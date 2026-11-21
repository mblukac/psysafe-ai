"""Training-only, family-separated model and alert-gate selection for surge replay.

The ensemble is fixed in advance: the arithmetic mean of the six native
either-risk probabilities. It and each single undergo the same one-score
calibration, gate search and training review budget. No evaluation clock,
receipts beyond the specified panel, or external calls enter selection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from .budget_fit import GRID
from .data import Turn
from .fit import grouped_folds
from .full_panel import MODEL_KEYS, PanelTurn
from .full_replay import TAGS, calibrated, encoded, endorsed
from .loss_workmatch import DEADLINE, SEED, SERVICE, dispatch_starts
from .staffing_queue import Trajectory
from .surge_replay import nominate
from .surge_stream import assign_paths, make_clock

SINGLES = tuple(sorted(MODEL_KEYS.values()))
ENSEMBLE = "mean6"
ARMS = (*SINGLES, ENSEMBLE)
N_ARRIVALS, N_STREAMS, START_BUDGET = 2000, 4, 100
CUTOFFS = tuple(c for c in GRID if c >= 0)


def encoded_arm(rows: Sequence[PanelTurn], arm: str) -> list[Turn]:
    """One binary reference and one scalar forecast per rated prefix; gaps stay gaps."""
    if arm in SINGLES:
        return encoded(rows, arm)
    if arm != ENSEMBLE:
        raise ValueError("unknown arm")
    result = []
    for r in rows:
        if set(r.scores) != set(SINGLES):
            raise ValueError("mean pool requires all six native forecasts")
        ps = [sum(r.scores[m][t] for t in TAGS) for m in SINGLES]
        if any(not math.isfinite(p) or not 0 <= p <= 1 + 1e-12 for p in ps):
            raise ValueError("invalid either-risk probability")
        p = math.fsum(min(1.0, x) for x in ps) / len(ps)
        result.append(
            Turn(r.dialogue_id, r.turn_index, ("self_harm" if endorsed(r.votes) else "safe",), (p, 0, 0), r.group_id)
        )
    return result


@dataclass(frozen=True)
class TrainingStream:
    times: tuple[float, ...]
    order: tuple[int, ...]


def training_streams(positive: Sequence[bool], fold: int) -> list[TrainingStream]:
    """Four entirely synthetic clocks per outer fold; different from all test streams.

    Per 400 arrivals, the first 100 gaps have a 60/hour exponential rate and
    the remaining 300 have a 20/hour rate. Place training paths using the same
    peak-enrichment convention as the project's surge design. Votes determine
    simulated mix but are never available to the dispatcher.
    """
    streams = []
    for rep in range(N_STREAMS):
        seed = SEED + 51_000_001 + 10_007 * fold + rep
        rng = Random(seed)  # noqa: S311 - reproducible synthetic training queues
        times = [0.0]
        for i in range(1, N_ARRIVALS):
            times.append(times[-1] + rng.expovariate(1.0 if i % 400 < 100 else 1.0 / 3.0))
        clock = make_clock(times)
        order, _ = assign_paths(positive, clock, seed=seed + 1_000_003, risk_burst=True)
        streams.append(TrainingStream(clock.arrivals, tuple(order)))
    return streams


@dataclass(frozen=True)
class Candidate:
    cutoff: float
    caught: int
    total_starts: int

    @property
    def ranking(self) -> tuple[int, int, float]:
        return self.caught, -self.total_starts, self.cutoff


def train_gate(
    paths: Sequence[Trajectory],
    streams: Sequence[TrainingStream],
    *,
    budget: int = START_BUDGET,
    cutoffs: Sequence[float] = CUTOFFS,
) -> Candidate:
    """Maximize total frozen supported catches on exactly budget starts in EACH stream.

    Compare every eligible candidate at equal actual work. Break equal catch
    totals by fewer uncapped starts and then a higher cutoff (within detector).
    Infeasibility is data, not extra free catches; a zero cutoff ensures a
    fully nominating alternative under the designated development simulator.
    """
    if budget < 1 or not streams or not paths or not cutoffs:
        raise ValueError("empty candidates, budget or training set")
    candidates = []
    for cutoff in cutoffs:
        noms = [nominate(path, cutoff) for path in paths]
        caught = total = 0
        for stream in streams:
            tickets = sorted(
                (at + 2 * (n.turn - 1), sid, n.score)
                for sid, (at, ix) in enumerate(zip(stream.times, stream.order, strict=True))
                if (n := noms[ix]).turn is not None and n.score is not None
            )
            starts = dispatch_starts(tickets, 1, "fifo")
            if len(starts) < budget:
                break
            total += len(starts)
            for sid, at in starts[:budget]:
                nom = noms[stream.order[sid]]
                if nom.supported and nom.onset_turn is not None:
                    onset = stream.times[sid] + 2 * (nom.onset_turn - 1)
                    caught += int(onset <= at <= onset + DEADLINE)
        else:
            candidates.append(Candidate(cutoff, caught, total))
    if not candidates:
        raise ValueError("no gate can fill the actual training budget on every paired stream")
    return max(candidates, key=lambda c: c.ranking)


def train_only(rows: list[PanelTurn], fold: int) -> tuple[str, dict[str, Candidate]]:
    """Train/evaluate gate candidates on OOF predictions using outer-training families only.

    The inner OOF labels are intentionally *selection* data, not estimates of
    generalization. Only the outer held-out replay evaluates the chosen policy.
    """
    ref = encoded_arm(rows, SINGLES[0])
    ids = sorted({r.dialogue_id for r in ref})
    positive = {r.dialogue_id for r in ref if r.votes == ("self_harm",)}
    streams = training_streams([d in positive for d in ids], fold)
    splits = [
        ({r.dialogue_id for r in train}, {r.dialogue_id for r in val})
        for train, val in grouped_folds(ref, 3, SEED + fold)
    ]
    candidates = {}
    for arm in ARMS:
        scores = encoded_arm(rows, arm)
        oof: dict[str, Trajectory] = {}
        for train_ids, val_ids in splits:
            train = [r for r in scores if r.dialogue_id in train_ids]
            val = [r for r in scores if r.dialogue_id in val_ids]
            if {r.group_id for r in train} & {r.group_id for r in val} or set(oof) & val_ids:
                raise ValueError("inner training family leak or duplicated validation dialogue")
            oof.update(calibrated(train, val))
        if set(oof) != set(ids):
            raise ValueError("incomplete training-only predictions")
        candidates[arm] = train_gate([oof[d] for d in ids], streams)
    # Threshold scales can differ; across detectors, compare only equal-work
    # catches, then uncapped work. Input tuple order supplies an alphabetical
    # last-resort tie-break. Never use the ensemble to choose its comparator.
    best = max(SINGLES, key=lambda m: (candidates[m].caught, -candidates[m].total_starts))
    return best, candidates


@dataclass(frozen=True)
class FoldFit:
    paths: dict[str, dict[str, Trajectory]]
    cutoffs: dict[str, float]
    selected_single: str
    audit: dict[str, Any]


def fit_fold(panel: Sequence[PanelTurn], train_ids: set[str], test_ids: set[str], fold: int) -> FoldFit:
    if not train_ids or not test_ids or train_ids & test_ids or train_ids | test_ids != {r.dialogue_id for r in panel}:
        raise ValueError("expected a full partition of the available conversations")
    train = [r for r in panel if r.dialogue_id in train_ids]
    test = [r for r in panel if r.dialogue_id in test_ids]
    if {r.group_id for r in train} & {r.group_id for r in test}:
        raise ValueError("outer train and test share an opening family")
    best, candidates = train_only(train, fold)
    # This is the first call where evaluation rows are scored. The selected
    # model and all cutoffs are frozen; labels are attached only for scoring.
    paths = {arm: calibrated(encoded_arm(train, arm), encoded_arm(test, arm)) for arm in ARMS}
    cutoffs = {arm: candidates[arm].cutoff for arm in ARMS}
    audit = dict(
        fold=fold,
        train_paths=len(train_ids),
        held_paths=len(test_ids),
        train_opening_families=len({r.group_id for r in train}),
        held_opening_families=len({r.group_id for r in test}),
        synthetic_training_streams=N_STREAMS,
        synthetic_arrivals_per_stream=N_ARRIVALS,
        equal_training_starts_per_stream=START_BUDGET,
        equal_training_minutes_per_stream=START_BUDGET * SERVICE,
        cutoffs=cutoffs,
        training_matched_catches={arm: c.caught for arm, c in candidates.items()},
        training_uncapped_starts={arm: c.total_starts for arm, c in candidates.items()},
        selected_single=best,
    )
    return FoldFit(paths, cutoffs, best, audit)
