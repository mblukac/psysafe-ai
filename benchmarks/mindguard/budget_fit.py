"""Nested, training-only price selection for the reviewer-minutes replay.

Budgets count starts (one start consumes 20 reviewer-minutes). The physical
queue, first-endorsement deadline, and disclosure cadence are identical for
training and held-out replay. No simulation ever sees a suffix when choosing
to stop. This tunes a fixed cutoff for each policy/budget, not a controller
that anticipates running out of monthly starts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import accumulate, pairwise
from math import isfinite
from typing import Any

from .data import Turn
from .fit import SINGLES, grouped_folds
from .staffing import placement
from .staffing_fit import train_policies
from .staffing_queue import Session, Trajectory, simulate

POLICIES = ("fixed", "rl", "rl_fifo")
# Common grid fixed before replay. Even max pacing pressure (+0.10) cannot
# prevent the -1.1 sentinel from admitting a first turn on either score scale.
GRID = (-1.1, -0.10, -0.025, 0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.95)
# Post hoc resolution sensitivity only. Integer ratios avoid accumulating
# floating-point increments or introducing near-duplicates of primary cuts.
DENSE_GRID = tuple(
    sorted(
        set(GRID)
        | {i / 40 for i in range(41)}
        | {
            -0.5,
            -0.25,
            -0.15,
            -0.075,
            -0.05,
            -0.01,
            -0.005,
            -0.001,
            0.0001,
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.0075,
            0.99,
        }
    )
)
SERVICE_MINUTES = 20.0
TURN_MINUTES = 2.0
DEADLINE_MINUTES = 10.0

TunedPath = tuple[Trajectory, float]
TunedPaths = dict[int, dict[str, list[TunedPath]]]
_Preds = dict[str, tuple[Trajectory, Trajectory]]  # calibrated and advantage
_Starts = tuple[int, int]  # (uncapped starts, onset hits in the capped prefix)


def _identity(value: float, _: int) -> float:
    return value


def _split_scores(train: list[Turn], test: list[Turn], index: int) -> _Preds:
    """Fit only on training paths, then produce both metrics without reading held-out labels as features."""

    def paths(rows: list[Turn]) -> dict[str, Trajectory]:
        by_id: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
        for row in rows:
            if not row.votes or any(v not in {"self_harm", "harm_others", "safe"} for v in row.votes):
                raise ValueError("require observed MindGuard votes; missing ratings are not negative labels")
            endorsed = int(row.votes.count("self_harm") * 2 > len(row.votes))
            by_id[row.dialogue_id].append((row.turn_index, row.scores[index], endorsed))
        return {d: tuple(sorted(path)) for d, path in by_id.items()}

    native_train, native_test = paths(train), paths(test)
    fitted = train_policies([native_train[d] for d in sorted(native_train)])
    return {
        d: (
            tuple((t, fitted.q(s), y) for t, s, y in native),
            tuple((t, fitted.advantage(s, t), y) for t, s, y in native),
        )
        for d, native in native_test.items()
    }


def _choice(
    candidates: dict[float, list[_Starts]], budget: int, *, require_fill: bool = True
) -> tuple[float, list[_Starts]]:
    """Maximize timely catches within B starts, with an optional fill constraint.

    Quota: prefer filling B in every training placement when possible. Across
    equal hits prefer greater worst fill. Ceiling: no minimum utilization;
    across equal hits prefer fewer total used starts. Finally break ties by
    higher cutoff. Use the same deterministic rule for every policy.
    """

    def rank(cut: float) -> tuple[bool, int, int, float]:
        stats = candidates[cut]
        if not require_fill:
            return True, sum(h for _, h in stats), -sum(min(n, budget) for n, _ in stats), cut
        minimum_used = min(min(n, budget) for n, _ in stats)
        return minimum_used == budget, sum(h for _, h in stats), minimum_used, cut

    winner = max(candidates, key=rank)
    return winner, candidates[winner]


def _tune(
    oof: _Preds,
    arrivals: Sequence[float],
    staff: int,
    budgets: tuple[int, ...],
    seed: int,
    reps: int,
    *,
    require_fill: bool = True,
    dense_grid: bool = False,
) -> tuple[dict[int, dict[str, float]], list[dict[str, Any]]]:
    grid = DENSE_GRID if dense_grid else GRID
    ordered = [p for _, p in sorted(oof.items())]
    observations: dict[tuple[str, int], dict[float, list[_Starts]]] = {
        (name, b): {cut: [] for cut in grid} for b in budgets for name in POLICIES
    }
    for rep in range(reps):
        indices = placement(len(ordered), len(arrivals), seed + rep)
        sessions = [
            [Session(at, ordered[i][field]) for at, i in zip(arrivals, indices, strict=True)] for field in (0, 1)
        ]
        for name in POLICIES:
            for cut in grid:
                flags: list[bool] = []
                simulate(
                    sessions[int(name != "fixed")],
                    reviewers=staff,
                    service_minutes=SERVICE_MINUTES,
                    turn_minutes=TURN_MINUTES,
                    deadline_minutes=DEADLINE_MINUTES,
                    policy="rl" if name == "rl" else "fixed",
                    metric=_identity,
                    threshold=cut,
                    starts_audit=flags,
                )
                cumulative = list(accumulate(flags, initial=0))
                for b in budgets:
                    observations[name, b][cut].append((len(flags), cumulative[min(b, len(flags))]))
    cuts: dict[int, dict[str, float]] = {b: {} for b in budgets}
    audit: list[dict[str, Any]] = []
    for b in budgets:
        for name in POLICIES:
            cut, stats = _choice(observations[name, b], b, require_fill=require_fill)
            cuts[b][name] = cut
            audit.append(
                {
                    "policy": name,
                    "selection_mode": "quota" if require_fill else "ceiling",
                    "budget_starts": b,
                    "cutoff": cut,
                    "inner_uncapped_starts": [n for n, _ in stats],
                    "inner_used_starts": [min(b, n) for n, _ in stats],
                    "inner_catches_by_onset_deadline": [h for _, h in stats],
                    "inner_full_budget_in_all_placements": all(n >= b for n, _ in stats),
                }
            )
    return cuts, audit


def tune_folds(
    rows: list[Turn],
    arrivals: Sequence[float],
    *,
    model: str,
    staff: int,
    budgets: tuple[int, ...],
    seed: int,
    inner_placements: int = 2,
    require_fill: bool = True,
    dense_grid: bool = False,
) -> tuple[TunedPaths, list[dict[str, Any]]]:
    """Return held-out original score paths paired with frozen per-fold cutoffs.

    Paths in every list are sorted by original dialogue ID. Inner simulations
    pool grouped three-fold OOF *training* scores, using common path placements
    across policies and candidates, distinct from the outer replay placements.
    Set require_fill=False for a budget ceiling with no minimum utilization.
    The post hoc dense grid is used only when explicitly requested.
    """
    if model not in ("jev", "gemini") or type(staff) is not int or staff < 1:
        raise ValueError("expected Jev or Gemini and at least one reviewer")
    if not budgets or len(set(budgets)) != len(budgets) or any(type(b) is not int or b < 0 for b in budgets):
        raise ValueError("supply distinct nonnegative start budgets")
    if not arrivals or any(not isfinite(t) or t < 0 for t in arrivals) or any(a > b for a, b in pairwise(arrivals)):
        raise ValueError("supply a finite, sorted, nonempty arrival trace")
    if type(inner_placements) is not int or inner_placements < 1:
        raise ValueError("need at least one inner placement")
    if type(require_fill) is not bool:
        raise ValueError("require_fill must be boolean")
    if type(dense_grid) is not bool:
        raise ValueError("dense_grid must be boolean")
    source_order = sorted({r.dialogue_id for r in rows})
    accum: dict[int, dict[str, dict[str, TunedPath]]] = {b: {p: {} for p in POLICIES} for b in budgets}
    audit: list[dict[str, Any]] = []
    index = SINGLES.index(model)
    for fold, (train, test) in enumerate(grouped_folds(rows, 5, seed)):
        inner: _Preds = {}
        for part, held in grouped_folds(train, 3, seed + fold):
            inner.update(_split_scores(part, held, index))
        cuts, details = _tune(
            inner,
            arrivals,
            staff,
            budgets,
            seed + 1_000_003 + fold * 10_007,
            inner_placements,
            require_fill=require_fill,
            dense_grid=dense_grid,
        )
        audit.extend(dict(row, outer_fold=fold + 1) for row in details)
        fitted = _split_scores(train, test, index)
        for b in budgets:
            for name in POLICIES:
                for d, values in fitted.items():
                    accum[b][name][d] = (values[int(name != "fixed")], cuts[b][name])
    return (
        {b: {p: [paths[d] for d in source_order] for p, paths in policies.items()} for b, policies in accum.items()},
        audit,
    )
