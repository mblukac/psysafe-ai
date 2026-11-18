"""One retrospective review slot per dialogue, versus one prospective first alert.

All inputs are aligned, out-of-sample turn predictions. Only the retrospective
policy may use future turns to choose which prefix to send for review.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from .data import Turn

Scores = Mapping[str, Sequence[float]]
Fold = tuple[Sequence[Turn], Scores, Sequence[Turn], Scores]


def _slots(n: int, budget: float) -> int:
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("budget must be a finite fraction in [0, 1]")
    return max(int(budget > 0), math.floor(n * budget + 0.5))


def allocate_slots(sizes: Sequence[int], total: int) -> list[int]:
    """Distribute exactly total places by fold size; ties go to the earlier fold."""
    if not sizes or any(type(n) is not int or n < 0 for n in sizes):
        raise ValueError("expected nonnegative integer fold sizes")
    n_all = sum(sizes)
    if type(total) is not int or not 0 <= total <= n_all:
        raise ValueError("total slots must be an integer bounded by the number of dialogues")
    if total == 0:
        return [0] * len(sizes)
    # Integer divmod avoids losing ties through floating-point arithmetic.
    quotients = [divmod(n * total, n_all) for n in sizes]
    allocation = [whole for whole, _ in quotients]
    order = sorted(range(len(sizes)), key=lambda i: (-quotients[i][1], i))
    for i in order[: total - sum(allocation)]:
        allocation[i] += 1
    return allocation


def _group(rows: Sequence[Turn]) -> dict[str, list[int]]:
    if not rows:
        raise ValueError("empty review fold")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.dialogue_id].append(i)
    for indexes in groups.values():
        indexes.sort(key=lambda i: rows[i].turn_index)
        if len({rows[i].turn_index for i in indexes}) != len(indexes):
            raise ValueError("duplicate turn_index within a dialogue")
    return groups


def _check_scores(scores: Sequence[float], rows: Sequence[Turn]) -> None:
    if len(scores) != len(rows) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("expected one finite probability in [0, 1] per row")


def _threshold(groups: dict[str, list[int]], scores: Sequence[float], slots: int) -> float:
    """Review at score >= threshold. Never split a training tie or exceed train capacity."""
    maxima = sorted((max(scores[i] for i in inds) for inds in groups.values()), reverse=True)
    if slots == len(maxima):
        return 0.0
    if slots == 0:
        return math.nextafter(1.0, math.inf)
    # Smallest representable threshold excluding the highest training peak
    # outside the budget. An indivisible tie block may therefore underfill it.
    return math.nextafter(maxima[slots], math.inf)


@dataclass(frozen=True)
class _Case:
    tie_key: bytes
    peak_score: float
    # Tuple of (turn index, fraction of panel endorsing self-harm).
    peak: tuple[int, float]
    crossing: tuple[int, float] | None
    onset: int | None


def _make_cases(
    rows: Sequence[Turn],
    scores: Sequence[float],
    groups: dict[str, list[int]],
    threshold: float,
) -> list[_Case]:
    fractions = [r.votes.count("self_harm") / len(r.votes) for r in rows]
    cases = []
    for dialogue_id in sorted(groups):
        inds = groups[dialogue_id]
        # max on a chronological sequence retains the earliest tied peak.
        peak = max(inds, key=lambda i: scores[i])
        first = next((i for i in inds if scores[i] >= threshold), None)
        onset = next((rows[i].turn_index for i in inds if fractions[i] > 0.5), None)
        cases.append(
            _Case(
                hashlib.sha256(dialogue_id.encode()).digest(),
                scores[peak],
                (rows[peak].turn_index, fractions[peak]),
                None if first is None else (rows[first].turn_index, fractions[first]),
                onset,
            ),
        )
    return cases


class Counts(TypedDict):
    reviews: int
    catches: int
    misses: int
    unsupported: int
    premature: int
    endorsed_vote_mass: float


def _count(cases: list[_Case], slots: int, live: bool) -> tuple[Counts, set[int]]:
    if live:
        selected = [(i, c.crossing) for i, c in enumerate(cases) if c.crossing is not None]
    else:
        order = sorted(range(len(cases)), key=lambda i: (-cases[i].peak_score, cases[i].tie_key))
        selected = [(i, cases[i].peak) for i in order[:slots]]
    caught = {i for i, (_, fraction) in selected if fraction > 0.5}
    premature = 0
    for i, (turn, _) in selected:
        onset = cases[i].onset
        premature += int(onset is not None and turn < onset)
    return {
        "reviews": len(selected),
        "catches": len(caught),
        "misses": sum(c.onset is not None for c in cases) - len(caught),
        "unsupported": len(selected) - len(caught),
        "premature": premature,
        "endorsed_vote_mass": sum(fraction for _, (_, fraction) in selected),
    }, caught


@dataclass
class _Prepared:
    cases: dict[str, list[_Case]]
    thresholds: dict[str, float]
    train_reviews: dict[str, int]
    train_size: int
    test_slots: int


def _prepare(
    train_rows: Sequence[Turn],
    train_scores: Scores,
    test_rows: Sequence[Turn],
    test_scores: Scores,
    budget: float,
    slots: int | None,
) -> _Prepared:
    train_groups, test_groups = _group(train_rows), _group(test_rows)
    if train_groups.keys() & test_groups.keys():
        raise ValueError("a dialogue appears in both development and test")
    if not train_scores or train_scores.keys() != test_scores.keys():
        raise ValueError("development and test methods must match")
    train_slots = _slots(len(train_groups), budget)
    if slots is not None and (type(slots) is not int or not 0 <= slots <= len(test_groups)):
        raise ValueError("test slots must be an integer bounded by the number of test dialogues")
    test_slots = _slots(len(test_groups), budget) if slots is None else slots
    prepared = _Prepared({}, {}, {}, len(train_groups), test_slots)
    for method, scores in train_scores.items():
        _check_scores(scores, train_rows)
        _check_scores(test_scores[method], test_rows)
        threshold = _threshold(train_groups, scores, train_slots)
        prepared.thresholds[method] = threshold
        prepared.train_reviews[method] = sum(
            max(scores[i] for i in inds) >= threshold for inds in train_groups.values()
        )
        prepared.cases[method] = _make_cases(test_rows, test_scores[method], test_groups, threshold)
    return prepared


class MethodReport(TypedDict):
    threshold: float
    train_live_reviews: int
    fixed: Counts
    live: Counts


class FoldReport(TypedDict):
    dialogues: int
    eligible_cases: int
    train_dialogues: int
    queue_slots: int
    methods: dict[str, MethodReport]


def evaluate_fold(
    train_rows: Sequence[Turn],
    train_scores: Scores,
    test_rows: Sequence[Turn],
    test_scores: Scores,
    *,
    budget: float,
    slots: int | None = None,
) -> FoldReport:
    """Select thresholds on training OOF scores; return anonymous held-out results."""
    prep = _prepare(train_rows, train_scores, test_rows, test_scores, budget, slots)
    cases = next(iter(prep.cases.values()))
    methods: dict[str, MethodReport] = {}
    for method, records in prep.cases.items():
        methods[method] = {
            "threshold": prep.thresholds[method],
            "train_live_reviews": prep.train_reviews[method],
            "fixed": _count(records, prep.test_slots, live=False)[0],
            "live": _count(records, prep.test_slots, live=True)[0],
        }
    return {
        "dialogues": len(cases),
        "eligible_cases": sum(c.onset is not None for c in cases),
        "train_dialogues": prep.train_size,
        "queue_slots": prep.test_slots,
        "methods": methods,
    }


def _paired(prep: _Prepared, indices: Sequence[int]) -> dict[str, int]:
    delta = {}
    a, b = ([prep.cases[name][i] for i in indices] for name in ("stack", "best_single"))
    for policy, live in (("fixed", False), ("live", True)):
        ac, a_hits = _count(a, prep.test_slots, live)
        bc, b_hits = _count(b, prep.test_slots, live)
        delta[f"{policy}_catch_delta"] = ac["catches"] - bc["catches"]
        delta[f"{policy}_review_delta"] = ac["reviews"] - bc["reviews"]
        delta[f"{policy}_stack_only_catches"] = len(a_hits - b_hits)
        delta[f"{policy}_single_only_catches"] = len(b_hits - a_hits)
    return delta


class Interval(TypedDict):
    observed: int
    percentile_95: list[int]


def paired_uncertainty(
    folds: Sequence[Fold],
    *,
    budget: float,
    slots_per_fold: Sequence[int] | None = None,
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Interval]:
    """Paired within-fold dialogue bootstrap conditional on already fitted predictions.

    Resample the dialogues with replacement, carrying every turn, model and
    reference together. Rebuild exact-budget queues, but keep train thresholds
    frozen. Percentile limits use empirical (nearest rank) quantiles. They
    describe case-mix variability, not model-selection or rater uncertainty.
    """
    if not folds or replicates < 1:
        raise ValueError("bootstrap requires folds and at least one replicate")
    if slots_per_fold is not None and len(slots_per_fold) != len(folds):
        raise ValueError("expected one slot allocation per fold")
    preps = [
        _prepare(*fold, budget, None if slots_per_fold is None else slots_per_fold[i]) for i, fold in enumerate(folds)
    ]
    if any(not {"stack", "best_single"} <= prep.cases.keys() for prep in preps):
        raise ValueError("bootstrap requires stack and best_single")
    seen: set[bytes] = set()
    for prep in preps:
        keys = {c.tie_key for c in prep.cases["stack"]}
        if seen & keys:
            raise ValueError("overlapping held-out dialogues across folds")
        seen |= keys
    rng = random.Random(seed)  # noqa: S311 - reproducible resampling, not security

    def totals(sample: bool) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for prep in preps:
            n = len(prep.cases["stack"])
            indices = rng.choices(range(n), k=n) if sample else list(range(n))
            for metric, count in _paired(prep, indices).items():
                result[metric] += count
        return result

    observed = totals(False)
    draws = [totals(True) for _ in range(replicates)]
    output: dict[str, Interval] = {}
    for metric, count in observed.items():
        ordered = sorted(d[metric] for d in draws)
        output[metric] = {
            "observed": count,
            "percentile_95": [ordered[math.ceil(q * replicates) - 1] for q in (0.025, 0.975)],
        }
    return output
