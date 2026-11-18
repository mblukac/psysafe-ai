"""Paired error dependence relative to MindGuard's observed panel, not latent truth."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import correlation
from typing import Any

from .data import LABELS
from .fit import SINGLES
from .review import Fold, _group, _prepare, _Prepared, _selected

_PAIRS = tuple(combinations(range(3), 2))
_Number = float | int
_Metrics = dict[tuple[str, ...], _Number | None]


@dataclass
class _FoldData:
    review: _Prepared
    # [dialogue][six columns: three native, then three calibrated][turn]
    residuals: list[list[list[float]]]


def _fold_data(fold: Fold, budget: float, slots: int) -> _FoldData:
    train, dev, test, held = fold
    if any(m not in dev or m not in held for m in SINGLES):
        raise ValueError("dependence requires all three singles in every fold")
    for row in test:
        if len(row.votes) != 3 or any(v not in LABELS for v in row.votes):
            raise ValueError("expected three clinician votes per held-out turn")
        if len(row.scores) != 3 or any(not 0 <= p <= 1 for p in row.scores):
            raise ValueError("expected three finite native probabilities")
    review = _prepare(train, {m: dev[m] for m in SINGLES}, test, {m: held[m] for m in SINGLES}, budget, slots)
    groups = _group(test)
    residuals = []
    h = [row.votes.count("self_harm") / 3 for row in test]
    # _make_cases uses the same sorted dialogue order.
    for key in sorted(groups):
        idx = groups[key]
        native = [[test[t].scores[m] - h[t] for t in idx] for m in range(3)]
        calibrated = [[held[m][t] - h[t] for t in idx] for m in SINGLES]
        residuals.append(native + calibrated)
    return _FoldData(review, residuals)


def _metrics(data: list[_FoldData], rng: random.Random | None) -> _Metrics:
    errors: list[list[float]] = [[] for _ in range(6)]
    # In positive dialogues the bit is a missed case; in negatives it is a false alert.
    masks = {(event, policy): [0] * 8 for event in ("miss", "false_alert") for policy in ("fixed", "live")}
    for fold in data:
        n = len(fold.residuals)
        indices = list(range(n)) if rng is None else rng.choices(range(n), k=n)
        for i in indices:
            for col, residual in zip(errors, fold.residuals[i], strict=True):
                col.extend(residual)
        for policy in ("fixed", "live"):
            catch_sets = []
            alert_sets = []
            for method in SINGLES:
                cases = [fold.review.cases[method][i] for i in indices]
                chosen = _selected(cases, fold.review.test_slots, live=policy == "live")
                catch_sets.append({i for i, (_, endorsed) in chosen if endorsed > 0.5})
                alert_sets.append({i for i, _ in chosen})
            for j, i in enumerate(indices):
                if fold.review.cases[SINGLES[0]][i].onset is not None:
                    mask = sum(1 << m for m, caught in enumerate(catch_sets) if j not in caught)
                    masks["miss", policy][mask] += 1
                else:
                    mask = sum(1 << m for m, alerted in enumerate(alert_sets) if j in alerted)
                    masks["false_alert", policy][mask] += 1

    out: _Metrics = {}
    for (event, policy), hist in masks.items():
        base = (f"{event}_overlap", policy)
        verb = "missed" if event == "miss" else "alerted"
        out[(*base, "eligible_conversations" if event == "miss" else "never_majority_positive")] = sum(hist)
        out[(*base, f"all_three_{verb}")] = hist[7]
        for i, j in _PAIRS:
            prefix = (*base, f"{SINGLES[i]}__{SINGLES[j]}")
            cells = [0] * 4
            for mask, count in enumerate(hist):
                cells[2 * ((mask >> i) & 1) + ((mask >> j) & 1)] += count
            neither, second, first, both = cells
            denom = math.sqrt((neither + second) * (first + both) * (neither + first) * (second + both))
            for cell, value in zip(("neither", "second_only", "first_only", "both"), cells, strict=True):
                out[(*prefix, f"{cell}_{verb}")] = value
            out[(*prefix, "phi")] = (both * neither - first * second) / denom if denom else None
    for kind, columns in (("native", errors[:3]), ("calibrated_oof", errors[3:])):
        # Pearson is invariant to positive rescaling. Avoid underflow if all
        # probabilities on a bootstrap sample of negative turns are near zero.
        scaled = []
        for col in columns:
            low, high = min(col), max(col)
            scaled.append([(x - low) / (high - low) for x in col] if low < high else [])
        for i, j in _PAIRS:
            rho = correlation(scaled[i], scaled[j]) if scaled[i] and scaled[j] else None
            out[("turn_residual_pearson", kind, f"{SINGLES[i]}__{SINGLES[j]}")] = rho
    return out


def error_dependence(
    folds: Sequence[Fold], *, budget: float, slots_per_fold: Sequence[int], replicates: int, seed: int
) -> dict[str, Any]:
    """Summarize held-out case overlap and turn-weighted signed residual correlations.

    Bootstrap paired dialogues within fold; re-rank queues but freeze the fits and
    live thresholds. Intervals for a coefficient use only defined replicates and
    are suppressed if the observed coefficient itself is undefined. Shared noisy
    clinician votes can induce residual correlation; neither metric tests DMM
    conditional independence or measures shared errors against a patient outcome.
    """
    if not folds or replicates < 1 or len(folds) != len(slots_per_fold):
        raise ValueError("need folds, one slot allocation per fold, and positive bootstrap count")
    data = []
    seen: set[str] = set()
    for fold, k in zip(folds, slots_per_fold, strict=True):
        keys = {turn.dialogue_id for turn in fold[2]}
        if keys & seen:
            raise ValueError("overlapping held-out dialogues across folds")
        seen.update(keys)
        data.append(_fold_data(fold, budget, k))
    observed = _metrics(data, None)
    draws: dict[tuple[str, ...], list[_Number]] = defaultdict(list)
    rng = random.Random(seed)  # noqa: S311 - reproducible case-mix resampling
    for _ in range(replicates):
        for name, value in _metrics(data, rng).items():
            if value is not None:
                draws[name].append(value)
    report: dict[str, Any] = {"bootstrap_replicates": replicates}
    for path, point in observed.items():
        ordered = sorted(draws[path])
        interval = None
        if ordered and point is not None:
            interval = [ordered[math.ceil(q * len(ordered)) - 1] for q in (0.025, 0.975)]
        node = report
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = {"observed": point, "percentile_95_if_defined": interval, "defined_replicates": len(ordered)}
    return report
