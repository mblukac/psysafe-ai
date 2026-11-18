"""Small, offline probability pools for MindGuard's self_harm vs. other vote.

One equally weighted example is one prefix. Its target is the fraction of
clinicians choosing self_harm (not the majority label or a clinical outcome).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
from math import exp, isfinite, log, log1p

from .data import Turn

SINGLES = ("jev", "gemini", "liquid")
_SUBSETS = dict(zip(SINGLES, ((0,), (1,), (2,)), strict=True)) | {"pair": (0, 1), "stack": (0, 1, 2)}
_EPS = 1e-6


def _prob(value: float) -> float:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("model scores must be finite probabilities in [0, 1]")
    return value


def _clipped(value: float) -> float:
    return max(_EPS, min(1 - _EPS, _prob(value)))


def _target(row: Turn) -> float:
    if not row.votes or any(v not in ("safe", "self_harm", "harm_others") for v in row.votes):
        raise ValueError("expected clinician votes using MindGuard labels")
    return row.votes.count("self_harm") / len(row.votes)


def log_loss(rows: Sequence[Turn], scores: Sequence[float]) -> float:
    """Average of per-clinician binary log losses; clip exact limits for reporting."""
    if not rows:
        raise ValueError("cannot score an empty set")
    total = 0.0
    for row, score in zip(rows, scores, strict=True):
        q, p = _target(row), _clipped(score)
        total -= q * log(p) + (1 - q) * log1p(-p)
    return total / len(rows)


def brier_loss(rows: Sequence[Turn], scores: Sequence[float]) -> float:
    """Vote-level Brier, including observed disagreement (not squared error to vote share)."""
    if not rows:
        raise ValueError("cannot score an empty set")
    total = 0.0
    for row, score in zip(rows, scores, strict=True):
        q, p = _target(row), _prob(score)
        total += q * (1 - p) ** 2 + (1 - q) * p**2
    return total / len(rows)


def _keys(rows: Sequence[Turn]) -> dict[tuple[str, int], int]:
    keys = {(r.dialogue_id, r.turn_index): i for i, r in enumerate(rows)}
    if len(keys) != len(rows):
        raise ValueError("duplicate dialogue/turn")
    return keys


def grouped_folds(rows: list[Turn], n_splits: int, seed: int) -> list[tuple[list[Turn], list[Turn]]]:
    """Partition whole dialogues, spreading majority-positive and minority-only cases."""
    _keys(rows)
    groups: dict[str, list[Turn]] = defaultdict(list)
    for row in rows:
        groups[row.dialogue_id].append(row)
    if not 2 <= n_splits <= len(groups):
        raise ValueError("fold count must be between 2 and the number of distinct dialogues")
    strata: dict[int, list[str]] = defaultdict(list)
    for group, turns in groups.items():
        peak = max(_target(turn) for turn in turns)
        strata[2 if peak > 0.5 else int(peak > 0)].append(group)
    bins: list[set[str]] = [set() for _ in range(n_splits)]
    lengths = [0] * n_splits
    for stratum in sorted(strata, reverse=True):
        counts = [0] * n_splits
        for group in sorted(strata[stratum], key=lambda g: sha256(f"{seed}:{g}".encode()).digest()):
            j = min(range(n_splits), key=lambda i: (counts[i], len(bins[i]), lengths[i], i))
            bins[j].add(group)
            lengths[j] += len(groups[group])
            counts[j] += 1
    return [([r for r in rows if r.dialogue_id not in b], [r for r in rows if r.dialogue_id in b]) for b in bins]


def _sigmoid(z: float) -> float:
    return 1 / (1 + exp(-z)) if z >= 0 else exp(z) / (1 + exp(z))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _features(row: Turn, columns: tuple[int, ...]) -> list[float]:
    if len(row.scores) != 3:
        raise ValueError("expected scores for Jev, Gemini and Liquid")
    probabilities = [_clipped(row.scores[c]) for c in columns]
    return [1.0, *(log(p) - log1p(-p) for p in probabilities)]


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Elimination on a positive-definite ridge Hessian, at most 4 by 4."""
    n = len(b)
    aug = [[*row, y] for row, y in zip(a, b, strict=True)]
    for j in range(n):
        pivot = aug[j][j]  # positive pivots for a positive-definite Hessian
        for i in range(j + 1, n):
            ratio = aug[i][j] / pivot
            for k in range(j + 1, n + 1):
                aug[i][k] -= ratio * aug[j][k]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (aug[i][n] - sum(aug[i][k] * x[k] for k in range(i + 1, n))) / aug[i][i]
    return x


def _fit(rows: list[Turn], columns: tuple[int, ...]) -> list[float]:
    """Damped Newton solve of regularized mean soft-label cross entropy.

    Fixed, prespecified penalties: 0.05/2 per squared slope deviation from
    equal logit weights, and 0.001/2 on intercept squared (finite for one-class folds).
    """
    design = [(_features(row, columns), _target(row)) for row in rows]
    prior = [0.0] + [1 / len(columns)] * len(columns)
    ridge = [0.001] + [0.05] * len(columns)
    beta = prior.copy()
    n = len(design)
    dim = len(prior)

    def objective(coef: list[float]) -> float:
        loss = 0.0
        for features, y in design:
            z = _dot(features, coef)
            loss += max(z, 0) - y * z + log1p(exp(-abs(z)))
        return loss / n + sum(0.5 * ridge[j] * (coef[j] - prior[j]) ** 2 for j in range(dim))

    for _ in range(64):
        grad = [n * r * (b - p) for r, b, p in zip(ridge, beta, prior, strict=True)]
        hess = [[n * r if j == k else 0.0 for k in range(dim)] for j, r in enumerate(ridge)]
        for features, y in design:
            p = _sigmoid(_dot(features, beta))
            for j, fj in enumerate(features):
                grad[j] += (p - y) * fj
                for k in range(dim):
                    hess[j][k] += p * (1 - p) * fj * features[k]
        if max(abs(g) for g in grad) <= 1e-8 * n:
            return beta
        direction = _solve(hess, grad)
        decrement = _dot(grad, direction) / n
        # Near the optimum loss differences can be below summation roundoff,
        # especially with 0/1 provider probabilities and rare endorsements.
        if decrement < 1e-13:
            return beta
        before = objective(beta)
        step = 1.0
        while step >= 2**-32:
            candidate = [b - step * d for b, d in zip(beta, direction, strict=True)]
            if objective(candidate) <= before - 1e-4 * step * decrement:
                beta = candidate
                break
            step *= 0.5
        else:
            raise ArithmeticError("ridge logit pool line search failed")
    raise ArithmeticError("ridge logit pool did not converge")


def _scores(train: list[Turn], test: list[Turn]) -> dict[str, list[float]]:
    result = {"equal": [sum(_prob(p) for p in r.scores) / 3 for r in test]}
    for method, columns in _SUBSETS.items():
        beta = _fit(train, columns)
        result[method] = [_sigmoid(_dot(_features(r, columns), beta)) for r in test]
    return result


def fold_predictions(
    train: list[Turn],
    test: list[Turn],
    *,
    seed: int = 0,
) -> tuple[dict[str, list[float]], dict[str, list[float]], str]:
    """Return inner out-of-fold train scores and outer held-out scores in input order.

    The best single is chosen by inner vote log loss (ties use Jev, Gemini, Liquid
    order). The outer fit never reads held-out votes. No hyperparameters are tuned.
    """
    position = _keys(train)
    _keys(test)
    groups = {r.dialogue_id for r in train}
    if not test or len(groups) < 2:
        raise ValueError("require a nonempty test and at least two training dialogues")
    if any(r.dialogue_id in groups for r in test):
        raise ValueError("dialogue appears in both train and test")
    oof = {method: [0.0] * len(train) for method in (*_SUBSETS, "equal")}
    for fit, held_out in grouped_folds(train, min(3, len(groups)), seed):
        for method, values in _scores(fit, held_out).items():
            for row, value in zip(held_out, values, strict=True):
                oof[method][position[(row.dialogue_id, row.turn_index)]] = value
    winner = min(SINGLES, key=lambda m: log_loss(train, oof[m]))
    final = _scores(train, test)
    oof["best_single"], final["best_single"] = oof[winner].copy(), final[winner].copy()
    return oof, final, winner
