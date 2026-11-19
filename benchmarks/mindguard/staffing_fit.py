"""Offline stopping values for first-referral replay, not an intervention model.

One model and one outer training set per fit. A stop earns the majority
endorsement on the nominated turn; waiting earns a discounted future stop.
An observation at the end of a complete path has no successor. Held-out path
length, clinician votes, timestamps, and unbought models are never features.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from math import inf, isfinite, log, log1p, nextafter

from .data import Turn
from .fit import _clipped, _fit, _sigmoid

Observation = tuple[int, float, int]
Path = Sequence[Observation]

DISCOUNT = 0.95
_SCORE_EDGES = (0.01, 0.05, 0.20, 0.50, 0.80)
_TURN_EDGES = (4, 9, 16)  # only elapsed turns; never use fraction of completed path
_AGES = len(_TURN_EDGES) + 1
_BANDS = len(_SCORE_EDGES) + 1
_STATES = _AGES * _BANDS
_PRIOR_WEIGHT = 4.0


def _checked_score(score: float) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score) or not 0 <= score <= 1:
        raise ValueError("expected a finite model probability in [0, 1]")
    return score


def _q(coef: tuple[float, float], score: float) -> float:
    p = _clipped(_checked_score(score))
    return _sigmoid(coef[0] + coef[1] * (log(p) - log1p(-p)))


def _state(q: float, turn: int) -> int:
    if type(turn) is not int or turn < 1:
        raise ValueError("turn must be a positive integer")
    return bisect_right(_SCORE_EDGES, q) * _AGES + bisect_right(_TURN_EDGES, turn)


@dataclass(frozen=True)
class StoppingPolicy:
    calibration: tuple[float, float]
    wait_values: tuple[float, ...]
    bellman_iterations: int

    def q(self, score: float) -> float:
        """Estimated probability of current majority endorsement, shared by all policies."""
        return _q(self.calibration, score)

    def advantage(self, score: float, turn: int) -> float:
        """Discounted stopping advantage, not a probability: reward now minus value of waiting."""
        q = self.q(score)
        return q - self.wait_values[_state(q, turn)]


def matched_cutoff(train_paths: Sequence[Path], policy: StoppingPolicy, base_threshold: float) -> float:
    """Match RL's training *no-congestion* admissions with a single q cutoff.

    A path consumes at most one ticket. Thus only each path's maximum matters
    for admission count (not which turn it nominates). Use the largest feasible
    count not exceeding RL's; a tied q at the boundary cannot be split. This
    freezes a comparator, not a guarantee about held-out, congested workload.
    """
    if not train_paths or any(not path for path in train_paths):
        raise ValueError("need nonempty, complete training paths")
    if not isfinite(base_threshold) or not 0 <= base_threshold <= 1:
        raise ValueError("expected a base threshold in [0, 1]")
    peaks = []
    target = 0
    for path in train_paths:
        if any(type(t) is not int or t != index for index, (t, _, _) in enumerate(path, start=1)):
            raise ValueError("training paths must start at turn 1 and be consecutive")
        peaks.append(max(policy.q(score) for _, score, _ in path))
        target += any(policy.advantage(score, t) >= base_threshold for t, score, _ in path)
    if target == 0:
        return nextafter(1.0, inf)
    if target == len(peaks):
        return 0.0
    return nextafter(sorted(peaks, reverse=True)[target], inf)


def train_policies(train_paths: Sequence[Path]) -> StoppingPolicy:
    """Fit a per-provider ridge logit calibrator and tabular fitted Q iteration.

    Each wait target is gamma * max(q(next), wait(next)), or zero on ending.
    Average targets in bins of current q and elapsed turn, shrinking each cell
    towards its score band's mean with four pseudo-observations. Unseen bands
    back off to the global training mean. This contraction learns the value of
    a multi-step policy, *not* the maximum future observed clinician label.

    The allocator separately sets admission/priority under its actual workload;
    this surrogate stopping objective does not optimize the full service queue.
    Callers must supply genuinely finished, fully rated paths, not truncations.
    """
    if not train_paths or any(not path for path in train_paths):
        raise ValueError("need nonempty, complete training paths")
    rows: list[Turn] = []
    for d, path in enumerate(train_paths):
        for expected, (t, score, y) in enumerate(path, start=1):
            if type(t) is not int or t != expected:
                raise ValueError("training paths must start at turn 1 and be consecutive; gaps are not terminals")
            if not isinstance(y, int) or y not in (0, 1):
                raise ValueError("need an observed binary majority label on every training turn")
            # Reuse the pilot's verified single-column logit fit. Only column 0
            # is accessed; the single target here is majority, not vote share.
            rows.append(Turn(str(d), t, ("self_harm" if y else "safe",), (_checked_score(score), 0.0, 0.0)))
    intercept, slope = _fit(rows, (0,))
    coef = (intercept, slope)
    # Keep the full denominator (including terminal observations) but only
    # record nonterminal successors; terminal value is exactly zero.
    counts = [0] * _STATES
    transitions: list[tuple[int, float, int]] = []
    for path in train_paths:
        previous = None
        for t, score, _ in path:
            q = _q(coef, score)
            state = _state(q, t)
            counts[state] += 1
            if previous is not None:
                transitions.append((previous, q, state))
            previous = state
    n = len(rows)
    band_counts = [sum(counts[i : i + _AGES]) for i in range(0, _STATES, _AGES)]
    wait = [0.0] * _STATES
    for iteration in range(1, 401):
        sums = [0.0] * _STATES
        for start, next_q, dest in transitions:
            sums[start] += DISCOUNT * max(next_q, wait[dest])
        global_mean = sum(sums) / n
        priors = [
            sum(sums[b * _AGES : (b + 1) * _AGES]) / count if count else global_mean
            for b, count in enumerate(band_counts)
        ]
        update = [
            (value + _PRIOR_WEIGHT * priors[s // _AGES]) / (counts[s] + _PRIOR_WEIGHT) for s, value in enumerate(sums)
        ]
        if max(abs(v - w) for v, w in zip(update, wait, strict=True)) < 1e-8:
            return StoppingPolicy(coef, tuple(update), iteration)
        wait = update
    raise ArithmeticError("discounted stopping-value iteration did not converge")
