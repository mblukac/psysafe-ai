"""Bayesian decision on the *gain from changing* the prefix highlighted for a reviewer.

No posteriors from nested prefixes are multiplied. For an original turn j and
the highest-scoring available later turn k, define two training-only rewards
U_j and U_k: is that turn endorsed AND is this review start within 10 minutes
of the first observed endorsement? Both arms use the very same start time.
In each relative-score cell their four joint outcomes have a Dirichlet prior.
Only P(U_j=0,U_k=1)-P(U_j=1,U_k=0) contributes to expected gain. The
posterior-mean utility maximizer switches iff that difference is positive.

Fractional family weights define a generalized (power-likelihood) posterior,
not an independent-turn clinical posterior or a calibrated uncertainty
interval. Freeze family membership, score calibration, and gate in the
development data. Age landmarks are an experimental design, not an estimated
waiting-time distribution. These choices, cells and prior are not optimized.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Order: neither, new only, frozen only, both. A small extra quarter-family
# of frozen-only evidence favors keeping the original in cells with no data.
PRIOR = (0.50, 0.50, 0.75, 0.50)
TURN = 2
DEADLINE = 10


def _valid_q(q: float) -> bool:
    return math.isfinite(q) and 0 <= q <= 1


def _cell(old_q: float, new_q: float) -> int:
    if not _valid_q(old_q) or not _valid_q(new_q):
        raise ValueError("Need finite probabilities")
    # Compare odds using cross multiplication; identical 0s or 1s are ties.
    a, b = new_q * (1 - old_q), old_q * (1 - new_q)
    if 2 * a < b:
        return 0
    return 2 if a > 2 * b else 1


@dataclass(frozen=True)
class PairRule:
    counts: tuple[tuple[float, ...], ...]  # family-weighted data, before PRIOR
    eligible_families: int
    eligible_paths: int
    landmarks: int

    def gain(self, original_q: float, challenger_q: float) -> float:
        n = self.counts[_cell(original_q, challenger_q)]
        return (n[1] + PRIOR[1] - n[2] - PRIOR[2]) / (sum(n) + sum(PRIOR))

    def replace(self, original_q: float, challenger_q: float) -> bool:
        return self.gain(original_q, challenger_q) > 1e-12  # exact numerical ties freeze

    def choose(self, visible: Sequence[tuple[int, float]], elapsed_minutes: float) -> int:
        """Return a turn index. No votes, true onset or predicted future turns are accepted."""
        if not visible or not math.isfinite(elapsed_minutes) or not 0 <= elapsed_minutes <= DEADLINE:
            raise ValueError("Expected a live, unexpired ticket")
        anchor, base = visible[0]
        previous = 0
        for t, q in visible:
            if type(t) is not int or t <= previous or (t - anchor) * TURN > elapsed_minutes or not _valid_q(q):
                raise ValueError("Expected only visible scores in strictly increasing turn order")
            previous = t
        if len(visible) == 1:
            return anchor
        new_turn, new_q = max(visible[1:], key=lambda r: r[1])  # earliest later turn wins a tie
        return new_turn if self.replace(base, new_q) else anchor


def fit_pair(
    paths: Mapping[str, Sequence[tuple[int, float, int]]],
    cutoff: float,
    *,
    families: Mapping[str, str] | None = None,
) -> PairRule:
    """Fit on original inner-OOF development paths, never resampled streams.

    If families is omitted each path is its own family (synthetic fixtures).
    At virtual review ages 0,...,10 minutes, expose only scored turns available
    at that instant; labels enter only after selecting the two candidate turns.
    The integer grid covers each distinct visibility/deadline interval and
    includes the inclusive expiry boundary. Skip landmarks with no alternative.
    Balance the remaining landmarks within each opening family to total one.
    """
    if not _valid_q(cutoff) or (families is not None and set(paths) - families.keys()):
        raise ValueError("Invalid gate or missing development-family mapping")
    grouped: dict[str, Counter[tuple[int, int]]] = {}
    eligible = landmarks = 0
    for key, path in paths.items():
        if not path:
            raise ValueError("Empty development path")
        previous = 0
        for t, q, y in path:
            if type(t) is not int or t <= previous or not _valid_q(q) or y not in (0, 1):
                raise ValueError("Invalid or nonchronological scored development turns")
            previous = t
        anchor = next((row for row in path if row[1] >= cutoff), None)
        if anchor is None:
            continue
        j, base, original_y = anchor
        onset = next((t for t, _, y in path if y), None)
        later = [row for row in path if 0 < (row[0] - j) * TURN <= DEADLINE]
        if not later:
            continue
        eligible += 1
        bucket = grouped.setdefault(key if families is None else families[key], Counter())
        for age in range(DEADLINE + 1):
            visible = [row for row in later if (row[0] - j) * TURN <= age]
            if not visible:
                continue
            _, q, new_y = max(visible, key=lambda row: row[1])
            start_since_onset = None if onset is None else (j - onset) * TURN + age
            timely = start_since_onset is not None and 0 <= start_since_onset <= DEADLINE
            outcome = 2 * int(timely and original_y == 1) + int(timely and new_y == 1)
            bucket[_cell(base, q), outcome] += 1
            landmarks += 1
    counts = [[0.0] * 4 for _ in range(3)]
    for bucket in grouped.values():
        total = sum(bucket.values())
        for (cell, outcome), n in bucket.items():
            counts[cell][outcome] += n / total
    return PairRule(tuple(tuple(row) for row in counts), len(grouped), eligible, landmarks)
