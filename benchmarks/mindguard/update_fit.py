"""A small autoregressive observation model for updating the evidence on a waiting ticket.

Target Y_t is *that turn's* binary majority endorsement, not a patient state,
an intervention effect, or whether any earlier turn was endorsed. Train on
uniquely counted development paths using calibrated inner-out-of-fold scores.
The caller must not train on replayed copies or on outer-test families.

At the start of each observed segment use the single-prefix estimate q_t.
For contiguous observations, only the transition from B_{t-1} to B_t is
used, where B bins the calibrated prefix score at fixed boundaries:

    u = (1 - p_prev) * A[0] + p_prev * A[1]
    p = u * L[1, B_prev, B] / (u * L[1, B_prev, B] + (1-u) * L[0, B_prev, B])

A is the next positive probability by previous label; L is the class-specific
probability of the current score bin conditional on its previous bin. Thus
two overlapping prefix estimates are not multiplied as independent opinions.
The working assumptions are first-order label dynamics and first-order
observation memory conditional on the current label. Higher-order residual
dependence, unobserved clinical states and parameter uncertainty are not solved.
This is a plug-in empirical-Bayes filter with shrunk conditional table estimates,
not a fully Bayesian posterior predictive integrated over all parameters.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

from .staffing_queue import Trajectory

# Fixed choices (not tuned on the small test release). Coarsening score changes
# is deliberate: fitting fine-grained conditional transitions to ~40 positive
# turns would mistake rare cells for reliable evidence.
BIN_EDGES = (0.10, 0.50, 0.80)
_K = len(BIN_EDGES) + 1
_LABEL_MASS = 2.0
_EMISSION_MASS = 4.0


def _bin(q: float) -> int:
    if isinstance(q, bool) or not isinstance(q, (float, int)) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("expected finite calibrated probability in [0, 1]")
    return bisect_right(BIN_EDGES, q)


def _index(t: int, previous: int) -> None:
    if type(t) is not int or t <= previous:
        raise ValueError("expected positive strictly increasing observed turn indices")


@dataclass(frozen=True)
class PrefixFilter:
    prevalence: float
    transition: tuple[float, float]
    # emission[y][previous_bin][current_bin]
    emission: tuple[tuple[tuple[float, ...], ...], ...]
    training_turns: int
    adjacent_pairs: int

    def posterior(self, observations: Iterable[tuple[int, float]]) -> tuple[float, ...]:
        """Return causal, label-blind scores; reset after gaps, never infer missing turns."""
        values = []
        prev = 0
        prev_bin = 0
        p = 0.0
        for t, q in observations:
            _index(t, prev)
            b = _bin(q)
            if prev and t == prev + 1:
                u = (1 - p) * self.transition[0] + p * self.transition[1]
                yes = u * self.emission[1][prev_bin][b]
                no = (1 - u) * self.emission[0][prev_bin][b]
                p = yes / (yes + no)
            else:
                p = float(q)
            values.append(p)
            prev, prev_bin = t, b
        return tuple(values)


def fit_filter(paths: Iterable[Trajectory]) -> PrefixFilter:
    """Fit once on labeled development paths, with no transition across a gap or path boundary.

    A_i uses two prior observations centered at Jeffreys-smoothed turn
    prevalence. Each emission row uses four prior observations centered at
    the pooled distribution of next bins for the same previous bin (with
    1/2 pseudocount per destination). Both classes have positive likelihood
    even in empty cells; a never-seen previous bin supplies neutral evidence.
    """
    transitions = [[0, 0] for _ in range(2)]
    counts = [[[0] * _K for _ in range(_K)] for _ in range(2)]
    n = positive = pairs = 0
    for path in paths:
        if not path:
            raise ValueError("empty development path")
        prev = prev_bin = prev_label = 0
        for t, q, y in path:
            _index(t, prev)
            b = _bin(q)
            if type(y) is not int or y not in (0, 1):
                raise ValueError("expected a binary observed endorsement")
            n += 1
            positive += y
            if prev and t == prev + 1:
                transitions[prev_label][y] += 1
                counts[y][prev_bin][b] += 1
                pairs += 1
            prev, prev_bin, prev_label = t, b, y
    if not n:
        raise ValueError("no labeled development turns")
    base = (positive + 0.5) / (n + 1)

    def transition(y: int) -> float:
        return (transitions[y][1] + _LABEL_MASS * base) / (sum(transitions[y]) + _LABEL_MASS)

    def emission(y: int, previous: int) -> tuple[float, ...]:
        pooled = [a + b for a, b in zip(counts[0][previous], counts[1][previous], strict=True)]
        totals = sum(pooled) + 0.5 * _K
        row = counts[y][previous]
        return tuple(
            (count + _EMISSION_MASS * (all_count + 0.5) / totals) / (sum(row) + _EMISSION_MASS)
            for count, all_count in zip(row, pooled, strict=True)
        )

    emissions = tuple(tuple(emission(y, previous) for previous in range(_K)) for y in (0, 1))
    return PrefixFilter(base, (transition(0), transition(1)), emissions, n, pairs)
