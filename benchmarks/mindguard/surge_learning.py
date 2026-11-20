"""Training-only offline RL for dispatch among already waiting nominations.

The fitted object chooses a *job*, not when to issue a referral or how to treat
a person. Training randomizes choices on simulated queues; the chosen start
changes which other tickets survive until the next dispatch. Optimization uses
the cached retrospective reference only after the entire trajectory is over.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from random import Random
from typing import Any

from .surge_queue import Candidate, Case, Context, rule_index, simulate

RULES = ("risk", "fifo", "aging")  # first wins ties/duplicate nominations
DISCOUNT = 0.98  # per actual review start, not per contested node or wall minute
_MIN_ACTION_VISITS = 6
_SHRINKAGE = 8.0
_MAX_EPISODES = 32
State = tuple[int, bool, bool, bool, bool]
Selector = Callable[[Sequence[Candidate], Context], int]


def _observation(candidates: Sequence[Candidate], context: Context, deadline: float) -> tuple[State, dict[int, int]]:
    """Queue features only. Never use eventual demand, labels, or duration left."""
    if not candidates:
        raise ValueError("dispatch requires a waiting ticket")
    winners: dict[int, int] = {}
    for a, name in enumerate(RULES):
        i = rule_index(candidates, mode=name, deadline_minutes=deadline)
        if i not in winners.values():
            winners[a] = i
    top = candidates[winners[0]].q
    oldest = candidates[0]  # simulator presents candidates in FIFO order
    state = (
        sum(1 << a for a in winners),
        len(candidates) >= max(3, 2 * (context.reviewers or 1)),
        oldest.age >= 0.65 * deadline,
        top - oldest.q >= 0.15,
        top >= 0.50,
    )
    return state, winners


@dataclass
class _Moment:
    count: int = 0
    reward: float = 0.0
    next_weights: dict[State, float] = field(default_factory=dict)


_Transitions = dict[State, dict[int, _Moment]]


def _collect(
    moments: _Transitions,
    choices: Sequence[tuple[State, int] | None],
    contests: Sequence[int],
    hits: Sequence[bool],
) -> None:
    """Aggregate semi-Markov transitions until the next *distinct* choice.

    Include hits at the current dispatch and at intervening unopposed starts.
    The final interval ends at the end of the supplied training episode; it
    has no continuation bootstrap. Do not compare to the best future gold.
    """
    if len(choices) != len(contests):
        raise ValueError("simulator decision and feedback lengths differ")
    last = -1
    actual = []
    for choice, i in zip(choices, contests, strict=True):
        if type(i) is not int or not last < i < len(hits):
            raise ValueError("simulator contested start positions must increase")
        last = i
        if choice is not None:
            actual.append((i, *choice))
    for j, (start, state, action) in enumerate(actual):
        end = actual[j + 1][0] if j + 1 < len(actual) else len(hits)
        moment = moments.setdefault(state, {}).setdefault(action, _Moment())
        moment.count += 1
        weight = 1.0
        for idx in range(start, end):
            moment.reward += weight * int(hits[idx])
            weight *= DISCOUNT
        if end < len(hits):
            future = actual[j + 1][1]
            moment.next_weights[future] = moment.next_weights.get(future, 0.0) + weight


def _fit_table(moments: _Transitions) -> tuple[dict[State, int], int]:
    """Batch, support-restricted fitted Q on empirical transition moments.

    Shrink each estimated action return toward the training return pooled
    across actions in the same observable state. Require observations for both
    the comparator and an alternative; otherwise default to risk priority.
    """
    if not moments:
        return {}, 0
    value = dict.fromkeys(moments, 0.0)
    for iteration in range(1, 1201):
        new: dict[State, float] = {}
        greedy: dict[State, int] = {}
        for s, row in moments.items():
            numerators = {a: m.reward + sum(w * value[t] for t, w in m.next_weights.items()) for a, m in row.items()}
            pooled = sum(numerators.values()) / sum(m.count for m in row.values())
            q = {a: (numerators[a] + _SHRINKAGE * pooled) / (m.count + _SHRINKAGE) for a, m in row.items()}
            # With no support for risk we have no supported comparative claim.
            # The pooled return then serves as the backed-up default estimate.
            allowed = [0]
            if 0 in row and row[0].count >= _MIN_ACTION_VISITS:
                allowed.extend(a for a in (1, 2) if a in row and row[a].count >= _MIN_ACTION_VISITS)
            greedy[s] = max(allowed, key=lambda a: q.get(a, pooled))
            new[s] = q.get(greedy[s], pooled)
        if max(abs(new[s] - value[s]) for s in value) < 1e-7:
            return greedy, iteration
        value = new
    raise ArithmeticError("discounted dispatch-value iteration did not converge")


@dataclass(frozen=True)
class DispatchPolicy:
    """Frozen lookup table. Evaluating it does not use or update reward data."""

    table: dict[State, int]
    deadline_minutes: float
    diagnostics: dict[str, Any]

    def _choose(self, candidates: Sequence[Candidate], context: Context, counts: Counter[str] | None) -> int:
        s, winners = _observation(candidates, context, self.deadline_minutes)
        a = self.table.get(s, 0) if len(winners) > 1 else 0
        if counts is not None:
            counts["contested"] += 1
            counts["choice_differs"] += int(len(winners) > 1)
            counts["nonrisk_selected"] += int(a != 0)
            counts[RULES[a]] += 1
        return winners[a]

    def __call__(self, candidates: Sequence[Candidate], context: Context) -> int:
        return self._choose(candidates, context, None)

    def selector(self, counts: Counter[str]) -> Selector:
        """Optional evaluation telemetry only: how often dispatch actually changed."""
        return lambda candidates, context: self._choose(candidates, context, counts)


def _explorer(rng: Random, choices: list[tuple[State, int] | None], deadline: float) -> Selector:
    def choose(candidates: Sequence[Candidate], context: Context) -> int:
        state, winners = _observation(candidates, context, deadline)
        if len(winners) == 1:
            choices.append(None)
            return winners[0]
        act = rng.choice(list(winners))
        choices.append((state, act))
        return winners[act]

    return choose


def train_dispatch(
    episodes: Iterable[Sequence[Case]],
    *,
    reviewers: int,
    seed: int,
    service_minutes: float = 20.0,
    deadline_minutes: float = 10.0,
) -> DispatchPolicy:
    """Learn only from supplied training placements; caller owns family separation.

    The learner uniformly explores *distinct nominations* on the queue it
    creates. One example per simulator start is a deadline-qualified supported
    referral (1) or no such catch (0). Subsequent noncontroversial starts and
    action-dependent queue trajectories contribute to the discounted return.
    No random decisions or training-time feedback are used on held-out shifts.
    """
    if type(reviewers) is not int or reviewers < 1:
        raise ValueError("training requires a positive finite team")
    if any(not isfinite(t) or t <= 0 for t in (service_minutes, deadline_minutes)):
        raise ValueError("service and deadline must be finite and positive")
    moments: _Transitions = {}
    decision_count = contests_total = starts_total = episodes_seen = 0
    for i, sessions in enumerate(episodes):
        if i >= _MAX_EPISODES or not sessions:
            raise ValueError("provide one to 32 nonempty training placements")
        rng = Random(seed + 100_003 * i)  # noqa: S311 - reproducible policy exploration
        choices: list[tuple[State, int] | None] = []

        contests: list[int] = []
        hits: list[bool] = []
        simulate(
            sessions,
            reviewers=reviewers,
            service_minutes=service_minutes,
            deadline_minutes=deadline_minutes,
            policy="learned",
            select=_explorer(rng, choices, deadline_minutes),
            contest_starts=contests,
            start_hits=hits,
        )
        _collect(moments, choices, contests, hits)
        episodes_seen += 1
        starts_total += len(hits)
        contests_total += len(contests)
        decision_count += sum(c is not None for c in choices)
    if not episodes_seen:
        raise ValueError("at least one training placement is required")
    table, iterations = _fit_table(moments)
    return DispatchPolicy(
        table,
        deadline_minutes,
        {
            "algorithm": "support_restricted_tabular_dispatch_fitted_q",
            "discount_per_start": DISCOUNT,
            "episodes": episodes_seen,
            "starts": starts_total,
            "contested_starts": contests_total,
            "distinct_choice_starts": decision_count,
            "exploratory_actions": {
                RULES[a]: sum(row[a].count for row in moments.values() if a in row) for a in range(3)
            },
            "observed_states": len(table),
            "learned_nonrisk_states": sum(a != 0 for a in table.values()),
            "minimum_training_visits_per_action": _MIN_ACTION_VISITS,
            "shrinkage_observations_per_action": _SHRINKAGE,
            "bellman_iterations": iterations,
        },
    )
