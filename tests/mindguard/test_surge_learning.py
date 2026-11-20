from collections import Counter

import pytest

from benchmarks.mindguard.surge_learning import (
    DISCOUNT,
    DispatchPolicy,
    State,
    _collect,
    _fit_table,
    _Moment,
    _Transitions,
    train_dispatch,
)
from benchmarks.mindguard.surge_queue import Candidate, Case, Context, simulate


def _episode() -> list[Case]:
    # Reviewer busy to t=5. At t=5 both waiting tickets are endorsed NOW, but
    # the oldest expires at 9; the higher-scoring one can still start at 10.
    return [
        Case(0, 0, 0.01, 1, None, False, False),
        Case(1, 1, 0.20, 1, 1, True, False),
        Case(3, 3, 0.90, 1, 3, True, False),
    ]


def test_learns_dispatch_from_future_service_not_merely_immediate_label() -> None:
    source = _episode()
    # Independent exploratory action draws on replicated toy training queues.
    train = train_dispatch([source] * 30, reviewers=1, seed=2026, service_minutes=5, deadline_minutes=8)
    baseline = simulate(source, policy="risk", reviewers=1, service_minutes=5, deadline_minutes=8)
    stats: Counter[str] = Counter()
    learned = simulate(
        source, policy="learned", select=train.selector(stats), reviewers=1, service_minutes=5, deadline_minutes=8
    )
    assert baseline["catches"] == 1
    assert learned["catches"] == 2
    assert stats["contested"] == stats["choice_differs"] == stats["nonrisk_selected"] == 1
    assert train.diagnostics["exploratory_actions"]["fifo"] >= 6
    assert train.diagnostics["exploratory_actions"]["risk"] >= 6
    assert train.diagnostics["exploratory_actions"]["aging"] == 0  # duplicate nomination
    assert train.diagnostics["learned_nonrisk_states"] == 1
    assert (
        train.table
        == train_dispatch([source] * 30, reviewers=1, seed=2026, service_minutes=5, deadline_minutes=8).table
    )


def test_discounted_transition_includes_uncontested_starts_and_no_terminal_bootstrap() -> None:
    a: State = (3, False, True, False, True)
    b: State = (7, False, True, True, True)
    data: _Transitions = {}
    _collect(data, [(a, 0), None, (b, 1)], [1, 2, 4], [True, False, True, True, True])
    assert data[a][0].count == 1
    assert data[a][0].reward == pytest.approx(DISCOUNT + DISCOUNT**2)
    assert data[a][0].next_weights == {b: pytest.approx(DISCOUNT**3)}
    assert data[b][1].reward == 1
    assert not data[b][1].next_weights


def test_bellman_backup_can_prefer_a_lower_immediate_reward_to_reach_a_better_next_queue() -> None:
    a: State = (3, True, False, False, True)
    b: State = (3, False, True, True, True)
    moments = {
        # Risk has 0.5 immediate reward and ends; FIFO earns 0 now but
        # reaches another *choice state* with a certain timely catch.
        a: {0: _Moment(20, 10.0), 1: _Moment(20, 0, {b: 20 * DISCOUNT})},
        b: {0: _Moment(20, 20), 1: _Moment(20, 20)},
    }
    rules, steps = _fit_table(moments)
    assert rules[a] == 1
    assert rules[b] == 0
    assert steps >= 3


def test_unseen_state_and_no_contention_default_to_risk_without_feedback() -> None:
    p = train_dispatch([_episode()[:1]], reviewers=2, seed=0)
    assert p.diagnostics["distinct_choice_starts"] == 0
    candidates = [Candidate(0.2, 4, 1), Candidate(0.9, 1, 5)]
    assert p(candidates, Context(2, 1, 0, 10)) == 1
    assert p(candidates, Context(2, 1, 10_000, 0)) == 1


def test_changing_test_labels_changes_only_outcomes_not_learned_dispatch() -> None:
    p = DispatchPolicy({}, 8, {})
    orig = _episode()
    relabeled = [Case(c.arrival, c.nomination, c.q, c.turn, None, False, c.peak) for c in orig]
    a: Counter[str] = Counter()
    b: Counter[str] = Counter()
    r1 = simulate(orig, reviewers=1, service_minutes=5, deadline_minutes=8, policy="learned", select=p.selector(a))
    r2 = simulate(relabeled, reviewers=1, service_minutes=5, deadline_minutes=8, policy="learned", select=p.selector(b))
    assert r1["starts"] == r2["starts"]
    assert r1["catches"] > r2["catches"]
    assert a == b


def test_feedback_audit_and_training_episodes_require_valid_support() -> None:
    with pytest.raises(ValueError, match="lengths"):
        _collect({}, [None], [], [])
    with pytest.raises(ValueError, match="increase"):
        _collect({}, [None, None], [0, 0], [True])
    with pytest.raises(ValueError, match="training placement"):
        train_dispatch([], reviewers=1, seed=3)
    with pytest.raises(ValueError, match="finite team"):
        train_dispatch([_episode()], reviewers=0, seed=3)
