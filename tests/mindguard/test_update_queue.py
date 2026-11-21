from math import nextafter

import pytest

from benchmarks.mindguard.staffing_queue import Trajectory
from benchmarks.mindguard.update_queue import ARMS, UpdatePath, prepare_path, replay_updates


class Higher:
    def replace(self, original_q: float, challenger_q: float) -> bool:
        return challenger_q > original_q


def prep(path: Trajectory, bayes: tuple[float, ...] | None = None) -> UpdatePath:
    return prepare_path(path, bayes or tuple(r[1] for r in path), 0.65, pair_rule=Higher())


def test_at_start_update_scores_rescue_with_identical_start_and_labor() -> None:
    blocker = prep(((1, 0.9, 0),))
    target = prep(((1, 0.68, 0), (2, 0.70, 0), (3, 0.88, 1), (4, 0.99, 0)))
    who: dict[str, tuple[set[int], set[int]]] = {}
    result = replay_updates(
        [blocker, target], [0, 1], [0, 1], [False] * 2, reviewers=1, service_minutes=5, affected_sources=who
    )
    assert result["starts"] == 2
    assert result["review_minutes"] == 10
    assert result["timely_updatable_starts"] == result["updatable_starts"] == 1
    assert result["timely_positive_starts"] == 1
    assert result["arms"]["frozen"]["catches"] == 0
    for arm in ARMS[1:]:
        assert result["arms"][arm] == dict(catches=1, rescued=1, spoiled=0, changed=1, unsupported_starts=1)
        assert who[arm] == ({1}, set())


def test_before_first_later_message_no_credit_and_after_onset_no_hindsight() -> None:
    blocker = prep(((1, 0.8, 0),))
    # Turn 2 has no rating / score. Turn 3 (arrival+4) is NOT compressed into turn 2.
    target = prep(((1, 0.66, 0), (3, 0.8, 1)))
    for service in (2.0, nextafter(5.0, 0.0)):
        r = replay_updates([blocker, target], [0, 1], [0, 1], [False] * 2, reviewers=1, service_minutes=service)
        assert r["before_onset_starts"] == 1
        assert r["timely_positive_starts"] == 0
        assert r["updatable_starts"] == 0
        assert all(v["catches"] == 0 for v in r["arms"].values())
    # Disclosing a stronger future turn must have no effect when review is immediate.
    immediate = replay_updates([target], [10.15], [0], [False], reviewers=None)
    assert immediate["starts"] == 1
    assert all(v["changed"] == 0 and v["catches"] == 0 for v in immediate["arms"].values())


def test_highest_calibrated_and_highest_bayes_and_latest_can_disagree() -> None:
    blocker = prep(((1, 0.7, 0),))
    target = prep(((1, 0.7, 1), (2, 0.7, 0), (3, 0.2, 0)), (0.8, 0.9, 0.1))
    who: dict[str, tuple[set[int], set[int]]] = {}
    r = replay_updates(
        [target, blocker], [0, 1], [1, 0], [False, False], reviewers=1, service_minutes=6, affected_sources=who
    )
    for name in ("frozen", "max_score", "paired_bayes", "visible_oracle"):
        assert r["arms"][name]["changed"] == 0
        assert r["arms"][name]["catches"] == 1
    for name in ("latest", "latest_qualifying", "bayes"):
        assert r["arms"][name]["spoiled"] == 1
        assert r["arms"][name]["catches"] == 0
        assert who[name] == (set(), {0})


def test_true_deadline_inclusive_but_alert_expiry_not_refreshed() -> None:
    blocker = prep(((1, 0.9, 0),))
    # First evidence at arrival; no alert until turn 2. Last, strongest turn at minute 12.
    p = prep(((1, 0.3, 1), (2, 0.70, 1), (7, 0.95, 1)))
    yes = replay_updates([blocker, p], [0, 0], [0, 1], [False] * 2, reviewers=1, service_minutes=10)
    late = replay_updates([blocker, p], [0, 0], [0, 1], [False] * 2, reviewers=1, service_minutes=12)
    expired = replay_updates([blocker, p], [0, 0], [0, 1], [False] * 2, reviewers=1, service_minutes=13)
    assert all(v["catches"] == 1 for v in yes["arms"].values())
    assert all(v["catches"] == 0 for v in late["arms"].values())
    assert late["after_deadline_starts"] == late["updatable_starts"] == 1
    assert late["expired"] == 0
    assert expired["starts"] == expired["expired"] == 1
    assert all(v["catches"] == 0 for v in expired["arms"].values())


def test_no_nomination_remains_in_denominator_and_zero_staff_is_empty() -> None:
    p = prep(((1, 0.2, 1),))
    q = prep(((1, 0.8, 1), (2, 0.99, 1)))
    r = replay_updates([p, q], [0.0, 0.0], [0, 1], [False, True], reviewers=0)
    assert (r["positives"], r["nominated"], r["starts"], r["expired"], r["review_minutes"]) == (2, 1, 0, 1, 0.0)
    assert all(v["catches"] == 0 for v in r["arms"].values())


def test_no_untrained_pair_or_mixed_panel_and_validate_chronology() -> None:
    z = prepare_path(((1, 0.9, 1),), [0.6], 0.5)
    assert "paired_bayes" not in replay_updates([z], [0], [0], [False], reviewers=None)["arms"]
    with pytest.raises(ValueError, match="consistent"):
        replay_updates([z, prep(((1, 0.9, 1),))], [0], [0], [False], reviewers=1)
    with pytest.raises(ValueError, match="chronological"):
        prep(((1, 0.9, 1), (1, 0.8, 0)))
    with pytest.raises(ValueError, match="invalid path"):
        prepare_path(((1, 0.9, 1),), [], 0.5)
