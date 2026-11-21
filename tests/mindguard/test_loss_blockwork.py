import pytest

from benchmarks.mindguard.loss_blockwork import compare
from benchmarks.mindguard.loss_workmatch import GATES, plan


def test_daily_caps_cover_later_arrivals_at_equal_cost_and_keep_negative_reviews() -> None:
    # Liberal serves the first two low-risk negatives each day. Tuned serves
    # its two positives. Only arrivals through day one's final start have
    # been seen when the whole-stream liberal budget is exhausted.
    paths = {
        g: [plan(((1, q, y),), cut) for q, y in ((0.2, 0), (0.9, 1))] for g, cut in zip(GATES, (0.8, 0.1), strict=True)
    }
    r = compare(paths, [0.0, 30.0, 60.0, 90.0, 1440.0, 1470.0, 1500.0, 1530.0], [0, 0, 1, 1, 0, 0, 1, 1])
    assert r["positives"] == 4
    assert r["day_caps"] == [2, 2]
    assert r["full_cap"] == 4
    assert r["full_hits"] == [4, 2, 2]
    assert r["day_hits"] == [4, 0, 0]
    assert r["full_arrival_fraction"] == [1.0, 0.5]


def test_cohort_uses_arrival_not_release_and_keeps_zero_budget_positives() -> None:
    # First conversation's flag is after midnight. In continuous service the
    # second conversation is already in service; in separate arrival cohorts
    # both can drain in isolation, including full post-boundary service costs.
    paths = {
        g: [plan(((1, 0.0, 0), (2, 0.9, 1)), 0.5), plan(((1, 0.9, 1),), 0.5), plan(((1, 0.0, 1),), 0.5)] for g in GATES
    }
    r = compare(paths, [1439.0, 1440.0, 2880.0], [0, 1, 2])
    assert r["positives"] == 3
    assert r["full_cap"] == 1
    assert r["day_caps"] == [1, 1, 0]
    assert r["day_hits"] == [2, 2, 2]


def test_zero_cap_has_no_fake_last_start() -> None:
    paths = {g: [plan(((1, 0.0, 1),), 0.5)] for g in GATES}
    r = compare(paths, [0.0], [0])
    assert r["full_last_days"] == []
    assert r["day_caps"] == [0]
    assert r["positives"] == 1
    with pytest.raises(ValueError, match="nonempty chronological"):
        compare(paths, [], [])
