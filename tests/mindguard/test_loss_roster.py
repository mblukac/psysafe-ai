from dataclasses import replace

import pytest

from benchmarks.mindguard.full_shifts import DAY, FIXED, fifo
from benchmarks.mindguard.loss_roster import fifo_trace, paired, summarize
from benchmarks.mindguard.surge_queue import Case


def case(at: float, *, supported: bool = True) -> Case:
    return Case(at, at, 0.8, 1, at if supported else None, supported, False)


def test_cap_counts_negatives_and_stops_both_arms_in_dispatch_order() -> None:
    # Fixed can take both 580 tickets at once. All 3 actual starts in each arm
    # include the negative first job; the dropped later positive still counts.
    cases = [case(580, supported=False), case(580), case(700), case(800)]
    r = paired(cases, 0, begin=0, switch=0, close=900, stop=DAY)
    assert (r["sessions"], r["positives"], r["starts_per_arm"], r["minutes_per_arm"]) == (4, 3, 3, 60)
    assert (r["fixed_uncapped_minutes"], r["peak_uncapped_minutes"]) == (80, 60)
    assert (r["fixed_matches"], r["peak_matches"]) == (2, 2)


def test_warmup_state_pending_warm_cases_and_outcome_work_consistent_in_drain() -> None:
    cases = [case(80), case(99), case(100), replace(case(125), nomination=140, onset=140)]
    r = paired(cases, 100, begin=0, switch=100, close=130, stop=200)
    assert (r["sessions"], r["positives"], r["minutes_per_arm"]) == (2, 2, 40)
    assert (r["fixed_warm_carry_starts"], r["peak_warm_carry_starts"]) == (1, 1)
    assert (r["fixed_drain_starts"], r["peak_drain_starts"]) == (1, 0)
    assert (r["fixed_matches"], r["peak_matches"]) == (1, 1)
    with pytest.raises(ValueError, match="enrollment"):
        paired([*cases, case(130)], 100, begin=0, switch=100, close=130, stop=200)


def test_identical_rosters_tie_and_match_existing_fifo_without_warm_or_drain() -> None:
    cases = [case(t) for t in (500, 520, 521, 535, 536, 537, 542, 543, 545, 550, 1020, 1021, 1022)]
    original = fifo(cases, FIXED, begin=0, end=DAY, score_from=0, score_to=DAY)
    r = paired(cases, FIXED, begin=0, switch=0, close=DAY, stop=DAY)
    assert original["review_minutes"] == r["minutes_per_arm"]
    assert original["catches"] == r["fixed_matches"] == r["peak_matches"]
    assert fifo_trace(cases, FIXED, 0, 540, DAY) == fifo_trace(
        [replace(c, supported=False, onset=None) for c in cases], FIXED, 0, 540, DAY
    )


def test_aggregate_pools_before_dividing() -> None:
    a = paired([case(580, supported=False), case(580), case(700)], 0, begin=0, switch=0, close=900, stop=DAY)
    b = paired([case(700)], 0, begin=0, switch=0, close=900, stop=DAY)
    r = summarize([(0, 0, 2, a), (1, 0, 1, b)])
    assert r["minutes_per_arm"] == pytest.approx((2 * 40 + 20) / 3)
    assert r["fixed_recall"] == pytest.approx((2 * a["fixed_matches"] + b["fixed_matches"]) / 5)
