from dataclasses import replace

import pytest

from benchmarks.mindguard.full_shifts import DAY, END, TRAIN_END, fifo, next_shift, peak_start, training_clock
from benchmarks.mindguard.surge_queue import Case, simulate


def case(t: float) -> Case:
    return Case(t, t, 0.9, 1, t, True, False)


def play(cases: list[Case], start: int) -> dict[str, float]:
    return fifo(cases, start, begin=0, end=DAY, score_from=0, score_to=DAY)


def test_shift_wrap_train_only_and_equal_hours() -> None:
    earlier = [0, 20, 23 * 60, DAY + 1, DAY + 20 * 60, DAY + 21 * 60, DAY + 22 * 60]
    offset = peak_start(earlier)
    assert offset == 17 * 60  # 17:00-01:00; deterministic tie break
    for hour in (offset, 9 * 60):
        assert sum(next_shift(float(t), hour)[0] == t for t in range(14 * DAY)) == 14 * 480
        assert next_shift(hour + 480, hour)[0] == hour + DAY
    with pytest.raises(ValueError, match="pre-split"):
        peak_start([*earlier, TRAIN_END])
    with pytest.raises(ValueError, match="pre-split"):
        training_clock([*earlier, TRAIN_END])
    clock = training_clock(earlier)
    assert clock == sorted(clock)
    assert all(t < END and t % TRAIN_END in earlier for t in clock)


def test_available_worker_dispatch_agrees_with_constant_staff() -> None:
    cases = [case(t) for t in (90, 90, 90, 99, 100, 110, 110, 110, 117, 119, 120, 120, 134)]
    for offset, n_staff in [(0, 2), (14 * 60, 1)]:
        actual = play(cases, offset)
        expected = simulate(cases, reviewers=n_staff, policy="fifo")
        for k in ("positives", "starts", "nominated", "catches", "review_minutes"):
            assert expected[k] == actual[k]
        assert actual["overtime_minutes"] == 0


def test_starts_not_preempted_and_shift_end_exclusive() -> None:
    # Base and second both start at 479, one minute before second's shift ends.
    # Ticket at 480 expires before either is free.
    held = [case(479), case(479), case(480)]
    actual = play(held, 0)
    assert (actual["starts"], actual["catches"], actual["review_minutes"], actual["overtime_minutes"]) == (2, 2, 40, 19)
    assert play([case(480), case(480)], 0)["starts"] == 1
    # The queued head has exactly ten minutes of patience remaining at start of shift.
    assert play([case(88), case(90)], 100)["catches"] == 2
    assert play([case(89), case(89)], 100)["catches"] == 1


def test_warmup_followthrough_and_distinct_labor_accounting() -> None:
    # Warm-up busy service blocks the first scored case until its deadline.
    # Second scored session nominated only during follow-through. Count its
    # outcome but not its working minutes during the scoring window.
    cases = [case(89), case(100), replace(case(120), nomination=130, onset=130)]
    r = fifo(cases, 600, begin=0, end=DAY, score_from=100, score_to=125)
    assert (r["sessions"], r["positives"], r["catches"], r["starts"], r["review_minutes"]) == (2, 2, 2, 1, 20)
    flip = [replace(c, supported=False, onset=None) for c in cases]
    s = fifo(flip, 600, begin=0, end=DAY, score_from=100, score_to=125)
    assert (r["starts"], r["overtime_minutes"]) == (s["starts"], s["overtime_minutes"])
    assert s["catches"] == 0
