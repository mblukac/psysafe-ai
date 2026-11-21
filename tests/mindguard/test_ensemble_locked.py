import pytest

from benchmarks.mindguard.ensemble_locked import first_of, frozen_trace, summary
from benchmarks.mindguard.surge_replay import Nomination


def test_first_alert_does_not_wait_for_the_correct_model() -> None:
    late_support = Nomination(3, 0.6, 3, True)
    early = Nomination(1, 0.8, 3, False)
    missing = Nomination(None, None, 3, False)
    assert first_of([late_support, missing, early]) == Nomination(1, 1.0, 3, False)
    assert first_of([missing, missing]) == missing
    with pytest.raises(ValueError, match="unaligned"):
        first_of([early, Nomination(None, None, None, False)])
    with pytest.raises(ValueError, match="different labels"):
        first_of([late_support, Nomination(3, 0.3, 3, False)])


def test_keeps_negatives_on_clock_and_ticket_at_exact_deadline() -> None:
    paths = [Nomination(1, 0.8, None, False), Nomination(1, 0.8, 1, True)]
    # Negative starts at zero. At 20, first positive has expired but the second is exactly on deadline.
    assert frozen_trace(paths, [0, 9, 10], [0, 1, 1]) == [0, 0, 1]


def test_work_cap_does_not_remove_positive_cases_or_select_hits() -> None:
    blocks = [
        dict(weight=1, fold=0, positives=3, traces={"a": [0], "b": [0, 1]}),
        dict(weight=1, fold=1, positives=2, traces={"a": [0, 1], "b": [0, 0, 1]}),
    ]
    r = summary(blocks, ("a", "b"))
    assert r["zero_cap_folds"] == [0]
    assert r["minutes"] == 10
    assert r["recall"] == {"a": 1 / 5, "b": 0}
