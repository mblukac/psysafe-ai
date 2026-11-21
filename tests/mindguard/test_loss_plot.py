import pytest

from benchmarks.mindguard.loss_plot import percentages


def test_plot_moves_late_first_flags_out_of_access_failures() -> None:
    rows = [
        dict(
            model=str(i),
            positives=20,
            no_ticket=2,
            late_first_flag=1,
            unserved_or_late_despite_flag_by_deadline=4,
            pre_onset_pickup=1,
            timely_no_supported_turn_available=1,
            timely_stale=2,
            caught=9,
        )
        for i in range(6)
    ]
    assert percentages(rows) == [15, 20, 5, 5, 10, 45]
    with pytest.raises(ValueError, match="six distinct"):
        percentages(rows[:-1])
    with pytest.raises(ValueError, match="same positive cases"):
        percentages([*rows[:-1], {**rows[-1], "positives": 21}])
    with pytest.raises(ValueError, match="incomplete loss partition"):
        percentages([*rows[:-1], {**rows[-1], "caught": 8}])
