import pytest

from benchmarks.mindguard.ensemble_replay import (
    BRIDGE,
    ENSEMBLE,
    NAMES,
    SELECTED,
    SINGLES,
    flag_status,
    match_block,
    scored_order,
    tranche,
)
from benchmarks.mindguard.surge_replay import Nomination


def test_paid_prefix_and_pair_caps_include_negative_starts() -> None:
    other = SINGLES[0]
    scores = {m: [-1, 1, 2] for m in NAMES}
    scores[SINGLES[-1]] = []
    scores[ENSEMBLE] = [0, 1]
    matched = match_block(scores, other)
    assert matched["all"][0] == 0
    assert not matched["all"][1][ENSEMBLE]
    assert matched["selected_pair"] == (2, {ENSEMBLE: {0, 1}, SELECTED: {1}})
    assert matched["pair_" + SINGLES[-1]][0] == 0
    assert match_block(scores, other)["pair_" + other][1][other] == {1}


def test_differences_are_paired_episode_not_independent_margins() -> None:
    arm = SINGLES[0]
    synthetic = [
        {"paired": (2, {arm: {0, 1}, ENSEMBLE: {1, 2}})},
        {"paired": (1, {arm: {3}, ENSEMBLE: {4}})},
    ]
    out = tranche(synthetic, [0, 1, 2, 0, 2], ["A", "B", "C"])["paired"]
    assert out["starts"] == 3
    assert out["paired"][arm]["rescued"] == out["paired"][arm]["lost"] == 2
    assert out["paired"][arm]["gf"] == {"C": 2}
    assert out["paired"][arm]["lf"] == {"A": 2}
    status = {arm: [BRIDGE[3], BRIDGE[3], BRIDGE[0]], ENSEMBLE: [BRIDGE[1], BRIDGE[3], BRIDGE[3]]}
    b = tranche(synthetic, [0, 1, 2, 0, 2], ["A", "B", "C"], status=status, nobody={2})["paired"]["paired"][arm][
        "bridge"
    ]
    assert b["gained_never_flagged"] == b["gained_all_six_never_flagged"] == {2: 2}
    assert b["lost_flagged_unsupported"] == {0: 2}
    assert not b["lost_never_flagged"]
    assert flag_status(Nomination(7, 0.9, 1, True)) == "supported_flag_late"
    assert flag_status(Nomination(6, 0.9, 1, True)) == "supported_flag_in_time"


def test_spent_on_early_late_unsupported_or_negative_and_inclusive_deadline() -> None:
    p = [
        Nomination(1, 0.9, 2, False),
        Nomination(1, 0.9, None, False),
        Nomination(2, 0.9, 1, True),
        Nomination(1, 0.9, 1, True),
        Nomination(1, 0.9, 1, True),
    ]
    # At minute 20, ticket 3 can still start at exact expiry and deadline.
    # Ticket 4 expires behind it. At 40 ticket 2 starts by ticket expiry
    # but too late relative to need. Negative ticket 1 still pays for a start.
    times = [0, 70, 29, 10, 11]
    tickets = [(0.0, 0, 0.9), (10.0, 3, 0.9), (11.0, 4, 0.9), (31.0, 2, 0.9), (70.0, 1, 0.9)]
    assert scored_order(tickets, 1, times, list(range(5)), p) == [-1, 3, -1, -1]
    with pytest.raises(ValueError, match="incomplete"):
        match_block({}, "unknown")
