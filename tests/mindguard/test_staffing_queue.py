import math
from collections.abc import Callable

import pytest

from benchmarks.mindguard.staffing_queue import Session, simulate


def chat(time: float, *scores_labels: tuple[float, int]) -> Session:
    return Session(time, tuple((t, score, label) for t, (score, label) in enumerate(scores_labels, 1)))


def identity(score: float, index: int) -> float:
    return score


def run(
    chats: list[Session],
    policy: str = "fixed",
    staff: int = 1,
    service: float = 10,
    deadline: float = 5,
    metric: Callable[[float, int], float] = identity,
) -> dict[str, int | float | None]:
    return simulate(
        chats,
        policy=policy,
        reviewers=staff,
        service_minutes=service,
        deadline_minutes=deadline,
        turn_minutes=1,
        metric=metric,
        threshold=0.5,
    )


def test_bursty_trace_workers_are_not_a_nonrenewing_slot_cap() -> None:
    chats = [chat(t, (1, 1)) for t in (0, 0, 0, 10, 10, 10)]
    one, two, three = (run(chats, staff=n) for n in range(1, 4))
    assert [r["catches"] for r in (one, two, three)] == [2, 4, 6]
    assert one["arrivals"] == one["positives"] == 6
    assert one["alerts"] == 6
    assert one["reviews"] == 2
    assert one["review_minutes"] == 20
    assert one["max_queue"] == 2
    assert one["expired"] == one["misses_late"] == 4
    assert one["misses_no_alert"] == one["misses_wrong_turn"] == 0


def test_deadline_is_inclusive_and_queue_drains_after_final_arrival() -> None:
    just_in_time = run([chat(0, (1, 0)), chat(1, (1, 1))], deadline=9)
    too_late = run([chat(0, (1, 0)), chat(0.5, (1, 1))], deadline=9)
    assert just_in_time["catches"] == 1
    assert just_in_time["mean_wait_minutes"] == 4.5
    assert too_late["catches"] == 0
    assert too_late["misses_late"] == 1


def test_secondary_deadline_measured_from_first_observed_majority_positive() -> None:
    # Onset is at minute 0 in all three dialogues; no interaction or staff congestion.
    # The third is reviewed early enough but on an unsupported prefix: neither measure gives credit.
    chats = [
        chat(0, (0.1, 1), (0.1, 0), (0.9, 1)),
        chat(0, (0.1, 1), (0.9, 1)),
        chat(0, (0.1, 1), (0.9, 0)),
    ]
    result = run(chats, staff=3, deadline=1)
    assert result["catches"] == 2
    assert result["catches_by_onset_deadline"] == 1
    assert result["recall_by_onset_deadline"] == 1 / 3
    assert result["recall"] == 2 / 3


def test_first_nomination_locked_through_service_or_expiry() -> None:
    result = run(
        [
            chat(0, (1, 0)),  # Occupies reviewer until minute 10.
            chat(1, (0.9, 0), (1, 1)),  # Later evidence cannot repair an unsupported ticket.
            chat(2, (0.1, 1)),  # Not nominated.
            chat(3, (1, 1)),  # Right turn; expires.
        ]
    )
    assert result["positives"] == 3
    assert result["misses_wrong_turn"] == result["misses_no_alert"] == result["misses_late"] == 1
    assert result["reviews"] == result["unsupported_reviews"] == 1
    assert result["catches"] == 0


def test_paced_congestion_and_priority_are_label_blind_and_shared_by_rl() -> None:
    chats = [chat(0, (1, 0)), chat(1, (0.54, 0)), chat(8, (0.7, 0)), chat(8, (0.9, 1))]
    fixed = run(chats)
    assert fixed["alerts"] == 4
    assert fixed["catches"] == 0  # FIFO takes first simultaneous ticket, other misses deadline.
    paced = run(chats, "paced")
    assert paced["alerts"] == 3  # Marginal ticket is suppressed during congestion.
    assert paced["reviews"] == 2
    assert paced["catches"] == 1
    assert run(chats, "rl") == paced  # Same metric implies exactly the same decisions.
    relabeled = [Session(c.arrival, tuple((i, s, 1 - y) for i, s, y in c.turns)) for c in chats]
    counter = run(relabeled, "paced")
    for name in ("alerts", "reviews", "mean_wait_minutes", "max_queue", "expired", "review_minutes"):
        assert counter[name] == paced[name]


def test_pacing_ages_tickets_without_refreshing_their_deadline() -> None:
    # At time 10 the older ticket's .03 age advantage beats a .02 gap in the raw score.
    old, new = chat(0, (0.70, 1)), chat(3, (0.72, 0))
    result = run([chat(0, (1, 0)), old, new], "paced", deadline=10)
    assert result["catches"] == result["expired"] == 1
    assert result["reviews"] == 2


def test_simultaneous_candidates_share_one_snapshot_and_one_dispatch() -> None:
    chats = [chat(0, (0.6, 0)), chat(0, (0.9, 1))]
    assert run(chats, "paced")["catches"] == 1
    assert run(chats[::-1], "paced")["catches"] == 1
    assert run(chats)["catches"] == 0  # FIFO tie-break is input order, not label.


def test_zero_staff_and_empty_positive_stratum_do_not_invent_recall() -> None:
    empty = run([chat(0, (1, 0))], "rl", staff=0)
    assert empty["recall"] is empty["mean_wait_minutes"] is None
    assert empty["alerts"] == empty["expired"] == 1
    assert empty["reviews"] == 0
    assert run([chat(0, (-0.5, 1))], "rl")["recall"] == 0


@pytest.mark.parametrize(
    "session",
    [
        Session(0, ()),
        Session(-1, ((1, 0, 0),)),
        Session(0, ((2, 0, 0),)),
        Session(0, ((1, math.nan, 1),)),
        Session(0, ((1, 1, 2),)),
    ],
)
def test_invalid_or_unrated_records_are_not_silently_negatives(session: Session) -> None:
    with pytest.raises(ValueError, match=r"empty session|arrival minutes|one-based|binary reference"):
        run([session])


def test_invalid_configuration_fails() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        run([], service=0)
    with pytest.raises(ValueError, match="nonnegative"):
        run([], staff=-1)
    with pytest.raises(ValueError, match="policy"):
        run([], policy="clairvoyant")
    with pytest.raises(ValueError, match="metric"):
        run([chat(0, (0.1, 1))], metric=lambda *_: math.nan)
