import math

import pytest

from benchmarks.mindguard.surge_queue import Candidate, Case, Context, rule_index, simulate


def ticket(at: float, q: float, y: bool, *, onset: float | None = None, peak: bool = False) -> Case:
    return Case(min(at, onset) if onset is not None else at, at, q, 1, at if onset is None and y else onset, y, peak)


def test_locked_nominees_dispatch_differs_by_policy_at_burst_not_by_gold() -> None:
    rows = [ticket(0, 0.7, False), ticket(0, 0.95, True, peak=True)]
    weak = simulate(rows, reviewers=1, policy="fifo")
    strong = simulate(rows, reviewers=1, policy="risk")
    assert weak["sessions"] == weak["nominated"] == strong["nominated"] == 2
    assert weak["starts"] == strong["starts"] == weak["contested_starts"] == 1
    assert weak["catches"] == 0
    assert weak["misses_queue_delay"] == weak["misses_queue_expired"] == 1
    assert strong["catches"] == strong["by_period"]["peak"]["catches"] == 1
    assert weak["p95_wait_started"] == 0
    assert weak["p95_wait_nominated"] == 10
    assert weak["review_minutes"] == 20
    assert weak["utilization"] == pytest.approx(20 / 43200)


def test_no_queue_oracle_is_not_oracle_of_nomination_and_preserves_miss_partition() -> None:
    cases = [
        ticket(1, 0.5, True, peak=True),
        ticket(4, 0.99, False, onset=25),  # positive later, wrong locked prefix
        ticket(10, 0.8, True, onset=0),  # inclusive onset deadline
        ticket(10.1, 0.8, True, onset=0),  # right prefix, but nomination already too late
        Case(8, None, None, None, 12, False, False),  # positive with no nomination
        Case(8, None, None, None, None, False, False),
    ]

    def no_contest(_: tuple[Candidate, ...], ctx: Context) -> int:
        raise AssertionError(f"no scarce choice in unrestricted service at {ctx.elapsed}")

    base = simulate(cases, reviewers=None, policy="fifo")
    other = simulate(cases, reviewers=None, policy="learned", select=no_contest)  # type: ignore[arg-type]
    assert base == other
    assert base["sessions"] == 6
    assert base["starts"] == base["nominated"] == 4
    assert base["catches"] == 2
    assert base["positives"] == 5
    assert base["misses_wrong_turn"] == base["misses_no_alert"] == base["misses_nomination_late"] == 1
    assert base["contested_starts"] == base["expired"] == base["misses_queue_delay"] == 0
    assert base["mean_wait_started"] == base["p95_wait_nominated"] == 0
    assert base["by_period"]["peak"]["recall"] == 1.0
    assert base["by_period"]["offpeak"]["recall"] == 0.25


def test_start_on_expiry_allowed_late_wait_separate_from_late_nomination() -> None:
    rows = [ticket(0, 0.8, False), ticket(10, 0.8, True, onset=1)]
    # The first review occupies the server until time 20. The second starts
    # on its ticket deadline (20), not by its earlier onset deadline (11).
    r = simulate(rows, reviewers=1)
    assert r["starts"] == 2
    assert r["expired"] == 0
    assert r["misses_queue_started_late"] == r["misses_queue_delay"] == 1
    assert r["misses_nomination_late"] == r["misses_queue_expired"] == 0
    assert r["mean_wait_started"] == 5
    assert r["p95_wait_started"] == 10


def test_current_candidate_features_only_and_audits_align_to_contests_post_rollout() -> None:
    rows = [ticket(0, 0.2, False), ticket(0, 0.3, True), ticket(0, 0.4, False), ticket(5, 0.9, True)]
    calls = []
    contests: list[int] = []
    hits: list[bool] = []

    def selector(candidates: tuple[Candidate, ...], ctx: Context) -> int:
        assert not contests
        assert not hits  # cannot read an outcome on-line even through the supplied reference
        calls.append((candidates, ctx))
        return rule_index(candidates, "risk")

    r = simulate(
        rows,
        reviewers=2,
        service_minutes=5,
        policy="learned",
        select=selector,  # type: ignore[arg-type]
        horizon_minutes=4,
        contest_starts=contests,
        start_hits=hits,
    )
    assert [c.elapsed for _, c in calls] == [0, 0]
    assert [c.remaining for _, c in calls] == [4, 4]
    assert [c.waiting for _, c in calls] == [3, 2]
    assert [v.q for v in calls[0][0]] == [0.2, 0.3, 0.4]
    assert [v.age for v in calls[0][0]] == [0, 0, 0]
    assert contests == [0, 1]
    assert hits == [False, True, False, True]
    assert len(hits) == r["starts"]
    assert sum(hits) == r["catches"]
    assert r["utilization"] == 1  # occupancy clipped to stated horizon [0,4]


def test_age_changes_winner_and_ties_are_stable() -> None:
    cands = [Candidate(0.61, 9, 1), Candidate(0.65, 1, 9)]
    assert rule_index(cands, "risk") == 1
    assert rule_index(cands, "aging") == 0
    assert rule_index([Candidate(0.9, 1, 3), Candidate(0.9, 1, 50)], "risk") == 0
    assert rule_index([Candidate(0.9, 1, 3), Candidate(0.9, 1, 50)], "aging") == 0


def test_zero_staff_and_zero_positive_periods() -> None:
    r = simulate([ticket(0, 0.8, True, peak=True)], reviewers=0)
    assert r["starts"] == r["catches"] == r["contested_starts"] == 0
    assert r["misses_queue_expired"] == 1
    assert r["by_period"]["offpeak"]["recall"] is None
    assert r["p95_wait_started"] is None
    assert r["p95_wait_nominated"] == 10


@pytest.mark.parametrize(
    "case",
    [
        Case(3, 1, 0.5, 1, None, False, False),
        Case(0, None, 1, None, 3, False, False),
        Case(0, 1, 0.5, 2, None, True, False),
        Case(0, 1, 0.5, 0, 0, True, False),
        Case(0, 1, math.nan, 1, None, False, False),
    ],
)
def test_inconsistent_case_rejected(case: Case) -> None:
    with pytest.raises(ValueError, match=r"ticket|nomination"):
        simulate([case], reviewers=1)


def test_invalid_action_and_horizon_rejected() -> None:
    with pytest.raises(ValueError, match="feasible"):
        simulate([ticket(0, 0.9, False)] * 2, reviewers=1, policy="learned", select=lambda *_: 5)
    with pytest.raises(ValueError, match="horizon"):
        simulate([], reviewers=1, horizon_minutes=math.inf)
