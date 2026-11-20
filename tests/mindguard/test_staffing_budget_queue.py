import pytest

from benchmarks.mindguard.staffing_queue import Session, simulate


def session(at: int, *turns: tuple[float, int]) -> Session:
    return Session(at, tuple((i, s, y) for i, (s, y) in enumerate(turns, 1)))


def run(
    sessions: list[Session],
    policy: str,
    audit: list[bool] | None,
    thresholds: list[float] | None = None,
    positions: list[int] | None = None,
) -> dict[str, float | int | None]:
    return simulate(
        sessions,
        reviewers=3,
        threshold=0.5,
        policy=policy,
        metric=lambda value, _: value,
        service_minutes=1,
        turn_minutes=1,
        deadline_minutes=1,
        starts_audit=audit,
        session_thresholds=thresholds,
        starts_session_ids=positions,
    )


@pytest.mark.parametrize(
    ("policy", "first_three"),
    [("fixed", [False, True, False]), ("paced", [True, False, False]), ("rl", [True, False, False])],
)
def test_prefix_budget_preserves_simultaneous_dispatch_order(policy: str, first_three: list[bool]) -> None:
    # Fixed breaks ties by input order; both priority policies break score ties
    # by input order, before moving on to the lower score. No fabricated starts.
    rows = [session(0, (0.6, 0)), session(0, (0.9, 1)), session(0, (0.9, 0)), session(3, (0.8, 1))]
    audit: list[bool] = []
    result = run(rows, policy, audit)
    assert result == run(rows, policy, None)
    assert audit == [*first_three, True]
    assert (len(audit[:1]), sum(audit[:1])) == (1, int(first_three[0]))
    assert len(audit) == result["reviews"]
    assert sum(audit) == result["catches_by_onset_deadline"]


def test_audit_only_credits_supported_nomination_by_first_positive_deadline() -> None:
    audit: list[bool] = []
    rows = [
        session(0, (0.9, 0)),  # no endorsement anywhere
        session(3, (0.1, 1), (0.1, 0), (0.9, 1)),  # supported at minute five; onset deadline four
        session(8, (0.1, 1), (0.9, 1)),  # supported exactly on the onset deadline
        session(15, (0.9, 0), (0.1, 1)),  # unsupported first alert
        session(20, (0.1, 1)),  # positive with no start: remains in denominator, not in ordered starts
    ]
    result = run(rows, "rl", audit)
    assert audit == [False, False, True, False]
    assert result["catches"] == 2
    assert result["positives"] == 4
    assert result["catches_by_onset_deadline"] == 1
    assert sum(audit[:2]) == 0
    assert sum(audit[:3]) == 1


@pytest.mark.parametrize("first_gate", [0.3, 0.8])
def test_identical_trajectory_different_thresholds_does_not_share_filtered_candidates(first_gate: float) -> None:
    rows = [session(0, (0.4, 0), (0.9, 1)), session(4, (0.4, 0), (0.9, 1))]
    audit: list[bool] = []
    other_gate = 0.8 if first_gate == 0.3 else 0.3
    run(rows, "paced", audit, thresholds=[first_gate, other_gate])
    assert audit == [first_gate == 0.8, other_gate == 0.8]


def test_adjusting_fold_gate_does_not_reorder_by_margin() -> None:
    # Raw priority 0.9 > 0.8, but acceptance margins 0.01 < 0.7.
    rows = [session(0, (0.8, 0)), session(0, (0.9, 1))]
    audit: list[bool] = []
    run(rows, "paced", audit, thresholds=[0.1, 0.89])
    assert audit == [True, False]


def test_started_positions_align_with_hit_audit_through_dispatch_ties() -> None:
    rows = [session(2, (0.95, 0)), session(0, (0.7, 0)), session(0, (0.9, 1)), session(0, (0.9, 0))]
    starts: list[int] = []
    hits: list[bool] = []
    assert run(rows, "paced", hits, positions=starts) == run(rows, "paced", None)
    assert starts == [2, 3, 1, 0]
    assert hits == [bool(rows[sid].turns[0][2]) for sid in starts]


def test_per_session_cutoff_gets_live_surcharge_on_top() -> None:
    rows = [session(0, (1.0, 0)) for _ in range(3)] + [Session(0.1, ((1, 0.56, 1),))]
    no_pressure = run(rows, "fixed", [], thresholds=[0, 0, 0, 0.5])
    pressured = run(rows, "paced", [], thresholds=[0, 0, 0, 0.5])
    assert no_pressure["alerts"] == 4
    assert pressured["alerts"] == 3


@pytest.mark.parametrize("thresholds", [[], [float("nan")], [0.5, 0.5]])
def test_per_session_cutoff_input_is_validated(thresholds: list[float]) -> None:
    with pytest.raises(ValueError, match="one finite cutoff per session"):
        run([session(0, (0.4, 0))], "fixed", None, thresholds=thresholds)
