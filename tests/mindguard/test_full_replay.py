from dataclasses import replace

import pytest

from benchmarks.mindguard.full_panel import PanelTurn
from benchmarks.mindguard.full_replay import (
    Job,
    calibrated,
    encoded,
    endpoint_counts,
    loss_rule,
    matched_work,
    prepare_hold,
    reference,
    run_job,
    run_mixed,
    summarize,
)
from benchmarks.mindguard.surge_queue import Candidate, Case, Context, simulate
from benchmarks.mindguard.surge_replay import Nomination, attach, nominate


def panel(d: str, t: int, votes: tuple[str, ...], sh: float = 0.15, other: float = 0.1) -> PanelTurn:
    return PanelTurn(d, d, t, votes, {"jev": {"safe": 1 - sh - other, "self_harm": sh, "harm_others": other}})


SAFE = ("safe",) * 3
SH = ("self_harm", "safe", "self_harm")
OTH = ("harm_others", "harm_others", "safe")


def test_union_outcome_is_not_union_of_category_majorities_and_calibration_keeps_gaps() -> None:
    rows = [panel("a", 1, SAFE, 0.01, 0), panel("a", 3, ("self_harm", "safe", "harm_others"), 0.22, 0.75)]
    enc = encoded(rows, "jev")
    assert enc[1].scores[0] == pytest.approx(0.97)
    assert enc[1].votes == ("self_harm",)
    paths = calibrated(enc, enc)
    assert [t for t, _, _ in paths["a"]] == [1, 3]
    nom = nominate(paths["a"], 0.5)
    assert nom.turn == nom.onset_turn == 3
    ref = reference(rows, nom)
    assert ref.firsts == (None, None)
    assert ref.first_gap == 2
    with pytest.raises(ValueError, match="sum outside"):
        encoded([panel("bad", 1, SAFE, 1, 0.2)], "jev")


def test_one_desk_can_be_late_for_union_but_timely_for_subtype_and_gap_bound_is_conservative() -> None:
    # Gap at turn 1 in the first path; the second has no observed endorsement
    # but one unrated turn. A third positive has an inclusive deadline tie.
    scored = [
        [panel("a", 2, SH), panel("a", 8, OTH)],
        [panel("b", 1, SAFE), panel("b", 3, SAFE)],
        [panel("c", 1, SH)],
    ]
    noms = [Nomination(8, 0.9, 2, True), Nomination(None, None, None, False), Nomination(1, 0.9, 1, True)]
    refs = [reference(path, nom) for path, nom in zip(scored, noms, strict=True)]
    cases = attach(noms, [0, 10, 0], [0, 1, 2], [False] * 3)
    starts = {0: 14.0, 2: 10.0}
    actual = endpoint_counts(cases, refs, [0, 1, 2], starts)
    assert (actual["self_harm_positives"], actual["self_harm_catches"]) == (2, 1)
    assert (actual["harm_others_positives"], actual["harm_others_catches"]) == (1, 1)
    assert (actual["possible_positives"], actual["all_gaps_endorsed_catches"]) == (3, 1)


def test_postdecision_audit_and_work_match_use_dispatch_order_not_session_ordinal() -> None:
    cases = [
        Case(0, 0, 0.2, 1, None, False, False),
        Case(0, 0, 0.9, 1, 0, True, False),
        Case(40, 40, 0.8, 1, 40, True, True),
    ]
    starts: dict[int, float] = {}
    stats = simulate(cases, reviewers=1, policy="risk", start_times=starts)
    assert list(starts.items()) == [(1, 0), (2, 40)]
    assert stats["catches"] == 2
    # Deliberately short arm. Budget of one start; no cherry-picking.
    assert matched_work(cases, {"risk": starts, "fifo": {0: 0}}) == (1, {"risk": 1, "fifo": 0})


def test_learned_loss_rule_has_no_hidden_endpoint_input_to_selector() -> None:
    training = [Nomination(8, 0.8, 3, True)] * 12 + [Nomination(1, 0.2, None, False)] * 12
    choose = loss_rule(training, 20, 2)
    candidates = [Candidate(0.2, 0, 1), Candidate(0.8, 0, 8)]
    assert choose(candidates, Context(2, 2, 0, 100)) == 1


def make_rows(prefix: str, n: int) -> list[PanelTurn]:
    return [
        panel(prefix + str(i), t, OTH if t == 3 and i % 4 == 0 else SAFE, 0.01, 0.98 if t == 3 and i % 4 == 0 else 0.04)
        for i in range(n)
        for t in (1, 3)
    ]


def test_full_fold_uses_all_gappy_paths_and_same_gate_for_every_dispatch_rule() -> None:
    raw = make_rows("test", 4)
    job = Job(
        "jev",
        0,
        encoded(make_rows("train", 12), "jev"),
        encoded(raw, "jev"),
        raw,
        [i / 10 for i in range(12)] + [70, 71, 130, 131, 190, 250, 310, 370],
        None,
        None,
        (1, 0),
        20,
        9,
        1,
        1,
        "liberal",
    )
    records, diag = run_job(job)
    assert diag["heldout_turns"] == 8
    assert diag["positive_heldout_paths"] == 1
    values = summarize(records, 20)
    assert {v["policy"] for v in values} == {"fifo", "risk", "aging", "loss_wait", "learned", "best_simple"}
    assert {v["positives"] for v in values} == {5}
    assert {v["possible_positives"] for v in values} == {20}
    for v in values:
        assert v["harm_others_recall"] == v["recall"]
        assert v["self_harm_recall"] is None
        assert v["all_gaps_endorsed_recall"] <= v["recall"] <= v["oracle_recall_bounds"][0]
        assert v["matched_minutes"] <= v["mean_review_minutes"]
        assert v["matched_recall"] <= v["recall"]
        if v["policy"] == "fifo":
            assert v["matched_delta_vs_fifo"] == 0
    with pytest.raises(ValueError, match="shared family"):
        prepare_hold(replace(job, train=job.hold + job.train))


def test_mixed_sensitivity_keeps_all_sources_and_only_untrained_rules() -> None:
    rows = make_rows("family", 12)
    halves = [[r for r in rows if int(r.dialogue_id[6:]) % 2 == i] for i in (0, 1)]
    jobs = [
        Job(
            "jev",
            i,
            encoded(halves[1 - i], "jev"),
            encoded(held, "jev"),
            held,
            [0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 15.0, 30.0, 30.0, 50.0, 60.0, 70.0],
            None,
            None,
            (0, 1, 2, 3),
            20,
            9,
            1,
            1,
            "liberal",
        )
        for i, held in enumerate(halves)
    ]
    rows_out, audit = run_mixed(jobs)
    assert sum(a["heldout_paths"] for a in audit) == 12
    assert {r["metrics"]["positives"] for r in rows_out} == {3}
    assert {r["policy"] for r in rows_out} == {"fifo", "risk", "aging"}
    with pytest.raises(ValueError, match="overlapping holdout"):
        run_mixed([jobs[0], jobs[0]])
