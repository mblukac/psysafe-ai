from typing import Any

import pytest

from benchmarks.mindguard.data import Turn
from benchmarks.mindguard.surge_queue import Case, simulate
from benchmarks.mindguard.surge_replay import Job, Nomination, aggregate, attach, fold_run, nominate


def test_nomination_uses_first_crossing_and_nothing_from_future() -> None:
    prefix = ((1, 0.2, 0), (2, 0.5, 0))
    # A more convincing later message cannot replace the already locked ticket.
    locked = nominate((*prefix, (3, 0.99, 1)), 0.5)
    assert locked == Nomination(2, 0.5, 3, False)
    # Replacing the later score and the entire reference trajectory changes
    # only evaluation fields, not the observed score or chosen turn.
    other = nominate(((1, 0.2, 1), (2, 0.5, 1), (3, 0.01, 0)), 0.5)
    assert (locked.turn, locked.score) == (other.turn, other.score)
    assert other == Nomination(2, 0.5, 1, True)
    assert nominate(((1, 0.01, 0), (2, 0.02, 1)), 0.5) == Nomination(None, None, 2, False)
    assigned = attach([locked, other], [10, 66], [1, 0], [False, True])
    assert assigned == [Case(10, 12, 0.5, 2, 10, True, False), Case(66, 68, 0.5, 2, 70, False, True)]


def _two_folds() -> list[dict[str, Any]]:
    # Fold 0: FIFO picks unsupported first, the positive ticket expires; one
    # other positive never nominated. Fold 1: everyone catches a singleton.
    paths = [
        [Nomination(1, 0.60, None, False), Nomination(1, 0.95, 1, True), Nomination(None, None, 1, False)],
        [Nomination(1, 0.75, 1, True)],
    ]
    result = []
    for fold, cohort in enumerate(paths):
        cases = attach(cohort, [0.0] * len(cohort), list(range(len(cohort))), [p.score == 0.95 for p in cohort])
        for staff, policy in ((1, "fifo"), (1, "risk"), (0, "fifo")):
            result.append(
                {
                    "fold": fold,
                    "placement": 0,
                    "weight": len(cohort),
                    "staff": staff,
                    "policy": policy,
                    "actions": {},
                    "horizon": 60,
                    "scenario": "observed",
                    "model": "jev",
                    "metrics": simulate(cases, reviewers=staff or None, policy=policy),
                }
            )
    return result


def test_case_weighted_recall_pools_counts_before_rates_and_queue_loss_matches_immediate() -> None:
    records = _two_folds()
    summary = {(r["staff"], r["policy"]): r for r in aggregate(records, service=20)}
    fifo, risk, immediate = summary[1, "fifo"], summary[1, "risk"], summary[0, "fifo"]
    # Fold weights 3:1, positive counts 2:1; 7 is the weighted positive
    # denominator. Neither an unweighted nor a size-weighted mean of rates works.
    assert fifo["recall"] == pytest.approx(1 / 7)
    assert risk["recall"] == pytest.approx(4 / 7)
    assert immediate["recall"] == pytest.approx(4 / 7)
    assert fifo["queue_loss"] == pytest.approx(immediate["recall"] - fifo["recall"])
    assert risk["queue_loss"] == 0
    assert risk["delta_vs_fifo"] == pytest.approx(3 / 7)
    assert risk["omit_one_fold_delta_range"] == pytest.approx([0.0, 0.5])
    assert fifo["by_period"]["peak"]["queue_loss"] == 1.0
    assert risk["by_period"]["offpeak"]["recall"] == fifo["by_period"]["offpeak"]["recall"]


def test_mismatched_tickets_and_nonpaired_immediate_reference_fail_closed() -> None:
    records = _two_folds()
    records[1]["metrics"]["nominated"] -= 1  # change only risk arm, same fold and placement
    with pytest.raises(ValueError, match="same cases and tickets"):
        aggregate(records, service=20)
    records = _two_folds()
    records[2]["placement"] = 42  # no-wait control has correct totals but comes from *another* draw
    with pytest.raises(ValueError, match="no-wait control"):
        aggregate(records, service=20)


def _turns(prefix: str, n: int) -> list[Turn]:
    cases = []
    for i in range(n):
        for t in range(1, 4):
            positive = t == 2 and i % 4 == 0
            score = 0.9 if positive else 0.15 if i % 3 == 0 else 0.02
            votes = ("self_harm",) * 3 if positive else ("safe",) * 3
            cases.append(Turn(f"{prefix}-{i}", t, votes, (score, score, score)))
    return cases


def test_fold_run_reuses_identical_case_multiset_across_clocks_pressure_and_dispatch() -> None:
    # Exercise fitting, cutoffs, training streams, queue learning, placement,
    # test dispatch and aggregation, without ever opening a data/cache file.
    arrivals = [t / 10 for t in range(12)] + [70, 71, 130, 131, 190, 250, 310, 370]
    job = Job(
        train=_turns("train", 12),
        test=_turns("test", 4),
        arrivals=arrivals,
        model="jev",
        fold=0,
        seed=9,
        placements=1,
        episodes=1,
        service=20,
        gate="liberal",
        azure=None,
        week=None,
        staff=(1, 0),
    )
    records, meta = fold_run(job)
    assert meta["heldout_dialogues"] == 4
    assert meta["positive_heldout_dialogues"] == 1
    assert meta["fixed_cutoff"] == 0.1
    assert {r["policy"] for r in records} == {"fifo", "risk", "aging", "learned"}
    scenarios = {r["scenario"] for r in records}
    assert {"steady", "observed", "risk_pulse", "double", "compound"} <= scenarios
    # Each scenario contains all four held-out paths exactly five times;
    # even peak enrichment only reorders, it cannot "create" positives.
    assert {r["metrics"]["sessions"] for r in records} == {20}
    assert {r["metrics"]["positives"] for r in records} == {5}
    assert len({r["metrics"]["nominated"] for r in records}) == 1
    immediate = {r["scenario"]: r["metrics"] for r in records if r["staff"] == 0}
    assert len({r["catches"] for r in immediate.values()}) == 1
    assert (
        immediate["observed"]["by_period"]["peak"]["positives"]
        <= immediate["risk_pulse"]["by_period"]["peak"]["positives"]
    )
    for r in aggregate(records, service=20):
        assert r["queue_loss"] == pytest.approx(immediate[r["scenario"]]["recall"] - r["recall"])
