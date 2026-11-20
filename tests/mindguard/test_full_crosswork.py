import argparse
from dataclasses import replace

import pytest

from benchmarks.mindguard.full_crosswork import build_jobs, cross_job, prefix_counts, summarize_work
from benchmarks.mindguard.full_panel import PanelTurn
from benchmarks.mindguard.surge_queue import Case


def test_same_ordinal_can_be_supported_for_one_detector_only_and_no_outcome_picking() -> None:
    yes = Case(0, 5, 0.9, 2, 0, True, False)
    wrong_turn = Case(0, 0, 0.9, 1, 5, False, False)
    same_onset = replace(yes, onset=5)
    # x serves two tickets, y only one. Outcomes must look up the correct
    # detector's nominated prefix; only the first start can count in x.
    assert prefix_counts({"x": ([same_onset, yes], {0: 5, 1: 7}), "y": ([wrong_turn, yes], {0: 0})}) == (
        1,
        {"x": 1, "y": 0},
    )
    # Late starts and a zero common budget cannot contribute timely catches.
    assert prefix_counts({"x": ([yes], {0: 11}), "y": ([yes], {0: 10})}) == (1, {"x": 0, "y": 1})
    assert prefix_counts({"x": ([yes], {}), "y": ([yes], {0: 10})}) == (0, {"x": 0, "y": 0})


def test_independent_gates_identical_case_order_and_ratio_of_pooled_counts() -> None:
    rows = []
    for i in range(14):
        for t in (1, 3):
            y = i % 3 == 0 and t == 3
            scores = {
                m: {"self_harm": q, "harm_others": 0.0, "safe": 1 - q}
                for m, q in (("jev", 0.99 if y else 0.01), ("liquid", 0.8 if i % 5 < 2 else 0.01))
            }
            rows.append(PanelTurn(f"p{i}", f"g{i}", t, ("self_harm" if y else "safe",) * 3, scores))
    args = argparse.Namespace(
        seed=99, azure=None, burst_week=None, placements=1, staff=(0, 1), service=20, gate="liberal"
    )
    arms = ("jev", "liquid")
    jobs = build_jobs(rows, arms, [0.0, 0.1, 0.4, 1.0, 3.0, 5.0, 10.0, 30.0, 60.0, 80.0, 90.0, 100.0], args)
    runs = [cross_job(j) for j in jobs]
    counts = [r for rec, _ in runs for r in rec]
    summary = summarize_work(counts, arms, 20)
    r = next(r for r in summary if r["scenario"] == "observed" and r["staff"] == 1)
    group = [j for j in counts if j["scenario"] == "observed" and j["staff"] == 1]
    assert r["models"]["jev"]["matched_recall"] == pytest.approx(
        sum(j["weight"] * j["matched_catches"]["jev"] for j in group) / sum(j["weight"] * j["positives"] for j in group)
    )
    for m in arms:
        assert r["matched_minutes"] <= r["models"][m]["uncapped_minutes"]
        assert r["models"][m]["matched_recall"] <= r["models"][m]["uncapped_recall"]
    with pytest.raises(ValueError, match="unpaired fold"):
        cross_job([jobs[0][0], jobs[1][1]])
