from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.loss_audit import PARTITION, audit, decompose
from benchmarks.mindguard.update_queue import ARMS


def example() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    arms = {a: dict(catches=2, rescued=0, spoiled=0) for a in ARMS}
    arms["visible_oracle"].update(catches=4, rescued=2)
    arms["latest"].update(catches=3, rescued=2, spoiled=1)
    point = dict(
        gate="tuned",
        model="test",
        scenario="example",
        staff=1,
        sessions=40,
        positives=20,
        nominated=30,
        starts=22,
        expired=8,
        review_minutes=440,
        before_onset_starts=1,
        after_deadline_starts=3,
        timely_positive_starts=6,
        arms=arms,
    )
    fifo = dict(misses_no_alert=4, misses_wrong_turn=7, misses_nomination_late=3, misses_queue_delay=4)
    immediate = {**point, "staff": 0, "starts": 30, "after_deadline_starts": 5}
    return point, fifo, immediate


def test_positive_partition_counts_and_late_gate_detail() -> None:
    r = decompose(*example())
    assert [r[k] for k in PARTITION] == [4, 6, 1, 3, 2, 2, 2]
    assert r["late_first_flag"] == 5
    assert r["unserved_or_late_despite_flag_by_deadline"] == 4
    assert r["arms"]["latest"] == {"caught": 3, "gained": 2, "lost": 1, "net": 1}


def test_rejects_different_control_and_impossible_subsets() -> None:
    point, fifo, control = example()
    with pytest.raises(ValueError, match="immediate-service control"):
        decompose(point, fifo, {**control, "model": "other"})
    with pytest.raises(ValueError, match="nonnegative"):
        decompose(point, fifo, {**control, "after_deadline_starts": 10})
    with pytest.raises(ValueError, match="all-case totals"):
        decompose({**point, "expired": 5}, fifo, control)
    point["arms"]["latest"].update(catches=3, rescued=3, spoiled=2)
    with pytest.raises(ValueError, match="evidence bound"):
        decompose(point, fifo, control)


def test_full_cache_verifies_all_workloads_and_immediate_control() -> None:
    report = audit(Path(__file__).resolve().parents[2] / "benchmarks" / "mindguard" / "figures")
    rows = report["results"]
    assert len(rows) == 6 * 2 * 9 * 4
    for r in rows:
        assert r["positives"] == pytest.approx(sum(r[k] for k in PARTITION))
        assert r["late_first_flag"] + r["unserved_or_late_despite_flag_by_deadline"] == pytest.approx(
            r["late_pickup"] + r["flagged_never_served"], abs=1e-8
        )
        if r["staff"] == 0:
            assert r["unserved_or_late_despite_flag_by_deadline"] == pytest.approx(0, abs=1e-8)
            assert r["timely_stale"] == pytest.approx(0, abs=1e-8)
            assert r["flagged_never_served"] == pytest.approx(0, abs=1e-8)
