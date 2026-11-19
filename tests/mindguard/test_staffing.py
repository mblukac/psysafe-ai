import hashlib
from pathlib import Path

import pytest

from benchmarks.mindguard import staffing
from benchmarks.mindguard.data import Turn


def test_pinned_trace_keeps_zero_gap_and_rejects_other_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "trace.csv"
    raw = b"1\n0\n" + b"2\n" * 9173
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="pinned"):
        staffing.load_arrivals(path)
    monkeypatch.setattr(staffing, "TRACE_SHA256", hashlib.sha256(raw).hexdigest())
    times = staffing.load_arrivals(path)
    assert times[:3] == [1, 1, 3]
    assert times[-1] == 18347


def test_balanced_whole_path_placements_are_repeatable() -> None:
    a = staffing.placement(5, 23, 4)
    assert set(a) == set(range(5))
    assert sorted(a.count(i) for i in range(5)) == [4, 4, 5, 5, 5]
    assert a == staffing.placement(5, 23, 4)
    assert a != staffing.placement(5, 23, 5)


def test_end_to_end_aggregates_do_not_contain_private_ids() -> None:
    rows = [
        Turn(f"private-{d}", t, ("self_harm" if t == 2 and d % 2 else "safe",) * 3, (p, p, p), str(d // 2))
        for d in range(10)
        for t, p in [(1, 0.03), (2, 0.7 if d % 2 else 0.05)]
    ]
    maps = {"toy": {(r.dialogue_id, r.turn_index): r.scores[0] for r in rows}}
    scores = staffing.crossfit(rows, maps, folds=3, seed=1, threshold=0.1)
    diag = staffing.forecast_diagnostics(scores["toy"])["native"]
    assert diag["positive_turns"] == 5
    assert diag["turns"] == 20
    results = staffing.replay(
        scores,
        list(range(40)),
        staffing=[0, 2],
        placements=2,
        seed=4,
        service=4,
        turn_spacing=2,
        deadline=2,
        threshold=0.1,
    )
    assert len(results) == 10
    assert "private-" not in str(results)
    assert all(r["mean_per_trace"]["positives"] == 20 for r in results)
    assert all(r["recall"] == 0 for r in results if r["reviewers"] == 0)
    for row in results:
        c = row["mean_per_trace"]
        assert c["catches"] + c["misses_late"] + c["misses_wrong_turn"] + c["misses_no_alert"] == c["positives"]


def test_refuses_incomparable_source_pools() -> None:
    t = ((1, 0.5, 0),)
    p = staffing.ScoredPath(t, t, t, t)
    with pytest.raises(ValueError, match="same source"):
        staffing.replay(
            {"one": [p], "two": []},
            [0],
            staffing=[1],
            placements=1,
            seed=1,
            service=1,
            turn_spacing=1,
            deadline=1,
            threshold=0.1,
        )
