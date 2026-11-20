import pytest

from benchmarks.mindguard.budget_fit import _split_scores
from benchmarks.mindguard.data import Turn
from benchmarks.mindguard.fit import grouped_folds
from benchmarks.mindguard.surge_mixed import nominations, stream_records
from benchmarks.mindguard.surge_replay import Nomination, Setting, aggregate, nominate
from benchmarks.mindguard.surge_stream import make_clock


def test_mixed_source_paths_each_use_their_own_frozen_outer_gate() -> None:
    rows = [
        Turn(str(d), t, (("self_harm" if d % 3 == 0 and t == 2 else "safe"),), (s, s, s))
        for d in range(10)
        for t, s in ((1, 0.12), (2, 0.92 if d % 3 == 0 else 0.02))
    ]
    name, frozen, cuts = nominations("jev", rows=rows, arrivals=[i * 3.0 for i in range(25)], seed=2026)
    expected = {}
    for i, (train, test) in enumerate(grouped_folds(rows, 5, 2026)):
        for d, (calibrated, _) in _split_scores(train, test, 0).items():
            expected[d] = nominate(calibrated, cuts[i])
    assert name == "jev"
    assert frozen == [p for _, p in sorted(expected.items())]
    assert len(frozen) == 10


def test_mixed_competition_pairs_all_arms_not_merely_separate_fold_counts() -> None:
    # Simultaneous unambiguous tickets: risk can put a supported case ahead
    # of an unsupported case coming from a different source path.
    sources = [Nomination(1, 0.20, None, False), Nomination(1, 0.9, 1, True)]
    clock = make_clock([0.0, 0.0, 30.0, 30.0])
    setting = Setting("simultaneous", clock, 60.0, False)
    records = [r for rep in range(10) for r in stream_records("jev", sources, setting, seed=313, rep=rep)]
    assert {(r["fold"], r["weight"]) for r in records} == {(0, 1)}
    assert {r["policy"] for r in records} == {"fifo", "risk", "aging"}
    assert {r["metrics"]["positives"] for r in records} == {2}
    assert {r["metrics"]["nominated"] for r in records} == {4}
    summary = {(r["staff"], r["policy"]): r for r in aggregate(records, service=20.0)}
    assert summary[1, "risk"]["recall"] > summary[1, "fifo"]["recall"]
    assert summary[1, "risk"]["starts"] == summary[1, "fifo"]["starts"] == 2
    assert summary[0, "fifo"]["recall"] == pytest.approx(1.0)
    assert summary[1, "risk"]["omit_one_fold_delta_range"] is None
