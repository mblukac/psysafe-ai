from __future__ import annotations

import json
from collections import Counter

import pytest

from benchmarks.mindguard.full_prevalence import paired_mixes, run
from benchmarks.mindguard.surge_replay import Nomination


def test_prevalence_nested_exact_and_balanced_on_whole_paths() -> None:
    ref = [True] * 17 + [False] * 50
    n = 9175
    mixes = paired_mixes(ref, n, 97)
    assert mixes == paired_mixes(ref, n, 97)
    assert mixes != paired_mixes(ref, n, 98)
    assert {k: sum(ref[i] for i in indices) for k, indices in mixes.items()} == {
        "one_percent": 92,
        "five_percent": 459,
        "source_fraction": 2328,
    }
    for indices in mixes.values():
        assert len(indices) == n
        for cls in (False, True):
            counts = Counter(i for i in indices if ref[i] == cls)
            assert len(counts) == ref.count(cls)
            assert max(counts.values()) - min(counts.values()) <= 1
    for a, b in (("one_percent", "five_percent"), ("five_percent", "source_fraction")):
        for x, y in zip(mixes[a], mixes[b], strict=True):
            assert not (ref[x] and not ref[y])
            if ref[x] == ref[y]:
                assert x == y


def test_fixed_gate_false_tickets_separate_path_vs_supported_turn() -> None:
    # First path: premature wrong-turn alert on eventual positive. Second:
    # supported onset nomination. One of two negative sources nominates.
    cases = [
        Nomination(1, 0.9, 2, False),
        Nomination(2, 0.9, 2, True),
        Nomination(1, 0.6, None, False),
        Nomination(None, None, None, False),
    ]
    report = run({"a": cases, "b": cases}, [float(i // 4) for i in range(400)], 2, 2026)
    assert "None" not in json.dumps(report, allow_nan=False)
    high = [r for r in report["results"] if r["mix"] == "source_fraction" and r["model"] == "a"]
    immediate = next(r for r in high if r["staff"] == 0)
    assert immediate["positives"] == 200
    assert immediate["negative_path_tickets"] == 100
    assert immediate["unsupported_tickets"] == 200
    assert immediate["timely_recall"] == 0.5
    assert immediate["contested_fraction_of_starts"] == 0
    for row in high:
        assert row["nominated"] == immediate["nominated"]
        assert row["review_minutes"] == 20 * row["starts"]
        assert row["timely_recall"] <= immediate["timely_recall"]
        assert row["misses_queue_delay"] == immediate["catches"] - row["catches"]
        twin = next(
            r for r in report["results"] if r["model"] == "b" and r["mix"] == row["mix"] and r["staff"] == row["staff"]
        )
        assert (row | {"model": "b"}) == twin
    assert next(r for r in high if r["staff"] == 1)["contested_starts"] > 0


def test_reject_absent_strata_or_misaligned_model_references() -> None:
    with pytest.raises(ValueError, match="both observed classes"):
        paired_mixes([True] * 3, 20, 3)
    cases = [Nomination(1, 0.9, 2, False), Nomination(None, None, None, False)]
    with pytest.raises(ValueError, match="misaligned"):
        run({"a": cases, "b": cases[::-1]}, [0.0, 1.0], 1, 4)
