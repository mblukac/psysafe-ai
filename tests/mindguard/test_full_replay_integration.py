"""Primary command end-to-end using invented source paths and a local toy clock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mindguard import full_replay
from benchmarks.mindguard.full_discrimination import audit
from benchmarks.mindguard.full_panel import PanelTurn


def test_primary_cli_consistent_from_outer_folds_through_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    # Twelve opening families; 3 have two paths sharing their safe opening.
    # Missing t=2 is never encoded, evaluated as negative, or erased from time.
    for family in range(12):
        for branch in range(1 + int(family in (0, 4, 8))):
            for t in (1, 3, 4):
                target = ("self_harm" if family % 4 == 0 else "harm_others") if family % 2 == 0 and t >= 3 else "safe"
                p = 0.85 if target != "safe" else 0.02
                rows.append(
                    PanelTurn(
                        f"invented-case-{family}-{branch}",
                        f"invented-opening-{family}",
                        t,
                        (target, target, "safe"),
                        {"jev": {"safe": 1 - p, "self_harm": p * 0.6, "harm_others": p * 0.4}},
                    )
                )

    # All fit/calibration, threshold tuning, dispatch learner, queue, oracle and
    # aggregate code are real. Only the two private, checksummed I/O boundaries
    # are replaced; this test reads no credentials, release data or network.
    monkeypatch.setattr(full_replay, "load_panel", lambda *_: rows)
    monkeypatch.setattr(full_replay, "load_arrivals", lambda *_: [float(3 * (i // 5)) for i in range(200)])
    output = tmp_path / "report.json"
    args = [
        "--annotations",
        "fixture",
        "--caches",
        "fixture",
        "--trace",
        "fixture",
        "--models",
        "jev",
        "--placements",
        "1",
        "--episodes",
        "1",
        "--staff",
        "1",
        "0",
        "--out",
        str(output),
    ]
    assert full_replay.main(args) == 0
    data = json.loads(output.read_text())
    assert (data["rated_turns"], data["dialogues"], data["families"]) == (45, 15, 12)
    assert data["counts"]["either"] == {"turns": 18, "dialogues": 9}
    assert data["coverage"]["unrated_positions"] == 15
    assert len(data["training"]) == 5
    assert sum(x["heldout_paths"] for x in data["training"]) == 15
    assert sum(x["positive_heldout_paths"] for x in data["training"]) == 9
    assert "invented-" not in json.dumps(data)
    diagnostic = audit(rows, data)
    assert diagnostic["dependence"]["positive_path_misses"]["units"] == 9
    assert diagnostic["probability"]["jev"]["calibrated"]["brier"] == pytest.approx(
        data["probability"]["jev"]["calibrated"]
    )
    scenarios = {s["scenario"] for s in data["results"]}
    assert scenarios == {"steady", "steady_pulse", "observed", "risk_pulse", "double", "compound"}
    for scenario in scenarios:
        records = [s for s in data["results"] if s["scenario"] == scenario]
        (unlimited,) = [s for s in records if s["staff"] == 0]
        finite = [s for s in records if s["staff"] == 1]
        assert unlimited["recall"] == 1
        assert unlimited["misses_queue_delay"] == 0
        assert len({s["matched_minutes"] for s in finite}) == 1
        assert {s["policy"] for s in finite} == {"fifo", "risk", "aging", "loss_wait", "learned", "best_simple"}
        for s in finite:
            assert s["positives"] == pytest.approx(s["self_harm_positives"] + s["harm_others_positives"])
            assert s["recall"] <= unlimited["recall"]
            assert s["misses_queue_delay"] == pytest.approx(unlimited["catches"] - s["catches"])
            assert s["matched_minutes"] <= s["mean_review_minutes"] + 1e-8
            assert s["recall"] <= s["oracle_recall_bounds"][0] + 1e-9
            assert s["oracle_recall_bounds"][0] <= s["oracle_recall_bounds"][1] + 1e-9
            assert s["all_gaps_endorsed_recall"] <= s["recall"] + 1e-9
    assert full_replay.main(args) == 2  # no overwrite of original analysis
