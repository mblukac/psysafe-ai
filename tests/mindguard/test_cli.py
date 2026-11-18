import json
from pathlib import Path

import pytest

from benchmarks.mindguard.__main__ import evaluate, main
from benchmarks.mindguard.data import read_cases
from benchmarks.mindguard.smoke import smoke_turns


def test_smoke_is_offline_and_reports_distinct_workloads(capsys, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("offline pilot attempted network access")

    monkeypatch.setattr("socket.socket.connect", no_network)
    assert main(["smoke", "--slots", "4", "--bootstrap", "5"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "synthetic_smoke_not_benchmark_results"
    assert (report["dialogues"], report["turns"]) == (18, 54)
    assert report["fixed_slots_per_method"] == 4
    assert len(report["methods"]) == 7
    for method in report["methods"].values():
        assert method["fixed"]["reviews"] == report["fixed_slots_per_method"]
        for policy in ("fixed", "live"):
            counts = method[policy]
            assert counts["catches"] + counts["misses"] == report["eligible_cases"]
            assert counts["catches"] + counts["unsupported"] == counts["reviews"]
    assert report["stack_minus_best_single"]["fixed_review_delta"]["observed"] == 0
    overlap = report["error_dependence"]["miss_overlap"]
    for policy in ("fixed", "live"):
        assert overlap[policy]["eligible_conversations"]["observed"] == report["eligible_cases"]
        both = overlap[policy]["jev__gemini"]["both_missed"]["observed"]
        assert both <= min(report["methods"][m][policy]["misses"] for m in ("jev", "gemini"))
    assert "Invented fixture" not in str(report)


def test_prepare_never_overwrites_private_requests(tmp_path, monkeypatch):
    target = tmp_path / "requests.jsonl"
    cases = read_cases(
        [
            {
                "prompt": [],
                "user_message": "Invented.",
                "annotator_labels": {
                    "annotator_1": "safe",
                    "annotator_2": "safe",
                    "annotator_3": "safe",
                },
            }
        ]
    )
    monkeypatch.setattr("benchmarks.mindguard.__main__.official_cases", lambda path: cases)
    cmd = ["prepare", "--annotations", "unused", "--out", str(target)]
    assert main(cmd) == 0
    assert Path(target).stat().st_mode & 0o077 == 0
    assert len(target.read_text().splitlines()) == 3
    assert "annotator" not in target.read_text()
    with pytest.raises(SystemExit) as exc:
        main(cmd)
    assert exc.value.code == 2


def test_complete_only_replay_is_a_restriction_of_same_heldout_predictions():
    turns = smoke_turns()
    all_ids = {row.dialogue_id for row in turns}
    args = {"folds": 3, "seed": 2026, "budget": 0.10, "replicates": 3, "synthetic": True}
    all_report = evaluate(turns, complete_ids=all_ids, **args)
    sub_report = evaluate(turns, complete_ids={min(all_ids)}, **args)
    assert sub_report["methods"] == all_report["methods"]
    assert sub_report["complete_dialogue_live_sensitivity"]["dialogues"] == 1
    for m, results in all_report["methods"].items():
        assert all_report["complete_dialogue_live_sensitivity"]["methods"][m] == results["live"]


def test_prepare_fails_closed_for_incomplete_local_release(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [],
                "user_message": "Synthetic.",
                "annotator_labels": {
                    "annotator_1": "safe",
                    "annotator_2": "safe",
                    "annotator_3": "safe",
                },
            }
        )
        + "\n"
    )
    out = tmp_path / "requests.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(["prepare", "--annotations", str(source), "--out", str(out)])
    assert exc.value.code == 2
    assert not out.exists()
