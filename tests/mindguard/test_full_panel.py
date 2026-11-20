import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.collect import _PROVIDERS
from benchmarks.mindguard.data import Case, digest, requests_for
from benchmarks.mindguard.full_panel import PanelTurn, _join, coverage_audit
from benchmarks.mindguard.providers import FLASH_ROUTES, LUNA_MODEL, MODEL_IDS, synthetic_response


def _cases() -> list[Case]:
    opening = (("user", "Synthetic opening"),)
    return [
        Case("a", 1, ("safe",) * 3, opening),
        Case(
            "a",
            3,
            ("self_harm", "harm_others", "harm_others"),
            (*opening, ("assistant", "A"), ("user", "Missing rating"), ("assistant", "B"), ("user", "Current")),
        ),
    ]


def _envelopes(cases: list[Case], model: str) -> list[dict[str, Any]]:
    rows = []
    for c in cases:
        (req,) = requests_for(c, models=(model,))
        q = {"safe": 0.75, "self_harm": 0.15, "harm_others": 0.1}
        if c.turn_index == 3:
            q = {"safe": 0.0, "self_harm": 0.0, "harm_others": 1.0}
        rows.append(
            {
                "request_id": req["request_id"],
                "model": model,
                "response": {
                    **synthetic_response(model, q),
                    "provider": _PROVIDERS[model],
                    "id": "gen-" + req["request_id"],
                    "usage": {"cost": 0.00001, "input_tokens": 123},
                },
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_joins_full_vectors_without_creating_negative_in_missing_slot(tmp_path: Path) -> None:
    cases = _cases()
    rows = _envelopes(cases, MODEL_IDS[0])
    luna = _envelopes(cases, LUNA_MODEL)
    panel = _join(
        cases,
        [
            _write(tmp_path / "z.jsonl", [luna[1], rows[0]]),
            _write(tmp_path / "a.jsonl", [luna[0], rows[1]]),
        ],
        expected_models=("luna", "jev"),
    )
    assert [(r.dialogue_id, r.turn_index) for r in panel] == [("a", 1), ("a", 3)]
    assert panel[0].group_id == panel[1].group_id == digest((("user", "Synthetic opening"),))
    assert panel[1].scores["luna"] == panel[1].scores["jev"] == {"safe": 0, "self_harm": 0, "harm_others": 1}
    assert sum(panel[0].scores["jev"][k] for k in ("self_harm", "harm_others")) == 0.25
    assert panel[1].votes == cases[1].votes


@pytest.mark.parametrize("fault", ["missing_turn", "missing_model", "bad_key", "wrong_route", "bad_vec", "cross_dup"])
def test_inexact_or_invalid_native_receipts_do_not_shrink_or_impute(fault: str, tmp_path: Path) -> None:
    cases = _cases()
    jev = _envelopes(cases, MODEL_IDS[0])
    luna = _envelopes(cases, LUNA_MODEL)
    other = []
    if fault == "missing_turn":
        luna.pop()
    elif fault == "missing_model":
        luna.clear()
    elif fault == "bad_key":
        jev[0]["request_id"] = "wrong"
    elif fault == "wrong_route":
        luna[0]["response"]["provider"] = "Wrong"
    elif fault == "bad_vec":
        del jev[0]["response"]["answers"]["mindguard"]["probabilities"]["harm_others"]
    elif fault == "cross_dup":
        other.append(_write(tmp_path / "shard.jsonl", [jev[0]]))
    cache = _write(tmp_path / "cache.jsonl", jev + luna)
    with pytest.raises(ValueError, match=r"Incomplete|do not match|mismatched|provider|probabilities|Duplicate"):
        _join(cases, [cache, *other], expected_models=("jev", "luna"))


def test_missing_empty_and_generation_reuse_across_files_fail(tmp_path: Path) -> None:
    cases = _cases()
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="does not exist"):
        _join(cases, [missing])
    with pytest.raises(ValueError, match="Empty"):
        _join(cases, [_write(missing, [])])
    jev = _envelopes(cases, MODEL_IDS[0])
    luna = _envelopes(cases, LUNA_MODEL)
    luna[1]["response"]["id"] = jev[1]["response"]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        _join(cases, [_write(missing, jev), _write(tmp_path / "dup.jsonl", luna)])


def test_observed_turn_audit_bounds_gaps_per_endpoint_without_exposing_ids() -> None:
    safe = ("safe",) * 3
    rows = [
        PanelTurn("first private ID", "family", 1, safe, {}),
        PanelTurn("first private ID", "family", 3, ("self_harm", "self_harm", "safe"), {}),
        PanelTurn("another private ID", "family", 2, safe, {}),
        PanelTurn("complete private ID", "other opening", 1, ("harm_others", "harm_others", "safe"), {}),
    ]
    audit = coverage_audit(rows)
    assert "private ID" not in json.dumps(audit)
    assert audit["unrated_positions"] == 2
    assert audit["paths_with_unrated_positions"] == 2
    assert audit["by_endpoint"]["either"] == {
        "observed_positive_paths": 2,
        "observed_positive_turns": 2,
        "positive_opening_families": 2,
        "possible_positive_paths_if_any_unrated_turn_qualifies": 3,
        "confirmed_paths_with_ambiguous_onset": 1,
        "max_earlier_onset_shift_turns": 1,
    }
    assert audit["by_endpoint"]["harm_others"]["max_earlier_onset_shift_turns"] == 0


def test_flash_rows_require_same_exact_join_including_route_and_cost(tmp_path: Path) -> None:
    cases = _cases()
    rows = [r for m in FLASH_ROUTES for r in _envelopes(cases, m)]
    path = tmp_path / "flash.jsonl"
    panel = _join(cases, [_write(path, rows)], expected_models=("decider_flash", "clef_flash"))
    assert panel[-1].scores["decider_flash"]["harm_others"] == panel[-1].scores["clef_flash"]["harm_others"] == 1.0
    rows[1]["response"]["usage"] = {}
    with pytest.raises(ValueError, match="charge"):
        _join(cases, [_write(path, rows)])
