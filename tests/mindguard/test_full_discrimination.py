from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.mindguard import full_replay
from benchmarks.mindguard.fit import grouped_folds
from benchmarks.mindguard.full_discrimination import (
    SourceOutcome,
    audit,
    average_precision,
    dependence,
    outcome,
    paired_errors,
)
from benchmarks.mindguard.full_panel import PanelTurn
from benchmarks.mindguard.surge_replay import Nomination


def test_ap_groups_ties_no_interpolation() -> None:
    pairs = [(0.5, 0), (0.8, 1), (0.5, 1), (0.3, 0)]
    assert average_precision(pairs) == pytest.approx(1 / 2 + 1 / 3)
    assert average_precision(list(reversed(pairs))) == average_precision(pairs)
    assert average_precision([(0.1, 0), (0.1, 0), (0.1, 1)]) == pytest.approx(1 / 3)
    assert average_precision([(0.6, 0)]) is None
    with pytest.raises(ValueError, match="invalid score"):
        average_precision([(float("nan"), 1)])


def test_first_supported_locked_ticket_in_time() -> None:
    assert outcome("family", Nomination(7, 0.8, 2, True)).timely
    assert not outcome("family", Nomination(8, 0.8, 2, True)).timely
    assert not outcome("family", Nomination(1, 0.8, 2, False)).timely
    assert outcome("family", Nomination(1, 0.8, 2, False)).positive
    assert outcome("family", Nomination(2, 0.8, None, False)).ticket


def test_pairwise_counts_score_true_misses_and_false_alarms_separately() -> None:
    counts = paired_errors([False, True, True, False], [False, False, True, False])
    assert (counts["both"], counts["first_only"], counts["neither"]) == (1, 1, 2)
    assert counts["expected_both_under_independent_errors"] == 0.5
    assert counts["phi"] == pytest.approx(1 / 3**0.5)
    assert paired_errors([True], [True])["phi"] is None
    with pytest.raises(ValueError, match="shorter"):
        paired_errors([True], [])
    first = {
        "d1": SourceOutcome("a", True, False, False),
        "d2": SourceOutcome("a", True, True, True),
        "d3": SourceOutcome("b", True, False, False),
        "d4": SourceOutcome("c", False, True, False),
        "d5": SourceOutcome("c", False, False, False),
        "d6": SourceOutcome("b", False, True, False),
    }
    second = first | {"d2": SourceOutcome("a", True, False, False), "d5": SourceOutcome("c", False, True, False)}
    dep = dependence({"x": first, "y": second})
    assert dep["positive_path_misses"]["units"] == 3
    assert dep["positive_path_misses"]["all_models_error"] == 2
    assert dep["positive_family_no_positive_path_caught"]["errors_per_model"] == {"x": 1, "y": 2}
    assert dep["negative_path_false_alerts"]["units"] == 3
    assert dep["negative_path_false_alerts"]["all_models_error"] == 2
    assert dep["all_negative_family_any_false_alert"]["units"] == 1
    with pytest.raises(ValueError, match="mismatched source"):
        dependence({"x": first, "y": second | {"d1": SourceOutcome("z", True, False, False)}})


@pytest.fixture
def fixture_replay(monkeypatch: pytest.MonkeyPatch) -> tuple[list[PanelTurn], dict[str, Any]]:
    rows = []
    for i in range(12):
        for t in (1, 3):
            positive = i % 3 == 0 and t == 3
            votes = ("safe", "harm_others", "self_harm") if positive else ("safe",) * 3
            p = 0.9 if positive else 0.03
            scores = {m: {"self_harm": p / 2, "harm_others": p / 2, "safe": 1 - p} for m in ("jev", "liquid")}
            rows.append(PanelTurn(f"secret-path-{i}", f"secret-family-{i//2}", t, votes, scores))
    # Cheap primary-gate fixture: the production cutoff fitter is exercised in
    # the replay tests; here the diagnostic must read exactly the audit's gate.
    monkeypatch.setattr(full_replay, "tuned_cutoff", lambda *_: 0.15)
    fits = []
    for m in ("jev", "liquid"):
        for fold, (train, hold) in enumerate(grouped_folds(full_replay.encoded(rows, m), 5, 18)):
            ids = {r.dialogue_id for r in hold}
            job = full_replay.Job(
                m,
                fold,
                train,
                hold,
                [r for r in rows if r.dialogue_id in ids],
                [0.0],
                None,
                None,
                (1,),
                20,
                18,
                1,
                1,
                "tuned",
            )
            fits.append(full_replay.prepare_hold(job)[3])
    return rows, dict(
        kind="mindguard_full_either_risk_v1",
        gate="tuned",
        deadline_minutes=10,
        turn_minutes=2,
        models=["jev", "liquid"],
        rated_turns=24,
        families=6,
        dialogues=12,
        seed=18,
        training=fits,
    )


def test_aggregate_only_oof_matches_actual_replay_partition(
    fixture_replay: tuple[list[PanelTurn], dict[str, Any]],
) -> None:
    rows, replay = fixture_replay
    result = audit(rows, replay)
    assert "secret-" not in json.dumps(result, allow_nan=False)
    assert result["positive_turns"] == result["dependence"]["positive_path_misses"]["units"] == 4
    assert sum(f["held_families"] for f in result["folds"]) == 6
    assert sum(f["held_turns"] for f in result["folds"]) == len(rows)
    assert result["probability"]["jev"]["native"]["average_precision"] == 1
    assert result["dependence"]["positive_path_misses"]["errors_per_model"] == {"jev": 0, "liquid": 0}
    replay["training"][0]["brier_native_sum"] += 0.01
    with pytest.raises(ValueError, match="scores do not match"):
        audit(rows, replay)


def test_wrong_or_incomplete_audit_rejected(fixture_replay: tuple[list[PanelTurn], dict[str, Any]]) -> None:
    rows, replay = fixture_replay
    with pytest.raises(ValueError, match="primary tuned"):
        audit(rows, replay | {"gate": "liberal"})
    with pytest.raises(ValueError, match="match supplied panel"):
        audit(rows, replay | {"models": ["jev"]})
    with pytest.raises(ValueError, match="duplicate frozen gate"):
        audit(rows, replay | {"training": replay["training"] * 2})
