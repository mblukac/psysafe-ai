import json

import pytest

from benchmarks.mindguard import staffing_dependence
from benchmarks.mindguard.data import Turn
from benchmarks.mindguard.staffing import ScoredPath, crossfit
from benchmarks.mindguard.staffing_dependence import MODELS, audit


def fixture_scores() -> dict[str, list[ScoredPath]]:
    out = {}
    for m, values in zip(
        MODELS,
        ((0.01, 0.1, 0.2, 0.01), (0.01, 0.01, 0.2, 0.1), (0.2, 0.2, 0.2, 0.2), (0.01, 0.01, 0.01, 0.01)),
        strict=True,
    ):
        t = tuple((i, s, int(i <= 3)) for i, s in enumerate(values, 1))
        out[m] = [ScoredPath(t, t, t, t)]
    return out


def test_positive_only_overlap_and_inclusive_boundary() -> None:
    report = audit(fixture_scores())
    assert report["source_turns"] == 4
    assert report["positive_turns"] == 3
    pairs = report["native"]["pairs"]
    assert len(pairs) == 6
    overlap = pairs["jev__gemini"]["positive_turn_misses"]
    assert [overlap[k] for k in ("neither", "first_only", "second_only", "both")] == [1, 0, 1, 1]
    assert overlap["jaccard"] == 0.5
    assert overlap["both_fraction_of_positive_turns"] == 1 / 3
    assert report["native"] == report["calibrated"]
    assert report["native"]["positive_turns_all_four_miss"] == 0
    assert report["native"]["positive_turns_any_model_misses"] == 3
    assert report["dialogue_miss_overlap"]["denominator_positive_dialogues"] == 1
    json.dumps(report, allow_nan=False)


def test_undefined_coefficients_and_no_positive_turns() -> None:
    t = ((1, 0.1, 0), (2, 0.1, 0))
    report = audit({m: [ScoredPath(t, t, t, t)] for m in MODELS})
    pair = report["native"]["pairs"]["jev__gemini"]
    assert pair["signed_majority_residual_pearson_all_turns"] is None
    assert pair["positive_turn_misses"]["phi"] is None
    assert pair["positive_turn_misses"]["jaccard"] is None
    assert pair["positive_turn_misses"]["both_fraction_of_positive_turns"] is None


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        audit(fixture_scores(), threshold=threshold)


def test_rejects_missing_model_and_misaligned_calibration() -> None:
    scores = fixture_scores()
    with pytest.raises(ValueError, match="exactly"):
        audit({m: v for m, v in scores.items() if m != "luna"})
    p = scores["luna"][0]
    scores["luna"] = [ScoredPath(((1, 0.1, 0),), p.advantage, p.native, p.matched_margin)]
    with pytest.raises(ValueError, match="unaligned"):
        audit(scores)


def test_audits_crossfit_outputs_without_replicating_source_turns() -> None:
    rows = [
        Turn(f"private-{d}", t, ("self_harm" if d % 2 and t == 2 else "safe",) * 3, (0.1, 0.2, 0.3))
        for d in range(6)
        for t in (1, 2)
    ]
    providers = {m: {(r.dialogue_id, r.turn_index): 0.05 + 0.1 * j for r in rows} for j, m in enumerate(MODELS)}
    scores = crossfit(rows, providers, folds=3, seed=2026, threshold=0.1)
    report = audit(scores)
    assert report["source_turns"] == 12
    assert report["source_dialogues"] == 6
    assert report["positive_turns"] == 3
    assert "private-" not in json.dumps(report, allow_nan=False)


def test_dialogue_overlap_locks_unsupported_first_alert_and_excludes_negative_paths() -> None:
    scores = {}
    for model in MODELS:
        # Jev alerts too early, Gemini catches, Liquid never alerts, Luna catches.
        values = {"jev": (0.2, 0.8), "gemini": (0.01, 0.1), "liquid": (0.01, 0.01), "luna": (0.01, 0.8)}[model]
        positive = tuple((i, s, int(i == 2)) for i, s in enumerate(values, 1))
        negative = ((1, 0.8, 0),)
        scores[model] = [ScoredPath(t, t, t, t) for t in (positive, negative)]
    overlap = audit(scores)["dialogue_miss_overlap"]
    assert overlap["denominator_positive_dialogues"] == 1
    for field in ("native", "calibrated"):
        assert overlap[field]["misses_per_model"] == {"jev": 1, "gemini": 0, "liquid": 1, "luna": 0}
        assert overlap[field]["pairs"]["jev__liquid"]["both"] == 1
        assert overlap[field]["pairs"]["jev__gemini"]["first_only"] == 1
        assert sum(overlap[field]["pairs"]["jev__gemini"].values()) == 1
        assert overlap[field]["positive_dialogues_all_four_miss"] == 0
        assert overlap[field]["positive_dialogues_any_model_misses"] == 1


def test_family_deletion_is_paired_fixed_prediction_sensitivity() -> None:
    scores = fixture_scores()
    for m in MODELS:
        scores[m] = scores[m] * 3
    report = audit(scores, groups=["private-family-a", "private-family-a", "private-family-b"])
    sensitivity = report["calibrated_residual_family_deletion_sensitivity"]
    assert sensitivity["families"] == 2
    for pair, result in sensitivity["pairs"].items():
        expected = report["calibrated"]["pairs"][pair]["signed_majority_residual_pearson_all_turns"]
        assert result["min_max_if_defined"] == pytest.approx([expected, expected])
        assert result["defined_deletions"] == 2
    assert "private-family" not in json.dumps(report)
    undefined = audit(scores, groups=["only-family"] * 3)["calibrated_residual_family_deletion_sensitivity"]
    assert undefined["pairs"]["jev__gemini"]["defined_deletions"] == 0
    with pytest.raises(ValueError, match="per source dialogue"):
        audit(scores, groups=["too-short"])


def test_cli_does_not_echo_private_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_: object) -> None:
        raise ValueError("private input text")

    monkeypatch.setattr(staffing_dependence, "official_cases", fail)
    assert staffing_dependence.main(["--annotations", "a", "--responses", "r", "--luna-responses", "l"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Staffing dependence audit failed: ValueError\n"
