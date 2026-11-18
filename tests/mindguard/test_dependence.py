from __future__ import annotations

from dataclasses import replace
from statistics import correlation

import pytest

from benchmarks.mindguard.data import Turn
from benchmarks.mindguard.dependence import error_dependence
from benchmarks.mindguard.fit import SINGLES
from benchmarks.mindguard.review import Fold


def fold() -> Fold:
    train = [Turn(f"train-{i}", 1, ("safe",) * 3, (0.1,) * 3) for i in range(2)]
    dev = {m: [0.8, 0.4] for m in SINGLES}
    # First four eligible: Jev catches 0/1, Gemini and Liquid catch 1/2.
    scores = [[0.9, 0.8, 0.1, 0, 0.2], [0.1, 0.8, 0.9, 0, 0.2], [0.1, 0.8, 0.9, 0, 0.2]]
    rows = [
        Turn(f"test-{t}", 1, ("self_harm",) * 2 + ("safe",) if t < 4 else ("safe",) * 3, (0.4,) * 3) for t in range(5)
    ]
    return train, dev, rows, dict(zip(SINGLES, scores, strict=True))


def test_matches_eligible_case_2x2_does_not_include_true_negatives():
    f = fold()
    result = error_dependence([f], budget=0.5, slots_per_fold=[2], replicates=60, seed=11)
    for policy in ("fixed", "live"):
        out = result["miss_overlap"][policy]
        assert out["eligible_conversations"]["observed"] == 4
        assert out["all_three_missed"]["observed"] == 1
        pair = out["jev__gemini"]
        assert [
            pair[k]["observed"] for k in ("both_missed", "first_only_missed", "second_only_missed", "neither_missed")
        ] == [1] * 4
        assert pair["phi"]["observed"] == 0
        assert out["gemini__liquid"]["phi"]["observed"] == 1
        assert 0 < pair["phi"]["defined_replicates"] < 60
    assert result == error_dependence([f], budget=0.5, slots_per_fold=[2], replicates=60, seed=11)
    # Native scores identical; correlation of native signed errors is 1,
    # NOT undefined as a correlation of the constant raw probability would be.
    assert result["turn_residual_pearson"]["native"]["jev__gemini"]["observed"] == pytest.approx(1)
    y = [2 / 3, 2 / 3, 2 / 3, 2 / 3, 0.0]
    err = [[p - h for p, h in zip(f[3][m], y, strict=True)] for m in SINGLES]
    assert result["turn_residual_pearson"]["calibrated_oof"]["jev__gemini"]["observed"] == pytest.approx(
        correlation(*err[:2])
    )


def test_zero_variance_and_zero_eligible_yield_null_not_zero():
    train, dev, held, scores = fold()
    all_safe = [replace(row, votes=("safe",) * 3) for row in held]
    result = error_dependence([(train, dev, all_safe, scores)], budget=0.5, slots_per_fold=[2], replicates=4, seed=4)
    unavailable = {"observed": None, "percentile_95_if_defined": None, "defined_replicates": 0}
    assert result["turn_residual_pearson"]["native"]["jev__liquid"] == unavailable
    fixed = result["miss_overlap"]["fixed"]
    assert fixed["eligible_conversations"]["observed"] == 0
    assert fixed["jev__liquid"]["phi"] == unavailable
    negatives = result["false_alert_overlap"]["fixed"]
    assert negatives["never_majority_positive"]["observed"] == 5
    assert negatives["all_three_alerted"]["observed"] == 1
    pair = negatives["jev__gemini"]
    assert [
        pair[k]["observed"] for k in ("both_alerted", "first_only_alerted", "second_only_alerted", "neither_alerted")
    ] == [1, 1, 1, 2]
    assert pair["phi"]["observed"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="overlapping"):
        error_dependence([fold(), fold()], budget=0.5, slots_per_fold=[2, 2], replicates=2, seed=1)


def test_live_premature_alert_contributes_miss_without_changing_retrospective_peak():
    train, dev, held, scores = fold()
    earlier = Turn(held[0].dialogue_id, 0, ("safe",) * 3, (0.6, 0, 0))
    with_earlier = {m: [*scores[m], earlier.scores[i]] for i, m in enumerate(SINGLES)}
    result = error_dependence(
        [(train, dev, [*held, earlier], with_earlier)], budget=0.5, slots_per_fold=[2], replicates=3, seed=1
    )
    assert result["miss_overlap"]["fixed"]["all_three_missed"]["observed"] == 1
    assert result["miss_overlap"]["live"]["all_three_missed"]["observed"] == 2
