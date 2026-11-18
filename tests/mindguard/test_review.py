import math

import pytest

from benchmarks.mindguard.data import Turn
from benchmarks.mindguard.review import allocate_slots, evaluate_fold, paired_uncertainty


def turn(dialogue: str, index: int, endorsements: int) -> Turn:
    votes = ("self_harm",) * endorsements + ("safe",) * (3 - endorsements)
    return Turn(dialogue, index, votes, (0.1, 0.2, 0.3))


def test_future_evidence_does_not_redeem_early_alert() -> None:
    train = [turn("train1", 1, 1), turn("train2", 1, 0)]
    # Unordered input; second turn has strongest score. Live must stop on the first.
    test = [turn("test", 2, 2), turn("safe", 1, 0), turn("test", 1, 1)]
    result = evaluate_fold(train, {"stack": [0.9, 0.5]}, test, {"stack": [0.9, 0.01, 0.6]}, budget=0.5)
    assert result["queue_slots"] == result["eligible_cases"] == 1
    fixed = result["methods"]["stack"]["fixed"]
    live = result["methods"]["stack"]["live"]
    assert fixed == {
        "reviews": 1,
        "catches": 1,
        "misses": 0,
        "unsupported": 0,
        "premature": 0,
        "endorsed_vote_mass": 2 / 3,
    }
    assert live == {
        "reviews": 1,
        "catches": 0,
        "misses": 1,
        "unsupported": 1,
        "premature": 1,
        "endorsed_vote_mass": 1 / 3,
    }


def test_train_ties_respect_cap_and_test_burden_is_not_forced() -> None:
    train = [turn(f"tr{i}", 1, 0) for i in range(3)]
    test = [turn(f"te{i}", 1, 2) for i in range(3)]
    report = evaluate_fold(train, {"m": [0.8] * 3}, test, {"m": [0.9] * 3}, budget=1 / 3)["methods"]["m"]
    assert report["threshold"] > 0.8
    assert report["train_live_reviews"] == 0
    assert report["fixed"]["reviews"] == 1
    assert report["live"]["reviews"] == 3


def test_queue_ties_are_label_blind_and_chronological() -> None:
    train = [turn("tr", 1, 0)]
    test = [turn("a", 2, 3), turn("a", 1, 0), turn("b", 1, 0)]
    report = evaluate_fold(train, {"m": [1.0]}, test, {"m": [0.9] * 3}, budget=1)
    assert report["methods"]["m"]["fixed"]["reviews"] == 2
    assert report["methods"]["m"]["fixed"]["catches"] == 0
    assert report == evaluate_fold(train, {"m": [1.0]}, test[::-1], {"m": [0.9] * 3}, budget=1)


def test_zero_budget_with_one_scores_disables_live_and_fixed() -> None:
    result = evaluate_fold([turn("tr", 1, 3)], {"m": [1.0]}, [turn("te", 1, 3)], {"m": [1.0]}, budget=0)
    assert result["methods"]["m"]["fixed"]["reviews"] == result["methods"]["m"]["live"]["reviews"] == 0


def test_paired_deltas_are_conditional_reproducible_and_rebuild_the_queue() -> None:
    train = [turn("tr1", 1, 1), turn("tr2", 1, 0)]
    test = [turn("te1", 1, 2), turn("te2", 1, 1)]
    tr = {"stack": [0.1, 0.2], "best_single": [0.7, 0.8]}
    te = {"stack": [0.9, 0.4], "best_single": [0.55, 0.60]}
    result = paired_uncertainty([(train, tr, test, te)], budget=0.5, seed=23, replicates=100)
    assert result == paired_uncertainty([(train, tr, test, te)], budget=0.5, seed=23, replicates=100)
    assert result["fixed_catch_delta"] == {"observed": 1, "percentile_95": [0, 1]}
    assert result["fixed_review_delta"] == {"observed": 0, "percentile_95": [0, 0]}
    assert result["fixed_stack_only_catches"]["observed"] == 1
    assert result["live_review_delta"]["observed"] == 2


def test_reject_leakage_and_invalid_scores() -> None:
    train, test = [turn("tr", 1, 0)], [turn("te", 1, 3)]
    with pytest.raises(ValueError, match="both development and test"):
        evaluate_fold(train, {"m": [0.7]}, train, {"m": [0.2]}, budget=0.1)
    with pytest.raises(ValueError, match="duplicate turn_index"):
        evaluate_fold(train, {"m": [0.7]}, test * 2, {"m": [0.2] * 2}, budget=0.1)
    with pytest.raises(ValueError, match="probability"):
        evaluate_fold(train, {"m": [0.7]}, test, {"m": [math.nan]}, budget=0.1)


def test_exact_budget_allocation_by_largest_remainder() -> None:
    assert allocate_slots([3, 3, 3], 2) == [1, 1, 0]
    assert allocate_slots([1, 4, 6], 3) == [0, 1, 2]
    for k in range(68):
        sizes = [14, 14, 13, 13, 13]
        alloc = allocate_slots(sizes, k)
        assert sum(alloc) == k
        assert all(0 <= n <= s for n, s in zip(alloc, sizes, strict=True))
    with pytest.raises(ValueError, match="total slots"):
        allocate_slots([1, 2], 4)


def test_slot_override_affects_queue_and_its_bootstrap_not_live() -> None:
    tr = {"stack": [0.1, 0.2], "best_single": [0.7, 0.8]}
    te = {"stack": [0.9, 0.4], "best_single": [0.55, 0.60]}
    tr_rows = [turn("tr1", 1, 1), turn("tr2", 1, 0)]
    folds = [(tr_rows, tr, [turn(f"{i}-a", 1, 2), turn(f"{i}-b", 1, 0)], te) for i in (1, 2)]
    slots = allocate_slots([2, 2], 1)
    reports = [evaluate_fold(*f, budget=0.5, slots=k) for f, k in zip(folds, slots, strict=True)]
    assert [r["methods"]["stack"]["fixed"]["reviews"] for r in reports] == [1, 0]
    assert [r["methods"]["stack"]["live"]["reviews"] for r in reports] == [2, 2]
    boot = paired_uncertainty(folds, budget=0.5, slots_per_fold=slots, replicates=100)
    assert boot["fixed_catch_delta"] == {"observed": 1, "percentile_95": [0, 1]}
    assert boot["live_review_delta"]["observed"] == 4
    with pytest.raises(ValueError, match="one slot allocation"):
        paired_uncertainty(folds, budget=0.5, slots_per_fold=[1])
