from dataclasses import dataclass, replace
from math import log

import pytest

from benchmarks.mindguard.data import Turn
from benchmarks.mindguard.fit import SINGLES, brier_loss, fold_predictions, grouped_folds, log_loss


def row(d: str, t: int, votes: tuple[str, ...], scores: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> Turn:
    return Turn(dialogue_id=d, turn_index=t, votes=votes, scores=scores)


def test_individual_vote_losses():
    split = [row("d", 0, ("self_harm", "self_harm", "harm_others"))]
    assert log_loss(split, [0.65]) == pytest.approx(-(2 * log(0.65) + log(0.35)) / 3)
    assert log_loss(split, [0.65]) < log_loss(split, [0.99])
    assert brier_loss(split, [0.65]) == pytest.approx((2 * 0.35**2 + 0.65**2) / 3)


def test_grouped_folds_partition_and_stratify_complete_dialogues():
    rows = [row(str(i), t, ("self_harm",) * 3 if i < 3 and t == 2 else ("safe",) * 3) for i in range(9) for t in (1, 2)]
    folds = grouped_folds(rows, 3, 42)
    ids = set()
    for train, test in folds:
        before = {r.dialogue_id for r in train}
        after = {r.dialogue_id for r in test}
        assert before.isdisjoint(after)
        assert ids.isdisjoint(after)
        assert len(after) == 3
        assert len(after & {"0", "1", "2"}) == 1
        ids.update(after)
    assert len(ids) == 9
    assert [sorted(r.dialogue_id for r in b) for _, b in folds] == [
        sorted(r.dialogue_id for r in b) for _, b in grouped_folds(list(reversed(rows)), 3, 42)
    ]


@dataclass(frozen=True)
class _FamilyTurn(Turn):
    group_id: str = ""


def test_shared_openings_travel_together_and_folds_balance_dialogues():
    rows = [
        _FamilyTurn(f"d{i}", 1, ("safe",) * 3, (0.5,) * 3, group_id="shared" if i < 3 else f"f{i}") for i in range(9)
    ]
    folds = grouped_folds(rows, 3, 42)
    assert all(len(test) == 3 for _, test in folds)
    for train, test in folds:
        assert {r.group_id for r in train}.isdisjoint(r.group_id for r in test)
    with pytest.raises(ValueError, match="inconsistent family"):
        grouped_folds([*rows, replace(rows[0], turn_index=2, group_id="different")], 3, 42)
    with pytest.raises(ValueError, match="both train and test"):
        fold_predictions(rows, [replace(rows[0], dialogue_id="other")])


def test_inner_oof_holds_out_entire_shared_family():
    rows = [_FamilyTurn(f"d{i}", 1, ("self_harm",) * 3, (0.5,) * 3, group_id="shared") for i in range(3)]
    lone = row("negative", 1, ("safe",) * 3)
    test = [row("new", 1, ("safe",) * 3)]
    oof, _, _ = fold_predictions([*rows, lone], test)
    assert all(p < 0.05 for p in oof["stack"][:3])
    assert oof["stack"][3] > 0.95
    with pytest.raises(ValueError, match="two independent"):
        fold_predictions(rows, test)


def test_scores_and_winner_are_independent_of_test_votes():
    train = [
        row(str(i), 1, (("safe",) if i % 2 == 0 else ("self_harm",)) * 3, (0.05 if i % 2 == 0 else 0.95, 0.5, 0.5))
        for i in range(8)
    ]
    test = [row("new", t, ("safe",) * 3, (x, 0.5, 0.5)) for t, x in enumerate((0, 1))]
    oof, out, winner = fold_predictions(train, test)
    changed_test = [replace(r, votes=("self_harm",) * 3) for r in test]
    assert (oof, out, winner) == fold_predictions(train, changed_test)
    assert winner == "jev"
    assert set(out) == set(oof) == {*SINGLES, "stack", "pair", "equal", "best_single"}
    assert out["equal"] == pytest.approx([1 / 3, 2 / 3])
    assert out["best_single"] == out["jev"]
    assert all(0 <= p <= 1 for scores in out.values() for p in scores)
    assert out["stack"][0] < out["stack"][1]


def test_training_scores_are_out_of_fold_even_in_one_class_folds():
    # A held-out dialogue whose own votes are *all positive* must receive a low
    # prediction if every other training dialogue has *only negative* votes.
    train = [row("positive", 1, ("self_harm",) * 3), row("negative", 1, ("safe",) * 3)]
    test = [row("new", 1, ("safe",) * 3)]
    oof, final, _ = fold_predictions(train, test)
    assert oof["stack"][0] < 0.05
    assert oof["stack"][1] > 0.95
    assert final["stack"] == pytest.approx([0.5])


def test_fails_closed_on_leakage_invalid_scores_and_invalid_folds():
    train = [row("one", 1, ("safe",) * 3), row("two", 1, ("harm_others",) * 3)]
    with pytest.raises(ValueError, match="both train and test"):
        fold_predictions(train, [replace(train[0], turn_index=2)])
    with pytest.raises(ValueError, match="duplicate"):
        grouped_folds(train + train[:1], 2, 0)
    with pytest.raises(ValueError, match="fold count"):
        grouped_folds(train, 3, 0)
    with pytest.raises(ValueError, match="finite probabilities"):
        brier_loss(train, [float("nan"), 0])
    with pytest.raises(ValueError, match="empty"):
        log_loss([], [])
