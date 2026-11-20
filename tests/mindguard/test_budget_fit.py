from collections.abc import Sequence

import pytest

from benchmarks.mindguard import budget_fit
from benchmarks.mindguard.data import Turn


def _rows() -> list[Turn]:
    return [
        Turn(str(d), t, (("self_harm" if d % 3 == 0 and t == 2 else "safe"),), (s, s, s))
        for d in range(10)
        for t, s in ((1, 0.11), (2, 0.81 if d % 3 == 0 else 0.05))
    ]


def test_choice_requires_full_budget_in_every_placement_if_possible() -> None:
    candidates = {0.05: [(15, 3), (12, 2)], 0.25: [(12, 6), (9, 7)]}
    assert budget_fit._choice(candidates, 10)[0] == 0.05
    candidates = {0.05: [(12, 6), (10, 6)], 0.2: [(12, 6), (10, 6)]}
    assert budget_fit._choice(candidates, 10)[0] == 0.2


def test_best_actual_captures_used_when_every_cutoff_underfills() -> None:
    candidates = {0.05: [(9, 4), (8, 4)], 0.2: [(7, 5), (7, 6)]}
    assert budget_fit._choice(candidates, 10)[0] == 0.2


def test_ceiling_prefers_hits_to_budget_exhaustion() -> None:
    candidates = {0.05: [(15, 3), (12, 2)], 0.25: [(12, 6), (9, 7)]}
    assert budget_fit._choice(candidates, 10, require_fill=False)[0] == 0.25


def test_ceiling_ties_prefer_fewer_capped_starts_then_higher_cutoff() -> None:
    # The .10 candidate uses 15 starts under the cap versus 16 at .25,
    # although it would have more starts without a cap.
    candidates = {0.05: [(20, 4), (5, 2)], 0.10: [(20, 4), (5, 2)], 0.25: [(8, 3), (8, 3)]}
    assert budget_fit._choice(candidates, 10, require_fill=False)[0] == 0.10


def test_dense_grid_is_optional_and_can_resolve_a_subcent_cutoff() -> None:
    assert set(budget_fit.GRID) < set(budget_fit.DENSE_GRID)
    path = ((1, 0.0001, 0), (2, 0.001, 1))
    oof: budget_fit._Preds = {"a": (path, path)}
    default = budget_fit._tune(oof, [0], 1, (1,), 0, 1)
    assert budget_fit._tune(oof, [0], 1, (1,), 0, 1, dense_grid=False) == default
    dense = budget_fit._tune(oof, [0], 1, (1,), 0, 1, dense_grid=True)
    assert default[0][1]["fixed"] == 0.0
    assert dense[0][1]["fixed"] == 0.001


@pytest.mark.parametrize("require_fill", [True, False])
def test_heldout_scores_ids_and_counts_are_consistent(require_fill: bool) -> None:
    rows = _rows()
    scored, audit = budget_fit.tune_folds(
        rows,
        [float(20 * i) for i in range(50)],
        model="gemini",
        staff=1,
        budgets=(4, 12),
        seed=2026,
        inner_placements=1,
        require_fill=require_fill,
    )
    assert len(audit) == 5 * 2 * 3
    for b in (4, 12):
        assert set(scored[b]) == {"fixed", "rl", "rl_fifo"}
        for sessions in scored[b].values():
            assert len(sessions) == 10
            for d, (path, price) in enumerate(sessions):
                assert path[1][2] == int(d % 3 == 0)
                assert price in budget_fit.GRID
        # Queueing changes the choice of cut, never the metric itself.
        assert [p for p, _ in scored[b]["rl"]] == [p for p, _ in scored[b]["rl_fifo"]]
    for details in audit:
        assert details["selection_mode"] == ("quota" if require_fill else "ceiling")
        assert details["inner_used_starts"] == [min(details["budget_starts"], details["inner_uncapped_starts"][0])]
        assert details["inner_catches_by_onset_deadline"][0] <= details["inner_used_starts"][0]


def test_outer_heldout_votes_never_enter_that_folds_tuning_or_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Freeze assignments since the official splitter deliberately stratifies on
    # reference labels. Conditioning on that split, only the evaluation targets
    # (not held-out metrics or prices) may change when relabeling a held-out set.
    def split(rows: list[Turn], n_splits: int, seed: int) -> list[tuple[list[Turn], list[Turn]]]:
        ids = sorted({r.dialogue_id for r in rows})
        names = [set(ids[f::n_splits]) for f in range(n_splits)]
        return [
            ([r for r in rows if r.dialogue_id not in test], [r for r in rows if r.dialogue_id in test])
            for test in names
        ]

    def fit(rows: list[Turn]) -> tuple[dict[int, dict[str, list[budget_fit.TunedPath]]], list[dict[str, object]]]:
        return budget_fit.tune_folds(
            rows, [i * 3.0 for i in range(14)], model="jev", staff=2, budgets=(8,), seed=21, inner_placements=1
        )

    monkeypatch.setattr(budget_fit, "grouped_folds", split)
    source = _rows()
    original, audit = fit(source)
    swapped, new_audit = fit(
        [
            (
                Turn(r.dialogue_id, r.turn_index, ("self_harm" if r.votes[0] == "safe" else "safe",), r.scores)
                if r.dialogue_id in {"0", "5"}
                else r
            )
            for r in source
        ]
    )
    assert [r for r in audit if r["outer_fold"] == 1] == [r for r in new_audit if r["outer_fold"] == 1]
    for name in budget_fit.POLICIES:
        for idx in (0, 5):
            (path, cut), (new_path, new_cut) = original[8][name][idx], swapped[8][name][idx]
            assert cut == new_cut
            assert [(t, s) for t, s, _ in path] == [(t, s) for t, s, _ in new_path]
            assert [y for _, _, y in path] != [y for _, _, y in new_path]


@pytest.mark.parametrize(("arrivals", "budgets"), [([], (1,)), ([1.0, 0.0], (1,)), ([1.0], (-1,)), ([0.0], (1, 1))])
def test_invalid_queue_inputs_raise(arrivals: Sequence[float], budgets: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"arrival|budget"):
        budget_fit.tune_folds(_rows(), arrivals, model="jev", staff=1, budgets=budgets, seed=1)
