import math

import pytest

from benchmarks.mindguard.staffing_queue import Trajectory
from benchmarks.mindguard.update_fit import fit_filter

PATHS: tuple[Trajectory, ...] = (
    ((1, 0.05, 0), (2, 0.90, 1)),
    ((1, 0.05, 0), (2, 0.03, 0)),
    ((1, 0.91, 1), (2, 0.99, 1)),
)


def test_bayes_step_by_hand_and_strict_probabilities() -> None:
    fitted = fit_filter(PATHS)
    # Three positives among six training turns; transitions 0->1:1/2, 1->1:1/1.
    assert (fitted.prevalence, fitted.training_turns, fitted.adjacent_pairs) == (0.5, 6, 3)
    assert fitted.transition == pytest.approx((0.5, 2 / 3))
    # From low to high bin: one positive, zero negative, pooled mass (1+.5)/(2+2).
    # Prior on each label-specific high-bin probability has total strength four.
    assert (fitted.emission[1][0][3], fitted.emission[0][0][3]) == pytest.approx((0.5, 0.3))
    u = 0.95 * 0.5 + 0.05 * 2 / 3
    hi = u * 0.5 / (u * 0.5 + (1 - u) * 0.3)
    p = fitted.posterior([(1, 0.05), (2, 1.0), (3, 0.0)])
    assert p[:2] == pytest.approx((0.05, hi))
    assert 0 < p[2] < 1
    assert hi > fitted.posterior([(1, 0.05), (2, 0.02)])[-1]
    for table in fitted.emission:
        for row in table:
            assert all(x > 0 for x in row)
            assert sum(row) == pytest.approx(1.0)


def test_gap_or_new_conversation_never_fabricates_a_label_transition() -> None:
    gap: Trajectory = ((1, 0.05, 0), (3, 0.9, 1), (4, 0.95, 1))
    f = fit_filter([gap])
    assert f == fit_filter([gap[:1], gap[1:]])
    assert f.adjacent_pairs == 1
    assert f.posterior([(1, 0.05), (3, 0.9), (4, 0.95)])[1:] == f.posterior([(3, 0.9), (4, 0.95)])
    assert f.posterior([(3, 1.0), (5, 0.0)]) == (1, 0)


def test_neutral_repeated_observations_do_not_compound_the_full_prefix_odds() -> None:
    # Both label populations emit the same repeated high bin. The evolution
    # is the label transition model only, NOT another factor of .99/.01.
    f = fit_filter([((1, 0.99, y), (2, 0.99, y)) for y in (0, 1)])
    p = f.posterior([(i, 0.99) for i in range(1, 18)])
    assert f.emission[0][3] == f.emission[1][3]
    assert p[1] == pytest.approx((1 - p[0]) * f.transition[0] + p[0] * f.transition[1])
    assert 0.5 <= p[-1] < p[1] < p[0]


def test_causal_prefix_and_empty_bins_or_single_training_class() -> None:
    f = fit_filter([((1, 0.0, 0), (2, 0.1, 0))])
    assert min(f.transition) > 0
    assert f.emission[0][3] == f.emission[1][3]
    obs = [(1, 0.82), (2, 0.5), (3, 0.80), (6, 0.70)]
    assert f.posterior(obs)[:2] == f.posterior(obs[:2])
    assert f.posterior([]) == ()
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in f.posterior(obs))


@pytest.mark.parametrize("q", [-0.1, float("nan"), float("inf"), 1.1, False])
def test_invalid_score_rejected_in_fit_and_predict(q: float) -> None:
    f = fit_filter(PATHS)
    with pytest.raises(ValueError, match="probability"):
        fit_filter([((1, q, 0),)])
    with pytest.raises(ValueError, match="probability"):
        f.posterior([(1, q)])


def test_reject_invalid_train_or_runtime_indices_and_labels() -> None:
    with pytest.raises(ValueError, match="no labeled"):
        fit_filter([])
    with pytest.raises(ValueError, match="empty development"):
        fit_filter([()])
    with pytest.raises(ValueError, match="binary"):
        fit_filter([((1, 0.2, 2),)])
    with pytest.raises(ValueError, match="increasing"):
        fit_filter([((1, 0.2, 0), (1, 0.3, 1))])
    with pytest.raises(ValueError, match="increasing"):
        fit_filter(PATHS).posterior([(2, 0.2), (1, 0.3)])
