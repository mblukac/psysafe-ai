import pytest

from benchmarks.mindguard.update_pair import fit_pair


def test_no_training_opportunities_keeps_frozen_and_requires_visibility() -> None:
    rule = fit_pair({"x": [(1, 0.3, 0), (7, 0.9, 1)]}, 0.3)
    assert rule.eligible_families == rule.eligible_paths == rule.landmarks == 0
    assert rule.choose([(1, 0.3), (3, 1.0)], 4) == 1
    assert rule.choose([(1, 0.3)], 0) == 1
    with pytest.raises(ValueError, match="visible"):
        rule.choose([(1, 0.3), (3, 1.0)], 3.99)
    with pytest.raises(ValueError, match="unexpired"):
        rule.choose([(1, 0.3)], 10.01)


def test_pair_posterior_expected_gain_and_false_switch_cost() -> None:
    gain = [(1, 0.5, 0), (2, 0.9, 1)]
    loss = [(1, 0.5, 1), (2, 0.9, 0)]
    positive = fit_pair({"a": gain}, 0.5)
    assert positive.landmarks == 9
    assert positive.counts[2] == (0, 1, 0, 0)
    assert positive.gain(0.5, 0.9) == pytest.approx(0.75 / 3.25)
    assert positive.replace(0.5, 0.9)
    assert positive.choose([(1, 0.5), (2, 0.9), (3, 0.9)], 4) == 2
    balanced = fit_pair({"a": gain, "b": loss}, 0.5)
    assert not balanced.replace(0.5, 0.9)
    assert balanced.eligible_families == 2
    assert not positive.replace(0, 0)
    assert not positive.replace(1, 1)


def test_no_pseudoreplication_from_copying_a_whole_family() -> None:
    src = [(1, 0.5, 0), (2, 0.9, 1)]
    one = fit_pair({"a": src}, 0.5)
    two = fit_pair({"a": src, "b": src}, 0.5, families={"a": "family", "b": "family"})
    assert one.counts == two.counts
    assert two.eligible_families == 1
    assert two.eligible_paths == 2
    assert two.landmarks == 2 * one.landmarks
    with pytest.raises(ValueError, match="family"):
        fit_pair({"a": src}, 0.5, families={"b": "family"})


def test_real_turn_gaps_and_true_deadline_matter_to_reward_not_action() -> None:
    # At age ten (not sooner) the later observed turn can first be selected.
    at_expiry = fit_pair({"x": [(2, 0.5, 0), (7, 0.9, 1)]}, 0.5)
    assert at_expiry.landmarks == 1
    assert at_expiry.replace(0.5, 0.9)
    # Both appear "supported" at some time, but true onset predates the
    # original alert by ten minutes; every later candidate is already late.
    stale = fit_pair({"x": [(1, 0.1, 1), (6, 0.5, 0), (7, 0.9, 1)]}, 0.5)
    assert stale.counts[2] == (1, 0, 0, 0)
    assert not stale.replace(0.5, 0.9)


def test_guarding_selection_trains_on_same_challenger_as_inference() -> None:
    # Current-strongest challenger changes at turn 4. One correct opportunity
    # must not become many spurious gains by comparing every available pair.
    rule = fit_pair({"x": [(1, 0.5, 0), (2, 0.6, 1), (4, 0.9, 0)]}, 0.5)
    assert rule.landmarks == 9
    assert rule.counts[1][1] == pytest.approx(4 / 9)
    assert rule.counts[2][0] == pytest.approx(5 / 9)
    assert rule.choose([(1, 0.5), (2, 0.6)], 2) == 2
    assert rule.choose([(1, 0.5), (2, 0.6), (4, 0.9)], 6) == 1
