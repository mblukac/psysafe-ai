from math import inf, nan

import pytest

from benchmarks.mindguard.staffing_fit import Observation, StoppingPolicy, matched_cutoff, train_policies
from benchmarks.mindguard.staffing_queue import Session, simulate


def test_one_turn_paths_have_no_continuation_even_for_unseen_states() -> None:
    policy = train_policies([[(1, 0.0, 0)], [(1, 1.0, 1)]])
    for score in (0.0, 0.2, 1.0):
        assert 0 < policy.q(score) < 1
        assert policy.advantage(score, 30) == policy.q(score)
    assert policy.q(1.0) > policy.q(0.0)


def test_same_marginal_predictions_different_trajectories_change_wait_value() -> None:
    # The scores/labels at every age have the same pooled distribution across
    # the two fits. An incoming 0.25 predicts a later high score only in A.
    mix_a = [[(1, 0.25, 0), (2, 0.95, 1)], [(1, 0.03, 0), (2, 0.01, 0)]]
    mix_b = [[(1, 0.25, 0), (2, 0.01, 0)], [(1, 0.03, 0), (2, 0.95, 1)]]
    a, b = train_policies(mix_a * 12), train_policies(mix_b * 12)
    assert a.q(0.25) == pytest.approx(b.q(0.25))
    assert a.advantage(0.25, 1) < b.advantage(0.25, 1) - 0.05


def test_bellman_backup_propagates_beyond_one_step() -> None:
    policy = train_policies([[(1, 0.001, 0), (2, 0.1, 0), (3, 0.95, 1)]] * 20)
    wait_value = policy.q(0.001) - policy.advantage(0.001, 1)
    assert wait_value > policy.q(0.1) + 0.1
    assert policy.bellman_iterations > 2


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ([], "nonempty"),
        ([(2, 0.5, 1)], "consecutive"),
        ([(1, 0.5, 0), (3, 0.3, 1)], "gaps"),
        ([(1, 0.5, -1)], "observed binary"),
        ([(1, 1.1, 1)], "finite"),
        ([(1, nan, 0)], "finite"),
    ],
)
def test_invalid_training_observations_fail_closed(path: list[Observation], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        train_policies([path])


def test_inference_rejects_invalid_scores_and_elapsed_turns() -> None:
    policy = train_policies([[(1, 0.0, 0)]])
    with pytest.raises(ValueError, match="finite"):
        policy.q(inf)
    with pytest.raises(ValueError, match="positive integer"):
        policy.advantage(0.4, 0)


def test_matched_score_rule_matches_rl_admissions_when_there_is_no_congestion() -> None:
    paths = [((1, 0.1, 0),), ((1, 0.5, 1),), ((1, 0.8, 1),)]
    # q is native; all training continuations valued at 0.2.
    policy = StoppingPolicy((0.0, 1.0), (0.2,) * 24, 0)
    threshold = matched_cutoff(paths, policy, 0.4)
    assert 0.5 < threshold <= 0.8
    sessions = [Session(float(i * 60), path) for i, path in enumerate(paths)]
    common = dict(service_minutes=1.0, turn_minutes=1.0, deadline_minutes=1.0)
    fixed = simulate(
        sessions, reviewers=1, policy="fixed", metric=lambda s, _: policy.q(s), threshold=threshold, **common
    )
    rl = simulate(sessions, reviewers=1, policy="rl", metric=policy.advantage, threshold=0.4, **common)
    assert fixed["alerts"] == rl["alerts"] == fixed["reviews"] == rl["reviews"] == 1


def test_matched_cutoff_conservatively_excludes_a_tied_boundary_block() -> None:
    paths = [[(1, 0.5, 1)], [(1, 0.1, 0), (2, 0.1, 0), (3, 0.1, 0), (4, 0.5, 1)]]
    # Only the late peak has positive stop advantage; q peaks tie.
    policy = StoppingPolicy((0.0, 1.0), (1.0, 0.0, 0.0, 0.0) * 6, 0)
    threshold = matched_cutoff(paths, policy, 0.25)
    assert threshold > policy.q(0.5)


def test_matched_cutoff_all_or_none_does_not_depend_on_vote_readout() -> None:
    paths = [[(1, 0.1, 0)], [(1, 0.8, 0)]]
    policy = StoppingPolicy((0.0, 1.0), (0.0,) * 24, 0)
    assert matched_cutoff(paths, policy, 0.0) == 0.0
    assert matched_cutoff(paths, policy, 1.0) > 1.0
