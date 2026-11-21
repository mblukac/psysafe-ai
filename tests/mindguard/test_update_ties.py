from benchmarks.mindguard.update_ties import choices


def test_latest_tie_does_not_turn_approximate_ties_into_ties() -> None:
    assert choices((0.5, 0.5, 0.5), (False, True, False)) == (0, 2, 1)
    assert choices((0.5, 0.5 - 1e-12), (False, True, False)) == (0, 0, 1)
    assert choices((0.5, 0.5 + 1e-12), (False, False, False)) == (1, 1, 0)
