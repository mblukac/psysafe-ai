from dataclasses import replace

import pytest

from benchmarks.mindguard.ensemble_fit import (
    ENSEMBLE,
    N_ARRIVALS,
    N_STREAMS,
    SINGLES,
    START_BUDGET,
    TrainingStream,
    encoded_arm,
    fit_fold,
    train_gate,
    training_streams,
)
from benchmarks.mindguard.full_panel import PanelTurn


def row(i, t, y):
    return PanelTurn(
        f"d{i}",
        f"g{i}",
        t,
        (("self_harm",) if y else ("safe",)) * 3,
        {m: {"self_harm": (j + 1) / 10, "harm_others": 0.05} for j, m in enumerate(SINGLES)},
    )


def test_equal_pool_uses_native_either_risk_not_votes_or_max_or_label():
    r = row(0, 3, 1)
    q = encoded_arm([r], ENSEMBLE)[0]
    assert q.scores[0] == pytest.approx(0.4)  # mean of .15, .25, ..., .65
    assert q.turn_index == 3
    assert encoded_arm([replace(r, votes=("safe",) * 3)], ENSEMBLE)[0].scores == q.scores
    with pytest.raises(ValueError, match="all six"):
        encoded_arm([replace(r, scores={})], ENSEMBLE)
    with pytest.raises(ValueError, match="invalid"):
        encoded_arm([replace(r, scores=r.scores | {SINGLES[0]: {"self_harm": 1, "harm_others": 1}})], ENSEMBLE)


def test_training_gate_requires_full_work_on_each_stream_and_counts_positive_starts_only_if_timely():
    paths = [((1, 0.9, 1),), ((1, 0.1, 0),), ((1, 0.4, 0), (3, 0.7, 1))]
    stream = TrainingStream((0.0, 30.0, 60.0), (2, 0, 1))
    got = train_gate(paths, [stream, stream], budget=2, cutoffs=[0.9, 0.6, 0.0])
    assert (got.cutoff, got.caught, got.total_starts) == (0.6, 4, 4)
    with pytest.raises(ValueError, match="no gate"):
        train_gate(paths, [stream], budget=2, cutoffs=[0.9])
    # At the lower gate, early start on path 2 earns no future credit.
    assert train_gate(paths, [stream], budget=2, cutoffs=[0.0]).caught == 1


def test_training_clocks_are_reproducible_independent_of_test_data():
    positives = [False, False, False, True]
    a = training_streams(positives, 0)
    assert len(a) == N_STREAMS
    assert all(len(s.times) == len(s.order) == N_ARRIVALS for s in a)
    assert a == training_streams(positives, 0)
    assert a != training_streams(positives, 1)


def test_outer_test_labels_and_forecasts_cannot_affect_gate_or_single_choice(monkeypatch):
    def sparse_clocks(positive, fold):
        return [
            TrainingStream(
                tuple(i * 40.0 for i in range(START_BUDGET * 2)),
                tuple(i % len(positive) for i in range(START_BUDGET * 2)),
            )
        ] * N_STREAMS

    monkeypatch.setattr("benchmarks.mindguard.ensemble_fit.training_streams", sparse_clocks)
    panel = [row(i, t, int(i % 2 == 0 and t == 3)) for i in range(7) for t in (1, 3)]
    changed = [
        (
            replace(r, votes=("self_harm",) * 3, scores={m: {"self_harm": 0.99, "harm_others": 0} for m in SINGLES})
            if r.dialogue_id == "d6"
            else r
        )
        for r in panel
    ]
    ids = {f"d{i}" for i in range(6)}
    a, b = [fit_fold(p, ids, {"d6"}, 0) for p in (panel, changed)]
    assert a.audit == b.audit
    assert a.selected_single in SINGLES
    assert a.cutoffs == b.cutoffs
    assert set(a.paths) == {*SINGLES, ENSEMBLE}
    assert a.paths[ENSEMBLE]["d6"] != b.paths[ENSEMBLE]["d6"]
    with pytest.raises(ValueError, match="opening family"):
        fit_fold([replace(r, group_id="g0") if r.dialogue_id == "d6" else r for r in panel], ids, {"d6"}, 0)
