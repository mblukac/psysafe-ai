import json
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard import update_pilot as pilot
from benchmarks.mindguard.full_panel import MODEL_KEYS
from benchmarks.mindguard.staffing import TRACE_SHA256
from benchmarks.mindguard.staffing_queue import Trajectory
from benchmarks.mindguard.surge_stream import AZURE_SHA256, BURST_SHA256
from benchmarks.mindguard.update_pair import fit_pair
from benchmarks.mindguard.update_queue import ARMS, prepare_path, replay_updates


def run_block(fold: int, placement: int) -> dict[str, Any]:
    # A synthetic waiting ticket gains endorsement (fold 0) or loses it (fold
    # 1) when the reviewer receives the latest prefix. Each placement repeats
    # the *same* source; only the fold weights and replicate positions vary.
    blocker: Trajectory = ((1, 0.9, 0),)
    waiting: Trajectory = ((1, 0.7, fold), (3, 0.9, 1 - fold))
    pair = fit_pair({"separate_train_family": ((1, 0.7, 0), (2, 0.9, 1))}, 0.65)
    plans = [prepare_path(s, tuple(q for _, q, _ in s), 0.65, pair_rule=pair) for s in (blocker, waiting)]
    affected: dict[str, tuple[set[int], set[int]]] = {}
    data = replay_updates(plans, [0, 16], [0, 1], [False, False], reviewers=1, affected_sources=affected)
    groups = {
        a: [
            (set(f"private source {fold} {i}" for i in ids), set(f"private family {fold} {i}" for i in ids))
            for ids in pair_ids
        ]
        for a, pair_ids in affected.items()
    }
    return dict(
        model="jev",
        gate="tuned",
        scenario="observed",
        staff=1,
        fold=fold,
        weight=2 - fold,
        placement=placement,
        data=data,
        affected=groups,
    )


def test_queue_to_aggregate_matches_realized_timing_weights_and_unique_sources() -> None:
    blocks = [run_block(f, p) for f, p in product((0, 1), repeat=2)]
    cell = pilot.aggregate(blocks)
    assert (cell["review_minutes"], cell["starts"], cell["updatable_starts"]) == (40, 2, 1)
    assert cell["arms"]["frozen"]["recall"] == pytest.approx(1 / 3)
    latest = cell["arms"]["latest"]
    assert latest["recall"] == pytest.approx(2 / 3)
    assert latest["delta"] == pytest.approx(1 / 3)
    assert (latest["rescued"], latest["spoiled"]) == pytest.approx((2 / 3, 1 / 3))
    assert latest["omit_fold_range"] == [-1, 1]
    assert latest["placement_range"] == pytest.approx([1 / 3, 1 / 3])
    assert (
        latest["rescued_paths"],
        latest["spoiled_paths"],
        latest["rescued_families"],
        latest["spoiled_families"],
    ) == (
        1,
        1,
        1,
        1,
    )
    assert cell["arms"]["visible_oracle"]["recall"] == 1.0
    assert "private" not in json.dumps(cell)


def test_aggregation_rejects_duplicate_layout_invalid_transition_or_ceiling() -> None:
    b = run_block(0, 0)
    with pytest.raises(ValueError, match="duplicate"):
        pilot.aggregate([b, b])
    with pytest.raises(ValueError, match="conditions"):
        pilot.aggregate([b, run_block(1, 0) | {"gate": "liberal"}])
    with pytest.raises(ValueError, match="immediate"):
        pilot.aggregate([b | {"staff": 0}])
    b["data"]["timely_positive_starts"] = 0
    with pytest.raises(ValueError, match="ceiling"):
        pilot.aggregate([b])
    b["data"]["arms"]["latest"]["rescued"] = 0
    with pytest.raises(ValueError, match="identity"):
        pilot.aggregate([b])


SCENARIOS = (
    "steady",
    "steady_pulse",
    "observed",
    "risk_pulse",
    "double",
    "compound",
    "azure",
    "burst_week",
    "burst_week_steady",
)


def fake_reference(tmp_path: Path) -> list[dict[str, Any]]:
    models = sorted(MODEL_KEYS.values())
    results = []
    for g in ("tuned", "liberal"):
        rows = []
        for m, s, k in product(models, SCENARIOS, (1, 2, 3, 0)):
            base = dict(
                model=m, scenario=s, staff=k, sessions=10.0, positives=3.0, nominated=6.0, starts=5.0, expired=1.0
            )
            vals = dict(**base, policy="fifo", mean_review_minutes=100, catches=1.0, unsupported_starts=4.0)
            rows.append(list(vals.values()))
            results.append(
                dict(
                    **base,
                    gate=g,
                    review_minutes=100,
                    arms={a: dict(catches=1.0, unsupported_starts=4.0) for a in ARMS},
                )
            )
        ref = dict(
            kind="mindguard_public_operating_points_v1",
            models=models,
            gate=g,
            service=20,
            deadline_minutes=10,
            turn_minutes=2,
            placements=pilot.PLACEMENTS,
            seed=pilot.SEED,
            rated_turns=1134,
            dialogues=67,
            families=50,
            trace_sha256=TRACE_SHA256,
            azure_sha256=AZURE_SHA256,
            burst_sha256=BURST_SHA256,
            rows=rows,
            columns=list(vals),
        )
        (tmp_path / f"full-panel-{g}.json").write_text(json.dumps(ref))
    return results


def test_valid_full_synthetic_grid_and_fail_closed_on_drift(tmp_path: Path) -> None:
    results = fake_reference(tmp_path)
    pilot.verify_previous(results, tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        pilot.verify_previous(results[:-1], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        pilot.verify_previous([results[1], *results[1:]], tmp_path)
    results[0]["review_minutes"] += 20
    with pytest.raises(ValueError, match="reproduce"):
        pilot.verify_previous(results, tmp_path)
    results[0]["review_minutes"] -= 20
    results[0]["arms"]["frozen"]["catches"] = 2
    with pytest.raises(ValueError, match="reproduce"):
        pilot.verify_previous(results, tmp_path)
    del results[0]["arms"]["visible_oracle"]
    with pytest.raises(ValueError, match="missing update arms"):
        pilot.verify_previous(results, tmp_path)


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("clock", "original clocks"),
        ("model", "original clocks"),
        ("scenario", "baseline FIFO"),
        ("duplicate", "duplicate baseline FIFO"),
        ("duplicate_column", "duplicate baseline columns"),
    ],
)
def test_malformed_published_reference_cannot_unlock_report(tmp_path: Path, fault: str, reason: str) -> None:
    results = fake_reference(tmp_path)
    path = tmp_path / "full-panel-tuned.json"
    ref = json.loads(path.read_text())
    if fault == "clock":
        ref["burst_sha256"] = "unverified synthetic digest"
    elif fault == "model":
        ref["models"][0] = "unknown-model"
    elif fault == "scenario":
        ref["rows"][0][ref["columns"].index("scenario")] = "unverified scenario"
        results[0]["scenario"] = "unverified scenario"
    elif fault == "duplicate":
        ref["rows"].append(ref["rows"][0])
    else:
        ref["columns"].append("catches")
    path.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match=reason):
        pilot.verify_previous(results, tmp_path)
