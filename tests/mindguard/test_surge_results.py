import json

import pytest

from benchmarks.mindguard.surge_results import public_points


def test_public_export_only_contains_aggregate_fields() -> None:
    row = dict(
        model="jev",
        scenario="steady",
        staff=1,
        policy="fifo",
        recall=0.25,
        delta_vs_fifo=0.0,
        queue_loss=0.25,
        mean_review_minutes=20,
        fraction_starts_contested=0.1,
        by_period={"peak": {"recall": None}, "offpeak": {"recall": 0.25}},
        text="PRIVATE",
    )
    report = dict(
        kind="mindguard_surge_v1", gate="tuned", seed=1, service=20, placements=8, training=["PRIVATE"], results=[row]
    )
    points = public_points(report)
    assert points["results"][0]["peak_recall"] is None
    assert points["gate"] == "tuned"
    assert "PRIVATE" not in json.dumps(points)
    row["scenario"] = "PRIVATE"
    with pytest.raises(ValueError, match="operating point"):
        public_points(report)
