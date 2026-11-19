from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.staffing_plot import main, plot_data, render


def report() -> dict[str, Any]:
    return {
        "kind": "trace_driven_counterfactual_staffing_not_production_recall",
        "trace": {"arrivals": 20},
        "source": {"dialogues": 3, "positive_dialogues": 1},
        "scenario": {"start_deadline_minutes": 5, "reviewers": [3, 1]},
        "results": [
            {
                "model": model,
                "reviewers": staff,
                "policy": policy,
                "recall": (staff / 4 if policy != "native_fixed" else 0),
            }
            for model in ("liquid", "luna", "jev", "gemini")
            for policy in ("native_fixed", "matched_fixed", "fixed", "paced", "rl")
            for staff in (3, 1)
        ],
    }


def test_figure_points_use_reported_recall_in_order_with_no_jitter_or_native_curve() -> None:
    d = plot_data(report())
    assert d.staffing == (1, 3)
    assert d.models == ("jev", "gemini", "liquid", "luna")
    assert len(d.percents) == 12
    assert d.percents["luna", "rl"] == d.percents["luna", "fixed"] == [25, 75]
    assert "5-minute" in d.subtitle
    assert "3 source dialogues (1 positive)" in d.subtitle


def test_missing_points_or_missing_outcomes_are_not_plotted_as_zero() -> None:
    r = report()
    r["results"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        plot_data(r)
    r = report()
    r["results"][-1]["recall"] = None
    with pytest.raises(ValueError, match="unassessable"):
        plot_data(r)


def test_plot_rejects_nonaggregate_data_and_invalid_destinations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aggregate"):
        plot_data({"raw_rows": []})
    with pytest.raises(ValueError, match="PNG or SVG"):
        render(plot_data(report()), tmp_path / "result.json")
    path = str(tmp_path / "missing.json")
    assert main(["--report", path, "--out", path, "--overwrite"]) == 2
