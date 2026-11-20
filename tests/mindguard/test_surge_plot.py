from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.surge_plot import main, plot_data, render


def example() -> dict[str, Any]:
    return {
        "kind": "mindguard_surge_v1",
        "results": [
            {
                "model": m,
                "scenario": s,
                "staff": n,
                "policy": p,
                "recall": {1: 0.5, 2: 0.6, 3: 0.75, 0: 0.9}[n] + {"fifo": 0.0, "learned": -0.12, "risk": 0.05}[p],
            }
            for m in ("gemini", "jev")
            for s in ("observed", "risk_pulse", "compound", "steady", "double")
            for n in (0, 3, 1, 2)
            for p in ("fifo", "learned", "risk")
        ],
    }


def test_arranges_exact_staff_counts_controls_and_signed_gains() -> None:
    d = plot_data(example())
    assert d.row_keys[:4] == (("jev", "steady"), ("jev", "observed"), ("jev", "risk_pulse"), ("jev", "compound"))
    assert d.row_keys[-1] == ("gemini", "compound")
    assert d.fifo_percent[0] == [50, 60, 75, 90]
    assert d.gain_pp[-1] == pytest.approx([5, -12])


def test_scenario_specific_fifo_must_be_the_comparator() -> None:
    r = example()
    row = next(
        row
        for row in r["results"]
        if (row["model"], row["scenario"], row["policy"], row["staff"]) == ("jev", "observed", "fifo", 1)
    )
    row["recall"] = 0.6
    d = plot_data(r)
    assert d.gain_pp[0] == pytest.approx([5, -12])
    assert d.gain_pp[1] == pytest.approx([-5, -22])


def test_weekly_pair_is_optional_but_both_controls_and_both_models_required_if_present() -> None:
    r = example()
    month = [row for row in r["results"] if row["scenario"] == "observed"]
    r["results"].extend({**row, "scenario": "burst_week"} for row in month)
    with pytest.raises(ValueError, match="incomplete"):
        plot_data(r)
    r["results"].extend({**row, "scenario": "burst_week_steady"} for row in month)
    d = plot_data(r)
    assert d.row_keys[4:6] == (("jev", "burst_week_steady"), ("jev", "burst_week"))
    assert d.row_keys[-2:] == (("gemini", "burst_week_steady"), ("gemini", "burst_week"))
    assert d.gain_pp[5] == pytest.approx([5, -12])


def test_rejects_duplicate_conditions_missing_unlimited_and_unassessable_rates() -> None:
    r = example()
    r["results"].append(r["results"][0])
    with pytest.raises(ValueError, match="duplicate"):
        plot_data(r)
    r = example()
    r["results"].pop(0)
    with pytest.raises(ValueError, match="incomplete"):
        plot_data(r)
    r = example()
    r["results"][0]["recall"] = None
    with pytest.raises(ValueError, match="unassessable"):
        plot_data(r)


def test_protects_input_and_existing_output_without_rendering(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aggregate"):
        plot_data({"kind": "raw"})
    path = tmp_path / "original.png"
    path.write_bytes(b"original")
    assert main(["--report", str(path), "--out", str(path), "--overwrite"]) == 2
    assert path.read_bytes() == b"original"
    with pytest.raises(FileExistsError):
        render(plot_data(example()), path)
    with pytest.raises(ValueError, match="PNG"):
        render(plot_data(example()), tmp_path / "output.csv")
