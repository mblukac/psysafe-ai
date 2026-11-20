from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.budget_plot import connect_points, main, plot_data, render


def report() -> dict[str, Any]:
    return {
        "kind": "mindguard_onset_budget_v1",
        "results": [
            {
                "model": m,
                "reviewers": s,
                "budget_minutes": b,
                "paired_mean_actual_minutes": b / 2,
                "fixed_paired_recall": 0.25,
                "rl_paired_recall": 0.5,
                "rl_fifo_paired_recall": 0.4,
            }
            for m in ("gemini", "jev")
            for s in (6, 1)
            for b in (200, 100)
        ],
    }


def test_uses_actual_work_in_target_order_and_keeps_duplicates() -> None:
    r = report()
    d = plot_data(r)
    assert d.models == ("jev", "gemini")
    assert d.staffing == (1, 6)
    assert [p[:2] for p in d.points["jev", 1]] == [(100, 50), (200, 100)]
    assert connect_points(d.points["jev", 1])
    r["results"][0]["paired_mean_actual_minutes"] = 50
    assert not connect_points(plot_data(r).points["gemini", 6])
    assert len(plot_data(r).points["gemini", 6]) == 2
    r["results"][0]["paired_mean_actual_minutes"] = 40
    assert not connect_points(plot_data(r).points["gemini", 6])


def test_optional_fifo_and_missing_recall() -> None:
    r = report()
    for row in r["results"]:
        del row["rl_fifo_paired_recall"]
    assert len(plot_data(r).methods) == 2
    r["results"][0]["rl_paired_recall"] = None
    with pytest.raises(ValueError, match="unassessable"):
        plot_data(r)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("paired_mean_actual_minutes", float("nan")),
        ("reviewers", True),
        ("budget_minutes", -1),
        ("fixed_paired_recall", 1.1),
    ],
)
def test_invalid_points(field: str, value: Any) -> None:
    r = report()
    r["results"][0][field] = value
    with pytest.raises(ValueError, match=r"invalid|expected|unassessable"):
        plot_data(r)


def test_duplicate_budget_rejected_and_fifo_ignored() -> None:
    r = report()
    r["results"].append(r["results"][0])
    with pytest.raises(ValueError, match="duplicate"):
        plot_data(r)
    r = report()
    del r["results"][0]["rl_fifo_paired_recall"]
    assert len(plot_data(r).methods) == 2


def test_output_protection_before_optional_dependency(tmp_path: Path) -> None:
    out = tmp_path / "figure.png"
    out.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        render(plot_data(report()), out)
    assert out.read_bytes() == b"existing"
    with pytest.raises(ValueError, match="PNG"):
        render(plot_data(report()), tmp_path / "figure.svg")
    assert main(["--report", str(out), "--out", str(out), "--overwrite"]) == 2


def test_renders_png_when_matplotlib_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "figure.png"
    render(plot_data(report()), out)
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
