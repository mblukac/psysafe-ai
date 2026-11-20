"""Plot aggregate paired-work onset recall; no text, identifiers, or model calls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import Any

_MODELS = {"jev": "Jev", "gemini": "Gemini Decision 1"}
_METHODS = {
    "fixed_paired_recall": "Fixed trained cutoff",
    "rl_paired_recall": "RL stopping + paced queue",
}


@dataclass(frozen=True)
class BudgetPlotData:
    models: tuple[str, ...]
    staffing: tuple[int, ...]
    methods: tuple[str, ...]
    # Rows remain ordered by target budget, never sorted or deduplicated by actual work.
    points: dict[tuple[str, int], tuple[tuple[int, float, tuple[float, ...]], ...]]
    regime: str
    n_cutoffs: int


def plot_data(report: dict[str, Any]) -> BudgetPlotData:
    if not isinstance(report, dict) or report.get("kind") != "mindguard_onset_budget_v1":
        raise ValueError("expected aggregate mindguard_onset_budget_v1 report")
    rows = report.get("results")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("expected nonempty aggregate results")
    # FIFO has separately matched work and must not use the primary paired x.
    methods = tuple(_METHODS)
    points: dict[tuple[str, int], list[tuple[int, float, tuple[float, ...]]]] = {}
    seen = set()
    for row in rows:
        model, staff, budget = (row[k] for k in ("model", "reviewers", "budget_minutes"))
        actual = row["paired_mean_actual_minutes"]
        if model not in _MODELS or type(staff) is not int or staff < 1:
            raise ValueError("expected Jev/Gemini and positive reviewer count")
        if type(budget) is not int or budget < 0:
            raise ValueError("invalid target budget")
        if type(actual) not in (int, float) or not math.isfinite(actual) or not 0 <= actual <= budget:
            raise ValueError("invalid paired actual reviewer-minutes")
        rates = tuple(row[k] for k in methods)
        if any(type(r) not in (int, float) or not math.isfinite(r) or not 0 <= r <= 1 for r in rates):
            raise ValueError("unassessable recall must not be plotted as zero")
        key = (model, staff, budget)
        if key in seen:
            raise ValueError("duplicate model/staffing/target-budget result")
        seen.add(key)
        points.setdefault((model, staff), []).append((budget, actual, rates))
    models = tuple(m for m in _MODELS if any(k[0] == m for k in points))
    staffing = tuple(sorted({k[1] for k in points}))
    if any((m, s) not in points for m in models for s in staffing):
        raise ValueError("incomplete model/staffing facet grid")
    regime = report.get("scenario", {}).get("training_budget", "ceiling_may_underspend")
    if regime not in ("ceiling_may_underspend", "fill_if_feasible"):
        raise ValueError("unknown training budget rule")
    n_cutoffs = len(report.get("scenario", {}).get("candidate_cutoffs", ()))
    return BudgetPlotData(
        models, staffing, methods, {k: tuple(sorted(v)) for k, v in points.items()}, regime, n_cutoffs
    )


def connect_points(points: tuple[tuple[int, float, tuple[float, ...]], ...]) -> bool:
    """No backward or vertical connectors when actual work repeats or reverses."""
    return all(a[1] < b[1] for a, b in pairwise(points))


def render(data: BudgetPlotData, out: Path, *, overwrite: bool = False) -> None:
    if out.suffix.lower() != ".png":
        raise ValueError("output must be PNG")
    if out.exists() and not overwrite:
        raise FileExistsError("output exists")
    import matplotlib as mpl  # type: ignore[import-not-found]
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-not-found]
    from matplotlib.figure import Figure  # type: ignore[import-not-found]
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter  # type: ignore[import-not-found]

    styles = (
        ("#25476b", "o", 8, "none", 1.8, 1.0),
        ("#b44f12", "s", 5.2, "none", 1.8, 1.0),
    )
    with mpl.rc_context({"font.family": "DejaVu Sans"}):
        fig = Figure(figsize=(max(7, 5.2 * len(data.models)), 3.0 * len(data.staffing) + 2.1), dpi=160)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(data.staffing), len(data.models), sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.22, hspace=0.30, wspace=0.12)
        fig.suptitle("Timely recall at the same reviewer-minutes", fontsize=18, weight="bold", y=0.97)
        label = "Budget ceiling" if data.regime == "ceiling_may_underspend" else "Fill budget if feasible"
        if data.n_cutoffs:
            label += f" · {data.n_cutoffs} candidate cutoffs"
        fig.text(
            0.5,
            0.91,
            f"{label}  ·  Review starts within 10 min of first clinician endorsement",
            ha="center",
            fontsize=11,
        )
        handles = []
        connected = all(connect_points(facet) for facet in data.points.values())
        for i, staff in enumerate(data.staffing):
            for j, model in enumerate(data.models):
                ax = axes[i, j]
                points = data.points[model, staff]
                x = [p[1] for p in points]
                ax.vlines(
                    x,
                    [min(p[2]) for p in points],
                    [max(p[2]) for p in points],
                    color="#b4b9bc",
                    linewidth=1.4,
                    zorder=1,
                )
                for k, method in enumerate(data.methods):
                    color, marker, size, face, width, alpha = styles[k]
                    (line,) = ax.plot(
                        x,
                        [p[2][k] for p in points],
                        color=color,
                        marker=marker,
                        markersize=size,
                        markerfacecolor=face,
                        markeredgewidth=1.3,
                        linewidth=width,
                        alpha=alpha,
                        linestyle=("-" if k == 0 else "--") if connected else "none",
                        label=_METHODS[method],
                        clip_on=False,
                    )
                    if i == j == 0:
                        handles.append(line)
                ax.set_title(f"{_MODELS[model]} · {staff} reviewer{'s' if staff != 1 else ''}", fontsize=12)
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                ax.grid(axis="y", color="#dce2e7", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.margins(x=0.06)
                ax.spines[["right", "top"]].set_visible(False)
                ax.tick_params(labelsize=10)
                if j == 0:
                    ax.set_ylabel("Timely recall", fontsize=11)
        fig.text(0.5, 0.155, "Actual reviewer-minutes per simulated month (placement mean)", ha="center", fontsize=12)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.085),
            ncol=len(handles),
            frameon=False,
            fontsize=9,
        )
        fig.text(
            0.5,
            0.025,
            "Vertical segments compare policies at equal work, not uncertainty intervals.\n"
            "Different training budgets can lead to the same actual spend; overlapping points are retained.",
            ha="center",
            fontsize=8,
            color="#586979",
        )
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=160)
    with out.open("wb" if overwrite else "xb") as dest:
        dest.write(buffer.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.report.resolve() == args.out.resolve():
            raise ValueError("input and output must differ")
        render(plot_data(json.loads(args.report.read_bytes())), args.out, overwrite=args.overwrite)
    except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Could not plot budget report ({type(exc).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
