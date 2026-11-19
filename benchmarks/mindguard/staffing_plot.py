"""Render timely recall by reviewer count from aggregate staffing replay JSON only.

Usage: python -m benchmarks.mindguard.staffing_plot --report staffing.json --out staffing.png
Requires matplotlib only for rendering. Points are pooled replay estimates, not
independent observations or confidence intervals. The native-score sensitivity
check is intentionally not among the three primary policy curves.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

_POLICIES = ("fixed", "paced", "rl")
_MODELS = {"jev": "Jev", "gemini": "Gemini Decision 1", "liquid": "Liquid D1 Ultra", "luna": "GPT-6 Luna Decisions"}
_LABELS = {"fixed": "Fixed threshold (FIFO)", "paced": "Paced queue", "rl": "RL stopping"}


@dataclass(frozen=True)
class PlotData:
    staffing: tuple[int, ...]
    models: tuple[str, ...]
    percents: dict[tuple[str, str], list[float]]
    subtitle: str


def plot_data(report: dict[str, Any]) -> PlotData:
    if (
        not isinstance(report, dict)
        or report.get("kind") != "trace_driven_counterfactual_staffing_not_production_recall"
    ):
        raise ValueError("expected an aggregate staffing replay report")
    source, scenario = report["source"], report["scenario"]
    dialogues, positives, arrivals = source["dialogues"], source["positive_dialogues"], report["trace"]["arrivals"]
    deadline = scenario["start_deadline_minutes"]
    if (
        any(type(n) is not int or n < 1 for n in (dialogues, positives, arrivals))
        or positives > dialogues
        or type(deadline) not in (int, float)
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise ValueError("invalid source counts, no assessable positives, or invalid deadline")
    levels = scenario["reviewers"]
    if not isinstance(levels, list) or not levels or any(type(n) is not int or n < 0 for n in levels):
        raise ValueError("expected nonempty integer staffing levels")
    staffing = tuple(sorted(levels))
    if len(set(staffing)) != len(staffing):
        raise ValueError("duplicate staffing level")

    points: dict[tuple[str, str, int], float] = {}
    for row in report["results"]:
        model, policy, staff, rate = (row[k] for k in ("model", "policy", "reviewers", "recall"))
        if policy in ("native_fixed", "matched_fixed"):
            continue
        if not isinstance(model, str) or not model or policy not in _POLICIES:
            raise ValueError("unknown policy or invalid model")
        if type(staff) is not int or staff not in staffing or type(rate) not in (int, float) or not 0 <= rate <= 1:
            raise ValueError("invalid staffing or unassessable recall")
        key = (model, policy, staff)
        if key in points:
            raise ValueError("duplicate result at a staffing level")
        points[key] = 100 * rate
    model_set = {m for m, _, _ in points}
    if len(model_set) not in (1, 2, 3, 4):
        raise ValueError("expected one to four assessed models")
    models = tuple(m for m in _MODELS if m in model_set) + tuple(sorted(model_set - _MODELS.keys()))
    if any((m, p, k) not in points for m in models for p in _POLICIES for k in staffing):
        raise ValueError("incomplete model/policy/staffing grid")
    curves = {(m, p): [points[m, p, k] for k in staffing] for m in models for p in _POLICIES}
    subtitle = (
        f"{dialogues:,} source dialogues ({positives:,} positive)   ·   {arrivals:,} trace arrivals"
        f"   ·   {deadline:g}-minute review-start deadline"
    )
    return PlotData(staffing, models, curves, subtitle)


def render(data: PlotData, out: Path, *, overwrite: bool = False) -> None:
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in ("png", "svg"):
        raise ValueError("output must be a PNG or SVG")
    # Aggregate transforms and tests need no plotting dependency.
    import matplotlib as mpl  # type: ignore[import-not-found]
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-not-found]
    from matplotlib.figure import Figure  # type: ignore[import-not-found]

    count = len(data.models)
    cols = count if count < 4 else 2
    rows = 2 if count == 4 else 1
    # Large hollow circle -> smaller hollow square -> small filled diamond.
    # At an exact tie all three can be seen without changing the data coordinates.
    styles = [
        ("#273b57", "-", "o", 10, "none", 2.8),
        ("#ba520b", "--", "s", 6.6, "none", 2.1),
        ("#007770", ":", "D", 3.5, "#007770", 1.8),
    ]
    with mpl.rc_context({"font.family": "DejaVu Sans", "svg.hashsalt": "psysafe-staffing-v1"}):
        fig = Figure(figsize=(max(6.8, 4.0 * cols + 0.4), 6.7 if rows == 2 else 4.1), dpi=160)
        FigureCanvasAgg(fig)
        fig.subplots_adjust(
            left=0.09 if cols < 3 else 0.065,
            right=0.976,
            top=0.83 if rows == 2 else 0.76,
            bottom=0.19 if rows == 2 else 0.27,
            wspace=0.18,
            hspace=0.42,
        )
        axes = fig.subplots(rows, cols, sharey=True, squeeze=False)
        fig.text(0.5, 0.967, "Timely recall by reviewer staffing", ha="center", va="top", fontsize=17, weight="bold")
        fig.text(0.5, 0.902 if rows == 2 else 0.865, data.subtitle, ha="center", fontsize=9, color="#435369")
        lines = []
        for i, model in enumerate(data.models):
            ax = axes[i // cols, i % cols]
            for p, (color, style, marker, size, face, width) in zip(_POLICIES, styles, strict=True):
                (line,) = ax.plot(
                    data.staffing,
                    data.percents[model, p],
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=size,
                    markerfacecolor=face,
                    markeredgewidth=1.4,
                    linewidth=width,
                    clip_on=False,
                    label=_LABELS[p],
                )
                if i == 0:
                    lines.append(line)
            ax.set_title(_MODELS.get(model, model), fontsize=11, pad=10, color="#203347")
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 25))
            ax.set_xticks(data.staffing)
            ax.margins(x=0.08)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="-", color="#d9dfe3", linewidth=0.7)
            for name, spine in ax.spines.items():
                if name in ("top", "right"):
                    spine.set_visible(False)
                else:
                    spine.set_color("#8998a3")
            ax.tick_params(colors="#435369", labelsize=9)
            if i % cols == 0:
                ax.set_ylabel("Timely recall (%)", fontsize=10)
        fig.text(0.5, 0.105 if rows == 2 else 0.15, "Concurrent reviewers", ha="center", fontsize=10)
        fig.legend(handles=lines, loc="lower center", bbox_to_anchor=(0.5, 0.016), ncol=3, frameon=False, fontsize=9)
        image = BytesIO()
        fig.savefig(image, format=fmt, dpi=160, metadata={"Date": None} if fmt == "svg" else None)
    with out.open("wb" if overwrite else "xb") as dest:
        dest.write(image.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot aggregate staffing replay results (matplotlib required)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output figure")
    args = parser.parse_args(argv)
    try:
        if args.report.resolve() == args.out.resolve():
            raise ValueError("input and output must differ")
        render(plot_data(json.loads(args.report.read_bytes())), args.out, overwrite=args.overwrite)
    except ImportError:
        print("Install the optional matplotlib package to render a figure.", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Could not plot staffing report ({type(exc).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
