"""Plot the public FIFO loss audit. No annotations, credentials, or model calls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SETTINGS = (
    ("observed", 0, "No wait"),
    ("observed", 3, "3 reviewers"),
    ("observed", 2, "2 reviewers"),
    ("observed", 1, "1 reviewer"),
    ("double", 1, "1 · twice the rate"),
    ("compound", 1, "1 · twice the rate + need pulse"),
)
LABELS = (
    "No flag by deadline",
    "Flag by deadline; not seen in time¹",
    "Started before need",
    "Timely; no eligible endorsed turn",
    "Timely; stale despite available evidence",
    "Timely, supported catch",
)
COLORS = ("#526781", "#c25a39", "#efb368", "#c4b7d1", "#865b9c", "#128578")


def percentages(rows: list[dict[str, Any]]) -> list[float]:
    """Equal detector weights, not a six-detector ensemble or extra cases."""
    if len({r["model"] for r in rows}) != 6 or len(rows) != 6:
        raise ValueError("each plotted condition must contain six distinct detectors")
    den = rows[0]["positives"]
    if den <= 0 or any(not math.isclose(r["positives"], den) for r in rows):
        raise ValueError("models must see the same positive cases")
    parts = []
    for r in rows:
        split = (
            r["no_ticket"] + r["late_first_flag"],
            r["unserved_or_late_despite_flag_by_deadline"],
            r["pre_onset_pickup"],
            r["timely_no_supported_turn_available"],
            r["timely_stale"],
            r["caught"],
        )
        if min(split) < -1e-8 or not math.isclose(sum(split), den):
            raise ValueError("incomplete loss partition")
        parts.append(split)
    return [100 * sum(values) / (len(rows) * den) for values in zip(*parts, strict=True)]


def main() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-not-found]
    from matplotlib.figure import Figure  # type: ignore[import-not-found]
    from matplotlib.patches import Patch  # type: ignore[import-not-found]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    rows = json.loads(args.audit.read_text())["results"]
    fig = Figure(figsize=(11.3, 5.25))
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=0.206, right=0.986, top=0.78, bottom=0.255, wspace=0.065)
    axes = fig.subplots(ncols=2, sharey=True)
    for ax, gate, title in zip(axes, ("tuned", "liberal"), ("Training-tuned gate", "Liberal gate"), strict=True):
        for i, (scenario, staff, _) in enumerate(SETTINGS):
            x = 0.0
            for width, color in zip(
                percentages([r for r in rows if (r["gate"], r["scenario"], r["staff"]) == (gate, scenario, staff)]),
                COLORS,
                strict=True,
            ):
                ax.barh(i, width, height=0.66, left=x, color=color, edgecolor="white", linewidth=0.5)
                if color == COLORS[-1] and width >= 9:
                    ax.text(x + width / 2, i, f"{width:.0f}", ha="center", va="center", color="white", fontsize=9)
                x += width
        ax.set_xlim(0, 100)
        ax.set_ylim(len(SETTINGS) - 0.4, -0.6)
        ax.axhline(3.5, lw=0.75, color="#b9b9b9")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_yticks(range(len(SETTINGS)), [name for _, _, name in SETTINGS], fontsize=9)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.tick_params(length=0, pad=4)
        ax.spines[:].set_visible(False)
    fig.text(0.02, 0.955, "Where timely human reviews are lost", fontsize=18, weight="bold")
    fig.text(0.02, 0.90, "Per 100 need-positive replayed conversations · six-detector average · one frozen FIFO ticket")
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS, strict=True)],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncols=3,
        frameon=False,
        fontsize=9.3,
    )
    fig.text(
        0.02,
        0.035,
        "¹ Expired or started after the deadline despite a flag by then; not all are recoverable by reordering.",
        fontsize=9,
    )
    with args.output.open("xb") as f:
        fig.savefig(f, format="png", dpi=155)


if __name__ == "__main__":
    main()
