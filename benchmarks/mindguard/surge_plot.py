"""Two-panel staffing and dispatch figure from aggregate surge experiment JSON only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

_MODELS = ("jev", "gemini")
_SCENARIOS = ("steady", "observed", "risk_pulse", "compound")
_WEEK = ("burst_week_steady", "burst_week")
_STAFF = (1, 2, 3, 0)  # zero is the report encoding for unlimited servers, not zero reviewers
_RULES = ("risk", "learned")


@dataclass(frozen=True)
class PlotData:
    row_keys: tuple[tuple[str, str], ...]
    fifo_percent: list[list[float]]
    gain_pp: list[list[float]]


def plot_data(report: dict[str, Any]) -> PlotData:
    if not isinstance(report, dict) or report.get("kind") != "mindguard_surge_v1":
        raise ValueError("expected aggregate mindguard_surge_v1 report")
    rows = report.get("results")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("expected aggregate results")
    scenarios = _SCENARIOS + (_WEEK if any(r.get("scenario") in _WEEK for r in rows) else ())
    scores = {}
    for row in rows:
        model, scenario, staff, policy = (row[k] for k in ("model", "scenario", "staff", "policy"))
        if model not in _MODELS or scenario not in scenarios:
            continue
        if type(staff) is not int:
            raise ValueError("expected integer reviewer count")
        if (policy != "fifo" or staff not in _STAFF) and (policy not in _RULES or staff != 1):
            continue
        key = (model, scenario, staff, policy)
        if key in scores:
            raise ValueError("duplicate condition; plot one service-time and gate regime per report")
        recall = row["recall"]
        if type(recall) not in (int, float) or not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError("invalid or unassessable recall")
        scores[key] = recall
    keys = tuple((m, s) for m in _MODELS for s in scenarios)
    fifo, gains = [], []
    for m, s in keys:
        try:
            fifo.append([100 * scores[m, s, k, "fifo"] for k in _STAFF])
            gains.append([100 * (scores[m, s, 1, p] - scores[m, s, 1, "fifo"]) for p in _RULES])
        except KeyError as exc:
            raise ValueError("incomplete Jev/Gemini scenario, staffing or policy grid") from exc
    return PlotData(keys, fifo, gains)


def render(data: PlotData, out: Path, *, overwrite: bool = False) -> None:
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in ("png", "svg"):
        raise ValueError("output must be PNG or SVG")
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    import matplotlib as mpl  # type: ignore[import-not-found]
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-not-found]
    from matplotlib.colors import LinearSegmentedColormap, Normalize  # type: ignore[import-not-found]
    from matplotlib.figure import Figure  # type: ignore[import-not-found]

    # Separate absolute and signed scales. Annotations carry exact (rounded)
    # units, including minus signs: color is never the sole encoding.
    limit = max(1, math.ceil(max(abs(v) for row in data.gain_pp for v in row)))
    panels = (
        (data.fifo_percent, Normalize(0, 100), ("#f0f5f7", "#75a5b9", "#174c64"), ("1", "2", "3", "No wait")),
        (data.gain_pp, Normalize(-limit, limit), ("#ad5428", "#f7f6f1", "#24796e"), ("Risk-first", "Learned RL")),
    )
    labels = {
        "steady": "Month: steady",
        "observed": "Month: observed",
        "risk_pulse": "Month: risk pulse",
        "compound": "Month: compound",
        "burst_week_steady": "AI week: steady",
        "burst_week": "AI week: observed",
    }
    with_week = any(s in _WEEK for _, s in data.row_keys)
    top, bottom = (0.77, 0.16) if with_week else (0.73, 0.15)
    with mpl.rc_context({"font.family": "DejaVu Sans", "svg.hashsalt": "psysafe-surge-v1"}):
        fig = Figure(figsize=(10.7, 6.7 if with_week else 5.6), dpi=160)
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(
            1, 2, left=0.291, right=0.975, bottom=bottom, top=top, width_ratios=(2.0, 1.12), wspace=0.19
        )
        fig.text(
            0.035,
            0.94,
            "What reviewer capacity and dispatch each contribute",
            fontsize=18,
            weight="bold",
            color="#203347",
        )
        fig.text(
            0.035,
            0.885,
            "Same nominees across policies. A catch requires review start on a supported ticket by the first-endorsement deadline.",
            fontsize=10,
            color="#435369",
        )
        for j, (values, norm, palette, columns) in enumerate(panels):
            ax = fig.add_subplot(grid[0, j])
            cmap = LinearSegmentedColormap.from_list(f"panel{j}", palette)
            ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax.xaxis.tick_top()
            ax.set_xticks(range(len(columns)), columns)
            ax.set_yticks(range(len(data.row_keys)))
            ax.set_yticklabels([labels[s] for _, s in data.row_keys] if not j else [])
            ax.tick_params(length=0, colors="#334858", labelsize=10, pad=7)
            ax.set_title(
                ("A   FIFO recall (%) by reviewers", "B   Gain vs FIFO (pp)")[j],
                loc="left",
                pad=30,
                fontsize=11,
                weight="bold",
                color="#203347",
            )
            for row, vals in enumerate(values):
                for col, value in enumerate(vals):
                    rgb = cmap(norm(value))[:3]
                    light = sum(
                        a * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
                        for a, v in zip((0.2126, 0.7152, 0.0722), rgb, strict=True)
                    )
                    s = f"{value:.1f}" if not j else (f"{value:+.1f}" if abs(value) >= 0.05 else "0.0")
                    ax.text(
                        col,
                        row,
                        s.replace("-", "\u2212"),
                        ha="center",
                        va="center",
                        color="#101d27" if light > 0.21 else "white",
                        fontsize=10.5,
                    )
            for y in range(1, len(values)):
                model_boundary = data.row_keys[y - 1][0] != data.row_keys[y][0]
                week_boundary = data.row_keys[y][1] == "burst_week_steady"
                ax.axhline(y - 0.5, color="white", linewidth=4.0 if model_boundary else 2.0 if week_boundary else 0.8)
            for x in range(1, len(columns)):
                ax.axvline(x - 0.5, color="white", linewidth=2 if j == 0 and x == 3 else 0.8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        for y_frac, model in ((0.75, "Jev"), (0.25, "Gemini\nDecision 1")):
            fig.text(
                0.035, bottom + y_frac * (top - bottom), model, va="center", fontsize=11, weight="bold", color="#203347"
            )
        fig.text(
            0.035,
            0.084 if with_week else 0.064,
            "No wait = immediate review start.   •   Panel B: one reviewer; positive values mean more timely catches.",
            fontsize=9,
            color="#435369",
        )
        if with_week:
            fig.text(
                0.035,
                0.043,
                "Month = Open Up arrivals; AI week = monitored subset of BurstGPT requests. Each steady control retains its trace's duration and count.",
                fontsize=8.1,
                color="#435369",
            )
        image = BytesIO()
        fig.savefig(image, format=fmt, dpi=160, metadata={"Date": None} if fmt == "svg" else None)
    with out.open("wb" if overwrite else "xb") as dest:
        dest.write(image.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.out.resolve() == args.report.resolve():
            raise ValueError("report cannot also be the output")
        render(plot_data(json.loads(args.report.read_bytes())), args.out, overwrite=args.overwrite)
    except ImportError:
        print("Install matplotlib to render surge figures.", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Cannot plot surge report ({type(exc).__name__}).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
