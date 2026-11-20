"""Export allowlisted aggregate operating points; optionally plot from that export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIELDS = (
    "model scenario staff policy sessions positives nominated starts catches recall mean_review_minutes "
    "matched_minutes matched_recall matched_delta_vs_fifo matched_delta_vs_best_simple "
    "self_harm_recall harm_others_recall all_gaps_endorsed_recall expired unsupported_starts "
    "misses_no_alert misses_wrong_turn misses_nomination_late misses_queue_delay "
    "fraction_starts_contested oracle_recall_bounds omit_one_fold_delta_range"
).split()
LABEL = {
    "jev": "Jev",
    "gemini": "Gemini Decision 1",
    "liquid": "Liquid D1 Ultra",
    "luna": "Luna Decisions",
    "clef_flash": "Clef Flash",
    "decider_flash": "Decider Flash",
}


def export(report: dict[str, Any]) -> dict[str, Any]:
    if report["kind"] != "mindguard_full_either_risk_v1":
        raise ValueError("expected full cohort report")
    keys: set[tuple[Any, ...]] = set()
    rows = []
    for r in report["results"]:
        key = tuple(r[k] for k in ("model", "scenario", "staff", "policy"))
        if key in keys or r["model"] not in report["models"] or r["model"] not in LABEL:
            raise ValueError("duplicate operating point or unknown detector")
        keys.add(key)
        row = {k: r[k] for k in FIELDS}
        for period in ("peak", "offpeak"):
            row[period + "_recall"] = r["by_period"][period]["recall"]
        rows.append(row)
    if not rows:
        raise ValueError("no operating points")
    return {
        "kind": "mindguard_public_operating_points_v1",
        **{
            k: report[k]
            for k in (
                "models",
                "gate",
                "service",
                "deadline_minutes",
                "turn_minutes",
                "placements",
                "training_episodes",
                "seed",
                "rated_turns",
                "dialogues",
                "families",
                "trace_sha256",
                "azure_sha256",
                "burst_sha256",
            )
        },
        "counts": {
            tag: {k: report["counts"][tag][k] for k in ("turns", "dialogues")}
            for tag in ("either", "self_harm", "harm_others")
        },
        "columns": list(rows[0]),
        "rows": [list(r.values()) for r in rows],
    }


def plot(data: dict[str, Any], dest: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # type: ignore[import-not-found]
    from matplotlib.figure import Figure  # type: ignore[import-not-found]
    from matplotlib.lines import Line2D  # type: ignore[import-not-found]

    rows = [dict(zip(data["columns"], values, strict=True)) for values in data["rows"]]
    values = {(r["model"], r["scenario"], r["staff"], r["policy"]): r for r in rows}
    models = sorted(data["models"], key=list(LABEL).index)
    colors = ("#00669a", "#b45100", "#00886c", "#764f9c", "#853647", "#454b50")
    fig = Figure(figsize=(12.0, 5.1), layout="constrained")
    FigureCanvasAgg(fig)
    left, right = fig.subplots(ncols=2, sharey=True)
    for ax, scenario, title in zip(
        (left, right),
        ("observed", "compound"),
        ("Observed arrivals", "2× arrival rate + concentrated need"),  # noqa: RUF001 - multiplication sign in title
        strict=True,
    ):
        for m, color in zip(models, colors, strict=False):
            series = [values[m, scenario, s, "fifo"] for s in (1, 2, 3, 0)]
            ax.plot(
                [r["mean_review_minutes"] / 1000 for r in series],
                [100 * r["recall"] for r in series],
                color=color,
                linewidth=1.5,
            )
            for r, marker in zip(series, ("o", "s", "^", "X"), strict=True):
                ax.plot(
                    r["mean_review_minutes"] / 1000,
                    100 * r["recall"],
                    marker=marker,
                    color=color,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Actual reviewer-minutes per replay (thousands)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("Timely recall (% of observed-positive cases)")
    fig.suptitle("Which cases reach review within ten minutes?  ·  First in, first out", fontsize=15)
    handles = [Line2D([], [], color=c, label=LABEL[m]) for m, c in zip(models, colors, strict=False)]
    handles += [
        Line2D([], [], color="#434343", linestyle="none", marker=s, label=n)
        for s, n in zip(("o", "s", "^", "X"), ("1 reviewer", "2 reviewers", "3 reviewers", "No wait"), strict=True)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=9, frameon=False)
    with dest.open("xb") as f:
        fig.savefig(f, format="png", dpi=155)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--plot", action="store_true", help="Read the public operating points and generate a PNG")
    args = p.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if args.plot:
        if data["kind"] != "mindguard_public_operating_points_v1":
            p.error("plot needs a public operating-points export")
        plot(data, args.output)
    else:
        serialized = json.dumps(export(data), separators=(",", ":"), allow_nan=False)
        with args.output.open("x", encoding="utf-8") as f:
            print(serialized, file=f)


if __name__ == "__main__":
    main()
