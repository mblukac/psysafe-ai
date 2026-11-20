"""Export aggregate operating points for a public figure without case or training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .surge_replay import POLICIES

_SCENARIOS = {
    "steady",
    "steady_pulse",
    "observed",
    "risk_pulse",
    "double",
    "compound",
    "azure",
    "burst_week",
    "burst_week_steady",
}


def public_points(report: dict[str, Any]) -> dict[str, Any]:
    if report["kind"] != "mindguard_surge_v1" or report["gate"] not in {"tuned", "liberal"}:
        raise ValueError("expected the isolated-fold surge report")
    points = []
    for r in report["results"]:
        if r["model"] not in {"jev", "gemini"} or r["policy"] not in POLICIES or r["scenario"] not in _SCENARIOS:
            raise ValueError("unexpected operating point")
        point = {k: r[k] for k in ("model", "scenario", "policy")}
        point["staff"] = int(r["staff"])
        for k in ("recall", "delta_vs_fifo", "queue_loss", "mean_review_minutes", "fraction_starts_contested"):
            point[k] = round(float(r[k]), 7) if r[k] is not None else None
        for period in ("peak", "offpeak"):
            val = r["by_period"][period]["recall"]
            point[period + "_recall"] = round(float(val), 7) if val is not None else None
        points.append(point)
    return {
        "kind": "mindguard_surge_v1",
        "gate": report["gate"],
        "seed": int(report["seed"]),
        "service_minutes": float(report["service"]),
        "placements": int(report["placements"]),
        "note": "staff=0 means immediate service. Recall and differences are fractions; staffing is concurrent.",
        "results": points,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    data = public_points(json.loads(args.report.read_text()))
    # Keep a reviewable one-condition-per-line diff without disclosing case-level data.
    rows = data.pop("results")
    prefix = json.dumps(data, indent=2, allow_nan=False).rstrip()
    print(prefix[:-2] + ',\n  "results": [')
    print(",\n".join("    " + json.dumps(r, allow_nan=False) for r in rows))
    print("  ]\n}")


if __name__ == "__main__":
    main()
