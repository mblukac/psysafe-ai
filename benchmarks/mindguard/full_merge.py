"""Merge independent *per-detector* full replays after an exact overlap replication.

Not valid for the across-detector work-match analysis: there a new detector
can change every existing detector's budget. Only the two listed single-model
report types are supported. Include a previously evaluated detector alongside
the new one, score both against the complete panel, and preserve that run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED = ("mindguard_full_either_risk_v1", "mindguard_full_mixed_sensitivity_v1")
MODEL_FIELDS = ("models", "training", "results", "probability")


def blocks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if report.get("kind") not in ALLOWED:
        raise ValueError("only independent per-model full replays may be merged")
    models = report["models"]
    if not models or len(set(models)) != len(models) or set(report["probability"]) != set(models):
        raise ValueError("inconsistent model tags")
    parts = {m: dict(probability=report["probability"][m]) for m in models}
    for key, fields in (("results", ("scenario", "staff", "policy")), ("training", ("fold",))):
        if {r["model"] for r in report[key]} != set(models):
            raise ValueError("missing results or training for one or more detectors")
        expected = None
        for m in models:
            rows = [r for r in report[key] if r["model"] == m]
            ids = {tuple(r[f] for f in fields) for r in rows}
            if len(ids) != len(rows) or (expected is not None and ids != expected):
                raise ValueError("duplicated or inconsistent experimental grid")
            expected = ids
            parts[m][key] = rows
    return parts


def merge_reports(previous: dict[str, Any], additional: dict[str, Any]) -> dict[str, Any]:
    # All fields not explicitly per-model must agree, including cohort, clock
    # hashes, gate, replicates, seeds, service and deadline settings.
    meta = {k: v for k, v in previous.items() if k not in MODEL_FIELDS}
    if meta != {k: v for k, v in additional.items() if k not in MODEL_FIELDS}:
        raise ValueError("incompatible study settings or coverage")
    first, second = blocks(previous), blocks(additional)
    shared = first.keys() & second.keys()
    if not shared or not second.keys() - first.keys():
        raise ValueError("require a replicated detector and a new one")
    if any(first[k] != second[k] for k in shared):
        raise ValueError("overlap does not exactly replicate every result, Brier score and training record")
    parts = first | second
    models = sorted(parts)
    return dict(
        **meta,
        models=models,
        probability={m: parts[m]["probability"] for m in models},
        training=[r for m in models for r in parts[m]["training"]],
        results=[r for m in models for r in parts[m]["results"]],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("previous", type=Path)
    p.add_argument("additional", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    report = merge_reports(json.loads(args.previous.read_text()), json.loads(args.additional.read_text()))
    with args.out.open("x", encoding="utf-8") as f:
        print(json.dumps(report, indent=2, allow_nan=False), file=f)


if __name__ == "__main__":
    main()
