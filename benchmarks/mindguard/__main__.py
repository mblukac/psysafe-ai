"""Offline smoke, request preparation and evaluation of already-cached decisions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION, Case, Turn, attach_responses, local_records, read_cases, requests_for
from .fit import brier_loss, fold_predictions, grouped_folds, log_loss
from .providers import MODEL_IDS
from .review import Fold, allocate_slots, evaluate_fold, paired_uncertainty
from .smoke import smoke_turns


def evaluate(
    turns: list[Turn],
    *,
    folds: int,
    seed: int,
    budget: float,
    replicates: int,
    synthetic: bool,
    slots: int | None = None,
) -> dict[str, Any]:
    if not math.isfinite(budget) or not 0 <= budget <= 1 or replicates < 1:
        raise ValueError("budget must be in [0,1] and bootstrap count must be positive")
    split_reports = []
    predictions: list[Fold] = []
    summary: dict[str, dict[str, Any]] = {}
    partitions = grouped_folds(turns, folds, seed)
    sizes = [len({row.dialogue_id for row in test}) for _, test in partitions]
    total_slots = math.floor(sum(sizes) * budget + 0.5) if slots is None else slots
    allocated = allocate_slots(sizes, total_slots)
    live_rate = budget if slots is None else slots / sum(sizes)
    for i, (train, test) in enumerate(partitions):
        development, heldout, winner = fold_predictions(train, test, seed=seed + i)
        predictions.append((train, development, test, heldout))
        review = evaluate_fold(train, development, test, heldout, budget=live_rate, slots=allocated[i])
        split_reports.append({"fold": i + 1, "best_single": winner, **review})
        for method, scores in heldout.items():
            totals = summary.setdefault(method, {"log_loss": 0.0, "brier": 0.0, "fixed": {}, "live": {}})
            totals["log_loss"] += log_loss(test, scores) * len(test) / len(turns)
            totals["brier"] += brier_loss(test, scores) * len(test) / len(turns)
            for policy in ("fixed", "live"):
                counts = review["methods"][method]["fixed" if policy == "fixed" else "live"]
                for metric, count in counts.items():
                    totals[policy][metric] = totals[policy].get(metric, 0) + count
    return {
        "kind": "synthetic_smoke_not_benchmark_results" if synthetic else "cached_mindguard_internal_pilot",
        "target": "self_harm vs other clinician vote",
        "model_ids_in_score_order": MODEL_IDS,
        "expected_dataset_revision": None if synthetic else DATASET_REVISION,
        "live_target_fraction": live_rate,
        "outer_folds": folds,
        "seed": seed,
        "bootstrap_replicates": replicates,
        "turns": len(turns),
        "dialogues": sum(r["dialogues"] for r in split_reports),
        "eligible_cases": sum(r["eligible_cases"] for r in split_reports),
        "fixed_slots_per_method": sum(r["queue_slots"] for r in split_reports),
        "methods": summary,
        "stack_minus_best_single": paired_uncertainty(
            predictions, budget=live_rate, slots_per_fold=allocated, replicates=replicates, seed=seed
        ),
        "folds": split_reports,
    }


def official_cases(path: Path) -> list[Case]:
    cases = read_cases(local_records(path))
    if len(cases) != 1134 or len({c.dialogue_id for c in cases}) != 67:
        raise ValueError("expected all 1,134 official prefixes and 67 unambiguous dialogue chains")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="invented inputs only; no network, key, or download")
    prepare = commands.add_parser(
        "prepare", help="emit PRIVATE request envelopes to a new local JSONL file; never send"
    )
    prepare.add_argument("--annotations", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    cached = commands.add_parser("evaluate", help="score a complete, locally cached official release")
    cached.add_argument("--annotations", required=True, type=Path)
    cached.add_argument("--responses", required=True, type=Path)
    for sub in (smoke, cached):
        workload = sub.add_mutually_exclusive_group()
        workload.add_argument("--budget", type=float, default=0.10, help="target review fraction (default 0.10)")
        workload.add_argument("--slots", type=int, help="exact total retrospective conversation reviews per method")
        sub.add_argument("--folds", type=int, default=3 if sub is smoke else 5)
        sub.add_argument("--bootstrap", type=int, default=50 if sub is smoke else 1000)
        sub.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            cases = official_cases(args.annotations)
            # Preflight all payloads before creating a sensitive local file.
            envelopes = [req for case in cases for req in requests_for(case)]
            with open(args.out, "x", encoding="utf-8", opener=_private_open) as handle:
                for env in envelopes:
                    handle.write(json.dumps(env, ensure_ascii=False) + "\n")
            print(f"Prepared {len(envelopes)} private requests locally. None sent.", file=sys.stderr)
            return 0
        synthetic = args.command == "smoke"
        rows = (
            smoke_turns()
            if synthetic
            else attach_responses(official_cases(args.annotations), local_records(args.responses))
        )
        report = evaluate(
            rows,
            folds=args.folds,
            seed=args.seed,
            budget=args.budget,
            slots=args.slots,
            replicates=args.bootstrap,
            synthetic=synthetic,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(2, f"Pilot stopped: {error}\n")
    return 0


def _private_open(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


if __name__ == "__main__":
    raise SystemExit(main())
