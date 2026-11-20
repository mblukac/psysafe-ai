"""Pool cross-fitted nominations into one queue to stress-test unlearned dispatch.

Each source path gets its family-held-out score and gate. These scores come
from different fits and other queue members can have informed those fits:
this is a mixing sensitivity, not independent evaluation of a deployed model.
No dispatcher is learned or tuned on the mixed stream.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from .__main__ import official_cases
from .budget_fit import tune_folds
from .data import Turn, attach_responses, complete_dialogues, local_records
from .staffing import TRACE_SHA256, load_arrivals
from .surge_queue import simulate
from .surge_replay import Nomination, Setting, aggregate, attach, nominate, schedules
from .surge_stream import AZURE_SHA256, BURST_SHA256, assign_paths


def nominations(
    model: str, *, rows: list[Turn], arrivals: list[float], seed: int
) -> tuple[str, list[Nomination], list[float]]:
    # Same grouped train/test partitions, inner OOF placements, grid, objective
    # and cutoff ties as surge_replay.fold_run(..., gate="tuned").
    paths, audit = tune_folds(
        rows, arrivals, model=model, staff=1, budgets=(750,), seed=seed, inner_placements=2, require_fill=False
    )
    frozen = [nominate(path, cutoff) for path, cutoff in paths[750]["fixed"]]
    cutoffs = [row["cutoff"] for row in audit if row["policy"] == "fixed"]
    return model, frozen, cutoffs


def stream_records(
    model: str, frozen: Sequence[Nomination], setting: Setting, *, seed: int, rep: int
) -> list[dict[str, Any]]:
    clock = setting.clock
    order, _ = assign_paths(
        [p.onset_turn is not None for p in frozen], clock, seed=seed + rep, risk_burst=setting.burst
    )
    sessions = attach(frozen, clock.arrivals, order, clock.peak_mask)
    return [
        {
            "model": model,
            "fold": 0,
            "weight": 1,
            "placement": rep,
            "scenario": setting.name,
            "staff": staff,
            "policy": policy,
            "actions": {},
            "horizon": setting.horizon,
            "metrics": simulate(
                sessions,
                reviewers=staff or None,
                service_minutes=20.0,
                deadline_minutes=10.0,
                horizon_minutes=setting.horizon,
                policy=policy,
            ),
        }
        for staff in (1, 2, 3, 0)
        for policy in (("fifo", "risk", "aging") if staff else ("fifo",))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "responses", "trace"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--burst-week", type=Path)
    parser.add_argument("--azure", type=Path)
    parser.add_argument("--out", type=Path, help="create a new aggregate JSON file (default stdout)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--placements", type=int, default=8)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    try:
        if args.placements < 1:
            raise ValueError("need positive placements")
        cases = official_cases(args.annotations)
        complete = complete_dialogues(cases)
        rows = [r for r in attach_responses(cases, local_records(args.responses)) if r.dialogue_id in complete]
        arrivals = load_arrivals(args.trace)
        fit = partial(nominations, rows=rows, arrivals=arrivals, seed=args.seed)
        models = ("jev", "gemini")
        if args.jobs == 1:
            fitted = list(map(fit, models))
        else:
            with ProcessPoolExecutor(max_workers=2) as pool:
                fitted = list(pool.map(fit, models))
        records = []
        for rep in range(args.placements):
            for setting in schedules(arrivals, args.azure, args.seed + 8_000_009 + rep, args.burst_week):
                for model, frozen, _ in fitted:
                    records.extend(stream_records(model, frozen, setting, seed=args.seed, rep=rep))
        result = {
            "kind": "mixed_queue_cross_fitted_nominations_sensitivity",
            "limitations": "Cross-fold calibrated scores compete; neither a single independent fitted pipeline nor a learned dispatch evaluation.",
            "seed": args.seed,
            "service": 20,
            "deadline": 10,
            "turn_minutes": 2,
            "gate": "tuned",
            "placements": args.placements,
            "source_dialogues": len(complete),
            "positive_dialogues": sum(p.onset_turn is not None for p in fitted[0][1]),
            "turns": len(rows),
            "trace_sha256": TRACE_SHA256,
            "burst_week_sha256": BURST_SHA256 if args.burst_week else None,
            "azure_sha256": AZURE_SHA256 if args.azure else None,
            "frozen": {
                m: {
                    "fold_cutoffs": cuts,
                    "source_nominations": sum(n.turn is not None for n in nom),
                    "distinct_nominated_scores": len({n.score for n in nom if n.turn is not None}),
                }
                for m, nom, cuts in fitted
            },
            "results": aggregate(records, service=20),
        }
        output = json.dumps(result, indent=2, allow_nan=False)
        if args.out is None:
            print(output)
        else:
            # Do not silently overwrite previous experimental evidence.
            with args.out.open("x", encoding="utf-8") as f:
                print(output, file=f)
    except (ValueError, KeyError, OSError, ArithmeticError, RuntimeError) as exc:
        print(f"Mixed replay failed ({type(exc).__name__}); check pinned inputs and output path.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
