"""Exact *across-detector* work matching at fixed FIFO dispatch on the full panel.

For every fold, arrival stream and staffing level, cap all detectors at the
smallest realized number of FIFO starts among those detectors. Evaluate each
one's first K actual starts in chronological dispatch order. This selects K
from volume, not held-out outcomes. Never drop unreviewed cases from recall's
denominator. K is retrospective and can differ by placement or comparison set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from .fit import grouped_folds
from .full_panel import PanelTurn, load_panel
from .full_replay import Job, encoded, prepare_hold
from .staffing import TRACE_SHA256, load_arrivals
from .surge_queue import Case, simulate
from .surge_replay import attach, ratio, schedules
from .surge_stream import AZURE_SHA256, BURST_SHA256, assign_paths


def prefix_counts(arms: Mapping[str, tuple[Sequence[Case], dict[int, float]]]) -> tuple[int, dict[str, int]]:
    """The nominated turn and thus its contemporaneous support are arm specific."""
    if len(arms) < 2:
        raise ValueError("require at least two detectors")
    cap = min(len(starts) for _, starts in arms.values())
    counts = {}
    for model, (cases, starts) in arms.items():
        hits = 0
        for sid, start in islice(starts.items(), cap):
            case = cases[sid]
            hits += int(case.supported and case.onset is not None and start <= case.onset + 10)
        counts[model] = hits
    return cap, counts


def cross_job(jobs: Sequence[Job]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One outer fold; use identical path multisets and clocks for all arms."""
    example = jobs[0]
    paths = {}
    audit = []
    expected_ids = sorted({r.dialogue_id for r in example.hold})
    for job in jobs:
        if job.fold != example.fold or sorted({r.dialogue_id for r in job.hold}) != expected_ids or job.model in paths:
            raise ValueError("unpaired fold")
        _, held, _, meta = prepare_hold(job)
        paths[job.model] = held
        audit.append(meta)
    firsts = [n.onset_turn for n in paths[example.model]]
    if any([n.onset_turn for n in held] != firsts for held in paths.values()):
        raise ValueError("detectors did not receive identical positive cases and onsets")
    source = [v is not None for v in firsts]
    records = []
    for rep in range(example.placements):
        for setting in schedules(example.arrivals, example.azure, example.seed + 8_000_009 + rep, example.week):
            clock = setting.clock
            order, _ = assign_paths(
                source, clock, seed=example.seed + 1009 * example.fold + rep, risk_burst=setting.burst
            )
            streams = {m: attach(h, clock.arrivals, order, clock.peak_mask) for m, h in paths.items()}
            for staff in example.staff:
                traces = {}
                uncapped = {}
                for model, cases in streams.items():
                    starts: dict[int, float] = {}
                    totals = simulate(
                        cases,
                        reviewers=staff or None,
                        service_minutes=example.service,
                        horizon_minutes=setting.horizon,
                        start_times=starts,
                    )
                    traces[model] = (cases, starts)
                    uncapped[model] = (totals["catches"], len(starts))
                cap, hits = prefix_counts(traces)
                records.append(
                    dict(
                        fold=example.fold,
                        weight=len(source),
                        placement=rep,
                        scenario=setting.name,
                        staff=staff,
                        positives=sum(source[i] for i in order),
                        sessions=len(order),
                        matched_starts=cap,
                        matched_catches=hits,
                        uncapped=uncapped,
                    )
                )
    return records, audit


def summarize_work(records: Sequence[dict[str, Any]], models: Sequence[str], service: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["scenario"], row["staff"]].append(row)
    results = []
    for (scenario, staff), members in groups.items():
        weights = sum(r["weight"] for r in members)
        positives = sum(r["weight"] * r["positives"] for r in members)
        starts = sum(r["weight"] * r["matched_starts"] for r in members)
        folds = sorted({r["fold"] for r in members})
        info = {}
        for model in models:
            hits = sum(r["weight"] * r["matched_catches"][model] for r in members)
            all_hits = sum(r["weight"] * r["uncapped"][model][0] for r in members)
            all_starts = sum(r["weight"] * r["uncapped"][model][1] for r in members)
            if hits > all_hits or starts > all_starts or (starts == all_starts and hits != all_hits):
                raise ValueError("truncated prefix cannot exceed full queue")
            info[model] = dict(
                matched_catches=hits / weights,
                matched_recall=ratio(hits, positives),
                uncapped_minutes=service * all_starts / weights,
                uncapped_recall=ratio(all_hits, positives),
            )
        # A row is a replicated queue, not an independent clinical case.
        # Drop all streams of one family-held-out fold at once as a sensitivity,
        # not a confidence interval or a model-refitting bootstrap.
        pairwise = []
        for i, left in enumerate(models):
            for right in models[i + 1 :]:
                diffs = []
                for skip in folds:
                    kept = [r for r in members if r["fold"] != skip]
                    delta = ratio(
                        sum(r["weight"] * (r["matched_catches"][left] - r["matched_catches"][right]) for r in kept),
                        sum(r["weight"] * r["positives"] for r in kept),
                    )
                    if delta is not None:
                        diffs.append(delta)
                pairwise.append(
                    dict(
                        left=left,
                        right=right,
                        delta=ratio(
                            sum(
                                r["weight"] * (r["matched_catches"][left] - r["matched_catches"][right])
                                for r in members
                            ),
                            positives,
                        ),
                        omit_one_fold_range=[min(diffs), max(diffs)] if diffs else None,
                    )
                )
        results.append(
            dict(
                scenario=scenario,
                staff=staff,
                positives=positives / weights,
                matched_minutes=service * starts / weights,
                sessions=sum(r["weight"] * r["sessions"] for r in members) / weights,
                zero_common_start_folds=[
                    fold for fold in folds if all(r["matched_starts"] == 0 for r in members if r["fold"] == fold)
                ],
                fraction_of_placements_with_zero_common_starts=sum(r["matched_starts"] == 0 for r in members)
                / len(members),
                models=info,
                pairwise=pairwise,
            )
        )
    return results


def build_jobs(
    panel: list[PanelTurn], models: Sequence[str], arrivals: list[float], args: argparse.Namespace
) -> list[list[Job]]:
    jobs: list[list[Job]] = [[] for _ in range(5)]
    for model in models:
        for fold, (train, hold) in enumerate(grouped_folds(encoded(panel, model), 5, args.seed)):
            ids = {r.dialogue_id for r in hold}
            jobs[fold].append(
                Job(
                    model,
                    fold,
                    train,
                    hold,
                    [r for r in panel if r.dialogue_id in ids],
                    arrivals,
                    args.burst_week,
                    args.azure,
                    tuple(args.staff),
                    args.service,
                    args.seed,
                    0,
                    args.placements,
                    args.gate,
                )
            )
    return jobs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--caches", nargs="+", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--burst-week", type=Path)
    p.add_argument("--azure", type=Path)
    p.add_argument("--models", nargs="+")
    p.add_argument("--gate", choices=("tuned", "liberal"), default="tuned")
    p.add_argument("--staff", type=int, nargs="+", default=(1, 2, 3, 0))
    p.add_argument("--service", type=float, default=20)
    p.add_argument("--placements", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if (
            args.out.exists()
            or min(args.jobs, args.placements) < 1
            or min(args.staff) < 0
            or len(set(args.staff)) != len(args.staff)
            or not math.isfinite(args.service)
            or args.service <= 0
        ):
            raise ValueError("invalid settings or existing output")
        panel = load_panel(args.annotations, args.caches)
        models = args.models or sorted(panel[0].scores)
        if len(set(models)) != len(models) or len(models) < 2 or not set(models) <= panel[0].scores.keys():
            raise ValueError("need at least two fully cached distinct models")
        jobs = build_jobs(panel, models, load_arrivals(args.trace), args)
        if args.jobs == 1:
            runs = list(map(cross_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=min(5, args.jobs)) as pool:
                runs = list(pool.map(cross_job, jobs))
        report = dict(
            kind="full_mindguard_exact_across_detector_fifo_work_match_v1",
            models=models,
            gate=args.gate,
            placements=args.placements,
            seed=args.seed,
            service_minutes=args.service,
            deadline_minutes=10,
            turn_minutes=2,
            source_paths=len({r.dialogue_id for r in panel}),
            trace_sha256=TRACE_SHA256,
            azure_sha256=AZURE_SHA256 if args.azure else None,
            burst_sha256=BURST_SHA256 if args.burst_week else None,
            training=[a for _, diagnostics in runs for a in diagnostics],
            results=summarize_work([r for records, _ in runs for r in records], models, args.service),
        )
        with args.out.open("x", encoding="utf-8") as f:
            print(json.dumps(report, indent=2, allow_nan=False), file=f)
    except (ValueError, KeyError, ArithmeticError, OSError, RuntimeError) as exc:
        print(
            f"Full work match failed ({type(exc).__name__}); verify configuration, private inputs and output path.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
