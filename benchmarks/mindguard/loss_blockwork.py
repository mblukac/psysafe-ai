"""Retrospective equal-work checks in isolated 24h arrival cohorts, versus the whole stream.

Only first K starts of each queue count as work. Each cohort drains independently
before reset: this is not a continuous staffing schedule. All eligible episodes
keep their original need and clock, including those arriving after the Kth start.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise, product
from pathlib import Path
from typing import Any

from .fit import grouped_folds
from .full_panel import MODEL_KEYS, coverage_audit, load_panel
from .full_replay import calibrated, encoded
from .loss_workmatch import (
    DISPATCH,
    GATES,
    REPS,
    ROOT,
    SEED,
    SERVICE,
    Job,
    PathPlan,
    dispatch_starts,
    hits,
    plan,
)
from .staffing import load_arrivals
from .surge_replay import schedules
from .surge_stream import assign_paths
from .update_pilot import write_report

DAY = 1440
ARMS = ("tuned/fifo/frozen", "liberal/fifo/frozen", "liberal/fifo/latest_on_max")
SCENARIOS = ("observed", "compound")
Release = tuple[float, int, float]


def evaluate(
    releases: dict[str, list[Release]], paths: dict[str, list[PathPlan]], times: Sequence[float], order: Sequence[int]
) -> tuple[int, list[int], list[float]]:
    """Same four-way K as the main factorial (not an outcome-selected cap)."""
    queues = {(g, d): dispatch_starts(releases[g], 1, d) for g, d in product(GATES, DISPATCH)}
    cap = min(map(len, queues.values()))
    scores: list[int] = []
    last: list[float] = []
    for g in GATES:
        starts = queues[g, "fifo"][:cap]
        scored, _ = hits(paths[g], times, order, starts, cap)
        scores.extend(scored[j] for j in ((0,) if g == "tuned" else (0, 3)))
        if starts:
            last.append(starts[-1][1])
    return cap, scores, last


def compare(paths: dict[str, list[PathPlan]], times: Sequence[float], order: Sequence[int]) -> dict[str, Any]:
    if not times or len(times) != len(order) or min(times) < 0 or any(a > b for a, b in pairwise(times)):
        raise ValueError("need a nonempty chronological workload")
    if [p.onset for p in paths["tuned"]] != [p.onset for p in paths["liberal"]]:
        raise ValueError("reference cohort must not depend on gate")
    positives = sum(paths["tuned"][i].onset is not None for i in order)
    releases: dict[str, list[Release]] = {}
    days: list[dict[str, list[Release]]] = [{g: [] for g in GATES} for _ in range(int(times[-1] // DAY) + 1)]
    for g in GATES:
        releases[g] = sorted(
            (times[s] + 2 * (p.flagged - 1), s, p.score)
            for s, idx in enumerate(order)
            if (p := paths[g][idx]).flagged is not None
        )
        for entry in releases[g]:
            # Assign by conversation arrival, NOT ticket time. Evidence and
            # positive/negative status stay attached even across midnight.
            days[int(times[entry[1]] // DAY)][g].append(entry)
    full_cap, full_hits, last = evaluate(releases, paths, times, order)
    caps, catches = [], [0] * len(ARMS)
    for day in days:
        cap, scored, _ = evaluate(day, paths, times, order)
        caps.append(cap)
        catches = [a + b for a, b in zip(catches, scored, strict=True)]
    return dict(
        positives=positives,
        sessions=len(order),
        full_cap=full_cap,
        full_hits=full_hits,
        day_caps=caps,
        day_hits=catches,
        full_last_days=[(t - times[0]) / DAY for t in last],
        full_arrival_fraction=[bisect_right(times, t) / len(times) for t in last],
        horizon_days=(times[-1] - times[0]) / DAY,
    )


def run(job: Job) -> list[dict[str, Any]]:
    held = calibrated(job.train, job.test)
    if (
        len(job.test) != job.fit["held_turns"]
        or len(held) != job.fit["held_paths"]
        or not math.isclose(
            sum((q - y) ** 2 for path in held.values() for _, q, y in path), job.fit["single_brier_sum"], abs_tol=1e-9
        )
        or {r.group_id for r in job.test} & {r.group_id for r in job.train}
    ):
        raise ValueError("mismatched training or held-out families")
    paths = {g: [plan(held[d], job.fit["cutoffs"][g]) for d in sorted(held)] for g in GATES}
    ref = [p.onset is not None for p in paths["tuned"]]
    rows = []
    for rep in range(REPS):
        # Both clocks are Open Up, not Azure/Burst: do not load irrelevant traces.
        for setting in schedules(job.clock, None, SEED + 8_000_009 + rep):
            if setting.name not in SCENARIOS:
                continue
            order, meta = assign_paths(ref, setting.clock, seed=SEED + 1009 * job.fold + rep, risk_burst=setting.burst)
            r = compare(paths, setting.clock.arrivals, order)
            if r["positives"] != meta["positives"]:
                raise ValueError("lost positive episodes")
            rows.append(dict(**r, model=job.model, scenario=setting.name, fold=job.fold, rep=rep, weight=len(held)))
    return rows


def summarize(rows: list[dict[str, Any]], previous: dict[str, Any], arms: list[str]) -> dict[str, Any]:
    if {(r["fold"], r["rep"]) for r in rows} != set(product(range(5), range(REPS))) or len(rows) != 5 * REPS:
        raise ValueError("incomplete or duplicate workload blocks")
    weight = sum(r["weight"] for r in rows)
    positives = sum(r["positives"] * r["weight"] for r in rows)
    if positives <= 0:
        raise ValueError("no eligible episodes")
    monthly = [sum(r["full_hits"][i] * r["weight"] for r in rows) / weight for i in range(len(ARMS))]
    mins = SERVICE * sum(r["full_cap"] * r["weight"] for r in rows) / weight
    matches = [(monthly[i], previous["catches"][arms.index(a)]) for i, a in enumerate(ARMS)]
    if not all(
        math.isclose(x, y, abs_tol=1e-8)
        for x, y in [*matches, (mins, previous["matched_minutes"]), (positives / weight, previous["positives"])]
    ):
        raise ValueError("full-stream controls did not reproduce original factorial")
    nonzero = [r for r in rows if r["full_cap"]]
    nz_weight = sum(r["weight"] for r in nonzero)

    def conditional(key: str) -> list[float | None]:
        return [sum(r[key][i] * r["weight"] for r in nonzero) / nz_weight if nz_weight else None for i in range(2)]

    def contrast(group: list[dict[str, Any]], a: int, b: int) -> float:
        return float(
            sum((r["day_hits"][a] - r["day_hits"][b]) * r["weight"] for r in group)
            / sum(r["positives"] * r["weight"] for r in group)
        )

    ranges = {}
    for name, a, b in (
        ("liberal_minus_tuned_frozen", 1, 0),
        ("liberal_refresh_minus_tuned_frozen", 2, 0),
        ("liberal_refresh_minus_liberal_frozen", 2, 1),
    ):
        values = [contrast([r for r in rows if r["fold"] != f], a, b) for f in range(5)]
        ranges[name] = [min(values), max(values)]
    return dict(
        model=rows[0]["model"],
        scenario=rows[0]["scenario"],
        positives=positives / weight,
        duration_days=rows[0]["horizon_days"],
        full_minutes=mins,
        full_catches=monthly,
        full_last_start_days_nonzero=conditional("full_last_days"),
        full_arrival_fraction_nonzero=conditional("full_arrival_fraction"),
        full_nonzero_weight=nz_weight / weight,
        day_minutes=SERVICE * sum(sum(r["day_caps"]) * r["weight"] for r in rows) / weight,
        day_catches=[sum(r["day_hits"][i] * r["weight"] for r in rows) / weight for i in range(len(ARMS))],
        day_omit_one_fold_delta=ranges,
        day_start_caps_by_fold_rep=[
            dict(fold=r["fold"], placement=r["rep"], cap=r["day_caps"])
            for r in sorted(rows, key=lambda r: (r["fold"], r["rep"]))
        ],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "private")
    p.add_argument("--reference-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    if args.jobs < 1 or args.out.exists():
        p.error("need a fresh output and positive job count")
    prior = json.loads((args.reference_dir / "ticket-updates.json").read_text())
    main_report = json.loads((args.reference_dir / "loss-workmatch.json").read_text())
    if main_report["kind"] != "mindguard_joint_minutes_v1" or any(
        main_report[k] != prior[k]
        for k in (
            "coverage",
            "dataset_revision",
            "trace_sha256",
            "models",
            "folds",
            "seed",
            "placements",
            "service_minutes",
            "turn_minutes",
            "deadline_minutes",
        )
    ):
        raise ValueError("reference files are not paired")
    folder = args.inputs
    panel = load_panel(
        folder / "mindguard.parquet",
        [
            folder / s
            for s in (
                "responses.jsonl",
                "luna-responses.jsonl",
                "clef-flash-responses.jsonl",
                "decider-flash-responses.jsonl",
            )
        ],
        expected_models=sorted(MODEL_KEYS.values()),
    )
    if coverage_audit(panel) != prior["coverage"]:
        raise ValueError("cache coverage does not match")
    clock = load_arrivals(folder / "real_interarrivals_nov.csv")
    fits = {(r["model"], r["fold"]): r for r in prior["fits"]}
    jobs = [
        Job(
            m,
            f,
            train,
            test,
            fits[m, f],
            clock,
            folder / "azure_conv_2023.csv",
            folder / "BurstGPT_1_v1.1.csv",
            SCENARIOS,
            (1,),
        )
        for m in prior["models"]
        for f, (train, test) in enumerate(grouped_folds(encoded(panel, m), 5, SEED))
    ]
    with ProcessPoolExecutor(max_workers=min(len(jobs), args.jobs)) as pool:
        batches = list(pool.map(run, jobs))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            groups[r["model"], r["scenario"]].append(r)
    controls = {(r["model"], r["scenario"]): r for r in main_report["results"] if r["staff"] == 1}
    out = dict(
        kind="mindguard_isolated_daily_equal_work_v1",
        arms=ARMS,
        coverage_gates=GATES,
        dataset_revision=prior["dataset_revision"],
        trace_sha256=prior["trace_sha256"],
        staff=1,
        service_minutes=SERVICE,
        units="Each day cap is starts per queue; every serviced start costs 20 minutes, including post-boundary work.",
        results=[summarize(rs, controls[k], main_report["arms"]) for k, rs in sorted(groups.items())],
    )
    write_report(out, args.out)


if __name__ == "__main__":
    main()
