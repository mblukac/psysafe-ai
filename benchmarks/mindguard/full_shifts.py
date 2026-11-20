"""Paired shift timing sensitivity: equal scheduled hours, FIFO, no network.

The Open Up file contains intervals, not an initial civil timestamp. All shifts
are on the elapsed-trace clock. The data-driven roster sees training-period
traffic only, no nominations, distress labels, or future realized arrivals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget_fit import GRID
from .data import Turn
from .fit import grouped_folds
from .full_panel import load_panel
from .full_replay import calibrated, encoded, tuned_cutoff
from .staffing import TRACE_SHA256, load_arrivals, placement
from .surge_queue import Case
from .surge_replay import attach, nominate

DAY = 1440
TRAIN_END = 14 * DAY
SCORE_FROM, SCORE_TO = 15 * DAY, 29 * DAY
END = 30 * DAY
SHIFT, SERVICE, DEADLINE = 480, 20, 10
FIXED = 9 * 60


def peak_start(training: Sequence[float]) -> int:
    """Predict one recurring contiguous eight-hour window; earliest wins ties."""
    if not training or any(not math.isfinite(t) or not 0 <= t < TRAIN_END for t in training):
        raise ValueError("need strictly pre-split training times")
    hourly = [0] * 24
    for t in training:
        hourly[int(t // 60) % 24] += 1
    return 60 * max(range(24), key=lambda h: sum(hourly[(h + i) % 24] for i in range(8)))


def training_clock(training: Sequence[float]) -> list[float]:
    """Tile earlier days to 30 days to preserve the primary 750-start tuning budget."""
    if not training or any(not 0 <= t < TRAIN_END for t in training) or list(training) != sorted(training):
        raise ValueError("training clock must consist solely of sorted pre-split traffic")
    return [t + offset for offset in range(0, END, TRAIN_END) for t in training if t + offset < END]


def next_shift(at: float, offset: int) -> tuple[float, float]:
    """Earliest eight-hour worker start >= at, plus the end of that shift."""
    begin = (math.floor((at - offset) / DAY) * DAY) + offset
    if at >= begin + SHIFT:
        begin += DAY
    return max(at, begin), begin + SHIFT


def fifo(
    cases: Sequence[Case],
    offset: int,
    *,
    begin: int = TRAIN_END,
    end: int = END,
    score_from: int = SCORE_FROM,
    score_to: int = SCORE_TO,
) -> dict[str, float]:
    """Base reviewer always on; extra worker can finish, not start, off shift.

    FIFO with a common expiry can be solved by traversing nominations in order:
    if the earliest free reviewer cannot serve the head by its expiry, no later
    allocation can serve that head. A server reservation is never made for an
    expired ticket. Labels are joined only after every start has been fixed.
    Baseline reviewer gets exact ties, identically for both arms.
    """
    if not 0 <= begin <= score_from < score_to <= end or not 0 <= offset < DAY:
        raise ValueError("invalid staffing or scoring horizon")
    if any(not begin <= c.arrival < end for c in cases):
        raise ValueError("operating window excludes a session arrival")
    tickets = sorted((t, i) for i, c in enumerate(cases) if (t := c.nomination) is not None)
    base_free = extra_free = float(begin)
    starts: dict[int, tuple[float, float]] = {}  # start, paid overtime past staffed coverage
    for t, i in tickets:
        base = max(t, base_free)
        extra, extra_end = next_shift(max(t, extra_free), offset)
        now = min(base, extra)
        if now > t + DEADLINE or now >= end:
            continue
        if base <= extra:
            base_free = base + SERVICE
            overtime = max(0.0, base_free - end)
        else:
            extra_free = extra + SERVICE
            overtime = max(0.0, extra_free - min(extra_end, end))
        starts[i] = (now, overtime)
    cohorts = [i for i, c in enumerate(cases) if score_from <= c.arrival < score_to]
    active = [v for v in starts.values() if score_from <= v[0] < score_to]
    return dict(
        sessions=len(cohorts),
        positives=sum(cases[i].onset is not None for i in cohorts),
        nominated=sum(cases[i].nomination is not None for i in cohorts),
        catches=sum(
            (onset := cases[i].onset) is not None
            and cases[i].supported
            and i in starts
            and starts[i][0] <= onset + DEADLINE
            for i in cohorts
        ),
        # Operational labor is attributed by service START, unlike the session-cohort outcome.
        starts=len(active),
        review_minutes=SERVICE * len(active),
        overtime_minutes=sum(v[1] for v in active),
    )


@dataclass(frozen=True)
class Job:
    model: str
    fold: int
    train: list[Turn]
    held: list[Turn]
    arrivals: list[float]
    primary: dict[str, Any]
    seed: int
    placements: int


def run_job(j: Job) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    earlier = [t for t in j.arrivals if t < TRAIN_END]
    offset = peak_start(earlier)
    trained = {}
    for a, b in grouped_folds(j.train, 3, j.seed + j.fold):
        trained.update(calibrated(a, b))
    causal_cut = tuned_cutoff(trained, training_clock(earlier), j.seed + 1_000_003 + 10_007 * j.fold)
    held = calibrated(j.train, j.held)
    brier = sum((q - y) ** 2 for path in held.values() for _, q, y in path)
    if (
        not math.isclose(brier, j.primary["brier_calibrated_sum"], abs_tol=1e-9)
        or j.primary["heldout_turns"] != len(j.held)
        or j.primary["heldout_paths"] != len(held)
        or j.primary["cutoff"] not in GRID
    ):
        raise ValueError("cannot reproduce primary's held-out forecast")
    cuts = {"frozen_primary": j.primary["cutoff"], "early_clock_only": causal_cut}
    by_gate = {g: [nominate(held[d], cut) for d in sorted(held)] for g, cut in cuts.items()}
    idx = [i for i, t in enumerate(j.arrivals) if TRAIN_END <= t < END]
    operational = [j.arrivals[i] for i in idx]
    results = []
    for p in range(j.placements):
        # These match the primary Open Up random assignments, including tie ordering.
        draw = placement(len(held), len(j.arrivals), j.seed + 1009 * j.fold + p)
        order = [draw[i] for i in idx]
        for g, noms in by_gate.items():
            cases = attach(noms, operational, order, [False] * len(order))
            for arm, start in (("fixed", FIXED), ("training_peak", offset)):
                results.append(
                    dict(
                        model=j.model, fold=j.fold, weight=len(held), placement=p, gate=g, arm=arm, **fifo(cases, start)
                    )
                )
    return dict(model=j.model, fold=j.fold, cutoffs=cuts), results


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["model"], r["gate"]].append(r)
    totals = []
    scheduled = (SCORE_TO - SCORE_FROM) * (1 + SHIFT / DAY)
    for (model, gate), members in groups.items():
        by_arm = {}
        for arm in ("fixed", "training_peak"):
            group = [r for r in members if r["arm"] == arm]
            denom = sum(r["weight"] for r in group)
            summary = {
                k: sum(r["weight"] * r[k] for r in group) / denom
                for k in (
                    "sessions",
                    "positives",
                    "nominated",
                    "catches",
                    "starts",
                    "review_minutes",
                    "overtime_minutes",
                )
            }
            by_arm[arm] = dict(
                **summary,
                recall=summary["catches"] / summary["positives"],
                scheduled_minutes=scheduled,
                paid_minutes=scheduled + summary["overtime_minutes"],
            )
        if any(by_arm["fixed"][k] != by_arm["training_peak"][k] for k in ("sessions", "positives", "nominated")):
            raise ValueError("unpaired arms")

        def delta(part: list[dict[str, Any]]) -> float:
            return float(
                sum(r["weight"] * r["catches"] * (1 if r["arm"] == "training_peak" else -1) for r in part)
                / sum(r["weight"] * r["positives"] for r in part if r["arm"] == "fixed")
            )

        reps = [delta([r for r in members if r["placement"] == p]) for p in sorted({r["placement"] for r in members})]
        folds = [delta([r for r in members if r["fold"] != f]) for f in sorted({r["fold"] for r in members})]
        totals.append(
            dict(
                model=model,
                gate=gate,
                arms=by_arm,
                delta_recall=delta(members),
                placement_delta_range=[min(reps), max(reps)],
                omit_fold_delta_range=[min(folds), max(folds)],
            )
        )
    return totals


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("annotations", "trace", "primary-report", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--caches", type=Path, nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args(argv)
    try:
        primary = json.loads(args.primary_report.read_text(encoding="utf-8"))
        if (
            args.out.exists()
            or args.jobs < 1
            or primary["kind"] != "mindguard_full_either_risk_v1"
            or primary["gate"] != "tuned"
            or primary["trace_sha256"] != TRACE_SHA256
            or primary["service"] != SERVICE
            or primary["deadline_minutes"] != DEADLINE
            or primary["turn_minutes"] != 2
            or not 1 <= primary["placements"]
        ):
            raise ValueError("need matching primary and new output path")
        panel = load_panel(args.annotations, args.caches, expected_models=primary["models"])
        arrivals = load_arrivals(args.trace)
        audits = {(r["model"], r["fold"]): r for r in primary["training"]}
        jobs = [
            Job(m, i, train, held, arrivals, audits[m, i], primary["seed"], primary["placements"])
            for m in primary["models"]
            for i, (train, held) in enumerate(grouped_folds(encoded(panel, m), 5, primary["seed"]))
        ]
        if args.jobs == 1:
            fits = list(map(run_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
                fits = list(pool.map(run_job, jobs))
        earlier = [t for t in arrivals if t < TRAIN_END]
        report = dict(
            kind="paired_equal_hours_shift_sensitivity_v1",
            trace_sha256=TRACE_SHA256,
            seed=primary["seed"],
            placements=primary["placements"],
            training_days=[1, 14],
            warmup_day=15,
            scored_days=[16, 29],
            follow_through_day=30,
            trace_relative_hours_not_local=True,
            fixed_start_hours=FIXED / 60,
            predicted_start_hours=peak_start(earlier) / 60,
            hours_per_shift=SHIFT / 60,
            base_reviewers=1,
            service_minutes=SERVICE,
            start_deadline_minutes=DEADLINE,
            train_arrivals=len(earlier),
            scored_arrivals=sum(SCORE_FROM <= t < SCORE_TO for t in arrivals),
            operational_arrivals=sum(TRAIN_END <= t < END for t in arrivals),
            post_month_arrivals_excluded=sum(t >= END for t in arrivals),
            source_dialogues=len({r.dialogue_id for r in panel}),
            source_rated_turns=len(panel),
            accounting="Outcomes by monitored session arrival; labor by starts in scoring period, including full service and overtime. Paid = scheduled plus overtime; review minutes are not additive to paid minutes.",
            interpretation="frozen_primary reuses gate tuned on training families with entire public clock. early_clock_only retunes using training families and only first 14 elapsed days tiled to 30.",
            training=[audit for audit, _ in fits],
            results=summarize([r for _, rows in fits for r in rows]),
        )
        with args.out.open("x", encoding="utf-8") as f:
            print(json.dumps(report, indent=2, allow_nan=False), file=f)
    except (KeyError, ArithmeticError, TypeError, OSError, ValueError, RuntimeError) as exc:
        print(
            f"Shift sensitivity failed ({type(exc).__name__}); check primary, panel, trace and output.", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
