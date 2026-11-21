"""Offline factorial comparison at identical *realized* reviewer-minutes within each paired stream."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from collections import defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION, Turn
from .fit import grouped_folds
from .full_panel import MODEL_KEYS, coverage_audit, load_panel
from .full_replay import calibrated, encoded
from .staffing import TRACE_SHA256, load_arrivals
from .staffing_queue import Trajectory
from .surge_replay import schedules
from .surge_stream import AZURE_SHA256, BURST_SHA256, assign_paths

ROOT = Path(__file__).resolve().parent
GATES = ("tuned", "liberal")
DISPATCH = ("fifo", "risk")
PRESENT = ("frozen", "latest_qualifying", "max_score", "latest_on_max")
ARMS = tuple("/".join(a) for a in product(GATES, DISPATCH, PRESENT))
DEMAND = (
    "steady",
    "steady_pulse",
    "observed",
    "risk_pulse",
    "double",
    "compound",
    "azure",
    "burst_week_steady",
    "burst_week",
)
SEED, REPS, SERVICE, DEADLINE = 2026, 8, 20, 10


@dataclass(frozen=True)
class PathPlan:
    onset: int | None
    flagged: int | None
    score: float
    turns: tuple[int, ...]
    bits: tuple[int, ...]  # outcome by visible landmark; never passed to the queue


def plan(path: Trajectory, cutoff: float) -> PathPlan:
    onset = next((t for t, _, y in path if y), None)
    ix = next((i for i, (_, q, _) in enumerate(path) if q >= cutoff), None)
    if ix is None:
        return PathPlan(onset, None, 0.0, (), ())
    frozen = latest = peak = last_peak = ix
    bits = []
    for i in range(ix, len(path)):
        if path[i][1] >= cutoff:
            latest = i
        if path[i][1] > path[peak][1]:
            peak = i
        if path[i][1] >= path[last_peak][1]:
            last_peak = i
        # Selection above sees no labels. Only now attach the evaluation outcomes.
        bits.append(sum(int(path[j][2]) << k for k, j in enumerate((frozen, latest, peak, last_peak))))
    return PathPlan(onset, path[ix][0], path[ix][1], tuple(t for t, _, _ in path[ix:]), tuple(bits))


def dispatch_starts(tickets: Sequence[tuple[float, int, float]], staff: int, mode: str) -> list[tuple[int, float]]:
    """Sorted (release, source-stream ordinal, score); only these operational fields are visible.

    Mirrors surge_queue.simulate, skipping expiry-only events: removing tickets strictly before
    a potential service start has the same result. At equality, tickets may still start.
    If there is no contest originals start in FIFO order even with risk dispatch.
    """
    if mode not in DISPATCH or staff < 0:
        raise ValueError("invalid queue")
    if not staff:  # conventional immediate-service control, not zero staffing
        return [(sid, t) for t, sid, _ in tickets]
    out: list[tuple[int, float]] = []
    free: list[float] = []
    waiting: dict[int, float] = {}
    expiries: deque[tuple[float, int]] = deque()
    n, i = len(tickets), 0
    while i < n or waiting:
        now = min(tickets[i][0] if i < n else math.inf, free[0] if free else math.inf)
        while free and free[0] <= now:
            heappop(free)
        while expiries and expiries[0][0] < now:
            _, sid = expiries.popleft()
            waiting.pop(sid, None)
        while i < n and tickets[i][0] == now:
            t, sid, q = tickets[i]
            waiting[sid] = q
            expiries.append((t + DEADLINE, sid))
            i += 1
        while waiting and len(free) < staff:
            if mode == "risk" and len(waiting) > staff - len(free):
                sid = max(waiting, key=waiting.__getitem__)  # FIFO on ties
            else:
                sid = next(iter(waiting))
            del waiting[sid]
            out.append((sid, now))
            heappush(free, now + SERVICE)
    return out


def hits(
    paths: list[PathPlan], times: Sequence[float], order: Sequence[int], starts: list[tuple[int, float]], cap: int
) -> tuple[list[int], list[int]]:
    """Return capped presentation-specific catches and uncapped catches for reproduction checks."""
    if not 0 <= cap <= len(starts):
        raise ValueError("cap exceeds realized starts")
    prefix, total = [0] * len(PRESENT), [0] * len(PRESENT)
    for rank, (sid, start) in enumerate(starts):
        p, at = paths[order[sid]], times[sid]
        if p.onset is None or not at + 2 * (p.onset - 1) <= start <= at + 2 * (p.onset - 1) + DEADLINE:
            continue
        # Compare in the exact arithmetic/convention used to release each turn, including ties.
        visible = bisect_right(p.turns, start, key=lambda t: at + 2 * (t - 1)) - 1
        if visible < 0:
            raise ValueError("review before nomination")
        mask = p.bits[visible]
        for j in range(len(PRESENT)):
            bit = (mask >> j) & 1
            total[j] += bit
            if rank < cap:
                prefix[j] += bit
    return prefix, total


@dataclass(frozen=True)
class Job:
    model: str
    fold: int
    train: list[Turn]
    test: list[Turn]
    fit: dict[str, Any]
    clock: list[float]
    azure: Path
    week: Path
    scenarios: tuple[str, ...]
    staff: tuple[int, ...]


def run(job: Job) -> list[dict[str, Any]]:
    held = calibrated(job.train, job.test)
    fit = job.fit
    if (
        len(held) != fit["held_paths"]
        or len(job.test) != fit["held_turns"]
        or not math.isclose(
            sum((q - y) ** 2 for p in held.values() for _, q, y in p), fit["single_brier_sum"], abs_tol=1e-9
        )
        or {r.group_id for r in job.test} & {r.group_id for r in job.train}
        or fit["cutoffs"]["liberal"] != 0.10
    ):
        raise ValueError("prior held-out fit mismatch")
    paths = {g: [plan(held[d], fit["cutoffs"][g]) for d in sorted(held)] for g in GATES}
    ref = [p.onset is not None for p in paths["tuned"]]
    rows = []
    for rep in range(REPS):
        for setting in schedules(job.clock, job.azure, SEED + 8_000_009 + rep, job.week):
            if setting.name not in job.scenarios:
                continue
            order, meta = assign_paths(ref, setting.clock, seed=SEED + 1009 * job.fold + rep, risk_burst=setting.burst)
            times = setting.clock.arrivals
            releases = {
                g: sorted(
                    (times[sid] + 2 * (p.flagged - 1), sid, p.score)
                    for sid, i in enumerate(order)
                    if (p := ps[i]).flagged is not None
                )
                for g, ps in paths.items()
            }
            for staff in job.staff:
                queues = [dispatch_starts(releases[g], staff, d) for g, d in product(GATES, DISPATCH)]
                # Count agreement is per stream, never obtained by matching average hours.
                cap = min(map(len, queues))
                capped, full = [], []
                for (g, _), starts in zip(product(GATES, DISPATCH), queues, strict=True):
                    first, all_ = hits(paths[g], times, order, starts, cap)
                    capped.extend(first)
                    full.extend(all_)
                rows.append(
                    dict(
                        model=job.model,
                        scenario=setting.name,
                        staff=staff,
                        fold=job.fold,
                        rep=rep,
                        weight=len(ref),
                        positives=meta["positives"],
                        cap=cap,
                        catches=capped,
                        full=full,
                        queue_starts=[len(q) for q in queues],
                    )
                )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or len({(r["fold"], r["rep"]) for r in rows}) != len(rows):
        raise ValueError("empty or duplicate block")
    folds = sorted({r["fold"] for r in rows})
    total_weight = sum(r["weight"] for r in rows)
    positives = sum(r["positives"] * r["weight"] for r in rows)
    if not positives:
        raise ValueError("empty reference positive cohort")

    def counts(rs: list[dict[str, Any]], key: str, i: int) -> int:
        return sum(r[key][i] * r["weight"] for r in rs)

    totals = [counts(rows, "catches", i) for i in range(len(ARMS))]
    groups = [[r for r in rows if r["fold"] == f] for f in folds]
    # Sufficient statistics for any paired contrast, including dropping one whole held-out fold.
    fold_stats = [
        dict(
            fold=f,
            weighted_positives=sum(r["positives"] * r["weight"] for r in members),
            weighted_catches=[counts(members, "catches", i) for i in range(len(ARMS))],
        )
        for f, members in zip(folds, groups, strict=True)
    ]
    return dict(
        **{k: rows[0][k] for k in ("model", "scenario", "staff")},
        positives=positives / total_weight,
        matched_minutes=sum(r["weight"] * r["cap"] * SERVICE for r in rows) / total_weight,
        zero_budget_folds=[f for f, grp in zip(folds, groups, strict=True) if not any(r["cap"] for r in grp)],
        catches=[t / total_weight for t in totals],
        recall=[t / positives for t in totals],
        leave_fold_sufficient=fold_stats,
        uncapped_catches=[counts(rows, "full", i) / total_weight for i in range(len(ARMS))],
        uncapped_minutes=[counts(rows, "queue_starts", i) * SERVICE / total_weight for i in range(4)],
    )


def verify(rows: list[dict[str, Any]], prior: dict[str, Any], directory: Path) -> None:
    ref = {(r["model"], r["scenario"], r["staff"], r["gate"]): r for r in prior["results"]}
    frozen = {}
    for g in GATES:
        d = json.loads((directory / f"full-panel-{g}.json").read_text())
        for values in d["rows"]:
            r = dict(zip(d["columns"], values, strict=True))
            if r["policy"] in DISPATCH:
                frozen[r["model"], r["scenario"], r["staff"], g, r["policy"]] = r
    for row in rows:
        key = tuple(row[k] for k in ("model", "scenario", "staff"))
        checks = []
        for i, (g, mode) in enumerate(product(GATES, DISPATCH)):
            expected = frozen[(*key, g, mode)]
            k = i * len(PRESENT)
            checks.extend(
                [
                    (row["positives"], expected["positives"]),
                    (row["uncapped_minutes"][i], expected["mean_review_minutes"]),
                    (row["uncapped_catches"][k], expected["catches"]),
                ]
            )
            if mode == "fifo":
                base = ref[(*key, g)]
                checks.extend(
                    (row["uncapped_catches"][k + j], base["arms"][arm]["catches"]) for j, arm in enumerate(PRESENT[:3])
                )
        if any(not math.isclose(a, b, abs_tol=1e-8) for a, b in checks):
            raise ValueError(f"could not reproduce original held-out results: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "private")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--scenarios", nargs="+", choices=DEMAND, default=DEMAND)
    parser.add_argument("--staff", type=int, nargs="+", choices=(0, 1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.out.exists()
        or args.jobs < 1
        or len(set(args.staff)) != len(args.staff)
        or len(set(args.scenarios)) != len(args.scenarios)
    ):
        parser.error("need unique settings, positive jobs, and a new output")
    p = args.inputs
    prior = json.loads((args.reference_dir / "ticket-updates.json").read_text())
    panel = load_panel(
        p / "mindguard.parquet",
        [
            p / s
            for s in (
                "responses.jsonl",
                "luna-responses.jsonl",
                "clef-flash-responses.jsonl",
                "decider-flash-responses.jsonl",
            )
        ],
        expected_models=sorted(MODEL_KEYS.values()),
    )
    contract = dict(
        kind="full_mindguard_ticket_updates_v1",
        seed=SEED,
        dataset_revision=DATASET_REVISION,
        folds=5,
        placements=REPS,
        service_minutes=SERVICE,
        deadline_minutes=DEADLINE,
        turn_minutes=2,
        trace_sha256=TRACE_SHA256,
        azure_sha256=AZURE_SHA256,
        burst_sha256=BURST_SHA256,
        coverage=coverage_audit(panel),
        models=sorted(MODEL_KEYS.values()),
    )
    if any(prior.get(k) != v for k, v in contract.items()):
        raise ValueError("mismatched published primary report")
    fits = {(f["model"], f["fold"]): f for f in prior["fits"]}
    clock = load_arrivals(p / "real_interarrivals_nov.csv")
    jobs = [
        Job(
            m,
            f,
            train,
            test,
            fits[m, f],
            clock,
            p / "azure_conv_2023.csv",
            p / "BurstGPT_1_v1.1.csv",
            tuple(args.scenarios),
            tuple(args.staff),
        )
        for m in prior["models"]
        for f, (train, test) in enumerate(grouped_folds(encoded(panel, m), 5, SEED))
    ]
    groups = defaultdict(list)
    if args.jobs == 1:
        batches = list(map(run, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(len(jobs), args.jobs)) as pool:
            batches = list(pool.map(run, jobs))
    for batch in batches:
        for r in batch:
            groups[r["model"], r["scenario"], r["staff"]].append(r)
    results = [aggregate(rs) for _, rs in sorted(groups.items())]
    verify(results, prior, args.reference_dir)
    doc = dict(
        **(contract | {"kind": "mindguard_joint_minutes_v1"}),
        arms=ARMS,
        queues=list(product(GATES, DISPATCH)),
        method="Chronological first K starts per queue, K=min across both gates and dispatchers per placement, "
        "fold, detector, scenario and staffing; retrospective cap, not a deployable staffing rule.",
        results=results,
    )
    with args.out.open("x") as f:
        json.dump(doc, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
