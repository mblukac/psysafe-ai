"""Offline, fully paired human-review replay on every clinician-rated MindGuard turn.

The desk takes either majority-endorsed risk, not a diagnosis. Native three-way
vectors become P(self harm)+P(harm to others). One first crossing per session,
only at observed rated positions; a gap is neither a negative nor a decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from .budget_fit import GRID
from .data import Turn
from .fit import _fit, grouped_folds
from .full_panel import PanelTurn, coverage_audit, load_panel
from .staffing import TRACE_SHA256, load_arrivals, placement
from .staffing_fit import _q
from .staffing_queue import Trajectory
from .surge_learning import DispatchPolicy, train_dispatch
from .surge_mixed import stream_records
from .surge_oracle import oracle_counts
from .surge_queue import Candidate, Case, Context, Select, simulate
from .surge_replay import Nomination, aggregate, attach, nominate, ratio, schedules, training_streams
from .surge_stream import AZURE_SHA256, BURST_SHA256, assign_paths

SIMPLE = ("fifo", "risk", "aging", "loss_wait")
TAGS = ("self_harm", "harm_others")


def endorsed(votes: Sequence[str], target: str = "either") -> bool:
    if len(votes) != 3 or any(v not in ("safe", *TAGS) for v in votes) or target not in ("either", *TAGS):
        raise ValueError("invalid panel votes or endpoint")
    return (sum(v != "safe" for v in votes) if target == "either" else votes.count(target)) >= 2


def encoded(rows: Sequence[PanelTurn], model: str) -> list[Turn]:
    """Represent *binary either-risk majority* in the existing one-column fitter."""
    out = []
    for r in rows:
        p = sum(r.scores[model][t] for t in TAGS)
        if not math.isfinite(p) or not 0 <= p <= 1 + 1e-12:
            raise ValueError("either-risk option sum outside probability range")
        out.append(
            Turn(
                r.dialogue_id,
                r.turn_index,
                ("self_harm" if endorsed(r.votes) else "safe",),
                (min(1.0, p), 0, 0),
                r.group_id,
            )
        )
    return out


def calibrated(train: list[Turn], held: list[Turn]) -> dict[str, Trajectory]:
    """No transition model or contiguous-index assumption in current-turn calibration."""
    beta0, beta1 = _fit(train, (0,))
    paths: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for r in held:
        paths[r.dialogue_id].append((r.turn_index, _q((beta0, beta1), r.scores[0]), int(r.votes[0] == "self_harm")))
    return {d: tuple(sorted(path)) for d, path in paths.items()}


def tuned_cutoff(oof: dict[str, Trajectory], arrivals: list[float], seed: int) -> float:
    """Fixed FIFO threshold, two training placements; prior 750-start ceiling.

    Each cutoff's reward counts catches from its first up to 750 chronological
    starts; ties prefer fewer uncapped starts, then higher cutoff.
    """
    paths = [p for _, p in sorted(oof.items())]
    draws = [placement(len(paths), len(arrivals), seed + rep) for rep in range(2)]
    score = []
    for cut in (c for c in GRID if c >= 0):
        paths_at_cut = [nominate(path, cut) for path in paths]
        reward = starts = 0
        for draw in draws:
            hits: list[bool] = []
            simulate(
                attach(paths_at_cut, arrivals, draw, [False] * len(draw)),
                reviewers=1,
                service_minutes=20,
                start_hits=hits,
            )
            reward += sum(hits[:750])
            starts += len(hits)
        score.append((reward, -starts, cut))
    return max(score)[-1]


@dataclass(frozen=True)
class Reference:
    # These fields are used only *after* dispatch, never in a policy observation.
    firsts: tuple[int | None, int | None]
    supported: tuple[bool, bool]
    first_gap: int | None


def reference(rows: Sequence[PanelTurn], nom: Nomination) -> Reference:
    observed = {r.turn_index: r.votes for r in rows}
    if len(observed) != len(rows) or not observed:
        raise ValueError("empty path or duplicated position")
    gap = next((t for t in range(1, max(observed) + 1) if t not in observed), None)

    def first(tag: str) -> int | None:
        return min((t for t, votes in observed.items() if endorsed(votes, tag)), default=None)

    def support(tag: str) -> bool:
        return nom.turn is not None and endorsed(observed[nom.turn], tag)

    return Reference((first(TAGS[0]), first(TAGS[1])), (support(TAGS[0]), support(TAGS[1])), gap)


def endpoint_counts(
    cases: Sequence[Case], refs: Sequence[Reference], order: Sequence[int], starts: dict[int, float]
) -> Counter[str]:
    """Evaluate both subtypes on *one desk*; bound missing labels at fixed decisions."""
    counts: Counter[str] = Counter()
    for sid, (case, i) in enumerate(zip(cases, order, strict=True)):
        ref = refs[i]
        start = starts.get(sid, math.inf)
        for tag, turn, ok in zip(TAGS, ref.firsts, ref.supported, strict=True):
            if turn is not None:
                counts[tag + "_positives"] += 1
                counts[tag + "_catches"] += int(ok and start <= case.arrival + 2 * (turn - 1) + 10)
        gap_time = math.inf if ref.first_gap is None else case.arrival + 2 * (ref.first_gap - 1)
        early = min(case.onset if case.onset is not None else math.inf, gap_time)
        if math.isfinite(early):
            counts["possible_positives"] += 1
            counts["all_gaps_endorsed_catches"] += int(case.supported and start <= early + 10)
    return counts


def matched_work(cases: Sequence[Case], traces: dict[str, dict[int, float]]) -> tuple[int, dict[str, int]]:
    """Evaluate each dispatcher's first K actual starts in dispatch order.

    K is the minimum realized starts across arms on that very same arrival
    stream, not a number selected from outcomes. Under nonanticipating rules,
    stopping after start K leaves that prefix of decisions unchanged. This
    retrospective comparison is within classifier/gate/team; models with
    different gates should not be described as work matched by this analysis.
    """
    cap = min(map(len, traces.values()))
    hits = {}
    for rule, starts in traces.items():
        count = 0
        for sid, start in list(starts.items())[:cap]:
            case = cases[sid]
            count += int(case.supported and case.onset is not None and start <= case.onset + 10)
        hits[rule] = count
    return cap, hits


def loss_rule(training: Sequence[Nomination], service: float, staff: int) -> Select:
    """Estimate lost timely credit from one slot interval, using training labels only.

    Histogram of remaining onset slack at nomination conditioned on score band
    and elapsed turn band; unsupported nominees have no credit at any age.
    Shrink each cell toward the score-weighted loss implied by uniform slack.
    The interval service/staff approximates the next slot, not its actual time.
    """
    cells: dict[tuple[int, bool], list[float]] = defaultdict(list)
    for p in training:
        if p.turn is not None and p.score is not None:
            slack = 10 - 2 * (p.turn - p.onset_turn) if p.supported and p.onset_turn is not None else -1
            cells[min(3, int(4 * p.score)), p.turn >= 8].append(slack)
    interval = service / staff

    def value(c: Candidate) -> float:
        slacks = cells.get((min(3, int(4 * c.q)), c.turn >= 8), [])
        prior = c.q * max(0.0, min(10.0, c.age + interval) - c.age) / 10
        return (sum(c.age <= slack < c.age + interval for slack in slacks) + 4 * prior) / (len(slacks) + 4)

    def choose(candidates: Sequence[Candidate], context: Context) -> int:
        return max(range(len(candidates)), key=lambda i: value(candidates[i]))

    return choose


def dispatch(
    cases: Sequence[Case],
    staff: int,
    name: str,
    service: float,
    horizon: float,
    learner: DispatchPolicy | None,
    loss: Select | None,
    actions: Counter[str],
    starts: dict[int, float],
) -> dict[str, Any]:
    select = (
        learner.selector(actions)
        if name == "learned" and learner is not None
        else loss if name == "loss_wait" else None
    )
    return simulate(
        cases,
        reviewers=staff or None,
        policy="learned" if select is not None else name,
        select=select,
        service_minutes=service,
        horizon_minutes=horizon,
        start_times=starts,
    )


@dataclass(frozen=True)
class Job:
    model: str
    fold: int
    train: list[Turn]
    hold: list[Turn]
    raw: list[PanelTurn]
    arrivals: list[float]
    week: Path | None
    azure: Path | None
    staff: tuple[int, ...]
    service: float
    seed: int
    episodes: int
    placements: int
    gate: str


def prepare_hold(job: Job) -> tuple[list[Nomination], list[Nomination], list[Reference], dict[str, Any]]:
    if {r.group_id for r in job.train} & {r.group_id for r in job.hold}:
        raise ValueError("shared family across fold")
    inner = {}
    for part, held in grouped_folds(job.train, 3, job.seed + job.fold):
        inner.update(calibrated(part, held))
    cut = tuned_cutoff(inner, job.arrivals, job.seed + 1_000_003 + 10_007 * job.fold) if job.gate == "tuned" else 0.10
    fitted = calibrated(job.train, job.hold)
    train = [nominate(p, cut) for _, p in sorted(inner.items())]
    hold = [nominate(p, cut) for _, p in sorted(fitted.items())]
    grouped = {d: list(g) for d, g in groupby(sorted(job.raw, key=lambda r: r.dialogue_id), lambda r: r.dialogue_id)}
    refs = [reference(grouped[d], nom) for d, nom in zip(sorted(fitted), hold, strict=True)]
    audit = dict(
        fold=job.fold,
        model=job.model,
        cutoff=cut,
        heldout_paths=len(hold),
        positive_heldout_paths=sum(p.onset_turn is not None for p in hold),
        heldout_turns=len(job.hold),
        brier_native_sum=sum((r.scores[0] - int(r.votes[0] == "self_harm")) ** 2 for r in job.hold),
        brier_calibrated_sum=sum((q - y) ** 2 for path in fitted.values() for _, q, y in path),
    )
    return train, hold, refs, audit


def run_mixed(folds: Sequence[Job]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All held-out paths in one waiting room; no dispatch fit on mixed data.

    Every path's gate excluded its own family, but its score fit can use
    labels of other waiting-room members. A mixing sensitivity, not an
    independent test of a single deployable fitted pipeline.
    """
    example = folds[0]
    if example.service != 20 or set(example.staff) != {0, 1, 2, 3}:
        raise ValueError("mixed sensitivity requires 20-minute service and staff 1/2/3/immediate")
    all_paths: dict[str, Nomination] = {}
    audit = []
    for job in folds:
        _, held, _, meta = prepare_hold(job)
        ids = sorted({r.dialogue_id for r in job.hold})
        if set(ids) & all_paths.keys() or job.model != example.model:
            raise ValueError("overlapping holdout paths or different score models")
        all_paths.update(zip(ids, held, strict=True))
        audit.append(meta)
    frozen = [p for _, p in sorted(all_paths.items())]
    rows = []
    for rep in range(example.placements):
        for setting in schedules(example.arrivals, example.azure, example.seed + 8_000_009 + rep, example.week):
            rows.extend(stream_records(example.model, frozen, setting, seed=example.seed, rep=rep))
    return rows, audit


def run_job(job: Job) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train, hold, refs, audit = prepare_hold(job)
    seed = job.seed + 1009 * job.fold
    learners = {}
    losses = {}
    best = {}
    for s in (x for x in job.staff if x):
        learners[s] = train_dispatch(
            training_streams(train, seed + 70_001, job.episodes),
            reviewers=s,
            seed=job.seed + 10_007 * s + job.fold,
            service_minutes=job.service,
        )
        losses[s] = loss_rule(train, job.service, s)
        scores = dict.fromkeys(SIMPLE, 0)
        for sample in training_streams(train, seed + 741_007, 6):
            for name in SIMPLE:
                scores[name] += dispatch(sample, s, name, job.service, 43200, None, losses[s], Counter(), {})["catches"]
        best[s] = max(SIMPLE, key=lambda name: scores[name])  # FIFO first on ties
    audit.update(learners={s: learner.diagnostics for s, learner in learners.items()}, development_selected=best)
    source = [p.onset_turn is not None for p in hold]
    rows = []
    for rep in range(job.placements):
        for setting in schedules(job.arrivals, job.azure, job.seed + 8_000_009 + rep, job.week):
            clock = setting.clock
            order, _ = assign_paths(source, clock, seed=seed + rep, risk_burst=setting.burst)
            cases = attach(hold, clock.arrivals, order, clock.peak_mask)
            for s in job.staff:
                lower, upper = oracle_counts(cases, s, job.service) if s else (0, 0)
                group: dict[str, dict[str, Any]] = {}
                traces = {}
                for p in (*SIMPLE, "learned") if s else ("fifo",):
                    starts: dict[int, float] = {}
                    actions: Counter[str] = Counter()
                    r = dispatch(
                        cases, s, p, job.service, setting.horizon, learners.get(s), losses.get(s), actions, starts
                    )
                    traces[p] = starts
                    other = endpoint_counts(cases, refs, order, starts)
                    group[p] = dict(
                        model=job.model,
                        fold=job.fold,
                        weight=len(hold),
                        placement=rep,
                        scenario=setting.name,
                        staff=s,
                        policy=p,
                        actions=actions,
                        metrics=r,
                        horizon=setting.horizon,
                        other=other,
                    )
                    if s and r["catches"] > upper:
                        raise ValueError("feasible catches above oracle upper bound")
                    lower = max(lower, r["catches"])
                if s:
                    group["best_simple"] = dict(group[best[s]], policy="best_simple")
                cap, hits = matched_work(cases, traces)
                hits["best_simple"] = hits[best[s]] if s else hits["fifo"]
                for name, row in group.items():
                    row["other"] = dict(
                        row["other"],
                        oracle_lower=lower,
                        oracle_upper=upper if s else lower,
                        matched_starts=cap,
                        matched_catches=hits[name],
                        matched_fifo=hits["fifo"],
                        matched_best_simple=hits["best_simple"],
                    )
                    rows.append(row)
    return rows, audit


def summarize(records: list[dict[str, Any]], service: float) -> list[dict[str, Any]]:
    values = aggregate(records, service)
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r["model"], r["scenario"], r["staff"], r["policy"]].append(r)
    for v in values:
        g = groups[v["model"], v["scenario"], v["staff"], v["policy"]]
        denom = sum(r["weight"] for r in g)
        for tag in (*TAGS,):
            for metric in ("positives", "catches"):
                key = tag + "_" + metric
                v[key] = sum(r["weight"] * r["other"].get(key, 0) for r in g) / denom
            v[tag + "_recall"] = ratio(v[tag + "_catches"], v[tag + "_positives"])
        for key in (
            "possible_positives",
            "all_gaps_endorsed_catches",
            "oracle_lower",
            "oracle_upper",
            "matched_starts",
            "matched_catches",
            "matched_fifo",
            "matched_best_simple",
        ):
            v[key] = sum(r["weight"] * r["other"].get(key, 0) for r in g) / denom
        v["all_gaps_endorsed_recall"] = ratio(v["all_gaps_endorsed_catches"], v["possible_positives"])
        v["oracle_recall_bounds"] = [ratio(v[k], v["positives"]) for k in ("oracle_lower", "oracle_upper")]
        v["matched_minutes"] = service * v["matched_starts"]
        v["matched_recall"] = ratio(v["matched_catches"], v["positives"])
        for rival in ("fifo", "best_simple"):
            v["matched_delta_vs_" + rival] = ratio(v["matched_catches"] - v["matched_" + rival], v["positives"])
    return values


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--caches", type=Path, nargs="+", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--burst-week", type=Path)
    p.add_argument("--azure", type=Path)
    p.add_argument("--models", nargs="+", help="Subset of completely cached model keys; default all supplied")
    p.add_argument("--gate", choices=("tuned", "liberal"), default="tuned")
    p.add_argument("--staff", type=int, nargs="+", default=(1, 2, 3, 0), help="Zero denotes immediate service")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--service", type=float, default=20)
    p.add_argument("--placements", type=int, default=8)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument(
        "--mixed-only",
        action="store_true",
        help="Sensitivity: pool all family-held-out tickets in a single desk; no learned dispatch",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if (
            args.out.exists()
            or min(args.placements, args.episodes, args.jobs) < 1
            or args.episodes > 32
            or len(set(args.staff)) != len(args.staff)
            or min(args.staff) < 0
            or not math.isfinite(args.service)
            or args.service <= 0
        ):
            raise ValueError("invalid parameters or output already exists")
        panel = load_panel(args.annotations, args.caches)
        arrivals = load_arrivals(args.trace)
        models = args.models or sorted(panel[0].scores)
        if len(set(models)) != len(models) or not set(models) <= panel[0].scores.keys():
            raise ValueError("missing or duplicated model")
        jobs = []
        for model in models:
            for fold, (train, hold) in enumerate(grouped_folds(encoded(panel, model), 5, args.seed)):
                ids = {r.dialogue_id for r in hold}
                jobs.append(
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
                        args.episodes,
                        args.placements,
                        args.gate,
                    )
                )
        if args.mixed_only:
            tasks = [[j for j in jobs if j.model == m] for m in models]
            if args.jobs == 1:
                mixed = list(map(run_mixed, tasks))
            else:
                with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
                    mixed = list(pool.map(run_mixed, tasks))
            training = [a for _, audits in mixed for a in audits]
            results = aggregate([r for rows, _ in mixed for r in rows], args.service)
        else:
            if args.jobs == 1:
                fits = list(map(run_job, jobs))
            else:
                with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
                    fits = list(pool.map(run_job, jobs))
            training = [audit for _, audit in fits]
            results = summarize([r for records, _ in fits for r in records], args.service)
        report = dict(
            kind="mindguard_full_mixed_sensitivity_v1" if args.mixed_only else "mindguard_full_either_risk_v1",
            cohort="all observed rated turns, including incomplete paths",
            gate=args.gate,
            service=args.service,
            deadline_minutes=10,
            turn_minutes=2,
            seed=args.seed,
            placements=args.placements,
            training_episodes=0 if args.mixed_only else args.episodes,
            models=models,
            coverage=coverage_audit(panel),
            rated_turns=len(panel),
            families=len({r.group_id for r in panel}),
            dialogues=len({r.dialogue_id for r in panel}),
            counts={
                k: dict(
                    turns=sum(endorsed(r.votes, k) for r in panel),
                    dialogues=len({r.dialogue_id for r in panel if endorsed(r.votes, k)}),
                )
                for k in ("either", *TAGS)
            },
            trace_sha256=TRACE_SHA256,
            burst_sha256=BURST_SHA256 if args.burst_week else None,
            azure_sha256=AZURE_SHA256 if args.azure else None,
            training=training,
            probability={
                m: {
                    kind: sum(r["brier_" + kind + "_sum"] for r in training if r["model"] == m) / len(panel)
                    for kind in ("native", "calibrated")
                }
                for m in models
            },
            results=results,
        )
        output = json.dumps(report, indent=2, allow_nan=False)
        with args.out.open("x", encoding="utf-8") as f:
            print(output, file=f)
    except (ValueError, OSError, KeyError, RuntimeError, ArithmeticError) as e:
        print(
            f"Full replay failed ({type(e).__name__}); check pinned inputs, available models and output path.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
