"""Offline surge experiment. Keep nomination fixed and change only reviewer dispatch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from random import Random
from typing import Any

from .__main__ import official_cases
from .budget_fit import _split_scores, _tune
from .data import Turn, attach_responses, complete_dialogues, local_records
from .fit import SINGLES, grouped_folds
from .staffing import TRACE_SHA256, load_arrivals, placement
from .staffing_queue import Trajectory
from .surge_learning import train_dispatch
from .surge_queue import Case, simulate
from .surge_stream import (
    AZURE_SHA256,
    BURST_SHA256,
    Clock,
    assign_paths,
    azure_episode,
    burst_week,
    load_azure_native,
    load_burst_week,
    make_clock,
)

POLICIES = ("fifo", "risk", "aging", "learned")
COUNT_FIELDS = (
    "sessions",
    "positives",
    "nominated",
    "starts",
    "catches",
    "expired",
    "unsupported_starts",
    "misses_no_alert",
    "misses_wrong_turn",
    "misses_nomination_late",
    "misses_queue_delay",
    "misses_queue_expired",
    "misses_queue_started_late",
)


@dataclass(frozen=True)
class Nomination:
    turn: int | None
    score: float | None
    onset_turn: int | None
    supported: bool


def nominate(path: Trajectory, cutoff: float) -> Nomination:
    # Lock the first qualifying turn using the present score only. Inspect votes
    # solely to annotate the already fixed decision and to construct stress mixes.
    chosen = next((entry for entry in path if entry[1] >= cutoff), None)
    onset = next((t for t, _, y in path if y), None)
    if chosen is None:
        return Nomination(None, None, onset, False)
    t, s, endorsed = chosen
    return Nomination(t, s, onset, bool(endorsed))


def attach(
    paths: Sequence[Nomination], times: Sequence[float], indices: Sequence[int], peak: Sequence[bool]
) -> list[Case]:
    result = []
    for at, i, flag in zip(times, indices, peak, strict=True):
        p = paths[i]
        result.append(
            Case(
                at,
                None if p.turn is None else at + 2 * (p.turn - 1),
                p.score,
                p.turn,
                None if p.onset_turn is None else at + 2 * (p.onset_turn - 1),
                p.supported,
                flag,
            )
        )
    return result


def training_streams(paths: Sequence[Nomination], seed: int, n_episodes: int) -> Iterable[list[Case]]:
    # Abstract monitored-session rates / hour, independent of test clocks and labels.
    for ep in range(n_episodes):
        rng = Random(seed + 31_003 * ep)  # noqa: S311 - reproducible simulation
        n = 2000
        rate = (13, 26, 52)[ep % 3] / 60
        times = list(accumulate(rng.expovariate(rate) for _ in range(n)))
        yield attach(paths, times, placement(len(paths), n, seed + 500_003 + ep), [False] * n)


@dataclass(frozen=True)
class Setting:
    name: str
    clock: Clock
    horizon: float
    burst: bool


def schedules(arrivals: list[float], azure: Path | None, seed: int, week: Path | None = None) -> list[Setting]:
    clocks = {
        "steady": make_clock(arrivals, steady=True),
        "observed": make_clock(arrivals),
        "double": make_clock(arrivals, load=2.0),
    }
    settings = [
        Setting(name, clocks[clock], arrivals[-1] / load, burst)
        for name, clock, load, burst in (
            ("steady", "steady", 1, False),
            ("steady_pulse", "steady", 1, True),
            ("observed", "observed", 1, False),
            ("risk_pulse", "observed", 1, True),
            ("double", "double", 2, False),
            ("compound", "double", 2, True),
        )
    ]
    if azure is not None:
        ref = load_azure_native(azure)
        c = make_clock(azure_episode(azure, n=60, seed=seed), peak_reference=ref, bin_minutes=5)
        settings.append(Setting("azure", c, ref[-1], False))
    if week is not None:
        native = load_burst_week(week)
        monitored = burst_week(week, n=2184, seed=seed)
        for name, steady in (("burst_week", False), ("burst_week_steady", True)):
            c = make_clock(monitored, peak_reference=native, steady=steady)
            settings.append(Setting(name, c, 7 * 24 * 60, False))
    return settings


@dataclass
class Job:
    train: list[Turn]
    test: list[Turn]
    arrivals: list[float]
    model: str
    fold: int
    seed: int
    placements: int
    episodes: int
    service: float
    gate: str
    azure: Path | None
    week: Path | None
    staff: tuple[int, ...]


def fold_run(job: Job) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idx = SINGLES.index(job.model)
    inner = {}
    for part, held in grouped_folds(job.train, 3, job.seed + job.fold):
        inner.update(_split_scores(part, held, idx))
    if job.gate == "tuned":
        # Exact fixed threshold from the previous study's 1-reviewer / 15k training
        # objective, including its 20-minute service; never refit the gate during stress.
        cuts, _ = _tune(inner, job.arrivals, 1, (750,), job.seed + 1_000_003 + job.fold * 10_007, 2, require_fill=False)
        cut = cuts[750]["fixed"]
    else:
        cut = 0.10
    training = [nominate(value[0], cut) for _, value in sorted(inner.items())]
    heldout = [nominate(value[0], cut) for _, value in sorted(_split_scores(job.train, job.test, idx).items())]
    finite = [s for s in job.staff if s]
    learned = {
        s: train_dispatch(
            training_streams(training, job.seed + 70_001 + 1009 * job.fold, job.episodes),
            reviewers=s,
            seed=job.seed + 10007 * s + job.fold,
            service_minutes=job.service,
        )
        for s in finite
    }
    source = [p.onset_turn is not None for p in heldout]
    rows = []
    for rep in range(job.placements):
        for setting in schedules(job.arrivals, job.azure, job.seed + 8_000_009 + rep, job.week):
            clock = setting.clock
            order, _ = assign_paths(source, clock, seed=job.seed + 1009 * job.fold + rep, risk_burst=setting.burst)
            cases = attach(heldout, clock.arrivals, order, clock.peak_mask)
            for s in job.staff:
                for name in POLICIES if s else ("fifo",):
                    actions: Counter[str] = Counter()
                    r = simulate(
                        cases,
                        reviewers=s or None,
                        service_minutes=job.service,
                        horizon_minutes=setting.horizon,
                        policy=name,
                        select=learned[s].selector(actions) if name == "learned" else None,
                    )
                    rows.append(
                        {
                            "model": job.model,
                            "fold": job.fold,
                            "weight": len(heldout),
                            "placement": rep,
                            "scenario": setting.name,
                            "staff": s,
                            "policy": name,
                            "actions": dict(actions),
                            "metrics": r,
                            "horizon": setting.horizon,
                        }
                    )
    return rows, {
        "model": job.model,
        "fold": job.fold,
        "heldout_dialogues": len(source),
        "positive_heldout_dialogues": sum(source),
        "fixed_cutoff": cut,
        "learners": {s: policy.diagnostics for s, policy in learned.items()},
    }


def ratio(n: float, d: float) -> float | None:
    return n / d if d else None


def combine(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Size-weight the isolated held-out fold queues; pool counts before taking rates."""
    sw = sum(r["weight"] for r in rows)

    def pool(name: str | None) -> dict[str, Any]:
        cs = {
            k: sum(r["weight"] * (r["metrics"]["by_period"][name] if name else r["metrics"])[k] for r in rows) / sw
            for k in COUNT_FIELDS
        }
        pos = cs["positives"]
        if not math.isclose(
            pos,
            cs["catches"]
            + sum(
                cs[k] for k in ("misses_no_alert", "misses_wrong_turn", "misses_nomination_late", "misses_queue_delay")
            ),
        ):
            raise ValueError("positive case partition failed")
        return dict(
            cs,
            recall=ratio(cs["catches"], pos),
            queue_loss=ratio(cs["misses_queue_delay"], pos),
            prevalence=ratio(pos, cs["sessions"]),
        )

    out = pool(None)
    out["by_period"] = {n: pool(n) for n in ("peak", "offpeak")}
    for name in ("contested_starts", "review_minutes", "utilization", "p95_wait_nominated"):
        valid = [r for r in rows if r["metrics"][name] is not None]
        out["mean_" + name] = ratio(
            sum(r["weight"] * r["metrics"][name] for r in valid), sum(r["weight"] for r in valid)
        )
    out["fraction_starts_contested"] = ratio(out["mean_contested_starts"], out["starts"])
    staff = rows[0]["staff"]
    out["offered_load"] = (
        sum(r["weight"] * r["metrics"]["nominated"] * r["service"] / (staff * r["horizon"]) for r in rows) / sw
        if staff
        else None
    )
    out["mean_rl_deviations_from_risk"] = sum(r["weight"] * r["actions"].get("nonrisk_selected", 0) for r in rows) / sw
    return out


def _paired_delta(members: list[dict[str, Any]], base: list[dict[str, Any]], skip: int) -> float | None:
    denom = sum(r["weight"] * r["metrics"]["positives"] for r in base if r["fold"] != skip)
    return ratio(
        sum(
            r["weight"] * (r["metrics"]["catches"] - b["metrics"]["catches"])
            for r, b in zip(members, base, strict=True)
            if r["fold"] != skip
        ),
        denom,
    )


def aggregate(records: list[dict[str, Any]], service: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r["model"], r["scenario"], r["staff"], r["policy"]].append({**r, "service": service})
    output = []
    for (m, scene, s, p), members in groups.items():
        val = combine(members)
        # Per-fold sensitivity: remove an entire held-out queue (including its
        # positives) from the final summary. No refitting or altered dispatch.
        base = groups[m, scene, s, "fifo"]
        if len(base) != len(members) or any(
            (a["fold"], a["placement"], a["metrics"]["nominated"], a["metrics"]["positives"])
            != (b["fold"], b["placement"], b["metrics"]["nominated"], b["metrics"]["positives"])
            for a, b in zip(base, members, strict=True)
        ):
            raise ValueError("dispatch arm did not see the same cases and tickets")

        immediate = groups.get((m, scene, 0, "fifo"))
        if immediate:
            for r, upper in zip(members, immediate, strict=True):
                metrics = r["metrics"]
                if (r["fold"], r["placement"]) != (upper["fold"], upper["placement"]) or (
                    metrics["catches"] + metrics["misses_queue_delay"] != upper["metrics"]["catches"]
                ):
                    raise ValueError("no-wait control and delay loss disagree")
        ranges = [v for f in {r["fold"] for r in base} if (v := _paired_delta(members, base, f)) is not None]
        output.append(
            dict(
                model=m,
                scenario=scene,
                staff=s,
                policy=p,
                **val,
                delta_vs_fifo=_paired_delta(members, base, -1),
                omit_one_fold_delta_range=[min(ranges), max(ranges)] if ranges else None,
            )
        )
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("annotations", "responses", "trace"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--azure", type=Path)
    p.add_argument("--burst-week", type=Path, help="Optional pinned BurstGPT v1.1 trace")
    p.add_argument("--models", nargs="+", choices=("jev", "gemini"), default=("jev", "gemini"))
    p.add_argument("--staff", type=int, nargs="+", default=(1, 2, 3, 0), help="0 denotes unlimited immediate service")
    p.add_argument("--service", type=float, default=20)
    p.add_argument("--gate", choices=("tuned", "liberal"), default="tuned")
    p.add_argument("--placements", type=int, default=8)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args(argv)
    try:
        if (
            min(a.placements, a.jobs, a.episodes) < 1
            or a.episodes > 32
            or min(a.staff) < 0
            or not math.isfinite(a.service)
            or a.service <= 0
            or any(len(set(seq)) != len(seq) for seq in (a.models, a.staff))
        ):
            raise ValueError("invalid replay parameters")
        cases = official_cases(a.annotations)
        complete = complete_dialogues(cases)
        rows = [r for r in attach_responses(cases, local_records(a.responses)) if r.dialogue_id in complete]
        arrivals = load_arrivals(a.trace)
        jobs = [
            Job(
                train,
                test,
                arrivals,
                model,
                fold,
                a.seed,
                a.placements,
                a.episodes,
                a.service,
                a.gate,
                a.azure,
                a.burst_week,
                tuple(a.staff),
            )
            for fold, (train, test) in enumerate(grouped_folds(rows, 5, a.seed))
            for model in a.models
        ]
        if a.jobs == 1:
            parts = [fold_run(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(a.jobs, len(jobs))) as pool:
                parts = list(pool.map(fold_run, jobs))
        report = {
            "kind": "mindguard_surge_v1",
            "service": a.service,
            "gate": a.gate,
            "seed": a.seed,
            "placements": a.placements,
            "training_episodes": a.episodes,
            "source_dialogues": len(complete),
            "turns": len(rows),
            "positive_dialogues": len({r.dialogue_id for r in rows if r.votes.count("self_harm") >= 2}),
            "trace_sha256": TRACE_SHA256,
            "azure_sha256": AZURE_SHA256 if a.azure else None,
            "burst_week_sha256": BURST_SHA256 if a.burst_week else None,
            "clocks_example_first_placement": {
                s.name: s.clock.summary for s in schedules(arrivals, a.azure, a.seed + 8_000_009, a.burst_week)
            },
            "training": [audit for _, audit in parts],
            "results": aggregate([r for records, _ in parts for r in records], a.service),
        }
        print(json.dumps(report, allow_nan=False, indent=2))
    except (ValueError, KeyError, OSError, ArithmeticError, RuntimeError) as exc:
        print(f"Surge replay failed ({type(exc).__name__}); check pinned inputs and settings.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
