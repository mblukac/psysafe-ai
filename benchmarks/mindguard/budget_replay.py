"""Offline comparison at identical realized review-minutes; no provider or network access."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .__main__ import official_cases
from .budget_fit import DEADLINE_MINUTES, DENSE_GRID, GRID, POLICIES, SERVICE_MINUTES, TURN_MINUTES, tune_folds
from .data import Turn, attach_responses, complete_dialogues, local_records
from .staffing import TRACE_SHA256, TRACE_URL, load_arrivals, placement
from .staffing_queue import Session, simulate

SERVICE = int(SERVICE_MINUTES)
MODELS = ("jev", "gemini")
# Declared minute caps, not points chosen using held-out recalls.
DEFAULT_MINUTES = (5_000, 10_000, 15_000, 18_000, 20_000, 25_000, 30_000)


def paired_prefix(a: Sequence[bool], b: Sequence[bool], cap: int) -> tuple[int, int, int]:
    """Use the same number of starts, never gold, to set the common work level."""
    if type(cap) is not int or cap < 0:
        raise ValueError("need a nonnegative integer review cap")
    n = min(cap, len(a), len(b))
    return n, sum(a[:n]), sum(b[:n])


def summarize(runs: list[tuple[int, dict[str, list[bool]]]], budget: int) -> dict[str, Any]:
    total_pos = sum(pos for pos, _ in runs)
    if not runs or total_pos <= 0:
        raise ValueError("need evaluable held-out placements")
    out: dict[str, Any] = {"budget_minutes": SERVICE * budget, "pooled_positive_arrivals": total_pos}
    for name in ("rl", "rl_fifo"):
        pairs = [paired_prefix(flags["fixed"], flags[name], budget) for _, flags in runs]
        minutes = [starts * SERVICE for starts, _, _ in pairs]
        a = sum(x[1] for x in pairs)
        b = sum(x[2] for x in pairs)
        comparison: dict[str, Any] = {
            "paired_mean_actual_minutes": sum(minutes) / len(minutes),
            "paired_min_max_actual_minutes": [min(minutes), max(minutes)],
            "placements_filling_cap_in_both": sum(m == budget * SERVICE for m in minutes),
            "fixed_paired_recall": a / total_pos,
            f"{name}_paired_recall": b / total_pos,
            "paired_delta_pp": 100 * (b - a) / total_pos,
            "paired_catches": {"fixed": a, name: b},
            "per_placement": [
                {"minutes_each": m, "positive_arrivals": pos, "fixed_catches": x, f"{name}_catches": y}
                for (pos, _), (_, x, y), m in zip(runs, pairs, minutes, strict=True)
            ],
        }
        if name == "rl":
            out.update(comparison)
        else:
            out["fifo_ablation"] = comparison
    # Independently implementable policy: stop at the originally announced hard cap
    # (may underspend). No adjustment to the common observed spend in this table.
    out["standalone_at_nominal_cap"] = {
        name: {
            "mean_actual_minutes": SERVICE * sum(min(budget, len(f[name])) for _, f in runs) / len(runs),
            "recall": sum(sum(f[name][:budget]) for _, f in runs) / total_pos,
        }
        for name in POLICIES
    }
    return out


def family_deletions(positives: Counter[str], fixed: Counter[str], rl: Counter[str]) -> dict[str, Any]:
    """Drop one positive opening family's scored outcomes, not its traffic or fitted models."""
    n, delta = sum(positives.values()), sum(rl.values()) - sum(fixed.values())
    diffs = [100 * (delta - rl[g] + fixed[g]) / (n - count) for g, count in positives.items() if 0 < count < n]
    return {
        "positive_families": len(positives),
        "min_max_delta_pp": [min(diffs), max(diffs)] if diffs else None,
        "scope": "omit one positive opening family's outcomes; fixed predictions, traffic and dispatch; not a CI",
    }


Job = tuple[list[Turn], list[float], str, int, tuple[int, ...], int, int, int, bool, bool]


def evaluate(job: Job) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, arrivals, model, staff, budgets, seed, inner_placements, placements, require_fill, dense_grid = job
    tuned, audit = tune_folds(
        rows,
        arrivals,
        model=model,
        staff=staff,
        budgets=budgets,
        seed=seed,
        inner_placements=inner_placements,
        require_fill=require_fill,
        dense_grid=dense_grid,
    )
    n = len({r.dialogue_id for r in rows})
    orders = [placement(n, len(arrivals), seed + i) for i in range(placements)]
    source_families = {row.dialogue_id: row.group_id or row.dialogue_id for row in rows}
    positive_ids = {r.dialogue_id for r in rows if 2 * r.votes.count("self_harm") > len(r.votes)}
    families = [source_families[d] if d in positive_ids else None for d in sorted(source_families)]
    results = []
    for budget in budgets:
        runs = []
        pos_groups: Counter[str] = Counter()
        catch_groups: dict[str, Counter[str]] = {name: Counter() for name in ("fixed", "rl")}
        specs = tuned[budget]
        if any(len(specs[name]) != n for name in POLICIES):
            raise ValueError("model paths missing from a held-out fold")
        for indices in orders:
            audits: dict[str, list[bool]] = {}
            dispatch: dict[str, list[int]] = {}
            positives = set()
            for name in POLICIES:
                paths = specs[name]
                stream = [Session(at, paths[i][0]) for at, i in zip(arrivals, indices, strict=True)]
                flags: list[bool] = []
                indices_started: list[int] = []
                result = simulate(
                    stream,
                    reviewers=staff,
                    service_minutes=SERVICE,
                    turn_minutes=TURN_MINUTES,
                    deadline_minutes=DEADLINE_MINUTES,
                    policy="rl" if name == "rl" else "fixed",
                    metric=lambda s, _: s,
                    threshold=0.0,
                    session_thresholds=[paths[i][1] for i in indices],
                    starts_audit=flags,
                    starts_session_ids=indices_started,
                )
                if len(flags) != result["reviews"] or sum(flags) != result["catches_by_onset_deadline"]:
                    raise RuntimeError("dispatch audit inconsistent with full replay")
                positives.add(int(result["positives"] or 0))
                audits[name] = flags
                dispatch[name] = indices_started
            if len(positives) != 1:
                raise ValueError("unpaired denominators")
            pos_groups.update(g for i in indices if (g := families[i]) is not None)
            common = min(budget, len(audits["fixed"]), len(audits["rl"]))
            for name, counter in catch_groups.items():
                for sid, hit in zip(dispatch[name][:common], audits[name][:common], strict=True):
                    if hit:
                        group = families[indices[sid]]
                        if group is None:
                            raise ValueError("a catch must belong to a positive family")
                        counter[group] += 1
            runs.append((positives.pop(), audits))
        summary = summarize(runs, budget)
        if sum(pos_groups.values()) != summary["pooled_positive_arrivals"] or any(
            sum(c.values()) != summary["paired_catches"][k] for k, c in catch_groups.items()
        ):
            raise RuntimeError("source family totals do not match paired results")
        results.append(
            {
                "model": model,
                "reviewers": staff,
                **summary,
                "positive_family_deletion": family_deletions(pos_groups, catch_groups["fixed"], catch_groups["rl"]),
            }
        )
    return results, [{"model": model, "reviewers": staff, **row} for row in audit]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations", "responses", "trace"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--models", choices=MODELS, nargs="+", default=MODELS)
    p.add_argument("--staffing", type=int, nargs="+", default=[1, 6])
    p.add_argument("--minutes", type=int, nargs="+", default=DEFAULT_MINUTES)
    p.add_argument("--placements", type=int, default=8)
    p.add_argument("--inner-placements", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--jobs", type=int, default=1, help="parallel local model/staff fits")
    p.add_argument("--dense-grid", action="store_true", help="sensitivity: finer shared candidate cutoff grid")
    p.add_argument(
        "--require-fill",
        action="store_true",
        help="sensitivity: require training policies to exhaust each budget if feasible",
    )
    args = p.parse_args(argv)
    try:
        if (
            any(x <= 0 or x % SERVICE for x in args.minutes)
            or min(args.staffing) < 1
            or min(args.jobs, args.inner_placements, args.placements) < 1
            or any(len(set(seq)) != len(seq) for seq in (args.minutes, args.models, args.staffing))
        ):
            raise ValueError("invalid budget grid, staffing, replicates or worker count")
        arrivals = load_arrivals(args.trace)
        cases = official_cases(args.annotations)
        complete = complete_dialogues(cases)
        rows = [r for r in attach_responses(cases, local_records(args.responses)) if r.dialogue_id in complete]
        positive = [r for r in rows if 2 * r.votes.count("self_harm") > len(r.votes)]
        budgets = tuple(m // SERVICE for m in sorted(args.minutes))
        jobs: list[Job] = [
            (
                rows,
                arrivals,
                m,
                k,
                budgets,
                args.seed,
                args.inner_placements,
                args.placements,
                args.require_fill,
                args.dense_grid,
            )
            for m in args.models
            for k in sorted(args.staffing)
        ]
        if args.jobs == 1:
            output = [evaluate(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(len(jobs), args.jobs)) as pool:
                output = list(pool.map(evaluate, jobs))
        report = {
            "kind": "mindguard_onset_budget_v1",
            "source": {
                "turns": len(rows),
                "positive_turns": len(positive),
                "dialogues": len(complete),
                "positive_dialogues": len({r.dialogue_id for r in positive}),
                "opening_families": len({r.group_id for r in rows}),
                "positive_families": len({r.group_id for r in positive}),
            },
            "scenario": {
                "training_budget": "fill_if_feasible" if args.require_fill else "ceiling_may_underspend",
                "candidate_cutoffs": list(DENSE_GRID if args.dense_grid else GRID),
                "seed": args.seed,
                "placements": args.placements,
                "inner_placements": args.inner_placements,
                "outer_folds": 5,
                "inner_folds": 3,
                "service_minutes": SERVICE,
                "turn_minutes": TURN_MINUTES,
                "deadline_from_first_endorsed_turn_minutes": DEADLINE_MINUTES,
                "trace_url": TRACE_URL,
                "trace_sha256": TRACE_SHA256,
                "arrivals": len(arrivals),
            },
            "results": [r for results, _ in output for r in results],
            "training_audit": [r for _, audit in output for r in audit],
        }
        print(json.dumps(report, allow_nan=False, indent=2))
    except (ArithmeticError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Offline replay failed ({type(exc).__name__}); check private inputs and parameters.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
