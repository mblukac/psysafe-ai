"""Paired, work-matched ensemble versus singles on evolving conversations. Entirely offline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, product
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION
from .fit import grouped_folds
from .full_panel import MODEL_KEYS, PanelTurn, coverage_audit, load_panel
from .full_replay import encoded
from .loss_workmatch import DEADLINE, REPS, ROOT, SEED, SERVICE, dispatch_starts
from .staffing import TRACE_SHA256, load_arrivals
from .staffing_queue import Trajectory
from .surge_replay import Nomination, nominate, schedules
from .surge_stream import assign_paths

SINGLES = tuple(sorted(MODEL_KEYS.values()))
ENSEMBLE = "mean6"
SELECTED = "selected_single"
NAMES = (*SINGLES, ENSEMBLE)
SCENARIOS = ("observed", "double", "compound")
Release = tuple[float, int, float]
BRIDGE = ("never_flagged", "flagged_unsupported", "supported_flag_late", "supported_flag_in_time")


def flag_status(p: Nomination) -> str:
    """Frozen nominal eligibility before sharing scarce slots; evaluated only after decisions."""
    if p.turn is None:
        return BRIDGE[0]
    if not p.supported:
        return BRIDGE[1]
    if p.onset_turn is None or p.turn < p.onset_turn:
        raise ValueError("supported nomination cannot precede first evidence")
    return BRIDGE[2] if 2 * (p.turn - p.onset_turn) > DEADLINE else BRIDGE[3]


def scored_order(
    releases: list[Release], staff: int, times: Sequence[float], order: Sequence[int], paths: Sequence[Nomination]
) -> list[int]:
    """One marker per actual review in chronological order; -1 means a paid noncatch."""
    starts = dispatch_starts(releases, staff, "fifo")
    out = []
    for sid, start in starts:
        n = paths[order[sid]]
        onset = times[sid] + 2 * (n.onset_turn - 1) if n.onset_turn is not None else math.inf
        out.append(sid if n.supported and onset <= start <= onset + DEADLINE else -1)
    return out


def match_block(scored: dict[str, list[int]], selected: str) -> dict[str, tuple[int, dict[str, set[int]]]]:
    """Pay for all actual starts, truncate without observing labels. Hit sets for paired rescues/losses."""
    if selected not in SINGLES or set(scored) != set(NAMES):
        raise ValueError("incomplete ensemble comparison")
    groups = {"all": NAMES, "selected_pair": (ENSEMBLE, selected)}
    groups.update({"pair_" + m: (ENSEMBLE, m) for m in SINGLES})
    out = {}
    cache: dict[tuple[str, int], set[int]] = {}
    for key, models in groups.items():
        cap = min(len(scored[m]) for m in models)
        hits = {}
        for m in models:
            if (m, cap) not in cache:
                cache[m, cap] = {sid for sid in scored[m][:cap] if sid >= 0}
            hits[SELECTED if key == "selected_pair" and m == selected else m] = cache[m, cap]
        if key == "all":
            hits[SELECTED] = hits[selected]
        out[key] = (cap, hits)
    return out


def tranche(
    groups: list[dict[str, tuple[int, dict[str, set[int]]]]],
    order: Sequence[int],
    family: list[str],
    *,
    status: dict[str, list[str]] | None = None,
    nobody: set[int] | None = None,
) -> dict[str, Any]:
    """Cohorts are disjoint. Return sums plus nonpublic counts by family, not conversations or predictions."""
    out = {}
    for key in groups[0]:
        caps = [g[key][0] for g in groups]
        hits = {m: set().union(*(g[key][1][m] for g in groups)) for m in groups[0][key][1]}
        paired = {}
        for m, baseline in hits.items():
            if m == ENSEMBLE:
                continue
            rescued, lost = hits[ENSEMBLE] - baseline, baseline - hits[ENSEMBLE]
            paired[m] = {
                "rescued": len(rescued),
                "lost": len(lost),
                "gf": Counter(family[order[sid]] for sid in rescued),
                "lf": Counter(family[order[sid]] for sid in lost),
            }
            if status is not None:
                # Local path ordinal is never exported. A cause is assigned using
                # the non-catching arm's frozen nomination on the SAME source.
                bridge = {}
                for label, episodes, missed_arm in (("gained", rescued, m), ("lost", lost, ENSEMBLE)):
                    counts = Counter(order[sid] for sid in episodes)
                    for reason in BRIDGE:
                        bridge[label + "_" + reason] = {
                            i: n for i, n in counts.items() if status[missed_arm][i] == reason
                        }
                    if label == "gained":
                        bridge["gained_all_six_never_flagged"] = {
                            i: n for i, n in counts.items() if nobody and i in nobody
                        }
                paired[m]["bridge"] = bridge
        out[key] = dict(
            starts=sum(caps),
            empty_subblocks=caps.count(0),
            subblocks=len(caps),
            hits={m: len(v) for m, v in hits.items()},
            paired=paired,
        )
    return out


def replay_fold(
    paths: dict[str, dict[str, Trajectory]],
    cutoffs: dict[str, float],
    selected_single: str,
    families: dict[str, str],
    clock: list[float],
    fold: int,
    *,
    scenarios: tuple[str, ...] = SCENARIOS,
    staff: tuple[int, ...] = (1, 2, 3),
) -> list[dict[str, Any]]:
    if not families or set(paths) != set(NAMES) or set(cutoffs) != set(NAMES):
        raise ValueError("incomplete predictions or thresholds")
    ids = sorted(families)
    if any(set(p) != set(ids) for p in paths.values()):
        raise ValueError("different held-out paths")
    for d in ids:
        reference = [(t, y) for t, _, y in paths[ENSEMBLE][d]]
        if any([(t, y) for t, _, y in paths[m][d]] != reference for m in SINGLES):
            raise ValueError("different reference turn labels or indices")
    frozen = {m: [nominate(paths[m][d], cutoffs[m]) for d in ids] for m in NAMES}
    status = {m: [flag_status(p) for p in noms] for m, noms in frozen.items()}
    status[SELECTED] = status[selected_single]
    nobody = {i for i in range(len(ids)) if all(status[m][i] == BRIDGE[0] for m in SINGLES)}
    family = [families[d] for d in ids]
    positive = [n.onset_turn is not None for n in frozen[ENSEMBLE]]
    posfamilies = {f for p, f in zip(positive, family, strict=True) if p}
    rows = []
    for rep in range(REPS):
        for setting in schedules(clock, None, SEED + 8_000_009 + rep):
            if setting.name not in scenarios:
                continue
            times = setting.clock.arrivals
            order, _ = assign_paths(positive, setting.clock, seed=SEED + 1009 * fold + rep, risk_burst=setting.burst)
            den = sum(positive[i] for i in order)
            byday = int(times[-1] // 1440) + 1
            tickets: dict[str, list[Release]] = {}
            days: list[dict[str, list[Release]]] = [defaultdict(list) for _ in range(byday)]
            for m, noms in frozen.items():
                tickets[m] = sorted(
                    (times[sid] + 2 * (n.turn - 1), sid, n.score)
                    for sid, i in enumerate(order)
                    if (n := noms[i]).turn is not None and n.score is not None
                )
                for item in tickets[m]:
                    days[int(times[item[1]] // 1440)][m].append(item)
            for s in staff:
                scored = {m: scored_order(tickets[m], s, times, order, frozen[m]) for m in NAMES}
                work = tranche([match_block(scored, selected_single)], order, family, status=status, nobody=nobody)
                daily = {}
                if s == 1 and setting.name in ("compound", "observed"):
                    budgets = []
                    for day in days:
                        table = {m: scored_order(day[m], s, times, order, frozen[m]) for m in NAMES}
                        budgets.append(match_block(table, selected_single))
                    daily = tranche(budgets, order, family, status=status, nobody=nobody)
                rows.append(
                    dict(
                        fold=fold,
                        rep=rep,
                        weight=len(ids),
                        scenario=setting.name,
                        staff=s,
                        sessions=len(order),
                        positives=den,
                        positive_families=posfamilies,
                        work=work,
                        day_region=daily,
                        full={m: (len(v), sum(i >= 0 for i in v)) for m, v in scored.items()},
                    )
                )
    return rows


def summarize_budget(rows: list[dict[str, Any]], col: str, key: str) -> dict[str, Any]:
    w = sum(r["weight"] for r in rows)
    den = sum(r["weight"] * r["positives"] for r in rows)
    if not den:
        raise ValueError("no positive episodes")
    family = set().union(*(r["positive_families"] for r in rows))
    raw = [r[col][key] for r in rows]
    folds = sorted({r["fold"] for r in rows})
    hits = {m: sum(r["weight"] * a["hits"][m] for r, a in zip(rows, raw, strict=True)) for m in raw[0]["hits"]}
    paired = {}
    for m in raw[0]["paired"]:

        def net(sub: list[dict[str, Any]], m: str = m) -> float:
            num = sum(r["weight"] * (r[col][key]["hits"][ENSEMBLE] - r[col][key]["hits"][m]) for r in sub)
            return float(num / sum(r["weight"] * r["positives"] for r in sub))

        rr = ll = 0
        credit: Counter[str] = Counter()
        causes: Counter[str] = Counter()
        paths: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for r, a in zip(rows, raw, strict=True):
            p = a["paired"][m]
            rr += r["weight"] * p["rescued"]
            ll += r["weight"] * p["lost"]
            for k, sign in (("gf", 1), ("lf", -1)):
                credit.update({f: sign * r["weight"] * n for f, n in p[k].items()})
            for reason, cases in p["bridge"].items():
                causes[reason] += r["weight"] * sum(cases.values())
                paths[reason].update((r["fold"], i) for i in cases)
        if hits[ENSEMBLE] - hits[m] != rr - ll or sum(credit.values()) != rr - ll:
            raise ValueError("discordant-case differences did not balance")
        if sum(causes["gained_" + k] for k in BRIDGE) != rr or sum(causes["lost_" + k] for k in BRIDGE) != ll:
            raise ValueError("paired sources did not partition discordant catches")
        omit = [net([r for r in rows if r["fold"] != f]) for f in folds] if len(folds) > 1 else []
        paired[m] = dict(
            delta=(rr - ll) / den,
            gained=rr / w,
            lost=ll / w,
            bridge={reason: dict(episodes=n / w, distinct_paths=len(paths[reason])) for reason, n in causes.items()},
            leave_fold_range=[min(omit), max(omit)] if omit else None,
            family_net_wins=sum(credit[f] > 0 for f in family),
            family_net_losses=sum(credit[f] < 0 for f in family),
            family_net_ties=sum(credit[f] == 0 for f in family),
        )
    return dict(
        minutes=SERVICE * sum(r["weight"] * a["starts"] for r, a in zip(rows, raw, strict=True)) / w,
        zero_budget_folds=[f for f in folds if not any(r[col][key]["starts"] for r in rows if r["fold"] == f)],
        zero_blocks=sum(a["starts"] == 0 for a in raw),
        blocks=len(rows),
        zero_subblocks=sum(a["empty_subblocks"] for a in raw),
        subblocks=sum(a["subblocks"] for a in raw),
        catches={m: h / w for m, h in hits.items()},
        recall={m: h / den for m, h in hits.items()},
        paired=paired,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        not rows
        or len({(r["scenario"], r["staff"]) for r in rows}) != 1
        or len(rows) != 5 * REPS
        or {(r["fold"], r["rep"]) for r in rows} != set(product(range(5), range(REPS)))
    ):
        raise ValueError("incomplete, duplicate or mixed replay conditions")
    w = sum(r["weight"] for r in rows)
    work = {k: summarize_budget(rows, "work", k) for k in rows[0]["work"]}
    days = {k: summarize_budget(rows, "day_region", k) for k in rows[0]["day_region"]}
    return dict(
        scenario=rows[0]["scenario"],
        staff=rows[0]["staff"],
        positives=sum(r["weight"] * r["positives"] for r in rows) / w,
        sessions=rows[0]["sessions"],
        uncapped={
            m: dict(
                minutes=SERVICE * sum(r["weight"] * r["full"][m][0] for r in rows) / w,
                catches=sum(r["weight"] * r["full"][m][1] for r in rows) / w,
            )
            for m in NAMES
        },
        work=work,
        daily=days,
    )


def one(
    job: tuple[list[PanelTurn], set[str], set[str], list[float], int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from .ensemble_fit import fit_fold

    panel, train, held, clock, f = job
    fit = fit_fold(panel, train, held, f)
    rows = replay_fold(
        fit.paths,
        fit.cutoffs,
        fit.selected_single,
        {r.dialogue_id: r.group_id for r in panel if r.dialogue_id in held},
        clock,
        f,
    )
    return fit.audit, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "private")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.jobs < 1:
        p.error("use a fresh output and positive workers")
    root = args.inputs
    panel = load_panel(
        root / "mindguard.parquet",
        [
            root / n
            for n in (
                "responses.jsonl",
                "luna-responses.jsonl",
                "clef-flash-responses.jsonl",
                "decider-flash-responses.jsonl",
            )
        ],
        expected_models=SINGLES,
    )
    coverage = coverage_audit(panel)
    if (coverage["paths"], coverage["rated_turns"], coverage["opening_families"]) != (67, 1134, 50):
        p.error("unexpected cached cohort")
    clock = load_arrivals(root / "real_interarrivals_nov.csv")
    jobs = [
        (panel, {r.dialogue_id for r in tr}, {r.dialogue_id for r in te}, clock, f)
        for f, (tr, te) in enumerate(grouped_folds(encoded(panel, SINGLES[0]), 5, SEED))
    ]
    with ProcessPoolExecutor(max_workers=min(5, args.jobs)) as pool:
        results = list(pool.map(one, jobs))
    rows = sorted((r for a in results for r in a[1]), key=lambda r: (r["scenario"], r["staff"]))
    out = dict(
        kind="mindguard_ensemble_work_matched_v1",
        dataset_revision=DATASET_REVISION,
        trace_sha256=TRACE_SHA256,
        coverage=coverage,
        placements=REPS,
        folds=5,
        seed=SEED,
        service_minutes=SERVICE,
        deadline_minutes=DEADLINE,
        turn_minutes=2,
        primary=["compound", 1],
        fit=[a[0] for a in results],
        method="First K actual FIFO starts including negatives per paired fold/placement; all positives retained. "
        "K is minimum across the named comparison set. Daily sensitivity resets and drains each arrival-day independently.",
        results=[summarize(list(g)) for _, g in groupby(rows, key=lambda r: (r["scenario"], r["staff"]))],
    )
    with args.out.open("x", encoding="utf-8") as fd:
        json.dump(out, fd, allow_nan=False, separators=(",", ":"))
        fd.write("\n")


if __name__ == "__main__":
    main()
