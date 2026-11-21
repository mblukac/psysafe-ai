"""Bridge the previously published thresholds to two six-detector policies in the same surge.

The OR arm changes nothing about the original single gates. It emits exactly
one frozen ticket, at the earliest crossing (not the best crossing in hindsight).
The mean arm uses the same inner-family tuning procedure as the original gates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from itertools import accumulate
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION
from .ensemble_fit import encoded_arm
from .fit import grouped_folds
from .full_panel import MODEL_KEYS, coverage_audit, load_panel
from .full_replay import calibrated, tuned_cutoff
from .loss_workmatch import DEADLINE, REPS, ROOT, SEED, SERVICE, dispatch_starts
from .staffing import TRACE_SHA256, load_arrivals
from .surge_replay import Nomination, nominate, schedules
from .surge_stream import assign_paths

SINGLE = tuple(sorted(MODEL_KEYS.values()))
POOL = ("mean6", "any6")
ALL = (*SINGLE, *POOL)


def first_of(noms: Sequence[Nomination]) -> Nomination:
    if not noms or len({n.onset_turn for n in noms}) != 1:
        raise ValueError("unaligned reference cases")
    live = [n for n in noms if n.turn is not None]
    if not live:
        return Nomination(None, None, noms[0].onset_turn, False)
    first = min(n.turn for n in live if n.turn is not None)
    votes = {n.supported for n in live if n.turn == first}
    if len(votes) != 1:
        raise ValueError("different labels at the same turn")
    return Nomination(first, 1.0, noms[0].onset_turn, votes.pop())


def frozen_trace(paths: list[Nomination], times: Sequence[float], order: Sequence[int]) -> list[int]:
    tickets = sorted(
        (times[i] + 2 * (p.turn - 1), i, 1.0) for i, j in enumerate(order) if (p := paths[j]).turn is not None
    )
    starts = dispatch_starts(tickets, 1, "fifo")
    flags = []
    for i, at in starts:
        p = paths[order[i]]
        flags.append(
            int(
                p.supported
                and p.onset_turn is not None
                and times[i] + 2 * (p.onset_turn - 1) <= at <= times[i] + 2 * (p.onset_turn - 1) + DEADLINE
            )
        )
    return list(accumulate(flags, initial=0))


def summary(blocks: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, Any]:
    weight = sum(b["weight"] for b in blocks)
    den = sum(b["positives"] * b["weight"] for b in blocks)
    if not weight or not den or len(set(arms)) != len(arms):
        raise ValueError("invalid comparison")
    folds = sorted({b["fold"] for b in blocks})
    results = []
    for b in blocks:
        k = min(len(b["traces"][arm]) - 1 for arm in arms)
        results.append((b["weight"], k, b["fold"], {a: b["traces"][a][k] for a in arms}))
    hits = {a: sum(w * vals[a] for w, _, _, vals in results) for a in arms}
    return dict(
        positives=den / weight,
        minutes=SERVICE * sum(w * k for w, k, _, _ in results) / weight,
        zero_cap_folds=[f for f in folds if all(k == 0 for _, k, fold, _ in results if f == fold)],
        catches={a: h / weight for a, h in hits.items()},
        recall={a: h / den for a, h in hits.items()},
    )


def analyze(inputs: Path, refs: Path) -> dict[str, Any]:
    prior = json.loads((refs / "ticket-updates.json").read_text())
    cross = json.loads((refs / "full-panel-crosswork.json").read_text())
    panel = load_panel(
        inputs / "mindguard.parquet",
        [
            inputs / p
            for p in (
                "responses.jsonl",
                "luna-responses.jsonl",
                "clef-flash-responses.jsonl",
                "decider-flash-responses.jsonl",
            )
        ],
        expected_models=list(SINGLE),
    )
    clock = load_arrivals(inputs / "real_interarrivals_nov.csv")
    if (
        prior["dataset_revision"] != DATASET_REVISION
        or prior["coverage"] != coverage_audit(panel)
        or prior["seed"] != SEED
        or prior["trace_sha256"] != TRACE_SHA256
        or prior["placements"] != REPS
    ):
        raise ValueError("published experiment mismatch")
    cuts = {(r["model"], r["fold"]): r for r in prior["fits"]}
    blocks: list[dict[str, Any]] = []
    totals = {arm: {"no_ticket": 0, "early_or_unsupported_or_late": 0, "supported_in_time": 0} for arm in ALL}
    rescued_never = {arm: {"mean6_flagged": 0, "mean6_supported_in_time": 0} for arm in (*SINGLE, "any6")}
    pool_cuts = []
    splits = {a: grouped_folds(encoded_arm(panel, a), 5, SEED) for a in (*SINGLE, "mean6")}
    for fold in range(5):
        paths: dict[str, list[Nomination]] = {}
        ids = sorted({r.dialogue_id for r in splits["mean6"][fold][1]})
        for arm in (*SINGLE, "mean6"):
            train, held = splits[arm][fold]
            traj = calibrated(train, held)
            if sorted(traj) != ids or {r.group_id for r in train} & {r.group_id for r in held}:
                raise ValueError("fold identity or training overlap")
            if arm == "mean6":
                inner = {}
                for learn, val in grouped_folds(train, 3, SEED + fold):
                    inner.update(calibrated(learn, val))
                cut = tuned_cutoff(inner, clock, SEED + 1_000_003 + 10_007 * fold)
                pool_cuts.append(cut)
            else:
                rec = cuts[arm, fold]
                cut = rec["cutoffs"]["tuned"]
                if not math.isclose(
                    sum((q - y) ** 2 for p in traj.values() for _, q, y in p), rec["single_brier_sum"], abs_tol=1e-8
                ):
                    raise ValueError("old calibrated probabilities do not reproduce")
            paths[arm] = [nominate(traj[i], cut) for i in ids]
        paths["any6"] = [first_of(ns) for ns in zip(*(paths[a] for a in SINGLE), strict=True)]
        positive = [p.onset_turn is not None for p in paths["mean6"]]
        for old, count in rescued_never.items():
            for p, q in zip(paths[old], paths["mean6"], strict=True):
                if p.onset_turn is not None and p.turn is None and q.turn is not None:
                    count["mean6_flagged"] += 1
                    count["mean6_supported_in_time"] += int(
                        q.supported and 0 <= 2 * (q.turn - p.onset_turn) <= DEADLINE
                    )
        for a, pp in paths.items():
            if [p.onset_turn for p in pp] != [p.onset_turn for p in paths["mean6"]]:
                raise ValueError("ensemble reference changed")
            for p in pp:
                if p.onset_turn is not None:
                    if p.turn is None:
                        key = "no_ticket"
                    elif p.supported and 0 <= 2 * (p.turn - p.onset_turn) <= DEADLINE:
                        key = "supported_in_time"
                    else:
                        key = "early_or_unsupported_or_late"
                    totals[a][key] += 1
        for rep in range(REPS):
            setting = next(s for s in schedules(clock, None, SEED + 8_000_009 + rep) if s.name == "compound")
            order, meta = assign_paths(positive, setting.clock, seed=SEED + 1009 * fold + rep, risk_burst=setting.burst)
            blocks.append(
                dict(
                    weight=len(ids),
                    fold=fold,
                    positives=meta["positives"],
                    traces={a: frozen_trace(pp, setting.clock.arrivals, order) for a, pp in paths.items()},
                )
            )
    baseline = summary(blocks, SINGLE)
    ref = next(r for r in cross["results"] if (r["scenario"], r["staff"]) == ("compound", 1))
    checks = [(baseline["positives"], ref["positives"]), (baseline["minutes"], ref["matched_minutes"])]
    checks += [(baseline["catches"][a], ref["models"][a]["matched_catches"]) for a in SINGLE]
    if not all(math.isclose(a, b, abs_tol=1e-8) for a, b in checks):
        raise ValueError("historical six-detector common-work baseline did not reproduce")
    return dict(
        kind="mindguard_ensemble_historical_gates_v1",
        dataset_revision=DATASET_REVISION,
        trace_sha256=TRACE_SHA256,
        seed=SEED,
        placements=REPS,
        folds=5,
        scenario="compound",
        staff=1,
        service_minutes=SERVICE,
        deadline_minutes=DEADLINE,
        turn_minutes=2,
        immediate_source_counts=totals,
        mean6_on_original_positive_no_tickets=rescued_never,
        mean6_fold_cutoffs=pool_cuts,
        original_six=baseline,
        all_eight=summary(blocks, ALL),
        pairwise={pool: {s: summary(blocks, (pool, s)) for s in SINGLE} for pool in POOL},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "private")
    p.add_argument("--refs", type=Path, default=ROOT / "figures")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a fresh output path")
    result = analyze(args.inputs, args.refs)
    with args.out.open("x", encoding="utf-8") as f:
        print(json.dumps(result, allow_nan=False, indent=2), file=f)


if __name__ == "__main__":
    main()
