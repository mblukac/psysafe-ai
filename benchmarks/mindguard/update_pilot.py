"""Offline, paired full-panel ticket-update pilot. Outputs only aggregate statistics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION, Turn
from .fit import grouped_folds
from .full_panel import MODEL_KEYS, coverage_audit, load_panel
from .full_replay import calibrated, encoded, tuned_cutoff
from .staffing import TRACE_SHA256, load_arrivals
from .surge_replay import schedules
from .surge_stream import AZURE_SHA256, BURST_SHA256, assign_paths
from .update_fit import fit_filter
from .update_pair import fit_pair
from .update_queue import ARMS, prepare_path, replay_updates

SEED = 2026
PLACEMENTS = 8
COMMON = (
    "sessions",
    "positives",
    "nominated",
    "expired",
    "starts",
    "review_minutes",
    "updatable_starts",
    "timely_updatable_starts",
    "timely_positive_starts",
    "before_onset_starts",
    "after_deadline_starts",
)
METRICS = ("catches", "rescued", "spoiled", "changed", "unsupported_starts")


@dataclass(frozen=True)
class Job:
    model: str
    fold: int
    train: list[Turn]
    test: list[Turn]
    clock: list[float]
    azure: Path
    week: Path


def families(rows: Sequence[Turn]) -> dict[str, str]:
    result: dict[str, str] = {}
    for r in rows:
        if r.group_id is None or result.setdefault(r.dialogue_id, r.group_id) != r.group_id:
            raise ValueError("missing or inconsistent opening family")
    return result


def run(job: Job) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_f, test_f = families(job.train), families(job.test)
    if set(train_f.values()) & set(test_f.values()):
        raise ValueError("leaked test opening family")
    inner = {}
    for learn, val in grouped_folds(job.train, 3, SEED + job.fold):
        inner.update(calibrated(learn, val))
    if set(inner) != set(train_f):
        raise ValueError("incomplete training crossfit")
    post = fit_filter(inner.values())
    held = calibrated(job.train, job.test)
    ids = sorted(held)
    refs = [any(y for _, _, y in held[d]) for d in ids]
    risk = {d: post.posterior((t, q) for t, q, _ in path) for d, path in held.items()}
    qerr = sum((q - y) ** 2 for p in held.values() for _, q, y in p)
    perr = sum((b - y) ** 2 for d, p in held.items() for (_, _, y), b in zip(p, risk[d], strict=True))
    cuts = dict(tuned=tuned_cutoff(inner, job.clock, SEED + 1_000_003 + 10_007 * job.fold), liberal=0.10)
    plans = {}
    pair_audit = {}
    for gate, cut in cuts.items():
        pair = fit_pair(inner, cut, families=train_f)
        plans[gate] = [prepare_path(held[d], risk[d], cut, pair_rule=pair) for d in ids]
        pair_audit[gate] = [bool(pair.replace(a, b)) for a, b in ((0.75, 0.25), (0.5, 0.5), (0.25, 0.75))]
    rows = []
    for rep in range(PLACEMENTS):
        for setting in schedules(job.clock, job.azure, SEED + 8_000_009 + rep, job.week):
            order, _ = assign_paths(refs, setting.clock, seed=SEED + 1009 * job.fold + rep, risk_burst=setting.burst)
            for gate, paths in plans.items():
                for s in (1, 2, 3, 0):
                    affected: dict[str, tuple[set[int], set[int]]] = {}
                    data = replay_updates(
                        paths,
                        setting.clock.arrivals,
                        order,
                        setting.clock.peak_mask,
                        reviewers=s or None,
                        horizon_minutes=setting.horizon,
                        affected_sources=affected,
                    )
                    rows.append(
                        dict(
                            model=job.model,
                            gate=gate,
                            scenario=setting.name,
                            staff=s,
                            fold=job.fold,
                            weight=len(ids),
                            placement=rep,
                            data=data,
                            # Only used in memory to deduplicate. Never emitted in the report.
                            affected={
                                a: [
                                    (set(ids[i] for i in indices), set(test_f[ids[i]] for i in indices))
                                    for indices in groups
                                ]
                                for a, groups in affected.items()
                            },
                        )
                    )
    return (
        dict(
            model=job.model,
            fold=job.fold,
            held_turns=len(job.test),
            held_paths=len(ids),
            cutoffs=cuts,
            single_brier_sum=qerr,
            filter_brier_sum=perr,
            training_adjacent_pairs=post.adjacent_pairs,
            paired_replace_low_mid_high=pair_audit,
        ),
        rows,
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired contrast per positive session, path-weighted exactly as the published baseline."""
    if not rows or len({(r["placement"], r["fold"]) for r in rows}) != len(rows):
        raise ValueError("empty or duplicate replay block")
    first = rows[0]
    axes = {k: first[k] for k in ("model", "gate", "scenario", "staff")}
    if any(any(r[k] != v for k, v in axes.items()) for r in rows):
        raise ValueError("different replay conditions in one block")
    w = sum(r["weight"] for r in rows)
    counts = {k: sum(r["weight"] * r["data"][k] for r in rows) / w for k in COMMON}
    if counts["positives"] <= 0:
        raise ValueError("no observed-positive sessions")
    arms = {}
    for a in ARMS:
        m = {k: sum(r["weight"] * r["data"]["arms"][a][k] for r in rows) / w for k in METRICS}
        if not math.isclose(
            m["catches"],
            (sum(r["weight"] * r["data"]["arms"]["frozen"]["catches"] for r in rows) / w + m["rescued"] - m["spoiled"]),
            abs_tol=1e-9,
        ):
            raise ValueError("rescue/spoil identity failed")
        ind = [[set(), set()], [set(), set()]]  # type: list[list[set[str]]]
        for r in rows:
            if a in r["affected"]:
                for pair, group in zip(ind, r["affected"][a], strict=True):
                    for col, entries in zip(pair, group, strict=True):
                        col.update(entries)

        def delta(subset: list[dict[str, Any]], arm: str) -> float | None:
            den = sum(r["weight"] * r["data"]["positives"] for r in subset)
            return (
                sum(
                    r["weight"] * (r["data"]["arms"][arm]["rescued"] - r["data"]["arms"][arm]["spoiled"])
                    for r in subset
                )
                / den
                if den
                else None
            )

        folds = [
            x
            for f in sorted({r["fold"] for r in rows})
            if (x := delta([r for r in rows if r["fold"] != f], a)) is not None
        ]
        placements = [
            x
            for p in sorted({r["placement"] for r in rows})
            if (x := delta([r for r in rows if r["placement"] == p], a)) is not None
        ]
        arms[a] = dict(
            **m,
            recall=m["catches"] / counts["positives"],
            delta=(m["rescued"] - m["spoiled"]) / counts["positives"],
            omit_fold_range=[min(folds), max(folds)] if folds else None,
            placement_range=[min(placements), max(placements)] if placements else None,
            rescued_paths=len(ind[0][0]),
            spoiled_paths=len(ind[1][0]),
            rescued_families=len(ind[0][1]),
            spoiled_families=len(ind[1][1]),
        )
    if (
        not all(v["catches"] <= arms["visible_oracle"]["catches"] + 1e-8 for v in arms.values())
        or arms["visible_oracle"]["catches"] > counts["timely_positive_starts"] + 1e-8
    ):
        raise ValueError("evidence/access ceilings not nested")
    if axes["staff"] == 0 and (counts["updatable_starts"] or any(v["changed"] for v in arms.values())):
        raise ValueError("update altered immediate service")
    return dict(**axes, **counts, arms=arms)


def verify_previous(results: list[dict[str, Any]], ref_dir: Path) -> None:
    """Verify each complete gate-by-detector grid against its corresponding published FIFO."""
    models = sorted(MODEL_KEYS.values())
    scenarios = (
        "steady",
        "steady_pulse",
        "observed",
        "risk_pulse",
        "double",
        "compound",
        "azure",
        "burst_week",
        "burst_week_steady",
    )
    expected = {(g, m, s, k) for g in ("tuned", "liberal") for m in models for s in scenarios for k in (0, 1, 2, 3)}
    columns = {
        "model",
        "scenario",
        "staff",
        "policy",
        "positives",
        "sessions",
        "nominated",
        "starts",
        "expired",
        "mean_review_minutes",
        "catches",
        "unsupported_starts",
    }
    contract = {
        "kind": "mindguard_public_operating_points_v1",
        "trace_sha256": TRACE_SHA256,
        "azure_sha256": AZURE_SHA256,
        "burst_sha256": BURST_SHA256,
        "placements": PLACEMENTS,
        "seed": SEED,
        "service": 20,
        "deadline_minutes": 10,
        "turn_minutes": 2,
        "rated_turns": 1134,
        "dialogues": 67,
        "families": 50,
    }
    previous = {}
    for gate in ("tuned", "liberal"):
        d = json.loads((ref_dir / f"full-panel-{gate}.json").read_text())
        if any(d.get(k) != v for k, v in (contract | {"gate": gate}).items()) or sorted(d.get("models", [])) != models:
            raise ValueError("mismatched original clocks, cohort, settings or gate")
        if (
            not isinstance(d.get("columns"), list)
            or len(d["columns"]) != len(set(d["columns"]))
            or not columns.issubset(d["columns"])
        ):
            raise ValueError("missing or duplicate baseline columns")
        for vals in d["rows"]:
            r = dict(zip(d["columns"], vals, strict=True))
            if r["policy"] == "fifo":
                key = (gate, r["model"], r["scenario"], r["staff"])
                if type(r["staff"]) is not int or key not in expected or key in previous:
                    raise ValueError("unexpected or duplicate baseline FIFO condition")
                previous[key] = r
    if set(previous) != expected or len(results) != len(expected):
        raise ValueError("incomplete six-model replay grid")
    for r in results:
        key = (r["gate"], r["model"], r["scenario"], r["staff"])
        if type(r["staff"]) is not int or key not in previous or set(r["arms"]) != set(ARMS):
            raise ValueError("unexpected or duplicate update condition, or missing update arms")
        old = previous.pop(key)
        checks = [(r[k], old[k]) for k in ("sessions", "positives", "nominated", "starts", "expired")]
        checks += [
            (r["review_minutes"], old["mean_review_minutes"]),
            (r["arms"]["frozen"]["catches"], old["catches"]),
            (r["arms"]["frozen"]["unsupported_starts"], old["unsupported_starts"]),
        ]
        if not all(
            all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (v, b))
            and math.isclose(v, b, abs_tol=1e-8, rel_tol=1e-12)
            for v, b in checks
        ):
            raise ValueError("frozen ticket did not reproduce published FIFO")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations", "trace", "azure", "burst-week", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--caches", type=Path, nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    if args.out.exists() or args.jobs < 1:
        p.error("use a fresh output file and positive worker count")
    models = sorted(MODEL_KEYS.values())
    panel = load_panel(args.annotations, args.caches, expected_models=models)
    clock = load_arrivals(args.trace)
    jobs = [
        Job(m, f, t, h, clock, args.azure, args.burst_week)
        for m in models
        for f, (t, h) in enumerate(grouped_folds(encoded(panel, m), 5, SEED))
    ]
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        fits = list(pool.map(run, jobs))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for _, records in fits:
        for r in records:
            groups[tuple(r[k] for k in ("gate", "model", "scenario", "staff"))].append(r)
    results = [aggregate(g) for _, g in sorted(groups.items())]
    verify_previous(results, Path(__file__).with_name("figures"))
    report = dict(
        kind="full_mindguard_ticket_updates_v1",
        dataset_revision=DATASET_REVISION,
        models=models,
        seed=SEED,
        placements=PLACEMENTS,
        folds=5,
        service_minutes=20,
        deadline_minutes=10,
        turn_minutes=2,
        trace_sha256=TRACE_SHA256,
        burst_sha256=BURST_SHA256,
        azure_sha256=AZURE_SHA256,
        coverage=coverage_audit(panel),
        fits=[a for a, _ in fits],
        results=results,
    )
    write_report(report, args.out)
    print(f"Verified {len(results)} paired conditions against both published six-model FIFO baselines.")


def write_report(report: dict[str, Any], out: Path) -> None:
    """Preserve numeric precision; keep each aggregate condition on its own diffable line."""
    header = {k: v for k, v in report.items() if k != "results"}
    prefix = json.dumps(header, indent=2, allow_nan=False).rstrip()
    rows = [json.dumps(r, separators=(",", ":"), allow_nan=False) for r in report["results"]]
    with out.open("x", encoding="utf-8") as f:
        print(prefix[:-2] + ',\n  "results": [\n' + ",\n".join("    " + row for row in rows) + "\n  ]\n}", file=f)


if __name__ == "__main__":
    main()
