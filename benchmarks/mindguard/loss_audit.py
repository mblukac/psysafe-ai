"""Account for every observed-positive replay episode using cached, aggregate sufficient statistics.

No model calls, raw conversations, individual predictions or inferred labels. The
FIFO and presentation reports share the same stream, folds, weights and starts.
Subtract *counts* on those shared episodes; do not subtract unweighted rates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION
from .full_panel import MODEL_KEYS
from .staffing import TRACE_SHA256
from .surge_stream import AZURE_SHA256, BURST_SHA256
from .update_pilot import PLACEMENTS, SEED, verify_previous, write_report
from .update_queue import ARMS

AXES = ("gate", "model", "scenario", "staff")
PARTITION = (
    "no_ticket",
    "flagged_never_served",
    "pre_onset_pickup",
    "late_pickup",
    "timely_no_supported_turn_available",
    "timely_stale",
    "caught",
)
TOL = 1e-8


def _count(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < -TOL:
        raise ValueError("expected a finite nonnegative episode count")
    return max(0.0, float(value))


def _equal(a: float, b: float, message: str) -> None:
    if not math.isclose(a, b, abs_tol=TOL, rel_tol=1e-12):
        raise ValueError(message)


def decompose(updates: dict[str, Any], fifo: dict[str, Any], immediate: dict[str, Any]) -> dict[str, Any]:
    """Partition positives by first-access outcome, then by evidence at that same pickup.

    Ticket expiry is anchored to release, not to the independently observed need.
    A positive case whose ticket expires is therefore 'flagged_never_served', not
    necessarily a late start. 'Stale' is restricted to misses recoverable by
    selecting another already-visible, rated endpoint since admission.
    """
    n = _count(updates["positives"])
    if immediate["staff"] != 0 or any(updates[k] != immediate[k] for k in AXES if k != "staff"):
        raise ValueError("need the immediate-service control from the same stream")
    for k in ("sessions", "positives", "nominated"):
        _equal(_count(updates[k]), _count(immediate[k]), "different immediate-control admissions or workload")
    _equal(_count(immediate["starts"]), _count(immediate["nominated"]), "immediate service dropped tickets")
    # With immediate service, start equals first flag. These late first
    # flags must be either unserved or served late under finite staffing.
    late_flags = _count(immediate["after_deadline_starts"])
    no_ticket = _count(fifo["misses_no_alert"])
    early = _count(updates["before_onset_starts"])
    late = _count(updates["after_deadline_starts"])
    timely = _count(updates["timely_positive_starts"])
    visible = _count(updates["arms"]["visible_oracle"]["catches"])
    frozen = _count(updates["arms"]["frozen"]["catches"])
    _equal(
        n,
        sum(_count(fifo[key]) for key in ("misses_wrong_turn", "misses_nomination_late", "misses_queue_delay"))
        + no_ticket
        + frozen,
        "cached source failure partition does not reconcile",
    )
    unserved = _count(n - no_ticket - early - late - timely)
    no_timely_pickup_after_timely_flag = _count(unserved + late - late_flags)
    if unserved > _count(updates["expired"]) + TOL or early + late + timely > _count(updates["starts"]) + TOL:
        raise ValueError("positive starts or expirations exceed all-case totals")
    no_evidence = _count(timely - visible)
    stale = _count(visible - frozen)
    partition = dict(zip(PARTITION, (no_ticket, unserved, early, late, no_evidence, stale, frozen), strict=True))
    _equal(n, sum(partition.values()), "positive cases do not partition")
    _equal(updates["starts"] + updates["expired"], updates["nominated"], "tickets do not partition")
    _equal(updates["starts"] * 20, updates["review_minutes"], "unexpected work per start")
    comparisons = {}
    for name in ARMS:
        a = updates["arms"][name]
        hit, rescued, spoiled = (_count(a[k]) for k in ("catches", "rescued", "spoiled"))
        _equal(hit - frozen, rescued - spoiled, "unpaired presentation contrast")
        if hit > visible + TOL or rescued > stale + TOL or spoiled > frozen + TOL:
            raise ValueError("presentation exceeds evidence bound or observed frozen hits")
        comparisons[name] = dict(caught=hit, gained=rescued, lost=spoiled, net=rescued - spoiled)
    return {
        **{k: updates[k] for k in AXES},
        **{k: _count(updates[k]) for k in ("sessions", "positives", "nominated", "starts", "review_minutes")},
        **partition,
        "late_first_flag": late_flags,
        "unserved_or_late_despite_flag_by_deadline": no_timely_pickup_after_timely_flag,
        "arms": comparisons,
    }


def audit(figures: Path) -> dict[str, Any]:
    src = json.loads((figures / "ticket-updates.json").read_text())
    meta = {
        "dataset_revision": DATASET_REVISION,
        "models": sorted(MODEL_KEYS.values()),
        "folds": 5,
        "placements": PLACEMENTS,
        "seed": SEED,
        "service_minutes": 20,
        "turn_minutes": 2,
        "deadline_minutes": 10,
        "trace_sha256": TRACE_SHA256,
        "azure_sha256": AZURE_SHA256,
        "burst_sha256": BURST_SHA256,
    }
    if any(src.get(k) != v for k, v in (meta | {"kind": "full_mindguard_ticket_updates_v1"}).items()):
        raise ValueError("unexpected update cache or replay settings")
    rows = src["results"]
    # Also enforces full 6 x 2 x 9 x 4 coverage, no duplicate keys, shared
    # clock, cohort, minutes, and matching FIFO counts at every condition.
    verify_previous(rows, figures)
    immediate = {(r["gate"], r["model"], r["scenario"]): r for r in rows if r["staff"] == 0}
    baseline = {}
    for gate in ("tuned", "liberal"):
        old = json.loads((figures / f"full-panel-{gate}.json").read_text())
        for raw in old["rows"]:
            r = dict(zip(old["columns"], raw, strict=True))
            if r["policy"] == "fifo":
                baseline[gate, r["model"], r["scenario"], r["staff"]] = r
    return {
        "kind": "mindguard_fifo_loss_partition_v1",
        **meta,
        "unit": "path-weighted mean replay episodes per stream; staff=0 is immediate service, not zero reviewers",
        "partition": PARTITION,
        "evidence_oracle": "best observable prefix endpoint since admission at those exact starts; not a policy",
        "results": [
            decompose(r, baseline[tuple(r[k] for k in AXES)], immediate[r["gate"], r["model"], r["scenario"]])
            for r in rows
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", type=Path, default=Path(__file__).with_name("figures"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.figures)
    write_report(report, args.out)
    print(f"Verified {len(report['results'])} FIFO loss partitions and same-start comparisons.")


if __name__ == "__main__":
    main()
