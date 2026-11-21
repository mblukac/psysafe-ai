"""Offline paired roster replay at equal scheduled coverage *and* actual busy minutes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION
from .fit import grouped_folds
from .full_panel import MODEL_KEYS, coverage_audit, load_panel
from .full_replay import calibrated, encoded
from .full_shifts import (
    DAY,
    DEADLINE,
    END,
    FIXED,
    SCORE_FROM,
    SCORE_TO,
    SERVICE,
    SHIFT,
    TRAIN_END,
    next_shift,
    peak_start,
)
from .staffing import TRACE_SHA256, load_arrivals, placement
from .surge_queue import Case
from .surge_replay import attach, nominate

FIGURES = Path(__file__).with_name("figures")


def fifo_trace(cases: Sequence[Case], offset: int, begin: int, switch: int, stop: int) -> list[tuple[int, float]]:
    """Common warm-up roster; after switch apply the repeating roster, including midnight overlap.

    Baseline worker is always on. Both workers finish previously started jobs. Ties go to the
    baseline worker, nomination ties to arrival ordinal. Never reserve a worker for an expired job.
    Only operational nomination times enter dispatch; replay is work-conserving FIFO.
    """
    if not 0 <= begin <= switch < stop or not 0 <= offset < DAY:
        raise ValueError("invalid roster interval")
    free_base = free_extra = float(begin)
    starts = []
    for release, sid in sorted((c.nomination, i) for i, c in enumerate(cases) if c.nomination is not None):
        base, ready = max(free_base, release), max(free_extra, release)
        extra = next_shift(ready, FIXED)[0] if ready < switch else math.inf
        if extra >= switch:
            extra = next_shift(max(ready, switch), offset)[0]
        at = min(base, extra)
        if at >= stop or at > release + DEADLINE:
            continue
        if base <= extra:
            free_base = at + SERVICE
        else:
            free_extra = at + SERVICE
        starts.append((sid, at))
    if any(b[1] < a[1] for a, b in pairwise(starts)):
        raise RuntimeError("nonchronological dispatch")
    return starts


def paired(
    cases: Sequence[Case],
    offset: int,
    *,
    begin: int = TRAIN_END,
    switch: int = SCORE_FROM,
    close: int = SCORE_TO,
    stop: int = END,
) -> dict[str, int]:
    """First K actual starts, not K positive cases or only in-period service starts.

    All admissions close at the end of the scored arrival cohort. The final day permits
    service of outstanding sessions but no new arrivals. A warm-up arrival served after
    the switch costs the same as a scored case and earns no reward. Later starts dropped
    by the cap earn no reward, regardless of when the corresponding case arrived.
    """
    if not begin <= switch < close <= stop or any(not begin <= c.arrival < close for c in cases):
        raise ValueError("session outside enrollment interval")
    arms = {arm: fifo_trace(cases, off, begin, switch, stop) for arm, off in (("fixed", FIXED), ("peak", offset))}
    if [s for s in arms["fixed"] if s[1] < switch] != [s for s in arms["peak"] if s[1] < switch]:
        raise RuntimeError("different warm-up trajectories")
    traces = {arm: [s for s in starts if s[1] >= switch] for arm, starts in arms.items()}
    k = min(map(len, traces.values()))
    out = dict(
        sessions=sum(c.arrival >= switch for c in cases),
        positives=sum(c.arrival >= switch and c.onset is not None for c in cases),
        starts_per_arm=k,
        minutes_per_arm=k * SERVICE,
    )
    for arm, trace in traces.items():
        selected = trace[:k]
        if len(selected) != k:
            raise RuntimeError("unequal capped work")
        out[arm + "_uncapped_minutes"] = SERVICE * len(trace)
        out[arm + "_matches"] = sum(
            c.arrival >= switch and c.onset is not None and c.onset <= at <= c.onset + DEADLINE and c.supported
            for sid, at in selected
            for c in (cases[sid],)
        )
        out[arm + "_warm_carry_starts"] = sum(cases[sid].arrival < switch for sid, _ in selected)
        out[arm + "_drain_starts"] = sum(at >= close for _, at in selected)
    return out


def summarize(rows: list[tuple[int, int, int, dict[str, int]]]) -> dict[str, Any]:
    if len({(f, p) for f, p, _, _ in rows}) != len(rows) or not rows or any(w <= 0 for _, _, w, _ in rows):
        raise ValueError("invalid blocks or weights")
    weight = sum(w for _, _, w, _ in rows)
    means = {key: sum(w * d[key] for _, _, w, d in rows) / weight for key in rows[0][3]}
    if means["positives"] <= 0 or not math.isclose(means["minutes_per_arm"], SERVICE * means["starts_per_arm"]):
        raise ValueError("invalid pooled workload or cohort")

    def delta(blocks: list[tuple[int, int, int, dict[str, int]]]) -> float:
        positives = sum(w * d["positives"] for _, _, w, d in blocks)
        return sum(w * (d["peak_matches"] - d["fixed_matches"]) for _, _, w, d in blocks) / positives

    folds = [delta([v for v in rows if v[0] != f]) for f in sorted({v[0] for v in rows})]
    placements = [delta([v for v in rows if v[1] == p]) for p in sorted({v[1] for v in rows})]
    return dict(
        **means,
        fixed_recall=means["fixed_matches"] / means["positives"],
        peak_recall=means["peak_matches"] / means["positives"],
        delta=delta(rows),
        omit_fold_delta_range=[min(folds), max(folds)],
        placement_delta_range=[min(placements), max(placements)],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("annotations", "trace", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--caches", type=Path, nargs="+", required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("need fresh output path")
    previous = json.loads((FIGURES / "full-panel-shifts.json").read_text())
    baseline = json.loads((FIGURES / "ticket-updates.json").read_text())
    models = sorted(MODEL_KEYS.values())
    if not (
        previous["kind"] == "paired_equal_hours_shift_sensitivity_v1"
        and previous["trace_sha256"] == TRACE_SHA256
        and previous["seed"] == 2026
        and previous["placements"] == 8
        and previous["service_minutes"] == SERVICE
        and previous["start_deadline_minutes"] == DEADLINE
        and previous["scored_days"] == [16, 29]
        and baseline["kind"] == "full_mindguard_ticket_updates_v1"
        and baseline["dataset_revision"] == DATASET_REVISION
        and baseline["trace_sha256"] == TRACE_SHA256
        and sorted(baseline["models"]) == models
    ):
        raise ValueError("unrecognized previous experiment")
    cut = {(v["model"], v["fold"]): v["cutoffs"] for v in previous["training"]}
    if len(previous["training"]) != 5 * len(models) or set(cut) != {(m, f) for m in models for f in range(5)}:
        raise ValueError("incomplete training gate grid")
    panel = load_panel(args.annotations, args.caches, expected_models=models)
    if coverage_audit(panel) != baseline["coverage"]:
        raise ValueError("rated release coverage differs from published panel")
    fits = {(v["model"], v["fold"]): v for v in baseline["fits"]}
    if len(fits) != len(baseline["fits"]) or set(fits) != set(cut):
        raise ValueError("incomplete held-out forecast audit")
    arrivals = load_arrivals(args.trace)
    offset = peak_start([t for t in arrivals if t < TRAIN_END])
    idx = [i for i, t in enumerate(arrivals) if TRAIN_END <= t < SCORE_TO]
    times = [arrivals[i] for i in idx]
    cohort = sum(SCORE_FROM <= t < SCORE_TO for t in times)
    if cohort != previous["scored_arrivals"] or offset != 60 * previous["predicted_start_hours"]:
        raise ValueError("clock or roster prediction differs from prior study")
    blocks: dict[tuple[str, str], list[tuple[int, int, int, dict[str, int]]]] = defaultdict(list)
    for model in models:
        for fold, (train, held) in enumerate(grouped_folds(encoded(panel, model), 5, 2026)):
            fitted = calibrated(train, held)
            paths = [fitted[key] for key in sorted(fitted)]
            fit = fits[model, fold]
            err = sum((q - y) ** 2 for path in paths for _, q, y in path)
            if (
                fit["held_paths"] != len(paths)
                or fit["held_turns"] != len(held)
                or fit["cutoffs"] != {"tuned": cut[model, fold]["frozen_primary"], "liberal": 0.10}
                or set(cut[model, fold]) != {"frozen_primary", "early_clock_only"}
                or not math.isclose(err, fit["single_brier_sum"], rel_tol=1e-12, abs_tol=1e-9)
            ):
                raise ValueError("cannot reproduce held-out forecasts or gates")
            for gate, threshold in (cut[model, fold] | {"liberal": 0.1}).items():
                noms = [nominate(path, threshold) for path in paths]
                for rep in range(8):
                    draw = placement(len(paths), len(arrivals), 2026 + 1009 * fold + rep)
                    cases = attach(noms, times, [draw[i] for i in idx], [False] * len(idx))
                    result = paired(cases, offset)
                    if result["sessions"] != cohort:
                        raise RuntimeError("changed scored cohort")
                    blocks[model, gate].append((fold, rep, len(paths), result))
    prior = {(r["model"], r["gate"]): r["arms"]["fixed"] for r in previous["results"]}
    results = []
    for (model, gate), values in sorted(blocks.items()):
        if len(values) != 40:
            raise RuntimeError("incomplete paired folds x placements")
        summary = summarize(values)
        ref = prior[model, "frozen_primary" if gate == "liberal" else gate]
        if any(not math.isclose(summary[k], ref[k], abs_tol=1e-9) for k in ("sessions", "positives")):
            raise RuntimeError("mismatched prior scored denominator")
        results.append(dict(model=model, gate=gate, **summary))
    output = dict(
        kind="mindguard_matched_busy_minutes_roster_v1",
        revision=DATASET_REVISION,
        trace_sha256=TRACE_SHA256,
        seed=2026,
        folds=5,
        placements=8,
        warmup_day=15,
        scored_arrival_days=[16, 29],
        drain_day=30,
        review_minutes_per_start=SERVICE,
        deadline_from_first_endorsed_turn_minutes=DEADLINE,
        fixed_hour=FIXED // 60,
        peak_hour=offset // 60,
        scored_arrivals=cohort,
        scheduled_minutes_per_arm_scoring_window=(SCORE_TO - SCORE_FROM) * (1 + SHIFT / DAY),
        scheduled_minutes_per_arm_including_drain=(END - SCORE_FROM) * (1 + SHIFT / DAY),
        accounting="Common 09-17 warmup. Fixed vs predicted extra eight-hour shift with one base reviewer. "
        "Post-switch start order includes warm carry-ins, safe cases and service during drain; "
        "no new day-30 arrivals. Truncate each paired stream at smaller realized number of starts, "
        "charge full 20 minutes even if service crosses a boundary. Only timely, supported frozen "
        "reviews of days 16-29 arrivals earn credit. Paid overtime is not work-matched.",
        results=results,
    )
    with args.out.open("x", encoding="utf-8") as f:
        print(json.dumps(output, indent=2, allow_nan=False), file=f)


if __name__ == "__main__":
    main()
