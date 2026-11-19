"""Trace-driven staffing replay. Offline only; stdout contains aggregate results, never text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any

from .__main__ import official_cases
from .data import Turn, attach_responses, complete_dialogues, local_records
from .fit import SINGLES, grouped_folds
from .staffing_fit import matched_cutoff, train_policies
from .staffing_queue import Session, Trajectory, simulate

TRACE_REVISION = "4fc679321b19bdff91d59f765537c86296cfa1fa"
TRACE_SHA256 = "1414173cc5fbf6d2f50ca1345b38ce6aedd0ec92793af5960a4fb646cb243e29"
TRACE_URL = f"https://raw.githubusercontent.com/cathug/crisis-des/{TRACE_REVISION}/real_interarrivals_nov.csv"
POLICIES = ("fixed", "paced", "rl")
ScoreMap = Mapping[tuple[str, int], float]


def load_arrivals(path: Path) -> list[float]:
    """All 9,175 published interarrival intervals in minutes, including the zero interval."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRACE_SHA256:
        raise ValueError("arrival trace differs from the pinned public file")
    gaps = list(map(float, raw.splitlines()))
    if len(gaps) != 9175 or any(not math.isfinite(t) or t < 0 for t in gaps):
        raise ValueError("unexpected interarrival format")
    return list(accumulate(gaps))


@dataclass(frozen=True)
class ScoredPath:
    calibrated: Trajectory
    advantage: Trajectory
    native: Trajectory
    matched_margin: Trajectory


def crossfit(
    rows: list[Turn], providers: Mapping[str, ScoreMap], *, folds: int, seed: int, threshold: float
) -> dict[str, list[ScoredPath]]:
    """Assign each unique, complete source path exactly one held-out fit per model."""
    by_id: dict[str, list[Turn]] = defaultdict(list)
    for row in rows:
        by_id[row.dialogue_id].append(row)
    order = sorted(by_id)
    learned: dict[str, dict[str, ScoredPath]] = {m: {} for m in providers}
    for group in by_id.values():
        group.sort(key=lambda r: r.turn_index)
    for train, test in grouped_folds(rows, folds, seed):
        training = sorted({r.dialogue_id for r in train})
        testing = sorted({r.dialogue_id for r in test})
        for m, scores in providers.items():

            def path(d: str, source: ScoreMap = scores) -> Trajectory:
                return tuple(
                    (r.turn_index, source[(d, r.turn_index)], int(2 * r.votes.count("self_harm") > len(r.votes)))
                    for r in by_id[d]
                )

            train_paths = [path(d) for d in training]
            policy = train_policies(train_paths)
            cut = matched_cutoff(train_paths, policy, threshold)
            for d in testing:
                native = path(d)
                learned[m][d] = ScoredPath(
                    tuple((t, policy.q(s), y) for t, s, y in native),
                    tuple((t, policy.advantage(s, t), y) for t, s, y in native),
                    native,
                    tuple((t, max(-1.0, policy.q(s) - cut), y) for t, s, y in native),
                )
    return {m: [paths[d] for d in order] for m, paths in learned.items()}


def _identity(value: float, _: int) -> float:
    return value


def forecast_diagnostics(paths: Sequence[ScoredPath]) -> dict[str, Any]:
    """Majority-target checks on unique source turns, never the traffic replicas."""
    out = {}
    for name in ("native", "calibrated"):
        pairs = [(s, y) for p in paths for _, s, y in getattr(p, name)]
        n = len(pairs)
        positives = sum(y for _, y in pairs)
        out[name] = {
            "turns": n,
            "positive_turns": positives,
            "mean_score": sum(s for s, _ in pairs) / n,
            "observed_fraction": positives / n,
            "brier": sum((s - y) ** 2 for s, y in pairs) / n,
        }
    return out


def placement(n_paths: int, n_arrivals: int, seed: int) -> list[int]:
    """Permute balanced path repetitions globally (no artificial within-block chronology)."""
    rng = random.Random(seed)  # noqa: S311 - experimental randomization, not cryptography
    full, extra = divmod(n_arrivals, n_paths)
    indices = list(range(n_paths)) * full + rng.sample(range(n_paths), extra)
    rng.shuffle(indices)
    return indices


def replay(
    scores: Mapping[str, list[ScoredPath]],
    arrivals: Sequence[float],
    *,
    staffing: Sequence[int],
    placements: int,
    seed: int,
    service: float,
    turn_spacing: float,
    deadline: float,
    threshold: float,
) -> list[dict[str, Any]]:
    if not scores or not arrivals or not staffing or not 1 <= placements or not 0 <= threshold <= 1:
        raise ValueError("need paths, arrivals, staffing levels, positive placements and a cutoff in [0,1]")
    if len(set(staffing)) != len(staffing) or any(type(k) is not int or k < 0 for k in staffing):
        raise ValueError("staffing levels must be distinct nonnegative integers")
    n = len(next(iter(scores.values())))
    if not n or any(len(paths) != n for paths in scores.values()):
        raise ValueError("all providers must cover the same source trajectories")
    output: dict[tuple[str, int, str], list[dict[str, int | float | None]]] = defaultdict(list)
    for rep in range(placements):
        ix = placement(n, len(arrivals), seed + rep)
        for model, paths in scores.items():
            for name, field, engine in (
                ("fixed", "calibrated", "fixed"),
                ("paced", "calibrated", "paced"),
                ("rl", "advantage", "rl"),
                ("native_fixed", "native", "fixed"),
                ("matched_fixed", "matched_margin", "fixed"),
            ):
                trajectories = [getattr(p, field) for p in paths]
                sessions = [Session(t, trajectories[i]) for t, i in zip(arrivals, ix, strict=True)]
                for k in staffing:
                    output[model, k, name].append(
                        simulate(
                            sessions,
                            reviewers=k,
                            service_minutes=service,
                            turn_minutes=turn_spacing,
                            deadline_minutes=deadline,
                            policy=engine,
                            metric=_identity,
                            threshold=0 if name == "matched_fixed" else threshold,
                        )
                    )
    results = []
    for (model, k, name), runs in output.items():
        keys = (
            "arrivals",
            "positives",
            "catches",
            "catches_by_onset_deadline",
            "alerts",
            "reviews",
            "misses_no_alert",
            "misses_wrong_turn",
            "misses_late",
            "expired",
            "unsupported_reviews",
            "review_minutes",
        )
        totals = {key: sum(float(r[key] or 0) for r in runs) for key in keys}
        pos = totals["positives"]
        recall = totals["catches"] / pos if pos else None
        rates = [float(r["recall"]) for r in runs if r["recall"] is not None]
        results.append(
            {
                "model": model,
                "reviewers": k,
                "policy": name,
                "recall": recall,
                "recall_by_onset_deadline": totals["catches_by_onset_deadline"] / pos if pos else None,
                "placement_min_max": [min(rates), max(rates)] if rates else None,
                "mean_per_trace": {key: v / placements for key, v in totals.items()},
                "start_delay_minutes": (
                    sum(float(r["mean_wait_minutes"] or 0) * float(r["reviews"] or 0) for r in runs) / totals["reviews"]
                    if totals["reviews"]
                    else None
                ),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("annotations", "responses", "trace"):
        parser.add_argument("--" + opt, required=True, type=Path)
    parser.add_argument("--staffing", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--placements", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--service-minutes", type=float, default=20)
    parser.add_argument("--turn-minutes", type=float, default=2)
    parser.add_argument("--deadline-minutes", type=float, default=10)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        arrivals = load_arrivals(args.trace)
        cases = official_cases(args.annotations)
        complete = complete_dialogues(cases)
        all_rows = attach_responses(cases, local_records(args.responses))
        rows = [r for r in all_rows if r.dialogue_id in complete]
        providers: dict[str, ScoreMap] = {
            m: {(r.dialogue_id, r.turn_index): r.scores[j] for r in rows} for j, m in enumerate(SINGLES)
        }
        scores = crossfit(rows, providers, folds=5, seed=args.seed, threshold=args.threshold)
        results = replay(
            scores,
            arrivals,
            staffing=args.staffing,
            placements=args.placements,
            seed=args.seed,
            service=args.service_minutes,
            turn_spacing=args.turn_minutes,
            deadline=args.deadline_minutes,
            threshold=args.threshold,
        )
        report = {
            "kind": "trace_driven_counterfactual_staffing_not_production_recall",
            "trace": {"url": TRACE_URL, "sha256": TRACE_SHA256, "arrivals": len(arrivals), "minutes": arrivals[-1]},
            "source": {
                "dialogues": len(complete),
                "turns": len(rows),
                "excluded_incomplete": 67 - len(complete),
                "positive_dialogues": len({r.dialogue_id for r in rows if r.votes.count("self_harm") >= 2}),
                "opening_families": len({r.group_id for r in rows}),
                "folds": 5,
            },
            "scenario": {
                "placements": args.placements,
                "seed": args.seed,
                "service_minutes": args.service_minutes,
                "turn_minutes": args.turn_minutes,
                "start_deadline_minutes": args.deadline_minutes,
                "base_threshold": args.threshold,
                "reviewers": args.staffing,
            },
            "forecast_diagnostics": {m: forecast_diagnostics(paths) for m, paths in scores.items()},
            "results": results,
        }
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, ArithmeticError) as exc:
        # Parsing a private input may throw an exception containing sensitive text.
        print(
            f"Replay failed ({type(exc).__name__}); check pinned inputs, cache coverage and scenario options.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
