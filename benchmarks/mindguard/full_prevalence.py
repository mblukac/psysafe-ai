"""Frozen-gate FIFO sensitivity to synthetic cohort prevalence on the observed arrival clock.

No provider calls or new ratings. This deliberately changes class-prior mix,
not score distributions within each of the two observed conversation strata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from random import Random
from typing import Any

from .fit import grouped_folds
from .full_discrimination import audit
from .full_panel import PanelTurn, load_panel
from .full_replay import calibrated, encoded
from .staffing import TRACE_SHA256, load_arrivals
from .surge_queue import simulate
from .surge_replay import Nomination, attach, nominate, ratio


def paired_mixes(positive: Sequence[bool], n: int, seed: int) -> dict[str, list[int]]:
    """Nested random strata assignments; retain path identity on unchanged slots.

    Draw permutations of whole source paths without replacement per cycle in
    each stratum. Counts within that stratum therefore differ by at most one.
    Rates are exact *counts*, rounded to nearest session, not random Bernoulli draws.
    """
    if any(type(v) is not bool for v in positive) or len(set(positive)) != 2 or n < 1:
        raise ValueError("need both observed classes and positive stream length")
    rng = Random(seed)  # noqa: S311 - reproducible simulation
    slots = list(range(n))
    rng.shuffle(slots)
    sizes = {
        "one_percent": (n + 50) // 100,
        "five_percent": (5 * n + 50) // 100,
        "source_fraction": (2 * n * sum(positive) + len(positive)) // (2 * len(positive)),
    }

    def stream(cls: bool, needed: int) -> list[int]:
        members = [i for i, v in enumerate(positive) if v == cls]
        out: list[int] = []
        while len(out) < needed:
            out.extend(rng.sample(members, len(members)))
        return out

    pos = stream(True, max(sizes.values()))
    neg = stream(False, n - min(sizes.values()))
    result = {}
    for name, k in sizes.items():
        indices = [0] * n
        for rank, slot in enumerate(slots):
            indices[slot] = pos[rank] if rank < k else neg[n - 1 - rank]
        result[name] = indices
    return result


def frozen_paths(panel: list[PanelTurn], replay: dict[str, Any]) -> dict[str, list[Nomination]]:
    # Re-use the audit's checks of reconstructed Brier sums/targets against
    # every held-out primary fold, exact panel membership and tuned gate.
    audit(panel, replay)
    cuts = {(r["model"], r["fold"]): r["cutoff"] for r in replay["training"]}
    frozen = {}
    keys = sorted({r.dialogue_id for r in panel})
    for model in replay["models"]:
        fitted = {}
        for f, (train, held) in enumerate(grouped_folds(encoded(panel, model), 5, replay["seed"])):
            fitted.update({d: nominate(p, cuts[model, f]) for d, p in calibrated(train, held).items()})
        frozen[model] = [fitted[d] for d in keys]
    return frozen


def run(paths: Mapping[str, Sequence[Nomination]], arrivals: list[float], placements: int, seed: int) -> dict[str, Any]:
    if (
        not paths
        or placements < 1
        or len(arrivals) < 2
        or not all(math.isfinite(t) and t >= 0 for t in arrivals)
        or any(a > b for a, b in pairwise(arrivals))
        or arrivals[-1] <= arrivals[0]
    ):
        raise ValueError("need an ordered, nonnegative clock and at least one placement")
    reference = [n.onset_turn for n in next(iter(paths.values()))]
    if any(reference != [n.onset_turn for n in v] for v in paths.values()):
        raise ValueError("model paths misaligned")
    positive = [t is not None for t in reference]
    stats: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for rep in range(placements):
        for mix, indices in paired_mixes(positive, len(arrivals), seed + rep).items():
            for m, trajectory in paths.items():
                cases = attach(trajectory, arrivals, indices, [False] * len(arrivals))
                false = sum(c.onset is None and c.nomination is not None for c in cases)
                unsupported = sum(c.nomination is not None and not c.supported for c in cases)
                for s in (0, 1, 2, 3):
                    r = simulate(
                        cases,
                        reviewers=s or None,
                        service_minutes=20,
                        deadline_minutes=10,
                        policy="fifo",
                        horizon_minutes=arrivals[-1],
                    )
                    row = stats[m, mix, s]
                    for k in (
                        "positives",
                        "nominated",
                        "catches",
                        "starts",
                        "review_minutes",
                        "contested_starts",
                        "unsupported_starts",
                        "misses_queue_delay",
                        "expired",
                    ):
                        row[k] += r[k]
                    row["negative_path_tickets"] += false
                    row["unsupported_tickets"] += unsupported
    results = []
    for (m, mix, s), totals in sorted(stats.items()):
        counts = {k: v / placements for k, v in totals.items()}
        results.append(
            dict(
                model=m,
                mix=mix,
                staff=s,
                **counts,
                realized_prevalence=counts["positives"] / len(arrivals),
                timely_recall=ratio(totals["catches"], totals["positives"]),
                unsupported_ticket_fraction=ratio(totals["unsupported_tickets"], totals["nominated"]),
                negative_path_fraction_of_tickets=ratio(totals["negative_path_tickets"], totals["nominated"]),
                contested_fraction_of_starts=ratio(totals["contested_starts"], totals["starts"]),
            )
        )
    return dict(
        kind="full_prevalence_fifo_sensitivity_v1",
        models=sorted(paths),
        placements=placements,
        seed=seed,
        source_paths=len(positive),
        source_positive_paths=sum(positive),
        arrivals=len(arrivals),
        policy="fifo",
        staff_zero_means="immediate service",
        service_minutes=20,
        deadline_minutes=10,
        turn_minutes=2,
        interpretation="Synthetic source-stratum prior shifts, not assumed clinical prevalence. Within-stratum balanced resampling; aggregates are means across placements, rates from pooled counts. Pre-existing tuned gates; one family-held-out prediction per path, pooled across fits. Not a prospective validation.",
        false_ticket_definitions="negative_path_tickets: nominated paths never observed endorsed; unsupported_tickets: no endorsement at nominated turn, including wrong turns on positive paths. No patient outcomes measured.",
        results=results,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("annotations", "replay", "trace", "out"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--caches", required=True, nargs="+", type=Path)
    parser.add_argument("--placements", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        replay = json.loads(args.replay.read_text(encoding="utf-8"))
        if args.out.exists() or replay["trace_sha256"] != TRACE_SHA256:
            raise ValueError("output exists or replay trace mismatched")
        panel = load_panel(args.annotations, args.caches, expected_models=replay["models"])
        result = run(frozen_paths(panel, replay), load_arrivals(args.trace), args.placements, replay["seed"])
        result["trace_sha256"] = TRACE_SHA256
        with args.out.open("x", encoding="utf-8") as f:
            print(json.dumps(result, indent=2, allow_nan=False), file=f)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ArithmeticError):
        print(
            "Prevalence sensitivity failed: check the frozen primary report, complete panel, clock and new output path.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
