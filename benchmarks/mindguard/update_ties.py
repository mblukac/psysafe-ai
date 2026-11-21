"""Post-hoc diagnostic only: does changing the exact-tie convention explain rescues?

Repeat the observed-clock, one-reviewer FIFO condition on the original eight
placements. Keep both primary learned decisions and gates frozen. For the
exploratory rule choose the *last*, rather than first, maximum of calibrated q
over visible turns. No approximate ties, new fits of the update rule, or costs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .fit import _clipped, grouped_folds
from .full_panel import MODEL_KEYS, PanelTurn, load_panel
from .full_replay import calibrated, encoded
from .staffing import load_arrivals
from .surge_queue import simulate
from .surge_replay import attach, nominate
from .surge_stream import assign_paths, make_clock
from .update_pair import _cell
from .update_pilot import PLACEMENTS, SEED, families


def choices(scores: Sequence[float], switches: Sequence[bool]) -> tuple[int, int, int]:
    """Earliest overall max, latest overall max, frozen trained paired choice."""
    if not scores or len(switches) != 3 or not all(math.isfinite(q) and 0 <= q <= 1 for q in scores):
        raise ValueError("invalid visible scores or paired decision table")
    first = max(range(len(scores)), key=scores.__getitem__)
    last = max(i for i, q in enumerate(scores) if q == scores[first])
    challenger = max(range(1, len(scores)), key=scores.__getitem__) if len(scores) > 1 else 0
    pair = challenger if switches[_cell(scores[0], scores[challenger])] else 0
    return first, last, pair


def evaluate(p: list[PanelTurn], clock: list[float], main: dict[str, Any], model: str) -> list[dict[str, Any]]:
    timing = make_clock(clock)
    # Exact decimal sums of the wire components; don't mistake float addition
    # artifacts or clipping in the calibration model for genuine model ties.
    native = {
        (r.dialogue_id, r.turn_index): sum(
            (Decimal(str(r.scores[model][tag])) for tag in ("self_harm", "harm_others")), Decimal(0)
        )
        for r in p
    }
    results = []
    for gate in ("tuned", "liberal"):
        counts: Counter[str] = Counter()
        ids_by: dict[str, set[str]] = defaultdict(set)
        fams_by: dict[str, set[str]] = defaultdict(set)
        total_w = 0
        for fold, (train, held) in enumerate(grouped_folds(encoded(p, model), 5, SEED)):
            foldfit = next(v for v in main["fits"] if (v["model"], v["fold"]) == (model, fold))
            cutoff = foldfit["cutoffs"][gate]
            switches = foldfit["paired_replace_low_mid_high"][gate]
            trajectories = calibrated(train, held)
            ids = sorted(trajectories)
            fams = families(held)
            paths = [trajectories[k] for k in ids]
            nominations = [nominate(z, cutoff) for z in paths]
            source = [n.onset_turn is not None for n in nominations]
            displays = []
            for path, nom in zip(paths, nominations, strict=True):
                turns = [r for r in path if nom.turn is not None and r[0] >= nom.turn]
                prefixes = []
                for i in range(len(turns)):
                    q = [r[1] for r in turns[: i + 1]]
                    first, last, pair = choices(q, switches)
                    compare = "equal" if q[0] == q[pair] else "higher" if q[pair] > q[0] else "lower"
                    band = ("low", "middle", "high")[_cell(q[0], q[pair])]
                    prefixes.append((first, last, pair, compare, band))
                displays.append((turns, prefixes))
            for rep in range(PLACEMENTS):
                order, _ = assign_paths(source, timing, seed=SEED + 1009 * fold + rep)
                cases = attach(nominations, timing.arrivals, order, timing.peak_mask)
                starts: dict[int, float] = {}
                base = simulate(cases, reviewers=1, start_times=starts, horizon_minutes=clock[-1])
                w = len(ids)
                total_w += w
                counts["positives"] += base["positives"] * w
                counts["starts"] += len(starts) * w
                for sid, at in starts.items():
                    case, k = cases[sid], order[sid]
                    turns, prefixes = displays[k]
                    i = 0
                    while i + 1 < len(turns) and case.arrival + 2 * (turns[i + 1][0] - 1) <= at:
                        i += 1
                    first, last, pair, compare, band = prefixes[i]
                    counts["pair_last_different_selection"] += (pair != last) * w
                    if case.onset is None or not case.onset <= at <= case.onset + 10:
                        continue
                    y0 = turns[0][2]
                    for arm, j in (("frozen", 0), ("max_score", first), ("last_max", last), ("paired_bayes", pair)):
                        y = turns[j][2]
                        counts[arm] += y * w
                        if y != y0:
                            outcome = "rescue" if y else "spoil"
                            keys = [arm + "_" + outcome]
                            if arm == "paired_bayes":
                                keys.append(f"{arm}_{outcome}_{compare}_{band}")
                                if compare == "equal":
                                    a, b = (native[ids[k], turns[pos][0]] for pos in (0, j))
                                    if a == b:
                                        origin = "native_one" if a == 1 else "native_zero" if a == 0 else "native_other"
                                    else:
                                        origin = (
                                            "clip_only"
                                            if _clipped(min(1, float(a))) == _clipped(min(1, float(b)))
                                            else "indistinguishable_after_calibration"
                                        )
                                    keys.append(f"{arm}_{outcome}_equal_{origin}")
                            for key in keys:
                                counts[key] += w
                                ids_by[key].add(ids[k])
                                fams_by[key].add(fams[ids[k]])
                    counts["pair_last_different_timely_outcome"] += (turns[pair][2] != turns[last][2]) * w
        ref = next(
            r
            for r in main["results"]
            if (r["model"], r["gate"], r["scenario"], r["staff"]) == (model, gate, "observed", 1)
        )
        for arm in ("frozen", "max_score", "paired_bayes"):
            if not math.isclose(counts[arm] / total_w, ref["arms"][arm]["catches"], abs_tol=1e-9):
                raise ValueError("post-hoc replay does not reproduce primary arm")
        if not math.isclose(counts["starts"] / total_w, ref["starts"], abs_tol=1e-9):
            raise ValueError("physical starts differ")
        results.append(
            dict(
                model=model,
                gate=gate,
                mean_counts={k: v / total_w for k, v in counts.items()},
                last_max_delta=(counts["last_max"] - counts["frozen"]) / counts["positives"],
                affected_paths={k: len(v) for k, v in ids_by.items()},
                affected_families={k: len(v) for k, v in fams_by.items()},
            )
        )
    return results


def main() -> None:
    arg = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "trace", "primary", "out"):
        arg.add_argument("--" + name, type=Path, required=True)
    arg.add_argument("--caches", nargs="+", type=Path, required=True)
    a = arg.parse_args()
    if a.out.exists():
        arg.error("need fresh output")
    panel = load_panel(a.annotations, a.caches, expected_models=sorted(MODEL_KEYS.values()))
    primary = json.loads(a.primary.read_text(encoding="utf-8"))
    if (
        primary["kind"] != "full_mindguard_ticket_updates_v1"
        or primary["seed"] != SEED
        or primary["placements"] != PLACEMENTS
    ):
        arg.error("mismatched primary report")
    clock = load_arrivals(a.trace)
    rows = [r for m in sorted(MODEL_KEYS.values()) for r in evaluate(panel, clock, primary, m)]
    with a.out.open("x", encoding="utf-8") as f:
        print(
            json.dumps(dict(kind="posthoc_latest_on_exact_score_ties_observed_one_reviewer", rows=rows), indent=2),
            file=f,
        )


if __name__ == "__main__":
    main()
