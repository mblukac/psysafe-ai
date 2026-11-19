"""Optional offline four-model staffing audit; aggregate unique source turns only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from statistics import correlation
from typing import Any

from .__main__ import official_cases
from .data import attach_responses, complete_dialogues, local_records
from .fit import SINGLES
from .luna import load_luna_scores
from .staffing import ScoredPath, crossfit

MODELS = (*SINGLES, "luna")


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2 or min(a) == max(a) or min(b) == max(b):
        return None
    return correlation(a, b)


def audit(
    scores: Mapping[str, Sequence[ScoredPath]], *, threshold: float = 0.10, groups: Sequence[str] | None = None
) -> dict[str, Any]:
    """Use aligned crossfit() output, not traffic replicas or vote-share forecasts.

    Native and OOF-calibrated signed residuals are score minus observed majority.
    Turn and uncongested first-ticket dialogue miss overlap have distinct
    denominators. No staffing/queue simulation is performed. Shared targets and case difficulty
    can induce correlation; these are not errors against future patient harm.
    Optional groups are opening-family IDs aligned with crossfit's sorted unique
    dialogue IDs; deletion sensitivity freezes held-out predictions (not a CI).
    """
    if set(scores) != set(MODELS):
        raise ValueError("expected exactly jev, gemini, liquid and luna")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")
    n = len(scores[MODELS[0]])
    if not n or any(len(paths) != n for paths in scores.values()):
        raise ValueError("models must have the same nonempty aligned source paths")
    if groups is not None and (len(groups) != n or any(not isinstance(g, str) or not g for g in groups)):
        raise ValueError("need one nonempty opening-family ID per source dialogue")
    reference = [[(t, y) for t, _, y in p.native] for p in scores[MODELS[0]]]
    if any(
        not path or any(t != i or type(y) is not int or y not in (0, 1) for i, (t, y) in enumerate(path, 1))
        for path in reference
    ):
        raise ValueError("need complete consecutive paths with binary majority targets")
    labels = [y for path in reference for _, y in path]
    report: dict[str, Any] = {
        "kind": "four_model_unique_source_turn_dependence_not_queue_recall",
        "target": "observed three-clinician self_harm majority, not vote share or patient harm",
        "source_dialogues": n,
        "source_turns": len(labels),
        "positive_turns": sum(labels),
        "threshold": threshold,
        "alert_rule": "score >= threshold; miss means score < threshold on a positive turn",
        "uncertainty": "descriptive point estimates only; turns within dialogues are dependent",
        "dialogue_miss_overlap": {
            "scope": "unique positive source dialogues; uncongested fixed first-ticket lock, not six-reviewer replay",
            "rule": "first threshold crossing locks the only ticket; catch iff that selected turn has majority support",
            "denominator_positive_dialogues": sum(any(y for _, y in path) for path in reference),
        },
    }
    for field in ("native", "calibrated"):
        columns = {}
        for model in MODELS:
            trajectories = [getattr(p, field) for p in scores[model]]
            if [[(t, y) for t, _, y in p] for p in trajectories] != reference:
                raise ValueError("unaligned model paths, turns or majority targets")
            column = [s for p in trajectories for _, s, _ in p]
            if any(isinstance(s, bool) or not math.isfinite(s) or not 0 <= s <= 1 for s in column):
                raise ValueError("expected finite probability scores")
            columns[model] = column
        residuals = {m: [s - y for s, y in zip(c, labels, strict=True)] for m, c in columns.items()}
        misses = {m: [s < threshold for s, y in zip(c, labels, strict=True) if y] for m, c in columns.items()}
        pairs = {}
        for a, b in combinations(MODELS, 2):
            cells = [0] * 4
            for x, y in zip(misses[a], misses[b], strict=True):
                cells[2 * x + y] += 1
            neither, second, first, both = cells
            union = first + second + both
            denom = math.sqrt((first + both) * (second + both) * (neither + first) * (neither + second))
            pairs[f"{a}__{b}"] = {
                "signed_majority_residual_pearson_all_turns": _pearson(residuals[a], residuals[b]),
                "positive_turn_misses": {
                    "denominator_positive_turns": sum(labels),
                    "neither": neither,
                    "first_only": first,
                    "second_only": second,
                    "both": both,
                    "both_fraction_of_positive_turns": both / sum(labels) if sum(labels) else None,
                    "jaccard": both / union if union else None,
                    "phi": (both * neither - first * second) / denom if denom else None,
                },
            }
        report[field] = {
            "pairs": pairs,
            "positive_turn_misses_per_model": {m: sum(v) for m, v in misses.items()},
            "positive_turns_all_four_miss": sum(all(v) for v in zip(*(misses[m] for m in MODELS), strict=True)),
            "positive_turns_any_model_misses": sum(any(v) for v in zip(*(misses[m] for m in MODELS), strict=True)),
        }
        if field == "calibrated" and groups is not None:
            turn_groups = [g for g, path in zip(groups, reference, strict=True) for _ in path]
            deleted = sorted(set(groups))
            sensitivity = {}
            for a, b in combinations(MODELS, 2):
                values = []
                for group in deleted:
                    keep = [i for i, g in enumerate(turn_groups) if g != group]
                    value = _pearson([residuals[a][i] for i in keep], [residuals[b][i] for i in keep])
                    if value is not None:
                        values.append(value)
                sensitivity[f"{a}__{b}"] = {
                    "min_max_if_defined": [min(values), max(values)] if values else None,
                    "defined_deletions": len(values),
                }
            report["calibrated_residual_family_deletion_sensitivity"] = {
                "interpretation": "descriptive leave-one-opening-family-out range, NOT a confidence interval; predictions fixed, no refits",
                "families": len(deleted),
                "pairs": sensitivity,
            }
        dialogue_misses = {}
        for model in MODELS:
            # Select using scores alone. Join clinician labels only afterwards.
            selected = [
                next((i for i, (_, s, _) in enumerate(getattr(p, field)) if s >= threshold), None)
                for p in scores[model]
            ]
            dialogue_misses[model] = [
                index is None or not labels[index][1]
                for index, labels in zip(selected, reference, strict=True)
                if any(y for _, y in labels)
            ]
        dialogue_pairs = {}
        for a, b in combinations(MODELS, 2):
            cells = [0] * 4
            for x, y in zip(dialogue_misses[a], dialogue_misses[b], strict=True):
                cells[2 * x + y] += 1
            dialogue_pairs[f"{a}__{b}"] = dict(
                zip(("neither", "second_only", "first_only", "both"), cells, strict=True)
            )
        report["dialogue_miss_overlap"][field] = {
            "pairs": dialogue_pairs,
            "misses_per_model": {m: sum(v) for m, v in dialogue_misses.items()},
            "positive_dialogues_all_four_miss": sum(
                all(v) for v in zip(*(dialogue_misses[m] for m in MODELS), strict=True)
            ),
            "positive_dialogues_any_model_misses": sum(
                any(v) for v in zip(*(dialogue_misses[m] for m in MODELS), strict=True)
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("annotations", "responses", "luna-responses"):
        parser.add_argument("--" + opt, required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        cases = official_cases(args.annotations)
        complete = complete_dialogues(cases)
        rows = [r for r in attach_responses(cases, local_records(args.responses)) if r.dialogue_id in complete]
        providers = {m: {(r.dialogue_id, r.turn_index): r.scores[j] for r in rows} for j, m in enumerate(SINGLES)}
        providers["luna"] = load_luna_scores(cases, args.luna_responses)
        scores = crossfit(rows, providers, folds=args.folds, seed=args.seed, threshold=args.threshold)
        families = {r.dialogue_id: r.group_id or r.dialogue_id for r in rows}
        report = audit(scores, threshold=args.threshold, groups=[families[d] for d in sorted(families)])
        report.update(folds=args.folds, seed=args.seed, split_unit="shared-opening family")
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, ArithmeticError) as exc:
        print(f"Staffing dependence audit failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
