"""Offline source-cohort diagnostics for the full either-risk replay.

Use the replay's five opening-family folds, single-score calibration and frozen
training cutoffs. No synthetic arrivals, provider calls, per-case exports or
threshold selection on evaluation labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, groupby
from pathlib import Path
from typing import Any

from .fit import grouped_folds
from .full_panel import PanelTurn, load_panel
from .full_replay import calibrated, encoded
from .surge_replay import Nomination, nominate


def average_precision(pairs: Sequence[tuple[float, int]]) -> float | None:
    """Non-interpolated AP: sum of precision at each *distinct* score times delta recall."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 or y not in (0, 1) for p, y in pairs):
        raise ValueError("invalid score or label")
    positives = sum(y for _, y in pairs)
    if not positives:
        return None
    tp = n = 0
    result = 0.0
    for _, grp in groupby(sorted(pairs, key=lambda x: -x[0]), key=lambda x: x[0]):
        labels = [y for _, y in grp]
        gained = sum(labels)
        tp += gained
        n += len(labels)
        result += (gained / positives) * (tp / n)
    return result


@dataclass(frozen=True)
class SourceOutcome:
    family: str
    positive: bool
    ticket: bool
    timely: bool


def outcome(family: str, nom: Nomination) -> SourceOutcome:
    # Replay primary endpoint under *immediate service*: a locked early wrong
    # turn cannot earn credit later, even if the dialogue eventually qualifies.
    timely = (
        nom.supported
        and nom.turn is not None
        and nom.onset_turn is not None
        and 0 <= 2 * (nom.turn - nom.onset_turn) <= 10
    )
    return SourceOutcome(family, nom.onset_turn is not None, nom.turn is not None, timely)


def paired_errors(first: Sequence[bool], second: Sequence[bool]) -> dict[str, int | float | None]:
    """2x2 table for binary errors on the *same* pre-specified eligible units."""
    cells = [0, 0, 0, 0]
    for a, b in zip(first, second, strict=True):
        cells[2 * a + b] += 1
    neither, just_b, just_a, both = cells
    n = len(first)
    den = math.sqrt((both + just_a) * (both + just_b) * (neither + just_b) * (neither + just_a))
    return {
        "n": n,
        "neither": neither,
        "first_only": just_a,
        "second_only": just_b,
        "both": both,
        "expected_both_under_independent_errors": (both + just_a) * (both + just_b) / n if n else None,
        "phi": (both * neither - just_a * just_b) / den if den else None,
    }


def dependence(outcomes: Mapping[str, Mapping[str, SourceOutcome]]) -> dict[str, Any]:
    """Source-level overlap; count every original path once, not every arrival."""
    models = sorted(outcomes)
    ref = outcomes[models[0]]
    if not ref or any(
        ref.keys() != col.keys()
        or any((r.family, r.positive) != (col[d].family, col[d].positive) for d, r in ref.items())
        for col in outcomes.values()
    ):
        raise ValueError("mismatched source paths, families or labels")
    pos = sorted(d for d, r in ref.items() if r.positive)
    neg = sorted(ref.keys() - set(pos))
    families: dict[str, list[str]] = defaultdict(list)
    for d, r in ref.items():
        families[r.family].append(d)
    pos_fam = sorted({ref[d].family for d in pos})
    neg_fam = sorted(families.keys() - set(pos_fam))
    miss = {m: [not outcomes[m][d].timely for d in pos] for m in models}
    fp = {m: [outcomes[m][d].ticket for d in neg] for m in models}
    miss_fam = {
        m: [all(not outcomes[m][d].timely for d in families[f] if ref[d].positive) for f in pos_fam] for m in models
    }
    fp_fam = {m: [any(outcomes[m][d].ticket for d in families[f]) for f in neg_fam] for m in models}

    def block(columns: dict[str, list[bool]]) -> dict[str, Any]:
        return {
            "units": len(columns[models[0]]),
            "errors_per_model": {m: sum(columns[m]) for m in models},
            "all_models_error": sum(all(v) for v in zip(*(columns[m] for m in models), strict=True)),
            "pairs": {f"{a}__{b}": paired_errors(columns[a], columns[b]) for a, b in combinations(models, 2)},
        }

    return {
        "positive_path_misses": block(miss),
        "negative_path_false_alerts": block(fp),
        "positive_family_no_positive_path_caught": block(miss_fam),
        "all_negative_family_any_false_alert": block(fp_fam),
    }


def audit(panel: list[PanelTurn], replay: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct scored OOF trajectories; reject a mismatched replay audit."""
    if (
        not panel
        or replay.get("kind") != "mindguard_full_either_risk_v1"
        or replay.get("gate") != "tuned"
        or replay.get("deadline_minutes") != 10
        or replay.get("turn_minutes") != 2
    ):
        raise ValueError("need primary tuned-gate full replay and nonempty source panel")
    models = sorted(replay["models"])
    if not models or len(set(models)) != len(models) or set(models) != set(panel[0].scores):
        raise ValueError("replay models must match supplied panel exactly")
    if (
        replay["rated_turns"] != len(panel)
        or replay["families"] != len({r.group_id for r in panel})
        or replay["dialogues"] != len({r.dialogue_id for r in panel})
    ):
        raise ValueError("replay source cohort disagrees")
    frozen = {(r["model"], r["fold"]): r for r in replay["training"]}
    if set(frozen) != {(m, f) for m in models for f in range(5)} or len(frozen) != len(replay["training"]):
        raise ValueError("missing or duplicate frozen gate")
    seed = replay["seed"]
    all_paths: dict[str, dict[str, SourceOutcome]] = {}
    probability: dict[str, Any] = {}
    folds: list[dict[str, int]] = []
    held_assignments: dict[str, int] = {}
    for model in models:
        all_paths[model] = {}
        scored: dict[str, list[tuple[float, int]]] = defaultdict(list)
        family_fold: dict[str, int] = {}
        unique: set[tuple[str, int]] = set()
        for fold, (train, held) in enumerate(grouped_folds(encoded(panel, model), 5, seed)):
            paths = calibrated(train, held)
            families = {r.dialogue_id: r.group_id for r in held}
            groups = set(families.values())
            if groups & {r.group_id for r in train} or None in groups:
                raise ValueError("ambiguous or leaked opening family")
            if any(family_fold.setdefault(g, fold) != fold for g in groups if g is not None):
                raise ValueError("family assigned to multiple folds")
            audit_fold = frozen[model, fold]
            cut = audit_fold["cutoff"]
            if (
                not isinstance(cut, (float, int))
                or isinstance(cut, bool)
                or not math.isfinite(cut)
                or not 0 <= cut <= 1
                or audit_fold["heldout_paths"] != len(paths)
                or audit_fold["heldout_turns"] != len(held)
            ):
                raise ValueError("unusable gate, path or turn count")
            observed = {(r.dialogue_id, r.turn_index): (r.scores[0], int(r.votes[0] == "self_harm")) for r in held}
            if unique & observed.keys():
                raise ValueError("repeated held-out source turn")
            unique.update(observed)
            train_prevalence = sum(r.votes[0] == "self_harm" for r in train) / len(train)
            held_brier = {"native": 0.0, "calibrated": 0.0}
            for d, path in paths.items():
                family = families[d]
                if family is None or d in all_paths[model]:
                    raise ValueError("missing or duplicate held-out source")
                all_paths[model][d] = outcome(family, nominate(path, cut))
                for t, q, y in path:
                    p, expected = observed[d, t]
                    if y != expected:
                        raise ValueError("OOF target changed")
                    for key, pred in (("native", p), ("calibrated", q)):
                        held_brier[key] += (pred - y) ** 2
                        scored[key].append((pred, y))
                    scored["training_prevalence"].append((train_prevalence, y))
            if not all(
                math.isclose(val, audit_fold[f"brier_{k}_sum"], rel_tol=1e-9, abs_tol=1e-10)
                for k, val in held_brier.items()
            ):
                raise ValueError("replay scores do not match reconstructed fold")
            positives = sum(all_paths[model][d].positive for d in paths)
            if positives != audit_fold["positive_heldout_paths"]:
                raise ValueError("replay case targets disagree")
            if model == models[0]:
                folds.append(
                    {
                        "fold": fold,
                        "held_families": len(groups),
                        "held_dialogues": len(paths),
                        "held_positive_dialogues": positives,
                        "held_turns": len(held),
                    }
                )
        if len(unique) != len(panel) or (held_assignments and held_assignments != family_fold):
            raise ValueError("incomplete or unpaired held-out folds")
        held_assignments = family_fold
        probability[model] = {
            kind: {
                "brier": sum((p - y) ** 2 for p, y in column) / len(column),
                "average_precision": average_precision(column),
            }
            for kind, column in scored.items()
        }
    return {
        "kind": "full_either_risk_unique_source_oof_v1",
        "seed": seed,
        "gate": "tuned",
        "target": "At least two clinicians voted either risk class on this observed turn; not actual patient harm.",
        "probability_estimand": "Turn-weighted pooled OOF AP (ties grouped) and mean squared error. Training prevalence is a fold-specific constant, not a calibrated clinician posterior.",
        "overlap_estimand": "Immediate service only: first model-specific calibrated crossing locked, supported at crossing, within 10 minutes of first observed endorsement, two-minute turn cadence. No alerts selected or rescored under test outcomes.",
        "uncertainty": "Descriptive paired counts, not independent replicas. Both-expected and phi measure association, not conditional independence given latent risk. Families count once in family-level tables.",
        "folds": folds,
        "models": models,
        "rated_turns": len(panel),
        "positive_turns": sum(y for _, y in scored["native"]),
        "probability": probability,
        "dependence": dependence(all_paths),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path, required=True, help="Matching full primary tuned replay JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        replay = json.loads(args.replay.read_text(encoding="utf-8"))
        if args.out.exists():
            raise ValueError("refusing to overwrite existing report")
        panel = load_panel(args.annotations, args.caches, expected_models=replay["models"])
        result = json.dumps(audit(panel, replay), indent=2, allow_nan=False)
        with args.out.open("x", encoding="utf-8") as handle:
            print(result, file=handle)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ArithmeticError):
        print(
            "Full discrimination failed: check complete receipt panel, matching tuned replay and new output path.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
