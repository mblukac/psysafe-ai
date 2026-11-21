"""Which distinct observed-positive conversations did the original tuned gates never flag?"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION
from .fit import grouped_folds
from .full_discrimination import outcome, paired_errors
from .full_panel import MODEL_KEYS, PanelTurn, coverage_audit, load_panel
from .full_replay import calibrated, encoded
from .surge_replay import Nomination, nominate

ROOT = Path(__file__).resolve().parent


def summarize(families: Mapping[str, str], predictions: Mapping[str, Mapping[str, Nomination]]) -> dict[str, Any]:
    """Unweighted source counts. No names, private per-case arrays, or resampled arrivals leave this function."""
    models = sorted(predictions)
    if not models or not families or any(set(predictions[m]) != set(families) for m in models):
        raise ValueError("need paired models and common conversation/family coverage")
    positive: set[str] = set()
    negative: set[str] = set()
    misses: dict[str, set[str]] = {m: set() for m in models}
    supported_hits: dict[str, set[str]] = {m: set() for m in models}
    negative_flags: dict[str, set[str]] = {m: set() for m in models}
    late: Counter[str] = Counter()
    or_hit = 0
    for d, fam in families.items():
        noms = {m: predictions[m][d] for m in models}
        if len({n.onset_turn for n in noms.values()}) != 1:
            raise ValueError("unpaired first evidence of need")
        firsts = sorted(n.turn for n in noms.values() if n.turn is not None)
        same_turn: dict[int, bool] = {}
        for n in noms.values():
            if n.turn is not None and same_turn.setdefault(n.turn, n.supported) != n.supported:
                raise ValueError("different endorsement at the same turn")
        if noms[models[0]].onset_turn is None:
            negative.add(d)
            for m, n in noms.items():
                if n.turn is not None:
                    negative_flags[m].add(d)
            continue
        positive.add(d)
        earliest_supported = False
        for m, n in noms.items():
            if n.turn is None:
                misses[m].add(d)
            elif outcome(fam, n).timely:
                supported_hits[m].add(d)
                earliest_supported |= n.turn == firsts[0]
            elif n.onset_turn is not None and 2 * (n.turn - n.onset_turn) > 10:
                late[m] += 1
        or_hit += int(earliest_supported)
    if not positive:
        raise ValueError("need a positive source denominator")
    n_models = len(models)
    freq = Counter(sum(d in misses[m] for m in models) for d in positive)
    all_miss = set.intersection(*misses.values())
    flags = {m: positive - misses[m] for m in models}
    union = set.union(*flags.values())
    pos_families = {families[d] for d in positive}
    counts = {m: [d in misses[m] for d in sorted(positive)] for m in models}
    pairs = {a: {b: paired_errors(counts[a], counts[b]) for b in models} for a in models}
    if sum(freq.values()) != len(positive) or sum(k * x for k, x in freq.items()) != sum(map(len, misses.values())):
        raise ValueError("miss histogram does not reconcile")
    by_model = {
        m: {
            "never_flagged_positive": len(misses[m]),
            "any_flag_positive": len(flags[m]),
            "first_flag_after_deadline": late[m],
            "timely_supported_if_immediate": len(supported_hits[m]),
            "never_flagged_here_but_flagged_elsewhere": len(union - flags[m]),
            "sole_detector_to_flag": sum(sum(d in other for other in flags.values()) == 1 for d in flags[m]),
            "any_flag_on_observed_negative": len(negative_flags[m]),
        }
        for m in models
    }
    return {
        "observed_positive_paths": len(positive),
        "observed_negative_paths": len(negative),
        "positive_opening_families": len(pos_families),
        "families_containing_a_path_missed_by_all": len({families[d] for d in all_miss}),
        "positive_families_with_no_positive_path_flagged_by_any": len(pos_families - {families[d] for d in union}),
        "by_model": by_model,
        "never_flagged_by_n_models_from_0_up": [freq[k] for k in range(n_models + 1)],
        "joint_never_flagged_counts": [[pairs[a][b]["both"] for b in models] for a in models],
        "pairwise_miss_phi": [[pairs[a][b]["phi"] for b in models] for a in models],
        "mean_model_no_flag_rate": sum(map(len, misses.values())) / (n_models * len(positive)),
        "never_flagged_by_any": len(all_miss),
        "flagged_by_any": len(union),
        "retrospective_union_of_immediate_supported_catches": len(set.union(*supported_hits.values())),
        "first_to_flag_or_immediate_supported_catches": or_hit,
        "observed_negative_flagged_by_any": len(set.union(*negative_flags.values())),
    }


def audit(panel: list[PanelTurn], reference: dict[str, Any], errors: dict[str, Any]) -> dict[str, Any]:
    models = sorted(MODEL_KEYS.values())
    coverage = coverage_audit(panel)
    required = dict(
        kind="full_mindguard_ticket_updates_v1",
        seed=2026,
        folds=5,
        dataset_revision=DATASET_REVISION,
        deadline_minutes=10,
        turn_minutes=2,
        models=models,
        coverage=coverage,
    )
    if any(reference.get(k) != v for k, v in required.items()):
        raise ValueError("wrong release, cohort, model set, folds or replay clock")
    fits = {(r["model"], r["fold"]): r for r in reference["fits"]}
    if set(fits) != set(product(models, range(5))) or len(fits) != len(reference["fits"]):
        raise ValueError("incomplete or duplicate frozen model/fold fit")
    families = {r.dialogue_id: r.group_id for r in panel}
    predictions: dict[str, dict[str, Nomination]] = {m: {} for m in models}
    fold_for_family: dict[str, int] = {}
    for m in models:
        for f, (train, held) in enumerate(grouped_folds(encoded(panel, m), 5, 2026)):
            ps = calibrated(train, held)
            test_groups = {families[d] for d in ps}
            fit = fits[m, f]
            cutoff = fit["cutoffs"]["tuned"]
            # Verify the exact held-out score calibration and family blocking before loading its frozen gate.
            if (
                test_groups & {r.group_id for r in train}
                or predictions[m].keys() & ps.keys()
                or any(fold_for_family.setdefault(g, f) != f for g in test_groups)
                or fit["held_paths"] != len(ps)
                or fit["held_turns"] != len(held)
                or not 0 <= cutoff <= 1
                or not math.isclose(
                    sum((q - y) ** 2 for p in ps.values() for _, q, y in p), fit["single_brier_sum"], abs_tol=1e-9
                )
            ):
                raise ValueError("inconsistent held-out forecasts or family folds")
            predictions[m].update({d: nominate(p, cutoff) for d, p in ps.items()})
    result = summarize(families, predictions)
    fold_counts = []
    for f in range(5):
        group = {d: fam for d, fam in families.items() if fold_for_family[fam] == f}
        # All five original folds have positives. Suppress per-path/private IDs.
        block = summarize(group, {m: {d: predictions[m][d] for d in group} for m in models})
        fold_counts.append(
            dict(
                fold=f,
                positive_paths=block["observed_positive_paths"],
                never_flagged_by_any=block["never_flagged_by_any"],
                first_to_flag_or_immediate_supported_catches=block["first_to_flag_or_immediate_supported_catches"],
                any_flag_positive_by_model=[block["by_model"][m]["any_flag_positive"] for m in models],
            )
        )
    errors_ref = errors["dependence"]["positive_path_misses"]
    if (
        errors.get("kind") != "full_either_risk_unique_source_oof_v1"
        or errors["gate"] != "tuned"
        or errors["seed"] != 2026
        or sorted(errors["models"]) != models
        or errors_ref["units"] != result["observed_positive_paths"]
        or any(
            errors_ref["units"] - errors_ref["errors_per_model"][m]
            != result["by_model"][m]["timely_supported_if_immediate"]
            for m in models
        )
    ):
        raise ValueError("does not reproduce previously published distinct-case reference")
    return dict(
        kind="mindguard_tuned_no_ticket_overlap_v1",
        dataset_revision=DATASET_REVISION,
        models_and_matrix_order=models,
        gate="original family-held-out tuned",
        matrix_unit="distinct observed-positive conversation, before resampling; miss = no first crossing anywhere",
        family_blocked_fold_counts=fold_counts,
        note="A flag does not mean an endorsed turn was reviewed. The retrospective union is not an ensemble policy. "
        "Miss correlations describe these role-plays; 17 positives in 16 families do not establish population dependence.",
        **result,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=ROOT / "private")
    p.add_argument("--reference-dir", type=Path, default=ROOT / "figures")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("use a fresh output path")
    panel = load_panel(
        args.inputs / "mindguard.parquet",
        [
            args.inputs / x
            for x in (
                "responses.jsonl",
                "luna-responses.jsonl",
                "clef-flash-responses.jsonl",
                "decider-flash-responses.jsonl",
            )
        ],
        expected_models=sorted(MODEL_KEYS.values()),
    )
    result = audit(
        panel,
        *[json.loads((args.reference_dir / x).read_text()) for x in ("ticket-updates.json", "full-panel-errors.json")],
    )
    with args.out.open("x", encoding="utf-8") as f:
        print(json.dumps(result, indent=2, allow_nan=False), file=f)


if __name__ == "__main__":
    main()
