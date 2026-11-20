"""Join private, complete decision-model receipt caches to the *rated* release turns.

No network calls, credential access, gap filling, or complete-dialogue filtering.
The scores are native three-option vectors, not clinician posteriors.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .__main__ import official_cases
from .collect import CollectionError, _cache, _receipts
from .data import Case, digest, requests_for
from .providers import CLEF_FLASH_MODEL, DECIDER_FLASH_MODEL, LUNA_MODEL, MODEL_IDS, extract_probabilities


@dataclass(frozen=True)
class PanelTurn:
    dialogue_id: str
    group_id: str
    turn_index: int
    votes: tuple[str, ...]
    scores: dict[str, dict[str, float]]


MODEL_KEYS = dict(zip(MODEL_IDS, ("jev", "gemini", "liquid"), strict=True)) | {
    LUNA_MODEL: "luna",
    CLEF_FLASH_MODEL: "clef_flash",
    DECIDER_FLASH_MODEL: "decider_flash",
}


def coverage_audit(rows: Sequence[PanelTurn]) -> dict[str, object]:
    """Release-level observed-vs-possible onset counts; no private identifiers.

    A missing user position before a confirmed onset can only move that onset
    earlier. A missing position in a never-endorsed path can change eligibility.
    These logical bounds describe this release, not an extrapolated prevalence
    or a confidence interval. The per-endpoint upper bounds need not be joint.
    """
    paths: dict[str, list[PanelTurn]] = defaultdict(list)
    for row in rows:
        paths[row.dialogue_id].append(row)
    missing = {
        d: set(range(1, max(r.turn_index for r in turns) + 1)) - {r.turn_index for r in turns}
        for d, turns in paths.items()
    }
    by_endpoint = {}
    for ep in ("either", "self_harm", "harm_others"):
        supported = {
            d: [r for r in turns if sum((v != "safe" if ep == "either" else v == ep) for v in r.votes) >= 2]
            for d, turns in paths.items()
        }
        positive = {d: min(r.turn_index for r in turns) for d, turns in supported.items() if turns}
        shifts = [max(0, onset - min(missing[d], default=onset)) for d, onset in positive.items()]
        ambiguous = sum(bool(holes) and d not in positive for d, holes in missing.items())
        by_endpoint[ep] = {
            "observed_positive_paths": len(positive),
            "observed_positive_turns": sum(len(turns) for turns in supported.values()),
            "positive_opening_families": len({paths[d][0].group_id for d in positive}),
            "possible_positive_paths_if_any_unrated_turn_qualifies": len(positive) + ambiguous,
            "confirmed_paths_with_ambiguous_onset": sum(s > 0 for s in shifts),
            "max_earlier_onset_shift_turns": max(shifts, default=0),
        }
    return {
        "rated_turns": len(rows),
        "paths": len(paths),
        "opening_families": len({r.group_id for r in rows}),
        "unrated_positions": sum(len(holes) for holes in missing.values()),
        "paths_with_unrated_positions": sum(bool(holes) for holes in missing.values()),
        "by_endpoint": by_endpoint,
    }


def load_panel(
    annotations: Path, cache_paths: Sequence[Path], *, expected_models: Sequence[str] | None = None
) -> list[PanelTurn]:
    """Require every observed model on every rated turn; missing paths/caches are errors.

    Supply `expected_models` (short keys) to additionally reject a wholly absent model.
    Do not remove a failing case or replace an unknown score with zero.
    """
    return _join(official_cases(annotations), cache_paths, expected_models=expected_models)


def _join(
    cases: Sequence[Case], cache_paths: Sequence[Path], *, expected_models: Sequence[str] | None = None
) -> list[PanelTurn]:
    if not cases or not cache_paths:
        raise CollectionError("Need annotations and one or more complete private receipt caches")
    rows = [PanelTurn(c.dialogue_id, digest(c.prefix[:1]), c.turn_index, c.votes, {}) for c in cases]
    if len({(r.dialogue_id, r.turn_index) for r in rows}) != len(rows):
        raise CollectionError("Duplicate release dialogue/turn")
    expected: dict[str, str] = {}
    indices: dict[str, int] = {}
    model_set: set[str] = set()
    generations: set[str] = set()
    for path in cache_paths:
        with _cache(path, write=False) as f:
            if f is None:
                raise CollectionError("Requested receipt cache does not exist")
            # Discover cohort membership while holding the same shared read lock.
            # Never trust filename, row order or case identifier from a response.
            found: set[str] = set()
            for line in f:
                obj = json.loads(line)
                model = obj.get("model") if isinstance(obj, dict) else None
                if not isinstance(model, str) or model not in MODEL_KEYS:
                    raise CollectionError("Unknown or missing model snapshot in cache")
                found.add(model)
            if not found:
                raise CollectionError("Empty receipt cache")
            for model in sorted(found - model_set):
                for i, c in enumerate(cases):
                    (req,) = requests_for(c, models=(model,))
                    rid = req["request_id"]
                    if rid in expected:
                        raise CollectionError("Non-unique request identity")
                    expected[rid], indices[rid] = model, i
            model_set.update(found)
            f.seek(0)
            # _receipts validates the wire choice/vector, allowed processor, model,
            # private envelope, request digest and generation ID for each receipt.
            for rid, generation, response in _receipts(f, expected):
                model = expected[rid]
                key, row = MODEL_KEYS[model], rows[indices[rid]]
                if key in row.scores or generation in generations:
                    raise CollectionError("Duplicate request or generation across caches")
                generations.add(generation)
                row.scores[key] = extract_probabilities(model, response)
    keys = {MODEL_KEYS[m] for m in model_set}
    if expected_models is not None and set(expected_models) != keys:
        raise CollectionError("Receipt models do not match the explicitly requested panel")
    if any(set(r.scores) != keys for r in rows):
        raise CollectionError("Incomplete receipt panel; require every rated turn for each selected model")
    return rows
