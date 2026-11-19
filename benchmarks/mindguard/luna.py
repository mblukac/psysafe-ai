"""Offline loading for the optional single-model Luna receipt cache."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .collect import CollectionError, _cache, _receipts
from .data import Case, requests_for
from .providers import LUNA_MODEL, extract_probabilities


def load_luna_scores(cases: Sequence[Case], cache_path: Path) -> dict[tuple[str, int], float]:
    """Require exact coverage and genuine-shaped receipts for the pinned request and processor.

    Pass all cases used for collection, before selecting complete dialogues.
    An absent, refused, malformed or duplicate response never becomes a zero.
    """
    by_id = {
        req["request_id"]: (c.dialogue_id, c.turn_index) for c in cases for req in requests_for(c, models=(LUNA_MODEL,))
    }
    if not cases or len(by_id) != len(cases) or len(set(by_id.values())) != len(cases):
        raise CollectionError("Expected unique, nonempty release case keys")
    expected = dict.fromkeys(by_id, LUNA_MODEL)
    with _cache(cache_path, write=False) as handle:
        scores = {
            by_id[rid]: extract_probabilities(LUNA_MODEL, response)["self_harm"]
            for rid, _, response in _receipts(handle, expected)
        }
    if len(scores) != len(by_id):
        raise CollectionError(f"Incomplete Luna cache: {len(scores)} of {len(by_id)} scores; cohort not assessable")
    return scores
