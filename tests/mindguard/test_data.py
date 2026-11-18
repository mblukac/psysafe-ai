from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.mindguard.data import attach_responses, read_cases, requests_for
from benchmarks.mindguard.providers import synthetic_response


def row(message: str, prompt: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "user_message": message,
        "label": "ignored",
        "annotator_labels": {"annotator_1": "self_harm", "annotator_2": "safe", "annotator_3": "safe"},
    }


HISTORY = [{"role": "user", "content": "Opening."}, {"role": "assistant", "content": "Reply."}]


def test_links_exact_prefixes_in_any_order_and_avoids_duplicate_current_turn():
    cases = read_cases([row("Next.", [*HISTORY, {"role": "user", "content": "Next."}]), row("Opening.", [])])
    assert cases == read_cases([row("Opening.", HISTORY[:1]), row("Next.", HISTORY)])
    assert [c.turn_index for c in cases] == [1, 2]
    assert cases[0].dialogue_id == cases[1].dialogue_id
    assert all(json.loads(r["payload"]["state"]) == {"conversation": HISTORY[:1]} for r in requests_for(cases[0]))


@pytest.mark.parametrize(
    "records",
    [
        [row("Opening.", []), row("Opening.", [])],
        [row("Opening.", []), row("X.", HISTORY), row("Y.", HISTORY)],
        [row("Missing previous.", HISTORY)],
        [row("Opening.", [{"role": "system", "content": "Hidden scenario"}])],
        [row("Not last.", HISTORY[:1])],
        [row("Opening.", []) | {"annotator_labels": {"annotator_1": "safe"}}],
    ],
)
def test_rejects_ambiguous_chains_and_incomplete_ratings(records):
    with pytest.raises(ValueError, match=r"ambiguous|incomplete|invalid prefix|does not match|three individual"):
        read_cases(records)


def test_cache_binds_all_models_and_exact_inputs():
    cases = read_cases([row("A", [])])
    reqs = list(requests_for(cases[0]))
    saved = [
        {
            "request_id": r["request_id"],
            "model": r["model"],
            "response": synthetic_response(r["model"], {"safe": 0.8, "self_harm": 0.15, "harm_others": 0.05}),
        }
        for r in reqs
    ]
    assert attach_responses(cases, reversed(saved))[0].scores == (0.15, 0.15, 0.15)
    for invalid in [
        saved[:2],
        [*saved, saved[0]],
        [*saved, saved[0] | {"request_id": "old"}],
        [saved[0] | {"model": saved[1]["model"]}, *saved[1:]],
    ]:
        with pytest.raises(ValueError, match=r"missing|duplicate|unexpected"):
            attach_responses(cases, invalid)
    with pytest.raises(ValueError, match="missing"):
        attach_responses(read_cases([row("Changed", [])]), saved)
