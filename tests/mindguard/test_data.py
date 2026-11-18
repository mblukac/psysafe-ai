from __future__ import annotations

import json
from typing import Any

import pytest

from benchmarks.mindguard.data import (
    attach_responses,
    complete_dialogues,
    digest,
    read_cases,
    read_release_cases,
    requests_for,
)
from benchmarks.mindguard.providers import MODEL_IDS, build_request, synthetic_response


def row(message: str, prompt: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "user_message": message,
        "label": "ignored",
        "annotator_labels": {"annotator_1": "self_harm", "annotator_2": "safe", "annotator_3": "safe"},
    }


HISTORY = [{"role": "user", "content": "Opening."}, {"role": "assistant", "content": "Reply."}]


def test_exact_release_spellings_preserve_each_raters_vote():
    source = row("Opening.", [])
    source["annotator_labels"] = {
        "annotator_3": "unsafe_harm_to_others",
        "annotator_2": "safe",
        "annotator_1": "unsafe_self_harm_risk",
    }
    assert read_cases([source])[0].votes == ("self_harm", "safe", "harm_others")


@pytest.mark.parametrize("unknown", ["unsafe", " unsafe_self_harm_risk", "unsafe_threats_to_others", None])
def test_unrecognized_vote_fails_closed(unknown):
    source = row("Opening.", [])
    source["annotator_labels"]["annotator_1"] = unknown
    with pytest.raises(ValueError, match="unknown individual clinician label"):
        read_cases([source])


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


def test_release_shared_root_has_distinct_case_keys_with_identical_payloads():
    opening = row("Opening.", [])
    records = [opening, row("Branch A", HISTORY), opening, row("Branch B", HISTORY)]
    cases = read_release_cases(records)
    assert read_release_cases(reversed(records)) == cases
    assert len(cases) == 4
    assert complete_dialogues(cases) == {digest(c.prefix) for c in cases if c.turn_index == 2}
    first = [c for c in cases if c.turn_index == 1]
    assert len(first) == 2
    assert first[0].votes == first[1].votes
    assert first[0].dialogue_id != first[1].dialogue_id
    reqs_a, reqs_b = [list(requests_for(c)) for c in first]
    for a, b in zip(reqs_a, reqs_b, strict=True):
        assert a["payload"] == b["payload"]
        assert a["request_id"] != b["request_id"]
        assert a["request_id"] == digest(["mindguard-case-request-v2", first[0].dialogue_id, 1, a["payload"]])
        assert first[0].dialogue_id not in str(a["payload"])
    cache = [
        {
            "request_id": req["request_id"],
            "model": req["model"],
            "response": synthetic_response(req["model"], {"safe": 0.5, "self_harm": 0.2, "harm_others": 0.3}),
        }
        for c in cases
        for req in requests_for(c)
    ]
    turns = attach_responses(cases, cache)
    assert {t.group_id for t in turns} == {digest(first[0].prefix)}
    assert {t.dialogue_id for t in turns} == complete_dialogues(cases)


def test_release_preserves_missing_ratings_and_absolute_turn_indices():
    long_history = [*HISTORY, {"role": "user", "content": "Missing vote."}, {"role": "assistant", "content": "Reply."}]
    other = row("Complete.", [])
    cases = read_release_cases([row("Third.", long_history), row("Opening.", []), other])
    assert complete_dialogues(cases) == {digest([("user", "Complete.")])}
    assert sorted(c.turn_index for c in cases) == [1, 1, 3]
    assert complete_dialogues(read_release_cases([row("Initial missing.", HISTORY)])) == set()


def test_release_rejects_ambiguous_assignment_of_shared_ratings():
    opening = row("Opening.", [])
    other_vote = row("Opening.", [])
    other_vote["annotator_labels"] = {"annotator_1": "safe", "annotator_2": "self_harm", "annotator_3": "safe"}
    terminals = [row("A", HISTORY), row("B", HISTORY)]
    with pytest.raises(ValueError, match="multiplicity"):
        read_release_cases([opening, *terminals])
    with pytest.raises(ValueError, match="conflicting ordered"):
        read_release_cases([opening, other_vote, *terminals])
    with pytest.raises(ValueError, match="multiplicity"):
        read_release_cases([opening, opening])


def test_release_literal_empty_user_is_a_rated_turn_not_a_placeholder():
    blank = {"role": "user", "content": ""}
    records = [row("", [blank]), row("Next.", [blank, {"role": "assistant", "content": "Reply."}])]
    first, second = read_release_cases(records)
    for case in [first, second]:
        state = json.loads(next(requests_for(case))["payload"]["state"])
        assert state["conversation"][0] == blank
    with pytest.raises(ValueError, match="current user message"):
        read_cases(records)
    with pytest.raises(ValueError, match="nonempty"):
        build_request(MODEL_IDS[0], [blank])
    with pytest.raises(ValueError, match="current user message"):
        read_release_cases([row(" ", [])])
    with pytest.raises(ValueError, match="invalid prefix content"):
        read_release_cases([row("Next.", [blank, {"role": "assistant", "content": ""}])])
