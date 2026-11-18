"""Invented conversations, votes and provider outputs exercising the complete pipeline."""

from __future__ import annotations

from typing import Any

from .data import Turn, attach_responses, read_cases, requests_for
from .providers import synthetic_response


def smoke_turns() -> list[Turn]:
    records: list[dict[str, Any]] = []
    for dialogue in range(18):
        history: list[dict[str, str]] = []
        for turn in range(3):
            message = f"Invented fixture {dialogue}, step {turn}."
            positive = dialogue % 3 == 0 and turn == 2
            votes = ["self_harm"] * (2 + dialogue % 2) if positive else ["safe"] * 2
            votes += ["harm_others" if dialogue % 4 == 0 else "safe"] * (3 - len(votes))
            if dialogue % 5 == 0 and turn == 1:
                votes[-1] = "self_harm"
            records.append(
                {
                    "prompt": history.copy(),
                    "user_message": message,
                    "annotator_labels": {f"annotator_{n}": vote for n, vote in enumerate(votes, 1)},
                }
            )
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": "Fixture reply."}])
    cases = read_cases(reversed(records))
    cache = []
    for i, case in enumerate(cases):
        for j, req in enumerate(requests_for(case)):
            # Vary scores, including misleading early peaks. No real provider outputs.
            p = (((i * (13 + 2 * j) + j * 3) % 91) + 2) / 100
            probs = {"self_harm": p, "safe": (1 - p) * 0.9, "harm_others": (1 - p) * 0.1}
            cache.append(
                {
                    "request_id": req["request_id"],
                    "model": req["model"],
                    "response": synthetic_response(req["model"], probs),
                }
            )
    return attach_responses(cases, reversed(cache))
