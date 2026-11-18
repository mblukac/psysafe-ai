"""Local, fail-closed pairing of MindGuard test turns and cached Decisions responses."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import MODEL_IDS, build_request, extract_probabilities

DATASET_REVISION = "1c082b67bc0a6574badc3bf4cbe3e12af130f606"
LABELS = {"safe", "self_harm", "harm_others"}
Message = tuple[str, str]


@dataclass(frozen=True)
class Turn:
    dialogue_id: str
    turn_index: int
    votes: tuple[str, ...]
    scores: tuple[float, float, float]


@dataclass(frozen=True)
class Case:
    dialogue_id: str
    turn_index: int
    votes: tuple[str, ...]
    prefix: tuple[Message, ...]


def digest(obj: object) -> str:
    encoded = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def local_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError:
            raise ValueError("Parquet requires local pyarrow; alternatively supply a local JSONL export") from None
        yield from pq.read_table(path).to_pylist()
        return
    if path.suffix != ".jsonl":
        raise ValueError("expected local .jsonl or .parquet input")
    with path.open(encoding="utf-8") as lines:
        for line_no, line in enumerate(lines, 1):
            try:
                row = json.loads(line, object_pairs_hook=_unique)
            except ValueError:
                raise ValueError(f"invalid JSON record on line {line_no}") from None
            if not isinstance(row, dict):
                raise ValueError(f"expected object on line {line_no}")
            yield row


def read_cases(records: Iterable[dict[str, Any]]) -> list[Case]:
    """Reconstruct exact prefix chains. Refuse missing turns, duplicates or branching roots."""
    groups: dict[str, dict[int, Case]] = defaultdict(dict)
    for row in records:
        prompt, current, ratings = row.get("prompt"), row.get("user_message"), row.get("annotator_labels")
        if not isinstance(prompt, list) or not isinstance(current, str) or not current.strip():
            raise ValueError("invalid MindGuard prompt or current user message")
        prefix: list[Message] = []
        for msg in prompt:
            if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
                raise ValueError("invalid prefix message")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise ValueError("invalid prefix content")
            prefix.append((msg["role"], msg["content"]))
        # The release may include the rated turn in prompt. Never duplicate it or use a subsequent reply.
        if prefix and prefix[-1][0] == "user":
            if prefix[-1][1] != current:
                raise ValueError("current user turn does not match the end of the prompt")
        else:
            prefix.append(("user", current))
        roles = [role for role, _ in prefix]
        if roles != ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))]:
            raise ValueError("expected an ordered user/assistant conversation prefix")
        if not isinstance(ratings, dict) or set(ratings) != {"annotator_1", "annotator_2", "annotator_3"}:
            raise ValueError("expected three individual clinician votes")
        votes = tuple(ratings[k] for k in sorted(ratings))
        if any(not isinstance(v, str) or v not in LABELS for v in votes):
            raise ValueError("unknown individual clinician label; inspect the source mapping locally")
        group = digest(prefix[:1])
        index = roles.count("user")
        if index in groups[group]:
            raise ValueError("duplicate or ambiguous dialogue/turn; cannot infer an independent chain")
        groups[group][index] = Case(group, index, votes, tuple(prefix))
    cases = []
    for group in sorted(groups):
        previous: tuple[Message, ...] = ()
        for expected, (index, case) in enumerate(sorted(groups[group].items()), 1):
            if index != expected or case.prefix[: len(previous)] != previous:
                raise ValueError("incomplete or branching prefix chain; verify source dialogue IDs before evaluation")
            previous = case.prefix
            cases.append(case)
    if not cases:
        raise ValueError("no annotation records")
    return cases


def requests_for(case: Case) -> Iterator[dict[str, Any]]:
    """Create private request envelopes locally. Never include votes in the provider payload."""
    messages = [{"role": role, "content": content} for role, content in case.prefix]
    for model in MODEL_IDS:
        payload = build_request(model, messages)
        yield {"request_id": digest(payload), "model": model, "payload": payload}


def attach_responses(cases: list[Case], responses: Iterable[dict[str, Any]]) -> list[Turn]:
    """Require exact coverage. Only model self_harm probabilities enter the fit."""
    by_id = {}
    for record in responses:
        rid = record.get("request_id")
        if not isinstance(rid, str) or rid in by_id or not isinstance(record.get("response"), dict):
            raise ValueError("invalid or duplicate cached response")
        by_id[rid] = record
    turns = []
    for case in cases:
        scores = []
        for req in requests_for(case):
            saved = by_id.pop(req["request_id"], None)
            if saved is None or saved.get("model") != req["model"]:
                raise ValueError("missing or mismatched model response for this exact prefix and rubric")
            scores.append(extract_probabilities(req["model"], saved["response"])["self_harm"])
        turns.append(Turn(case.dialogue_id, case.turn_index, case.votes, (scores[0], scores[1], scores[2])))
    if by_id:
        raise ValueError("unexpected cached responses (wrong dataset, prompt or model revision)")
    return turns
