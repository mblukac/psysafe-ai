"""Opt-in OpenRouter collection. Without --allow-paid, only inspect local inputs.

The POSIX cache records successful responses only. An interrupted request may have
been billed; reconcile it with the provider before explicitly resuming. The cap is
per invocation, not a dollar limit. OpenRouter controls its own internal routing.
"""

from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from .__main__ import official_cases
from .data import _unique, digest, requests_for
from .providers import MODEL_IDS, build_request, extract_probabilities

Envelope = dict[str, Any]
Transport = Callable[[Envelope], Envelope]
_PROVIDERS = dict(zip(MODEL_IDS, ("TypeSafe", "Google AI Studio", "Liquid"), strict=True))
_CANARY_TEXT = "I spent the morning looking for my library card and found it in my coat."


class CollectionError(ValueError):
    """Operator-facing messages must never contain request or provider response text."""


def canary_requests() -> list[Envelope]:
    batch = []
    for model in MODEL_IDS:
        payload = build_request(model, [{"role": "user", "content": _CANARY_TEXT}])
        batch.append({"request_id": digest(payload), "model": model, "payload": payload})
    return batch


def _post(payload: Envelope) -> Envelope:
    """Exactly one direct TLS POST, with no redirect, retry or user-configurable host."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise CollectionError("OPENROUTER_API_KEY is required for an authorized send")
    conn = http.client.HTTPSConnection("openrouter.ai", timeout=45)
    try:
        conn.request(
            "POST",
            "/api/alpha/decisions",
            body=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        if response.status != 200:
            raise CollectionError(f"OpenRouter returned HTTP {response.status}; no automatic retry")
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise CollectionError("Oversized Decisions response")
        result = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(result, dict):
            raise CollectionError("Invalid Decisions response")
        return result
    finally:
        conn.close()


@contextmanager
def _cache(path: Path, *, write: bool) -> Iterator[TextIO | None]:
    """POSIX: no symlinks, hard links, shared permissions, FIFOs or concurrent writers."""
    if path.suffix != ".jsonl":
        raise CollectionError("The private cache must be a .jsonl file")
    flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_RDWR | os.O_CREAT | os.O_APPEND if write else os.O_RDONLY)
    try:
        fd = os.open(path, flags, 0o600)
    except FileNotFoundError:
        if write:
            raise
        yield None
        return
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise CollectionError("Cache must be a private owner-only regular file")
        lock = fcntl.LOCK_EX if write else fcntl.LOCK_SH
        fcntl.flock(fd, lock | fcntl.LOCK_NB)
        with os.fdopen(fd, "a+" if write else "r", encoding="utf-8", closefd=False) as stream:
            stream.seek(0)
            yield stream
    finally:
        os.close(fd)


def _validate(model: str, response: object) -> str:
    if (
        not isinstance(response, dict)
        or response.get("provider") != _PROVIDERS[model]
        or not isinstance(response.get("id"), str)
        or not response["id"].startswith("gen-")
    ):
        raise CollectionError("Missing generation receipt or unexpected provider")
    extract_probabilities(model, response)
    return str(response["id"])


def _seen(stream: TextIO | None, expected: dict[str, str]) -> dict[str, str]:
    seen: dict[str, str] = {}
    generations: set[str] = set()
    if stream is None:
        return seen
    for line in stream:
        if not line.endswith("\n"):
            raise CollectionError("Incomplete cache record; reconcile before resuming")
        row = json.loads(line, object_pairs_hook=_unique)
        if not isinstance(row, dict) or set(row) != {"request_id", "model", "response"}:
            raise CollectionError("Invalid cached envelope")
        rid = row["request_id"]
        if not isinstance(rid, str) or rid in seen or rid not in expected or row["model"] != expected[rid]:
            raise CollectionError("Duplicate, unexpected or mismatched cached request")
        generation = _validate(row["model"], row["response"])
        if generation in generations:
            raise CollectionError("Duplicate generation receipt")
        seen[rid] = generation
        generations.add(generation)
    return seen


def _require_canary(path: Path) -> None:
    expected = {r["request_id"]: r["model"] for r in canary_requests()}
    with _cache(path, write=False) as handle:
        if set(_seen(handle, expected)) != set(expected):
            raise CollectionError("A complete, current three-model synthetic canary cache is required")


def main(argv: list[str] | None = None, *, transport: Transport = _post) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    canary = commands.add_parser("canary", help="one fixed, invented prefix per pinned model")
    batch = commands.add_parser("batch", help="official local annotations only")
    for sub in (canary, batch):
        sub.add_argument("--cache", required=True, type=Path, help="owner-only local response .jsonl")
        sub.add_argument("--max-requests", required=True, type=int, help="maximum new requests in this invocation")
        sub.add_argument("--allow-paid", action="store_true", help="authorize real, billable requests")
    batch.add_argument("--annotations", required=True, type=Path)
    batch.add_argument("--canary-cache", required=True, type=Path)
    batch.add_argument("--allow-gated-provider-processing", action="store_true", help="confirm permission to send text")
    args = parser.parse_args(argv)
    try:
        if args.max_requests < 0:
            raise CollectionError("Request cap must not be negative")
        if args.mode == "batch":
            if args.allow_paid and not args.allow_gated_provider_processing:
                raise CollectionError("Confirm permission for gated text using --allow-gated-provider-processing")
            if args.cache.resolve() in (args.annotations.resolve(), args.canary_cache.resolve()):
                raise CollectionError("Dataset, batch cache and canary cache must be separate")
            if args.allow_paid:
                _require_canary(args.canary_cache)
            requests = [r for case in official_cases(args.annotations) for r in requests_for(case)]
        else:
            requests = canary_requests()

        expected = {r["request_id"]: r["model"] for r in requests}
        if len(expected) != len(requests):
            raise CollectionError("Duplicate prepared request")
        with _cache(args.cache, write=args.allow_paid) as handle:
            seen = _seen(handle, expected)
            todo = [r for r in requests if r["request_id"] not in seen]
            permitted = todo[: args.max_requests]
            print(
                f"Prepared: {len(expected)}; cached: {len(seen)}; remaining: {len(todo)}; within cap: {len(permitted)}",
                flush=True,
            )
            if not args.allow_paid:
                print("Preflight only. No requests sent; no canary or provider access tested.")
                return 0
            if handle is None:
                raise CollectionError("Private response cache unavailable")
            generations = set(seen.values())
            for req in permitted:
                result = transport(req["payload"])
                generation = _validate(req["model"], result)
                if generation in generations:
                    raise CollectionError("Duplicate generation receipt")
                generations.add(generation)
                row = {"request_id": req["request_id"], "model": req["model"], "response": result}
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"Saved {len(permitted)} new responses; {len(todo) - len(permitted)} remain.")
    except CollectionError as exc:
        print(f"Collection stopped: {exc}", file=sys.stderr)
        if args.allow_paid:
            print(
                "Check provider activity before retrying; an unrecorded request may have been billed.", file=sys.stderr
            )
        return 2
    except (OSError, ValueError, http.client.HTTPException, UnicodeError):
        # Provider/Parquet/codec diagnostics may echo sensitive input, credentials or response text.
        print(
            "Collection stopped: local data, transport or response validation failed. No automatic retry.",
            file=sys.stderr,
        )
        if args.allow_paid:
            print(
                "Check provider activity before retrying; an unrecorded request may have been billed.", file=sys.stderr
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
