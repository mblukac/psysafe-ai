"""Opt-in OpenRouter collection. Without --allow-paid, only inspect local inputs.

The POSIX cache records successful responses only. An interrupted request may have
been billed; reconcile it with the provider before explicitly resuming. The cap is
per invocation, not a dollar limit. The optional Luna cohort disallows fallback
and restricts routing to OpenAI; the historical three-model payload is unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from threading import Event
from typing import Any, TextIO

from .__main__ import official_cases
from .data import _unique, digest, requests_for
from .providers import FLASH_ROUTES, LUNA_MODEL, MODEL_IDS, build_request, extract_probabilities

Envelope = dict[str, Any]
Transport = Callable[[Envelope], Envelope]
_PROVIDERS = dict(zip(MODEL_IDS, ("TypeSafe", "Google AI Studio", "Liquid"), strict=True))
_PROVIDERS[LUNA_MODEL] = "OpenAI"
_PROVIDERS.update({model: route.name for model, route in FLASH_ROUTES.items()})
_CANARY_TEXT = "I spent the morning looking for my library card and found it in my coat."
_LUNA_BUDGET = Decimal("0.30")
# Entire dated snapshot input context at the price ceiling in the outgoing
# request. We reserve the entire context, NOT a heuristic prompt token count.
_LUNA_MAX_CHARGE = Decimal("0.105")  # 1,050,000 tokens * $0.10 / 1M


class CollectionError(ValueError):
    """Operator-facing messages must never contain request or provider response text."""


class _HTTPStatusError(CollectionError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"OpenRouter returned HTTP {status}; no automatic retry")


class _MalformedWireError(CollectionError):
    """HTTP 200 with a malformed, oversized, or ambiguous response body."""


class _FailedRequestError(Exception):
    def __init__(self, diagnostic: Envelope) -> None:
        self.diagnostic = diagnostic
        super().__init__("Budgeted Decisions request failed")


def _luna_charge(response: Envelope) -> Decimal:
    usage = response.get("usage")
    amount = usage.get("cost") if isinstance(usage, dict) else None
    if type(amount) not in (int, float):
        raise CollectionError("Missing billed Luna usage.cost; stop and reconcile")
    usd = Decimal(str(amount))
    if not usd.is_finite() or not 0 <= usd <= _LUNA_MAX_CHARGE:
        raise CollectionError("Invalid Luna charge; stop and reconcile")
    return usd


def _flash_charge(model: str, response: Envelope) -> Decimal:
    route = FLASH_ROUTES[model]
    usage = response.get("usage")
    amount = usage.get("cost") if isinstance(usage, dict) else None
    count = usage.get("input_tokens") if isinstance(usage, dict) else None
    if type(amount) not in (int, float) or type(count) is not int or not 1 <= count <= route.context:
        raise CollectionError("Missing or invalid Flash charge or input token count; stop and reconcile")
    usd = Decimal(str(amount))
    if not usd.is_finite() or not 0 <= usd <= route.reservation:
        raise CollectionError("Flash charge outside full-context reservation; stop and reconcile")
    return usd


def canary_requests(models: tuple[str, ...] = MODEL_IDS) -> list[Envelope]:
    batch = []
    for model in models:
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
            # Don't parse an HTTP error body: it may echo private conversation text.
            raise _HTTPStatusError(response.status)
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise _MalformedWireError("Oversized Decisions response")
        try:
            result = json.loads(raw, object_pairs_hook=_unique)
        except (ValueError, UnicodeError):
            raise _MalformedWireError("Invalid Decisions response encoding or structure") from None
        if not isinstance(result, dict):
            raise _MalformedWireError("Invalid Decisions response")
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
    if model == LUNA_MODEL:
        _luna_charge(response)
    if model in FLASH_ROUTES:
        _flash_charge(model, response)
    return str(response["id"])


def _receipts(stream: TextIO | None, expected: dict[str, str]) -> Iterator[tuple[str, str, Envelope]]:
    seen: set[str] = set()
    generations: set[str] = set()
    if stream is None:
        return
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
        seen.add(rid)
        generations.add(generation)
        yield rid, generation, row["response"]


def _seen(stream: TextIO | None, expected: dict[str, str]) -> dict[str, str]:
    return {rid: generation for rid, generation, _ in _receipts(stream, expected)}


def _require_canary(path: Path, models: tuple[str, ...]) -> Decimal:
    expected = {r["request_id"]: r["model"] for r in canary_requests(models)}
    with _cache(path, write=False) as handle:
        rows = list(_receipts(handle, expected))
        if {r[0] for r in rows} != set(expected):
            raise CollectionError("A complete synthetic canary cache for the selected cohort is required")
        return (
            sum((_luna_charge(response) for _, _, response in rows), Decimal(0))
            if models == (LUNA_MODEL,)
            else Decimal(0)
        )


def _save(handle: TextIO, generations: set[str], req: Envelope, result: Envelope, generation: str) -> None:
    if generation in generations:
        raise CollectionError("Duplicate generation receipt")
    row = {"request_id": req["request_id"], "model": req["model"], "response": result}
    handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    generations.add(generation)


def _failure_metadata(model: str, result: object, exc: Exception, *, duplicate: bool = False) -> Envelope:
    """Whitelist only diagnostic metadata. Never copy exception text or answer data.

    A generation-shaped ID, route and charge are evidence for an operator to
    reconcile, not authorization to settle a reservation/retry this request.
    """
    if isinstance(exc, _HTTPStatusError):
        return {"category": "http_status", "http_status": exc.status}
    if isinstance(exc, TimeoutError):
        return {"category": "timeout"}
    if isinstance(exc, _MalformedWireError):
        return {"category": "malformed_wire", "http_status": 200}
    if not isinstance(result, dict):
        return {"category": "transport_or_local_failure"}
    diagnostic: Envelope = {"category": "invalid_receipt", "http_status": 200}
    if result.get("model") != model or result.get("provider") != _PROVIDERS[model]:
        diagnostic["category"] = "route_or_model_mismatch"
        return diagnostic
    generation = result.get("id")
    # Deliberately narrower than the common legacy extraction contract. Don't
    # persist a free-form string under an "id" key. Unknown formats are omitted.
    if isinstance(generation, str) and re.fullmatch(r"gen-(?:dec-)?[0-9]{10}-[a-zA-Z0-9]{20}", generation):
        diagnostic["generation_id"] = generation
        try:
            charge = _flash_charge(model, result) if model in FLASH_ROUTES else _luna_charge(result)
            diagnostic["reported_cost_usd"] = str(charge)
        except CollectionError:
            pass
    answers = result.get("answers")
    answer = answers.get("mindguard") if isinstance(answers, dict) else None
    if duplicate:
        diagnostic["category"] = "duplicate_generation"
    elif result.get("error") or (isinstance(answer, dict) and answer.get("error")):
        diagnostic["category"] = "upstream_error"
    elif result.get("refusal") or (
        isinstance(answer, dict) and (answer.get("type") == "refusal" or answer.get("refusal"))
    ):
        diagnostic["category"] = "provider_refusal"
    elif isinstance(answer, dict) and answer.get("type") == "choice":
        try:
            extract_probabilities(model, result)
        except ValueError:
            diagnostic["category"] = "invalid_choice"
        else:
            diagnostic["category"] = "invalid_billing_or_generation"
    return diagnostic


def _parallel(
    permitted: list[Envelope],
    workers: int,
    transport: Transport,
    handle: TextIO,
    generations: set[str],
    on_failure: Callable[[str, Envelope], None] | None = None,
) -> None:
    """Bound in-flight calls. On first failure stop dispatch; drain validated receipts."""
    failed = Event()

    def request(req: Envelope) -> tuple[Envelope, str]:
        result = None
        try:
            result = transport(req["payload"])
            return result, _validate(req["model"], result)
        except Exception as exc:
            # Set before the future completes; never relay untrusted exception text.
            failed.set()
            if on_failure is not None:
                raise _FailedRequestError(_failure_metadata(req["model"], result, exc)) from None
            raise

    submitted = saved = errors = 0
    interrupted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future[tuple[Envelope, str]], Envelope] = {}
        while pending or (not failed.is_set() and submitted < len(permitted)):
            while not failed.is_set() and len(pending) < workers and submitted < len(permitted):
                req = permitted[submitted]
                pending[pool.submit(request, req)] = req
                submitted += 1
            if not pending:
                break
            try:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                failed.set()
                interrupted = True
                continue
            for future in done:
                req = pending.pop(future)
                try:
                    result, generation = future.result()
                except Exception as exc:  # noqa: BLE001 - no provider text may escape
                    failed.set()
                    errors += 1
                    if on_failure is not None and isinstance(exc, _FailedRequestError):
                        on_failure(req["request_id"], exc.diagnostic)
                    continue
                try:
                    _save(handle, generations, req, result, generation)
                except CollectionError as exc:  # Duplicate receipt: preserve other valid calls in flight.
                    failed.set()
                    errors += 1
                    if on_failure is not None:
                        on_failure(req["request_id"], _failure_metadata(req["model"], result, exc, duplicate=True))
                    continue
                saved += 1
                if saved % 100 == 0:
                    print(f"Batch progress: {saved}/{len(permitted)} new responses saved", flush=True)
    if errors or interrupted:
        raise CollectionError(f"Batch stopped: {submitted} submitted; {saved} saved; {errors} failed. No retry.")


def _reconciled_usd(text: object) -> Decimal:
    if not isinstance(text, str) or not re.fullmatch(r"0(?:\.[0-9]{1,15})?", text):
        raise CollectionError("Reconciled USD must be a nonnegative decimal amount within the original reservation")
    amount = Decimal(text)
    if amount > _LUNA_MAX_CHARGE:
        raise CollectionError("Reconciled charge exceeds original reservation")
    return amount


def _read_reconciliation(path: Path, expected: dict[str, str]) -> tuple[str, Decimal] | None:
    with _cache(path, write=False) as stream:
        if stream is None:
            return None
        lines = list(stream)
        if len(lines) != 1 or not lines[0].endswith("\n"):
            raise CollectionError("Invalid or ambiguous Luna reconciliation journal")
        row = json.loads(lines[0], object_pairs_hook=_unique)
        if (
            not isinstance(row, dict)
            or set(row) != {"request_id", "original_charge_usd"}
            or not isinstance(row["request_id"], str)
            or row["request_id"] not in expected
        ):
            raise CollectionError("Invalid Luna reconciliation record")
        return row["request_id"], _reconciled_usd(row["original_charge_usd"])


def _confirm_retryable_failure(path: Path, rid: str) -> None:
    with _cache(path, write=False) as stream:
        if stream is None:
            return  # historical collector had no failure diagnostics
        for line in stream:
            row = json.loads(line, object_pairs_hook=_unique)
            if not line.endswith("\n") or not isinstance(row, dict) or not isinstance(row.get("request_id"), str):
                raise CollectionError("Invalid failure record; no Luna retry")
            if row["request_id"] == rid and row.get("category") not in {"timeout", "transport_or_local_failure"}:
                status = row.get("http_status")
                if (
                    row.get("category") != "http_status"
                    or type(status) is not int
                    or not (status in (408, 429) or 500 <= status < 600)
                ):
                    raise CollectionError("Recorded Luna failure is not eligible for retry")


def _luna_budgeted(
    path: Path,
    permitted: list[Envelope],
    workers: int,
    transport: Transport,
    handle: TextIO,
    expected: dict[str, str],
    prior_charge: Decimal,
    retry: tuple[str, Decimal] | None = None,
) -> None:
    """Reserve each in-flight request's maximum charge; resume only from known receipts.

    A write-ahead list of IDs prevents retrying potentially billed timeouts or
    interrupted calls as if they had been free. No prompt or secret is journaled.
    """
    handle.seek(0)
    seen = {}
    spent = prior_charge
    for rid, generation, response in _receipts(handle, expected):
        seen[rid] = generation
        spent += _luna_charge(response)
    generations = set(seen.values())
    journal = path.with_suffix(".attempts.jsonl")

    def record_failure(rid: str, diagnostic: Envelope) -> None:
        # The diagnostic is for future failures only; absence beside an old
        # orphan does not establish success, refusal, stage of failure or cost.
        with _cache(path.with_suffix(".failures.jsonl"), write=True) as stream:
            if stream is None:
                raise CollectionError("Cannot save private Luna failure metadata")
            stream.write(json.dumps({"request_id": rid, **diagnostic}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"Luna failure category: {diagnostic['category']}. See owner-only failure metadata.", file=sys.stderr)

    with _cache(journal, write=True) as attempts:
        if attempts is None:
            raise CollectionError("Luna reservation journal unavailable")
        counts: Counter[str] = Counter()
        for line in attempts:
            rid = json.loads(line)
            if not line.endswith("\n") or not isinstance(rid, str) or rid not in expected:
                raise CollectionError("Invalid Luna reservation journal; stop and reconcile")
            counts[rid] += 1
        ledger_path = path.with_suffix(".reconciliations.jsonl")
        recorded = _read_reconciliation(ledger_path, expected)
        if recorded is not None:
            if retry is not None or counts[recorded[0]] != 2:
                raise CollectionError("A Luna retry has already been authorized; no second retry")
            spent += recorded[1]
        if any(n != (2 if recorded and rid == recorded[0] else 1) for rid, n in counts.items()):
            raise CollectionError("Unexpected duplicate Luna reservation; stop and reconcile")
        if retry is not None:
            rid, usd = retry
            if set(counts) != set(seen) | {rid} or rid in seen or [r["request_id"] for r in permitted] != [rid]:
                raise CollectionError("Retry must name the single unscored, previously attempted Luna request")
            _confirm_retryable_failure(path.with_suffix(".failures.jsonl"), rid)
            if spent + usd + _LUNA_MAX_CHARGE > _LUNA_BUDGET:
                raise CollectionError("Luna budget cannot reserve the retry including its reconciled original charge")
            # The cache and attempts are exclusively locked. First durably
            # attest prior billing, then append the second reservation below.
            # A crash between these writes blocks recovery, never an extra send.
            with _cache(ledger_path, write=True) as ledger:
                if ledger is None or ledger.read():
                    raise CollectionError("Luna reconciliation already exists")
                ledger.write(json.dumps({"request_id": rid, "original_charge_usd": format(usd, "f")}) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            spent += usd
        elif set(counts) != set(seen):
            raise CollectionError("Unreconciled Luna charges or reservations; do not retry")
        start = 0
        while start < len(permitted):
            slots = min(workers, int((_LUNA_BUDGET - spent) // _LUNA_MAX_CHARGE))
            if slots <= 0:
                raise CollectionError("Luna budget cannot reserve another request; no further submissions")
            wave = permitted[start : start + slots]
            for req in wave:
                attempts.write(json.dumps(req["request_id"]) + "\n")
            attempts.flush()
            os.fsync(attempts.fileno())
            offset = handle.tell()
            _parallel(wave, len(wave), transport, handle, generations, on_failure=record_failure)
            handle.seek(offset)
            spent += sum((_luna_charge(response) for _, _, response in _receipts(handle, expected)), Decimal(0))
            start += len(wave)
            if start // 100 != (start - len(wave)) // 100:
                print(f"Luna progress: {start}/{len(permitted)} new responses saved; billed total ${spent}", flush=True)
        print(f"Luna billed total including prior receipts and canary: ${spent}", flush=True)


def main(argv: list[str] | None = None, *, transport: Transport = _post) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    canary = commands.add_parser("canary", help="one fixed, invented prefix per pinned model")
    batch = commands.add_parser("batch", help="official local annotations only")
    for sub in (canary, batch):
        sub.add_argument(
            "--cohort",
            choices=("pilot", "luna"),
            default="pilot",
            help="historical three-model pilot (default), or OpenAI Luna only, with separate cache and canary",
        )
        sub.add_argument("--cache", required=True, type=Path, help="owner-only local response .jsonl")
        sub.add_argument("--max-requests", required=True, type=int, help="maximum new requests in this invocation")
        sub.add_argument("--allow-paid", action="store_true", help="authorize real, billable requests")
    batch.add_argument("--annotations", required=True, type=Path)
    batch.add_argument("--canary-cache", required=True, type=Path)
    batch.add_argument(
        "--workers", type=int, choices=range(1, 13), default=1, help="maximum concurrent requests (default 1)"
    )
    batch.add_argument("--allow-gated-provider-processing", action="store_true", help="confirm permission to send text")
    batch.add_argument("--retry-reconciled-request", help="one Luna request ID verified as attempted but unscored")
    batch.add_argument("--reconciled-charge-usd", help="independently verified original charge for this exact request")
    args = parser.parse_args(argv)
    try:
        models = (LUNA_MODEL,) if args.cohort == "luna" else MODEL_IDS
        prior_charge = Decimal(0)
        retry = None
        if args.max_requests < 0:
            raise CollectionError("Request cap must not be negative")
        if args.mode == "batch":
            if args.retry_reconciled_request is not None or args.reconciled_charge_usd is not None:
                if (
                    args.cohort != "luna"
                    or args.max_requests != 1
                    or args.workers != 1
                    or not re.fullmatch("[0-9a-f]{64}", args.retry_reconciled_request or "")
                ):
                    raise CollectionError(
                        "Explicit recovery requires a Luna batch, one valid request ID and one worker/request"
                    )
                retry = (args.retry_reconciled_request, _reconciled_usd(args.reconciled_charge_usd))
            if args.allow_paid and not args.allow_gated_provider_processing:
                raise CollectionError("Confirm permission for gated text using --allow-gated-provider-processing")
            if args.cache.resolve() in (args.annotations.resolve(), args.canary_cache.resolve()):
                raise CollectionError("Dataset, batch cache and canary cache must be separate")
            if args.allow_paid:
                prior_charge = _require_canary(args.canary_cache, models)
            requests = [r for case in official_cases(args.annotations) for r in requests_for(case, models=models)]
        else:
            requests = canary_requests(models)

        expected = {r["request_id"]: r["model"] for r in requests}
        if len(expected) != len(requests):
            raise CollectionError("Duplicate prepared request")
        with _cache(args.cache, write=args.allow_paid) as handle:
            seen = _seen(handle, expected)
            todo = [r for r in requests if r["request_id"] not in seen]
            permitted = todo[: args.max_requests]
            if retry is not None:
                permitted = [r for r in todo if r["request_id"] == retry[0]]
                if not permitted:
                    raise CollectionError("Recovery ID is unknown or already has a valid receipt")
            print(
                f"Prepared: {len(expected)}; cached: {len(seen)}; remaining: {len(todo)}; within cap: {len(permitted)}",
                flush=True,
            )
            if not args.allow_paid:
                if args.cohort == "luna":
                    # Rough estimate only: exact tokenization and wrapper overhead are provider-specific.
                    input_bytes = sum(
                        len(
                            json.dumps(
                                [r["payload"]["questions"], r["payload"]["state"]],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        )
                        for r in permitted
                    )
                    print(
                        f"Luna within-cap estimate: ~${input_bytes / 4 * 1e-7:.5f} "
                        "(4 UTF-8 bytes/token, $0.10/M input); not a billing limit."
                    )
                print("Preflight only. No requests sent; no canary or provider access tested.")
                return 0
            if handle is None:
                raise CollectionError("Private response cache unavailable")
            generations = set(seen.values())
            if args.cohort == "luna":
                _luna_budgeted(
                    args.cache, permitted, getattr(args, "workers", 1), transport, handle, expected, prior_charge, retry
                )
            elif args.mode == "batch" and args.workers > 1:
                _parallel(permitted, args.workers, transport, handle, generations)
            else:
                for req in permitted:
                    result = transport(req["payload"])
                    _save(handle, generations, req, result, _validate(req["model"], result))
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
