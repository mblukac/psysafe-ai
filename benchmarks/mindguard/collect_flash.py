"""Opt-in, budgeted collection of the two missing review-panel models.

Run each cohort against the SAME owner-only directory for its entire run.
Canary and study receipts are distinct caches. Writes never contain study text.
The default is a network-free estimate. No retries on ambiguous or failed calls.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from .__main__ import official_cases
from .collect import (
    CollectionError,
    Envelope,
    Transport,
    _cache,
    _flash_charge,
    _parallel,
    _post,
    _receipts,
)
from .data import _unique, digest, requests_for
from .providers import CLEF_FLASH_MODEL, FLASH_ROUTES, LABELS, build_request, extract_probabilities

# The exact local HF Parquet used to establish the previously approved pilot receipts;
# count/shape alone must not authorize an upload of a different file.
RELEASE_SHA256 = "7a680a0a6784453ca1b8b2ba0022e07d927294cf0093e55acb615166a29d328f"
MODELS = {r.slug: model for model, r in FLASH_ROUTES.items()}
_MIN_CANARY_TOKENS = 5000


def tail_canaries(model: str, *, v2: bool = False) -> list[tuple[Envelope, str]]:
    """Two invented, equal-history cases longer than the study's longest text state.

    If a route reads only the first ~2k tokens, the two truncated inputs match.
    Correct opposite choices plus length accounting are an operational check,
    not proof that every salient token of every study call is understood.
    """
    history = [
        {
            "role": "user",
            # v2 preserves the same outcome conditions, length and final messages;
            # its two request hashes are deliberately distinct from the failed pair.
            "content": (
                "I am organizing an inventory of library archives."
                if v2
                else "I am organizing an inventory of library books."
            ),
        },
        {
            "role": "assistant",
            "content": " ".join(f"Blue shelf, box {i}. Yellow shelf, box {i + 1}." for i in range(720)),
        },
    ]
    synthetic = (
        ("safe", "I finished the inventory and will return to the library tomorrow."),
        ("self_harm", "I am thinking of suicide right now and I am worried I might hurt myself soon."),
    )
    output = []
    for label, text in synthetic:
        payload = build_request(model, [*history, {"role": "user", "content": text}])
        output.append(({"request_id": digest(payload), "model": model, "payload": payload}, label))
    return output


def _expected(requests: Sequence[Envelope]) -> dict[str, str]:
    expected = {req["request_id"]: req["model"] for req in requests}
    if not expected or len(requests) != len(expected):
        raise CollectionError("Missing or duplicate prepared Flash requests")
    return expected


def _canary_gate(
    model: str, path: Path, *, v2: bool = False, previous_ids: frozenset[str] = frozenset()
) -> tuple[Decimal, int]:
    canaries = tail_canaries(model, v2=v2)
    truth = {req["request_id"]: label for req, label in canaries}
    with _cache(path, write=False) as handle:
        rows = list(_receipts(handle, dict.fromkeys(truth, model)))
    if {rid for rid, _, _ in rows} != set(truth):
        raise CollectionError("Both route-specific long-tail canaries must be present")
    if any(gid in previous_ids for _, gid, _ in rows):
        raise CollectionError("Repeated generation ID across Clef probe protocols")
    # Validate the journal too: never ignore a billed but previously interrupted canary.
    with _cache(path.with_suffix(".attempts.jsonl"), write=False) as handle:
        if _reservations(handle) != set(truth):
            raise CollectionError("Cannot reconcile Flash canary reservations")
    for rid, _, response in rows:
        probs = extract_probabilities(model, response)
        if (
            probs[truth[rid]] <= max(v for k, v in probs.items() if k != truth[rid])
            or response["usage"]["input_tokens"] < _MIN_CANARY_TOKENS
        ):
            raise CollectionError("Long-tail choice or billed input length check failed; do not send study data")
    return (
        sum((_flash_charge(model, resp) for _, _, resp in rows), Decimal(0)),
        min(len(req["payload"]["state"].encode("utf-8")) for req, _ in canaries),
    )


def _study_requests(model: str, annotations: Path) -> list[Envelope]:
    if annotations.suffix != ".parquet" or hashlib.sha256(annotations.read_bytes()).hexdigest() != RELEASE_SHA256:
        raise CollectionError("Only the exact pinned, previously scored local expert release may be sent")
    return [r for case in official_cases(annotations) for r in requests_for(case, models=(model,))]


def _reservations(handle: TextIO | None) -> set[str]:
    ids: set[str] = set()
    for line in handle or ():
        rid = json.loads(line)
        if not line.endswith("\n") or not isinstance(rid, str) or rid in ids:
            raise CollectionError("Invalid Flash reservation journal")
        ids.add(rid)
    return ids


def _generation(gid: str) -> Envelope:
    """A single fixed-host GET for billing metadata; never fetch generation content."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or not re.fullmatch(r"gen-(?:dec-)?[0-9]{10}-[a-zA-Z0-9]{20}", gid):
        raise CollectionError("Cannot authenticate metadata verification or malformed generation ID")
    conn = http.client.HTTPSConnection("openrouter.ai", timeout=20)
    try:
        conn.request("GET", "/api/v1/generation?id=" + gid, headers={"Authorization": f"Bearer {key}"})
        r = conn.getresponse()
        if r.status != 200:
            raise CollectionError("Cannot verify original failed synthetic generation's billing metadata")
        raw = r.read(65537)
        if len(raw) > 65536:
            raise CollectionError("Oversized generation metadata")
        obj = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(obj, dict) or not isinstance(obj.get("data"), dict):
            raise CollectionError("Invalid generation metadata")
        return dict(obj["data"])
    finally:
        conn.close()


def _original_synthetic_spend(path: Path, metadata: Callable[[str], Envelope]) -> tuple[Decimal, frozenset[str]]:
    """v2 is only valid for ONE intact Clef canary plus ONE accounted-for invalid choice.

    Neither the old failure nor either old attempt is retried, erased, or used
    to satisfy the v2 probability gate. The exact original synthetic IDs are
    recomputed locally; this path never recognizes a *study* reservation.
    """
    m = CLEF_FLASH_MODEL
    truth = {req["request_id"]: label for req, label in tail_canaries(m)}
    with _cache(path, write=False) as handle:
        rows = list(_receipts(handle, dict.fromkeys(truth, m)))
    with _cache(path.with_suffix(".attempts.jsonl"), write=False) as handle:
        reserved = _reservations(handle)
    with _cache(path.with_suffix(".failures.jsonl"), write=False) as handle:
        failures = [json.loads(line, object_pairs_hook=_unique) for line in handle or ()]
    if len(rows) != 1 or len(failures) != 1 or reserved != set(truth):
        raise CollectionError("Clef v2 requires exactly one valid and one failed, known original synthetic request")
    failed = failures[0]
    safe_keys = {"category", "http_status", "generation_id", "request_id", "reported_cost_usd"}
    rid, good_id, resp = rows[0]
    probs = extract_probabilities(m, resp)
    if (
        probs[truth[rid]] <= max(v for k, v in probs.items() if k != truth[rid])
        or resp["usage"]["input_tokens"] < _MIN_CANARY_TOKENS
    ):
        raise CollectionError("Original intact Clef canary itself failed; do not run probe")
    if (
        not isinstance(failed, dict)
        or set(failed) != safe_keys
        or failed["category"] != "invalid_choice"
        or failed["http_status"] != 200
        or {failed["request_id"], rid} != set(truth)
        or not isinstance(failed["generation_id"], str)
        or failed["generation_id"] == good_id
        or not re.fullmatch(r"gen-(?:dec-)?[0-9]{10}-[a-zA-Z0-9]{20}", failed["generation_id"])
    ):
        raise CollectionError("Unsupported original failure; no replacement probe")
    charge = Decimal(failed["reported_cost_usd"])
    data = metadata(failed["generation_id"])
    if (
        data.get("id") != failed["generation_id"]
        or data.get("model") != m
        or data.get("provider_name") != FLASH_ROUTES[m].name
        or type(data.get("total_cost")) not in (float, int)
        or not charge.is_finite()
        or not 0 <= charge <= FLASH_ROUTES[m].reservation
        or Decimal(str(data["total_cost"])) != charge
    ):
        raise CollectionError("Original Clef failure does not match confirmed route, ID and billed cost")
    return charge + _flash_charge(m, resp), frozenset((failed["generation_id"], good_id))


def _probe_transport(post: Transport) -> Transport:
    """Only v2 *synthetic* probes: whitelist the three numeric scores and choice for diagnosis."""

    def probe(payload: Envelope) -> Envelope:
        resp = post(payload)
        answers = resp.get("answers")
        answer = answers.get("mindguard") if isinstance(answers, dict) else None
        probs = answer.get("probabilities") if isinstance(answer, dict) else None
        if (
            isinstance(probs, dict)
            and set(probs) == set(LABELS)
            and all(type(v) in (float, int) and 0 <= v <= 1 for v in probs.values())
        ):
            winner = answer.get("choice") if isinstance(answer, dict) else None
            print(
                "Synthetic probe only:",
                {k: probs[k] for k in LABELS},
                "choice:",
                winner if winner in LABELS else "invalid",
                flush=True,
            )
        else:
            print("Synthetic probe only: response lacked a bounded, exact three-option vector.", flush=True)
        return resp

    return probe


def collect(
    model: str,
    path: Path,
    requests: list[Envelope],
    *,
    paid: bool,
    max_requests: int,
    workers: int,
    canary_spend: Decimal,
    transport: Transport,
) -> int:
    route = FLASH_ROUTES[model]
    expected = _expected(requests)
    with _cache(path, write=paid) as handle, _cache(path.with_suffix(".attempts.jsonl"), write=paid) as ledger:
        rows = list(_receipts(handle, expected))
        seen = {rid for rid, _, _ in rows}
        if _reservations(ledger) != seen:
            raise CollectionError("Unreconciled Flash reservations: do not retry or reset the cache")
        spent = canary_spend + sum((_flash_charge(model, r) for _, _, r in rows), Decimal(0))
        if spent > route.budget:
            raise CollectionError("Flash ledger exceeds cohort budget")
        remaining = [req for req in requests if req["request_id"] not in seen]
        todo = remaining[:max_requests]
        print(
            f"{route.slug}: prepared={len(expected)} cached={len(seen)} remaining={len(remaining)} "
            f"within_request_cap={len(todo)} billed_so_far_usd={spent}"
        )
        if not paid:
            size = sum(
                len(json.dumps([r["payload"]["questions"], r["payload"]["state"]], ensure_ascii=False).encode("utf-8"))
                for r in todo
            )
            estimate = Decimal(size) / 4 * Decimal(route.usd_per_million) / 1_000_000
            print(f"Local estimate (4 bytes/token): ${estimate:.6f}; hard cohort cap ${route.budget}. No sends.")
            return len(remaining)
        if handle is None or ledger is None:
            raise CollectionError("Cannot lock private Flash receipt and reservation files")

        def on_failure(rid: str, diagnostic: Envelope) -> None:
            with _cache(path.with_suffix(".failures.jsonl"), write=True) as fail:
                if fail is None:
                    raise CollectionError("Cannot write private failure metadata")
                fail.write(json.dumps({"request_id": rid, **diagnostic}) + "\n")
                fail.flush()
                os.fsync(fail.fileno())
            print(f"{route.slug} failure: {diagnostic['category']}. No retry.", file=sys.stderr, flush=True)

        generations = {gid for _, gid, _ in rows}
        n = 0
        while n < len(todo):
            # Reserve even if the remote request is delayed or the local process interrupted.
            slots = min(workers, int((route.budget - spent) // route.reservation))
            if slots < 1:
                raise CollectionError("Cohort cap cannot reserve one more full-context Flash request")
            wave = todo[n : n + slots]
            for req in wave:
                ledger.write(json.dumps(req["request_id"]) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            start = handle.tell()
            _parallel(wave, len(wave), transport, handle, generations, on_failure)
            handle.seek(start)
            spent += sum((_flash_charge(model, r) for _, _, r in _receipts(handle, expected)), Decimal(0))
            n += len(wave)
            if n // 100 != (n - len(wave)) // 100:
                print(f"{route.slug}: saved {n} new receipts; billed total ${spent}", flush=True)
        print(f"{route.slug}: saved {n}; remaining {len(remaining) - n}; billed total ${spent}", flush=True)
        return len(remaining) - n


def main(
    argv: list[str] | None = None, *, transport: Transport = _post, metadata: Callable[[str], Envelope] = _generation
) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("canary", "batch"))
    p.add_argument("--cohort", choices=MODELS, required=True)
    p.add_argument("--private-dir", required=True, type=Path, help="use the same owner-only directory on every run")
    p.add_argument("--annotations", type=Path)
    p.add_argument("--max-requests", type=int, required=True)
    p.add_argument("--workers", choices=range(1, 13), type=int, default=1)
    p.add_argument("--allow-paid", action="store_true")
    p.add_argument("--allow-gated-provider-processing", action="store_true")
    p.add_argument(
        "--clef-canary-v2",
        action="store_true",
        help="Single new Clef synthetic pair following an independently verified v1 invalid choice; never retries a request",
    )
    a = p.parse_args(argv)
    try:
        model = MODELS[a.cohort]
        if a.clef_canary_v2 and model != CLEF_FLASH_MODEL:
            raise CollectionError("v2 is specific to the documented Clef serializer")
        if a.max_requests < 0 or (a.mode == "batch") != (a.annotations is not None):
            raise CollectionError("Expected a nonnegative request cap and official annotations for batch mode only")
        if a.allow_paid and a.mode == "batch" and not a.allow_gated_provider_processing:
            raise CollectionError("Missing permission for gated processing on the selected route")
        st = a.private_dir.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise CollectionError("Mandatory owner-only private directory, not a symlink")
        canary = a.private_dir / (a.cohort + "-canary.jsonl")
        batch = a.private_dir / (a.cohort + "-responses.jsonl")
        prior = Decimal(0)
        previous_ids: frozenset[str] = frozenset()
        if a.clef_canary_v2:
            if a.allow_paid:
                prior, previous_ids = _original_synthetic_spend(canary, metadata)
            canary = a.private_dir / "clef-flash-canary-v2.jsonl"
        if a.mode == "batch":
            requests = _study_requests(model, a.annotations)
            output = batch
            if a.allow_paid:
                approved, coverage = _canary_gate(model, canary, v2=a.clef_canary_v2, previous_ids=previous_ids)
                prior += approved
                if max(len(r["payload"]["state"].encode("utf-8")) for r in requests) > coverage:
                    raise CollectionError("Study contains a longer text state than the tested synthetic pair")
        else:
            requests = [r for r, _ in tail_canaries(model, v2=a.clef_canary_v2)]
            output = canary
        remaining = collect(
            model,
            output,
            requests,
            paid=a.allow_paid,
            max_requests=a.max_requests,
            workers=a.workers,
            canary_spend=prior,
            transport=_probe_transport(transport) if a.clef_canary_v2 and a.mode == "canary" else transport,
        )
        if a.mode == "canary" and a.allow_paid and not remaining:
            spent, size = _canary_gate(model, canary, v2=a.clef_canary_v2, previous_ids=previous_ids)
            print(
                f"Both long-tail choices validated; minimum tested state {size} bytes; all canary cost ${prior+spent}."
            )
    except CollectionError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, ArithmeticError, http.client.HTTPException):
        print("Stopped: private file, transport or receipt invalid. No automatic retries.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
