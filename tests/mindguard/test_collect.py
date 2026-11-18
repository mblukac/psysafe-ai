from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest

from benchmarks.mindguard.collect import canary_requests, main
from benchmarks.mindguard.data import Case, digest
from benchmarks.mindguard.providers import MODEL_IDS, synthetic_response


def fake(payload: dict[str, Any]) -> dict[str, Any]:
    result = synthetic_response(payload["model"], {"safe": 0.9, "self_harm": 0.09, "harm_others": 0.01})
    provider = dict(zip(MODEL_IDS, ["TypeSafe", "Google AI Studio", "Liquid"], strict=True))
    return {**result, "provider": provider[payload["model"]], "id": "gen-test-" + digest(payload)}


def never_send(_: object) -> dict[str, Any]:
    pytest.fail("Preflight, failed validation or fully cached run must not use inference")


def canary(path: Path, cap: int = 3) -> list[str]:
    return ["canary", "--cache", str(path), "--max-requests", str(cap)]


def test_default_canary_preflight_is_offline_and_creates_no_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "receipt.jsonl"
    assert main(canary(path, 1), transport=never_send) == 0
    assert "within cap: 1" in capsys.readouterr().out
    assert not path.exists()


def test_batch_requires_both_permissions_valid_canary_and_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache, receipts = tmp_path / "batch.jsonl", tmp_path / "canary.jsonl"
    batch = [
        "batch",
        "--annotations",
        str(tmp_path / "dataset.jsonl"),
        "--cache",
        str(cache),
        "--canary-cache",
        str(receipts),
        "--max-requests",
        "1",
    ]
    # Tests use an invented case; the CLI's real path requires official count/chain checks.
    monkeypatch.setattr(
        "benchmarks.mindguard.collect.official_cases",
        lambda _: [
            Case("test", 1, ("safe",) * 3, (("user", "Synthetic case"),)),
        ],
    )
    assert main(batch, transport=never_send) == 0
    assert main([*batch, "--allow-paid"], transport=never_send) == 2
    approved = [*batch, "--allow-paid", "--allow-gated-provider-processing"]
    assert main(approved, transport=never_send) == 2
    assert main([*canary(receipts, 1), "--allow-paid"], transport=fake) == 0
    assert main(approved, transport=never_send) == 2
    assert main([*canary(receipts), "--allow-paid"], transport=fake) == 0
    assert main(approved, transport=fake) == 0
    assert len(cache.read_text().splitlines()) == 1
    assert "Synthetic case" not in cache.read_text()
    assert cache.stat().st_mode & 0o077 == 0
    assert main(approved, transport=fake) == 0
    assert len(cache.read_text().splitlines()) == 2
    # Wrong generation for the expected pinned model invalidates even a full canary cache.
    records = [json.loads(line) for line in receipts.read_text().splitlines()]
    records[0]["response"]["model"] = MODEL_IDS[2]
    receipts.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert main(approved, transport=never_send) == 2


@pytest.mark.parametrize(
    "problem", ["shared", "symlink", "bad_id", "duplicate", "partial", "wrong_provider", "error", "refusal"]
)
def test_invalid_cache_or_receipt_never_sent(tmp_path: Path, problem: str) -> None:
    path = tmp_path / "rows.jsonl"
    r = canary_requests()[0]
    record = {"request_id": r["request_id"], "model": r["model"], "response": fake(r["payload"])}
    if problem == "bad_id":
        record["request_id"] = "not-a-request"
    if problem == "wrong_provider":
        record["response"]["provider"] = "unknown"
    if problem == "error":
        record["response"] = {"error": {"message": "sensitive response text"}}
    if problem == "refusal":
        record["response"]["answers"]["mindguard"] = {"type": "refusal"}
    line = json.dumps(record) + "\n"
    path.write_text(line * (2 if problem == "duplicate" else 1))
    path.chmod(0o600)
    if problem == "shared":
        path.chmod(0o644)
    if problem == "symlink":
        link = tmp_path / "link.jsonl"
        link.symlink_to(path)
        path = link
    if problem == "partial":
        path.write_text(line.rstrip())
    assert main([*canary(path), "--allow-paid"], transport=never_send) == 2


def test_transport_failure_stops_immediately_and_redacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls = []
    cache = tmp_path / "receipt.jsonl"

    def fails_on_second(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload["model"])
        if len(calls) == 2:
            raise OSError("sensitive prefix or authorization material")
        return fake(payload)

    assert main([*canary(cache), "--allow-paid"], transport=fails_on_second) == 2
    assert calls == list(MODEL_IDS[:2])
    assert len(cache.read_text().splitlines()) == 1
    out = capsys.readouterr()
    assert "sensitive prefix" not in out.out + out.err
    assert main([*canary(cache), "--allow-paid"], transport=fake) == 0
    assert main([*canary(cache), "--allow-paid"], transport=never_send) == 0


def test_parallel_batch_drains_failure_and_resumes_within_cap(tmp_path, monkeypatch, capsys):
    receipts, cache = tmp_path / "synthetic-canary.jsonl", tmp_path / "private.jsonl"
    assert main([*canary(receipts), "--allow-paid"], transport=fake) == 0
    monkeypatch.setattr(
        "benchmarks.mindguard.collect.official_cases",
        lambda _: [Case(f"fixture-{i}", 1, ("safe",) * 3, (("user", f"Invented {i}."),)) for i in range(35)],
    )
    args = [
        "batch",
        "--cache",
        str(cache),
        "--annotations",
        str(tmp_path / "synthetic.parquet"),
        "--canary-cache",
        str(receipts),
        "--workers",
        "3",
        "--max-requests",
        "7",
    ]
    assert main(args, transport=never_send) == 0
    assert not cache.exists()
    approved = [*args, "--allow-paid", "--allow-gated-provider-processing"]
    stop, started = Event(), Barrier(3, timeout=5)
    monkeypatch.setattr("benchmarks.mindguard.collect.Event", lambda: stop)

    def refusal(payload):
        started.wait()
        response = fake(payload)
        if payload["model"] == MODEL_IDS[0]:
            response["answers"]["mindguard"] = {"type": "refusal", "text": "PRIVATE MARKER"}
        else:
            assert stop.wait(5)
        return response

    assert main(approved, transport=refusal) == 2
    rows = [json.loads(s) for s in cache.read_text().splitlines()]
    assert {r["model"] for r in rows} == set(MODEL_IDS[1:])
    out = capsys.readouterr()
    assert "3 submitted; 2 saved; 1 failed" in out.err
    assert "PRIVATE MARKER" not in out.out + out.err + cache.read_text()
    stop.clear()
    assert main(approved, transport=fake) == 0
    assert len(cache.read_text().splitlines()) == 9
    # Start over with an empty, separate private cache: prove progress reporting and no duplication.
    args_all = [*approved, "--cache", str(tmp_path / "other.jsonl"), "--max-requests", "1000"]
    assert main(args_all, transport=fake) == 0
    rows = [json.loads(line) for line in (tmp_path / "other.jsonl").read_text().splitlines()]
    assert len({r["request_id"] for r in rows}) == len(rows) == 105
    assert "100/105" in capsys.readouterr().out


@pytest.mark.parametrize(("mode", "workers"), [("canary", 2), ("batch", 0), ("batch", 13)])
def test_invalid_concurrency_flags_rejected_before_access(tmp_path, mode, workers):
    args = [mode, "--cache", str(tmp_path / "cache.jsonl"), "--max-requests", "3", "--workers", str(workers)]
    if mode == "batch":
        args.extend(["--annotations", str(tmp_path / "unused"), "--canary-cache", str(tmp_path / "unused-canary")])
    with pytest.raises(SystemExit) as exit_info:
        main(args, transport=never_send)
    assert exit_info.value.code == 2
