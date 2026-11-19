from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.collect import _post, canary_requests, main
from benchmarks.mindguard.data import Case, digest, requests_for
from benchmarks.mindguard.luna import load_luna_scores
from benchmarks.mindguard.providers import LUNA_MODEL, MODEL_IDS, synthetic_response


def case(index: int) -> Case:
    return Case("invented", index, ("safe",) * 3, (("user", f"Invented message {index}"),))


def fake_luna(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["model"] == LUNA_MODEL
    assert payload["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "max_price": {"prompt": "0.10", "completion": "0", "request": "0"},
    }
    return {
        **synthetic_response(LUNA_MODEL, {"safe": 0.9, "self_harm": 0.1, "harm_others": 0.0}),
        "provider": "OpenAI",
        "id": "gen-synthetic-" + digest(payload),
        "usage": {"cost": 0.000132},
    }


def test_opt_in_and_shared_rubric() -> None:
    old = list(requests_for(case(1)))
    assert tuple(r["model"] for r in old) == MODEL_IDS
    assert all("provider" not in r["payload"] for r in old)
    assert [r["model"] for r in canary_requests()] == list(MODEL_IDS)
    (newer,) = requests_for(case(1), models=(LUNA_MODEL,))
    assert old[0]["payload"]["questions"] == newer["payload"]["questions"]
    assert old[0]["payload"]["state"] == newer["payload"]["state"]
    assert "votes" not in newer["payload"]
    assert newer["request_id"] == digest(["mindguard-case-request-v2", "invented", 1, newer["payload"]])


def test_luna_single_canary_authorization_and_resume(tmp_path, monkeypatch, capsys) -> None:
    canary, cache = tmp_path / "luna-canary.jsonl", tmp_path / "luna.jsonl"
    requests = [case(1), case(2)]
    monkeypatch.setattr("benchmarks.mindguard.collect.official_cases", lambda _: requests)
    canary_args = ["canary", "--cohort", "luna", "--cache", str(canary), "--max-requests", "1"]
    batch = [
        "batch",
        "--cohort",
        "luna",
        "--annotations",
        str(tmp_path / "data.parquet"),
        "--canary-cache",
        str(canary),
        "--cache",
        str(cache),
        "--max-requests",
        "1",
    ]

    def prohibited(_: object) -> dict[str, Any]:
        pytest.fail("unexpected transmission")

    assert main(batch, transport=prohibited) == 0
    assert "Luna within-cap estimate" in capsys.readouterr().out
    assert not cache.exists()
    assert main([*batch, "--allow-paid", "--allow-gated-provider-processing"], transport=prohibited) == 2
    assert main([*canary_args, "--allow-paid"], transport=fake_luna) == 0
    assert main([*batch, "--allow-paid"], transport=prohibited) == 2
    approved = [*batch, "--allow-paid", "--allow-gated-provider-processing"]
    assert main(approved, transport=fake_luna) == 0
    with pytest.raises(ValueError, match="Incomplete"):
        load_luna_scores(requests, cache)
    assert main(approved, transport=fake_luna) == 0
    assert main(approved, transport=prohibited) == 0
    assert load_luna_scores(requests, cache) == {("invented", 1): 0.1, ("invented", 2): 0.1}
    assert cache.stat().st_mode & 0o077 == 0
    assert "Invented message" not in cache.read_text()


def test_budget_reserves_full_context_and_includes_canary(tmp_path, monkeypatch) -> None:
    canary = tmp_path / "canary.jsonl"
    cache = tmp_path / "luna.jsonl"

    def max_cost(payload):
        return fake_luna(payload) | {"usage": {"cost": 0.105}}

    assert (
        main(
            ["canary", "--cohort", "luna", "--cache", str(canary), "--max-requests", "1", "--allow-paid"],
            transport=max_cost,
        )
        == 0
    )
    monkeypatch.setattr("benchmarks.mindguard.collect.official_cases", lambda _: [case(1), case(2), case(3)])
    args = [
        "batch",
        "--cohort",
        "luna",
        "--annotations",
        str(tmp_path / "data.parquet"),
        "--cache",
        str(cache),
        "--canary-cache",
        str(canary),
        "--allow-paid",
        "--allow-gated-provider-processing",
        "--max-requests",
        "3",
        "--workers",
        "12",
    ]
    assert main(args, transport=max_cost) == 2
    # 0.105 canary + 0.105 batch = 0.210. Even one more call might exceed 0.30.
    assert len(cache.read_text().splitlines()) == 1
    assert main(args, transport=lambda _: pytest.fail("budget exhausted")) == 2


def test_refusal_or_ambiguous_charge_is_not_automatically_retried(tmp_path) -> None:
    canary = tmp_path / "canary.jsonl"
    args = ["canary", "--cohort", "luna", "--cache", str(canary), "--max-requests", "1", "--allow-paid"]
    assert main(args, transport=lambda payload: fake_luna(payload) | {"usage": {}}) == 2
    assert canary.read_text() == ""
    assert len(canary.with_suffix(".attempts.jsonl").read_text().splitlines()) == 1
    assert main(args, transport=lambda _: pytest.fail("ambiguous prior charge")) == 2


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    canary = tmp_path / "canary.jsonl"
    cache = tmp_path / "luna.jsonl"
    assert (
        main(
            ["canary", "--cohort", "luna", "--cache", str(canary), "--max-requests", "1", "--allow-paid"],
            transport=fake_luna,
        )
        == 0
    )
    cases = [case(1), case(2), case(3)]
    monkeypatch.setattr("benchmarks.mindguard.collect.official_cases", lambda _: cases)
    args = [
        "batch",
        "--cohort",
        "luna",
        "--allow-paid",
        "--allow-gated-provider-processing",
        "--cache",
        str(cache),
        "--canary-cache",
        str(canary),
        "--annotations",
        str(tmp_path / "fixture.parquet"),
        "--max-requests",
        "1",
    ]
    assert main(args, transport=fake_luna) == 0

    def timeout(_):
        raise TimeoutError

    assert main(args, transport=timeout) == 2
    ids = [next(requests_for(c, models=(LUNA_MODEL,)))["request_id"] for c in cases]
    return args, cache, ids


def test_explicit_single_retry_is_durable_and_includes_original_charge(recovery, capsys):
    args, cache, ids = recovery
    retry = [*args, "--retry-reconciled-request", ids[1], "--reconciled-charge-usd", "0.010"]

    def only_orphan(payload):
        assert json.loads(payload["state"])["conversation"][-1]["content"] == "Invented message 2"
        return fake_luna(payload)

    assert main(retry, transport=only_orphan) == 0
    assert json.loads(cache.with_suffix(".reconciliations.jsonl").read_text()) == {
        "request_id": ids[1],
        "original_charge_usd": "0.010",
    }
    assert cache.with_suffix(".reconciliations.jsonl").stat().st_mode & 0o077 == 0
    assert [json.loads(line) for line in cache.with_suffix(".attempts.jsonl").read_text().splitlines()] == [
        ids[0],
        ids[1],
        ids[1],
    ]
    assert main(retry, transport=lambda _: pytest.fail("duplicate retry")) == 2
    assert main(args, transport=fake_luna) == 0
    assert "including prior receipts and canary: $0.010528" in capsys.readouterr().out


@pytest.mark.parametrize(
    "issue",
    [
        "unattempted",
        "scored",
        "too_large",
        "negative",
        "no_amount",
        "two_missing",
        "refusal",
        "budget",
        "crash",
        "second_failure",
    ],
)
def test_recovery_does_not_loosen_refusal_budget_or_journal_guards(recovery, issue, monkeypatch):
    args, cache, ids = recovery
    rid = ids[1]
    amount = "0.0001"
    if issue == "unattempted":
        rid = ids[2]
    if issue == "scored":
        rid = ids[0]
    if issue == "too_large":
        amount = "0.11"
    if issue == "negative":
        amount = "-0.001"
    if issue == "budget":
        from decimal import Decimal

        monkeypatch.setattr("benchmarks.mindguard.collect._LUNA_BUDGET", Decimal("0.1053"))
    if issue == "two_missing":
        with cache.with_suffix(".attempts.jsonl").open("a") as h:
            h.write(json.dumps(ids[2]) + "\n")
    if issue == "refusal":
        cache.with_suffix(".failures.jsonl").write_text(
            json.dumps({"request_id": rid, "category": "provider_refusal"}) + "\n"
        )
    if issue == "crash":
        ledger = cache.with_suffix(".reconciliations.jsonl")
        ledger.write_text(json.dumps({"request_id": rid, "original_charge_usd": "0"}) + "\n")
        ledger.chmod(0o600)
    retry = [*args, "--retry-reconciled-request", rid]
    if issue != "no_amount":
        retry.extend(["--reconciled-charge-usd", amount])
    if issue == "second_failure":

        def timeout(_):
            raise TimeoutError

        assert main(retry, transport=timeout) == 2
    assert main(retry, transport=lambda _: pytest.fail("unsafe retry")) == 2


@pytest.mark.parametrize(
    "kind", ["refusal", "invalid_choice", "timeout", "http", "invalid_wire", "wrong_route", "bad_id"]
)
def test_luna_errors_are_classified_without_persisting_sensitive_upstream_data(tmp_path, monkeypatch, capsys, kind):
    marker = "PRIVATE EXAMPLE not safe to log"
    cache = tmp_path / "luna.jsonl"
    args = ["canary", "--cohort", "luna", "--cache", str(cache), "--max-requests", "1", "--allow-paid"]
    generation_id = "gen-dec-1700000000-0123456789abcdefghij"

    class FakeResponse:
        status = 429 if kind == "http" else 200

        def read(self, _: int) -> bytes:
            if kind == "http":
                pytest.fail("HTTP error bodies must not be copied or parsed")
            return marker.encode()

    class FakeConnection:
        def __init__(self, *_, **__):
            pass

        def request(self, *_, **__):
            pass

        def getresponse(self):
            return FakeResponse()

        def close(self):
            pass

    def transport(payload):
        if kind == "timeout":
            raise TimeoutError(marker)
        if kind in ("http", "invalid_wire"):
            return _post(payload)
        result = fake_luna(payload) | {"id": generation_id, "private_extra": marker}
        if kind == "wrong_route":
            result["provider"] = marker
        elif kind == "invalid_choice":
            result["answers"]["mindguard"]["probabilities"] = {marker: 1}
        else:
            result["answers"] = {"mindguard": {"type": "refusal", "text": marker}}
            if kind == "bad_id":
                result["id"] = "gen-" + marker
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", marker)
    monkeypatch.setattr("benchmarks.mindguard.collect.http.client.HTTPSConnection", FakeConnection)
    assert main(args, transport=transport) == 2
    sidecar = cache.with_suffix(".failures.jsonl")
    assert sidecar.stat().st_mode & 0o077 == 0
    saved = json.loads(sidecar.read_text())
    category = {
        "refusal": "provider_refusal",
        "http": "http_status",
        "invalid_wire": "malformed_wire",
        "wrong_route": "route_or_model_mismatch",
        "bad_id": "provider_refusal",
    }.get(kind, kind)
    assert saved["category"] == category
    assert saved["request_id"] == canary_requests((LUNA_MODEL,))[0]["request_id"]
    if kind == "http":
        assert saved["http_status"] == 429
    if kind in ("refusal", "invalid_choice"):
        assert saved["generation_id"] == generation_id
        assert saved["reported_cost_usd"] == "0.000132"
    else:
        assert "generation_id" not in saved
        assert "reported_cost_usd" not in saved
    output = capsys.readouterr()
    for value in (
        output.out,
        output.err,
        cache.read_text(),
        sidecar.read_text(),
        cache.with_suffix(".attempts.jsonl").read_text(),
    ):
        assert marker not in value
    assert main(args, transport=lambda _: pytest.fail("failure must not be retried")) == 2


@pytest.mark.parametrize("problem", ["id", "repeat", "generation", "provider", "model", "refused", "incomplete"])
def test_luna_cache_never_substitutes_missing_or_wrong_evidence(tmp_path: Path, problem: str) -> None:
    cases = [case(1), case(2)]
    reqs = [r for c in cases for r in requests_for(c, models=(LUNA_MODEL,))]
    rows = [{k: v for k, v in req.items() if k != "payload"} | {"response": fake_luna(req["payload"])} for req in reqs]
    if problem == "id":
        rows[0]["request_id"] = canary_requests((LUNA_MODEL,))[0]["request_id"]
    if problem == "repeat":
        rows.append(rows[0])
    if problem == "generation":
        rows[1]["response"]["id"] = rows[0]["response"]["id"]
    if problem == "provider":
        rows[0]["response"]["provider"] = "Azure"
    if problem == "model":
        rows[0]["response"]["model"] = MODEL_IDS[0]
    if problem == "refused":
        rows[0]["response"]["answers"]["mindguard"] = {"type": "refusal"}
    if problem == "incomplete":
        rows.pop()
    cache = tmp_path / "luna.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cache.chmod(0o600)
    with pytest.raises(ValueError, match=r"Duplicate|Incomplete|Missing|non-refused"):
        load_luna_scores(cases, cache)
