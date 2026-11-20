from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from benchmarks.mindguard.collect import _validate
from benchmarks.mindguard.collect_flash import MODELS, collect, main, tail_canaries
from benchmarks.mindguard.data import Case, digest, requests_for
from benchmarks.mindguard.providers import (
    CLEF_FLASH_MODEL,
    DECIDER_FLASH_MODEL,
    FLASH_ROUTES,
    MODEL_IDS,
    extract_probabilities,
    synthetic_response,
)


def fixture_requests(model: str, _: Path) -> list[dict[str, Any]]:
    return [
        next(requests_for(Case("invented", i, ("safe",) * 3, (("user", f"Invented study turn {i}"),)), models=(model,)))
        for i in (1, 2, 3)
    ]


def fake(payload: dict[str, Any]) -> dict[str, Any]:
    m = payload["model"]
    newest = json.loads(payload["state"])["conversation"][-1]["content"]
    p = 0.8 if "suicide" in newest else 0.1
    return {
        **synthetic_response(m, {"safe": round(1 - p, 2), "self_harm": p, "harm_others": 0}),
        "id": "gen-invented-" + digest(payload),
        "provider": FLASH_ROUTES[m].name,
        "usage": {"input_tokens": 7123, "cost": 0.0001},
    }


def no_send(_: dict[str, Any]) -> dict[str, Any]:
    pytest.fail("unexpected inference transport")


def cli(d: Path, name: str, mode: str) -> list[str]:
    args = [mode, "--cohort", name, "--private-dir", str(d), "--max-requests", "3", "--workers", "2"]
    return args if mode == "canary" else [*args, "--annotations", str(d / "input.parquet")]


@pytest.mark.parametrize("name", list(MODELS))
def test_flash_wire_and_resume(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = MODELS[name]
    route = FLASH_ROUTES[model]
    canary, batch = cli(tmp_path, name, "canary"), cli(tmp_path, name, "batch")
    paid = ["--allow-paid", "--allow-gated-provider-processing"]
    monkeypatch.setattr("benchmarks.mindguard.collect_flash._study_requests", fixture_requests)
    samples = tail_canaries(model)
    heads = [json.loads(r["payload"]["state"])["conversation"][:-1] for r, _ in samples]
    assert heads[0] == heads[1]
    assert len(samples[0][0]["payload"]["state"].encode()) > 21_000
    historical = fixture_requests(MODEL_IDS[0], tmp_path)[0]["payload"]
    new = fixture_requests(model, tmp_path)[0]["payload"]
    assert "provider" not in historical
    assert new["state"] == historical["state"]
    assert new["questions"] == historical["questions"]
    assert new["provider"] == {
        "only": [route.provider],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
        "max_price": {"prompt": route.usd_per_million, "completion": "0", "request": "0"},
    }
    assert main(batch, transport=no_send) == 0
    assert not list(tmp_path.iterdir())
    assert main([*batch, *paid], transport=no_send) == 2
    assert main([*canary, "--allow-paid"], transport=fake) == 0
    assert main([*batch, "--allow-paid"], transport=no_send) == 2
    assert main([*batch, *paid], transport=fake) == 0
    assert main([*batch, *paid], transport=no_send) == 0
    out = tmp_path / (route.slug + "-responses.jsonl")
    assert len(out.read_text().splitlines()) == 3
    assert not out.stat().st_mode & 0o077
    assert "Invented study" not in out.read_text()


@pytest.mark.parametrize("failure", ["wrong_choice", "short_tokens", "wrong_host"])
def test_canary_blocks_study_and_never_repeats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    canary = cli(tmp_path, "clef-flash", "canary")
    batch = cli(tmp_path, "clef-flash", "batch")
    monkeypatch.setattr("benchmarks.mindguard.collect_flash._study_requests", fixture_requests)

    def bad(payload: dict[str, Any]) -> dict[str, Any]:
        resp = fake(payload)
        if failure == "wrong_choice":
            resp.update(synthetic_response(payload["model"], {"safe": 1, "harm_others": 0, "self_harm": 0}))
        if failure == "short_tokens":
            resp["usage"]["input_tokens"] = 1999
        if failure == "wrong_host":
            resp["provider"] = "Cloudflare"
        return resp

    assert main([*canary, "--allow-paid"], transport=bad) == 2
    assert main([*canary, "--allow-paid"], transport=no_send) == 2
    assert main([*batch, "--allow-paid", "--allow-gated-provider-processing"], transport=no_send) == 2


def test_charge_integrity_and_full_context_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = DECIDER_FLASH_MODEL
    r = FLASH_ROUTES[model]
    allowed = r.reservation + Decimal("0.00005")
    monkeypatch.setitem(FLASH_ROUTES, model, replace(r, budget=allowed))
    reqs = fixture_requests(model, tmp_path)
    for usage in (
        {"input_tokens": 0, "cost": 0},
        {"input_tokens": 1, "cost": float(r.reservation + 1)},
        {"input_tokens": True, "cost": 0},
        {"input_tokens": 100},
    ):
        response = {**fake(reqs[0]["payload"]), "usage": usage}
        with pytest.raises(ValueError, match="Flash"):
            _validate(model, response)
    file = tmp_path / "study.jsonl"
    with pytest.raises(ValueError, match="cap cannot reserve"):
        collect(
            model, file, reqs, paid=True, max_requests=3, workers=3, canary_spend=Decimal("0.00004"), transport=fake
        )
    assert len(file.read_text().splitlines()) == 1
    assert len(file.with_suffix(".attempts.jsonl").read_text().splitlines()) == 1


def test_no_wrong_release_or_public_directory(tmp_path: Path) -> None:
    (tmp_path / "input.parquet").write_bytes(b"NOT PINNED")
    assert main(cli(tmp_path, "clef-flash", "batch"), transport=no_send) == 2
    tmp_path.chmod(0o755)
    try:
        assert main(cli(tmp_path, "clef-flash", "canary"), transport=no_send) == 2
    finally:
        tmp_path.chmod(0o700)


def test_flash_context_reservation() -> None:
    assert FLASH_ROUTES[CLEF_FLASH_MODEL].reservation == Decimal("0.000466944")
    assert FLASH_ROUTES[DECIDER_FLASH_MODEL].reservation == Decimal("0.00262144")


def test_only_clef_exact_four_decimal_grid_accepts_one_unit_error() -> None:
    m = CLEF_FLASH_MODEL
    for v in [
        {"safe": 0.3333, "self_harm": 0.3333, "harm_others": 0.3333},
        {"safe": 0.6001, "self_harm": 0.2, "harm_others": 0.2},
    ]:
        r = synthetic_response(m, v)
        assert extract_probabilities(m, r) == v
        r["model"] = DECIDER_FLASH_MODEL
        with pytest.raises(ValueError, match="sum"):
            extract_probabilities(DECIDER_FLASH_MODEL, r)
    for x in (0.3332, 0.333333, 0.32):
        with pytest.raises(ValueError, match="sum"):
            synthetic_response(m, {"safe": x, "self_harm": 0.3333, "harm_others": 0.3333})
    with pytest.raises(ValueError, match="sum"):
        synthetic_response(m, {"safe": 0.33, "self_harm": 0.33, "harm_others": 0.33})


def test_single_fresh_probe_requires_reconciled_known_failure_and_preserves_all_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("benchmarks.mindguard.collect_flash._study_requests", fixture_requests)
    canary = [*cli(tmp_path, "clef-flash", "canary"), "--allow-paid", "--workers", "1"]
    second = [*canary, "--clef-canary-v2"]
    batch = [
        *cli(tmp_path, "clef-flash", "batch"),
        "--allow-paid",
        "--allow-gated-provider-processing",
        "--clef-canary-v2",
    ]
    gid = "gen-dec-1795110000-00000000000000000000"

    def original(payload: dict[str, Any]) -> dict[str, Any]:
        r = fake(payload)
        if r["answers"]["mindguard"]["choice"] == "self_harm":
            r["id"] = gid
            r["answers"]["mindguard"]["choice"] = "invalid"
        return r

    def billing(i: str) -> dict[str, Any]:
        assert i == gid
        return {"id": i, "model": CLEF_FLASH_MODEL, "provider_name": "DeepInfra", "total_cost": 0.0001}

    assert main(second, transport=no_send, metadata=billing) == 2
    assert main(canary, transport=original) == 2
    old = {p.name: p.read_bytes() for p in tmp_path.glob("*.jsonl")}
    assert main(batch, transport=no_send, metadata=billing) == 2
    for k, v in (("id", "other"), ("provider_name", "other"), ("total_cost", 0), ("model", "other")):
        assert main(second, transport=no_send, metadata=lambda i, key=k, val=v: {**billing(i), key: val}) == 2
    assert main(second, transport=fake, metadata=billing) == 0
    assert "all canary cost $0.0004" in capsys.readouterr().out
    assert main(second, transport=no_send, metadata=billing) == 0
    assert main(canary, transport=no_send) == 2
    assert main(batch, transport=fake, metadata=billing) == 0
    assert "billed total $0.0007" in capsys.readouterr().out
    assert all((tmp_path / name).read_bytes() == content for name, content in old.items())
