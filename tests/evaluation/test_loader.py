"""Synthetic contract tests; they make no classifier-quality claim."""

import json

import pytest
from pydantic import ValidationError

from psysafe.core.contracts import Sensitivity
from psysafe.evaluation import (
    MAX_GOLDEN_CASES,
    EvaluationSplit,
    ExpectedBoundary,
    ExpectedSignal,
    GoldenCase,
    GoldenCaseLoadError,
    GoldenCaseLoadReason,
    audit_split_families,
    load_golden_cases,
)


def _signal(name: str, boundary: str) -> dict[str, str]:
    return {"name": name, "boundary": boundary}


def _payload(
    case_id: str,
    family_id: str,
    split: str = "holdout",
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": case_id,
        "family_id": family_id,
        "split": split,
        "slices": ["synthetic"],
        "conversation": {"messages": [{"role": "user", "content": "Synthetic contract fixture."}]},
        "expected_boundary": "never",
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path, *payloads: dict[str, object]) -> None:
    path.write_text("".join(json.dumps(payload) + "\n" for payload in payloads), encoding="utf-8")


def _library_traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if "/psysafe/evaluation/" in traceback.tb_frame.f_code.co_filename:
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


def test_loads_strict_jsonl_and_keeps_tuning_and_holdout_explicit(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    _write_jsonl(
        path,
        _payload(
            "case-1",
            "family-1",
            expected_boundary="balanced",
            expected_signals=[_signal("test_signal", "balanced")],
        ),
        _payload("case-2", "family-2", split="tuning"),
    )

    cases = load_golden_cases(path)

    assert tuple(case.case_id for case in cases) == ("case-1", "case-2")
    assert cases[0].split is EvaluationSplit.HOLDOUT
    assert cases[1].split is EvaluationSplit.TUNING
    assert cases[0].expected_boundary is ExpectedBoundary.BALANCED
    assert cases[0].expected_signals == (ExpectedSignal(name="test_signal", boundary=ExpectedBoundary.BALANCED),)
    assert cases[0].signals_at(Sensitivity.PRECISE) == ()
    assert cases[0].signals_at(Sensitivity.BALANCED) == ("test_signal",)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": True},
        {"case_id": "unsafe/id"},
        {"family_id": "unsafe/id"},
        {"slices": ["Unsafe Slice"]},
        {"slices": ["same", "same"]},
        {"expected_boundary": "never", "expected_signals": [_signal("test_signal", "precise")]},
        {"expected_boundary": "balanced", "expected_signals": [_signal("test_signal", "precise")]},
        {"expected_boundary": "precise", "expected_signals": [_signal("test_signal", "balanced")]},
        {"expected_review_signals": [_signal("human_review", "balanced")]},
    ],
)
def test_golden_case_rejects_unsafe_or_incoherent_fields(overrides: dict[str, object]) -> None:
    payload = _payload("case-1", "family-1")
    payload.update(overrides)
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(payload)


@pytest.mark.parametrize(
    ("payloads", "reason"),
    [
        (
            (_payload("same", "family-1"), _payload("same", "family-2")),
            GoldenCaseLoadReason.DUPLICATE_CASE_ID,
        ),
        (
            (_payload("case-1", "family-1"), _payload("case-2", "family-1", split="tuning")),
            GoldenCaseLoadReason.FAMILY_SPLIT_CONFLICT,
        ),
    ],
)
def test_loader_rejects_duplicate_ids_and_family_split_leakage(tmp_path, payloads, reason) -> None:
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, *payloads)

    with pytest.raises(GoldenCaseLoadError) as caught:
        load_golden_cases(path)

    assert caught.value.reason is reason
    assert caught.value.line_number == 2


def test_public_audit_detects_family_leakage_across_separate_manifests(tmp_path) -> None:
    tuning_path = tmp_path / "tuning.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    _write_jsonl(tuning_path, _payload("tuning-case", "shared-family", split="tuning"))
    _write_jsonl(holdout_path, _payload("holdout-case", "shared-family"))

    tuning = load_golden_cases(tuning_path)
    holdout = load_golden_cases(holdout_path)

    with pytest.raises(GoldenCaseLoadError) as caught:
        audit_split_families((*tuning, *holdout))

    assert caught.value.reason is GoldenCaseLoadReason.FAMILY_SPLIT_CONFLICT
    assert "Synthetic contract fixture" not in _library_traceback_locals(caught.value)


class _HostileCaseList(list[GoldenCase]):
    iterated = False

    def __iter__(self):
        type(self).iterated = True
        return super().__iter__()


def test_public_family_audit_bounds_containers_before_iteration() -> None:
    golden_case = GoldenCase.model_validate(_payload("case-1", "family-1"))
    hostile = _HostileCaseList([golden_case])
    oversized = [golden_case] * (MAX_GOLDEN_CASES + 1)

    with pytest.raises(GoldenCaseLoadError) as hostile_error:
        audit_split_families(hostile)
    with pytest.raises(GoldenCaseLoadError) as oversized_error:
        audit_split_families(oversized)

    assert hostile_error.value.reason is GoldenCaseLoadReason.INVALID_CASE
    assert oversized_error.value.reason is GoldenCaseLoadReason.TOO_MANY_CASES
    assert _HostileCaseList.iterated is False


def test_loader_errors_are_fixed_and_do_not_retain_rejected_content(tmp_path) -> None:
    path = tmp_path / "private-dataset-name.jsonl"
    path.write_text('{"secret":"PRIVATE GOLDEN CONTENT"', encoding="utf-8")

    with pytest.raises(GoldenCaseLoadError) as caught:
        load_golden_cases(path)

    assert caught.value.reason is GoldenCaseLoadReason.INVALID_JSON
    assert str(caught.value) == "could not load golden cases (invalid_json)"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    library_locals = _library_traceback_locals(caught.value)
    assert "PRIVATE GOLDEN CONTENT" not in library_locals
    assert "private-dataset-name" not in library_locals


class _ExplodingPath:
    def __fspath__(self) -> str:
        raise RuntimeError("PRIVATE PATH ACCESSOR")


def test_loader_sanitizes_unknown_path_adapter_failures() -> None:
    with pytest.raises(GoldenCaseLoadError) as caught:
        load_golden_cases(_ExplodingPath())

    assert caught.value.reason is GoldenCaseLoadReason.INVALID_SOURCE
    assert "PRIVATE PATH ACCESSOR" not in _library_traceback_locals(caught.value)


def test_loader_rejects_blank_and_empty_datasets_categorically(tmp_path) -> None:
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(GoldenCaseLoadError) as blank_error:
        load_golden_cases(blank)
    with pytest.raises(GoldenCaseLoadError) as empty_error:
        load_golden_cases(empty)

    assert blank_error.value.reason is GoldenCaseLoadReason.INVALID_JSON
    assert empty_error.value.reason is GoldenCaseLoadReason.EMPTY_DATASET
