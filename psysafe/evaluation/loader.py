"""Bounded JSONL loading with categorical, sanitized failures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import NoReturn, TextIO, cast

from psysafe.evaluation.models import EvaluationSplit, GoldenCase

MAX_GOLDEN_CASES = 10_000
MAX_GOLDEN_LINE_CHARS = 4_000_000
MAX_GOLDEN_FILE_CHARS = 64_000_000


class GoldenCaseLoadReason(str, Enum):
    """Safe categories for JSONL loading failures."""

    INVALID_SOURCE = "invalid_source"
    NOT_FOUND = "not_found"
    IO_ERROR = "io_error"
    INVALID_ENCODING = "invalid_encoding"
    INVALID_JSON = "invalid_json"
    INVALID_CASE = "invalid_case"
    DUPLICATE_CASE_ID = "duplicate_case_id"
    FAMILY_SPLIT_CONFLICT = "family_split_conflict"
    LINE_TOO_LARGE = "line_too_large"
    DATASET_TOO_LARGE = "dataset_too_large"
    TOO_MANY_CASES = "too_many_cases"
    EMPTY_DATASET = "empty_dataset"


class GoldenCaseLoadError(ValueError):
    """A fixed-message load error that never embeds paths or case content."""

    def __init__(self, reason: GoldenCaseLoadReason, *, line_number: int | None = None) -> None:
        self.reason = reason
        self.line_number = line_number
        super().__init__(f"could not load golden cases ({reason.value})")


def _raise_load_error(reason: GoldenCaseLoadReason, line_number: int | None) -> NoReturn:
    raise GoldenCaseLoadError(reason, line_number=line_number) from None


def _families_cross_splits(cases: Sequence[GoldenCase]) -> bool:
    family_splits: dict[str, EvaluationSplit] = {}
    for golden_case in cases:
        known_split = family_splits.get(golden_case.family_id)
        if known_split is not None and known_split is not golden_case.split:
            return True
        family_splits[golden_case.family_id] = golden_case.split
    return False


def audit_split_families(cases: Sequence[GoldenCase]) -> None:
    """Reject family leakage across separately loaded tuning/holdout sets.

    Call this on the concatenation of every manifest before selecting a split.
    The runner performs the same audit automatically on its supplied cases.
    """

    invalid = type(cases) not in {tuple, list}
    too_many = False
    values: tuple[GoldenCase, ...] | list[GoldenCase] = ()
    bounded_cases: tuple[GoldenCase, ...] = ()
    if not invalid:
        values = cast(tuple[GoldenCase, ...] | list[GoldenCase], cases)
        too_many = len(values) > MAX_GOLDEN_CASES
        if not too_many:
            invalid = any(type(golden_case) is not GoldenCase for golden_case in values)
            if not invalid:
                bounded_cases = tuple(values)
    conflict = _families_cross_splits(bounded_cases) if not invalid and not too_many else False
    if conflict or invalid or too_many:
        if conflict:
            reason = GoldenCaseLoadReason.FAMILY_SPLIT_CONFLICT
        elif too_many:
            reason = GoldenCaseLoadReason.TOO_MANY_CASES
        else:
            reason = GoldenCaseLoadReason.INVALID_CASE
        del cases, values, bounded_cases, conflict, invalid, too_many
        _raise_load_error(reason, None)


def load_golden_cases(path: str | PathLike[str]) -> tuple[GoldenCase, ...]:
    """Load a bounded UTF-8 JSONL dataset without exposing rejected data.

    Blank lines and non-object JSON values are invalid. Case IDs must be unique,
    and all cases sharing a family ID must remain in one split.
    """

    source: Path | None = None
    stream: TextIO | None = None
    cases: list[GoldenCase] = []
    case_ids: set[str] = set()
    family_splits: dict[str, EvaluationSplit] = {}
    raw_line = ""
    stripped = ""
    parsed: object | None = None
    golden_case: GoldenCase | None = None
    line_number = 0
    total_chars = 0
    failure: GoldenCaseLoadReason | None = None
    failure_line: int | None = None

    try:
        try:
            source = Path(path)
        except Exception:  # noqa: BLE001 - path-like objects are an input boundary.
            failure = GoldenCaseLoadReason.INVALID_SOURCE

        if failure is None and source is not None:
            try:
                stream = source.open("r", encoding="utf-8", newline="")
            except FileNotFoundError:
                failure = GoldenCaseLoadReason.NOT_FOUND
            except Exception:  # noqa: BLE001 - filesystem adapters are an input boundary.
                failure = GoldenCaseLoadReason.IO_ERROR

        while failure is None and stream is not None:
            try:
                raw_line = stream.readline(MAX_GOLDEN_LINE_CHARS + 1)
            except UnicodeError:
                failure = GoldenCaseLoadReason.INVALID_ENCODING
                failure_line = line_number + 1
                break
            except Exception:  # noqa: BLE001 - filesystem adapters are an input boundary.
                failure = GoldenCaseLoadReason.IO_ERROR
                failure_line = line_number + 1
                break

            if raw_line == "":
                break
            line_number += 1
            total_chars += len(raw_line)
            if len(raw_line) > MAX_GOLDEN_LINE_CHARS:
                failure = GoldenCaseLoadReason.LINE_TOO_LARGE
                failure_line = line_number
                break
            if total_chars > MAX_GOLDEN_FILE_CHARS:
                failure = GoldenCaseLoadReason.DATASET_TOO_LARGE
                failure_line = line_number
                break

            stripped = raw_line.strip()
            if not stripped:
                failure = GoldenCaseLoadReason.INVALID_JSON
                failure_line = line_number
                break
            try:
                parsed = json.loads(stripped)
            except Exception:  # noqa: BLE001 - parser failures remain categorical.
                failure = GoldenCaseLoadReason.INVALID_JSON
                failure_line = line_number
                continue
            if not isinstance(parsed, dict):
                failure = GoldenCaseLoadReason.INVALID_CASE
                failure_line = line_number
                break
            try:
                golden_case = GoldenCase.model_validate(parsed)
            except Exception:  # noqa: BLE001 - schema validators are an input boundary.
                failure = GoldenCaseLoadReason.INVALID_CASE
                failure_line = line_number
                continue

            if golden_case.case_id in case_ids:
                failure = GoldenCaseLoadReason.DUPLICATE_CASE_ID
                failure_line = line_number
                break
            known_split = family_splits.get(golden_case.family_id)
            if known_split is not None and known_split is not golden_case.split:
                failure = GoldenCaseLoadReason.FAMILY_SPLIT_CONFLICT
                failure_line = line_number
                break
            case_ids.add(golden_case.case_id)
            family_splits[golden_case.family_id] = golden_case.split
            cases.append(golden_case)
            if len(cases) > MAX_GOLDEN_CASES:
                failure = GoldenCaseLoadReason.TOO_MANY_CASES
                failure_line = line_number
                break

        if failure is None and not cases:
            failure = GoldenCaseLoadReason.EMPTY_DATASET
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:  # noqa: BLE001 - filesystem adapters are an input boundary.
                if failure is None:
                    failure = GoldenCaseLoadReason.IO_ERROR

    if failure is not None:
        safe_failure = failure
        safe_line = failure_line
        del path, source, stream, cases, case_ids, family_splits, raw_line, stripped, parsed, golden_case
        _raise_load_error(safe_failure, safe_line)
    loaded = tuple(cases)
    audit_failure = False
    try:
        audit_split_families(loaded)
    except GoldenCaseLoadError:
        audit_failure = True
    if audit_failure:
        del path, source, stream, cases, case_ids, family_splits, raw_line, stripped, parsed, golden_case, loaded
        _raise_load_error(GoldenCaseLoadReason.FAMILY_SPLIT_CONFLICT, None)
    return loaded


__all__ = [
    "MAX_GOLDEN_CASES",
    "MAX_GOLDEN_FILE_CHARS",
    "MAX_GOLDEN_LINE_CHARS",
    "GoldenCaseLoadError",
    "GoldenCaseLoadReason",
    "audit_split_families",
    "load_golden_cases",
]
