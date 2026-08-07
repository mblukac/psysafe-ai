"""Strict categorical contracts for golden-case evaluation.

The models in this module intentionally retain only opaque identifiers and
categorical values. Conversations are inputs to a run and never appear in
case results or reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psysafe.core.contracts import MAX_ASSESSMENT_SIGNALS, Conversation, IndeterminateReason, Outcome, Sensitivity

MAX_CASE_ID_CHARS = 80
MAX_FAMILY_ID_CHARS = 80
MAX_SLICE_NAME_CHARS = 64
MAX_CASE_SLICES = 32

CaseId = Annotated[
    str,
    Field(min_length=1, max_length=MAX_CASE_ID_CHARS, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
FamilyId = Annotated[
    str,
    Field(min_length=1, max_length=MAX_FAMILY_ID_CHARS, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
SliceName = Annotated[
    str,
    Field(min_length=1, max_length=MAX_SLICE_NAME_CHARS, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
SignalName = Annotated[
    str,
    Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
ClassifierId = Annotated[
    str,
    Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
PolicyVersion = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$"),
]


class EvaluationSplit(str, Enum):
    """Dataset partition with an explicit tuning/holdout boundary."""

    TUNING = "tuning"
    HOLDOUT = "holdout"


class ExpectedBoundary(str, Enum):
    """The narrowest sensitivity expected to match a golden case."""

    PRECISE = "precise"
    BALANCED = "balanced"
    PRECAUTIONARY = "precautionary"
    NEVER = "never"
    INDETERMINATE = "indeterminate"

    def expected_outcome(self, sensitivity: Sensitivity) -> Outcome:
        """Return the categorical expectation at one named sensitivity."""

        normalized = Sensitivity(sensitivity)
        if self is ExpectedBoundary.INDETERMINATE:
            return Outcome.INDETERMINATE
        if self is ExpectedBoundary.NEVER:
            return Outcome.NOT_MATCHED
        minimum_rank = {
            ExpectedBoundary.PRECISE: 0,
            ExpectedBoundary.BALANCED: 1,
            ExpectedBoundary.PRECAUTIONARY: 2,
        }[self]
        sensitivity_rank = {
            Sensitivity.PRECISE: 0,
            Sensitivity.BALANCED: 1,
            Sensitivity.PRECAUTIONARY: 2,
        }[normalized]
        return Outcome.MATCHED if sensitivity_rank >= minimum_rank else Outcome.NOT_MATCHED


class ExpectedSignal(BaseModel):
    """One categorical signal and the narrowest boundary where it appears."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: SignalName
    boundary: ExpectedBoundary

    @field_validator("boundary")
    @classmethod
    def boundary_must_match_some_sensitivity(cls, value: ExpectedBoundary) -> ExpectedBoundary:
        if value in {ExpectedBoundary.NEVER, ExpectedBoundary.INDETERMINATE}:
            raise ValueError("signal boundaries must be named sensitivities")
        return value


class GoldenCase(BaseModel):
    """One strict JSONL golden case.

    ``family_id`` groups variants derived from the same source. A loader and
    runner reject a family that crosses the tuning/holdout boundary.
    Signal expectations carry their own minimum boundary, producing an exact
    unordered set at each sensitivity while allowing broader boundaries to add
    legitimate signals monotonically.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_id: CaseId
    family_id: FamilyId
    split: EvaluationSplit
    slices: tuple[SliceName, ...] = Field(default_factory=tuple, max_length=MAX_CASE_SLICES)
    conversation: Conversation
    expected_boundary: ExpectedBoundary
    expected_signals: tuple[ExpectedSignal, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ASSESSMENT_SIGNALS,
    )
    expected_review_boundary: ExpectedBoundary | None = None
    expected_review_signals: tuple[ExpectedSignal, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ASSESSMENT_SIGNALS,
    )

    @field_validator("slices")
    @classmethod
    def slice_values_must_be_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("categorical labels must be unique")
        return values

    @field_validator("expected_signals", "expected_review_signals")
    @classmethod
    def signal_values_must_be_unique(
        cls,
        values: tuple[ExpectedSignal, ...] | None,
    ) -> tuple[ExpectedSignal, ...] | None:
        if values is not None and len({value.name for value in values}) != len(values):
            raise ValueError("expected signal names must be unique")
        return values

    @model_validator(mode="after")
    def signal_expectations_must_have_matching_boundaries(self) -> GoldenCase:
        if self.expected_signals is not None and self.expected_boundary in {
            ExpectedBoundary.NEVER,
            ExpectedBoundary.INDETERMINATE,
        }:
            raise ValueError("expected signals require a matching expected boundary")
        if self.expected_review_signals is not None and self.expected_review_boundary is None:
            raise ValueError("expected review signals require an expected review boundary")
        if self.expected_review_signals is not None and self.expected_review_boundary in {
            ExpectedBoundary.NEVER,
            ExpectedBoundary.INDETERMINATE,
        }:
            raise ValueError("expected review signals require a matching review boundary")
        self._validate_signal_boundaries(self.expected_boundary, self.expected_signals)
        if self.expected_review_boundary is not None:
            self._validate_signal_boundaries(self.expected_review_boundary, self.expected_review_signals)
        return self

    @staticmethod
    def _validate_signal_boundaries(
        decision_boundary: ExpectedBoundary,
        signals: tuple[ExpectedSignal, ...] | None,
    ) -> None:
        if signals is None or decision_boundary in {ExpectedBoundary.NEVER, ExpectedBoundary.INDETERMINATE}:
            return
        first_sensitivity = Sensitivity(decision_boundary.value)
        if any(signal.boundary.expected_outcome(first_sensitivity) is Outcome.MATCHED for signal in signals):
            if all(
                decision_boundary.expected_outcome(Sensitivity(signal.boundary.value)) is Outcome.MATCHED
                for signal in signals
            ):
                return
        raise ValueError("signal boundaries must begin at or after their decision boundary")

    def signals_at(self, sensitivity: Sensitivity, *, review: bool = False) -> tuple[str, ...] | None:
        """Return the exact unordered signal expectation at one sensitivity."""

        expectations = self.expected_review_signals if review else self.expected_signals
        if expectations is None:
            return None
        return tuple(
            expectation.name
            for expectation in expectations
            if expectation.boundary.expected_outcome(sensitivity) is Outcome.MATCHED
        )


class BoundaryOutcome(str, Enum):
    """Categorical comparison for one case at one sensitivity."""

    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    EXPECTED_INDETERMINATE = "expected_indeterminate"
    UNEXPECTED_INDETERMINATE = "unexpected_indeterminate"
    UNEXPECTED_DECISION = "unexpected_decision"
    SIGNAL_MISMATCH = "signal_mismatch"


class CaseOutcome(str, Enum):
    """Categorical summary for a case across every sensitivity."""

    PASSED = "passed"
    EXPECTED_INDETERMINATE = "expected_indeterminate"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    UNEXPECTED_INDETERMINATE = "unexpected_indeterminate"
    UNEXPECTED_DECISION = "unexpected_decision"
    SIGNAL_MISMATCH = "signal_mismatch"
    MONOTONICITY_VIOLATION = "monotonicity_violation"
    MULTIPLE_FAILURES = "multiple_failures"


class MonotonicityStatus(str, Enum):
    """Whether broader sensitivities preserve every narrower match."""

    VERIFIED = "verified"
    VIOLATED = "violated"
    INDETERMINATE = "indeterminate"


class BoundaryResult(BaseModel):
    """A value-free categorical result for one sensitivity."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sensitivity: Sensitivity
    expected_outcome: Outcome
    actual_outcome: Outcome
    outcome: BoundaryOutcome
    expected_signals: tuple[SignalName, ...] | None = None
    actual_signals: tuple[SignalName, ...] = Field(default_factory=tuple, max_length=MAX_ASSESSMENT_SIGNALS)
    indeterminate_reason: IndeterminateReason | None = None
    expected_review_outcome: Outcome | None = None
    actual_review_outcome: Outcome
    review_outcome: BoundaryOutcome | None = None
    expected_review_signals: tuple[SignalName, ...] | None = None
    actual_review_signals: tuple[SignalName, ...] = Field(default_factory=tuple, max_length=MAX_ASSESSMENT_SIGNALS)


class CaseResult(BaseModel):
    """Categorical evaluation result with no conversation or exception text."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: ClassifierId
    policy_version: PolicyVersion
    case_id: CaseId
    family_id: FamilyId
    split: EvaluationSplit
    slices: tuple[SliceName, ...] = Field(default_factory=tuple, max_length=MAX_CASE_SLICES)
    expected_boundary: ExpectedBoundary
    expected_review_boundary: ExpectedBoundary | None = None
    outcome: CaseOutcome
    monotonicity: MonotonicityStatus
    review_monotonicity: MonotonicityStatus | None = None
    boundaries: tuple[BoundaryResult, BoundaryResult, BoundaryResult]


class MetricSummary(BaseModel):
    """Aggregate decision metrics over case-by-sensitivity evaluations.

    Confusion counts include only rows with a binary expectation and binary
    decision. ``coverage`` is decided binary rows / expected binary rows.
    Precision is TP/(TP+FP); selective recall and FNR use TP+FN; FPR uses
    FP+TN. ``effective_recall`` uses TP+FN+positive-indeterminate so positive
    abstentions count as misses. ``indeterminate_rate`` and
    ``intervention_rate`` use all rows; intervention means matched or
    indeterminate. Signal match rate uses only rows with an exact signal
    expectation. A rate is ``None`` when its denominator is zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    evaluations: int = Field(ge=0)
    expected_binary: int = Field(ge=0)
    expected_indeterminate: int = Field(ge=0)
    decided_binary: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    positive_indeterminate: int = Field(ge=0)
    negative_indeterminate: int = Field(ge=0)
    unexpected_decision: int = Field(ge=0)
    actual_matched: int = Field(ge=0)
    actual_not_matched: int = Field(ge=0)
    actual_indeterminate: int = Field(ge=0)
    signal_expectations: int = Field(ge=0)
    signal_matches: int = Field(ge=0)
    signal_mismatches: int = Field(ge=0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    false_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_negative_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    indeterminate_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    intervention_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    signal_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class SensitivityReport(BaseModel):
    """Aggregate metrics for one named sensitivity."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sensitivity: Sensitivity
    metrics: MetricSummary
    review_metrics: MetricSummary | None = None


class SliceReport(BaseModel):
    """Aggregate metrics for cases carrying one safe slice label."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    slice: SliceName
    case_count: int = Field(ge=1)
    metrics: MetricSummary
    review_metrics: MetricSummary | None = None
    by_sensitivity: tuple[SensitivityReport, SensitivityReport, SensitivityReport]


class EvaluationReport(BaseModel):
    """Complete value-free report for exactly one dataset split.

    ``classifier_id`` and ``policy_version`` are validated policy labels
    snapshotted before case execution. They are not authenticated execution
    provenance. Provider/model metadata is deliberately omitted because the
    classifier contract does not guarantee that descriptive metadata is
    stable across routed executions.

    Holdout reports should normally keep ``diagnostics_revealed`` false.
    Revealing per-case holdout diagnostics enables adaptive tuning and repeated
    inspection invalidates the holdout as an unbiased evaluation set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: ClassifierId
    policy_version: PolicyVersion
    split: EvaluationSplit
    case_count: int = Field(ge=1)
    diagnostics_revealed: bool
    case_results: tuple[CaseResult, ...] = Field(default_factory=tuple)
    metrics: MetricSummary
    review_metrics: MetricSummary | None = None
    by_sensitivity: tuple[SensitivityReport, SensitivityReport, SensitivityReport]
    slices: tuple[SliceReport, ...]
    monotonicity_verified: int = Field(ge=0)
    monotonicity_violations: int = Field(ge=0)
    monotonicity_indeterminate: int = Field(ge=0)
    review_monotonicity_violations: int = Field(ge=0)

    @model_validator(mode="after")
    def diagnostics_flag_must_match_case_results(self) -> EvaluationReport:
        if not self.diagnostics_revealed:
            if self.case_results:
                raise ValueError("sealed reports cannot contain case diagnostics")
            if self.slices:
                raise ValueError("sealed reports cannot contain slice diagnostics")
        if self.diagnostics_revealed:
            if len(self.case_results) != self.case_count:
                raise ValueError("revealed diagnostics must include every case")
            if any(result.split is not self.split for result in self.case_results):
                raise ValueError("case diagnostics must match the report split")
            if any(
                result.classifier_id != self.classifier_id or result.policy_version != self.policy_version
                for result in self.case_results
            ):
                raise ValueError("case diagnostics must match the report policy")
        return self


__all__ = [
    "BoundaryOutcome",
    "BoundaryResult",
    "CaseOutcome",
    "CaseResult",
    "EvaluationReport",
    "EvaluationSplit",
    "ExpectedBoundary",
    "ExpectedSignal",
    "GoldenCase",
    "MetricSummary",
    "MonotonicityStatus",
    "SensitivityReport",
    "SliceReport",
]
