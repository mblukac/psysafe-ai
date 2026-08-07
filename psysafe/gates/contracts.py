"""Safe, framework-neutral contracts for workflow gate routing."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psysafe.core.classifier import AsyncTargetedClassifier, TargetedClassifier
from psysafe.core.contracts import (
    MAX_ASSESSMENT_SIGNALS,
    MAX_CONVERSATION_MESSAGES,
    Assessment,
    EvidenceDirectness,
    IndeterminateReason,
    Outcome,
    Sensitivity,
)

MAX_GATE_CLASSIFIERS = 64
MAX_GATE_REVIEW_SIGNALS = 64
MAX_GATE_ARTIFACT_ID_CHARS = 128


class Checkpoint(str, Enum):
    """Boundaries in a sequential or agentic execution workflow."""

    INPUT = "input"
    TASK_SELECTION = "task_selection"
    EXECUTION = "execution"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    COMMUNICATION = "communication"


class GateAction(str, Enum):
    """Application action requested by a gate decision."""

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@runtime_checkable
class ReviewSignalProvider(Protocol):
    """Assessment result exposing untrusted categorical review signals."""

    @property
    def review_signals(self) -> tuple[str, ...]: ...


@runtime_checkable
class GateClassifier(TargetedClassifier, Protocol):
    """Target-aware classifier with a finite trusted signal vocabulary."""

    @property
    def allowed_signals(self) -> tuple[str, ...]: ...

    @property
    def allowed_review_signals(self) -> tuple[str, ...]: ...


@runtime_checkable
class AsyncGateClassifier(AsyncTargetedClassifier, Protocol):
    """Async target-aware classifier with a finite trusted signal vocabulary."""

    @property
    def allowed_signals(self) -> tuple[str, ...]: ...

    @property
    def allowed_review_signals(self) -> tuple[str, ...]: ...


def _bounded_labels(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    for value in values:
        if not value or len(value) > 100:
            raise ValueError(f"{field_name} must contain between 1 and 100 characters")
        if not value[0].isalpha() or any(
            not (character.islower() or character.isdigit() or character in "_.-") for character in value
        ):
            raise ValueError(f"{field_name} must be lowercase identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")
    return values


class GateAssessment(BaseModel):
    """A routing-only copy of an assessment with explicit review signals.

    Classifier-specific findings, citations, adaptations, provider responses,
    and arbitrary subclass fields never cross this boundary. Applications that
    need those details can use the classifier API directly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$")
    policy_version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
    )
    sensitivity: Sensitivity = Sensitivity.BALANCED
    outcome: Outcome
    evidence_directness: EvidenceDirectness = EvidenceDirectness.NONE
    signals: tuple[str, ...] = Field(default_factory=tuple, max_length=MAX_ASSESSMENT_SIGNALS)
    indeterminate_reason: IndeterminateReason | None = None
    review_signals: tuple[str, ...] = Field(default_factory=tuple, max_length=MAX_GATE_REVIEW_SIGNALS)

    @field_validator("signals")
    @classmethod
    def signals_must_be_bounded_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _bounded_labels(values, field_name="signal labels")

    @field_validator("review_signals")
    @classmethod
    def review_signals_must_be_bounded_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _bounded_labels(values, field_name="review signal labels")

    @model_validator(mode="after")
    def categorical_fields_must_form_an_assessment(self) -> GateAssessment:
        # Reuse the canonical outcome invariants without copying metadata or a
        # potentially data-bearing classifier subclass into the gate result.
        Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=self.sensitivity,
            outcome=self.outcome,
            evidence_directness=self.evidence_directness,
            signals=self.signals,
            indeterminate_reason=self.indeterminate_reason,
        )
        if self.outcome is Outcome.INDETERMINATE and self.review_signals:
            raise ValueError("indeterminate gate assessments cannot assert review signals")
        return self


class GateDecision(BaseModel):
    """A deterministic decision correlated to one immutable artifact version.

    ``artifact_id`` is caller-supplied opaque correlation data, not a content
    hash. It must identify the exact immutable version checked, and the caller
    should consume the decision immediately to avoid time-of-check/time-of-use
    substitution. The gate never stores the ID or workflow content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    checkpoint: Checkpoint
    artifact_id: str = Field(
        min_length=1,
        max_length=MAX_GATE_ARTIFACT_ID_CHARS,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    target_message_index: int | None = Field(ge=0, lt=MAX_CONVERSATION_MESSAGES)
    action: GateAction
    assessments: tuple[GateAssessment, ...] = Field(
        min_length=1,
        max_length=MAX_GATE_CLASSIFIERS,
    )

    @model_validator(mode="after")
    def decision_must_be_fail_safe(self) -> GateDecision:
        classifier_ids = self.classifier_ids
        if len(set(classifier_ids)) != len(classifier_ids):
            raise ValueError("gate decisions require unique classifier IDs")
        if self.action is GateAction.ALLOW:
            if self.target_message_index is None:
                raise ValueError("an unbound gate decision cannot allow execution")
            if any(assessment.review_signals for assessment in self.assessments):
                raise ValueError("a gate decision with review signals cannot allow execution")
            if any(assessment.outcome is Outcome.INDETERMINATE for assessment in self.assessments):
                raise ValueError("a gate decision with an indeterminate assessment cannot allow execution")
        return self

    @property
    def classifier_ids(self) -> tuple[str, ...]:
        """Classifier IDs in deterministic evaluation order."""

        return tuple(assessment.classifier_id for assessment in self.assessments)

    @property
    def is_allowed(self) -> bool:
        """Whether the workflow may continue without review or blocking."""

        return self.action is GateAction.ALLOW


__all__ = [
    "MAX_GATE_ARTIFACT_ID_CHARS",
    "MAX_GATE_CLASSIFIERS",
    "MAX_GATE_REVIEW_SIGNALS",
    "AsyncGateClassifier",
    "Checkpoint",
    "GateAction",
    "GateAssessment",
    "GateClassifier",
    "GateDecision",
    "ReviewSignalProvider",
]
