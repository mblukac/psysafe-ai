"""Calibrated distress signals for adapting an application's response style."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator

from psysafe.backends.base import StructuredBackend
from psysafe.classifiers.base import (
    MAX_FINDINGS,
    Finding,
    Observation,
    ObservationRecord,
    PolicyClassifier,
    select_findings,
)
from psysafe.classifiers.context import EvidenceSubject, SourceContext, is_direct_user_evidence
from psysafe.classifiers.prompting import PromptSpec
from psysafe.core.classifier import FailurePolicy
from psysafe.core.contracts import Assessment, MessageRole, Outcome, Sensitivity


class DistressSignal(str, Enum):
    """Observable affective signals covered by the response-adaptation policy."""

    OVERWHELM = "overwhelm"
    PANIC = "panic"
    GRIEF = "grief"
    LONELINESS = "loneliness"
    LOW_MOOD = "low_mood"


class ResponseAdaptation(str, Enum):
    """Non-clinical communication adaptations for a downstream responder."""

    ACKNOWLEDGE_EMOTION = "acknowledge_emotion"
    CALM_CLEAR_LANGUAGE = "calm_clear_language"
    REDUCE_COGNITIVE_LOAD = "reduce_cognitive_load"
    OFFER_MANAGEABLE_CHOICES = "offer_manageable_choices"
    AVOID_PRESSURE = "avoid_pressure"
    INVITE_PAUSE = "invite_pause"


class DistressFinding(Finding):
    """One observable distress signal and its safe evidence locations."""

    signal: DistressSignal = Field(description="Observable distress-support signal.")
    subject: EvidenceSubject = Field(description="Whom the distress signal concerns.")
    source_context: SourceContext = Field(description="How the evidence is presented.")
    response_adaptations: tuple[ResponseAdaptation, ...] = Field(default_factory=tuple, max_length=6)

    @field_validator("response_adaptations")
    @classmethod
    def adaptations_must_be_unique(
        cls,
        values: tuple[ResponseAdaptation, ...],
    ) -> tuple[ResponseAdaptation, ...]:
        if len(values) != len(set(values)):
            raise ValueError("response adaptations must be unique")
        return values


class DistressObservation(Observation[DistressFinding]):
    """Sensitivity-independent distress observations."""


class DistressAssessment(Assessment):
    """A calibrated distress-support match without diagnostic conclusions."""

    findings: tuple[DistressFinding, ...] = Field(default_factory=tuple, max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def findings_must_match_assessment(self) -> DistressAssessment:
        if self.outcome is Outcome.MATCHED:
            signals = {finding.signal.value for finding in self.findings}
            if not self.findings or signals != set(self.signals):
                raise ValueError("matched assessment signals must exactly describe its distress findings")
        elif self.findings:
            raise ValueError("only matched assessments may include distress findings")
        return self

    @property
    def response_adaptations(self) -> tuple[ResponseAdaptation, ...]:
        """Deduplicated adaptations retained after calibration."""

        return tuple(
            dict.fromkeys(adaptation for finding in self.findings for adaptation in finding.response_adaptations),
        )


_DISTRESS_PROMPT = PromptSpec.from_package(
    package="psysafe.classifiers",
    resource="policies/distress.md",
)
_DISTRESS_SIGNALS = frozenset(signal.value for signal in DistressSignal)


class DistressSupportClassifier(PolicyClassifier[DistressObservation]):
    """Detect support-relevant distress language without diagnosing a person."""

    _result_model = DistressAssessment

    def __init__(
        self,
        backend: StructuredBackend,
        *,
        failure_policy: FailurePolicy = FailurePolicy.RETURN_INDETERMINATE,
    ) -> None:
        super().__init__(
            classifier_id="distress_support",
            policy_version="2026.08.2",
            prompt=_DISTRESS_PROMPT,
            backend=backend,
            observation_model=DistressObservation,
            allowed_signals=_DISTRESS_SIGNALS,
            evidence_role=MessageRole.USER,
            failure_policy=failure_policy,
        )

    def calibrate(
        self,
        record: ObservationRecord[DistressObservation],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> DistressAssessment:
        preflight = super().calibrate(record, sensitivity=sensitivity)
        if preflight.outcome is Outcome.INDETERMINATE:
            return DistressAssessment(**preflight.model_dump())
        observation = self._observation_from_record(record)
        gate_ready_findings = tuple(
            finding
            for finding in observation.findings
            if is_direct_user_evidence(finding.subject, finding.source_context)
        )
        gate_ready_observation = observation.model_copy(update={"findings": gate_ready_findings})
        gate_ready_record = record.model_copy(update={"observation": gate_ready_observation})
        assessment = super().calibrate(gate_ready_record, sensitivity=sensitivity)
        selected = (
            select_findings(gate_ready_findings, Sensitivity(sensitivity))
            if assessment.outcome is Outcome.MATCHED
            else ()
        )
        payload = assessment.model_dump()
        payload["findings"] = selected
        return DistressAssessment.model_validate(payload)


__all__ = [
    "DistressAssessment",
    "DistressFinding",
    "DistressObservation",
    "DistressSignal",
    "DistressSupportClassifier",
    "ResponseAdaptation",
]
