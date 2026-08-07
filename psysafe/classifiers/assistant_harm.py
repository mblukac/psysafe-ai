"""Calibrated classification of harmful assistant behavior in context."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from psysafe.backends.base import StructuredBackend
from psysafe.classifiers.base import (
    MAX_FINDINGS,
    Finding,
    Observation,
    ObservationRecord,
    PolicyClassifier,
    select_findings,
)
from psysafe.classifiers.calibration import least_direct
from psysafe.classifiers.prompting import PromptSpec
from psysafe.core.classifier import FailurePolicy
from psysafe.core.contracts import (
    Assessment,
    EvidenceDirectness,
    MessageRole,
    Outcome,
    Sensitivity,
)


class AssistantHarmSignal(str, Enum):
    """Harm domain and assistant behavior represented as one unambiguous label."""

    SELF_HARM_ENCOURAGEMENT = "self_harm.encouragement"
    SELF_HARM_ENDORSEMENT = "self_harm.endorsement"
    SELF_HARM_ACTIONABLE_INSTRUCTIONS = "self_harm.actionable_instructions"
    EATING_DISORDER_ENCOURAGEMENT = "eating_disorder.encouragement"
    EATING_DISORDER_ENDORSEMENT = "eating_disorder.endorsement"
    EATING_DISORDER_ACTIONABLE_INSTRUCTIONS = "eating_disorder.actionable_instructions"
    DANGEROUS_SUBSTANCE_USE_ENCOURAGEMENT = "dangerous_substance_use.encouragement"
    DANGEROUS_SUBSTANCE_USE_ENDORSEMENT = "dangerous_substance_use.endorsement"
    DANGEROUS_SUBSTANCE_USE_ACTIONABLE_INSTRUCTIONS = "dangerous_substance_use.actionable_instructions"
    OTHER_SELF_DESTRUCTIVE_BEHAVIOR_ENCOURAGEMENT = "other_self_destructive_behavior.encouragement"
    OTHER_SELF_DESTRUCTIVE_BEHAVIOR_ENDORSEMENT = "other_self_destructive_behavior.endorsement"
    OTHER_SELF_DESTRUCTIVE_BEHAVIOR_ACTIONABLE_INSTRUCTIONS = "other_self_destructive_behavior.actionable_instructions"


class AssistantHarmFinding(Finding):
    """One harmful assistant behavior with safe message provenance."""

    signal: AssistantHarmSignal = Field(
        description="Observable assistant behavior, including both the behavior and harm domain.",
    )


class AssistantHarmObservation(Observation[AssistantHarmFinding]):
    """Structured assistant-behavior observations made before calibration."""


class AssistantHarmAssessment(Assessment):
    """A calibrated result retaining categorical behavior and safe provenance."""

    findings: tuple[AssistantHarmFinding, ...] = Field(default_factory=tuple, max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def findings_must_match_assessment(self) -> AssistantHarmAssessment:
        if self.outcome is Outcome.MATCHED:
            expected_signals = tuple(dict.fromkeys(finding.signal.value for finding in self.findings))
            if not self.findings or self.signals != expected_signals:
                raise ValueError("matched assessment signals must describe its findings")
            if self.evidence_directness is not least_direct(
                tuple(EvidenceDirectness(finding.directness) for finding in self.findings),
            ):
                raise ValueError("assessment directness must summarize its findings")
        elif self.findings:
            raise ValueError("only matched assessments may include findings")
        return self


_ASSISTANT_HARM_SIGNALS = frozenset(signal.value for signal in AssistantHarmSignal)
_ASSISTANT_HARM_PROMPT = PromptSpec.from_package(
    package="psysafe.classifiers",
    resource="policies/assistant_harm.md",
)


class AssistantHarmClassifier(PolicyClassifier[AssistantHarmObservation]):
    """Detect policy violations by an assistant without grading response quality."""

    _result_model = AssistantHarmAssessment

    def __init__(
        self,
        backend: StructuredBackend,
        *,
        failure_policy: FailurePolicy = FailurePolicy.RETURN_INDETERMINATE,
    ) -> None:
        super().__init__(
            classifier_id="assistant_harm",
            policy_version="2026.08.2",
            prompt=_ASSISTANT_HARM_PROMPT,
            backend=backend,
            observation_model=AssistantHarmObservation,
            allowed_signals=_ASSISTANT_HARM_SIGNALS,
            evidence_role=MessageRole.ASSISTANT,
            failure_policy=failure_policy,
        )

    def calibrate(
        self,
        record: ObservationRecord[AssistantHarmObservation],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> AssistantHarmAssessment:
        normalized_sensitivity = Sensitivity(sensitivity)
        observation = self._observation_from_record(record)
        assessment = super().calibrate(record, sensitivity=normalized_sensitivity)
        findings = (
            select_findings(observation.findings, normalized_sensitivity)
            if assessment.outcome is Outcome.MATCHED
            else ()
        )
        payload = assessment.model_dump()
        payload["findings"] = findings
        return AssistantHarmAssessment.model_validate(payload)


__all__ = [
    "AssistantHarmAssessment",
    "AssistantHarmClassifier",
    "AssistantHarmFinding",
    "AssistantHarmObservation",
    "AssistantHarmSignal",
]
