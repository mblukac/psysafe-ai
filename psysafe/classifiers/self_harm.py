"""Calibrated screening for observable self-harm and suicide signals."""

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


class SelfHarmSignal(str, Enum):
    """Observable signals covered by the screening policy."""

    IDEATION = "ideation"
    INTENT = "intent"
    PLAN_OR_ACCESS = "plan_or_access"
    PREPARATORY_BEHAVIOR = "preparatory_behavior"
    SUICIDE_ATTEMPT = "suicide_attempt"
    SELF_INJURY_UNCLEAR_INTENT = "self_injury_unclear_intent"
    NONSUICIDAL_SELF_INJURY = "nonsuicidal_self_injury"


class SelfHarmSubject(str, Enum):
    """Whose experience the language describes."""

    SELF = "self"
    THIRD_PARTY = "third_party"
    UNCLEAR = "unclear"


class SelfHarmSourceContext(str, Enum):
    """How evidence is presented, with fictional taking precedence over quoted."""

    DIRECT = "direct"
    QUOTED = "quoted"
    FICTIONAL = "fictional"
    UNCLEAR = "unclear"


class SelfHarmTimeframe(str, Enum):
    """When the described signal applies."""

    CURRENT = "current"
    RECENT = "recent"
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    UNCLEAR = "unclear"


class SelfHarmFinding(Finding):
    """One observable signal with its own context and message provenance."""

    signal: SelfHarmSignal = Field(
        description="The observable self-harm or suicide signal category.",
    )
    subject: SelfHarmSubject = Field(description="Whose experience is described.")
    source_context: SelfHarmSourceContext = Field(description="How the evidence is presented.")
    timeframe: SelfHarmTimeframe = Field(description="When the described signal applies.")


class SelfHarmObservation(Observation[SelfHarmFinding]):
    """Structured self-harm observations made before calibration."""


class SelfHarmAssessment(Assessment):
    """A calibrated result retaining categorical context and safe provenance."""

    findings: tuple[SelfHarmFinding, ...] = Field(default_factory=tuple, max_length=MAX_FINDINGS)

    @model_validator(mode="after")
    def findings_must_match_assessment(self) -> SelfHarmAssessment:
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


_SELF_HARM_SIGNALS = frozenset(signal.value for signal in SelfHarmSignal)
_SELF_HARM_PROMPT = PromptSpec.from_package(
    package="psysafe.classifiers",
    resource="policies/self_harm.md",
)


class SelfHarmClassifier(PolicyClassifier[SelfHarmObservation]):
    """Screen for evidence categories without making a diagnosis."""

    _result_model = SelfHarmAssessment

    def __init__(
        self,
        backend: StructuredBackend,
        *,
        failure_policy: FailurePolicy = FailurePolicy.RETURN_INDETERMINATE,
    ) -> None:
        super().__init__(
            classifier_id="self_harm_and_suicide_signals",
            policy_version="2026.08.2",
            prompt=_SELF_HARM_PROMPT,
            backend=backend,
            observation_model=SelfHarmObservation,
            allowed_signals=_SELF_HARM_SIGNALS,
            evidence_role=MessageRole.USER,
            failure_policy=failure_policy,
        )

    def calibrate(
        self,
        record: ObservationRecord[SelfHarmObservation],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> SelfHarmAssessment:
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
        return SelfHarmAssessment.model_validate(payload)


__all__ = [
    "SelfHarmAssessment",
    "SelfHarmClassifier",
    "SelfHarmFinding",
    "SelfHarmObservation",
    "SelfHarmSignal",
    "SelfHarmSourceContext",
    "SelfHarmSubject",
    "SelfHarmTimeframe",
]
