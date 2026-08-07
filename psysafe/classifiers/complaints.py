"""Complaint classification and categorical escalation routing."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator

from psysafe.backends.base import BackendInvalidResponseError, StructuredBackend
from psysafe.classifiers.base import (
    MAX_FINDINGS,
    Finding,
    Observation,
    ObservationRecord,
    PolicyClassifier,
    _target_evidence_id,
    select_findings,
)
from psysafe.classifiers.context import EvidenceSubject, SourceContext, is_direct_user_evidence
from psysafe.classifiers.prompting import PromptSpec, encoded_message_ids, encoded_messages
from psysafe.core.classifier import FailurePolicy
from psysafe.core.contracts import (
    Assessment,
    Conversation,
    IndeterminateReason,
    MessageRole,
    Outcome,
    Sensitivity,
)


class ComplaintCategory(str, Enum):
    """Mutually exclusive category applied to each complaint finding."""

    SERVICE_QUALITY = "service_quality"
    PRODUCT_OR_OUTCOME = "product_or_outcome"
    BILLING_OR_PAYMENT = "billing_or_payment"
    ACCESS_OR_COMMUNICATION = "access_or_communication"
    STAFF_CONDUCT = "staff_conduct"
    PRIVACY_OR_DATA = "privacy_or_data"
    OTHER = "other"


class EscalationReason(str, Enum):
    """Observable reasons a downstream workflow may choose human review."""

    EXPLICIT_HUMAN_REQUEST = "explicit_human_request"
    REPEATED_UNRESOLVED = "repeated_unresolved"
    LEGAL_OR_REGULATORY_CONCERN = "legal_or_regulatory_concern"
    SAFETY_OR_SUPPORT_NEED = "safety_or_support_need"


MAX_COMPLAINT_ESCALATIONS = MAX_FINDINGS * len(EscalationReason)


class ComplaintEscalation(Finding):
    """One independently calibrated escalation reason."""

    signal: EscalationReason = Field(description="Observable reason for possible human review.")
    subject: EvidenceSubject = Field(description="Whom the escalation evidence concerns.")
    source_context: SourceContext = Field(description="How the escalation evidence is presented.")

    @property
    def reason(self) -> EscalationReason:
        return self.signal


class ComplaintFinding(Finding):
    """One categorized complaint with safe evidence locations."""

    signal: ComplaintCategory = Field(description="Category for this complaint finding.")
    subject: EvidenceSubject = Field(description="Whose dissatisfaction is expressed.")
    source_context: SourceContext = Field(description="How the evidence is presented.")

    @property
    def category(self) -> ComplaintCategory:
        """Domain-specific alias for the shared categorical signal."""

        return self.signal


class ComplaintsObservation(Observation[ComplaintFinding]):
    """Sensitivity-independent complaint observations."""

    escalations: tuple[ComplaintEscalation, ...] = Field(
        default_factory=tuple,
        max_length=MAX_COMPLAINT_ESCALATIONS,
    )

    @field_validator("findings", "escalations")
    @classmethod
    def exact_duplicates_are_not_allowed(
        cls,
        values: tuple[ComplaintFinding, ...] | tuple[ComplaintEscalation, ...],
    ) -> tuple[ComplaintFinding, ...] | tuple[ComplaintEscalation, ...]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate complaint observations are not allowed")
        return values

    @model_validator(mode="after")
    def insufficient_context_has_no_escalations(self) -> ComplaintsObservation:
        if self.insufficient_context and self.escalations:
            raise ValueError("an insufficient-context observation cannot assert escalations")
        return self


class ComplaintsAssessment(Assessment):
    """Calibrated complaint categories and explicit escalation reasons."""

    findings: tuple[ComplaintFinding, ...] = Field(default_factory=tuple, max_length=MAX_FINDINGS)
    escalations: tuple[ComplaintEscalation, ...] = Field(
        default_factory=tuple,
        max_length=MAX_FINDINGS * len(EscalationReason),
    )

    @field_validator("escalations")
    @classmethod
    def escalations_must_be_unique(
        cls,
        values: tuple[ComplaintEscalation, ...],
    ) -> tuple[ComplaintEscalation, ...]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate calibrated escalations are not allowed")
        return values

    @model_validator(mode="after")
    def findings_must_match_assessment(self) -> ComplaintsAssessment:
        if self.outcome is Outcome.MATCHED:
            categories = {finding.signal.value for finding in self.findings}
            if not self.findings or categories != set(self.signals):
                raise ValueError("matched assessment signals must exactly describe complaint categories")
        elif self.findings:
            raise ValueError("only matched assessments may include complaint findings")
        if self.outcome is Outcome.INDETERMINATE and self.escalations:
            raise ValueError("indeterminate assessments cannot assert escalation evidence")
        return self

    @property
    def escalation_reasons(self) -> tuple[EscalationReason, ...]:
        """Deduplicated routing reasons retained after calibration."""

        return tuple(dict.fromkeys(escalation.signal for escalation in self.escalations))

    @property
    def review_signals(self) -> tuple[str, ...]:
        """Categorical human-review reasons, independent of complaint match."""

        return tuple(reason.value for reason in self.escalation_reasons)


_COMPLAINTS_PROMPT = PromptSpec.from_package(
    package="psysafe.classifiers",
    resource="policies/complaints.md",
)
_COMPLAINT_SIGNALS = frozenset(category.value for category in ComplaintCategory)


class ComplaintsClassifier(PolicyClassifier[ComplaintsObservation]):
    """Categorize complaints; application code chooses any escalation action."""

    _result_model = ComplaintsAssessment

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        """Finite escalation vocabulary emitted by complaint assessments."""

        return tuple(reason.value for reason in EscalationReason)

    def __init__(
        self,
        backend: StructuredBackend,
        *,
        failure_policy: FailurePolicy = FailurePolicy.RETURN_INDETERMINATE,
    ) -> None:
        super().__init__(
            classifier_id="complaints",
            policy_version="2026.08.2",
            prompt=_COMPLAINTS_PROMPT,
            backend=backend,
            observation_model=ComplaintsObservation,
            allowed_signals=_COMPLAINT_SIGNALS,
            evidence_role=MessageRole.USER,
            failure_policy=failure_policy,
        )

    def _observation_is_saturated(self, observation: ComplaintsObservation) -> bool:
        """Treat either bounded observation collection as possibly truncated."""

        return (
            super()._observation_is_saturated(observation) or len(observation.escalations) >= MAX_COMPLAINT_ESCALATIONS
        )

    def calibrate(
        self,
        record: ObservationRecord[ComplaintsObservation],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> ComplaintsAssessment:
        return self._calibrate(
            record,
            sensitivity=sensitivity,
            target_message_id=None,
        )

    def _calibrate_target_record(
        self,
        record: ObservationRecord[ComplaintsObservation],
        *,
        target_message_index: int,
        sensitivity: Sensitivity,
    ) -> ComplaintsAssessment:
        """Calibrate complaint and escalation evidence for a validated target."""

        target_message_id = _target_evidence_id(target_message_index)
        if record.target_message_id not in {None, target_message_id}:
            return ComplaintsAssessment(
                **self._indeterminate_from_record(
                    record,
                    sensitivity=Sensitivity(sensitivity),
                    reason=IndeterminateReason.INVALID_RESPONSE,
                ).model_dump(),
            )
        unscoped_record = record.model_copy(update={"target_message_id": None})
        return self._calibrate(
            unscoped_record,
            sensitivity=sensitivity,
            target_message_id=target_message_id,
        )

    def _calibrate(
        self,
        record: ObservationRecord[ComplaintsObservation],
        *,
        sensitivity: Sensitivity,
        target_message_id: str | None,
    ) -> ComplaintsAssessment:
        normalized_sensitivity = Sensitivity(sensitivity)
        preflight = super().calibrate(record, sensitivity=normalized_sensitivity)
        if preflight.outcome is Outcome.INDETERMINATE:
            return ComplaintsAssessment(**preflight.model_dump())
        observation = self._observation_from_record(record)
        gate_ready_findings = tuple(
            finding
            for finding in observation.findings
            if is_direct_user_evidence(finding.subject, finding.source_context)
            and (target_message_id is None or target_message_id in finding.message_ids)
        )
        gate_ready_observation = observation.model_copy(update={"findings": gate_ready_findings})
        gate_ready_record = record.model_copy(update={"observation": gate_ready_observation})
        assessment = super().calibrate(gate_ready_record, sensitivity=normalized_sensitivity)
        selected_findings = (
            select_findings(gate_ready_findings, normalized_sensitivity)
            if assessment.outcome is Outcome.MATCHED
            else ()
        )
        selected_escalations = (
            select_findings(
                tuple(
                    escalation
                    for escalation in observation.escalations
                    if is_direct_user_evidence(escalation.subject, escalation.source_context)
                    and (target_message_id is None or target_message_id in escalation.message_ids)
                ),
                normalized_sensitivity,
            )
            if assessment.outcome is not Outcome.INDETERMINATE
            else ()
        )
        payload = assessment.model_dump()
        payload["findings"] = selected_findings
        payload["escalations"] = selected_escalations
        return ComplaintsAssessment.model_validate(payload)

    def _validate_observation(self, observation: ComplaintsObservation, conversation: Conversation) -> None:
        super()._validate_observation(observation, conversation)
        known_ids = encoded_message_ids(conversation)
        user_ids = {
            encoded["id"]
            for encoded, message in zip(encoded_messages(conversation), conversation.messages, strict=True)
            if message.role is MessageRole.USER
        }
        for escalation in observation.escalations:
            if not set(escalation.message_ids) <= known_ids or not (set(escalation.message_ids) & user_ids):
                raise BackendInvalidResponseError from None

    def _validate_target_observation(
        self,
        observation: ComplaintsObservation,
        target_message_index: int,
    ) -> None:
        super()._validate_target_observation(observation, target_message_index)
        target_id = _target_evidence_id(target_message_index)
        if any(target_id not in escalation.message_ids for escalation in observation.escalations):
            raise BackendInvalidResponseError from None


__all__ = [
    "MAX_COMPLAINT_ESCALATIONS",
    "ComplaintCategory",
    "ComplaintEscalation",
    "ComplaintFinding",
    "ComplaintsAssessment",
    "ComplaintsClassifier",
    "ComplaintsObservation",
    "EscalationReason",
]
