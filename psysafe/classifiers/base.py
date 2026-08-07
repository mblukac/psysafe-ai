"""Reusable structured policy classifier with deterministic calibration."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psysafe.backends.base import (
    BackendConfigurationError,
    BackendError,
    BackendInvalidResponseError,
    BackendProviderError,
    BackendTimeoutError,
    StructuredBackend,
    _fresh_backend_error,
    _fresh_configuration_error,
    _raise_backend_error,
    _raise_configuration_error,
)
from psysafe.classifiers.calibration import least_direct, matches_sensitivity, sensitivity_boundaries
from psysafe.classifiers.prompting import (
    CONVERSATION_FORMAT,
    PromptSpec,
    conversation_input_schema,
    encoded_message_ids,
    encoded_messages,
)
from psysafe.core.classifier import FailurePolicy, _resolve_classification_failure
from psysafe.core.contracts import (
    MAX_CONVERSATION_CONTENT_CHARS,
    MAX_CONVERSATION_MESSAGES,
    MAX_MESSAGE_CONTENT_CHARS,
    Assessment,
    AssessmentMetadata,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    MessageRole,
    Outcome,
    Sensitivity,
)

MAX_FINDINGS = 64
MAX_FINDING_MESSAGE_IDS = 16

FindingDirectness = Literal[
    EvidenceDirectness.AMBIGUOUS,
    EvidenceDirectness.CONTEXTUAL,
    EvidenceDirectness.EXPLICIT,
]
EvidenceMessageId = Annotated[
    str,
    Field(min_length=2, max_length=8, pattern=r"^m(?:0|[1-9][0-9]*)$"),
]


class Finding(BaseModel):
    """One independently calibratable observation made by the model.

    ``message_ids`` cite opaque positional IDs in the encoded input. Caller IDs
    never cross the provider boundary or enter results.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    signal: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$")
    directness: FindingDirectness
    message_ids: tuple[EvidenceMessageId, ...] = Field(min_length=1, max_length=MAX_FINDING_MESSAGE_IDS)

    @field_validator("message_ids")
    @classmethod
    def message_ids_must_be_safe_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("message IDs must be unique within a finding")
        return values


FindingT = TypeVar("FindingT", bound=Finding)
FindingT_co = TypeVar("FindingT_co", bound=Finding, covariant=True)


def select_findings(
    findings: tuple[FindingT, ...],
    sensitivity: Sensitivity,
) -> tuple[FindingT, ...]:
    """Filter findings with the shared deterministic sensitivity boundary."""

    normalized_sensitivity = Sensitivity(sensitivity)
    return tuple(finding for finding in findings if matches_sensitivity(finding.directness, normalized_sensitivity))


class Observation(BaseModel, Generic[FindingT_co]):
    """The complete sensitivity-independent model output for a conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    findings: tuple[FindingT_co, ...] = Field(max_length=MAX_FINDINGS)
    insufficient_context: bool

    @model_validator(mode="after")
    def insufficient_context_must_not_assert_findings(self) -> Observation[FindingT_co]:
        if self.insufficient_context and self.findings:
            raise ValueError("an insufficient-context observation cannot assert findings")
        return self


ObservationT = TypeVar("ObservationT", bound=Observation[Any])


class ObservationRecord(BaseModel, Generic[ObservationT]):
    """A policy-labelled observation, not proof of execution provenance.

    Records can be serialized or constructed by callers, so their labels and
    metadata are assertions rather than an authentication mechanism. Records
    returned directly by :meth:`PolicyClassifier.observe` have been checked
    against that call's conversation. Revalidate restored or external records
    with :meth:`PolicyClassifier.validate_record` before relying on citations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$")
    policy_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
    observation: ObservationT
    metadata: AssessmentMetadata = Field(default_factory=AssessmentMetadata)


class ConversationInputConstraints(BaseModel):
    """Relational input limits that JSON Schema cannot fully express."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_messages: int = MAX_CONVERSATION_MESSAGES
    max_message_content_chars: int = MAX_MESSAGE_CONTENT_CHARS
    max_total_content_chars: int = MAX_CONVERSATION_CONTENT_CHARS
    message_id_sequence: Literal["m0_to_mN_in_message_order"] = "m0_to_mN_in_message_order"


class ClassifierSpec(BaseModel):
    """Portable policy and schema contract for non-Python runtimes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: str
    policy_version: str
    instructions: str
    input_format: str
    input_schema: dict[str, object]
    input_constraints: ConversationInputConstraints
    allowed_signals: tuple[str, ...] | None
    evidence_role: MessageRole | None
    sensitivity_boundaries: dict[str, tuple[str, ...]]
    observation_schema: dict[str, object]


def _signal_value(signal: object) -> str:
    if isinstance(signal, Enum):
        value = signal.value
        if isinstance(value, str):
            return value
    if isinstance(signal, str):
        return signal
    raise BackendInvalidResponseError from None


def _resolve_typed_failure(
    *,
    classifier_id: str,
    policy_version: str,
    sensitivity: Sensitivity,
    reason: IndeterminateReason,
    policy: FailurePolicy,
    metadata: AssessmentMetadata,
    assessment_model: type[Assessment],
) -> Assessment:
    assessment = _resolve_classification_failure(
        classifier_id=classifier_id,
        policy_version=policy_version,
        sensitivity=sensitivity,
        reason=reason,
        policy=policy,
        metadata=metadata,
    )
    if isinstance(assessment, assessment_model):
        return assessment
    return assessment_model.model_validate(assessment.model_dump())


class PolicyClassifier(Generic[ObservationT]):
    """Make one structured observation, then apply sensitivity locally.

    The model never receives the selected sensitivity. Callers retain the
    policy-bound ``ObservationRecord`` returned by ``observe`` and recalibrate
    it without another provider request. The classifier retains no records.
    """

    _result_model: type[Assessment] = Assessment

    def __init__(
        self,
        *,
        classifier_id: str,
        policy_version: str,
        prompt: PromptSpec,
        backend: StructuredBackend,
        observation_model: type[ObservationT],
        allowed_signals: frozenset[str] | None = None,
        evidence_role: MessageRole | None = None,
        failure_policy: FailurePolicy = FailurePolicy.RETURN_INDETERMINATE,
    ) -> None:
        if not issubclass(observation_model, Observation):
            raise TypeError("observation_model must be an Observation model")
        if not issubclass(self._result_model, Assessment):
            raise TypeError("classifier result model must be an Assessment model")
        metadata = AssessmentMetadata(provider=backend.provider, model=backend.model)
        # Reuse the public Assessment validation for classifier and policy IDs.
        Assessment(
            classifier_id=classifier_id,
            policy_version=policy_version,
            outcome=Outcome.NOT_MATCHED,
            metadata=metadata,
        )
        self._classifier_id = classifier_id
        self._policy_version = policy_version
        self._prompt = prompt
        self._backend = backend
        self._observation_model = observation_model
        self._assessment_model = self._result_model
        self._allowed_signals = frozenset(allowed_signals) if allowed_signals is not None else None
        self._evidence_role = evidence_role
        self._failure_policy = FailurePolicy(failure_policy)
        self._metadata = metadata

    @property
    def classifier_id(self) -> str:
        return self._classifier_id

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def assessment_metadata(self) -> AssessmentMetadata:
        """Safe provider/model provenance for custom assessment subclasses."""

        return self._metadata

    def export_spec(self) -> ClassifierSpec:
        """Export the fixed cross-runtime contract without execution data."""

        return ClassifierSpec(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            instructions=self._prompt.instructions,
            input_format=CONVERSATION_FORMAT,
            input_schema=conversation_input_schema(),
            input_constraints=ConversationInputConstraints(),
            allowed_signals=tuple(sorted(self._allowed_signals)) if self._allowed_signals is not None else None,
            evidence_role=self._evidence_role,
            sensitivity_boundaries=sensitivity_boundaries(),
            observation_schema=self._observation_model.model_json_schema(),
        )

    def bind(self, observation: ObservationT) -> ObservationRecord[ObservationT]:
        """Bind a trusted local observation to this classifier and policy.

        This validates the observation type, signal taxonomy, and opaque ID
        syntax. Only :meth:`observe` can additionally validate citations
        against a particular conversation and required message role.
        """

        return self._bind(observation, metadata=AssessmentMetadata())

    def _bind(
        self,
        observation: ObservationT,
        *,
        metadata: AssessmentMetadata,
    ) -> ObservationRecord[ObservationT]:
        self._validate_observation_contract(observation)
        record_model = cast(
            type[ObservationRecord[ObservationT]],
            ObservationRecord.__class_getitem__(self._observation_model),
        )
        return record_model(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            observation=observation,
            metadata=metadata,
        )

    def observe(self, conversation: Conversation) -> ObservationRecord[ObservationT]:
        """Request and validate one sensitivity-independent observation."""

        failure: BackendError | None = None
        configuration_failure: BackendConfigurationError | None = None
        observation: ObservationT | None = None
        try:
            observation = self._backend.complete(
                instructions=self._prompt.instructions,
                input_text=self._prompt.encode(conversation),
                output_type=self._observation_model,
            )
            self._validate_observation(observation, conversation)
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        except TimeoutError:
            failure = BackendTimeoutError()
        # Custom backends are untrusted integration boundaries; sanitize failures.
        except Exception:  # noqa: BLE001
            failure = BackendProviderError()
        if configuration_failure is not None:
            del self, conversation, observation, failure
            _raise_configuration_error(configuration_failure)
        if failure is not None:
            del self, conversation, observation, configuration_failure
            _raise_backend_error(failure)
        if observation is None:
            failure = BackendInvalidResponseError()
            del self, conversation, observation, configuration_failure
            _raise_backend_error(failure)
        return self._bind(observation, metadata=self.assessment_metadata)

    async def aobserve(self, conversation: Conversation) -> ObservationRecord[ObservationT]:
        """Async equivalent of :meth:`observe`."""

        failure: BackendError | None = None
        configuration_failure: BackendConfigurationError | None = None
        observation: ObservationT | None = None
        try:
            observation = await self._backend.acomplete(
                instructions=self._prompt.instructions,
                input_text=self._prompt.encode(conversation),
                output_type=self._observation_model,
            )
            self._validate_observation(observation, conversation)
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        except TimeoutError:
            failure = BackendTimeoutError()
        # Custom backends are untrusted integration boundaries; sanitize failures.
        except Exception:  # noqa: BLE001
            failure = BackendProviderError()
        if configuration_failure is not None:
            del self, conversation, observation, failure
            _raise_configuration_error(configuration_failure)
        if failure is not None:
            del self, conversation, observation, configuration_failure
            _raise_backend_error(failure)
        if observation is None:
            failure = BackendInvalidResponseError()
            del self, conversation, observation, configuration_failure
            _raise_backend_error(failure)
        return self._bind(observation, metadata=self.assessment_metadata)

    def calibrate(
        self,
        record: ObservationRecord[ObservationT],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        """Apply a named boundary to a policy-bound observation record.

        This is a pure policy operation and does not authenticate serialized
        record labels or citations. Call :meth:`validate_record` first when a
        record came from storage, a queue, or another trust boundary.
        """

        normalized_sensitivity = Sensitivity(sensitivity)
        observation = self._observation_from_record(record)
        if observation.insufficient_context:
            return Assessment.indeterminate(
                classifier_id=self.classifier_id,
                policy_version=self.policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INSUFFICIENT_INPUT,
                metadata=record.metadata,
            )

        selected = select_findings(observation.findings, normalized_sensitivity)
        if not selected:
            return Assessment(
                classifier_id=self.classifier_id,
                policy_version=self.policy_version,
                sensitivity=normalized_sensitivity,
                outcome=Outcome.NOT_MATCHED,
                metadata=record.metadata,
            )

        signals = tuple(dict.fromkeys(_signal_value(finding.signal) for finding in selected))
        directness = least_direct(tuple(EvidenceDirectness(finding.directness) for finding in selected))
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=normalized_sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=directness,
            signals=signals,
            metadata=record.metadata,
        )

    def recalibrate(
        self,
        record: ObservationRecord[ObservationT],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        """Explicit alias for calibrating the same observation again."""

        return self.calibrate(record, sensitivity=sensitivity)

    def validate_record(
        self,
        record: ObservationRecord[ObservationT],
        conversation: Conversation,
    ) -> ObservationRecord[ObservationT]:
        """Revalidate an external record against the conversation it cites.

        Message order must be the same as at observation time because evidence
        IDs are opaque positions. Provider/model metadata remains descriptive,
        not authenticated provenance.
        """

        observation = self._observation_from_record(record)
        try:
            self._validate_observation(observation, conversation)
        except BackendInvalidResponseError:
            raise ValueError("observation record citations do not match this conversation") from None
        return record

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        normalized_sensitivity = Sensitivity(sensitivity)
        failure_reason: IndeterminateReason | None = None
        configuration_failure: BackendConfigurationError | None = None
        record: ObservationRecord[ObservationT] | None = None
        try:
            record = self.observe(conversation)
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
        except BackendError as caught:
            failure_reason = caught.reason
        if configuration_failure is not None:
            del self, conversation, record, failure_reason
            _raise_configuration_error(configuration_failure)
        if failure_reason is not None:
            classifier_id = self.classifier_id
            policy_version = self.policy_version
            policy = self._failure_policy
            metadata = self.assessment_metadata
            assessment_model = self._assessment_model
            if policy is FailurePolicy.RAISE:
                del self, conversation, record
            return _resolve_typed_failure(
                classifier_id=classifier_id,
                policy_version=policy_version,
                sensitivity=normalized_sensitivity,
                reason=failure_reason,
                policy=policy,
                metadata=metadata,
                assessment_model=assessment_model,
            )
        if record is None:
            classifier_id = self.classifier_id
            policy_version = self.policy_version
            policy = self._failure_policy
            metadata = self.assessment_metadata
            assessment_model = self._assessment_model
            if policy is FailurePolicy.RAISE:
                del self, conversation, record
            return _resolve_typed_failure(
                classifier_id=classifier_id,
                policy_version=policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=policy,
                metadata=metadata,
                assessment_model=assessment_model,
            )
        return self.calibrate(record, sensitivity=normalized_sensitivity)

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        normalized_sensitivity = Sensitivity(sensitivity)
        failure_reason: IndeterminateReason | None = None
        configuration_failure: BackendConfigurationError | None = None
        record: ObservationRecord[ObservationT] | None = None
        try:
            record = await self.aobserve(conversation)
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
        except BackendError as caught:
            failure_reason = caught.reason
        if configuration_failure is not None:
            del self, conversation, record, failure_reason
            _raise_configuration_error(configuration_failure)
        if failure_reason is not None:
            classifier_id = self.classifier_id
            policy_version = self.policy_version
            policy = self._failure_policy
            metadata = self.assessment_metadata
            assessment_model = self._assessment_model
            if policy is FailurePolicy.RAISE:
                del self, conversation, record
            return _resolve_typed_failure(
                classifier_id=classifier_id,
                policy_version=policy_version,
                sensitivity=normalized_sensitivity,
                reason=failure_reason,
                policy=policy,
                metadata=metadata,
                assessment_model=assessment_model,
            )
        if record is None:
            classifier_id = self.classifier_id
            policy_version = self.policy_version
            policy = self._failure_policy
            metadata = self.assessment_metadata
            assessment_model = self._assessment_model
            if policy is FailurePolicy.RAISE:
                del self, conversation, record
            return _resolve_typed_failure(
                classifier_id=classifier_id,
                policy_version=policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=policy,
                metadata=metadata,
                assessment_model=assessment_model,
            )
        return self.calibrate(record, sensitivity=normalized_sensitivity)

    def _observation_from_record(self, record: ObservationRecord[ObservationT]) -> ObservationT:
        if not isinstance(record, ObservationRecord):
            raise TypeError("record must be an ObservationRecord")
        if record.classifier_id != self.classifier_id or record.policy_version != self.policy_version:
            raise ValueError("observation record does not match this classifier policy")
        self._validate_observation_contract(record.observation)
        return record.observation

    def _validate_observation_contract(self, observation: ObservationT) -> None:
        if not isinstance(observation, self._observation_model):
            raise TypeError("observation does not match this classifier schema")
        for finding in observation.findings:
            signal = _signal_value(finding.signal)
            if self._allowed_signals is not None and signal not in self._allowed_signals:
                raise ValueError("observation contains a signal outside this classifier policy")

    def _validate_observation(self, observation: ObservationT, conversation: Conversation) -> None:
        try:
            self._validate_observation_contract(observation)
        except (TypeError, ValueError):
            raise BackendInvalidResponseError from None

        known_ids = encoded_message_ids(conversation)
        required_role_ids = (
            {
                encoded["id"]
                for encoded, message in zip(encoded_messages(conversation), conversation.messages, strict=True)
                if message.role is self._evidence_role
            }
            if self._evidence_role is not None
            else None
        )
        for finding in observation.findings:
            if not set(finding.message_ids) <= known_ids:
                raise BackendInvalidResponseError from None
            if required_role_ids is not None and not (set(finding.message_ids) & required_role_ids):
                raise BackendInvalidResponseError from None


__all__ = [
    "MAX_FINDINGS",
    "MAX_FINDING_MESSAGE_IDS",
    "ClassifierSpec",
    "ConversationInputConstraints",
    "Finding",
    "FindingDirectness",
    "FindingT",
    "Observation",
    "ObservationRecord",
    "ObservationT",
    "PolicyClassifier",
    "select_findings",
]
