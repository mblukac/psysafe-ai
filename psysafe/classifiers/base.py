"""Reusable structured policy classifier with deterministic calibration."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Annotated, Any, Generic, Literal, NoReturn, TypeVar, cast

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
    _raise_cancelled,
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


def _raise_target_error() -> NoReturn:
    raise ValueError("target message does not satisfy the classifier boundary") from None


def _raise_sensitivity_error() -> NoReturn:
    raise ValueError("value is not a valid Sensitivity") from None


def _raise_record_validation_error() -> NoReturn:
    raise ValueError("observation record does not match this conversation") from None


def _coerce_sensitivity(value: object) -> Sensitivity:
    """Accept only exact public enum/string values without invoking subclasses."""

    if type(value) is Sensitivity:
        return value
    if type(value) is str:
        try:
            return Sensitivity(value)
        except ValueError:
            pass
    _raise_sensitivity_error()


def _validated_target_index(
    conversation: Conversation,
    target_message_index: object,
    evidence_role: MessageRole | None,
) -> int:
    if type(conversation) is not Conversation:
        raise TypeError("conversation must be a Conversation")
    if type(target_message_index) is not int:
        raise ValueError("invalid target message index")
    if target_message_index < 0 or target_message_index >= len(conversation.messages):
        raise ValueError("invalid target message index")
    if evidence_role is not None and conversation.messages[target_message_index].role is not evidence_role:
        raise ValueError("target message role does not match classifier evidence role")
    return target_message_index


def _target_evidence_id(target_message_index: int) -> str:
    if type(target_message_index) is not int or not 0 <= target_message_index < MAX_CONVERSATION_MESSAGES:
        raise ValueError("target message index is outside the conversation contract")
    return f"m{target_message_index}"


class Observation(BaseModel, Generic[FindingT_co]):
    """The complete sensitivity-independent model output for a conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    findings: tuple[FindingT_co, ...] = Field(max_length=MAX_FINDINGS)
    insufficient_context: bool
    output_truncated: bool = False

    @model_validator(mode="after")
    def insufficient_context_must_not_assert_findings(self) -> Observation[FindingT_co]:
        if self.insufficient_context and (self.findings or self.output_truncated):
            raise ValueError("an insufficient-context observation cannot assert or truncate findings")
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
    target_message_id: EvidenceMessageId | None = Field(
        default=None,
        description="Target scope used during observation; None means exhaustive conversation scope.",
    )
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
    allowed_review_signals: tuple[str, ...]
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
    return _safe_typed_indeterminate(
        assessment,
        assessment_model=assessment_model,
    )


def _safe_typed_indeterminate(
    assessment: Assessment,
    *,
    assessment_model: type[Assessment],
) -> Assessment:
    """Use a typed result only when it preserves the sealed failure state."""

    if assessment_model is Assessment:
        return assessment
    candidate: Assessment | None = None
    try:
        candidate = assessment_model.model_validate(assessment.model_dump())
        if type(candidate) is not assessment_model:
            return assessment
        state = object.__getattribute__(candidate, "__dict__")
        if not isinstance(state, dict):
            return assessment
        if (
            state.get("classifier_id") != assessment.classifier_id
            or state.get("policy_version") != assessment.policy_version
            or state.get("sensitivity") is not assessment.sensitivity
            or state.get("outcome") is not Outcome.INDETERMINATE
            or state.get("evidence_directness") is not EvidenceDirectness.NONE
            or type(state.get("signals")) is not tuple
            or state.get("signals") != ()
            or state.get("indeterminate_reason") is not assessment.indeterminate_reason
            or type(state.get("metadata")) is not AssessmentMetadata
            or state.get("metadata") != assessment.metadata
        ):
            return assessment
        base_fields = frozenset(Assessment.model_fields)
        for field_name, value in state.items():
            if field_name in base_fields:
                continue
            if value is None or value is False:
                continue
            if type(value) in {tuple, list, dict, set, frozenset} and len(value) == 0:
                continue
            return assessment
    except (asyncio.CancelledError, Exception):
        return assessment
    return candidate or assessment


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

    @property
    def evidence_role(self) -> MessageRole | None:
        """Role that every actionable finding must cite, or ``None`` for any role."""

        return self._evidence_role

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        """Finite categorical vocabulary safe for gates and evaluation output."""

        return tuple(sorted(self._allowed_signals or ()))

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        """Independent routing vocabulary; empty for ordinary classifiers."""

        return ()

    def export_spec(self) -> ClassifierSpec:
        """Export the fixed cross-runtime contract without execution data."""

        return ClassifierSpec(
            classifier_id=self._classifier_id,
            policy_version=self._policy_version,
            instructions=self._prompt.provider_instructions(),
            input_format=CONVERSATION_FORMAT,
            input_schema=conversation_input_schema(),
            input_constraints=ConversationInputConstraints(),
            allowed_signals=tuple(sorted(self._allowed_signals)) if self._allowed_signals is not None else None,
            allowed_review_signals=self.allowed_review_signals,
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

        return self._bind(
            observation,
            metadata=AssessmentMetadata(),
            target_message_id=None,
        )

    def _bind(
        self,
        observation: ObservationT,
        *,
        metadata: AssessmentMetadata,
        target_message_id: str | None,
    ) -> ObservationRecord[ObservationT]:
        self._validate_observation_contract(observation)
        record_model = cast(
            type[ObservationRecord[ObservationT]],
            ObservationRecord.__class_getitem__(self._observation_model),
        )
        return record_model(
            classifier_id=self._classifier_id,
            policy_version=self._policy_version,
            target_message_id=target_message_id,
            observation=observation,
            metadata=metadata,
        )

    def observe(
        self,
        conversation: Conversation,
        *,
        target_message_index: int | None = None,
    ) -> ObservationRecord[ObservationT]:
        """Request and validate one sensitivity-independent observation."""

        target_failure = False
        normalized_target: int | None = None
        if target_message_index is not None:
            try:
                normalized_target = _validated_target_index(
                    conversation,
                    target_message_index,
                    self._evidence_role,
                )
            except Exception:  # noqa: BLE001 - subclass properties are untrusted.
                target_failure = True
        if target_failure:
            del conversation
            _raise_target_error()
        failure: BackendError | None = None
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        observation: ObservationT | None = None
        record: ObservationRecord[ObservationT] | None = None
        try:
            observation = self._backend.complete(
                instructions=self._prompt.provider_instructions(),
                input_text=self._prompt.encode(
                    conversation,
                    target_message_index=normalized_target,
                ),
                output_type=self._observation_model,
            )
            self._validate_observation(observation, conversation)
            if normalized_target is not None:
                self._validate_target_observation(observation, normalized_target)
            record = self._bind(
                observation,
                metadata=self._metadata,
                target_message_id=(_target_evidence_id(normalized_target) if normalized_target is not None else None),
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
            if configuration_failure is None:
                failure = BackendProviderError()
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        except TimeoutError:
            failure = BackendTimeoutError()
        # Custom backends are untrusted integration boundaries; sanitize failures.
        except Exception:  # noqa: BLE001
            failure = BackendProviderError()
        if cancelled:
            del self, conversation, observation, record, failure, configuration_failure
            _raise_cancelled()
        if configuration_failure is not None:
            del self, conversation, observation, record, failure
            _raise_configuration_error(configuration_failure)
        if failure is not None:
            del self, conversation, observation, record, configuration_failure
            _raise_backend_error(failure)
        if observation is None or record is None:
            failure = BackendInvalidResponseError()
            del self, conversation, observation, record, configuration_failure
            _raise_backend_error(failure)
        return record

    async def aobserve(
        self,
        conversation: Conversation,
        *,
        target_message_index: int | None = None,
    ) -> ObservationRecord[ObservationT]:
        """Async equivalent of :meth:`observe`."""

        target_failure = False
        normalized_target: int | None = None
        if target_message_index is not None:
            try:
                normalized_target = _validated_target_index(
                    conversation,
                    target_message_index,
                    self._evidence_role,
                )
            except Exception:  # noqa: BLE001 - subclass properties are untrusted.
                target_failure = True
        if target_failure:
            del conversation
            _raise_target_error()
        failure: BackendError | None = None
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        observation: ObservationT | None = None
        record: ObservationRecord[ObservationT] | None = None
        try:
            observation = await self._backend.acomplete(
                instructions=self._prompt.provider_instructions(),
                input_text=self._prompt.encode(
                    conversation,
                    target_message_index=normalized_target,
                ),
                output_type=self._observation_model,
            )
            self._validate_observation(observation, conversation)
            if normalized_target is not None:
                self._validate_target_observation(observation, normalized_target)
            record = self._bind(
                observation,
                metadata=self._metadata,
                target_message_id=(_target_evidence_id(normalized_target) if normalized_target is not None else None),
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
            if configuration_failure is None:
                failure = BackendProviderError()
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        except TimeoutError:
            failure = BackendTimeoutError()
        # Custom backends are untrusted integration boundaries; sanitize failures.
        except Exception:  # noqa: BLE001
            failure = BackendProviderError()
        if cancelled:
            del self, conversation, observation, record, failure, configuration_failure
            _raise_cancelled()
        if configuration_failure is not None:
            del self, conversation, observation, record, failure
            _raise_configuration_error(configuration_failure)
        if failure is not None:
            del self, conversation, observation, record, configuration_failure
            _raise_backend_error(failure)
        if observation is None or record is None:
            failure = BackendInvalidResponseError()
            del self, conversation, observation, record, configuration_failure
            _raise_backend_error(failure)
        return record

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
        if record.target_message_id is not None:
            return self._indeterminate_from_record(
                record,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INVALID_RESPONSE,
            )
        if observation.output_truncated or self._observation_is_saturated(observation):
            return self._indeterminate_from_record(
                record,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INVALID_RESPONSE,
            )
        if observation.insufficient_context:
            return self._indeterminate_from_record(
                record,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INSUFFICIENT_INPUT,
            )

        selected = select_findings(observation.findings, normalized_sensitivity)
        if not selected:
            return Assessment(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=normalized_sensitivity,
                outcome=Outcome.NOT_MATCHED,
                metadata=record.metadata,
            )

        signals = tuple(dict.fromkeys(_signal_value(finding.signal) for finding in selected))
        directness = least_direct(tuple(EvidenceDirectness(finding.directness) for finding in selected))
        return Assessment(
            classifier_id=self._classifier_id,
            policy_version=self._policy_version,
            sensitivity=normalized_sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=directness,
            signals=signals,
            metadata=record.metadata,
        )

    def _indeterminate_from_record(
        self,
        record: ObservationRecord[ObservationT],
        *,
        sensitivity: Sensitivity,
        reason: IndeterminateReason,
    ) -> Assessment:
        """Build the configured categorical result without execution data."""

        assessment = Assessment.indeterminate(
            classifier_id=self._classifier_id,
            policy_version=self._policy_version,
            sensitivity=sensitivity,
            reason=reason,
            metadata=record.metadata,
        )
        return _safe_typed_indeterminate(
            assessment,
            assessment_model=self._assessment_model,
        )

    def _observation_is_saturated(self, observation: ObservationT) -> bool:
        """Fail safely when the fixed output cap may have omitted evidence."""

        return len(observation.findings) >= MAX_FINDINGS

    def recalibrate(
        self,
        record: ObservationRecord[ObservationT],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        """Explicit alias for calibrating the same observation again."""

        return self.calibrate(record, sensitivity=sensitivity)

    def calibrate_target(
        self,
        record: ObservationRecord[ObservationT],
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        """Revalidate context, then calibrate evidence tied to one target.

        The observation may use the full conversation for context, while the
        resulting decision is bound to evidence that cites ``mN`` for the
        selected message. Domain classifiers may override this for nested,
        independently calibrated evidence.
        """

        target_failure = False
        normalized_target = 0
        try:
            normalized_target = _validated_target_index(
                conversation,
                target_message_index,
                self._evidence_role,
            )
            self.validate_record(record, conversation)
        except Exception:  # noqa: BLE001 - records and subclass properties are untrusted.
            target_failure = True
        if target_failure:
            del record, conversation, target_message_index, sensitivity
            _raise_target_error()
        del conversation
        return self._calibrate_target_record(
            record,
            target_message_index=normalized_target,
            sensitivity=sensitivity,
        )

    def _calibrate_target_record(
        self,
        record: ObservationRecord[ObservationT],
        *,
        target_message_index: int,
        sensitivity: Sensitivity,
    ) -> Assessment:
        """Calibrate a target already validated against its conversation."""

        target_id = _target_evidence_id(target_message_index)
        observation = self._observation_from_record(record)
        if record.target_message_id not in {None, target_id}:
            return self._indeterminate_from_record(
                record,
                sensitivity=Sensitivity(sensitivity),
                reason=IndeterminateReason.INVALID_RESPONSE,
            )
        if observation.output_truncated or self._observation_is_saturated(observation):
            return self._indeterminate_from_record(
                record,
                sensitivity=Sensitivity(sensitivity),
                reason=IndeterminateReason.INVALID_RESPONSE,
            )
        scoped_observation = observation.model_copy(
            update={
                "findings": tuple(finding for finding in observation.findings if target_id in finding.message_ids),
            },
        )
        scoped_record = record.model_copy(
            update={
                "observation": scoped_observation,
                "target_message_id": None,
            },
        )
        return self.calibrate(scoped_record, sensitivity=sensitivity)

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

        validation_failed = False
        observation: ObservationT | None = None
        try:
            observation = self._observation_from_record(record)
            self._validate_observation(observation, conversation)
            if record.target_message_id is not None:
                target_index = int(record.target_message_id[1:])
                _validated_target_index(conversation, target_index, self._evidence_role)
                self._validate_target_observation(observation, target_index)
        except Exception:  # noqa: BLE001 - external records are an untrusted boundary.
            validation_failed = True
        if validation_failed or observation is None:
            del self, record, conversation, observation
            _raise_record_validation_error()
        return record

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        invalid_sensitivity = False
        normalized_sensitivity = Sensitivity.BALANCED
        try:
            normalized_sensitivity = _coerce_sensitivity(sensitivity)
        except Exception:  # noqa: BLE001 - enum-like inputs are an untrusted boundary.
            invalid_sensitivity = True
        if invalid_sensitivity:
            del conversation, sensitivity
            _raise_sensitivity_error()
        attempt = self._observe_attempt(conversation, target_message_index=None)
        del conversation
        result: Assessment | None = None
        cancelled = False
        try:
            result = self._assessment_from_attempt(
                attempt,
                sensitivity=normalized_sensitivity,
                target_message_index=None,
            )
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            del attempt, result
            _raise_cancelled()
        if result is None:
            del attempt
            return _resolve_typed_failure(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=self._failure_policy,
                metadata=self._metadata,
                assessment_model=self._assessment_model,
            )
        return result

    def classify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        """Classify with full context while binding evidence to one message."""

        invalid_sensitivity = False
        normalized_sensitivity = Sensitivity.BALANCED
        try:
            normalized_sensitivity = _coerce_sensitivity(sensitivity)
        except Exception:  # noqa: BLE001 - subclass properties are untrusted.
            invalid_sensitivity = True
        if invalid_sensitivity:
            del conversation, sensitivity
            _raise_sensitivity_error()
        target_failure = False
        normalized_target = 0
        try:
            normalized_target = _validated_target_index(
                conversation,
                target_message_index,
                self._evidence_role,
            )
        except Exception:  # noqa: BLE001 - subclass properties are untrusted.
            target_failure = True
        if target_failure:
            del conversation, target_message_index
            _raise_target_error()
        attempt = self._observe_attempt(
            conversation,
            target_message_index=normalized_target,
        )
        del conversation
        result: Assessment | None = None
        cancelled = False
        try:
            result = self._assessment_from_attempt(
                attempt,
                sensitivity=normalized_sensitivity,
                target_message_index=normalized_target,
            )
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            del attempt, result
            _raise_cancelled()
        if result is None:
            del attempt
            return _resolve_typed_failure(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=self._failure_policy,
                metadata=self._metadata,
                assessment_model=self._assessment_model,
            )
        return result

    def _observe_attempt(
        self,
        conversation: Conversation,
        *,
        target_message_index: int | None,
    ) -> tuple[
        ObservationRecord[ObservationT] | None,
        IndeterminateReason | None,
        BackendConfigurationError | None,
        bool,
    ]:
        """Observe once and return only sanitized categorical failure state."""

        failure_reason: IndeterminateReason | None = None
        configuration_failure: BackendConfigurationError | None = None
        cancelled = False
        record: ObservationRecord[ObservationT] | None = None
        try:
            record = self.observe(
                conversation,
                target_message_index=target_message_index,
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
            if configuration_failure is None:
                failure_reason = IndeterminateReason.PROVIDER_ERROR
        except BackendError as caught:
            failure_reason = _fresh_backend_error(caught).reason
        except TimeoutError:
            failure_reason = IndeterminateReason.TIMEOUT
        except Exception:  # noqa: BLE001 - subclasses are an untrusted boundary.
            failure_reason = IndeterminateReason.INTERNAL_ERROR
        if record is None and failure_reason is None and configuration_failure is None and not cancelled:
            failure_reason = IndeterminateReason.INTERNAL_ERROR
        return record, failure_reason, configuration_failure, cancelled

    def _assessment_from_attempt(
        self,
        attempt: tuple[
            ObservationRecord[ObservationT] | None,
            IndeterminateReason | None,
            BackendConfigurationError | None,
            bool,
        ],
        *,
        sensitivity: Sensitivity,
        target_message_index: int | None,
    ) -> Assessment:
        """Calibrate sanitized observation state after raw input is out of scope."""

        record, failure_reason, configuration_failure, cancelled = attempt
        if cancelled:
            del attempt, record, configuration_failure, failure_reason
            _raise_cancelled()
        if configuration_failure is not None:
            _raise_configuration_error(configuration_failure)
        if failure_reason is not None or record is None:
            del attempt, record
            return _resolve_typed_failure(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=sensitivity,
                reason=failure_reason or IndeterminateReason.INTERNAL_ERROR,
                policy=self._failure_policy,
                metadata=self._metadata,
                assessment_model=self._assessment_model,
            )
        calibration_failed = False
        calibration_cancelled = False
        result: Assessment | None = None
        try:
            if target_message_index is None:
                result = self.calibrate(record, sensitivity=sensitivity)
            else:
                result = self._calibrate_target_record(
                    record,
                    target_message_index=target_message_index,
                    sensitivity=sensitivity,
                )
            if (
                result is None
                or type(result) is not self._assessment_model
                or result.classifier_id != self._classifier_id
                or result.policy_version != self._policy_version
                or result.sensitivity is not sensitivity
                or result.metadata != record.metadata
            ):
                calibration_failed = True
        except asyncio.CancelledError:
            calibration_cancelled = True
        except Exception:  # noqa: BLE001 - subclass calibration is untrusted.
            calibration_failed = True
        if calibration_cancelled:
            del attempt, record, result
            _raise_cancelled()
        if calibration_failed or result is None:
            del attempt, record, result
            return _resolve_typed_failure(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=self._failure_policy,
                metadata=self._metadata,
                assessment_model=self._assessment_model,
            )
        return result

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        invalid_sensitivity = False
        normalized_sensitivity = Sensitivity.BALANCED
        try:
            normalized_sensitivity = _coerce_sensitivity(sensitivity)
        except Exception:  # noqa: BLE001 - enum-like inputs are an untrusted boundary.
            invalid_sensitivity = True
        if invalid_sensitivity:
            del conversation, sensitivity
            _raise_sensitivity_error()
        attempt = await self._aobserve_attempt(conversation, target_message_index=None)
        del conversation
        result: Assessment | None = None
        cancelled = False
        try:
            result = self._assessment_from_attempt(
                attempt,
                sensitivity=normalized_sensitivity,
                target_message_index=None,
            )
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            del attempt, result
            _raise_cancelled()
        if result is None:
            del attempt
            return _resolve_typed_failure(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=self._failure_policy,
                metadata=self._metadata,
                assessment_model=self._assessment_model,
            )
        return result

    async def aclassify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        """Async target-bound classification with full conversation context."""

        invalid_sensitivity = False
        normalized_sensitivity = Sensitivity.BALANCED
        try:
            normalized_sensitivity = _coerce_sensitivity(sensitivity)
        except Exception:  # noqa: BLE001 - subclass properties are untrusted.
            invalid_sensitivity = True
        if invalid_sensitivity:
            del conversation, sensitivity
            _raise_sensitivity_error()
        target_failure = False
        normalized_target = 0
        try:
            normalized_target = _validated_target_index(
                conversation,
                target_message_index,
                self._evidence_role,
            )
        except Exception:  # noqa: BLE001 - subclass properties are untrusted.
            target_failure = True
        if target_failure:
            del conversation, target_message_index
            _raise_target_error()
        attempt = await self._aobserve_attempt(
            conversation,
            target_message_index=normalized_target,
        )
        del conversation
        result: Assessment | None = None
        cancelled = False
        try:
            result = self._assessment_from_attempt(
                attempt,
                sensitivity=normalized_sensitivity,
                target_message_index=normalized_target,
            )
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            del attempt, result
            _raise_cancelled()
        if result is None:
            del attempt
            return _resolve_typed_failure(
                classifier_id=self._classifier_id,
                policy_version=self._policy_version,
                sensitivity=normalized_sensitivity,
                reason=IndeterminateReason.INTERNAL_ERROR,
                policy=self._failure_policy,
                metadata=self._metadata,
                assessment_model=self._assessment_model,
            )
        return result

    async def _aobserve_attempt(
        self,
        conversation: Conversation,
        *,
        target_message_index: int | None,
    ) -> tuple[
        ObservationRecord[ObservationT] | None,
        IndeterminateReason | None,
        BackendConfigurationError | None,
        bool,
    ]:
        """Async observation with the same sanitized categorical state."""

        failure_reason: IndeterminateReason | None = None
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        record: ObservationRecord[ObservationT] | None = None
        try:
            record = await self.aobserve(
                conversation,
                target_message_index=target_message_index,
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
            if configuration_failure is None:
                failure_reason = IndeterminateReason.PROVIDER_ERROR
        except BackendError as caught:
            failure_reason = _fresh_backend_error(caught).reason
        except TimeoutError:
            failure_reason = IndeterminateReason.TIMEOUT
        except Exception:  # noqa: BLE001 - subclasses are an untrusted boundary.
            failure_reason = IndeterminateReason.INTERNAL_ERROR
        if record is None and failure_reason is None and configuration_failure is None and not cancelled:
            failure_reason = IndeterminateReason.INTERNAL_ERROR
        return record, failure_reason, configuration_failure, cancelled

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

    def _validate_target_observation(
        self,
        observation: ObservationT,
        target_message_index: int,
    ) -> None:
        """Require every targeted finding to bind actionable evidence to mN."""

        target_id = _target_evidence_id(target_message_index)
        if any(target_id not in finding.message_ids for finding in observation.findings):
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
