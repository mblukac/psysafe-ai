"""Synchronous and asynchronous multi-classifier workflow gates."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, NoReturn, TypeVar, cast

from pydantic import ValidationError

from psysafe.backends.base import BackendConfigurationError, _fresh_configuration_error, _raise_cancelled
from psysafe.core.classifier import ClassificationError, _classification_error_reason
from psysafe.core.contracts import (
    MAX_ASSESSMENT_SIGNALS,
    MAX_MESSAGE_CONTENT_CHARS,
    Assessment,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    MessageRole,
    Outcome,
    Sensitivity,
)
from psysafe.gates.contracts import (
    MAX_GATE_ARTIFACT_ID_CHARS,
    MAX_GATE_CLASSIFIERS,
    MAX_GATE_REVIEW_SIGNALS,
    AsyncGateClassifier,
    Checkpoint,
    GateAction,
    GateAssessment,
    GateClassifier,
    GateDecision,
    ReviewSignalProvider,
)
from psysafe.gates.policy import GatePolicy

MAX_GATE_TEXT_CHARS = MAX_MESSAGE_CONTENT_CHARS

_ACTION_PRECEDENCE = {
    GateAction.ALLOW: 0,
    GateAction.REVIEW: 1,
    GateAction.BLOCK: 2,
}

_CHECKPOINT_ROLES = {
    Checkpoint.INPUT: MessageRole.USER,
    Checkpoint.TASK_SELECTION: MessageRole.ASSISTANT,
    Checkpoint.EXECUTION: MessageRole.ASSISTANT,
    Checkpoint.TOOL_INPUT: MessageRole.ASSISTANT,
    Checkpoint.TOOL_OUTPUT: MessageRole.TOOL,
    Checkpoint.COMMUNICATION: MessageRole.ASSISTANT,
}

_ProviderExtra = Literal["anthropic", "openai"]
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


@dataclass(frozen=True, slots=True)
class _SyncBinding:
    classifier: GateClassifier
    classifier_id: str
    policy_version: str
    evidence_role: MessageRole | None
    allowed_signals: frozenset[str]
    allowed_review_signals: frozenset[str]


@dataclass(frozen=True, slots=True)
class _AsyncBinding:
    classifier: AsyncGateClassifier
    classifier_id: str
    policy_version: str
    evidence_role: MessageRole | None
    allowed_signals: frozenset[str]
    allowed_review_signals: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ConfigurationFailure:
    """A sanitized marker kept out of public gate decisions."""

    extra: _ProviderExtra


@dataclass(frozen=True, slots=True)
class _Cancellation:
    """Cancellation marker used to clear workflow input before re-raising."""


BindingT = TypeVar("BindingT", _SyncBinding, _AsyncBinding)
ClassifierT = TypeVar("ClassifierT")


def _bounded_classifier_items(classifiers: Iterable[ClassifierT]) -> tuple[ClassifierT, ...]:
    """Materialize at most one item beyond the public classifier cap."""

    if type(classifiers) is list:
        concrete = cast(list[ClassifierT], classifiers)
        if len(concrete) > MAX_GATE_CLASSIFIERS:
            raise ValueError(f"workflow gates accept at most {MAX_GATE_CLASSIFIERS} classifiers")
        return tuple(concrete)
    if type(classifiers) is tuple:
        concrete_tuple = cast(tuple[ClassifierT, ...], classifiers)
        if len(concrete_tuple) > MAX_GATE_CLASSIFIERS:
            raise ValueError(f"workflow gates accept at most {MAX_GATE_CLASSIFIERS} classifiers")
        return concrete_tuple
    try:
        iterator = iter(classifiers)
    except Exception:  # noqa: BLE001 - custom iterables are untrusted.
        raise TypeError("classifiers must be a finite iterable") from None
    items: list[ClassifierT] = []
    for _ in range(MAX_GATE_CLASSIFIERS + 1):
        try:
            items.append(next(iterator))
        except StopIteration:
            break
        except Exception:  # noqa: BLE001 - custom iterators are untrusted.
            raise TypeError("classifiers must be a finite iterable") from None
    if len(items) > MAX_GATE_CLASSIFIERS:
        raise ValueError(f"workflow gates accept at most {MAX_GATE_CLASSIFIERS} classifiers")
    return tuple(items)


def _validated_identity(classifier_id: object, policy_version: object) -> tuple[str, str]:
    if type(classifier_id) is not str or type(policy_version) is not str:
        raise TypeError("classifiers must expose string classifier_id and policy_version properties")
    try:
        assessment = Assessment(
            classifier_id=classifier_id,
            policy_version=policy_version,
            outcome=Outcome.NOT_MATCHED,
        )
    except ValidationError as error:
        raise ValueError("classifier identity does not satisfy the public assessment contract") from error
    return assessment.classifier_id, assessment.policy_version


def _validated_evidence_role(value: object) -> MessageRole | None:
    if value is None:
        return None
    if type(value) is MessageRole:
        return value
    if type(value) is not str:
        raise ValueError("classifier evidence_role must be a MessageRole or None")
    try:
        return MessageRole(value)
    except (TypeError, ValueError) as error:
        raise ValueError("classifier evidence_role must be a MessageRole or None") from error


def _validated_allowed_signals(value: object) -> frozenset[str]:
    if type(value) is not tuple or not value:
        raise ValueError("classifier allowed_signals must be a non-empty tuple")
    if len(value) > MAX_ASSESSMENT_SIGNALS:
        raise ValueError("classifier allowed_signals exceed the bounded vocabulary limit")
    if any(type(signal) is not str for signal in value):
        raise ValueError("classifier allowed_signals must contain exact string labels")
    try:
        assessment = Assessment(
            classifier_id="gate_vocabulary",
            policy_version="1",
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=value,
        )
    except ValidationError as error:
        raise ValueError("classifier allowed_signals must be a bounded categorical vocabulary") from error
    return frozenset(assessment.signals)


def _validated_allowed_review_signals(value: object) -> frozenset[str]:
    if type(value) is not tuple:
        raise ValueError("classifier allowed_review_signals must be a tuple")
    if len(value) > MAX_GATE_REVIEW_SIGNALS:
        raise ValueError("classifier allowed_review_signals exceed the bounded vocabulary limit")
    if any(type(signal) is not str for signal in value):
        raise ValueError("classifier allowed_review_signals must contain exact string labels")
    try:
        assessment = GateAssessment(
            classifier_id="gate_vocabulary",
            policy_version="1",
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=("policy_match",),
            review_signals=value,
        )
    except ValidationError as error:
        raise ValueError("classifier allowed_review_signals must be a bounded categorical vocabulary") from error
    if len(set(assessment.review_signals)) != len(assessment.review_signals):
        raise ValueError("classifier allowed_review_signals must be unique")
    return frozenset(assessment.review_signals)


def _sync_bindings(classifiers: Iterable[GateClassifier]) -> tuple[_SyncBinding, ...]:
    bindings: list[_SyncBinding] = []
    for classifier in _bounded_classifier_items(classifiers):
        if not isinstance(classifier, GateClassifier):
            raise TypeError("every synchronous gate classifier must satisfy GateClassifier")
        classifier_id, policy_version = _validated_identity(classifier.classifier_id, classifier.policy_version)
        bindings.append(
            _SyncBinding(
                classifier=classifier,
                classifier_id=classifier_id,
                policy_version=policy_version,
                evidence_role=_validated_evidence_role(classifier.evidence_role),
                allowed_signals=_validated_allowed_signals(classifier.allowed_signals),
                allowed_review_signals=_validated_allowed_review_signals(
                    classifier.allowed_review_signals,
                ),
            ),
        )
    return _validated_bindings(tuple(bindings))


def _async_bindings(classifiers: Iterable[AsyncGateClassifier]) -> tuple[_AsyncBinding, ...]:
    bindings: list[_AsyncBinding] = []
    for classifier in _bounded_classifier_items(classifiers):
        if not isinstance(classifier, AsyncGateClassifier):
            raise TypeError("every asynchronous gate classifier must satisfy AsyncGateClassifier")
        classifier_id, policy_version = _validated_identity(classifier.classifier_id, classifier.policy_version)
        bindings.append(
            _AsyncBinding(
                classifier=classifier,
                classifier_id=classifier_id,
                policy_version=policy_version,
                evidence_role=_validated_evidence_role(classifier.evidence_role),
                allowed_signals=_validated_allowed_signals(classifier.allowed_signals),
                allowed_review_signals=_validated_allowed_review_signals(
                    classifier.allowed_review_signals,
                ),
            ),
        )
    return _validated_bindings(tuple(bindings))


def _validated_bindings(bindings: tuple[BindingT, ...]) -> tuple[BindingT, ...]:
    if not bindings:
        raise ValueError("workflow gates require at least one classifier")
    if len(bindings) > MAX_GATE_CLASSIFIERS:
        raise ValueError(f"workflow gates accept at most {MAX_GATE_CLASSIFIERS} classifiers")
    classifier_ids = tuple(binding.classifier_id for binding in bindings)
    if len(set(classifier_ids)) != len(classifier_ids):
        raise ValueError("workflow gate classifier IDs must be unique")
    return bindings


def _indeterminate_assessment(
    *,
    classifier_id: str,
    policy_version: str,
    sensitivity: Sensitivity,
    reason: IndeterminateReason,
) -> GateAssessment:
    return GateAssessment(
        classifier_id=classifier_id,
        policy_version=policy_version,
        sensitivity=sensitivity,
        outcome=Outcome.INDETERMINATE,
        indeterminate_reason=reason,
    )


def _review_signal_label(value: object) -> str:
    if type(value) is str:
        return value
    raise ValueError("review signals must be categorical string labels")


def _assessment_review_signals(
    value: Assessment,
    allowed_review_signals: frozenset[str],
) -> tuple[str, ...]:
    if not isinstance(value, ReviewSignalProvider):
        return ()
    raw_signals = value.review_signals
    if type(raw_signals) is not tuple:
        raise ValueError("review signals must be returned as a bounded tuple")
    if len(raw_signals) > MAX_GATE_REVIEW_SIGNALS:
        raise ValueError("review signals exceed the bounded result limit")
    signals = tuple(_review_signal_label(signal) for signal in raw_signals)
    if not set(signals) <= allowed_review_signals:
        raise ValueError("review signals must occur in the declared vocabulary")
    return signals


def _sanitized_assessment(
    value: object,
    *,
    classifier_id: str,
    policy_version: str,
    sensitivity: Sensitivity,
    allowed_signals: frozenset[str],
    allowed_review_signals: frozenset[str],
) -> GateAssessment:
    if not isinstance(value, Assessment):
        return _indeterminate_assessment(
            classifier_id=classifier_id,
            policy_version=policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INVALID_RESPONSE,
        )
    try:
        if (
            value.classifier_id != classifier_id
            or value.policy_version != policy_version
            or value.sensitivity is not sensitivity
        ):
            raise ValueError("classifier result identity does not match its binding")
        raw_signals = value.signals
        if type(raw_signals) is not tuple:
            raise ValueError("classifier signals must be returned as a bounded tuple")
        if len(raw_signals) > MAX_ASSESSMENT_SIGNALS:
            raise ValueError("classifier signals exceed the bounded result limit")
        if any(type(signal) is not str for signal in raw_signals):
            raise ValueError("classifier signals must contain exact string labels")
        signals = raw_signals
        if not set(signals) <= allowed_signals:
            raise ValueError("classifier signals must occur in the configured vocabulary")
        review_signals = _assessment_review_signals(value, allowed_review_signals)
        return GateAssessment(
            classifier_id=value.classifier_id,
            policy_version=value.policy_version,
            sensitivity=value.sensitivity,
            outcome=value.outcome,
            evidence_directness=value.evidence_directness,
            signals=signals,
            indeterminate_reason=value.indeterminate_reason,
            review_signals=review_signals,
        )
    except Exception:  # noqa: BLE001 - classifier result subclasses are an untrusted boundary.
        return _indeterminate_assessment(
            classifier_id=classifier_id,
            policy_version=policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INVALID_RESPONSE,
        )


def _configuration_failure(error: BackendConfigurationError) -> _ConfigurationFailure | None:
    fresh = _fresh_configuration_error(error)
    if fresh is None:
        return None
    state = object.__getattribute__(fresh, "__dict__")
    extra = state.get("extra") if isinstance(state, dict) else None
    if extra not in {"anthropic", "openai"}:
        return None
    return _ConfigurationFailure(cast("_ProviderExtra", extra))


def _sync_assessment(
    binding: _SyncBinding,
    conversation: Conversation,
    target_message_index: int,
    sensitivity: Sensitivity,
) -> GateAssessment | _ConfigurationFailure | _Cancellation:
    try:
        value = binding.classifier.classify_target(
            conversation,
            target_message_index=target_message_index,
            sensitivity=sensitivity,
        )
    except asyncio.CancelledError:
        return _Cancellation()
    except BackendConfigurationError as error:
        failure = _configuration_failure(error)
        if failure is not None:
            return failure
        return _indeterminate_assessment(
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INTERNAL_ERROR,
        )
    except ClassificationError as error:
        return _indeterminate_assessment(
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            reason=_classification_error_reason(error),
        )
    except Exception:  # noqa: BLE001 - classifier implementations are an untrusted boundary.
        return _indeterminate_assessment(
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INTERNAL_ERROR,
        )
    try:
        return _sanitized_assessment(
            value,
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            allowed_signals=binding.allowed_signals,
            allowed_review_signals=binding.allowed_review_signals,
        )
    except asyncio.CancelledError:
        return _Cancellation()


async def _async_assessment(
    binding: _AsyncBinding,
    conversation: Conversation,
    target_message_index: int,
    sensitivity: Sensitivity,
) -> GateAssessment | _ConfigurationFailure | _Cancellation:
    try:
        value = await binding.classifier.aclassify_target(
            conversation,
            target_message_index=target_message_index,
            sensitivity=sensitivity,
        )
    except asyncio.CancelledError:
        return _Cancellation()
    except BackendConfigurationError as error:
        failure = _configuration_failure(error)
        if failure is not None:
            return failure
        return _indeterminate_assessment(
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INTERNAL_ERROR,
        )
    except ClassificationError as error:
        return _indeterminate_assessment(
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            reason=_classification_error_reason(error),
        )
    except Exception:  # noqa: BLE001 - classifier implementations are an untrusted boundary.
        return _indeterminate_assessment(
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INTERNAL_ERROR,
        )
    try:
        return _sanitized_assessment(
            value,
            classifier_id=binding.classifier_id,
            policy_version=binding.policy_version,
            sensitivity=sensitivity,
            allowed_signals=binding.allowed_signals,
            allowed_review_signals=binding.allowed_review_signals,
        )
    except asyncio.CancelledError:
        return _Cancellation()


def _decision(
    *,
    checkpoint: Checkpoint,
    artifact_id: str,
    target_message_index: int | None,
    policy: GatePolicy,
    assessments: tuple[GateAssessment, ...],
) -> GateDecision:
    actions: list[GateAction] = []
    for assessment in assessments:
        action = policy.action_for(assessment.outcome, classifier_id=assessment.classifier_id)
        if assessment.review_signals:
            action = max(
                action,
                policy.review_action_for(assessment.classifier_id),
                key=_ACTION_PRECEDENCE.__getitem__,
            )
        actions.append(action)
    return GateDecision(
        checkpoint=checkpoint,
        artifact_id=artifact_id,
        target_message_index=target_message_index,
        action=max(actions, key=_ACTION_PRECEDENCE.__getitem__),
        assessments=assessments,
    )


def _input_failure_decision(
    *,
    checkpoint: Checkpoint,
    artifact_id: str,
    target_message_index: int | None,
    policy: GatePolicy,
    classifier_ids_and_versions: tuple[tuple[str, str], ...],
    sensitivity: Sensitivity,
) -> GateDecision:
    assessments = tuple(
        _indeterminate_assessment(
            classifier_id=classifier_id,
            policy_version=policy_version,
            sensitivity=sensitivity,
            reason=IndeterminateReason.INSUFFICIENT_INPUT,
        )
        for classifier_id, policy_version in classifier_ids_and_versions
    )
    return _decision(
        checkpoint=checkpoint,
        artifact_id=artifact_id,
        target_message_index=target_message_index,
        policy=policy,
        assessments=assessments,
    )


def _validated_target_index(
    checkpoint: Checkpoint,
    conversation: object,
    *,
    target_message_index: int | None,
) -> int | None:
    """Resolve and role-check one opaque position in the full conversation."""

    if type(conversation) is not Conversation:
        return None
    if target_message_index is None:
        index = len(conversation.messages) - 1
    elif type(target_message_index) is not int:
        return None
    else:
        index = target_message_index
    if index < 0 or index >= len(conversation.messages):
        return None
    target = conversation.messages[index]
    expected_role = _CHECKPOINT_ROLES[checkpoint]
    if target.role is not expected_role:
        return None
    return index


def _text_target(checkpoint: Checkpoint, text: object) -> Conversation | None:
    if type(text) is not str:
        return None
    try:
        return Conversation.from_text(text, role=_CHECKPOINT_ROLES[checkpoint])
    except (TypeError, ValueError, ValidationError):
        return None


def _sync_gate_result(
    *,
    bindings: tuple[_SyncBinding, ...],
    checkpoint: Checkpoint,
    artifact_id: str,
    conversation: Conversation,
    target_message_index: int,
    sensitivity: Sensitivity,
    policy: GatePolicy,
) -> GateDecision | _ConfigurationFailure | _Cancellation:
    assessments: list[GateAssessment] = []
    for binding in bindings:
        value = _sync_assessment(binding, conversation, target_message_index, sensitivity)
        if isinstance(value, _Cancellation):
            return value
        if isinstance(value, _ConfigurationFailure):
            return value
        assessments.append(value)
    return _decision(
        checkpoint=checkpoint,
        artifact_id=artifact_id,
        target_message_index=target_message_index,
        policy=policy,
        assessments=tuple(assessments),
    )


async def _async_gate_result(
    *,
    bindings: tuple[_AsyncBinding, ...],
    checkpoint: Checkpoint,
    artifact_id: str,
    conversation: Conversation,
    target_message_index: int,
    sensitivity: Sensitivity,
    policy: GatePolicy,
) -> GateDecision | _ConfigurationFailure | _Cancellation:
    try:
        values = await asyncio.gather(
            *(_async_assessment(binding, conversation, target_message_index, sensitivity) for binding in bindings),
        )
    except asyncio.CancelledError:
        return _Cancellation()
    assessments: list[GateAssessment] = []
    for value in values:
        if isinstance(value, _Cancellation):
            return value
        if isinstance(value, _ConfigurationFailure):
            return value
        assessments.append(value)
    return _decision(
        checkpoint=checkpoint,
        artifact_id=artifact_id,
        target_message_index=target_message_index,
        policy=policy,
        assessments=tuple(assessments),
    )


def _raise_configuration_failure(extra: _ProviderExtra) -> NoReturn:
    raise BackendConfigurationError(extra) from None


def _normalized_checkpoint(value: object) -> Checkpoint | None:
    if type(value) is Checkpoint:
        return value
    if type(value) is not str:
        return None
    try:
        return Checkpoint(value)
    except (TypeError, ValueError):
        return None


def _normalized_sensitivity(value: object) -> Sensitivity | None:
    if type(value) is Sensitivity:
        return value
    if type(value) is not str:
        return None
    try:
        return Sensitivity(value)
    except Exception:  # noqa: BLE001 - enum-like caller values are untrusted.
        return None


def _normalized_artifact_id(value: object) -> str | None:
    if type(value) is not str:
        return None
    if not 1 <= len(value) <= MAX_GATE_ARTIFACT_ID_CHARS:
        return None
    if _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        return None
    return value


def _raise_invalid_checkpoint() -> NoReturn:
    raise ValueError("invalid workflow gate checkpoint") from None


def _raise_invalid_sensitivity() -> NoReturn:
    raise ValueError("invalid workflow gate sensitivity") from None


def _raise_invalid_artifact_id() -> NoReturn:
    raise ValueError("invalid workflow gate artifact_id") from None


def _validate_policy_bindings(policy: GatePolicy, classifier_ids: tuple[str, ...]) -> None:
    unknown_classifier_ids = set(policy.override_classifier_ids) - set(classifier_ids)
    if unknown_classifier_ids:
        raise ValueError("gate policy overrides must refer to configured classifier IDs")


def _validate_checkpoint_bindings(
    checkpoint: Checkpoint,
    bindings: tuple[_SyncBinding, ...] | tuple[_AsyncBinding, ...],
) -> None:
    expected_role = _CHECKPOINT_ROLES[checkpoint]
    if any(binding.evidence_role not in {None, expected_role} for binding in bindings):
        raise ValueError("classifier evidence_role is incompatible with the gate checkpoint")


class WorkflowGate:
    """Evaluate targeted classifiers at one role-bound workflow checkpoint.

    The full conversation provides context, while the target-aware classifier
    binds actionable evidence to one validated message position. The gate
    retains only its checkpoint, immutable policy, and classifier bindings.
    """

    __slots__ = ("_bindings", "_checkpoint", "_policy")

    def __init__(
        self,
        classifiers: Iterable[GateClassifier],
        *,
        checkpoint: Checkpoint,
        policy: GatePolicy,
    ) -> None:
        if not isinstance(policy, GatePolicy):
            raise TypeError("policy must be a GatePolicy")
        normalized_checkpoint = _normalized_checkpoint(checkpoint)
        if normalized_checkpoint is None:
            _raise_invalid_checkpoint()
        self._bindings = _sync_bindings(classifiers)
        _validate_policy_bindings(policy, tuple(binding.classifier_id for binding in self._bindings))
        _validate_checkpoint_bindings(normalized_checkpoint, self._bindings)
        self._checkpoint = normalized_checkpoint
        self._policy = policy

    @property
    def checkpoint(self) -> Checkpoint:
        """The workflow boundary this gate is configured to protect."""

        return self._checkpoint

    @property
    def policy(self) -> GatePolicy:
        """Return the immutable action policy."""

        return self._policy

    @property
    def classifier_ids(self) -> tuple[str, ...]:
        """Classifier IDs in deterministic evaluation order."""

        return tuple(binding.classifier_id for binding in self._bindings)

    def evaluate(
        self,
        conversation: Conversation,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
        target_message_index: int | None = None,
    ) -> GateDecision:
        """Evaluate an immutable artifact target and consume the decision immediately."""

        normalized_artifact_id = _normalized_artifact_id(artifact_id)
        if normalized_artifact_id is None:
            del conversation, artifact_id, sensitivity, target_message_index
            _raise_invalid_artifact_id()
        normalized_sensitivity = _normalized_sensitivity(sensitivity)
        if normalized_sensitivity is None:
            del conversation, sensitivity, target_message_index
            _raise_invalid_sensitivity()
        target_index = _validated_target_index(
            self._checkpoint,
            conversation,
            target_message_index=target_message_index,
        )
        if target_index is None:
            return _input_failure_decision(
                checkpoint=self._checkpoint,
                artifact_id=normalized_artifact_id,
                target_message_index=None,
                policy=self._policy,
                classifier_ids_and_versions=tuple(
                    (binding.classifier_id, binding.policy_version) for binding in self._bindings
                ),
                sensitivity=normalized_sensitivity,
            )
        result = _sync_gate_result(
            bindings=self._bindings,
            checkpoint=self._checkpoint,
            artifact_id=normalized_artifact_id,
            conversation=conversation,
            target_message_index=target_index,
            sensitivity=normalized_sensitivity,
            policy=self._policy,
        )
        if isinstance(result, _Cancellation):
            del conversation, target_index, result
            _raise_cancelled()
        if isinstance(result, _ConfigurationFailure):
            extra = result.extra
            del conversation, target_index, result
            _raise_configuration_failure(extra)
        return result

    def evaluate_text(
        self,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> GateDecision:
        """Evaluate one immutable text artifact and consume the decision immediately."""

        normalized_artifact_id = _normalized_artifact_id(artifact_id)
        if normalized_artifact_id is None:
            del text, artifact_id, sensitivity
            _raise_invalid_artifact_id()
        normalized_sensitivity = _normalized_sensitivity(sensitivity)
        if normalized_sensitivity is None:
            del text, sensitivity
            _raise_invalid_sensitivity()
        target = _text_target(self._checkpoint, text)
        if target is None:
            return _input_failure_decision(
                checkpoint=self._checkpoint,
                artifact_id=normalized_artifact_id,
                target_message_index=None,
                policy=self._policy,
                classifier_ids_and_versions=tuple(
                    (binding.classifier_id, binding.policy_version) for binding in self._bindings
                ),
                sensitivity=normalized_sensitivity,
            )
        result = _sync_gate_result(
            bindings=self._bindings,
            checkpoint=self._checkpoint,
            artifact_id=normalized_artifact_id,
            conversation=target,
            target_message_index=0,
            sensitivity=normalized_sensitivity,
            policy=self._policy,
        )
        if isinstance(result, _Cancellation):
            del text, target, result
            _raise_cancelled()
        if isinstance(result, _ConfigurationFailure):
            extra = result.extra
            del text, target, result
            _raise_configuration_failure(extra)
        return result


class AsyncWorkflowGate:
    """Concurrently evaluate async classifiers at role-bound checkpoints.

    Results remain ordered by classifier configuration, independent of task
    completion order. The gate retains no input, assessment, or decision data.
    """

    __slots__ = ("_bindings", "_checkpoint", "_policy")

    def __init__(
        self,
        classifiers: Iterable[AsyncGateClassifier],
        *,
        checkpoint: Checkpoint,
        policy: GatePolicy,
    ) -> None:
        if not isinstance(policy, GatePolicy):
            raise TypeError("policy must be a GatePolicy")
        normalized_checkpoint = _normalized_checkpoint(checkpoint)
        if normalized_checkpoint is None:
            _raise_invalid_checkpoint()
        self._bindings = _async_bindings(classifiers)
        _validate_policy_bindings(policy, tuple(binding.classifier_id for binding in self._bindings))
        _validate_checkpoint_bindings(normalized_checkpoint, self._bindings)
        self._checkpoint = normalized_checkpoint
        self._policy = policy

    @property
    def checkpoint(self) -> Checkpoint:
        """The workflow boundary this gate is configured to protect."""

        return self._checkpoint

    @property
    def policy(self) -> GatePolicy:
        """Return the immutable action policy."""

        return self._policy

    @property
    def classifier_ids(self) -> tuple[str, ...]:
        """Classifier IDs in deterministic result order."""

        return tuple(binding.classifier_id for binding in self._bindings)

    async def aevaluate(
        self,
        conversation: Conversation,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
        target_message_index: int | None = None,
    ) -> GateDecision:
        """Evaluate an immutable artifact target concurrently, then consume immediately."""

        normalized_artifact_id = _normalized_artifact_id(artifact_id)
        if normalized_artifact_id is None:
            del conversation, artifact_id, sensitivity, target_message_index
            _raise_invalid_artifact_id()
        normalized_sensitivity = _normalized_sensitivity(sensitivity)
        if normalized_sensitivity is None:
            del conversation, sensitivity, target_message_index
            _raise_invalid_sensitivity()
        target_index = _validated_target_index(
            self._checkpoint,
            conversation,
            target_message_index=target_message_index,
        )
        if target_index is None:
            return _input_failure_decision(
                checkpoint=self._checkpoint,
                artifact_id=normalized_artifact_id,
                target_message_index=None,
                policy=self._policy,
                classifier_ids_and_versions=tuple(
                    (binding.classifier_id, binding.policy_version) for binding in self._bindings
                ),
                sensitivity=normalized_sensitivity,
            )
        result = await _async_gate_result(
            bindings=self._bindings,
            checkpoint=self._checkpoint,
            artifact_id=normalized_artifact_id,
            conversation=conversation,
            target_message_index=target_index,
            sensitivity=normalized_sensitivity,
            policy=self._policy,
        )
        if isinstance(result, _Cancellation):
            del conversation, target_index, result
            _raise_cancelled()
        if isinstance(result, _ConfigurationFailure):
            extra = result.extra
            del conversation, target_index, result
            _raise_configuration_failure(extra)
        return result

    async def aevaluate_text(
        self,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> GateDecision:
        """Evaluate one immutable text artifact and consume the decision immediately."""

        normalized_artifact_id = _normalized_artifact_id(artifact_id)
        if normalized_artifact_id is None:
            del text, artifact_id, sensitivity
            _raise_invalid_artifact_id()
        normalized_sensitivity = _normalized_sensitivity(sensitivity)
        if normalized_sensitivity is None:
            del text, sensitivity
            _raise_invalid_sensitivity()
        target = _text_target(self._checkpoint, text)
        if target is None:
            return _input_failure_decision(
                checkpoint=self._checkpoint,
                artifact_id=normalized_artifact_id,
                target_message_index=None,
                policy=self._policy,
                classifier_ids_and_versions=tuple(
                    (binding.classifier_id, binding.policy_version) for binding in self._bindings
                ),
                sensitivity=normalized_sensitivity,
            )
        result = await _async_gate_result(
            bindings=self._bindings,
            checkpoint=self._checkpoint,
            artifact_id=normalized_artifact_id,
            conversation=target,
            target_message_index=0,
            sensitivity=normalized_sensitivity,
            policy=self._policy,
        )
        if isinstance(result, _Cancellation):
            del text, target, result
            _raise_cancelled()
        if isinstance(result, _ConfigurationFailure):
            extra = result.extra
            del text, target, result
            _raise_configuration_failure(extra)
        return result


__all__ = [
    "MAX_GATE_TEXT_CHARS",
    "AsyncWorkflowGate",
    "WorkflowGate",
]
