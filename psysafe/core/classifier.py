"""Classifier protocols and explicit backend-failure behavior."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from psysafe.core.contracts import (
    Assessment,
    AssessmentMetadata,
    Conversation,
    IndeterminateReason,
    MessageRole,
    Sensitivity,
)


class ClassificationError(RuntimeError):
    """A sanitized classifier failure suitable for application error handling."""

    def __init__(
        self,
        *,
        classifier_id: str,
        policy_version: str,
        reason: IndeterminateReason,
    ) -> None:
        super().__init__(f"classifier {classifier_id!r} could not produce an assessment ({reason.value})")
        self.classifier_id = classifier_id
        self.policy_version = policy_version
        self.reason = reason


class IndeterminateAssessmentError(ClassificationError):
    """Raised when a caller requests a boolean from an indeterminate result."""


def _classification_error_reason(error: ClassificationError) -> IndeterminateReason:
    """Read categorical state only from an exact library-created error."""

    if type(error) is not ClassificationError:
        return IndeterminateReason.INTERNAL_ERROR
    state = object.__getattribute__(error, "__dict__")
    reason = state.get("reason") if isinstance(state, dict) else None
    return reason if isinstance(reason, IndeterminateReason) else IndeterminateReason.INTERNAL_ERROR


class FailurePolicy(str, Enum):
    """Allowed responses to classifier backend failures.

    There is deliberately no policy that converts a failure into
    ``not_matched``.
    """

    RAISE = "raise"
    RETURN_INDETERMINATE = "return_indeterminate"


def _resolve_classification_failure(
    *,
    classifier_id: str,
    policy_version: str,
    sensitivity: Sensitivity,
    reason: IndeterminateReason,
    policy: FailurePolicy,
    metadata: AssessmentMetadata | None = None,
) -> Assessment:
    """Apply a fail-closed policy after the active provider exception is gone."""

    if policy is FailurePolicy.RAISE:
        sanitized_error = ClassificationError(
            classifier_id=classifier_id,
            policy_version=policy_version,
            reason=reason,
        )
        raise sanitized_error from None

    return Assessment.indeterminate(
        classifier_id=classifier_id,
        policy_version=policy_version,
        sensitivity=sensitivity,
        reason=reason,
        metadata=metadata,
    )


@runtime_checkable
class Classifier(Protocol):
    """Synchronous framework-neutral classifier contract."""

    @property
    def classifier_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment: ...


@runtime_checkable
class AsyncClassifier(Protocol):
    """Asynchronous framework-neutral classifier contract."""

    @property
    def classifier_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment: ...


@runtime_checkable
class TargetedClassifier(Classifier, Protocol):
    """Classifier that can bind a decision to one message while using context."""

    @property
    def evidence_role(self) -> MessageRole | None: ...

    def classify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment: ...


@runtime_checkable
class AsyncTargetedClassifier(AsyncClassifier, Protocol):
    """Async classifier that binds a decision to one contextualized message."""

    @property
    def evidence_role(self) -> MessageRole | None: ...

    async def aclassify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment: ...


__all__ = [
    "AsyncClassifier",
    "AsyncTargetedClassifier",
    "ClassificationError",
    "Classifier",
    "FailurePolicy",
    "IndeterminateAssessmentError",
    "TargetedClassifier",
]
