"""Classifier protocols and explicit backend-failure behavior."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from psysafe.core.contracts import (
    Assessment,
    AssessmentMetadata,
    Conversation,
    IndeterminateReason,
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


class FailurePolicy(str, Enum):
    """Allowed responses to classifier backend failures.

    There is deliberately no policy that converts a failure into
    ``not_matched``.
    """

    RAISE = "raise"
    RETURN_INDETERMINATE = "return_indeterminate"


def resolve_classification_failure(
    *,
    classifier_id: str,
    policy_version: str,
    sensitivity: Sensitivity,
    reason: IndeterminateReason,
    policy: FailurePolicy,
    metadata: AssessmentMetadata | None = None,
    error: BaseException | None = None,
) -> Assessment:
    """Apply a fail-closed policy without retaining raw exception content."""

    if policy is FailurePolicy.RAISE:
        sanitized_error = ClassificationError(
            classifier_id=classifier_id,
            policy_version=policy_version,
            reason=reason,
        )
        if error is None:
            raise sanitized_error
        raise sanitized_error from error

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


__all__ = [
    "AsyncClassifier",
    "ClassificationError",
    "Classifier",
    "FailurePolicy",
    "IndeterminateAssessmentError",
    "resolve_classification_failure",
]
