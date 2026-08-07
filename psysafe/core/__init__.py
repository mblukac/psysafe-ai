"""Public contracts for building calibrated PsySafe classifiers."""

from psysafe.core.classifier import (
    AsyncClassifier,
    ClassificationError,
    Classifier,
    FailurePolicy,
    IndeterminateAssessmentError,
    resolve_classification_failure,
)
from psysafe.core.contracts import (
    MAX_CONVERSATION_CONTENT_CHARS,
    MAX_CONVERSATION_MESSAGES,
    MAX_MESSAGE_CONTENT_CHARS,
    Assessment,
    AssessmentMetadata,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    Message,
    MessageRole,
    Outcome,
    Sensitivity,
)

__all__ = [
    "MAX_CONVERSATION_CONTENT_CHARS",
    "MAX_CONVERSATION_MESSAGES",
    "MAX_MESSAGE_CONTENT_CHARS",
    "Assessment",
    "AssessmentMetadata",
    "AsyncClassifier",
    "ClassificationError",
    "Classifier",
    "Conversation",
    "EvidenceDirectness",
    "FailurePolicy",
    "IndeterminateAssessmentError",
    "IndeterminateReason",
    "Message",
    "MessageRole",
    "Outcome",
    "Sensitivity",
    "resolve_classification_failure",
]
