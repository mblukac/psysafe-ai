"""Framework-neutral contracts for PsySafe classifiers.

These models deliberately keep classification evidence categorical.  They do
not expose confidence scores, risk scores, or raw provider responses.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_MESSAGE_CONTENT_CHARS = 100_000
MAX_CONVERSATION_MESSAGES = 128
MAX_CONVERSATION_CONTENT_CHARS = 500_000


class MessageRole(str, Enum):
    """Roles shared by common chat and agent runtimes."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A validated text message at a classifier boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    role: MessageRole
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CONTENT_CHARS)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class Conversation(BaseModel):
    """An immutable, size-bounded sequence of messages."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    messages: tuple[Message, ...] = Field(
        min_length=1,
        max_length=MAX_CONVERSATION_MESSAGES,
    )

    @model_validator(mode="after")
    def content_must_fit_total_limit(self) -> Conversation:
        total_chars = sum(len(message.content) for message in self.messages)
        if total_chars > MAX_CONVERSATION_CONTENT_CHARS:
            raise ValueError(f"conversation content exceeds {MAX_CONVERSATION_CONTENT_CHARS} characters")
        message_ids = [message.id for message in self.messages if message.id is not None]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("conversation message IDs must be unique")
        return self

    @classmethod
    def from_text(
        cls,
        content: str,
        *,
        role: MessageRole = MessageRole.USER,
        message_id: str | None = None,
    ) -> Conversation:
        """Create a one-message conversation without weakening validation."""

        return cls(messages=(Message(id=message_id, role=role, content=content),))


class Sensitivity(str, Enum):
    """Named classifier boundaries, from narrowest to broadest.

    The legacy labels ``low``, ``medium``, and ``high`` are accepted when
    parsing input and normalize to the named values.  Serialized output always
    uses the named value.
    """

    PRECISE = "precise"
    BALANCED = "balanced"
    PRECAUTIONARY = "precautionary"

    @classmethod
    def _missing_(cls, value: object) -> Sensitivity | None:
        aliases = {
            "low": cls.PRECISE,
            "medium": cls.BALANCED,
            "high": cls.PRECAUTIONARY,
        }
        if isinstance(value, str):
            return aliases.get(value.strip().lower())
        return None


class Outcome(str, Enum):
    """Whether the classifier policy matched the supplied conversation."""

    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    INDETERMINATE = "indeterminate"


class EvidenceDirectness(str, Enum):
    """How directly the input supports the classifier's category."""

    NONE = "none"
    AMBIGUOUS = "ambiguous"
    CONTEXTUAL = "contextual"
    EXPLICIT = "explicit"


class IndeterminateReason(str, Enum):
    """Sanitized reasons for an assessment without a policy decision."""

    INSUFFICIENT_INPUT = "insufficient_input"
    REFUSED = "refused"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    INTERNAL_ERROR = "internal_error"


class AssessmentMetadata(BaseModel):
    """Non-sensitive execution provenance retained with an assessment.

    Arbitrary dictionaries are intentionally not accepted here: credentials,
    request payloads, and raw responses must not become result metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: str | None = Field(default=None, min_length=1, max_length=80)
    model: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("provider", "model")
    @classmethod
    def metadata_value_must_be_single_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("metadata values must not be blank")
        if any(character in normalized for character in ("\n", "\r", "\x00")):
            raise ValueError("metadata values must be single-line identifiers")
        return normalized


class Assessment(BaseModel):
    """A categorical classifier result safe to pass between workflow stages."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    classifier_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$")
    policy_version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
    )
    sensitivity: Sensitivity = Sensitivity.BALANCED
    outcome: Outcome
    evidence_directness: EvidenceDirectness = EvidenceDirectness.NONE
    signals: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    indeterminate_reason: IndeterminateReason | None = None
    metadata: AssessmentMetadata = Field(default_factory=AssessmentMetadata)

    @field_validator("signals")
    @classmethod
    def signals_must_be_bounded_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or len(value) > 100:
                raise ValueError("signal labels must contain between 1 and 100 characters")
            if not value[0].isalpha() or any(
                not (character.islower() or character.isdigit() or character in "_.-") for character in value
            ):
                raise ValueError("signal labels must be lowercase identifiers")
        if len(set(values)) != len(values):
            raise ValueError("signal labels must be unique")
        return values

    @model_validator(mode="after")
    def outcome_must_match_asserted_evidence(self) -> Assessment:
        is_indeterminate = self.outcome is Outcome.INDETERMINATE
        if is_indeterminate and self.indeterminate_reason is None:
            raise ValueError("indeterminate assessments require an indeterminate_reason")
        if not is_indeterminate and self.indeterminate_reason is not None:
            raise ValueError("only indeterminate assessments may include an indeterminate_reason")
        if is_indeterminate and (self.evidence_directness is not EvidenceDirectness.NONE or self.signals):
            raise ValueError("indeterminate assessments cannot assert evidence or signals")
        if self.outcome is Outcome.MATCHED and (
            self.evidence_directness is EvidenceDirectness.NONE or not self.signals
        ):
            raise ValueError("matched assessments require directness and at least one signal")
        if self.outcome is Outcome.NOT_MATCHED and (
            self.evidence_directness is not EvidenceDirectness.NONE or self.signals
        ):
            raise ValueError("not_matched assessments cannot assert evidence or signals")
        return self

    @classmethod
    def indeterminate(
        cls,
        *,
        classifier_id: str,
        policy_version: str,
        sensitivity: Sensitivity,
        reason: IndeterminateReason,
        metadata: AssessmentMetadata | None = None,
    ) -> Assessment:
        """Build an explicit non-decision for a failed or incomplete check."""

        return cls(
            classifier_id=classifier_id,
            policy_version=policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.INDETERMINATE,
            indeterminate_reason=reason,
            metadata=metadata or AssessmentMetadata(),
        )

    def require_match_decision(self) -> bool:
        """Return the decision, raising instead of treating indeterminate as false."""

        if self.outcome is Outcome.INDETERMINATE:
            from psysafe.core.classifier import IndeterminateAssessmentError

            raise IndeterminateAssessmentError(
                classifier_id=self.classifier_id,
                policy_version=self.policy_version,
                reason=self.indeterminate_reason or IndeterminateReason.INTERNAL_ERROR,
            )
        return self.outcome is Outcome.MATCHED


__all__ = [
    "MAX_CONVERSATION_CONTENT_CHARS",
    "MAX_CONVERSATION_MESSAGES",
    "MAX_MESSAGE_CONTENT_CHARS",
    "Assessment",
    "AssessmentMetadata",
    "Conversation",
    "EvidenceDirectness",
    "IndeterminateReason",
    "Message",
    "MessageRole",
    "Outcome",
    "Sensitivity",
]
