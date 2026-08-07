"""Fixed policy prompts with a separate JSON boundary for untrusted input."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import PurePosixPath
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psysafe.core.contracts import (
    MAX_CONVERSATION_CONTENT_CHARS,
    MAX_CONVERSATION_MESSAGES,
    MAX_MESSAGE_CONTENT_CHARS,
    Conversation,
    MessageRole,
)

MAX_POLICY_INSTRUCTIONS_CHARS = 100_000
CONVERSATION_FORMAT = "psysafe.conversation.v1"
RUNTIME_OBSERVATION_INSTRUCTIONS = (
    "Return every applicable observation that fits the supplied schema. Set `output_truncated` to true if a "
    "collection limit causes any applicable observation to be omitted. When the input contains "
    "`target_message_id`, use the full exchange as context but emit only findings and independent review "
    "observations whose actionable evidence cites that ID; cite it in every emitted item and return empty "
    "collections when none apply."
)


class EncodedMessage(BaseModel):
    """Portable provider-input message with an opaque positional ID."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str = Field(pattern=r"^m(?:0|[1-9][0-9]*)$")
    role: MessageRole
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CONTENT_CHARS)


class EncodedConversation(BaseModel):
    """Complete JSON input contract exported to other runtimes."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    format: Literal["psysafe.conversation.v1"] = "psysafe.conversation.v1"
    messages: tuple[EncodedMessage, ...] = Field(min_length=1, max_length=MAX_CONVERSATION_MESSAGES)
    target_message_id: str | None = Field(
        default=None,
        pattern=r"^m(?:0|[1-9][0-9]*)$",
        description="When present, report only observations actionable for this message.",
    )

    @model_validator(mode="after")
    def aggregate_and_positional_invariants(self) -> EncodedConversation:
        if sum(len(message.content) for message in self.messages) > MAX_CONVERSATION_CONTENT_CHARS:
            raise ValueError(
                f"conversation content exceeds {MAX_CONVERSATION_CONTENT_CHARS} characters",
            )
        expected_ids = tuple(f"m{index}" for index in range(len(self.messages)))
        if tuple(message.id for message in self.messages) != expected_ids:
            raise ValueError("encoded message IDs must be positional and contiguous from m0")
        if self.target_message_id is not None and self.target_message_id not in expected_ids:
            raise ValueError("target message ID must identify an encoded message")
        return self


class PromptResourceError(ValueError):
    """A policy resource could not be loaded safely."""


def _raise_encode_target_error() -> NoReturn:
    raise ValueError("target message index does not identify an encoded message") from None


class PromptSpec(BaseModel):
    """Immutable fixed instructions for one versioned classifier policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    instructions: str = Field(min_length=1, max_length=MAX_POLICY_INSTRUCTIONS_CHARS)

    @field_validator("instructions")
    @classmethod
    def instructions_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("policy instructions must not be blank")
        return normalized

    @classmethod
    def from_package(cls, *, package: str, resource: str) -> PromptSpec:
        """Load policy text through package resources, independent of the CWD."""

        path = PurePosixPath(resource)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise PromptResourceError("resource must be a relative package path")

        failure = False
        text = ""
        try:
            target = resources.files(package).joinpath(*path.parts)
            text = target.read_text(encoding="utf-8")
        except (OSError, ModuleNotFoundError, TypeError, UnicodeError):
            failure = True
        if failure:
            raise PromptResourceError(f"could not load policy resource {package}:{resource}")
        return cls(instructions=text)

    def encode(
        self,
        conversation: Conversation,
        *,
        target_message_index: int | None = None,
    ) -> str:
        """Serialize untrusted messages as data, never as policy instructions."""

        if type(conversation) is not Conversation:
            del conversation, target_message_index
            raise TypeError("conversation must be a Conversation") from None
        if target_message_index is not None and (
            type(target_message_index) is not int or not 0 <= target_message_index < len(conversation.messages)
        ):
            del conversation, target_message_index
            _raise_encode_target_error()
        payload: dict[str, object] = {
            "format": CONVERSATION_FORMAT,
            "messages": encoded_messages(conversation),
        }
        if target_message_index is not None:
            payload["target_message_id"] = f"m{target_message_index}"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def provider_instructions(self) -> str:
        """Combine the domain policy with the fixed observation protocol."""

        return f"{self.instructions}\n\n{RUNTIME_OBSERVATION_INSTRUCTIONS}"


def encoded_messages(conversation: Conversation) -> list[dict[str, str]]:
    """Return messages with opaque positional IDs for evidence citations.

    Caller IDs remain local because they can contain patient, account, or
    internal identifiers. The encoded ``m0``, ``m1`` sequence maps directly to
    message order and is stable for a given conversation.
    """

    encoded: list[dict[str, str]] = []
    for index, message in enumerate(conversation.messages):
        encoded.append(
            {
                "id": f"m{index}",
                "role": message.role.value,
                "content": message.content,
            },
        )
    return encoded


def encoded_message_ids(conversation: Conversation) -> frozenset[str]:
    """Return the exact evidence IDs visible to the model."""

    return frozenset(message["id"] for message in encoded_messages(conversation))


def conversation_input_schema() -> dict[str, object]:
    """Return the JSON Schema for the opaque, positional provider input."""

    return EncodedConversation.model_json_schema()


__all__ = [
    "CONVERSATION_FORMAT",
    "MAX_POLICY_INSTRUCTIONS_CHARS",
    "RUNTIME_OBSERVATION_INSTRUCTIONS",
    "EncodedConversation",
    "EncodedMessage",
    "PromptResourceError",
    "PromptSpec",
    "conversation_input_schema",
    "encoded_message_ids",
    "encoded_messages",
]
