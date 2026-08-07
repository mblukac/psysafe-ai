"""Fixed policy prompts with a separate JSON boundary for untrusted input."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import PurePosixPath
from typing import Literal

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

    @model_validator(mode="after")
    def aggregate_and_positional_invariants(self) -> EncodedConversation:
        if sum(len(message.content) for message in self.messages) > MAX_CONVERSATION_CONTENT_CHARS:
            raise ValueError(
                f"conversation content exceeds {MAX_CONVERSATION_CONTENT_CHARS} characters",
            )
        expected_ids = tuple(f"m{index}" for index in range(len(self.messages)))
        if tuple(message.id for message in self.messages) != expected_ids:
            raise ValueError("encoded message IDs must be positional and contiguous from m0")
        return self


class PromptResourceError(ValueError):
    """A policy resource could not be loaded safely."""


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

    def encode(self, conversation: Conversation) -> str:
        """Serialize untrusted messages as data, never as policy instructions."""

        payload = {
            "format": CONVERSATION_FORMAT,
            "messages": encoded_messages(conversation),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
    "EncodedConversation",
    "EncodedMessage",
    "PromptResourceError",
    "PromptSpec",
    "conversation_input_schema",
    "encoded_message_ids",
    "encoded_messages",
]
