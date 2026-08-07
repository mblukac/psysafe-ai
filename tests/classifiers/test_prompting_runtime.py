import json

import pytest
from pydantic import ValidationError

from psysafe.classifiers.prompting import (
    EncodedConversation,
    EncodedMessage,
    PromptResourceError,
    PromptSpec,
    encoded_message_ids,
)
from psysafe.core.contracts import Conversation, Message, MessageRole


def test_policy_resources_load_outside_the_repository(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    prompt = PromptSpec.from_package(
        package="psysafe.classifiers",
        resource="policies/self_harm.md",
    )

    assert "self-harm" in prompt.instructions


def test_resource_paths_cannot_escape_the_package() -> None:
    with pytest.raises(PromptResourceError):
        PromptSpec.from_package(package="psysafe.classifiers", resource="../README.md")
    with pytest.raises(PromptResourceError):
        PromptSpec.from_package(package="psysafe.classifiers", resource="/absolute/policy.md")


def test_encoding_uses_opaque_positional_ids_without_exposing_caller_ids() -> None:
    conversation = Conversation(
        messages=(
            Message(role=MessageRole.USER, content="first"),
            Message(id="caller-id", role=MessageRole.ASSISTANT, content="second"),
            Message(role=MessageRole.USER, content="third"),
        ),
    )

    payload = json.loads(PromptSpec(instructions="fixed").encode(conversation))

    assert [message["id"] for message in payload["messages"]] == ["m0", "m1", "m2"]
    assert encoded_message_ids(conversation) == frozenset({"m0", "m1", "m2"})
    assert "caller-id" not in json.dumps(payload)


def test_caller_ids_cannot_change_evidence_ids() -> None:
    conversation = Conversation(
        messages=(
            Message(role=MessageRole.USER, content="first"),
            Message(id="private-account-id", role=MessageRole.ASSISTANT, content="second"),
            Message(role=MessageRole.USER, content="third"),
        ),
    )

    payload = json.loads(PromptSpec(instructions="fixed").encode(conversation))

    assert [message["id"] for message in payload["messages"]] == ["m0", "m1", "m2"]
    assert "caller-id" not in json.dumps(payload)


def test_portable_input_enforces_positional_ids_and_aggregate_limit() -> None:
    with pytest.raises(ValidationError, match="positional and contiguous"):
        EncodedConversation(
            messages=(EncodedMessage(id="m7", role="user", content="test"),),
        )

    oversized = tuple(EncodedMessage(id=f"m{index}", role="user", content="x" * 100_000) for index in range(6))
    with pytest.raises(ValidationError, match="conversation content exceeds"):
        EncodedConversation(messages=oversized)
