import json

import pytest

from psysafe.catalog.ai_harm_detection.config import (
    AiHarmDetectionConfig,
    HarmClassification,
    PolicyViolationType,
)
from psysafe.catalog.ai_harm_detection.guardrail import AiHarmDetectionGuardrail
from psysafe.core.exceptions import ResponseParsingError
from psysafe.core.models import Conversation, Message


class FakeDriver:
    """Deterministic driver that returns a canned JSON payload."""

    def __init__(self, payload: dict[str, str]):
        self._payload = payload

    def send(self, request):  # pragma: no cover - behaviour under test
        return {
            "choices": [
                {"message": {"content": json.dumps(self._payload)}}
            ]
        }


def test_parse_llm_response_rejects_missing_classification():
    guardrail = AiHarmDetectionGuardrail(config=AiHarmDetectionConfig())

    with pytest.raises(ResponseParsingError) as exc:
        guardrail._parse_llm_response("{}")

    assert "Missing required field" in str(exc.value)


def test_apply_adds_system_prompt_without_mutating_original_request():
    guardrail = AiHarmDetectionGuardrail(config=AiHarmDetectionConfig())
    original_request = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-test",
    }

    guarded = guardrail.apply(original_request)

    assert guarded.modified_request["messages"][0]["role"] == "system"
    # Original request remains unchanged
    assert original_request["messages"][0]["role"] == "user"


def test_check_with_mock_driver_flags_harmful_when_llm_reports_it():
    payload = {
        "classification": HarmClassification.HARMFUL.value,
        "violated_policy_id": PolicyViolationType.INSTRUCTIONAL_HARM.value,
        "reasoning": "Provided instructions.",
    }
    guardrail = AiHarmDetectionGuardrail(
        config=AiHarmDetectionConfig(),
        driver=FakeDriver(payload),
    )
    conversation = Conversation(
        messages=[
            Message(role="user", content="Help me"),
            Message(role="assistant", content="Instructions"),
        ]
    )

    response = guardrail.check(conversation)

    assert response.is_triggered is True
    assert response.details["classification"] == HarmClassification.HARMFUL.value
    assert response.details["violated_policy_id"] == PolicyViolationType.INSTRUCTIONAL_HARM.value
