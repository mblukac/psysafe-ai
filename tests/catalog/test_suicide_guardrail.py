from typing import Optional
# tests/catalog/test_suicide_guardrail.py
import pytest
from unittest.mock import patch, MagicMock

from psysafe.catalog.suicide_prevention.guardrail import (
    SuicidePreventionGuardrail,
    Sensitivity
)
from psysafe.core.template import PromptTemplate
from psysafe.core.models import Conversation, Message, CheckOutput # Added CheckOutput
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from utils.llm_utils import LLMResponseParseError # Added LLMResponseParseError
import json # Added json

@pytest.mark.parametrize("sensitivity", list(Sensitivity))
@pytest.mark.parametrize("reasoning", [True, False])
@pytest.mark.parametrize("confidence", [True, False])
def test_suicide_prevention_guardrail_initialization(
    sensitivity, reasoning, confidence
):
    """
    Tests initialization of SuicidePreventionGuardrail with Sensitivity enum
    and boolean flags for reasoning and confidence.
    """
    guardrail = SuicidePreventionGuardrail(
        sensitivity=sensitivity,
        reasoning=reasoning,
        confidence=confidence
    )
    assert guardrail.sensitivity == sensitivity
    assert guardrail.reasoning == reasoning
    assert guardrail.confidence == confidence
    assert isinstance(guardrail.template, PromptTemplate)

def test_suicide_prevention_guardrail_initialization_default_flags():
    """Tests initialization with default reasoning and confidence flags."""
    guardrail = SuicidePreventionGuardrail(sensitivity=Sensitivity.LOW)
    assert guardrail.reasoning is True # Default in guardrail is True
    assert guardrail.confidence is False # Default in guardrail is False

@patch("psysafe.catalog.suicide_prevention.guardrail.PromptTemplate.from_file")
def test_suicide_prevention_guardrail_apply_method(mock_from_file, mocker):
    """Tests the apply method of SuicidePreventionGuardrail."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Suicide Prevention Prompt"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = SuicidePreventionGuardrail(
        sensitivity=Sensitivity.HIGH,
        reasoning=True,
        confidence=True
    )

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [
            {"role": "system", "content": "Initial system message."},
            {"role": "user", "content": "User expresses concerning thoughts."},
        ],
        "model": "test-model",
        "driver_type": "openai"
    }

    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/suicide_prevention/prompt.md"
    )

    assert mock_template_instance.render.call_count == 1
    render_call_args = mock_template_instance.render.call_args
    assert render_call_args is not None
    render_ctx = render_call_args[0][0]

    assert render_ctx.variables["user_context"] == "User expresses concerning thoughts."
    # RISK_INDICATORS is a constant in the guardrail module
    from psysafe.catalog.suicide_prevention.guardrail import RISK_INDICATORS
    assert render_ctx.variables["risk_indicators_text"] == RISK_INDICATORS
    # The sensitivity block text is now directly from constants
    from psysafe.catalog.suicide_prevention.guardrail import HIGH_SENSITIVITY
    assert render_ctx.variables["sensitivity_block_text"] == HIGH_SENSITIVITY
    assert render_ctx.variables["reasoning"] is True
    assert render_ctx.variables["confidence"] is True

    # Verify the structure of the modified_request
    # The guardrail prepends the new system message
    assert len(modified_request.modified_request["messages"]) == 3
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "system"
    assert modified_request.modified_request["messages"][1].get("content") == "Initial system message."
    assert modified_request.modified_request["messages"][2].get("role") == "user"
    assert modified_request.modified_request["messages"][2].get("content") == "User expresses concerning thoughts."


@patch("psysafe.catalog.suicide_prevention.guardrail.PromptTemplate.from_file")
def test_suicide_prevention_guardrail_apply_no_initial_system(mock_from_file, mocker):
    """Tests apply when the original request has no system message."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Suicide Prompt No System"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = SuicidePreventionGuardrail(sensitivity=Sensitivity.MEDIUM)

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [{"role": "user", "content": "User query."}],
        "model": "test-model",
        "driver_type": "openai"
    }
    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/suicide_prevention/prompt.md"
    )
    mock_template_instance.render.assert_called_once()
    render_ctx = mock_template_instance.render.call_args[0][0]
    assert render_ctx.variables["user_context"] == "User query."
    from psysafe.catalog.suicide_prevention.guardrail import MEDIUM_SENSITIVITY
    assert render_ctx.variables["sensitivity_block_text"] == MEDIUM_SENSITIVITY


    assert len(modified_request.modified_request["messages"]) == 2
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "user"
    assert modified_request.modified_request["messages"][1].get("content") == "User query."


def test_get_sensitivity_block_text():
    """Tests the _get_sensitivity_block_text helper method."""
    guardrail_low = SuicidePreventionGuardrail(sensitivity=Sensitivity.LOW)
    text_low = guardrail_low._get_sensitivity_block_text()
    from psysafe.catalog.suicide_prevention.guardrail import LOW_SENSITIVITY, MEDIUM_SENSITIVITY, HIGH_SENSITIVITY
    assert text_low == LOW_SENSITIVITY

    guardrail_medium = SuicidePreventionGuardrail(sensitivity=Sensitivity.MEDIUM)
    text_medium = guardrail_medium._get_sensitivity_block_text()
    assert text_medium == MEDIUM_SENSITIVITY

    guardrail_high = SuicidePreventionGuardrail(sensitivity=Sensitivity.HIGH)
    text_high = guardrail_high._get_sensitivity_block_text()
    assert text_high == HIGH_SENSITIVITY

# Mock driver for testing the check method
class MockLLMDriver:
    def __init__(self, response_content: str = "", raise_exception: Optional[Exception] = None):
        self.response_content = response_content
        self.raise_exception = raise_exception
        self.send_called_with = None

    def send(self, request_payload: OpenAIChatRequest) -> dict: # type: ignore
        self.send_called_with = request_payload
        if self.raise_exception:
            raise self.raise_exception
        
        # Simulate an OpenAI-like response structure
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.response_content,
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

@pytest.fixture
def suicide_guardrail_for_check():
    """Fixture to create a SuicidePreventionGuardrail instance for check tests."""
    return SuicidePreventionGuardrail(sensitivity=Sensitivity.MEDIUM, reasoning=True, confidence=True)

@pytest.fixture
def sample_conversation():
    """Fixture for a sample conversation."""
    return Conversation(messages=[
        Message(role="user", content="I feel terrible and don't want to live anymore."),
        Message(role="assistant", content="I'm here to listen. Can you tell me more?"),
        Message(role="user", content="There's no point.")
    ])

def test_check_method_success_direct_json(suicide_guardrail_for_check, sample_conversation, mocker):
    """Tests the check method with a successful direct JSON response from LLM."""
    guardrail = suicide_guardrail_for_check
    llm_response_json = {
        "risk_detected": True,
        "risk": 2, # Changed from risk_score to risk, as expected by the guardrail
        "analysis": "User expresses suicidal ideation.",
        "reasoning": "Explicit statements of wanting to die.", # Ensure this matches prompt.md if reasoning is true
        "confidence_level": 0.9 # Ensure this matches prompt.md if confidence is true
    }
    mock_driver = MockLLMDriver(response_content=json.dumps(llm_response_json))
    guardrail.bind(mock_driver) # type: ignore

    # Mock the logger within the guardrail instance for this test
    mock_logger = MagicMock(spec=logging.Logger)
    mocker.patch.object(guardrail, 'logger', mock_logger)


    output = guardrail.check(sample_conversation)

    assert output.is_triggered is True
    assert output.risk_score == 2.0
    assert output.details["reasoning"] == "Explicit statements of wanting to die."
    assert output.details["confidence_level"] == 0.9
    assert output.details["parsed_llm_output"] == llm_response_json
    assert not output.errors
    mock_logger.debug.assert_any_call("Successfully parsed raw_response as direct JSON.")


def test_check_method_success_markdown_json(suicide_guardrail_for_check, sample_conversation, mocker):
    """Tests the check method with JSON in a markdown block."""
    guardrail = suicide_guardrail_for_check
    llm_response_data = {
        "risk_detected": True, "risk": 3, "analysis": "Plan mentioned.", # Changed from risk_score to risk
        "reasoning": "User mentioned a plan.", "confidence_level": 0.95
    }
    raw_llm_content = f"```json\n{json.dumps(llm_response_data)}\n```"
    mock_driver = MockLLMDriver(response_content=raw_llm_content)
    guardrail.bind(mock_driver) # type: ignore
    
    mock_logger = MagicMock(spec=logging.Logger)
    mocker.patch.object(guardrail, 'logger', mock_logger)

    output = guardrail.check(sample_conversation)

    assert output.is_triggered is True
    assert output.risk_score == 3.0
    assert output.details["parsed_llm_output"] == llm_response_data
    assert not output.errors
    mock_logger.debug.assert_any_call("Successfully parsed JSON from Markdown code block.")


def test_check_method_success_xml_like(suicide_guardrail_for_check, sample_conversation, mocker):
    """Tests the check method with a simple XML-like response."""
    guardrail = suicide_guardrail_for_check
    # Based on the updated prompt, the LLM should output JSON.
    # However, parse_llm_response can handle XML, so we test this path.
    # The prompt now strictly asks for JSON, so this case tests the parser's fallback.
    # Suicide guardrail expects a "risk" field, not "risk_score" from the LLM.
    xml_content = "<risk_detected>true</risk_detected><risk>1</risk><analysis>Indirect statements.</analysis><reasoning>User sounds hopeless.</reasoning><confidence_level>0.7</confidence_level>"
    expected_parsed_dict = {
        "risk_detected": "true", "risk": "1", "analysis": "Indirect statements.", # Changed risk_score to risk
        "reasoning": "User sounds hopeless.", "confidence_level": "0.7"
    }
    mock_driver = MockLLMDriver(response_content=xml_content)
    guardrail.bind(mock_driver) # type: ignore

    mock_logger = MagicMock(spec=logging.Logger)
    mocker.patch.object(guardrail, 'logger', mock_logger)

    output = guardrail.check(sample_conversation)
    
    # Note: parse_llm_response for XML will parse "true" as a string.
    # The CheckOutput conversion `bool(is_triggered)` handles this.
    assert output.is_triggered is True
    assert output.risk_score == 1.0
    assert output.details["parsed_llm_output"] == expected_parsed_dict
    assert not output.errors
    mock_logger.debug.assert_any_call(f"Successfully parsed simple XML-like response: {expected_parsed_dict}")


def test_check_method_parse_error(suicide_guardrail_for_check, sample_conversation, mocker):
    """Tests the check method when LLM response is unparseable."""
    guardrail = suicide_guardrail_for_check
    raw_unparseable_content = "This is not valid JSON or XML."
    mock_driver = MockLLMDriver(response_content=raw_unparseable_content)
    guardrail.bind(mock_driver) # type: ignore

    mock_logger = MagicMock(spec=logging.Logger)
    mocker.patch.object(guardrail, 'logger', mock_logger)

    output = guardrail.check(sample_conversation)

    assert output.is_triggered is False # Default on error
    assert output.risk_score is None
    assert output.raw_llm_response == raw_unparseable_content
    assert len(output.errors) == 1
    assert "Failed to parse LLM response" in output.errors[0]
    # Check for a more specific message if available, or the general one
    assert "XML parsed to an empty dictionary" in output.errors[0] or "All parsing attempts failed" in output.errors[0]
    # Logger will get the specific error from parse_llm_response
    mock_logger.error.assert_any_call(f"LLMResponseParseError in check method: All parsing attempts failed (direct JSON, Markdown JSON, simple XML)., Raw Response: {raw_unparseable_content[:200]}")


def test_check_method_llm_call_exception(suicide_guardrail_for_check, sample_conversation, mocker):
    """Tests the check method when the LLM call itself raises an exception."""
    guardrail = suicide_guardrail_for_check
    mock_driver = MockLLMDriver(raise_exception=ValueError("LLM API Error"))
    guardrail.bind(mock_driver) # type: ignore

    mock_logger = MagicMock(spec=logging.Logger) # Mock logger if it's used before/during LLM call error handling
    mocker.patch.object(guardrail, 'logger', mock_logger)

    output = guardrail.check(sample_conversation)

    assert output.is_triggered is False
    assert output.risk_score is None
    assert "Error during LLM call: LLM API Error" in output.errors
    assert output.raw_llm_response == "LLM API Error" # Exception string becomes raw response

def test_check_method_empty_llm_response_content(suicide_guardrail_for_check, sample_conversation, mocker):
    """Tests check method when LLM returns empty content, leading to LLMResponseParseError."""
    guardrail = suicide_guardrail_for_check
    mock_driver = MockLLMDriver(response_content="") # Empty content
    guardrail.bind(mock_driver) # type: ignore

    mock_logger = MagicMock(spec=logging.Logger)
    mocker.patch.object(guardrail, 'logger', mock_logger)

    output = guardrail.check(sample_conversation)

    assert output.is_triggered is False
    assert output.risk_score is None
    assert output.raw_llm_response == ""
    assert len(output.errors) == 1
    assert "Could not extract content from LLM response." in output.errors[0]
    # parse_llm_response is not called if raw_llm_response_content is empty after driver call,
    # so the logger call for LLMResponseParseError from parse_llm_response won't happen in this specific path.
    # We can check if the guardrail's own logger was called if it logs this specific scenario,
    # but for now, let's remove the check for the parse_llm_response specific log.
    # If there's a general error log from the guardrail for this, it could be checked.
    # For now, focusing on the error message in CheckOutput.


def test_check_method_driver_not_bound(suicide_guardrail_for_check, sample_conversation):
    """Tests that check method raises RuntimeError if driver is not bound."""
    guardrail = suicide_guardrail_for_check
    # Do not bind driver
    with pytest.raises(RuntimeError) as excinfo:
        guardrail.check(sample_conversation)
    assert "LLM driver not bound" in str(excinfo.value)

def test_check_method_llm_returns_non_dict_json(suicide_guardrail_for_check, sample_conversation, mocker):
    """
    Tests how check method handles LLM response that is valid JSON but not a dict,
    which should be caught by parse_llm_response if it's not a simple type or
    if the prompt strictly expects a JSON object.
    The current parse_llm_response will raise LLMResponseParseError if JSON is not an object for direct/markdown.
    For XML, it constructs a dict.
    """
    guardrail = suicide_guardrail_for_check
    # This should be caught by parse_llm_response if it's not a dictionary as expected by JSON object prompts
    llm_response_list_json = '[{"item": "not_an_object"}]'
    mock_driver = MockLLMDriver(response_content=llm_response_list_json)
    guardrail.bind(mock_driver) # type: ignore

    mock_logger = MagicMock(spec=logging.Logger)
    mocker.patch.object(guardrail, 'logger', mock_logger)

    output = guardrail.check(sample_conversation)

    assert output.is_triggered is False
    assert output.risk_score is None
    assert output.raw_llm_response == llm_response_list_json
    assert len(output.errors) == 1
    # The error message depends on how parse_llm_response handles non-dict JSON.
    # Assuming it raises LLMResponseParseError because it expects a dictionary structure.
    assert "Failed to parse LLM response" in output.errors[0]
    # Check if the logger was called with the specific error from parse_llm_response
    # This part of the assert might need adjustment based on the exact error message from parse_llm_response
    # For example, if parse_llm_response itself logs "JSON is not an object"
    mock_logger.error.assert_any_call(f"LLMResponseParseError in check method: Parsed JSON is not a dictionary (got <class 'list'>)., Raw Response: {llm_response_list_json[:200]}")

# Need to import logging for the mock_logger
import logging