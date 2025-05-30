# tests/catalog/test_complaints_guardrail.py
import pytest
import json
from unittest.mock import patch, MagicMock, call

from psysafe.catalog.complaints_handling.guardrail import ComplaintsHandlingGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.typing.requests import OpenAIChatRequest
from psysafe.core.models import Conversation, Message, CheckOutput
from utils.llm_utils import LLMResponseParseError

# Mock the logger for all tests in this file
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("psysafe.catalog.complaints_handling.guardrail.logging.getLogger") as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance


def test_complaints_handling_guardrail_initialization(mock_logger):
    """Tests basic initialization of ComplaintsHandlingGuardrail."""
    guardrail = ComplaintsHandlingGuardrail()
    assert isinstance(guardrail, ComplaintsHandlingGuardrail)
    assert isinstance(guardrail.template, PromptTemplate)
    assert guardrail.logger is not None


@patch("psysafe.catalog.complaints_handling.guardrail.PromptTemplate.from_file")
def test_complaints_handling_guardrail_apply_method(mock_from_file, mock_logger):
    """Tests the apply method of ComplaintsHandlingGuardrail."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Complaints Handling Prompt"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = ComplaintsHandlingGuardrail()
    guardrail.name = "complaints_handling" # Set name attribute for GuardedRequest

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [
            {"role": "system", "content": "Initial system message."},
            {"role": "user", "content": "I am very unhappy with the service."},
        ],
        "model": "test-model",
        "driver_type": "openai"
    }

    guarded_req_obj = guardrail.apply(original_request)
    modified_request = guarded_req_obj.modified_request


    mock_from_file.assert_called_once_with(
        "psysafe/catalog/complaints_handling/prompt.md"
    )

    assert mock_template_instance.render.call_count == 1
    render_call_args = mock_template_instance.render.call_args
    assert render_call_args is not None
    render_ctx = render_call_args[0][0]

    assert render_ctx.variables["user_input"] == "I am very unhappy with the service."
    assert len(modified_request["messages"]) == 3
    assert modified_request["messages"][0].get("role") == "system"
    assert modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request["messages"][1].get("role") == "system"
    assert modified_request["messages"][1].get("content") == "Initial system message."
    assert guarded_req_obj.applied_guardrails == ["complaints_handling"]


@patch("psysafe.catalog.complaints_handling.guardrail.PromptTemplate.from_file")
def test_complaints_handling_guardrail_apply_no_user_input(mock_from_file, mock_logger):
    """Tests apply when no user input can be extracted."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_from_file.return_value = mock_template_instance
    guardrail = ComplaintsHandlingGuardrail()
    guardrail.name = "complaints_handling"

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [{"role": "assistant", "content": "How can I help?"}],
        "model": "test-model"
    }
    guarded_req_obj = guardrail.apply(original_request)
    assert not guarded_req_obj.is_modified
    assert guarded_req_obj.modified_request == original_request
    mock_template_instance.render.assert_not_called()


@pytest.fixture
def guardrail_with_mock_driver(mock_logger):
    """Fixture to provide a guardrail instance with a bound mock LLM driver."""
    guardrail = ComplaintsHandlingGuardrail()
    guardrail.name = "complaints_handling" # Set name for metadata
    mock_driver = MagicMock()
    mock_driver.send = MagicMock() # Mock the send method specifically
    guardrail.bind(mock_driver)
    return guardrail, mock_driver

# --- Test Cases for check() method ---

def test_check_successful_complaint_detected_direct_json(guardrail_with_mock_driver, mock_logger):
    """Test check method with a successful complaint detection (direct JSON response)."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="This is terrible service!")])
    
    llm_response_json = {
        "complaint_detected": True,
        "category": "Service Issue",
        "summary": "User is unhappy with the service.",
        "escalation_needed": True
    }
    mock_driver.send.return_value = {
        "choices": [{"message": {"content": json.dumps(llm_response_json)}}]
    }

    result = guardrail.check(conversation)

    assert result.is_triggered is True
    assert result.details["category"] == "Service Issue"
    assert result.details["summary"] == "User is unhappy with the service."
    assert result.details["escalation_needed"] is True
    assert not result.errors
    mock_driver.send.assert_called_once()


def test_check_successful_no_complaint_markdown_json(guardrail_with_mock_driver, mock_logger):
    """Test check method with no complaint detected (JSON in Markdown)."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="Everything is great!")])
    
    llm_response_json = {
        "complaint_detected": False,
        "category": "N/A",
        "summary": "N/A",
        "escalation_needed": False
    }
    # Simulate LLM wrapping JSON in markdown
    mock_driver.send.return_value = {
        "choices": [{"message": {"content": f"```json\n{json.dumps(llm_response_json)}\n```"}}]
    }

    result = guardrail.check(conversation)

    assert result.is_triggered is False
    assert result.details["category"] == "N/A"
    assert result.details["escalation_needed"] is False
    assert not result.errors


def test_check_llm_response_parse_error(guardrail_with_mock_driver, mock_logger):
    """Test check method when LLM response is unparsable."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="My product broke.")])
    
    raw_bad_response = "This is not JSON."
    mock_driver.send.return_value = {"choices": [{"message": {"content": raw_bad_response}}]}

    result = guardrail.check(conversation)

    assert result.is_triggered is False
    assert len(result.errors) == 1
    assert "Failed to parse LLM response" in result.errors[0]
    assert result.raw_llm_response == raw_bad_response
    assert "parser_error_type" in result.metadata
    assert result.metadata["parser_error_type"] == "LLMResponseParseError"
    # Check if logger was called with error
    # The parse_llm_response now raises a more generic message if all strategies fail.
    # The guardrail's except block logs e.message from LLMResponseParseError.
    expected_error_message_from_parser = "All parsing attempts failed (direct JSON, Markdown JSON, simple XML)."
    mock_logger.error.assert_any_call(
        f"LLMResponseParseError in check method: {expected_error_message_from_parser}. Raw Response: {raw_bad_response[:200]}",
        exc_info=True
    )


def test_check_llm_returns_non_boolean_fields(guardrail_with_mock_driver, mock_logger):
    """Test check method when LLM returns non-boolean for boolean fields."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="This is okay, I guess.")])
    
    llm_response_json = {
        "complaint_detected": "true_string", # Invalid boolean
        "category": "Feedback",
        "summary": "User provided neutral feedback.",
        "escalation_needed": "maybe" # Invalid boolean
    }
    mock_driver.send.return_value = {
        "choices": [{"message": {"content": json.dumps(llm_response_json)}}]
    }

    result = guardrail.check(conversation)

    assert result.is_triggered is False # Defaulted due to invalid "complaint_detected"
    assert result.details["category"] == "Feedback"
    assert result.details["escalation_needed"] is False # Defaulted
    assert len(result.errors) == 2 # Two errors for invalid booleans
    assert "LLM returned non-boolean 'complaint_detected'" in result.errors[0]
    assert "LLM returned non-boolean 'escalation_needed'" in result.errors[1]
    mock_logger.warning.assert_any_call("LLM returned non-boolean 'complaint_detected': true_string. Defaulting to False.")
    mock_logger.warning.assert_any_call("LLM returned non-boolean 'escalation_needed': maybe. Defaulting to False.")


def test_check_llm_call_fails(guardrail_with_mock_driver, mock_logger):
    """Test check method when the call to LLM driver fails."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="Help me.")])
    
    mock_driver.send.side_effect = Exception("Network Error")

    result = guardrail.check(conversation)

    assert result.is_triggered is False
    assert len(result.errors) == 1
    assert "Error during LLM call: Network Error" in result.errors[0]
    assert result.raw_llm_response == "Network Error"
    mock_logger.error.assert_called_once_with("Error during LLM call: Network Error", exc_info=True)


def test_check_llm_no_content_in_response(guardrail_with_mock_driver, mock_logger):
    """Test check method when LLM response has no content."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="Is this working?")])

    mock_driver.send.return_value = {"choices": [{"message": {"content": None}}]} # No content

    result = guardrail.check(conversation)

    assert result.is_triggered is False
    assert len(result.errors) == 1
    assert "Could not extract content from LLM response." in result.errors[0]
    assert result.raw_llm_response is None


def test_check_llm_empty_content_in_response(guardrail_with_mock_driver, mock_logger):
    """Test check method when LLM response has empty string content."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="Test.")])

    mock_driver.send.return_value = {"choices": [{"message": {"content": ""}}]} # Empty content

    result = guardrail.check(conversation)
    
    assert result.is_triggered is False
    assert len(result.errors) == 1
    assert "LLM response content was empty after driver call." in result.errors[0]
    assert result.raw_llm_response == ""


def test_check_driver_not_bound(mock_logger):
    """Test check method raises RuntimeError if driver is not bound."""
    guardrail = ComplaintsHandlingGuardrail() # No driver bound
    conversation = Conversation(messages=[Message(role="user", content="Test")])
    
    with pytest.raises(RuntimeError) as excinfo:
        guardrail.check(conversation)
    assert "LLM driver not bound" in str(excinfo.value)


def test_check_driver_lacks_send_method(mock_logger):
    """Test check method if bound driver does not have a 'send' method."""
    guardrail = ComplaintsHandlingGuardrail()
    mock_driver_no_send = MagicMock()
    # del mock_driver_no_send.send # Ensure it doesn't have 'send'
    # Or, more explicitly:
    if hasattr(mock_driver_no_send, 'send'):
        del mock_driver_no_send.send

    guardrail.bind(mock_driver_no_send)
    conversation = Conversation(messages=[Message(role="user", content="Test")])

    result = guardrail.check(conversation)
    assert result.is_triggered is False
    assert len(result.errors) == 1
    assert f"Bound driver of type {type(mock_driver_no_send).__name__} does not have a 'send' method." in result.errors[0]

def test_check_successful_xml_like_input_parsed(guardrail_with_mock_driver, mock_logger):
    """Test check method with XML-like input that parse_llm_response can handle."""
    guardrail, mock_driver = guardrail_with_mock_driver
    conversation = Conversation(messages=[Message(role="user", content="My order is late.")])
    
    # This XML-like format is what the old prompt produced.
    # parse_llm_response should be able to convert this to a dict.
    xml_like_response = """
    <complaint_detected>True</complaint_detected>
    <category>Service Issue</category>
    <summary>Order is late.</summary>
    <escalation_needed>False</escalation_needed>
    """
    mock_driver.send.return_value = {
        "choices": [{"message": {"content": xml_like_response}}]
    }

    result = guardrail.check(conversation)

    assert result.is_triggered is True # complaint_detected was "True" string, parsed to True bool
    assert result.details["category"] == "Service Issue"
    assert result.details["summary"] == "Order is late."
    assert result.details["escalation_needed"] is False # escalation_needed was "False" string
    assert not result.errors
    # Check that the parsed_llm_output contains the expected dictionary
    expected_parsed_dict = {
        "complaint_detected": True, # parse_llm_response converts "True" to True
        "category": "Service Issue",
        "summary": "Order is late.",
        "escalation_needed": False # parse_llm_response converts "False" to False
    }
    assert result.details["parsed_llm_output"] == expected_parsed_dict