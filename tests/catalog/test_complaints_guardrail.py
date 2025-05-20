# tests/catalog/test_complaints_guardrail.py
import pytest
from unittest.mock import patch, MagicMock

from psysafe.catalog.complaints_handling.guardrail import ComplaintsHandlingGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage

def test_complaints_handling_guardrail_initialization():
    """Tests basic initialization of ComplaintsHandlingGuardrail."""
    guardrail = ComplaintsHandlingGuardrail()
    # Check if the template is loaded (or if there's a specific attribute to check)
    # For now, just ensure it's an instance of the guardrail
    assert isinstance(guardrail, ComplaintsHandlingGuardrail)
    assert isinstance(guardrail.template, PromptTemplate) # Assuming it loads template on init

@patch("psysafe.catalog.complaints_handling.guardrail.PromptTemplate.from_file")
def test_complaints_handling_guardrail_apply_method(mock_from_file, mocker):
    """Tests the apply method of ComplaintsHandlingGuardrail."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Complaints Handling Prompt"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = ComplaintsHandlingGuardrail()

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [
            {"role": "system", "content": "Initial system message."},
            {"role": "user", "content": "I am very unhappy with the service."},
        ],
        "model": "test-model", # Add other required fields for OpenAIChatRequest if any
        "driver_type": "openai" # Example
    }

    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/complaints_handling/prompt.md"
    )

    assert mock_template_instance.render.call_count == 1
    render_call_args = mock_template_instance.render.call_args
    assert render_call_args is not None
    render_ctx = render_call_args[0][0]

    assert render_ctx.variables["user_input"] == "I am very unhappy with the service."

    # Verify the structure of the modified_request
    # The guardrail prepends the new system message
    assert len(modified_request.modified_request["messages"]) == 3
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "system"
    assert modified_request.modified_request["messages"][1].get("content") == "Initial system message."
    assert modified_request.modified_request["messages"][2].get("role") == "user"
    assert modified_request.modified_request["messages"][2].get("content") == "I am very unhappy with the service."

@patch("psysafe.catalog.complaints_handling.guardrail.PromptTemplate.from_file")
def test_complaints_handling_guardrail_apply_no_initial_system(mock_from_file, mocker):
    """Tests apply when the original request has no system message."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Complaints Prompt No System"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = ComplaintsHandlingGuardrail()

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [{"role": "user", "content": "This is a complaint."}],
        "model": "test-model",
        "driver_type": "openai"
    }
    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/complaints_handling/prompt.md"
    )
    mock_template_instance.render.assert_called_once()
    render_ctx = mock_template_instance.render.call_args[0][0]
    assert render_ctx.variables["user_input"] == "This is a complaint."

    assert len(modified_request.modified_request["messages"]) == 2
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "user"
    assert modified_request.modified_request["messages"][1].get("content") == "This is a complaint."