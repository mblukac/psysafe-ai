# tests/catalog/test_mental_health_guardrail.py
import pytest
from unittest.mock import patch, MagicMock

from psysafe.catalog.mental_health_support.guardrail import MentalHealthSupportGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage

def test_mental_health_support_guardrail_initialization():
    """Tests basic initialization of MentalHealthSupportGuardrail."""
    guardrail = MentalHealthSupportGuardrail()
    assert isinstance(guardrail, MentalHealthSupportGuardrail)
    assert isinstance(guardrail.template, PromptTemplate) # Assuming it loads template on init

@patch("psysafe.catalog.mental_health_support.guardrail.PromptTemplate.from_file")
def test_mental_health_support_guardrail_apply_method(mock_from_file, mocker):
    """Tests the apply method of MentalHealthSupportGuardrail."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Mental Health Support Prompt"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = MentalHealthSupportGuardrail()

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [
            {"role": "system", "content": "Initial system message."},
            {"role": "user", "content": "I'm feeling very down today."},
        ],
        "model": "test-model",
        "driver_type": "openai"
    }

    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/mental_health_support/prompt.md"
    )

    assert mock_template_instance.render.call_count == 1
    render_call_args = mock_template_instance.render.call_args
    assert render_call_args is not None
    render_ctx = render_call_args[0][0]

    assert render_ctx.variables["user_input_context"] == "I'm feeling very down today."

    # Verify the structure of the modified_request
    # The guardrail now prepends the new system message
    assert len(modified_request.modified_request["messages"]) == 3
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "system"
    assert modified_request.modified_request["messages"][1].get("content") == "Initial system message."
    assert modified_request.modified_request["messages"][2].get("role") == "user"
    assert modified_request.modified_request["messages"][2].get("content") == "I'm feeling very down today."

@patch("psysafe.catalog.mental_health_support.guardrail.PromptTemplate.from_file")
def test_mental_health_support_guardrail_apply_no_initial_system(mock_from_file, mocker):
    """Tests apply when the original request has no system message."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered Mental Health Prompt No System"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = MentalHealthSupportGuardrail()

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [{"role": "user", "content": "I need someone to talk to."}],
        "model": "test-model",
        "driver_type": "openai"
    }
    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/mental_health_support/prompt.md"
    )
    mock_template_instance.render.assert_called_once()
    render_ctx = mock_template_instance.render.call_args[0][0]
    assert render_ctx.variables["user_input_context"] == "I need someone to talk to."

    assert len(modified_request.modified_request["messages"]) == 2
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "user"
    assert modified_request.modified_request["messages"][1].get("content") == "I need someone to talk to."