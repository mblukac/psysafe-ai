# tests/catalog/test_pii_guardrail.py
import pytest
from unittest.mock import patch, MagicMock

from psysafe.catalog.pii_protection.guardrail import PiiProtectionGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage

def test_pii_protection_guardrail_initialization():
    """Tests basic initialization of PiiProtectionGuardrail."""
    guardrail = PiiProtectionGuardrail()
    assert isinstance(guardrail, PiiProtectionGuardrail)
    assert isinstance(guardrail.template, PromptTemplate) # Assuming it loads template on init

@patch("psysafe.catalog.pii_protection.guardrail.PromptTemplate.from_file")
def test_pii_protection_guardrail_apply_method(mock_from_file, mocker):
    """Tests the apply method of PiiProtectionGuardrail."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered PII Protection Prompt"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = PiiProtectionGuardrail()

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [
            {"role": "system", "content": "Initial system message."},
            {"role": "user", "content": "My email is test@example.com and phone is 123-456-7890."},
        ],
        "model": "test-model",
        "meta": {"driver_type": "openai", "model_name": "test-model"} # Added meta for driver/model
    }

    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/pii_protection/prompt.md"
    )

    assert mock_template_instance.render.call_count == 1
    render_call_args = mock_template_instance.render.call_args
    assert render_call_args is not None
    render_ctx = render_call_args[0][0]

    # The PII guardrail joins messages with their roles for context
    expected_context = "system: Initial system message.\nuser: My email is test@example.com and phone is 123-456-7890."
    assert render_ctx.variables["user_input_context"] == expected_context

    # Verify the structure of the modified_request
    # The guardrail modifies the existing system message or prepends a new one.
    # In this case, it modifies the existing one.
    assert len(modified_request.modified_request["messages"]) == 2
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert mock_render_output in modified_request.modified_request["messages"][0].get("content")
    assert "Initial system message." in modified_request.modified_request["messages"][0].get("content")
    assert modified_request.modified_request["messages"][1].get("role") == "user"
    assert modified_request.modified_request["messages"][1].get("content") == "My email is test@example.com and phone is 123-456-7890."

@patch("psysafe.catalog.pii_protection.guardrail.PromptTemplate.from_file")
def test_pii_protection_guardrail_apply_no_initial_system(mock_from_file, mocker):
    """Tests apply when the original request has no system message."""
    mock_template_instance = MagicMock(spec=PromptTemplate)
    mock_render_output = "Rendered PII Prompt No System"
    mock_template_instance.render.return_value = mock_render_output
    mock_from_file.return_value = mock_template_instance

    guardrail = PiiProtectionGuardrail()

    original_request: OpenAIChatRequest = { # type: ignore
        "messages": [{"role": "user", "content": "My address is 123 Main St."}],
        "model": "test-model",
        "meta": {"driver_type": "openai", "model_name": "test-model"}
    }
    modified_request = guardrail.apply(original_request)

    mock_from_file.assert_called_once_with(
        "psysafe/catalog/pii_protection/prompt.md"
    )
    mock_template_instance.render.assert_called_once()
    render_ctx = mock_template_instance.render.call_args[0][0]
    expected_context_no_system = "user: My address is 123 Main St."
    assert render_ctx.variables["user_input_context"] == expected_context_no_system

    # Prepends a new system message
    assert len(modified_request.modified_request["messages"]) == 2
    assert modified_request.modified_request["messages"][0].get("role") == "system"
    assert modified_request.modified_request["messages"][0].get("content") == mock_render_output
    assert modified_request.modified_request["messages"][1].get("role") == "user"
    assert modified_request.modified_request["messages"][1].get("content") == "My address is 123 Main St."