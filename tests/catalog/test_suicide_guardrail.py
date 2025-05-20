# tests/catalog/test_suicide_guardrail.py
import pytest
from unittest.mock import patch, MagicMock

from psysafe.catalog.suicide_prevention.guardrail import (
    SuicidePreventionGuardrail,
    Sensitivity
)
from psysafe.core.template import PromptTemplate
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage

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