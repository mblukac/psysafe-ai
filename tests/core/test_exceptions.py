import pytest

from psysafe.core.exceptions import (
    GuardrailError,
    GuardrailConfigError,
    LLMDriverError,
    ResponseParsingError,
    ValidationError as PsySafeValidationError, # Alias to avoid conflict with pydantic
    TimeoutError as PsySafeTimeoutError # Alias if needed, though pytest.raises is specific
)


def test_guardrail_error_base_exception():
    """Tests the GuardrailError base exception."""
    message = "Base guardrail error occurred."
    guardrail_name = "TestGuardrail"
    context = {"detail": "Some context"}

    # Test with message only
    err_msg_only = GuardrailError(message=message)
    assert err_msg_only.message == message
    assert err_msg_only.guardrail_name is None
    assert err_msg_only.context == {}
    assert str(err_msg_only) == message # Default __str__ is usually the first arg to __init__

    # Test with message and guardrail_name
    err_with_name = GuardrailError(message=message, guardrail_name=guardrail_name)
    assert err_with_name.message == message
    assert err_with_name.guardrail_name == guardrail_name
    assert err_with_name.context == {}

    # Test with message, guardrail_name, and context
    err_full = GuardrailError(message=message, guardrail_name=guardrail_name, context=context)
    assert err_full.message == message
    assert err_full.guardrail_name == guardrail_name
    assert err_full.context == context
    assert str(err_full) == message

    # Test raising and catching
    with pytest.raises(GuardrailError) as exc_info:
        raise GuardrailError(message="Raising test", guardrail_name="CatchMe")
    assert exc_info.value.message == "Raising test"
    assert exc_info.value.guardrail_name == "CatchMe"


def test_guardrail_error_is_exception_subclass():
    """Ensures GuardrailError is a subclass of Exception."""
    assert issubclass(GuardrailError, Exception)


def test_guardrail_config_error():
    """Tests GuardrailConfigError."""
    message = "Invalid configuration."
    guardrail_name = "ConfigTest"
    context = {"field": "api_key"}
    with pytest.raises(GuardrailConfigError) as exc_info:
        raise GuardrailConfigError(message=message, guardrail_name=guardrail_name, context=context)
    assert exc_info.value.message == message
    assert exc_info.value.guardrail_name == guardrail_name
    assert exc_info.value.context == context
    assert issubclass(GuardrailConfigError, GuardrailError)
    # No specific attributes to test beyond base GuardrailError for GuardrailConfigError


def test_llm_driver_error():
    """Tests LLMDriverError."""
    message = "LLM connection failed."
    guardrail_name = "LLMTest"
    driver_type = "OpenAI"
    context = {"endpoint": "v1/chat/completions"}
    with pytest.raises(LLMDriverError) as exc_info:
        raise LLMDriverError(message=message, guardrail_name=guardrail_name, driver_type=driver_type, context=context)
    assert exc_info.value.message == message
    assert exc_info.value.guardrail_name == guardrail_name
    assert exc_info.value.driver_type == driver_type
    assert exc_info.value.context == context
    assert issubclass(LLMDriverError, GuardrailError)
    assert exc_info.value.driver_type == "OpenAI" # Corrected


def test_response_parsing_error():
    """Tests ResponseParsingError."""
    message = "Failed to parse LLM response."
    guardrail_name = "ParsingTest"
    raw_response = "{'malformed_json': ..."
    context = {"parser_used": "json.loads"}
    with pytest.raises(ResponseParsingError) as exc_info:
        raise ResponseParsingError(message=message, guardrail_name=guardrail_name, raw_response=raw_response, context=context)
    assert exc_info.value.message == message
    assert exc_info.value.guardrail_name == guardrail_name
    assert exc_info.value.raw_response == raw_response
    assert exc_info.value.context == context
    assert issubclass(ResponseParsingError, GuardrailError)
    assert exc_info.value.raw_response == raw_response # Corrected


def test_psysafe_validation_error():
    """Tests psysafe.core.exceptions.ValidationError."""
    message = "Input validation failed."
    guardrail_name = "ValidationTest"
    context = {"input_field": "user_text", "reason": "too short"}
    with pytest.raises(PsySafeValidationError) as exc_info:
        raise PsySafeValidationError(message=message, guardrail_name=guardrail_name, context=context)
    assert exc_info.value.message == message
    assert exc_info.value.guardrail_name == guardrail_name
    assert exc_info.value.context == context
    assert issubclass(PsySafeValidationError, GuardrailError)
    assert exc_info.value.context.get("reason") == "too short" # Corrected


def test_psysafe_timeout_error():
    """Tests psysafe.core.exceptions.TimeoutError."""
    message = "Operation timed out."
    guardrail_name = "TimeoutTest"
    context = {"timeout_duration": 30}
    with pytest.raises(PsySafeTimeoutError) as exc_info:
        raise PsySafeTimeoutError(message=message, guardrail_name=guardrail_name, context=context)
    assert exc_info.value.message == message
    assert exc_info.value.guardrail_name == guardrail_name
    assert exc_info.value.context == context
    assert issubclass(PsySafeTimeoutError, GuardrailError)
    assert exc_info.value.message == message # Corrected