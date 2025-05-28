import pytest
from pydantic import ValidationError

from psysafe.core.types import (
    SensitivityLevel,
    VulnerabilityIndicators,
    VulnerabilityCheckOutput,
    SuicideRiskOutput,
    GuardrailResponse
)


def test_sensitivity_level_enum_values():
    """Tests that SensitivityLevel enum has the correct members and values."""
    assert SensitivityLevel.LOW.value == "low"
    assert SensitivityLevel.MEDIUM.value == "medium"
    assert SensitivityLevel.HIGH.value == "high"
    # CRITICAL is not in the actual enum
    assert len(SensitivityLevel) == 3

def test_sensitivity_level_member_access():
    """Tests member access for SensitivityLevel."""
    assert SensitivityLevel["LOW"] == SensitivityLevel.LOW
    assert SensitivityLevel["MEDIUM"] == SensitivityLevel.MEDIUM
    assert SensitivityLevel["HIGH"] == SensitivityLevel.HIGH
    with pytest.raises(KeyError):
        _ = SensitivityLevel["NON_EXISTENT"]
    with pytest.raises(KeyError): # CRITICAL was removed from the enum in psysafe.core.types
        _ = SensitivityLevel["CRITICAL"]


def test_vulnerability_indicators_enum_values():
    """Tests that VulnerabilityIndicators enum has the correct members and values."""
    assert VulnerabilityIndicators.HEALTH_CONDITIONS.value == "health_conditions"
    assert VulnerabilityIndicators.LIFE_EVENTS.value == "life_events"
    assert VulnerabilityIndicators.RESILIENCE.value == "resilience"
    assert VulnerabilityIndicators.CAPABILITY.value == "capability"
    assert len(VulnerabilityIndicators) == 4

def test_vulnerability_indicators_member_access():
    """Tests member access for VulnerabilityIndicators."""
    assert VulnerabilityIndicators["HEALTH_CONDITIONS"] == VulnerabilityIndicators.HEALTH_CONDITIONS
    assert VulnerabilityIndicators["LIFE_EVENTS"] == VulnerabilityIndicators.LIFE_EVENTS
    assert VulnerabilityIndicators["RESILIENCE"] == VulnerabilityIndicators.RESILIENCE
    assert VulnerabilityIndicators["CAPABILITY"] == VulnerabilityIndicators.CAPABILITY
    with pytest.raises(KeyError):
        _ = VulnerabilityIndicators["NON_EXISTENT"]

def test_vulnerability_check_output_valid():
    """Tests valid VulnerabilityCheckOutput model."""
    data = {
        "is_vulnerable": True,
        "confidence_score": 0.85,
        "severity_level": SensitivityLevel.HIGH,
        "indicators_detected": [VulnerabilityIndicators.HEALTH_CONDITIONS, VulnerabilityIndicators.LIFE_EVENTS],
        "reasoning": "Patient expressed concerns about recent diagnosis and job loss.",
        "raw_response": "{'original': 'data'}",
        "metadata": {"source_id": "123"}
    }
    output = VulnerabilityCheckOutput(**data)
    assert output.is_vulnerable is True
    assert output.confidence_score == 0.85
    assert output.severity_level == SensitivityLevel.HIGH
    assert output.indicators_detected == [VulnerabilityIndicators.HEALTH_CONDITIONS, VulnerabilityIndicators.LIFE_EVENTS]
    assert output.reasoning == "Patient expressed concerns about recent diagnosis and job loss."
    assert output.raw_response == "{'original': 'data'}"
    assert output.metadata == {"source_id": "123"}

def test_vulnerability_check_output_minimal():
    """Tests VulnerabilityCheckOutput with minimal required fields."""
    data = {"is_vulnerable": False} # is_vulnerable is the only truly required field based on the model
    output = VulnerabilityCheckOutput(**data)
    assert output.is_vulnerable is False
    assert output.confidence_score is None
    assert output.severity_level is None
    assert output.indicators_detected == []
    assert output.reasoning is None
    assert output.raw_response is None
    assert output.metadata == {}

def test_vulnerability_check_output_invalid_confidence():
    """Tests VulnerabilityCheckOutput with invalid confidence_score."""
    data = {
        "is_vulnerable": True,
        "confidence_score": "not_a_float", # Intentionally incorrect
    }
    with pytest.raises(ValidationError):
        VulnerabilityCheckOutput(**data)


def test_suicide_risk_output_valid():
    """Tests valid SuicideRiskOutput model."""
    data = {
        "risk_level": "high",
        "risk_score": 0.92,
        "indicators_present": ["hopelessness", "ideation"],
        "reasoning": "User expressed direct suicidal thoughts.",
        "confidence_level": "very_high", # Assuming string for now as per model
        "raw_response": "{'llm_output': 'critical risk'}",
        "metadata": {"session_id": "xyz789"}
    }
    output = SuicideRiskOutput(**data)
    assert output.risk_level == "high"
    assert output.risk_score == 0.92
    assert output.indicators_present == ["hopelessness", "ideation"]
    assert output.reasoning == "User expressed direct suicidal thoughts."
    assert output.confidence_level == "very_high"
    assert output.raw_response == "{'llm_output': 'critical risk'}"
    assert output.metadata == {"session_id": "xyz789"}

def test_suicide_risk_output_minimal():
    """Tests SuicideRiskOutput with minimal required fields."""
    data = {"risk_level": "none"}
    output = SuicideRiskOutput(**data)
    assert output.risk_level == "none"
    assert output.risk_score is None
    assert output.indicators_present == []
    assert output.reasoning is None
    assert output.confidence_level is None
    assert output.raw_response is None
    assert output.metadata == {}

def test_suicide_risk_output_invalid_risk_level():
    """Tests SuicideRiskOutput with invalid risk_level."""
    data = {"risk_level": "catastrophic"} # Intentionally incorrect
    with pytest.raises(ValidationError):
        SuicideRiskOutput(**data)

def test_suicide_risk_output_invalid_risk_score():
    """Tests SuicideRiskOutput with invalid risk_score."""
    data = {
        "risk_level": "medium",
        "risk_score": "not_a_float" # Intentionally incorrect
    }
    with pytest.raises(ValidationError):
        SuicideRiskOutput(**data)

# Assuming confidence_level is a string for now. If it has specific constraints, more tests would be needed.
def test_suicide_risk_output_optional_fields_none():
    """Tests SuicideRiskOutput with optional fields explicitly set to None."""
    data = {
        "risk_level": "low",
        "risk_score": None,
        "indicators_present": [], # or None, model allows list[str] = []
        "reasoning": None,
        "confidence_level": None,
        "raw_response": None,
        "metadata": {} # or None, model allows dict = {}
    }
    output = SuicideRiskOutput(**data)
    assert output.risk_level == "low"
    assert output.risk_score is None
    assert output.indicators_present == []
    assert output.reasoning is None
    assert output.confidence_level is None
    assert output.raw_response is None
    assert output.metadata == {}


def test_guardrail_response_valid():
    """Tests valid GuardrailResponse model."""
    data = {
        "is_triggered": True,
        "risk_score": 0.75,
        "details": {"reason": "High risk detected", "action_taken": "alert_sent"},
        "raw_llm_response": "{'llm_data': 'complex_output'}",
        "errors": [],
        "metadata": {"request_id": "req_abc123"}
    }
    response = GuardrailResponse(**data)
    assert response.is_triggered is True
    assert response.risk_score == 0.75
    assert response.details == {"reason": "High risk detected", "action_taken": "alert_sent"}
    assert response.raw_llm_response == "{'llm_data': 'complex_output'}"
    assert response.errors == []
    assert response.metadata == {"request_id": "req_abc123"}

def test_guardrail_response_minimal():
    """Tests GuardrailResponse with minimal required fields."""
    data = {"is_triggered": False} # is_triggered is the only truly required field
    response = GuardrailResponse(**data)
    assert response.is_triggered is False
    assert response.risk_score is None
    assert response.details == {}
    assert response.raw_llm_response is None
    assert response.errors == []
    assert response.metadata == {}

def test_guardrail_response_with_errors():
    """Tests GuardrailResponse when errors are present."""
    data = {
        "is_triggered": False, # Can be false even if errors occurred during processing
        "errors": ["LLM call failed", "Parsing error"],
        "metadata": {"attempt": 3}
    }
    response = GuardrailResponse(**data)
    assert response.is_triggered is False
    assert response.errors == ["LLM call failed", "Parsing error"]
    assert response.metadata == {"attempt": 3}

def test_guardrail_response_invalid_risk_score():
    """Tests GuardrailResponse with invalid risk_score type."""
    data = {
        "is_triggered": True,
        "risk_score": "high_risk" # Intentionally incorrect type
    }
    with pytest.raises(ValidationError):
        GuardrailResponse(**data)

def test_guardrail_response_invalid_details_type():
    """Tests GuardrailResponse with invalid details type."""
    data = {
        "is_triggered": True,
        "details": "just a string" # Intentionally incorrect type, should be a dict
    }
    with pytest.raises(ValidationError):
        GuardrailResponse(**data)

def test_guardrail_response_extra_fields_ignored():
    """Tests that extra fields are ignored by default."""
    data = {
        "is_triggered": True,
        "risk_score": 0.5,
        "unexpected_field": "this should be ignored"
    }
    # Pydantic models ignore extra fields by default unless configured otherwise
    response = GuardrailResponse(**data)
    assert response.is_triggered is True
    assert response.risk_score == 0.5
    assert not hasattr(response, "unexpected_field")

def test_vulnerability_check_output_invalid_severity():
    """Tests VulnerabilityCheckOutput with invalid severity_level."""
    data = {
        "is_vulnerable": True,
        "severity_level": "invalid_severity", # Intentionally incorrect
    }
    with pytest.raises(ValidationError):
        VulnerabilityCheckOutput(**data)

def test_vulnerability_check_output_invalid_indicators():
    """Tests VulnerabilityCheckOutput with invalid indicators_detected."""
    data = {
        "is_vulnerable": True,
        "indicators_detected": ["not_an_indicator_enum"], # Intentionally incorrect
    }
    with pytest.raises(ValidationError):
        VulnerabilityCheckOutput(**data)