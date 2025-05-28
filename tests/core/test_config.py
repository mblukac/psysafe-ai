import pytest
from pydantic import ValidationError

from psysafe.core.config import GuardrailConfig, VulnerabilityConfig, SuicidePreventionConfig
from psysafe.core.types import SensitivityLevel, VulnerabilityIndicators


def test_guardrail_config_defaults():
    """Tests GuardrailConfig default values."""
    config = GuardrailConfig()
    assert config.sensitivity == SensitivityLevel.MEDIUM
    assert config.reasoning_enabled is True
    assert config.confidence_enabled is False # Corrected based on actual default
    assert config.temperature == 0.1
    assert config.max_tokens is None
    assert config.timeout_seconds == 30
    assert config.retry_attempts == 3

def test_guardrail_config_custom_values():
    """Tests GuardrailConfig with custom values."""
    custom_data = {
        "sensitivity": SensitivityLevel.HIGH,
        "reasoning_enabled": False,
        "confidence_enabled": True,
        "temperature": 1.5,
        "max_tokens": 100,
        "timeout_seconds": 60,
        "retry_attempts": 1
    }
    config = GuardrailConfig(**custom_data)
    assert config.sensitivity == SensitivityLevel.HIGH
    assert config.reasoning_enabled is False
    assert config.confidence_enabled is True
    assert config.temperature == 1.5
    assert config.max_tokens == 100
    assert config.timeout_seconds == 60
    assert config.retry_attempts == 1

def test_guardrail_config_invalid_temperature_too_low():
    """Tests GuardrailConfig with temperature below valid range (ge=0.0)."""
    with pytest.raises(ValidationError):
        GuardrailConfig(temperature=-0.1)

def test_guardrail_config_invalid_temperature_too_high():
    """Tests GuardrailConfig with temperature above valid range (le=2.0)."""
    with pytest.raises(ValidationError):
        GuardrailConfig(temperature=2.1)

def test_guardrail_config_invalid_sensitivity_type():
    """Tests GuardrailConfig with invalid type for sensitivity."""
    with pytest.raises(ValidationError):
        GuardrailConfig(sensitivity="very_high_string")

def test_guardrail_config_invalid_max_tokens():
    """Tests GuardrailConfig with invalid max_tokens (gt=0)."""
    with pytest.raises(ValidationError):
        GuardrailConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        GuardrailConfig(max_tokens=-10)
    # Valid case
    config = GuardrailConfig(max_tokens=1)
    assert config.max_tokens == 1


def test_guardrail_config_invalid_timeout_seconds():
    """Tests GuardrailConfig with invalid timeout_seconds (gt=0)."""
    with pytest.raises(ValidationError):
        GuardrailConfig(timeout_seconds=0)
    with pytest.raises(ValidationError):
        GuardrailConfig(timeout_seconds=-5)
    # Valid case
    config = GuardrailConfig(timeout_seconds=1)
    assert config.timeout_seconds == 1


def test_guardrail_config_invalid_retry_attempts():
    """Tests GuardrailConfig with invalid retry_attempts (ge=0)."""
    with pytest.raises(ValidationError):
        GuardrailConfig(retry_attempts=-1)
    # Valid case
    config = GuardrailConfig(retry_attempts=0)
    assert config.retry_attempts == 0


def test_vulnerability_config_defaults():
    """Tests VulnerabilityConfig default values, inheriting from GuardrailConfig."""
    config = VulnerabilityConfig()
    # Inherited defaults
    assert config.sensitivity == SensitivityLevel.MEDIUM
    assert config.reasoning_enabled is True
    assert config.confidence_enabled is False
    assert config.temperature == 0.1
    assert config.max_tokens is None
    assert config.timeout_seconds == 30
    assert config.retry_attempts == 3
    # VulnerabilityConfig specific defaults
    assert config.indicators == list(VulnerabilityIndicators) # Default factory
    assert config.threshold_score == 0.5 # Corrected based on actual default

def test_vulnerability_config_custom_values():
    """Tests VulnerabilityConfig with custom values."""
    custom_data = {
        "sensitivity": SensitivityLevel.LOW,
        "reasoning_enabled": False,
        "confidence_enabled": True,
        "temperature": 0.9,
        "max_tokens": 200,
        "timeout_seconds": 10,
        "retry_attempts": 0,
        "indicators": [VulnerabilityIndicators.HEALTH_CONDITIONS, VulnerabilityIndicators.LIFE_EVENTS], # Using actual members
        "threshold_score": 0.75
    }
    config = VulnerabilityConfig(**custom_data)
    assert config.sensitivity == SensitivityLevel.LOW
    assert config.reasoning_enabled is False
    assert config.confidence_enabled is True
    assert config.temperature == 0.9
    assert config.max_tokens == 200
    assert config.timeout_seconds == 10
    assert config.retry_attempts == 0
    assert config.indicators == [VulnerabilityIndicators.HEALTH_CONDITIONS, VulnerabilityIndicators.LIFE_EVENTS]
    assert config.threshold_score == 0.75

def test_vulnerability_config_invalid_threshold_score_too_low():
    """Tests VulnerabilityConfig with threshold_score below valid range (ge=0.0)."""
    with pytest.raises(ValidationError):
        VulnerabilityConfig(threshold_score=-0.1)

def test_vulnerability_config_invalid_threshold_score_too_high():
    """Tests VulnerabilityConfig with threshold_score above valid range (le=1.0)."""
    with pytest.raises(ValidationError):
        VulnerabilityConfig(threshold_score=1.1)

def test_vulnerability_config_invalid_indicators_type():
    """Tests VulnerabilityConfig with invalid type for indicators."""
    with pytest.raises(ValidationError):
        VulnerabilityConfig(indicators="not_a_list")
    with pytest.raises(ValidationError):
        VulnerabilityConfig(indicators=["not_an_indicator_enum"])


def test_suicide_prevention_config_defaults():
    """Tests SuicidePreventionConfig default values, inheriting from GuardrailConfig."""
    config = SuicidePreventionConfig()
    # Inherited defaults
    assert config.sensitivity == SensitivityLevel.MEDIUM
    assert config.reasoning_enabled is True
    assert config.confidence_enabled is False
    assert config.temperature == 0.1
    assert config.max_tokens is None
    assert config.timeout_seconds == 30
    assert config.retry_attempts == 3
    # SuicidePreventionConfig specific defaults
    assert config.risk_threshold == 0.3
    assert config.emergency_contact_enabled is False
    assert config.crisis_resources_enabled is True # Corrected based on actual default

def test_suicide_prevention_config_custom_values():
    """Tests SuicidePreventionConfig with custom values."""
    custom_data = {
        "sensitivity": SensitivityLevel.HIGH,
        "reasoning_enabled": False,
        "confidence_enabled": True,
        "temperature": 1.0,
        "max_tokens": 50,
        "timeout_seconds": 45,
        "retry_attempts": 2,
        "risk_threshold": 0.85,
        "emergency_contact_enabled": True,
        "crisis_resources_enabled": False
    }
    config = SuicidePreventionConfig(**custom_data)
    assert config.sensitivity == SensitivityLevel.HIGH
    assert config.reasoning_enabled is False
    assert config.confidence_enabled is True
    assert config.temperature == 1.0
    assert config.max_tokens == 50
    assert config.timeout_seconds == 45
    assert config.retry_attempts == 2
    assert config.risk_threshold == 0.85
    assert config.emergency_contact_enabled is True
    assert config.crisis_resources_enabled is False

def test_suicide_prevention_config_invalid_risk_threshold_too_low():
    """Tests SuicidePreventionConfig with risk_threshold below valid range (ge=0.0)."""
    with pytest.raises(ValidationError):
        SuicidePreventionConfig(risk_threshold=-0.01)

def test_suicide_prevention_config_invalid_risk_threshold_too_high():
    """Tests SuicidePreventionConfig with risk_threshold above valid range (le=1.0)."""
    with pytest.raises(ValidationError):
        SuicidePreventionConfig(risk_threshold=1.01)

def test_suicide_prevention_config_boolean_fields():
    """Tests boolean fields in SuicidePreventionConfig."""
    config_true = SuicidePreventionConfig(emergency_contact_enabled=True, crisis_resources_enabled=True)
    assert config_true.emergency_contact_enabled is True
    assert config_true.crisis_resources_enabled is True

    config_false = SuicidePreventionConfig(emergency_contact_enabled=False, crisis_resources_enabled=False)
    assert config_false.emergency_contact_enabled is False
    assert config_false.crisis_resources_enabled is False

    with pytest.raises(ValidationError): # Test invalid type
        SuicidePreventionConfig(emergency_contact_enabled="not_a_bool")
    with pytest.raises(ValidationError): # Test invalid type
        SuicidePreventionConfig(crisis_resources_enabled="not_a_bool")