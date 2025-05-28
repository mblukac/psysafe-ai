from typing import Dict, Any

class GuardrailError(Exception):
    """Base exception for all guardrail-related errors"""
    def __init__(self, message: str, guardrail_name: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.guardrail_name = guardrail_name
        self.context = context or {}

class GuardrailConfigError(GuardrailError):
    """Raised when guardrail configuration is invalid"""
    pass

class LLMDriverError(GuardrailError):
    """Raised when LLM driver encounters an error"""
    def __init__(self, message: str, driver_type: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.driver_type = driver_type

class ResponseParsingError(GuardrailError):
    """Raised when LLM response cannot be parsed"""
    def __init__(self, message: str, raw_response: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.raw_response = raw_response

class ValidationError(GuardrailError):
    """Raised when input validation fails"""
    pass

class TimeoutError(GuardrailError):
    """Raised when guardrail operation times out"""
    pass