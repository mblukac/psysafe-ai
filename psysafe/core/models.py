# psysafe/core/models.py
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from pydantic import BaseModel, field_validator
from .types import VulnerabilityCheckOutput, SuicideRiskOutput, GuardrailResponse

RequestT = TypeVar('RequestT')

class GuardedRequest(BaseModel, Generic[RequestT]):
    original_request: RequestT
    modified_request: RequestT
    is_modified: bool = False
    applied_guardrails: List[str] = []
    metadata: dict[str, Any] = {} # Allow more flexible metadata, including lists and nested dicts

    @field_validator("metadata")
    @classmethod
    def validate_metadata_keys_are_strings(cls, v: dict) -> dict:
        if not v: # Allow empty dict by default
            return v
        for key in v.keys():
            if not isinstance(key, str):
                raise ValueError(f"Metadata keys must be strings, got {type(key).__name__} for key '{key}'")
        return v

class ValidationSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Violation(BaseModel):
    severity: ValidationSeverity
    code: str
    message: str
    context: Dict[str, Any] = {}

class ValidationReport(BaseModel):
    is_valid: bool
    violations: List[Violation] = []
    metadata: dict[str, Union[str, int, float, bool]] = {}

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        merged_violations = self.violations + other.violations
        merged_is_valid = self.is_valid and other.is_valid
        merged_metadata = {**self.metadata, **other.metadata}
        return ValidationReport(
            is_valid=merged_is_valid,
            violations=merged_violations,
            metadata=merged_metadata
        )

class PromptRenderCtx(BaseModel):
    driver_type: str
    model_name: str
    request_type: str # e.g., "chat", "completion"
    variables: Dict[str, Any] = {}
class Message(BaseModel):
    role: str  # e.g., "user", "assistant", "system"
    content: str

class Conversation(BaseModel):
    messages: List[Message]
class CheckOutput(BaseModel):
    """
    Represents the output of a guardrail's check method,
    typically after an LLM interaction.
    """
    is_triggered: bool
    risk_score: Optional[float] = None # Or appropriate type
    details: dict[str, Any] = {} # For any other structured data from the LLM, allowing nested dicts
    raw_llm_response: Optional[Any] = None # The raw response from the LLM
    errors: List[str] = [] # Any errors encountered during the check
    metadata: dict[str, Any] = {} # Additional metadata from the guardrail, allowing nested dicts