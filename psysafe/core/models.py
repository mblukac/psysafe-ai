# psysafe/core/models.py
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel

RequestT = TypeVar('RequestT')

class GuardedRequest(BaseModel, Generic[RequestT]):
    original_request: RequestT
    modified_request: RequestT
    metadata: Dict[str, Any] = {}

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
    metadata: Dict[str, Any] = {}

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