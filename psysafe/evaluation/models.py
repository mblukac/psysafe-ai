# psysafe/evaluation/models.py
from typing import Any

from pydantic import BaseModel

from psysafe.core.models import ValidationReport  # For actual validation output


class TestCase(BaseModel):
    id: str
    description: str | None = None
    input_request: dict[str, Any]  # Represents a driver request (e.g., OpenAIChatRequest)
    expected_outcome: dict[str, Any] | None = None  # e.g., expected PII found, expected vulnerability level
    # Could also include expected ValidationReport structure or specific violations
    expected_validation_report: ValidationReport | None = None  # More specific


class MetricResult(BaseModel):
    metric_name: str
    value: Any
    description: str | None = None


class EvaluationResult(BaseModel):
    guardrail_name: str
    test_case_id: str
    passed: bool
    actual_validation_report: ValidationReport | None = None
    # actual_llm_response: Optional[Dict[str, Any]] = None # The raw response from LLM after guardrail
    metrics: list[MetricResult] = []
    details: str | None = None  # For any additional notes or error messages
