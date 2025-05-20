# psysafe/evaluation/models.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from psysafe.core.models import ValidationReport # For actual validation output

class TestCase(BaseModel):
    id: str
    description: Optional[str] = None
    input_request: Dict[str, Any] # Represents a driver request (e.g., OpenAIChatRequest)
    expected_outcome: Optional[Dict[str, Any]] = None # e.g., expected PII found, expected vulnerability level
    # Could also include expected ValidationReport structure or specific violations
    expected_validation_report: Optional[ValidationReport] = None # More specific

class MetricResult(BaseModel):
    metric_name: str
    value: Any
    description: Optional[str] = None

class EvaluationResult(BaseModel):
    guardrail_name: str
    test_case_id: str
    passed: bool
    actual_validation_report: Optional[ValidationReport] = None
    # actual_llm_response: Optional[Dict[str, Any]] = None # The raw response from LLM after guardrail
    metrics: List[MetricResult] = []
    details: Optional[str] = None # For any additional notes or error messages