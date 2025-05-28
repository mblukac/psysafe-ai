from typing import TypedDict, Literal, Union, Optional
from enum import Enum
from pydantic import BaseModel

class SensitivityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class VulnerabilityIndicators(Enum):
    HEALTH_CONDITIONS = "health_conditions"
    LIFE_EVENTS = "life_events"
    RESILIENCE = "resilience"
    CAPABILITY = "capability"

class VulnerabilityCheckOutput(BaseModel):
    is_vulnerable: bool
    confidence_score: Optional[float] = None
    severity_level: Optional[SensitivityLevel] = None
    indicators_detected: list[VulnerabilityIndicators] = []
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: dict[str, Union[str, int, float, bool]] = {}

class SuicideRiskOutput(BaseModel):
    risk_level: Literal["none", "low", "medium", "high", "critical"]
    risk_score: Optional[float] = None
    indicators_present: list[str] = []
    reasoning: Optional[str] = None
    confidence_level: Optional[str] = None
    raw_response: Optional[str] = None
    metadata: dict[str, Union[str, int, float, bool]] = {}

class GuardrailResponse(BaseModel):
    """Unified response type for all guardrails"""
    is_triggered: bool
    risk_score: Optional[float] = None
    details: dict[str, Union[str, int, float, bool, list]] = {}
    raw_llm_response: Optional[str] = None
    errors: list[str] = []
    metadata: dict[str, Union[str, int, float, bool]] = {}