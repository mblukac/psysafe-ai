from enum import Enum
from typing import Literal

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
    confidence_score: float | None = None
    severity_level: SensitivityLevel | None = None
    indicators_detected: list[VulnerabilityIndicators] = []
    reasoning: str | None = None
    raw_response: str | None = None
    metadata: dict[str, str | int | float | bool] = {}


class SuicideRiskOutput(BaseModel):
    risk_level: Literal["none", "low", "medium", "high", "critical"]
    risk_score: float | None = None
    indicators_present: list[str] = []
    reasoning: str | None = None
    confidence_level: str | None = None
    raw_response: str | None = None
    metadata: dict[str, str | int | float | bool] = {}


class GuardrailResponse(BaseModel):
    """Unified response type for all guardrails."""

    is_triggered: bool
    risk_score: float | None = None
    details: dict[str, str | int | float | bool | list] = {}
    raw_llm_response: str | None = None
    errors: list[str] = []
    metadata: dict[str, str | int | float | bool] = {}
