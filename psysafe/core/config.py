from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from psysafe.core.types import SensitivityLevel, VulnerabilityIndicators

class GuardrailConfig(BaseModel):
    """Unified configuration for all guardrails"""
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
    reasoning_enabled: bool = True
    confidence_enabled: bool = False
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    timeout_seconds: int = Field(default=30, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    
class VulnerabilityConfig(GuardrailConfig):
    """Vulnerability-specific configuration"""
    indicators: list[VulnerabilityIndicators] = Field(
        default_factory=lambda: list(VulnerabilityIndicators)
    )
    threshold_score: float = Field(default=0.5, ge=0.0, le=1.0)

class SuicidePreventionConfig(GuardrailConfig):
    """Suicide prevention-specific configuration"""
    risk_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    emergency_contact_enabled: bool = False
    crisis_resources_enabled: bool = True