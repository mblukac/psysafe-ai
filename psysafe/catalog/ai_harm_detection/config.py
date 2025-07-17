from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from psysafe.core.config import GuardrailConfig


class HarmClassification(Enum):
    """Classification levels for AI harm detection"""
    HARMFUL = "HARMFUL"
    BORDERLINE = "BORDERLINE"
    SAFE = "SAFE"


class PolicyViolationType(Enum):
    """Types of policy violations for AI harm detection"""
    DIRECT_AGREEMENT = "1.1"
    ENCOURAGEMENT_GLORIFICATION = "1.2"
    INSTRUCTIONAL_HARM = "1.3"
    EATING_DISORDERS = "2.1"
    SUBSTANCE_ABUSE = "2.2"
    SELF_DESTRUCTIVE_PATTERNS = "2.3"
    
    @property
    def policy_id(self) -> str:
        """Get the policy ID string"""
        return self.value


class AiHarmDetectionConfig(GuardrailConfig):
    """Configuration for AI harm detection guardrail"""
    
    # Detection sensitivity
    detection_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Threshold for triggering harm detection"
    )
    
    # Policy violations to monitor
    monitored_policies: List[PolicyViolationType] = Field(
        default_factory=lambda: list(PolicyViolationType),
        description="List of policy violations to monitor"
    )
    
    # Output configuration
    include_policy_details: bool = Field(
        default=True,
        description="Include detailed policy violation information in output"
    )
    
    # Response format configuration
    require_step_by_step_reasoning: bool = Field(
        default=True,
        description="Require step-by-step reasoning in LLM response"
    )
    
    # Safety configuration
    fail_safe_mode: bool = Field(
        default=True,
        description="Default to BORDERLINE classification on parsing errors"
    )
    
    # Classification thresholds
    borderline_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold between SAFE and BORDERLINE classifications"
    )
    
    harmful_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold between BORDERLINE and HARMFUL classifications"
    )
    
    class Config:
        """Pydantic config"""
        use_enum_values = True
        validate_assignment = True
        
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.borderline_threshold >= self.harmful_threshold:
            raise ValueError("borderline_threshold must be less than harmful_threshold")