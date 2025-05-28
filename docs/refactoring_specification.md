# PsySafe-AI Codebase Refactoring Specification

## Executive Summary

This specification outlines a comprehensive refactoring plan to address critical code quality issues in the psysafe-ai codebase, focusing on module organization, type safety, error handling consistency, and configuration management.

## Current State Analysis

### 1. Module Organization Issues

#### 1.1 Duplicated Functionality
- **Location**: [`utils/__init__.py:173-246`](utils/__init__.py:173-246) and [`guardrails/vulnerability/run.py:10-79`](guardrails/vulnerability/run.py:10-79)
- **Issue**: Identical `analyze_text_vulnerability()` function exists in both locations
- **Impact**: Code maintenance burden, potential inconsistencies, unclear ownership

#### 1.2 Architectural Inconsistency
- **Legacy Structure**: [`guardrails/`](guardrails/) directory contains older implementation patterns
- **Modern Structure**: [`psysafe/catalog/`](psysafe/catalog/) contains newer, more structured implementations
- **Issue**: Two parallel systems for the same functionality

#### 1.3 Empty Directories
- **Location**: [`evaluators/`](evaluators/) directory exists but contains minimal functionality
- **Issue**: Unclear purpose and incomplete implementation

### 2. Type Safety Weaknesses

#### 2.1 Generic Type Usage
- **Current**: Extensive use of `Dict[str, Any]` throughout codebase
- **Locations**: 
  - [`psysafe/core/models.py`](psysafe/core/models.py) - metadata fields
  - [`psysafe/catalog/vulnerability_detection/guardrail.py`](psysafe/catalog/vulnerability_detection/guardrail.py) - LLM responses
  - [`utils/llm_utils.py`](utils/llm_utils.py) - function returns

#### 2.2 Missing Typed Models
- **VulnerabilityCheckOutput**: No structured type for vulnerability assessment results
- **GuardrailConfig**: Configuration parameters lack type safety
- **LLMResponse**: Raw LLM responses not properly typed

### 3. Error Handling Inconsistency

#### 3.1 Parsing Error Strategies
- **Vulnerability Guardrail**: Uses [`LLMResponseParseError`](utils/llm_utils.py:343-351) with detailed error context
- **Suicide Prevention**: Similar error handling but different default behaviors
- **Legacy Guardrails**: Silent failures with `False` defaults

#### 3.2 Missing Error Hierarchy
- **Current**: Ad-hoc exception handling
- **Need**: Unified `GuardrailError` base class with specific subtypes

### 4. Configuration Management Issues

#### 4.1 Sensitivity Handling
- **Vulnerability**: [`Sensitivity.LOW/MEDIUM/HIGH`](psysafe/catalog/vulnerability_detection/guardrail.py:13-16)
- **Suicide Prevention**: [`Sensitivity.LOW/MEDIUM/HIGH`](psysafe/catalog/suicide_prevention/guardrail.py:13-16) (different enum values)
- **Issue**: Inconsistent enum definitions and handling

#### 4.2 Configuration Scattered
- Each guardrail implements its own configuration logic
- No unified configuration management system

## Proposed New Structure

### 1. Module Reorganization

```
psysafe/
├── core/
│   ├── base.py              # Abstract base classes
│   ├── models.py            # Core data models
│   ├── config.py            # NEW: Unified configuration
│   ├── exceptions.py        # NEW: Error hierarchy
│   └── types.py             # NEW: Type definitions
├── catalog/
│   ├── base/                # NEW: Base guardrail implementations
│   │   ├── __init__.py
│   │   ├── prompt_guardrail.py
│   │   └── llm_guardrail.py
│   ├── vulnerability/       # RENAMED from vulnerability_detection
│   ├── suicide_prevention/
│   ├── complaints_handling/
│   ├── mental_health_support/
│   └── pii_protection/
├── utils/
│   ├── __init__.py          # CLEANED: Remove duplicated functions
│   ├── llm_utils.py         # ENHANCED: Better typing
│   └── parsing.py           # NEW: Centralized response parsing
└── evaluation/              # ENHANCED: Move from top-level evaluators/
    ├── __init__.py
    ├── metrics.py
    ├── runner.py
    └── datasets/
```

### 2. Type Safety Enhancements

#### 2.1 Core Type Definitions

```python
# psysafe/core/types.py

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
```

#### 2.2 Configuration Types

```python
# psysafe/core/config.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

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
```

### 3. Error Handling Hierarchy

```python
# psysafe/core/exceptions.py

class GuardrailError(Exception):
    """Base exception for all guardrail-related errors"""
    def __init__(self, message: str, guardrail_name: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
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
```

### 4. Unified Response Parsing

```python
# psysafe/utils/parsing.py

from typing import Dict, Any, Type, TypeVar, Optional
import json
import re
import logging
from xml.etree import ElementTree as ET

from psysafe.core.exceptions import ResponseParsingError
from psysafe.core.types import GuardrailResponse

T = TypeVar('T', bound=BaseModel)

class ResponseParser:
    """Centralized response parsing with multiple format support"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def parse_to_model(self, raw_response: str, model_class: Type[T]) -> T:
        """Parse raw response to specific Pydantic model"""
        try:
            parsed_dict = self._parse_to_dict(raw_response)
            return model_class.model_validate(parsed_dict)
        except Exception as e:
            raise ResponseParsingError(
                f"Failed to parse response to {model_class.__name__}: {str(e)}",
                raw_response=raw_response
            )
    
    def _parse_to_dict(self, raw_response: str) -> Dict[str, Any]:
        """Parse raw response to dictionary using multiple strategies"""
        if not raw_response or not raw_response.strip():
            raise ResponseParsingError("Empty response", raw_response=raw_response)
        
        # Strategy 1: Direct JSON
        try:
            result = json.loads(raw_response)
            if isinstance(result, dict):
                self.logger.debug("Parsed as direct JSON")
                return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_response, re.IGNORECASE)
        if json_match:
            try:
                result = json.loads(json_match.group(1).strip())
                if isinstance(result, dict):
                    self.logger.debug("Parsed JSON from markdown")
                    return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: XML-like key-value pairs
        try:
            return self._parse_xml_like(raw_response)
        except Exception:
            pass
        
        raise ResponseParsingError(
            "Could not parse response with any strategy",
            raw_response=raw_response
        )
    
    def _parse_xml_like(self, raw_response: str) -> Dict[str, Any]:
        """Parse simple XML-like key-value pairs"""
        # Implementation similar to existing parse_llm_response
        # but with better error handling and type safety
        pass
```

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)

#### 1.1 Create Core Infrastructure
- [ ] Implement [`psysafe/core/types.py`](psysafe/core/types.py)
- [ ] Implement [`psysafe/core/config.py`](psysafe/core/config.py)  
- [ ] Implement [`psysafe/core/exceptions.py`](psysafe/core/exceptions.py)
- [ ] Implement [`psysafe/utils/parsing.py`](psysafe/utils/parsing.py)

#### 1.2 Update Core Models
- [ ] Enhance [`psysafe/core/models.py`](psysafe/core/models.py) with new types
- [ ] Update [`CheckOutput`](psysafe/core/models.py:51-61) to use typed fields
- [ ] Add validation to [`GuardedRequest`](psysafe/core/models.py:8-11)

#### 1.3 Testing Infrastructure
- [ ] Create unit tests for new core types
- [ ] Create integration tests for parsing utilities
- [ ] Set up type checking with mypy

### Phase 2: Guardrail Modernization (Weeks 3-4)

#### 2.1 Create Base Guardrail Classes
```python
# psysafe/catalog/base/prompt_guardrail.py

from abc import abstractmethod
from typing import TypeVar, Generic
from psysafe.core.base import GuardrailBase
from psysafe.core.config import GuardrailConfig
from psysafe.core.types import GuardrailResponse
from psysafe.core.models import Conversation

T = TypeVar('T', bound=GuardrailConfig)

class ModernPromptGuardrail(GuardrailBase, Generic[T]):
    """Modern base class for prompt-based guardrails"""
    
    def __init__(self, config: T):
        self.config = config
        self.parser = ResponseParser()
    
    @abstractmethod
    def check(self, conversation: Conversation) -> GuardrailResponse:
        """Check conversation with typed response"""
        pass
    
    def _validate_config(self) -> None:
        """Validate guardrail configuration"""
        if not isinstance(self.config, GuardrailConfig):
            raise GuardrailConfigError("Invalid configuration type")
```

#### 2.2 Migrate Existing Guardrails
- [ ] Migrate [`VulnerabilityDetectionGuardrail`](psysafe/catalog/vulnerability_detection/guardrail.py:26-304)
- [ ] Migrate [`SuicidePreventionGuardrail`](psysafe/catalog/suicide_prevention/guardrail.py:51-356)
- [ ] Update other catalog guardrails

#### 2.3 Remove Duplications
- [ ] Remove [`analyze_text_vulnerability`](utils/__init__.py:173-246) from utils
- [ ] Deprecate [`guardrails/`](guardrails/) directory
- [ ] Update imports across codebase

### Phase 3: Enhanced Error Handling (Week 5)

#### 3.1 Implement Error Hierarchy
- [ ] Replace generic exceptions with typed exceptions
- [ ] Add error context and metadata
- [ ] Implement retry logic with exponential backoff

#### 3.2 Update Error Handling Patterns
```python
# Example: Enhanced error handling in guardrails

def check(self, conversation: Conversation) -> GuardrailResponse:
    try:
        # Guardrail logic
        raw_response = self._call_llm(conversation)
        parsed_response = self.parser.parse_to_model(raw_response, VulnerabilityCheckOutput)
        return self._convert_to_guardrail_response(parsed_response)
    
    except ResponseParsingError as e:
        self.logger.error(f"Parse error in {self.__class__.__name__}: {e}")
        return GuardrailResponse(
            is_triggered=False,
            errors=[f"Parse error: {e.message}"],
            raw_llm_response=e.raw_response,
            metadata={"error_type": "parsing", "guardrail": self.__class__.__name__}
        )
    
    except LLMDriverError as e:
        self.logger.error(f"LLM error in {self.__class__.__name__}: {e}")
        return GuardrailResponse(
            is_triggered=False,
            errors=[f"LLM error: {str(e)}"],
            metadata={"error_type": "llm_driver", "driver_type": e.driver_type}
        )
    
    except Exception as e:
        self.logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
        raise GuardrailError(f"Unexpected error: {str(e)}", guardrail_name=self.__class__.__name__)
```

### Phase 4: Configuration Management (Week 6)

#### 4.1 Unified Configuration System
- [ ] Implement configuration validation
- [ ] Add configuration file support (YAML/JSON)
- [ ] Create configuration migration utilities

#### 4.2 Environment-Specific Configs
```python
# config/development.yaml
guardrails:
  vulnerability:
    sensitivity: "high"
    reasoning_enabled: true
    confidence_enabled: true
    indicators: ["health_conditions", "life_events", "resilience", "capability"]
    threshold_score: 0.3
  
  suicide_prevention:
    sensitivity: "high"
    risk_threshold: 0.2
    emergency_contact_enabled: false
    crisis_resources_enabled: true

# config/production.yaml  
guardrails:
  vulnerability:
    sensitivity: "medium"
    reasoning_enabled: false
    confidence_enabled: false
    threshold_score: 0.5
```

### Phase 5: Testing and Validation (Week 7)

#### 5.1 Comprehensive Testing
- [ ] Unit tests for all new components
- [ ] Integration tests for guardrail workflows
- [ ] Performance benchmarks
- [ ] Type checking validation

#### 5.2 Backward Compatibility
- [ ] Deprecation warnings for old APIs
- [ ] Migration guides for users
- [ ] Compatibility shims where needed

## Edge Cases and Constraints

### 1. Backward Compatibility
- **Constraint**: Existing users depend on current APIs
- **Solution**: Gradual deprecation with clear migration paths
- **Timeline**: 6-month deprecation period for major changes

### 2. Performance Impact
- **Constraint**: Type validation and parsing overhead
- **Mitigation**: 
  - Lazy validation where possible
  - Caching for repeated operations
  - Performance benchmarks to track regression

### 3. Configuration Migration
- **Edge Case**: Users with custom configuration patterns
- **Solution**: Configuration migration utilities and validation

### 4. LLM Response Variability
- **Edge Case**: Different LLM providers return different formats
- **Solution**: Robust parsing with multiple fallback strategies

### 5. Error Recovery
- **Edge Case**: Partial failures in composite guardrails
- **Solution**: Graceful degradation with detailed error reporting

## Success Criteria

### 1. Code Quality Metrics
- [ ] **Type Coverage**: >95% of codebase properly typed
- [ ] **Test Coverage**: >90% line coverage for core modules
- [ ] **Cyclomatic Complexity**: <10 for all functions
- [ ] **Duplication**: <5% code duplication across modules

### 2. Performance Benchmarks
- [ ] **Response Time**: <2s for single guardrail checks
- [ ] **Memory Usage**: <50MB baseline memory footprint
- [ ] **Throughput**: >100 requests/second for simple checks

### 3. Developer Experience
- [ ] **IDE Support**: Full autocomplete and type hints
- [ ] **Error Messages**: Clear, actionable error messages
- [ ] **Documentation**: Complete API documentation with examples
- [ ] **Migration**: Zero-downtime migration path

### 4. Reliability Metrics
- [ ] **Error Rate**: <1% unhandled exceptions
- [ ] **Recovery**: 100% graceful error recovery
- [ ] **Consistency**: Identical behavior across environments

## Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 2 weeks | Core infrastructure, types, exceptions |
| Phase 2 | 2 weeks | Modernized guardrails, removed duplications |
| Phase 3 | 1 week | Enhanced error handling |
| Phase 4 | 1 week | Unified configuration system |
| Phase 5 | 1 week | Testing, validation, documentation |

**Total Duration**: 7 weeks

## Risk Mitigation

### 1. Breaking Changes
- **Risk**: Refactoring breaks existing integrations
- **Mitigation**: Feature flags, gradual rollout, extensive testing

### 2. Performance Regression
- **Risk**: New type system impacts performance
- **Mitigation**: Continuous benchmarking, optimization passes

### 3. Adoption Resistance
- **Risk**: Team resistance to new patterns
- **Mitigation**: Training sessions, clear documentation, gradual migration

## Conclusion

This refactoring specification provides a comprehensive roadmap for modernizing the psysafe-ai codebase. The phased approach ensures minimal disruption while delivering significant improvements in code quality, type safety, and maintainability.

The success of this refactoring will result in:
- **Cleaner Architecture**: Well-organized, single-responsibility modules
- **Type Safety**: Comprehensive typing reducing runtime errors
- **Consistent Error Handling**: Predictable error behavior across all components
- **Unified Configuration**: Centralized, validated configuration management
- **Better Developer Experience**: Enhanced IDE support and debugging capabilities

Implementation should begin with Phase 1 to establish the foundation, followed by systematic migration of existing components while maintaining backward compatibility throughout the process.