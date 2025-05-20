# tests/core/test_models.py
import pytest
from pydantic import ValidationError
from typing import Any, Dict, List, Optional

from psysafe.core.models import (
    GuardedRequest,
    ValidationReport,
    Violation,
    ValidationSeverity,
    PromptRenderCtx
)

# --- Test ValidationReport.merge() ---
def test_validation_report_merge_two_valid():
    report1 = ValidationReport(is_valid=True, violations=[], metadata={"source": "report1", "common_key": "val1"})
    report2 = ValidationReport(is_valid=True, violations=[], metadata={"checker": "report2", "common_key": "val2_override"})
    
    merged_report = report1.merge(report2)
    
    assert merged_report.is_valid
    assert not merged_report.violations
    assert merged_report.metadata == {"source": "report1", "checker": "report2", "common_key": "val2_override"}

def test_validation_report_merge_valid_and_invalid():
    v1 = Violation(severity=ValidationSeverity.ERROR, code="E1", message="Error 1")
    report1 = ValidationReport(is_valid=True, metadata={"source": "report1"})
    report2 = ValidationReport(is_valid=False, violations=[v1], metadata={"source": "report2_override"})
    
    merged_report_v_i = report1.merge(report2)
    assert not merged_report_v_i.is_valid
    assert len(merged_report_v_i.violations) == 1
    assert merged_report_v_i.violations[0].code == "E1"
    assert merged_report_v_i.metadata == {"source": "report2_override"}
    
    merged_report_i_v = report2.merge(report1) # Order matters for is_valid
    assert not merged_report_i_v.is_valid
    assert len(merged_report_i_v.violations) == 1
    assert merged_report_i_v.violations[0].code == "E1"
    # Metadata: report1's "source" will override report2's if report2 is self
    assert merged_report_i_v.metadata == {"source": "report1"}


def test_validation_report_merge_two_invalid_different_violations():
    v1 = Violation(severity=ValidationSeverity.WARNING, code="W1", message="Warning 1")
    v2 = Violation(severity=ValidationSeverity.ERROR, code="E1", message="Error 1")
    report1 = ValidationReport(is_valid=False, violations=[v1], metadata={"r1_key": "v1"})
    report2 = ValidationReport(is_valid=False, violations=[v2], metadata={"r2_key": "v2"})
    
    merged_report = report1.merge(report2)
    
    assert not merged_report.is_valid
    assert len(merged_report.violations) == 2
    codes = {v.code for v in merged_report.violations}
    assert "W1" in codes
    assert "E1" in codes
    assert merged_report.metadata == {"r1_key": "v1", "r2_key": "v2"}

def test_validation_report_merge_with_and_without_metadata():
    v1 = Violation(severity=ValidationSeverity.INFO, code="I1", message="Info 1")
    report1 = ValidationReport(is_valid=True, violations=[v1], metadata={"key1": "val1"})
    report2 = ValidationReport(is_valid=True, violations=[]) # No metadata
    
    merged12 = report1.merge(report2)
    assert merged12.is_valid
    assert len(merged12.violations) == 1
    assert merged12.metadata == {"key1": "val1"}
    
    merged21 = report2.merge(report1)
    assert merged21.is_valid
    assert len(merged21.violations) == 1
    assert merged21.metadata == {"key1": "val1"}

def test_validation_report_merge_empty_violations_lists():
    report1 = ValidationReport(is_valid=True, metadata={"m1": "v1"})
    report2 = ValidationReport(is_valid=True, metadata={"m2": "v2"})
    
    merged = report1.merge(report2)
    assert merged.is_valid
    assert not merged.violations
    assert merged.metadata == {"m1": "v1", "m2": "v2"}

def test_validation_report_merge_self_does_not_mutate():
    v1 = Violation(severity=ValidationSeverity.ERROR, code="E1", message="Error 1")
    report1 = ValidationReport(is_valid=False, violations=[v1], metadata={"orig_key": "orig_val"})
    report1_copy_meta = report1.metadata.copy()
    report1_copy_violations = report1.violations[:]

    v2 = Violation(severity=ValidationSeverity.WARNING, code="W1", message="Warning 1")
    report2 = ValidationReport(is_valid=True, violations=[v2], metadata={"new_key": "new_val"})
    
    report1.merge(report2) # Call merge
    
    # Assert report1 is unchanged
    assert report1.is_valid is False
    assert report1.violations == report1_copy_violations
    assert report1.violations[0].code == "E1"
    assert len(report1.violations) == 1
    assert report1.metadata == report1_copy_meta

# --- Test GuardedRequest ---
def test_guarded_request_valid_data():
    original_req = {"query": "hello"}
    modified_req = {"query": "hello world"}
    gr = GuardedRequest[Dict](original_request=original_req, modified_request=modified_req, metadata={"id": 1})
    assert gr.original_request == original_req
    assert gr.modified_request == modified_req
    assert gr.metadata == {"id": 1}

def test_guarded_request_missing_fields():
    with pytest.raises(ValidationError) as excinfo:
        GuardedRequest() # Missing original_request and modified_request
    errors = excinfo.value.errors()
    assert len(errors) == 2
    error_locs = {e["loc"][0] for e in errors}
    assert "original_request" in error_locs
    assert "modified_request" in error_locs

    with pytest.raises(ValidationError) as excinfo:
        GuardedRequest(original_request={"query": "test"}) # Missing modified_request
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("modified_request",)

def test_guarded_request_default_metadata():
    gr = GuardedRequest(original_request=1, modified_request=2)
    assert gr.metadata == {}

# Note: Pydantic handles type validation for original_request and modified_request based on RequestT.
# Testing specific type errors for these generic fields is less about GuardedRequest itself
# and more about how Pydantic handles the provided TypeVar.
# We'll assume Pydantic's generic handling is correct.
# For metadata, it must be a dict.
def test_guarded_request_metadata_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        GuardedRequest(original_request=1, modified_request=2, metadata="not_a_dict")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("metadata",)
    assert errors[0]["type"] == "dict_type"

# --- Test ValidationReport ---
def test_validation_report_valid_data_is_valid_true():
    v1 = Violation(severity=ValidationSeverity.INFO, code="I1", message="Info")
    report = ValidationReport(is_valid=True, violations=[v1], metadata={"check": "manual"})
    assert report.is_valid
    assert len(report.violations) == 1
    assert report.violations[0].code == "I1"
    assert report.metadata == {"check": "manual"}

def test_validation_report_valid_data_is_valid_false():
    v1 = Violation(severity=ValidationSeverity.ERROR, code="E1", message="Error")
    report = ValidationReport(is_valid=False, violations=[v1], metadata={"checker_id": 5})
    assert not report.is_valid
    assert len(report.violations) == 1
    assert report.violations[0].code == "E1"
    assert report.metadata == {"checker_id": 5}

def test_validation_report_default_fields():
    report = ValidationReport(is_valid=True) # violations and metadata should have defaults
    assert report.is_valid
    assert report.violations == []
    assert report.metadata == {}

def test_validation_report_missing_is_valid():
    with pytest.raises(ValidationError) as excinfo:
        ValidationReport() # Missing is_valid
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("is_valid",)
    assert errors[0]["type"] == "missing"

def test_validation_report_is_valid_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        ValidationReport(is_valid="not_a_bool")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("is_valid",)
    assert errors[0]["type"] == "bool_parsing"

def test_validation_report_violations_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        ValidationReport(is_valid=True, violations="not_a_list")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("violations",)
    assert errors[0]["type"] == "list_type"

def test_validation_report_violations_list_with_incorrect_item_type():
    v1 = Violation(severity=ValidationSeverity.INFO, code="I1", message="Info")
    with pytest.raises(ValidationError) as excinfo:
        ValidationReport(is_valid=True, violations=[v1, "not_a_violation_object"])
    errors = excinfo.value.errors()
    # Example: [{'type': 'model_type', 'loc': ('violations', 1), 'msg': 'Input should be a valid dictionary or instance of Violation', 'input': 'not_a_violation_object', 'ctx': {'class_name': 'Violation'}, ...}]
    assert len(errors) == 1
    assert errors[0]["loc"] == ("violations", 1)
    assert "model_type" in errors[0]["type"] or "is_instance_of" in errors[0]["type"] # Pydantic v2 can vary here
    assert "Violation" in errors[0]["msg"]


def test_validation_report_metadata_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        ValidationReport(is_valid=True, metadata="not_a_dict")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("metadata",)
    assert errors[0]["type"] == "dict_type"

# --- Test Violation ---
def test_violation_valid_data():
    v = Violation(
        severity=ValidationSeverity.ERROR,
        code="ERR001",
        message="An error occurred.",
        context={"details": "some_details", "value": 42}
    )
    assert v.severity == ValidationSeverity.ERROR
    assert v.code == "ERR001"
    assert v.message == "An error occurred."
    assert v.context == {"details": "some_details", "value": 42}

def test_violation_default_context():
    v = Violation(
        severity=ValidationSeverity.WARNING,
        code="WARN001",
        message="A warning."
    )
    assert v.severity == ValidationSeverity.WARNING
    assert v.code == "WARN001"
    assert v.message == "A warning."
    assert v.context == {}

def test_violation_missing_required_fields():
    with pytest.raises(ValidationError) as excinfo:
        Violation(code="C1", message="M1") # Missing severity
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("severity",)

    with pytest.raises(ValidationError) as excinfo:
        Violation(severity=ValidationSeverity.INFO, message="M1") # Missing code
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("code",)

    with pytest.raises(ValidationError) as excinfo:
        Violation(severity=ValidationSeverity.INFO, code="C1") # Missing message
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("message",)

def test_violation_severity_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        Violation(severity="NOT_A_SEVERITY", code="C1", message="M1")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("severity",)
    assert errors[0]["type"] == "enum"
    # Pydantic v2 provides a more specific message for enum errors
    assert "Input should be" in errors[0]["msg"] and all(
        member.value in errors[0]["msg"] for member in ValidationSeverity
    )


def test_violation_code_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        Violation(severity=ValidationSeverity.INFO, code=123, message="M1") # Code should be str
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("code",)
    assert errors[0]["type"] == "string_type"

def test_violation_message_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        Violation(severity=ValidationSeverity.INFO, code="C1", message=123) # Message should be str
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("message",)
    assert errors[0]["type"] == "string_type"

def test_violation_context_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        Violation(severity=ValidationSeverity.INFO, code="C1", message="M1", context="not_a_dict")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("context",)
    assert errors[0]["type"] == "dict_type"

# --- Test PromptRenderCtx ---
def test_prompt_render_ctx_valid_data():
    ctx = PromptRenderCtx(
        driver_type="openai",
        model_name="gpt-4",
        request_type="chat",
        variables={"user_name": "Alice", "topic": "testing"}
    )
    assert ctx.driver_type == "openai"
    assert ctx.model_name == "gpt-4"
    assert ctx.request_type == "chat"
    assert ctx.variables == {"user_name": "Alice", "topic": "testing"}

def test_prompt_render_ctx_default_variables():
    ctx = PromptRenderCtx(
        driver_type="anthropic",
        model_name="claude-2",
        request_type="completion"
    )
    assert ctx.variables == {}

def test_prompt_render_ctx_missing_required_fields():
    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(model_name="m", request_type="r") # Missing driver_type
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("driver_type",)

    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(driver_type="d", request_type="r") # Missing model_name
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("model_name",)

    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(driver_type="d", model_name="m") # Missing request_type
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("request_type",)

def test_prompt_render_ctx_driver_type_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(driver_type=123, model_name="m", request_type="r")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("driver_type",)
    assert errors[0]["type"] == "string_type"

def test_prompt_render_ctx_model_name_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(driver_type="d", model_name=456, request_type="r")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("model_name",)
    assert errors[0]["type"] == "string_type"

def test_prompt_render_ctx_request_type_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(driver_type="d", model_name="m", request_type=789)
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("request_type",)
    assert errors[0]["type"] == "string_type"

def test_prompt_render_ctx_variables_incorrect_type():
    with pytest.raises(ValidationError) as excinfo:
        PromptRenderCtx(driver_type="d", model_name="m", request_type="r", variables="not_a_dict")
    errors = excinfo.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("variables",)
    assert errors[0]["type"] == "dict_type"

# --- Test ValidationSeverity ---
def test_validation_severity_enum_members():
    assert ValidationSeverity.INFO.value == "INFO"
    assert ValidationSeverity.WARNING.value == "WARNING"
    assert ValidationSeverity.ERROR.value == "ERROR"
    assert ValidationSeverity.CRITICAL.value == "CRITICAL"

    # Check that all expected members are present
    expected_members = {"INFO", "WARNING", "ERROR", "CRITICAL"}
    actual_members = {member.name for member in ValidationSeverity}
    assert actual_members == expected_members

def test_validation_severity_assignment_and_comparison():
    severity_info = ValidationSeverity.INFO
    assert severity_info == ValidationSeverity.INFO
    assert severity_info.value == "INFO"

    severity_error = ValidationSeverity("ERROR") # Can also be instantiated from value
    assert severity_error == ValidationSeverity.ERROR

def test_validation_severity_invalid_value():
    with pytest.raises(ValueError): # Enum raises ValueError for invalid value
        ValidationSeverity("NON_EXISTENT_SEVERITY")