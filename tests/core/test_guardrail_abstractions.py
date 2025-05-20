# tests/core/test_guardrail_abstractions.py
import pytest
from typing import List, Any, Dict
from psysafe.core.base import GuardrailBase
from psysafe.core.prompt import PromptGuardrail
from psysafe.core.check import CheckGuardrail, Validator
from psysafe.core.composite import CompositeGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.core.models import GuardedRequest, ValidationReport, Violation, ValidationSeverity, PromptRenderCtx
from psysafe.typing.requests import RequestT, OpenAIChatRequest # Using OpenAIChatRequest as a concrete example
from psysafe.typing.responses import ResponseT, OpenAIChatResponse # Using OpenAIChatResponse as a concrete example

# Dummy Request/Response types for testing if not using specific ones
DummyRequest = Dict[str, Any]
DummyResponse = Dict[str, Any]

# --- Test PromptGuardrail ---
def test_prompt_guardrail_apply_openai_no_system_message():
    template = PromptTemplate.from_string("System: {{ instruction }}")
    guardrail_openai = PromptGuardrail[OpenAIChatRequest, OpenAIChatResponse](
        template=template,
        template_variables={"instruction": "Be helpful."}
    )
    openai_request: OpenAIChatRequest = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
    guarded_req = guardrail_openai.apply(openai_request)

    assert guarded_req.original_request == openai_request
    assert isinstance(guarded_req.modified_request, dict)
    assert len(guarded_req.modified_request["messages"]) == 2 # New system message + original user message
    assert guarded_req.modified_request["messages"][0]["role"] == "system"
    assert guarded_req.modified_request["messages"][0]["content"] == "System: Be helpful."
    assert guarded_req.modified_request["messages"][1] == openai_request["messages"][0] # User message preserved
    assert guarded_req.metadata["guardrail_type"] == "PromptGuardrail"

def test_prompt_guardrail_apply_openai_existing_system_message():
    template = PromptTemplate.from_string("System instruction: {{ custom_var }}")
    guardrail = PromptGuardrail[OpenAIChatRequest, OpenAIChatResponse](
        template=template,
        template_variables={"custom_var": "Be concise."}
    )
    initial_request: OpenAIChatRequest = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Initial system message."},
            {"role": "user", "content": "Hello"}
        ]
    }
    guarded_req = guardrail.apply(initial_request)
    expected_system_content = "Initial system message.\nSystem instruction: Be concise."
    assert guarded_req.modified_request["messages"][0]["role"] == "system"
    assert guarded_req.modified_request["messages"][0]["content"] == expected_system_content
    assert len(guarded_req.modified_request["messages"]) == 2
    assert guarded_req.modified_request["messages"][1]["role"] == "user" # User message preserved

def test_prompt_guardrail_apply_openai_empty_messages():
    template = PromptTemplate.from_string("System: {{ instruction }}")
    guardrail = PromptGuardrail[OpenAIChatRequest, OpenAIChatResponse](
        template=template,
        template_variables={"instruction": "Be safe."}
    )
    initial_request: OpenAIChatRequest = {"model": "test-model", "messages": []}
    guarded_req = guardrail.apply(initial_request)
    assert guarded_req.modified_request["messages"][0]["role"] == "system"
    assert guarded_req.modified_request["messages"][0]["content"] == "System: Be safe."
    assert len(guarded_req.modified_request["messages"]) == 1

def test_prompt_guardrail_apply_generic_prompt_key():
    template = PromptTemplate.from_string("Prefix: {{ var }}")
    # Using Dict[str, Any] as RequestT for this generic case
    guardrail = PromptGuardrail[Dict[str, Any], Dict[str, Any]](
        template=template,
        template_variables={"var": "Important"}
    )
    initial_request: Dict[str, Any] = {"prompt": "User query."}
    guarded_req = guardrail.apply(initial_request)
    expected_prompt = "Prefix: Important\n\nUser query."
    assert guarded_req.modified_request["prompt"] == expected_prompt

def test_prompt_guardrail_validate():
    template = PromptTemplate.from_string("test")
    guardrail = PromptGuardrail[DummyRequest, DummyResponse](template=template)
    report = guardrail.validate({"response": "data"})
    assert report.is_valid
    assert not report.violations

def test_prompt_guardrail_from_string():
    guardrail = PromptGuardrail.from_string("Hello {{ name }}", {"name": "Test"})
    assert isinstance(guardrail.template, PromptTemplate)
    assert guardrail.template.template_string == "Hello {{ name }}"
    assert guardrail.template_variables == {"name": "Test"}
    
    # Test rendering through apply to ensure template_variables are used
    request: OpenAIChatRequest = {"model": "test", "messages": [{"role": "user", "content": "Hi"}]}
    guarded_req = guardrail.apply(request) # type: ignore
    assert "Hello Test" in guarded_req.modified_request["messages"][0]["content"]

def test_prompt_guardrail_from_file(tmp_path):
    template_content = "File instruction: {{ item }}"
    template_file = tmp_path / "test_pg_template.md"
    template_file.write_text(template_content)

    guardrail = PromptGuardrail.from_file(str(template_file), {"item": "Setup"}) # type: ignore
    assert isinstance(guardrail.template, PromptTemplate)
    assert guardrail.template.template_path == template_file
    assert guardrail.template_variables == {"item": "Setup"}

    request: OpenAIChatRequest = {"model": "test", "messages": [{"role": "user", "content": "Go"}]}
    guarded_req = guardrail.apply(request)
    assert "File instruction: Setup" in guarded_req.modified_request["messages"][0]["content"]

def test_prompt_guardrail_repr():
    template = PromptTemplate.from_string("Repr test")
    guardrail = PromptGuardrail(template=template)
    assert repr(guardrail) == f"<PromptGuardrail template={template}>"

# --- Test CheckGuardrail ---
def passing_validator(response: DummyResponse) -> ValidationReport:
    return ValidationReport(is_valid=True, metadata={"validator_name": "passing_validator"})

def failing_validator(response: DummyResponse) -> ValidationReport:
    return ValidationReport(is_valid=False, violations=[Violation(severity=ValidationSeverity.ERROR, code="FAIL", message="Failed")], metadata={"validator_name": "failing_validator"})

# Helper validators for new CheckGuardrail tests
def validator_raises_exception(response: DummyResponse) -> ValidationReport:
    raise ValueError("Test exception from validator")

def validator_returns_malformed(response: DummyResponse) -> Any: # Returns Any, not ValidationReport
    return "This is not a ValidationReport"

def test_check_guardrail_apply():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[passing_validator])
    request_data: DummyRequest = {"data": "input"}
    guarded_req = guardrail.apply(request_data)
    assert guarded_req.original_request == request_data
    assert guarded_req.modified_request == request_data # CheckGuardrail doesn't modify

def test_check_guardrail_validate_single_pass():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[passing_validator])
    report = guardrail.validate({"response": "data"})
    assert report.is_valid

def test_check_guardrail_validate_single_fail():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[failing_validator])
    report = guardrail.validate({"response": "data"})
    assert not report.is_valid
    assert len(report.violations) == 1

def test_check_guardrail_validate_multiple():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[passing_validator, failing_validator])
    report = guardrail.validate({"response": "data"})
    assert not report.is_valid # Merged report
    assert len(report.violations) == 1

def test_check_guardrail_validate_no_validators():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[])
    report = guardrail.validate({"response": "data"})
    assert report.is_valid
    assert not report.violations
    assert report.metadata.get("message") == "No validators configured."
    assert report.metadata.get("num_validators_executed") == 0

def test_check_guardrail_validate_validator_raises_exception():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[validator_raises_exception])
    report = guardrail.validate({"response": "data"})
    assert not report.is_valid
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.severity == ValidationSeverity.ERROR
    assert violation.code == "VALIDATOR_EXCEPTION"
    assert "Test exception from validator" in violation.message
    assert violation.context["validator_name"] == "validator_raises_exception"
    assert report.metadata.get("num_validators_executed") == 1

def test_check_guardrail_validate_validator_returns_malformed():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[validator_returns_malformed])
    report = guardrail.validate({"response": "data"})
    assert not report.is_valid
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.severity == ValidationSeverity.ERROR
    assert violation.code == "INVALID_VALIDATOR_RETURN"
    assert "did not return a ValidationReport" in violation.message
    assert violation.context["validator_name"] == "validator_returns_malformed"
    assert report.metadata.get("num_validators_executed") == 1

def test_check_guardrail_validate_mixed_validators_with_exception_and_malformed():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[
        passing_validator,
        validator_raises_exception,
        failing_validator,
        validator_returns_malformed
    ])
    report = guardrail.validate({"response": "data"})
    assert not report.is_valid
    # Expect 3 violations: one from failing_validator, one from exception, one from malformed
    assert len(report.violations) == 3
    codes = {v.code for v in report.violations}
    assert "FAIL" in codes
    assert "VALIDATOR_EXCEPTION" in codes
    assert "INVALID_VALIDATOR_RETURN" in codes
    assert report.metadata.get("num_validators_executed") == 4

def test_check_guardrail_init_non_callable_validator():
    with pytest.raises(TypeError, match="All items in 'validators' must be callable."):
        CheckGuardrail[DummyRequest, DummyResponse](validators=[passing_validator, "not_a_callable"]) # type: ignore

def test_check_guardrail_repr():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[passing_validator, failing_validator])
    expected_repr = "<CheckGuardrail validators=[passing_validator, failing_validator]>"
    assert repr(guardrail) == expected_repr

def test_check_guardrail_repr_no_validators():
    guardrail = CheckGuardrail[DummyRequest, DummyResponse](validators=[])
    expected_repr = "<CheckGuardrail validators=[]>"
    assert repr(guardrail) == expected_repr

# --- Test CompositeGuardrail ---

# Helper classes for CompositeGuardrail tests
class ModifyingPromptGuardrail(PromptGuardrail[DummyRequest, DummyResponse]):
    def __init__(self, id_char: str, prompt_text: str = "TestPrompt"):
        super().__init__(PromptTemplate.from_string(prompt_text))
        self.id_char = id_char
        self.prompt_text = prompt_text # Store for metadata check

    def apply(self, request: DummyRequest) -> GuardedRequest[DummyRequest]:
        modified_req_data = request.copy()
        # Use a minimal render_ctx for this test guardrail
        render_ctx = PromptRenderCtx(driver_type="test", model_name="test", request_type="chat", variables={})
        rendered_prompt = self.template.render(render_ctx)

        current_content = modified_req_data.get("content", "")
        # Prepend "ID:prompt_text|" to content
        modified_req_data["content"] = f"{self.id_char}:{rendered_prompt}|{current_content}"
        
        return GuardedRequest(
            original_request=request,
            modified_request=modified_req_data,
            metadata={"applied_by": self.id_char, "prompt_used": rendered_prompt, "guardrail_type": self.__class__.__name__}
        )
    
    def __repr__(self): # To make __repr__ in CompositeGuardrail work
        return f"<ModifyingPromptGuardrail id_char='{self.id_char}'>"

class StubCheckGuardrail(CheckGuardrail[DummyRequest, DummyResponse]):
    def __init__(self, name: str, validation_report: ValidationReport, raise_on_validate: bool = False):
        super().__init__(validators=[]) # No actual validators needed for stub
        self.name = name
        self._validation_report = validation_report
        self._raise_on_validate = raise_on_validate
        # Ensure metadata has guardrail_type for merging tests
        if "guardrail_type" not in self._validation_report.metadata: # Check before setting
            self._validation_report.metadata["guardrail_type"] = self.__class__.__name__
        if "validator_name" not in self._validation_report.metadata: # Check before setting
            self._validation_report.metadata["validator_name"] = self.name


    def validate(self, response: DummyResponse) -> ValidationReport:
        if self._raise_on_validate:
            raise RuntimeError(f"Exception from {self.name}")
        # Return a copy to avoid modification issues if report is reused
        report_copy = self._validation_report.model_copy(deep=True)
        if "guardrail_name_in_report" not in report_copy.metadata: # Check before setting
            report_copy.metadata["guardrail_name_in_report"] = self.name # Add for easier debugging
        return report_copy

    def __repr__(self): # To make __repr__ in CompositeGuardrail work
        return f"<StubCheckGuardrail name='{self.name}'>"

class NotAGuardrail:
    pass

# Initialization Tests
def test_composite_guardrail_init_empty_list():
    with pytest.raises(ValueError, match="CompositeGuardrail requires at least one guardrail."):
        CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[])

def test_composite_guardrail_init_non_guardrail_base_object():
    g1 = ModifyingPromptGuardrail(id_char="A")
    with pytest.raises(TypeError, match="All items in 'guardrails' must be instances of GuardrailBase."):
        CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, NotAGuardrail()]) # type: ignore

def test_composite_guardrail_init_valid():
    g1 = ModifyingPromptGuardrail(id_char="A")
    g2 = StubCheckGuardrail("stub", ValidationReport(is_valid=True))
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2])
    assert len(composite.guardrails) == 2
    assert composite.guardrails[0] == g1
    assert composite.guardrails[1] == g2

# Apply Tests
def test_composite_guardrail_apply_single_guardrail():
    g1 = ModifyingPromptGuardrail(id_char="A", prompt_text="PromptA")
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1])
    initial_request: DummyRequest = {"content": "UserQuery"}
    
    guarded_req = composite.apply(initial_request)
    
    assert guarded_req.original_request == initial_request
    assert guarded_req.modified_request["content"] == "A:PromptA|UserQuery"
    assert "step_0_ModifyingPromptGuardrail" in guarded_req.metadata
    assert guarded_req.metadata["step_0_ModifyingPromptGuardrail"]["applied_by"] == "A"

def test_composite_guardrail_apply_multiple_modifying_guardrails():
    g1 = ModifyingPromptGuardrail(id_char="A", prompt_text="PromptA")
    g2 = ModifyingPromptGuardrail(id_char="B", prompt_text="PromptB")
    g3 = ModifyingPromptGuardrail(id_char="C", prompt_text="PromptC")
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2, g3])
    initial_request: DummyRequest = {"content": "Start"}
    
    guarded_req = composite.apply(initial_request)
    
    assert guarded_req.original_request == initial_request
    # g1 applies first: A:PromptA|Start
    # g2 applies next: B:PromptB|A:PromptA|Start
    # g3 applies last: C:PromptC|B:PromptB|A:PromptA|Start
    expected_content = "C:PromptC|B:PromptB|A:PromptA|Start"
    assert guarded_req.modified_request["content"] == expected_content
    
    assert "composite_guardrail_sequence" in guarded_req.metadata
    assert guarded_req.metadata["composite_guardrail_sequence"] == ["ModifyingPromptGuardrail", "ModifyingPromptGuardrail", "ModifyingPromptGuardrail"]
    assert "step_0_ModifyingPromptGuardrail" in guarded_req.metadata
    assert guarded_req.metadata["step_0_ModifyingPromptGuardrail"]["applied_by"] == "A"
    assert "step_1_ModifyingPromptGuardrail" in guarded_req.metadata
    assert guarded_req.metadata["step_1_ModifyingPromptGuardrail"]["applied_by"] == "B"
    assert "step_2_ModifyingPromptGuardrail" in guarded_req.metadata
    assert guarded_req.metadata["step_2_ModifyingPromptGuardrail"]["applied_by"] == "C"

def test_composite_guardrail_apply_mixed_guardrails_modification():
    # PromptGuardrail modifies, CheckGuardrail does not
    g1_modify = ModifyingPromptGuardrail(id_char="M1", prompt_text="Mod1")
    g2_check = StubCheckGuardrail("check1", ValidationReport(is_valid=True)) # apply is pass-through
    g3_modify = ModifyingPromptGuardrail(id_char="M2", prompt_text="Mod2")
    
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1_modify, g2_check, g3_modify])
    initial_request: DummyRequest = {"content": "Initial"}
    
    guarded_req = composite.apply(initial_request)
    
    assert guarded_req.original_request == initial_request
    # g1_modify: M1:Mod1|Initial
    # g2_check: (no change to content)
    # g3_modify: M2:Mod2|M1:Mod1|Initial
    expected_content = "M2:Mod2|M1:Mod1|Initial"
    assert guarded_req.modified_request["content"] == expected_content
    
    assert "step_0_ModifyingPromptGuardrail" in guarded_req.metadata
    assert guarded_req.metadata["step_0_ModifyingPromptGuardrail"]["applied_by"] == "M1"
    assert "step_1_StubCheckGuardrail" in guarded_req.metadata # CheckGuardrail's apply metadata
    assert "step_2_ModifyingPromptGuardrail" in guarded_req.metadata
    assert guarded_req.metadata["step_2_ModifyingPromptGuardrail"]["applied_by"] == "M2"

# Validate Tests
def test_composite_guardrail_validate_all_pass():
    g1 = StubCheckGuardrail("pass1", ValidationReport(is_valid=True, violations=[], metadata={"m1": "v1"}))
    g2 = StubCheckGuardrail("pass2", ValidationReport(is_valid=True, violations=[], metadata={"m2": "v2"}))
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2])
    response_data: DummyResponse = {"output": "clean"}
    
    final_report = composite.validate(response_data)
    
    assert final_report.is_valid
    assert not final_report.violations
    assert final_report.metadata["guardrail_type"] == "CompositeGuardrail"
    assert final_report.metadata["num_composed_guardrails"] == 2
    # Check if metadata from individual reports is merged (ValidationReport.merge behavior)
    assert final_report.metadata.get("m1") == "v1" # Using .get for safety if merge logic changes
    assert final_report.metadata.get("m2") == "v2"


def test_composite_guardrail_validate_one_fails():
    v_fail = Violation(severity=ValidationSeverity.CRITICAL, code="V_FAIL", message="Failed validation")
    g1 = StubCheckGuardrail("pass1", ValidationReport(is_valid=True))
    g2 = StubCheckGuardrail("fail1", ValidationReport(is_valid=False, violations=[v_fail], metadata={"failed_by": "g2"}))
    g3 = StubCheckGuardrail("pass2", ValidationReport(is_valid=True))
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2, g3])
    response_data: DummyResponse = {"output": "problematic"}
    
    final_report = composite.validate(response_data)
    
    assert not final_report.is_valid
    assert len(final_report.violations) == 1
    assert final_report.violations[0].code == "V_FAIL"
    assert final_report.metadata.get("failed_by") == "g2"
    assert final_report.metadata["num_composed_guardrails"] == 3

def test_composite_guardrail_validate_multiple_fail_reports_merged():
    v1 = Violation(severity=ValidationSeverity.WARNING, code="V1", message="Issue 1")
    v2 = Violation(severity=ValidationSeverity.CRITICAL, code="V2", message="Issue 2")
    g1 = StubCheckGuardrail("fail1", ValidationReport(is_valid=False, violations=[v1], metadata={"f1_meta": "val1"}))
    g2 = StubCheckGuardrail("pass1", ValidationReport(is_valid=True, metadata={"pass_meta": "val_pass"}))
    g3 = StubCheckGuardrail("fail2", ValidationReport(is_valid=False, violations=[v2], metadata={"f2_meta": "val2"}))
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2, g3])
    
    final_report = composite.validate({"output": "many issues"})
    
    assert not final_report.is_valid
    assert len(final_report.violations) == 2
    codes = {v.code for v in final_report.violations}
    assert "V1" in codes
    assert "V2" in codes
    assert final_report.metadata.get("f1_meta") == "val1"
    assert final_report.metadata.get("f2_meta") == "val2"
    assert final_report.metadata.get("pass_meta") == "val_pass" # Metadata from passing guardrails also merged

def test_composite_guardrail_validate_guardrail_raises_exception():
    v_pass = Violation(severity=ValidationSeverity.INFO, code="PASSED_PREVIOUSLY", message="Passed before exception")
    g1 = StubCheckGuardrail("pass_before_ex", ValidationReport(is_valid=False, violations=[v_pass], metadata={"meta1": "val1"}))
    g2 = StubCheckGuardrail("raiser", ValidationReport(is_valid=True), raise_on_validate=True)
    g3 = StubCheckGuardrail("pass_after_ex", ValidationReport(is_valid=True, metadata={"meta3": "val3"}))
    
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2, g3])
    final_report = composite.validate({"output": "trigger exception"})
    
    assert not final_report.is_valid
    # Expect 2 violations: one from g1, one from the exception in g2
    assert len(final_report.violations) == 2
    codes = {v.code for v in final_report.violations}
    assert "PASSED_PREVIOUSLY" in codes
    assert "COMPOSITE_VALIDATION_EXCEPTION" in codes
    
    exception_violation = next(v for v in final_report.violations if v.code == "COMPOSITE_VALIDATION_EXCEPTION")
    assert "Exception from raiser" in exception_violation.message
    assert exception_violation.context["guardrail_name"] == "StubCheckGuardrail"
    assert exception_violation.context["exception_type"] == "RuntimeError"
    assert final_report.metadata["num_composed_guardrails"] == 3
    assert final_report.metadata.get("meta1") == "val1" # Metadata from g1 should be present
    assert final_report.metadata.get("meta3") == "val3" # Metadata from g3 (after exception) should also be present

def test_composite_guardrail_validate_modifying_prompt_guardrail_in_chain():
    # PromptGuardrails usually return is_valid=True from their validate method
    g1 = ModifyingPromptGuardrail("M1")
    v_fail = Violation(severity=ValidationSeverity.ERROR, code="CHECK_FAIL", message="Check failed")
    g2 = StubCheckGuardrail("checker", ValidationReport(is_valid=False, violations=[v_fail], metadata={"checker_meta": "yes"}))
    
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2])
    final_report = composite.validate({"output": "test"})
    
    assert not final_report.is_valid
    assert len(final_report.violations) == 1
    assert final_report.violations[0].code == "CHECK_FAIL"
    # Check metadata merging. PromptGuardrail.validate adds `guardrail_type`.
    # StubCheckGuardrail also adds `guardrail_type` and `validator_name`.
    # The merge logic of ValidationReport (later overwrites) applies for child metadata.
    # CompositeGuardrail then sets its own type.
    assert final_report.metadata.get("guardrail_type") == "CompositeGuardrail" # CompositeGuardrail sets its own type after merging children.
    assert final_report.metadata.get("checker_meta") == "yes"
    # PromptGuardrail's validate() adds its own metadata too.
    # We need to check what PromptGuardrail.validate() adds to its report's metadata.
    # It adds `guardrail_type`: "PromptGuardrail" (or specific subclass name if overridden)
    # So, if g1 is ModifyingPromptGuardrail, its validate() (from PromptGuardrail) adds
    # metadata={"guardrail_type": "ModifyingPromptGuardrail"}
    # This would be overwritten by g2's metadata if the key is the same.
    # Let's assume ValidationReport.merge updates dicts.
    # The `g1.validate()` report will have `{'guardrail_type': 'ModifyingPromptGuardrail'}`.
    # The `g2.validate()` report will have `{'guardrail_type': 'StubCheckGuardrail', 'validator_name': 'checker', 'checker_meta': 'yes'}`.
    # Merged, `guardrail_type` will be from `g2`.
    assert final_report.metadata.get("validator_name") == "checker" # From g2

# Repr Test
def test_composite_guardrail_repr():
    g1 = ModifyingPromptGuardrail(id_char="A")
    g2 = StubCheckGuardrail("stub_validator", ValidationReport(is_valid=True))
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1, g2])
    
    # CompositeGuardrail uses g.__class__.__name__ for its repr
    expected_repr = "<CompositeGuardrail guardrails=[ModifyingPromptGuardrail, StubCheckGuardrail]>"
    assert repr(composite) == expected_repr

def test_composite_guardrail_repr_single_guardrail():
    g1 = ModifyingPromptGuardrail(id_char="B")
    composite = CompositeGuardrail[DummyRequest, DummyResponse](guardrails=[g1])
    expected_repr = "<CompositeGuardrail guardrails=[ModifyingPromptGuardrail]>"
    assert repr(composite) == expected_repr