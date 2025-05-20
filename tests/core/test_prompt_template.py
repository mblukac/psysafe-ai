# tests/core/test_prompt_template.py
import pytest
from pathlib import Path
from psysafe.core.template import PromptTemplate
from psysafe.core.models import PromptRenderCtx

def test_prompt_template_from_string():
    template_str = "Hello {{ name }}!"
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"name": "World"})
    pt = PromptTemplate.from_string(template_str)
    assert pt.render(ctx) == "Hello World!"

def test_prompt_template_from_file(tmp_path: Path):
    template_content = "Test file content: {{ item }}."
    template_file = tmp_path / "test_template.md"
    template_file.write_text(template_content)
    
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"item": "example"})
    pt = PromptTemplate.from_file(template_file)
    assert pt.render(ctx) == "Test file content: example."
    assert pt.template_path == template_file

def test_prompt_template_file_not_found():
    with pytest.raises(FileNotFoundError):
        PromptTemplate.from_file("non_existent_template.md")

def test_prompt_template_render_with_loop():
    template_str = "Items:{% for item in items %} - {{ item }}{% endfor %}"
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"items": ["apple", "banana", "cherry"]})
    pt = PromptTemplate.from_string(template_str)
    expected_output = "Items: - apple - banana - cherry"
    assert pt.render(ctx) == expected_output

def test_prompt_template_render_with_conditional():
    template_str = "{% if show_greeting %}Hello {{ name }}!{% else %}Goodbye {{ name }}.{% endif %}"
    ctx_greeting = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"show_greeting": True, "name": "Alice"})
    ctx_no_greeting = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"show_greeting": False, "name": "Bob"})
    pt = PromptTemplate.from_string(template_str)
    assert pt.render(ctx_greeting) == "Hello Alice!"
    assert pt.render(ctx_no_greeting) == "Goodbye Bob."

def test_prompt_template_render_missing_variable():
    template_str = "Hello {{ name }}!"
    # 'name' is missing from variables
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"age": 30})
    pt = PromptTemplate.from_string(template_str)
    assert pt.render(ctx) == "Hello !"

def test_prompt_template_render_none_variable():
    template_str = "Value: {{ value }}"
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"value": None})
    pt = PromptTemplate.from_string(template_str)
    assert pt.render(ctx) == "Value: "

def test_prompt_template_render_empty_string_variable():
    template_str = "Text: {{ text }}"
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"text": ""})
    pt = PromptTemplate.from_string(template_str)
    assert pt.render(ctx) == "Text: "

def test_prompt_template_repr_from_string():
    template_str = "Hello {{ name }}"
    pt = PromptTemplate.from_string(template_str)
    expected_repr = f"<PromptTemplate source='(string)', length={len(template_str)}>"
    assert repr(pt) == expected_repr

def test_prompt_template_repr_from_file(tmp_path: Path):
    template_content = "Test file content."
    template_file = tmp_path / "test_repr_template.md"
    template_file.write_text(template_content)
    pt = PromptTemplate.from_file(template_file)
    expected_repr = f"<PromptTemplate source='{template_file}'>"
    assert repr(pt) == expected_repr

def test_prompt_template_autoescape_off_by_default_for_string():
    template_str = "Data: {{ value }}"
    html_value = "<script>alert('XSS')</script> & an ampersand"
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"value": html_value})
    pt = PromptTemplate.from_string(template_str)
    assert pt.render(ctx) == f"Data: {html_value}"

def test_prompt_template_autoescape_off_by_default_for_file(tmp_path: Path):
    template_content = "Data: {{ value }}"
    template_file = tmp_path / "test_autoescape_template.md" # .md extension
    template_file.write_text(template_content)

    html_value = "<p>Test & Fun</p>"
    ctx = PromptRenderCtx(driver_type="test", model_name="test_model", request_type="chat", variables={"value": html_value})
    pt = PromptTemplate.from_file(template_file)
    # Expecting no escaping as from_file uses from_string logic internally for compilation
    assert pt.render(ctx) == f"Data: {html_value}"

# Add more tests for different rendering contexts, missing variables, etc.