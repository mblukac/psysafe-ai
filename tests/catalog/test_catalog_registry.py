# tests/catalog/test_catalog_registry.py
import pytest
from psysafe.catalog.registry import GuardrailCatalog
from psysafe.core.base import GuardrailBase
from psysafe.core.prompt import PromptGuardrail # For a concrete type
from psysafe.core.template import PromptTemplate

class MockGuardrail(PromptGuardrail): # Use a concrete subclass
    def __init__(self, param=None, kwarg1=None, kwarg2="default"):
        super().__init__(PromptTemplate.from_string("mock template {{var}}"))
        self.param = param
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2

@pytest.fixture(autouse=True)
def clear_catalog_registry():
    GuardrailCatalog.clear_registry()
    yield
    GuardrailCatalog.clear_registry()

def test_register_and_load_guardrail():
    GuardrailCatalog.register("mock_g", MockGuardrail)
    assert "mock_g" in GuardrailCatalog.list_available()
    
    loaded_list = GuardrailCatalog.load("mock_g", param="test_value")
    assert len(loaded_list) == 1
    loaded_g = loaded_list[0]
    assert isinstance(loaded_g, MockGuardrail)
    assert loaded_g.param == "test_value"

def test_load_multiple_guardrails():
    GuardrailCatalog.register("mock_g1", MockGuardrail)
    GuardrailCatalog.register("mock_g2", MockGuardrail)
    loaded_list = GuardrailCatalog.load(["mock_g1", "mock_g2"])
    assert len(loaded_list) == 2
    assert isinstance(loaded_list[0], MockGuardrail)
    assert isinstance(loaded_list[1], MockGuardrail)

def test_compose_guardrails():
    GuardrailCatalog.register("mock_g_comp1", MockGuardrail)
    GuardrailCatalog.register("mock_g_comp2", MockGuardrail)
    composite = GuardrailCatalog.compose(["mock_g_comp1", "mock_g_comp2"], param="composed")
    assert len(composite.guardrails) == 2
    assert isinstance(composite.guardrails[0], MockGuardrail)
    assert composite.guardrails[0].param == "composed"

def test_register_duplicate_name_raises_error():
    GuardrailCatalog.register("duplicate_mock", MockGuardrail)
    with pytest.raises(ValueError, match="already registered"):
        GuardrailCatalog.register("duplicate_mock", MockGuardrail)
 
def test_load_unknown_guardrail_raises_error():
    with pytest.raises(ValueError, match="Unknown guardrail"):
        GuardrailCatalog.load("unknown_g")

class NotAGuardrail: pass
def test_register_invalid_type_raises_error():
    with pytest.raises(TypeError, match="must be a subclass of GuardrailBase"):
        GuardrailCatalog.register("invalid_type_g", NotAGuardrail) # type: ignore
def test_load_guardrail_with_kwargs():
    GuardrailCatalog.register("mock_g_kwargs", MockGuardrail)
    loaded_list = GuardrailCatalog.load("mock_g_kwargs", param="val_p", kwarg1="val_k1", kwarg2="val_k2")
    assert len(loaded_list) == 1
    loaded_g = loaded_list[0]
    assert isinstance(loaded_g, MockGuardrail)
    assert loaded_g.param == "val_p"
    assert loaded_g.kwarg1 == "val_k1"
    assert loaded_g.kwarg2 == "val_k2"

    # Test with default kwarg2
    loaded_list_default = GuardrailCatalog.load("mock_g_kwargs", param="val_p_def", kwarg1="val_k1_def")
    assert len(loaded_list_default) == 1
    loaded_g_default = loaded_list_default[0]
    assert loaded_g_default.kwarg2 == "default" # Default value from MockGuardrail

def test_load_guardrail_with_mismatched_kwargs_raises_typeerror():
    GuardrailCatalog.register("mock_g_mismatch", MockGuardrail)
    with pytest.raises(TypeError): # Python raises TypeError for unexpected keyword arguments
        GuardrailCatalog.load("mock_g_mismatch", unexpected_kwarg="should_fail")

def test_compose_guardrails_with_kwargs():
    GuardrailCatalog.register("mock_g_comp_kw1", MockGuardrail)
    GuardrailCatalog.register("mock_g_comp_kw2", MockGuardrail)
    composite = GuardrailCatalog.compose(
        ["mock_g_comp_kw1", "mock_g_comp_kw2"],
        param="composed_val",
        kwarg1="comp_k1",
        kwarg2="comp_k2"
    )
    assert len(composite.guardrails) == 2
    for g in composite.guardrails:
        assert isinstance(g, MockGuardrail)
        assert g.param == "composed_val"
        assert g.kwarg1 == "comp_k1"
        assert g.kwarg2 == "comp_k2"

def test_get_guardrail_class():
    GuardrailCatalog.register("mock_g_get", MockGuardrail)
    klass = GuardrailCatalog.get_guardrail_class("mock_g_get")
    assert klass == MockGuardrail

def test_get_unknown_guardrail_class_raises_error():
    with pytest.raises(ValueError, match=r"Unknown guardrail: 'unknown_class_g'. Available: \[\]"):
        GuardrailCatalog.get_guardrail_class("unknown_class_g")

def test_clear_registry_empties_registry():
    GuardrailCatalog.register("mock_g_clear", MockGuardrail)
    assert "mock_g_clear" in GuardrailCatalog.list_available()
    GuardrailCatalog.clear_registry()
    assert GuardrailCatalog.list_available() == []
    # Ensure it's truly empty by trying to load
    with pytest.raises(ValueError, match="Unknown guardrail"):
        GuardrailCatalog.load("mock_g_clear")
    # And re-registering is possible
    GuardrailCatalog.register("mock_g_clear", MockGuardrail)
    assert "mock_g_clear" in GuardrailCatalog.list_available()