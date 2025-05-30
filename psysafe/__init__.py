# psysafe/__init__.py
__version__ = "0.1.0.dev0" # Example version

# Expose key components for easier import by users of the SDK
from .core.base import GuardrailBase
from .core.prompt import PromptGuardrail
from .core.check import CheckGuardrail
from .core.composite import CompositeGuardrail
from .core.template import PromptTemplate
from .core.models import GuardedRequest, ValidationReport, Violation, ValidationSeverity, PromptRenderCtx

from .catalog import GuardrailCatalog
# Specific guardrails can be imported directly if desired, or loaded via catalog

from .drivers.base import ChatDriverABC
# Specific drivers

__all__ = [
    "__version__",
    "GuardrailBase",
    "PromptGuardrail",
    "CheckGuardrail",
    "CompositeGuardrail",
    "PromptTemplate",
    "GuardedRequest",
    "ValidationReport",
    "Violation",
    "ValidationSeverity",
    "PromptRenderCtx",
    "GuardrailCatalog",
    # "VulnerabilityDetectionGuardrail", # If directly exposing
    # "SuicidePreventionGuardrail",    # If directly exposing
    "ChatDriverABC",
    # "OpenAIChatDriver",              # If directly exposing
]