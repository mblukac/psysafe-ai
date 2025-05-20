# psysafe/catalog/__init__.py
# This can be expanded to automatically discover and register guardrails
# or to provide a convenient access point to the catalog.

# For now, let's import and expose the GuardrailCatalog class
from .registry import GuardrailCatalog

__all__ = ["GuardrailCatalog"]