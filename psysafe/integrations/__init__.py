"""Optional framework adapters for PsySafe workflow gates.

Import a concrete adapter module only when its corresponding extra is installed.
The package root deliberately avoids importing either agent SDK.
"""

from psysafe.integrations._serialization import IntegrationInputError

__all__ = ["IntegrationInputError"]
