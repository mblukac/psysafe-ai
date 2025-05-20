# psysafe/core/base.py
from abc import ABC, abstractmethod
from typing import Any, Generic # Removed TypeVar as it's not directly used here, but in requests/responses

# Assuming RequestT and ResponseT will be imported from psysafe.typing
from psysafe.typing.requests import RequestT
from psysafe.typing.responses import ResponseT
from psysafe.core.models import GuardedRequest, ValidationReport

class GuardrailBase(Generic[RequestT, ResponseT], ABC):
    """Abstract base class for all guardrails."""

    @abstractmethod
    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """Apply the guardrail to a request.
        This method should modify the incoming request as needed (e.g., add system prompts,
        reformat input) and return a GuardedRequest object containing both the original
        and modified request, along with any relevant metadata.
        """
        pass

    @abstractmethod
    def validate(self, response: ResponseT) -> ValidationReport:
        """Validate a response against the guardrail.
        This method should inspect the LLM's response and return a ValidationReport
        indicating whether the response is valid according to the guardrail's criteria,
        along with any violations found and associated metadata.
        """
        pass

    def bind(self, driver: Any) -> Any:
        """
        Binds the guardrail to a specific LLM driver instance.
        This is an optional method that can be implemented by guardrails
        if they need to interact with or configure the driver.
        For example, a guardrail might set specific parameters on the driver
        or use driver-specific functionalities.

        By default, it returns the driver unchanged.
        Subclasses can override this to provide custom binding logic.
        """
        # Placeholder for now.
        # Actual implementation might involve wrapping the driver or
        # storing a reference to it if the guardrail needs to call driver methods.
        # For a simple case, it might do nothing or return a modified driver.
        # print(f"Binding {self.__class__.__name__} to {driver.__class__.__name__}")
        return driver # Or a wrapped/modified driver