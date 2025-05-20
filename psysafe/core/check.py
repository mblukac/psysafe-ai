# psysafe/core/check.py
from typing import Any, Callable, List, Generic

# Core imports
from psysafe.core.base import GuardrailBase
from psysafe.core.models import GuardedRequest, ValidationReport, Violation # Added Violation for Validator type hint

# Typing imports
from psysafe.typing.requests import RequestT
from psysafe.typing.responses import ResponseT

# Type alias for a validator function
# A validator takes a response and returns a list of Violations.
# It could also return a full ValidationReport, but for simplicity in definition,
# let's assume it returns List[Violation] and CheckGuardrail wraps it.
# Or, more flexibly, it returns a ValidationReport directly. Let's go with ValidationReport for consistency.
Validator = Callable[[ResponseT], ValidationReport]


class CheckGuardrail(GuardrailBase[RequestT, ResponseT], Generic[RequestT, ResponseT]):
    """
    Guardrail that validates responses by applying a list of validator functions.
    It does not modify the outgoing request.
    """

    def __init__(self, validators: List[Validator]):
        """
        Initializes a CheckGuardrail.

        Args:
            validators: A list of validator functions. Each function must accept
                        a response object (ResponseT) and return a ValidationReport.
        """
        if not all(callable(v) for v in validators):
            raise TypeError("All items in 'validators' must be callable.")
        self.validators = validators

    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """
        Applies the guardrail to a request.
        CheckGuardrails do not modify the request; they only validate responses.

        Args:
            request: The original request object.

        Returns:
            A GuardedRequest object with the original request unchanged.
        """
        return GuardedRequest[RequestT](
            original_request=request,
            modified_request=request, # Request is not modified
            metadata={
                "guardrail_type": self.__class__.__name__,
                "num_validators": len(self.validators)
            }
        )

    def validate(self, response: ResponseT) -> ValidationReport:
        """
        Validates the response by applying all registered validator functions.
        The reports from each validator are merged.

        Args:
            response: The response object from the LLM.

        Returns:
            A single ValidationReport aggregating results from all validators.
        """
        if not self.validators:
            return ValidationReport(
                is_valid=True,
                violations=[],
                metadata={
                    "message": "No validators configured.",
                    "num_validators_executed": 0,
                    "guardrail_type": self.__class__.__name__
                }
            )

        final_report = ValidationReport(is_valid=True) # Start with a valid report

        for validator_func in self.validators:
            try:
                # Each validator is expected to return a ValidationReport
                report_from_validator = validator_func(response)
                if not isinstance(report_from_validator, ValidationReport):
                    # Handle cases where a validator might return something else, or log a warning
                    # For now, we'll create a default error violation for this misbehaving validator
                    error_violation = Violation(
                        severity="ERROR", # Assuming ValidationSeverity.ERROR exists
                        code="INVALID_VALIDATOR_RETURN",
                        message=f"Validator {validator_func.__name__} did not return a ValidationReport.",
                        context={"validator_name": validator_func.__name__}
                    )
                    # Create a report for this specific error
                    misbehaving_validator_report = ValidationReport(is_valid=False, violations=[error_violation])
                    final_report = final_report.merge(misbehaving_validator_report)
                else:
                    final_report = final_report.merge(report_from_validator)
            except Exception as e:
                # If a validator raises an exception, treat it as a validation failure for that validator
                exception_violation = Violation(
                    severity="ERROR", # Assuming ValidationSeverity.ERROR exists
                    code="VALIDATOR_EXCEPTION",
                    message=f"Validator {validator_func.__name__} raised an exception: {str(e)}",
                    context={"validator_name": validator_func.__name__, "exception_type": e.__class__.__name__}
                )
                exception_report = ValidationReport(is_valid=False, violations=[exception_violation])
                final_report = final_report.merge(exception_report)
        
        final_report.metadata["guardrail_type"] = self.__class__.__name__
        final_report.metadata["num_validators_executed"] = len(self.validators)
        return final_report

    def __repr__(self) -> str:
        return f"<CheckGuardrail validators=[{', '.join(v.__name__ for v in self.validators)}]>"