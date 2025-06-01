# psysafe/core/composite.py
from typing import List, Generic

# Core imports
from psysafe.core.base import GuardrailBase
from psysafe.core.models import GuardedRequest, ValidationReport

# Typing imports
from psysafe.typing.requests import RequestT
from psysafe.typing.responses import ResponseT


class CompositeGuardrail(GuardrailBase[RequestT, ResponseT], Generic[RequestT, ResponseT]):
    """
    A guardrail that composes multiple guardrails, applying them sequentially.
    The `apply` method chains request modifications, and the `validate` method
    merges validation reports from all composed guardrails.
    """

    def __init__(self, guardrails: List[GuardrailBase[RequestT, ResponseT]]):
        """
        Initializes a CompositeGuardrail.

        Args:
            guardrails: A list of GuardrailBase instances to be composed.
                        They will be applied in the order they appear in this list.
        """
        if not guardrails:
            raise ValueError("CompositeGuardrail requires at least one guardrail.")
        if not all(isinstance(g, GuardrailBase) for g in guardrails):
            raise TypeError("All items in 'guardrails' must be instances of GuardrailBase.")
        self.guardrails = guardrails

    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """
        Applies all composed guardrails sequentially to the request.
        The `modified_request` from one guardrail becomes the input `request`
        for the next. The `original_request` in the final GuardedRequest
        refers to the initial request passed to the composite.

        Args:
            request: The initial request object.

        Returns:
            A GuardedRequest object where `original_request` is the initial request
            and `modified_request` is the result after all guardrails have been applied.
            Metadata from all guardrails is collected (though a merging strategy might be needed).
        """
        current_request_data = request
        cumulative_metadata = {"composite_guardrail_sequence": [g.__class__.__name__ for g in self.guardrails]}
        
        # The 'original_request' for the final GuardedRequest from the composite
        # should be the very first request it received.
        initial_original_request = request 

        for i, guardrail_instance in enumerate(self.guardrails):
            # The input to each subsequent guardrail's apply method is the
            # modified_request from the previous one.
            guarded_step_result = guardrail_instance.apply(current_request_data)
            current_request_data = guarded_step_result.modified_request
            
            # Collect metadata - simple update, might need more sophisticated merging
            # Prefixing with guardrail name/index could be an option.
            cumulative_metadata[f"step_{i}_{guardrail_instance.__class__.__name__}"] = guarded_step_result.metadata

        return GuardedRequest[RequestT](
            original_request=initial_original_request, # The very first request
            modified_request=current_request_data,    # The final modified request
            metadata=cumulative_metadata
        )

    def validate(self, response: ResponseT) -> ValidationReport:
        """
        Validates the response by applying all composed guardrails sequentially.
        Validation reports from each guardrail are merged.

        Args:
            response: The response object from the LLM.

        Returns:
            A single ValidationReport aggregating results from all composed guardrails.
        """
        final_report = ValidationReport(is_valid=True) # Start with a clean, empty report

        for guardrail_instance in self.guardrails:
            try:
                report_from_guardrail = guardrail_instance.validate(response)
                final_report = final_report.merge(report_from_guardrail)
            except Exception as e:
                # Similar to CheckGuardrail, handle exceptions from individual guardrails
                # during validation.
                from psysafe.core.models import Violation # Local import to avoid circularity if models grow
                exception_violation = Violation(
                    severity="ERROR", # Assuming ValidationSeverity.ERROR exists
                    code="COMPOSITE_VALIDATION_EXCEPTION",
                    message=f"Guardrail {guardrail_instance.__class__.__name__} in composite raised an exception during validation: {str(e)}",
                    context={"guardrail_name": guardrail_instance.__class__.__name__, "exception_type": e.__class__.__name__}
                )
                exception_report = ValidationReport(is_valid=False, violations=[exception_violation])
                final_report = final_report.merge(exception_report)
        
        # Add composite-specific metadata after all children have been merged
        final_report.metadata["guardrail_type"] = self.__class__.__name__
        final_report.metadata["num_composed_guardrails"] = len(self.guardrails)
        return final_report

    def __repr__(self) -> str:
        return f"<CompositeGuardrail guardrails=[{', '.join(g.__class__.__name__ for g in self.guardrails)}]>"