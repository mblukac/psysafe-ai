# psysafe/core/prompt.py
from typing import Any, Generic, Dict # Optional removed as not directly used in this snippet

# Core imports
from psysafe.core.base import GuardrailBase
from psysafe.core.models import GuardedRequest, ValidationReport, PromptRenderCtx
from psysafe.core.template import PromptTemplate

# Typing imports
from psysafe.typing.requests import RequestT
from psysafe.typing.responses import ResponseT


class PromptGuardrail(GuardrailBase[RequestT, ResponseT], Generic[RequestT, ResponseT]):
    """
    Guardrail that modifies requests by adding or transforming prompt instructions
    based on a provided template.
    """

    def __init__(self, template: PromptTemplate, template_variables: Dict[str, Any] = None):
        """
        Initializes a PromptGuardrail.

        Args:
            template: A PromptTemplate instance to be used for rendering.
            template_variables: A dictionary of variables that will be available
                                during template rendering, in addition to those
                                provided in PromptRenderCtx. These can be static
                                configurations for the guardrail.
        """
        self.template = template
        self.template_variables = template_variables or {}

    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """
        Applies the guardrail by rendering the prompt template and modifying the request.

        The actual modification of the request (e.g., adding a system message,
        prepending to user query) is highly dependent on the structure of `RequestT`.
        This implementation provides a basic structure; specific adapters or strategies
        might be needed for different request types.

        Args:
            request: The original request object.

        Returns:
            A GuardedRequest object containing the original and modified request.
        """
        # For demonstration, we assume RequestT might be a dictionary or an object
        # where prompts can be injected. This part will need to be more robust
        # and adaptable to various LLM client request structures.

        # Create a default PromptRenderCtx. In a real scenario,
        # this context would be more dynamically populated, perhaps by the driver
        # or a higher-level orchestrator.
        # For now, some fields might be placeholders.
        render_ctx = PromptRenderCtx(
            driver_type="unknown", # This should ideally come from the bound driver
            model_name="unknown",  # This should ideally come from the bound driver
            request_type="chat",   # Example, could be 'completion', etc.
            variables=self.template_variables # Pass guardrail-specific static vars
        )

        rendered_prompt = self.template.render(render_ctx)

        # --- Placeholder for request modification logic ---
        # This is the most complex part and highly dependent on RequestT.
        # Example: If RequestT is a dict for OpenAI chat:
        # modified_request_data = request.copy() # Assuming request is a dict
        # if "messages" in modified_request_data:
        #   modified_request_data["messages"].insert(0, {"role": "system", "content": rendered_prompt})
        # else:
        #   # Handle other request structures or raise an error
        #   pass # For now, we'll just pass the original request if not dict/messages
        #
        # For a generic approach, we might need a strategy pattern or expect
        # RequestT to have a specific method for modification.
        # For now, let's assume a simple pass-through or a very basic modification.
        # A more robust solution would involve request adapters.

        modified_request = request # Default to original if no modification logic implemented yet

        # Example of a simple modification if request is a dict and has 'prompt' key
        if isinstance(request, dict) and "prompt" in request:
            modified_request_data = request.copy()
            modified_request_data["prompt"] = rendered_prompt + "\n\n" + modified_request_data["prompt"]
            modified_request = modified_request_data
        elif isinstance(request, dict) and "messages" in request: # Basic OpenAI chat format
            modified_request_data = request.copy()
            # Ensure messages is a list and make a deep copy of it
            if isinstance(modified_request_data.get("messages"), list):
                modified_request_data["messages"] = [msg.copy() for msg in modified_request_data["messages"]]
            else:
                modified_request_data["messages"] = []

            # Check if a system message already exists
            has_system_message = False
            for message in modified_request_data["messages"]:
                if message.get("role") == "system":
                    # Append to existing system message or replace? For now, append.
                    message["content"] += "\n" + rendered_prompt
                    has_system_message = True
                    break
            if not has_system_message:
                modified_request_data["messages"].insert(0, {"role": "system", "content": rendered_prompt})
            modified_request = modified_request_data
        # --- End Placeholder ---

        return GuardedRequest[RequestT](
            original_request=request,
            modified_request=modified_request, # This would be the actual modified request
            metadata={
                "guardrail_type": self.__class__.__name__,
                "template_used": str(self.template.template_path) if self.template.template_path else "string_template",
                "rendered_prompt_snippet": rendered_prompt[:100] + "..." if len(rendered_prompt) > 100 else rendered_prompt
            }
        )

    def validate(self, response: ResponseT) -> ValidationReport:
        """
        PromptGuardrails typically do not validate responses themselves,
        as their primary role is to modify the outgoing request.
        Validation is usually handled by CheckGuardrails or other mechanisms.

        Args:
            response: The response object from the LLM.

        Returns:
            A ValidationReport indicating the response is valid (as this guardrail doesn't perform checks).
        """
        return ValidationReport(is_valid=True, violations=[], metadata={"guardrail_type": self.__class__.__name__})

    @classmethod
    def from_string(cls, prompt_text: str, template_variables: Dict[str, Any] = None) -> "PromptGuardrail[Any, Any]":
        """
        Factory method to create a PromptGuardrail directly from a prompt string.

        Args:
            prompt_text: The raw string to be used as the prompt template.
            template_variables: Optional static variables for the template.

        Returns:
            An instance of PromptGuardrail.
        """
        template = PromptTemplate.from_string(prompt_text)
        return cls(template=template, template_variables=template_variables)

    @classmethod
    def from_file(cls, template_file_path: str, template_variables: Dict[str, Any] = None) -> "PromptGuardrail[Any, Any]":
        """
        Factory method to create a PromptGuardrail from a template file.

        Args:
            template_file_path: Path to the file containing the prompt template.
            template_variables: Optional static variables for the template.

        Returns:
            An instance of PromptGuardrail.
        """
        template = PromptTemplate.from_file(template_file_path)
        return cls(template=template, template_variables=template_variables)

    def __repr__(self) -> str:
        return f"<PromptGuardrail template={self.template}>"