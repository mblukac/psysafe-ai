from typing import List, Optional, Any

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.core.models import GuardedRequest # GuardedRequest is a model
from psysafe.catalog import GuardrailCatalog


class ComplaintsHandlingGuardrail(PromptGuardrail[OpenAIChatRequest, Any]): # Added type hints
    """
    A guardrail to detect and categorize user complaints.
    """

    def __init__(self, escalation_keywords: Optional[List[str]] = None):
        """
        Initializes the ComplaintsHandlingGuardrail.

        Args:
            escalation_keywords: Optional list of keywords that trigger escalation.
                                 (Currently not used but planned for future customization)
        """
        template = PromptTemplate.from_file(
            "psysafe/catalog/complaints_handling/prompt.md"
        )
        super().__init__(template=template)
        self.escalation_keywords = escalation_keywords if escalation_keywords else []

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        """
        Applies the complaints handling guardrail to the request.

        It extracts the user input, renders a prompt to analyze the input for complaints,
        and prepends this analysis prompt (as a system message) to the original request.

        Args:
            request: The incoming OpenAI chat request.

        Returns:
            A GuardedRequest object containing the modified request.
        """
        if not request.get("messages"):
            # Or handle as an error, depending on expected behavior
            return GuardedRequest(original_request=request, modified_request=request, is_modified=False)

        # Assuming the last message is the user's current input
        user_input = ""
        messages = request.get("messages", [])
        if messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str):
            user_input = messages[-1].get("content")
        elif len(messages) > 1 and messages[-1].get("role") == "assistant" and messages[-2].get("role") == "user" and isinstance(messages[-2].get("content"), str):
            # If the last message is from assistant, take the one before
            user_input = messages[-2].get("content")
        # Add more sophisticated logic if needed to extract relevant user input

        render_ctx = PromptRenderCtx(
            driver_type=request.get("driver_type", "openai"), # Default to openai or get from request
            model_name=request.get("model", "unknown"),    # Get from request
            request_type="chat",  # Assuming chat type for OpenAIChatRequest
            variables={"user_input": user_input}
        )

        rendered_prompt = self.template.render(render_ctx)

        # Create a new system message with the rendered prompt
        system_message: OpenAIMessage = {"role": "system", "content": rendered_prompt}

        # Prepend the system message to the request's messages
        modified_messages = [system_message] + messages
        modified_request_data = request.copy()
        modified_request_data["messages"] = modified_messages
        # Ensure all keys from original request are present, then update messages
        modified_request: OpenAIChatRequest = {**request, "messages": modified_messages} # type: ignore

        return GuardedRequest(original_request=request, modified_request=modified_request, is_modified=True)


# Register the guardrail with the catalog
GuardrailCatalog.register("complaints_handling", ComplaintsHandlingGuardrail)