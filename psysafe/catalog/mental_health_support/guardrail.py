from typing import Any # For ResponseT type hint
from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.core.models import GuardedRequest, PromptRenderCtx
from psysafe.typing.requests import OpenAIChatRequest
from psysafe.catalog import GuardrailCatalog

class MentalHealthSupportGuardrail(PromptGuardrail[OpenAIChatRequest, Any]): # Added type hints
    """
    A guardrail that modifies requests to include a system prompt for mental health support.
    """
    def __init__(self):
        template = PromptTemplate.from_file("psysafe/catalog/mental_health_support/prompt.md")
        super().__init__(template=template)

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        """
        Applies the mental health support prompt to the request.
        """
        user_input_context = ""  # Placeholder, adapt as needed for actual context extraction
        # Attempt to get the last user message as context
        messages = request.get("messages", [])
        if messages and messages[-1].get("role") == "user":
            content = messages[-1].get("content")
            if isinstance(content, str):
                user_input_context = content
            # If content is a list (e.g. for vision models), try to find text parts
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_input_context = item.get("text", "")
                        break


        render_ctx = PromptRenderCtx(
            driver_type=request.get("driver_type", "openai"),
            model_name=request.get("model", "unknown"),
            request_type="chat",
            variables={
                "user_input_context": user_input_context
            }
        )
        rendered_prompt = self.template.render(render_ctx)

        # Insert the rendered prompt as a system message
        # We'll add it as the first message if no system message exists,
        # or append to an existing one. For simplicity, let's prepend.
        system_message_dict = {"role": "system", "content": rendered_prompt}
        modified_messages = [system_message_dict] + messages

        # Create a new request object
        modified_request_data = request.copy() # Assuming OpenAIChatRequest is a dict
        modified_request_data["messages"] = modified_messages
        modified_request: OpenAIChatRequest = {**request, "messages": modified_messages} # type: ignore

        return GuardedRequest(
            original_request=request,
            modified_request=modified_request,
            is_modified=True,
            applied_guardrails=[self.name if hasattr(self, 'name') else "MentalHealthSupportGuardrail"]
        )

GuardrailCatalog.register("mental_health_support", MentalHealthSupportGuardrail)