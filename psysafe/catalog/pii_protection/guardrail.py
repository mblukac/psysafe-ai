from typing import Dict, Any, List

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.core.models import GuardedRequest
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.catalog import GuardrailCatalog


class PiiProtectionGuardrail(PromptGuardrail[OpenAIChatRequest, Any]): # Added type hints
    """
    A guardrail to detect and prevent the leakage of Personally Identifiable Information (PII).
    It uses a predefined prompt to instruct the LLM on how to handle PII.
    """

    def __init__(self): # Removed **kwargs as they are not used for specific PII config
        template = PromptTemplate.from_file(
            "psysafe/catalog/pii_protection/prompt.md"
        )
        super().__init__(template=template)

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        """
        Applies the PII protection prompt to the incoming request.

        Args:
            request: The original OpenAI chat request.

        Returns:
            A GuardedRequest object containing the modified request.
        """
        messages: List[OpenAIMessage] = request.get("messages", [])
        user_input_context = "\n".join(
            [f"{msg.get('role')}: {msg.get('content')}" for msg in messages]
        )

        render_ctx = PromptRenderCtx(
            driver_type=request.get("meta", {}).get("driver_type", "openai"), # type: ignore
            model_name=request.get("meta", {}).get("model_name", "unknown"), # type: ignore
            request_type="chat",
            variables={"user_input_context": user_input_context}
        )

        rendered_prompt = self.template.render(render_ctx)

        modified_messages: List[OpenAIMessage] = []
        system_message_inserted = False

        # Prepend to existing system message or insert a new one
        for message in messages:
            if message.get("role") == "system":
                original_content = message.get("content", "")
                modified_content = f"{rendered_prompt}\n\n{original_content}".strip()
                modified_messages.append(
                    {"role": "system", "content": modified_content}
                )
                system_message_inserted = True
            else:
                modified_messages.append(message)

        if not system_message_inserted:
            modified_messages.insert(
                0, {"role": "system", "content": rendered_prompt}
            )
        
        # Create a new request object or modify a copy
        # Modifying a copy is safer for TypedDicts
        modified_request_data = request.copy()
        modified_request_data["messages"] = modified_messages

        # Ensure the modified_request_data conforms to OpenAIChatRequest structure
        # This might involve casting or careful construction if strict type checking is enforced elsewhere.
        # For now, we assume direct update is fine for the TypedDict.
        modified_request: OpenAIChatRequest = {**request, "messages": modified_messages} # type: ignore

        return GuardedRequest(original_request=request, modified_request=modified_request, is_modified=True)


GuardrailCatalog.register("pii_protection", PiiProtectionGuardrail)