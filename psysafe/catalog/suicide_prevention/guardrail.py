from enum import Enum
from typing import Dict, Any

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.core.models import GuardedRequest # GuardedRequest is a model
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.catalog import GuardrailCatalog

class Sensitivity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

RISK_INDICATORS = """
- Direct statements of intent ("I want to die", "I'm going to kill myself")
- Indirect statements ("I can't go on", "Life isn't worth living")
- Talking or writing about death or suicide
- Expressing feelings of hopelessness, helplessness, or worthlessness
- Increased substance use
- Withdrawal from friends, family, and activities
- Giving away prized possessions
- Making a plan or researching methods
- Sudden mood changes or improvement after a period of depression
- Previous suicide attempts
"""

LOW_SENSITIVITY = """
Focus on explicit, unambiguous statements of suicidal intent or planning.
Avoid flagging figurative language, expressions of frustration, or general distress
unless accompanied by clear suicidal indicators.
"""

MEDIUM_SENSITIVITY = """
Balance detection of clear suicidal statements with sensitivity to strong
indirect indicators. Flag expressions of severe hopelessness, helplessness,
or worthlessness if they suggest a significant risk. Be cautious with
ambiguous language but err on the side of safety if context is concerning.
"""

HIGH_SENSITIVITY = """
Prioritise safety by flagging any language that could potentially indicate
suicidal ideation, even if indirect or ambiguous. This includes expressions
of severe distress, hopelessness, or talk of being a burden, even without
explicit suicidal statements. Interpret any doubt in favour of potential risk.
"""

class SuicidePreventionGuardrail(PromptGuardrail[OpenAIChatRequest, Any]): # Added Any for ResponseT
    def __init__(
        self,
        sensitivity: Sensitivity,
        reasoning: bool = True,
        confidence: bool = False,
    ):
        template = PromptTemplate.from_file(
            "psysafe/catalog/suicide_prevention/prompt.md"
        )
        super().__init__(template=template)
        self.sensitivity = sensitivity
        self.reasoning = reasoning
        self.confidence = confidence

    def _get_sensitivity_block_text(self) -> str:
        if self.sensitivity == Sensitivity.LOW:
            return LOW_SENSITIVITY
        elif self.sensitivity == Sensitivity.MEDIUM:
            return MEDIUM_SENSITIVITY
        elif self.sensitivity == Sensitivity.HIGH:
            return HIGH_SENSITIVITY
        # Should not happen due to enum typing, but as a fallback
        raise ValueError(f"Unknown sensitivity level: {self.sensitivity}")

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        # Extract user context - assuming messages are in chronological order
        # and we want to analyze all user messages.
        # This might need adjustment based on how user_context is expected to be formatted.
        # For now, concatenating content of all 'user' role messages.
        user_messages_content = [
            msg.get("content") for msg in request.get("messages", []) if msg.get("role") == "user" and msg.get("content")
        ]
        user_context = "\n".join(user_messages_content)

        sensitivity_block_text = self._get_sensitivity_block_text()

        # Assuming driver_type and model_name might be available in request.model or request.extras
        # For now, providing placeholders if not directly available.
        driver_type = request.get("driver_type", "openai")
        model_name = request.get("model", "unknown")

        render_ctx = PromptRenderCtx(
            driver_type=driver_type,
            model_name=model_name,
            request_type="chat",
            variables={
                "user_context": user_context,
                "risk_indicators_text": RISK_INDICATORS,
                "sensitivity_block_text": sensitivity_block_text,
                "reasoning": self.reasoning,
                "confidence": self.confidence,
            }
        )

        rendered_prompt = self.template.render(render_ctx)

        # Create a new list of messages with the system prompt inserted at the beginning
        # or prepended to an existing system message.
        # For simplicity, inserting as the first message.
        # More sophisticated logic might be needed to merge with existing system prompts.
        modified_messages = [{"role": "system", "content": rendered_prompt}] + request.get("messages", []) # Ensure dicts

        # Create a new request object with the modified messages
        # Assuming OpenAIChatRequest can be instantiated or copied this way
        # This might need to be request.copy(update={"messages": modified_messages})
        # or similar depending on Pydantic model behavior.
        
        # Create a new request object or modify in place if mutable
        # For immutability, it's better to create a new one.
        # If OpenAIChatRequest is a Pydantic model, model_copy is preferred for deep copies.
        
        # Simplest approach: create a new instance if all fields are known or can be copied
        # This assumes OpenAIChatRequest can be created by passing all its fields.
        # A more robust way would be to use model_copy if it's a Pydantic model.
        
        # Let's assume request.messages is mutable for now for simplicity,
        # though creating a new request object is generally safer.
        # request.messages.insert(0, Message(role="system", content=rendered_prompt))
        # guarded_request_payload = request

        # A safer way with Pydantic models:
        modified_request_data = request.copy() # Assuming request is a dict
        modified_request_data["messages"] = modified_messages # Already dicts
        
        # Re-create the request object. This assumes OpenAIChatRequest can be created from its dict representation.
        # This part is tricky without knowing the exact structure and Pydantic version.
        # Let's try to modify the original request's messages list if it's mutable,
        # or create a new request if not.
        # For now, let's assume we create a new request object.
        # This might require all fields of OpenAIChatRequest.
        
        # Create a new list of messages for the new request
        # The system prompt should be the first message.
        # Create a new list of messages for the new request
        # The system prompt should be the first message.
        final_messages = [{"role": "system", "content": rendered_prompt}]
        for msg_dict in request.get("messages", []): # Assuming request.messages are already dicts
            # Avoid duplicating system messages if the original request already had one.
            # This simple logic just prepends; a more robust solution might merge system prompts.
            final_messages.append(msg_dict.copy()) # Create a copy of the dict

        # Create a new request object
        # We need to ensure all required fields of OpenAIChatRequest are passed.
        # If OpenAIChatRequest has other fields, they need to be copied from the original request.
        
        # Using model_copy for a safer update if OpenAIChatRequest is a Pydantic model
        updated_request_dict = request.copy() # Assuming request is a dict
        updated_request_dict['messages'] = final_messages # Already list of dicts
        
        # Create a new OpenAIChatRequest instance from the dictionary
        # This assumes OpenAIChatRequest can be initialized this way.
        # If not, a more specific way to construct it is needed.
        # For example, if it has a .construct or .parse_obj method.
        
        # Let's assume OpenAIChatRequest can be created by unpacking the dict.
        # This is a common pattern for Pydantic models.
        guarded_request_payload: OpenAIChatRequest = {**request, "messages": final_messages} # type: ignore


        # The GuardedRequest model expects the modified request directly
        return GuardedRequest(
            original_request=request, # Pass the original request
            modified_request=guarded_request_payload,
            is_modified=True, # Indicate that the request was modified
            applied_guardrails=[self.name if hasattr(self, 'name') else "SuicidePreventionGuardrail"], # self.name should be defined
            metadata={"original_request_hash": hash(str(request))} # Example meta, ensure request is hashable
        )

GuardrailCatalog.register("suicide_prevention", SuicidePreventionGuardrail)