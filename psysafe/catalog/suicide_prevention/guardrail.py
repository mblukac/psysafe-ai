from enum import Enum
from typing import Dict, Any, List, Optional
import json
import logging # Added import

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.core.models import GuardedRequest, Conversation, Message, CheckOutput # GuardedRequest is a model
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.catalog import GuardrailCatalog
from utils.llm_utils import parse_llm_response, LLMResponseParseError # Added import

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
        self.logger = logging.getLogger(__name__) # Added logger

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

    def _format_conversation_for_apply(self, conversation: Conversation) -> OpenAIChatRequest:
        """
        Formats a Conversation object into the OpenAIChatRequest structure
        expected by the apply method.
        """
        messages_for_llm: List[Dict[str, str]] = []
        for msg in conversation.messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})
        
        # The apply method expects a dict-like structure for OpenAIChatRequest
        # We only need 'messages' for the current implementation of apply.
        # Other fields like 'model' might be needed if apply evolves.
        return {"messages": messages_for_llm} # type: ignore

    def check(self, conversation: Conversation) -> CheckOutput:
        """
        Checks a conversation for suicide risk using the bound LLM driver.

        Args:
            conversation: The Conversation object to check.

        Returns:
            A CheckOutput object with the assessment results.
        """
        if not self.driver:
            raise RuntimeError(
                "LLM driver not bound. Please call `guardrail.bind(driver)` before using `check`."
            )

        # 1. Format the input Conversation for the guardrail's apply method
        llm_request_input = self._format_conversation_for_apply(conversation)

        # 2. Apply the guardrail to get the modified request (with system prompt)
        guarded_request = self.apply(llm_request_input)
        
        # The modified_request from apply() is already in the correct format for the driver
        # if the driver expects an OpenAIChatRequest-like dictionary.
        modified_llm_request = guarded_request.modified_request

        # 3. Send to LLM via the driver
        #    We assume the driver has a method like `chat_completion` or similar
        #    that takes the request and returns a response object with a 'content' attribute.
        #    This part is highly dependent on the actual driver interface.
        #    For now, let's assume a `driver.chat_completion(request_payload)` method
        #    that returns a response from which we can extract text content.
        
        raw_llm_response_content: Optional[str] = None
        llm_errors: List[str] = []
        llm_metadata: Dict[str, Any] = {}

        try:
            # This is a placeholder for the actual driver call.
            # The exact method and response structure will depend on the driver.
            # Example: response = self.driver.chat.completions.create(**modified_llm_request)
            # raw_llm_response_content = response.choices[0].message.content
            
            # For now, let's assume the driver has a method `invoke` or `call`
            # and returns an object with a `content` attribute or similar.
            # This needs to be adapted to the actual driver API.
            # OpenAIChatDriver has a `send` method.
            if hasattr(self.driver, 'send'):
                # The `send` method of OpenAIChatDriver expects an OpenAIChatRequest (dict)
                # and returns an OpenAIChatResponse (dict).
                # The response structure is like:
                # {
                #   "id": "chatcmpl-...",
                #   "object": "chat.completion",
                #   "created": 1677652288,
                #   "model": "gpt-3.5-turbo-0613",
                #   "choices": [{
                #     "index": 0,
                #     "message": {
                #       "role": "assistant",
                #       "content": "\n\nHello there, how may I assist you today?",
                #     },
                #     "finish_reason": "stop"
                #   }],
                #   "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
                # }
                llm_response_dict = self.driver.send(modified_llm_request) # type: ignore
                
                if llm_response_dict and llm_response_dict.get("choices"):
                    first_choice = llm_response_dict["choices"][0]
                    if first_choice and first_choice.get("message"):
                        raw_llm_response_content = first_choice["message"].get("content")
                
                if not raw_llm_response_content:
                    llm_errors.append("Could not extract content from LLM response.")
                    llm_metadata["raw_llm_response_dict"] = llm_response_dict # Store for debugging
                    # If content extraction fails, return immediately with error
                    return CheckOutput(
                        is_triggered=False,
                        errors=llm_errors,
                        raw_llm_response=raw_llm_response_content, # Should be None or empty if extraction failed
                        metadata=llm_metadata
                    )
            else:
                llm_errors.append(f"Bound driver of type {type(self.driver).__name__} does not have a 'send' method for LLM calls.")
                return CheckOutput(is_triggered=False, errors=llm_errors, metadata={"info": "LLM call not possible with this driver.", **llm_metadata})

        except Exception as e:
            llm_errors.append(f"Error during LLM call: {str(e)}")
            return CheckOutput(is_triggered=False, errors=llm_errors, raw_llm_response=str(e), metadata=llm_metadata)

        if not raw_llm_response_content: # This check might be redundant if the above return is hit, but good for safety
            llm_errors.append("LLM response content was empty or could not be extracted after driver call.")
            # Ensure raw_llm_response is the actual (empty) content
            return CheckOutput(is_triggered=False, errors=llm_errors, raw_llm_response=raw_llm_response_content, metadata=llm_metadata)

        # 4. Parse LLM response using the new utility
        try:
            # Pass the logger to the parsing utility
            llm_output = parse_llm_response(raw_llm_response_content, logger=self.logger)
            llm_metadata["parsed_response_type"] = type(llm_output).__name__

            # The parse_llm_response function is expected to return a dict.
            # Additional checks for dict type might be redundant if the function guarantees it,
            # but can be kept for robustness if needed.
            # For now, assuming parse_llm_response handles non-dict JSON correctly by raising an error or as per its spec.

            # Attempt to extract and process the risk value
            raw_risk_from_llm = llm_output.get("risk") # Key from LLM is 'risk'
            
            final_risk_score: Optional[float] = None
            final_is_triggered: bool = False

            if raw_risk_from_llm is not None:
                try:
                    # Convert to float. Handles int, float, and string representations of numbers.
                    numeric_value = float(raw_risk_from_llm)
                    final_risk_score = numeric_value
                    # Determine if triggered based on the numeric risk score.
                    # Assuming risk score > 0 indicates the guardrail is triggered.
                    if numeric_value > 0:
                        final_is_triggered = True
                except (ValueError, TypeError):
                    # If 'risk' is present but not a valid number, log it and default
                    llm_errors.append(f"LLM output contained a non-numeric 'risk' value: {raw_risk_from_llm}")
                    # final_is_triggered remains False, final_risk_score remains None

            # Construct details dictionary
            current_details = {
                "reasoning": llm_output.get("reasoning", "N/A"),
                "confidence_level": llm_output.get("confidence_level", "N/A"),
                "parsed_llm_output": llm_output # Store the parsed dict
            }
            llm_metadata["constructed_details_preview"] = str(current_details)[:200]

            return CheckOutput(
                is_triggered=final_is_triggered,
                risk_score=final_risk_score,
                details=current_details,
                raw_llm_response=raw_llm_response_content,
                errors=llm_errors, # Include any new errors from risk parsing
                metadata=llm_metadata
            )
        except LLMResponseParseError as e:
            self.logger.error(f"LLMResponseParseError in check method: {e.message}, Raw Response: {e.raw_response[:200]}")
            llm_errors.append(f"Failed to parse LLM response: {e.message}. Response snippet: {e.raw_response[:500] if e.raw_response else 'N/A'}")
            return CheckOutput(
                is_triggered=False, # Default to not triggered on parse error
                errors=llm_errors,
                raw_llm_response=e.raw_response, # Use raw response from exception
                metadata={"warning": "Could not parse LLM output.", "parser_error_type": type(e).__name__, **llm_metadata}
            )
        except Exception as e: # Catch any other unexpected errors during this stage
            self.logger.error(f"Unexpected error processing LLM response after parsing attempt: {str(e)}")
            llm_errors.append(f"Unexpected error processing LLM response: {str(e)}. Response snippet: {raw_llm_response_content[:500] if raw_llm_response_content else 'N/A'}")
            return CheckOutput(
                is_triggered=False,
                errors=llm_errors,
                raw_llm_response=raw_llm_response_content,
                metadata={"warning": "Unexpected error during LLM response processing.", "error_type": type(e).__name__, **llm_metadata}
            )

GuardrailCatalog.register("suicide_prevention", SuicidePreventionGuardrail)