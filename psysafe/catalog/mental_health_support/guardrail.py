from typing import Any, List, Dict, Optional
import logging

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.core.models import GuardedRequest, PromptRenderCtx, Conversation, Message, CheckOutput
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.catalog import GuardrailCatalog
from utils.llm_utils import parse_llm_response, LLMResponseParseError

class MentalHealthSupportGuardrail(PromptGuardrail[OpenAIChatRequest, Any]):
    """
    A guardrail to detect signs of mental health distress in user input
    and assess if suggesting professional help is warranted.
    """
    def __init__(self):
        template = PromptTemplate.from_file("psysafe/catalog/mental_health_support/prompt.md")
        super().__init__(template=template)
        self.logger = logging.getLogger(__name__)
        # Ensure the name attribute is set, typically done by the catalog or base class
        if not hasattr(self, 'name'):
            self.name = "mental_health_support"


    def _extract_user_input_context(self, request: OpenAIChatRequest) -> str:
        """
        Extracts a consolidated string of user messages from the request.
        This might need to be more sophisticated based on how much history is relevant.
        For now, concatenates all user messages.
        """
        user_messages_content: List[str] = []
        messages = request.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    user_messages_content.append(content)
                elif isinstance(content, list): # Handle multi-modal content if text part exists
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_messages_content.append(item.get("text", ""))
        return "\n".join(user_messages_content)

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        """
        Applies the mental health support analysis prompt to the request.
        This method prepares the request for the LLM call made by the `check` method.
        """
        user_input_context = self._extract_user_input_context(request)

        if not user_input_context: # If no user input, don't modify
            return GuardedRequest(
                original_request=request,
                modified_request=request,
                is_modified=False,
                applied_guardrails=[self.name]
            )

        render_ctx = PromptRenderCtx(
            driver_type=request.get("driver_type", "openai"),
            model_name=request.get("model", "unknown"),
            request_type="chat",
            variables={"user_input_context": user_input_context}
        )
        rendered_prompt = self.template.render(render_ctx)

        system_message: OpenAIMessage = {"role": "system", "content": rendered_prompt}
        
        original_messages = request.get("messages", [])
        # Prepend the system message for analysis
        modified_messages = [system_message] + original_messages
        
        modified_request: OpenAIChatRequest = {**request, "messages": modified_messages}

        return GuardedRequest(
            original_request=request,
            modified_request=modified_request,
            is_modified=True,
            applied_guardrails=[self.name]
        )

    def _format_conversation_for_apply(self, conversation: Conversation) -> OpenAIChatRequest:
        """
        Formats a Conversation object into the OpenAIChatRequest structure
        expected by the apply method.
        """
        messages_for_llm: List[Dict[str, str]] = []
        for msg in conversation.messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})
        
        # Include other necessary fields if apply() starts depending on them from this formatted request
        return {"messages": messages_for_llm} # type: ignore


    def check(self, conversation: Conversation) -> CheckOutput:
        """
        Checks a conversation for mental health distress using the bound LLM driver.

        Args:
            conversation: The Conversation object to check.

        Returns:
            A CheckOutput object with the assessment results.
        """
        if not self.driver:
            raise RuntimeError(
                "LLM driver not bound. Please call `guardrail.bind(driver)` before using `check`."
            )

        llm_request_input = self._format_conversation_for_apply(conversation)
        guarded_request = self.apply(llm_request_input) # apply() now prepares the request for this check
        modified_llm_request = guarded_request.modified_request

        raw_llm_response_content: Optional[str] = None
        llm_errors: List[str] = []
        llm_metadata: Dict[str, Any] = {}

        try:
            if hasattr(self.driver, 'send'):
                llm_response_dict = self.driver.send(modified_llm_request) # type: ignore
                if llm_response_dict and llm_response_dict.get("choices"):
                    first_choice = llm_response_dict["choices"][0]
                    if first_choice and first_choice.get("message"):
                        raw_llm_response_content = first_choice["message"].get("content")
                
                if not raw_llm_response_content:
                    llm_errors.append("Could not extract content from LLM response.")
                    llm_metadata["raw_llm_response_dict"] = llm_response_dict
                    return CheckOutput(is_triggered=False, errors=llm_errors, raw_llm_response=raw_llm_response_content, metadata=llm_metadata)
            else:
                llm_errors.append(f"Bound driver of type {type(self.driver).__name__} does not have a 'send' method.")
                return CheckOutput(is_triggered=False, errors=llm_errors, metadata={"info": "LLM call not possible.", **llm_metadata})

        except Exception as e:
            self.logger.error(f"Error during LLM call: {str(e)}", exc_info=True)
            llm_errors.append(f"Error during LLM call: {str(e)}")
            return CheckOutput(is_triggered=False, errors=llm_errors, raw_llm_response=str(e), metadata=llm_metadata)

        if not raw_llm_response_content:
            llm_errors.append("LLM response content was empty after driver call.")
            return CheckOutput(is_triggered=False, errors=llm_errors, raw_llm_response=raw_llm_response_content, metadata=llm_metadata)

        try:
            llm_output = parse_llm_response(raw_llm_response_content, logger=self.logger)
            llm_metadata["parsed_response_type"] = type(llm_output).__name__

            distress_level = llm_output.get("distress_level", "none")
            key_phrases = llm_output.get("key_phrases_detected", [])
            concerns = llm_output.get("concerns_identified", [])
            suggestion_needed = llm_output.get("suggestion_needed", False)
            summary = llm_output.get("summary", "N/A")

            if not isinstance(suggestion_needed, bool):
                self.logger.warning(f"LLM returned non-boolean 'suggestion_needed': {suggestion_needed}. Defaulting to False.")
                llm_errors.append(f"LLM returned non-boolean 'suggestion_needed': {suggestion_needed}")
                suggestion_needed = False
            
            # The guardrail is "triggered" if suggestion_needed is true.
            is_triggered = bool(suggestion_needed)

            current_details = {
                "distress_level": distress_level,
                "key_phrases_detected": key_phrases,
                "concerns_identified": concerns,
                "suggestion_needed": suggestion_needed, # This is the primary trigger indicator
                "summary": summary,
                "parsed_llm_output": llm_output
            }
            llm_metadata["constructed_details_preview"] = str(current_details)[:200]

            return CheckOutput(
                is_triggered=is_triggered,
                details=current_details,
                raw_llm_response=raw_llm_response_content,
                errors=llm_errors,
                metadata=llm_metadata
            )
        except LLMResponseParseError as e:
            self.logger.error(f"LLMResponseParseError in check method: {e.message}, Raw Response: {e.raw_response[:200]}", exc_info=True)
            llm_errors.append(f"Failed to parse LLM response: {e.message}. Snippet: {e.raw_response[:100] if e.raw_response else 'N/A'}")
            return CheckOutput(
                is_triggered=False, errors=llm_errors, raw_llm_response=e.raw_response,
                metadata={"warning": "Could not parse LLM output.", "parser_error_type": type(e).__name__, **llm_metadata}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error processing LLM response: {str(e)}", exc_info=True)
            llm_errors.append(f"Unexpected error processing LLM response: {str(e)}. Snippet: {raw_llm_response_content[:100] if raw_llm_response_content else 'N/A'}")
            return CheckOutput(
                is_triggered=False, errors=llm_errors, raw_llm_response=raw_llm_response_content,
                metadata={"warning": "Unexpected error during LLM response processing.", "error_type": type(e).__name__, **llm_metadata}
            )

GuardrailCatalog.register("mental_health_support", MentalHealthSupportGuardrail)