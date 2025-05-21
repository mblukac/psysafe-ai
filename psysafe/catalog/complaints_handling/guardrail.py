from typing import List, Optional, Any, Dict
import logging

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.core.models import GuardedRequest, Conversation, CheckOutput # GuardedRequest is a model
from psysafe.catalog import GuardrailCatalog
from utils.llm_utils import parse_llm_response, LLMResponseParseError


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
        self.logger = logging.getLogger(__name__)

    def _get_user_input_from_request(self, request: OpenAIChatRequest) -> str:
        """Extracts user input from the request messages."""
        user_input = ""
        messages = request.get("messages", [])
        if not messages:
            return ""

        # Try to get the last user message
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                user_input = message.get("content")
                break
        
        # Fallback: if last is assistant, try message before that if it's user
        if not user_input and len(messages) > 1:
            if messages[-1].get("role") == "assistant" and \
               messages[-2].get("role") == "user" and \
               isinstance(messages[-2].get("content"), str):
                user_input = messages[-2].get("content")
        return user_input

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
            return GuardedRequest(original_request=request, modified_request=request, is_modified=False, applied_guardrails=[self.name])

        user_input = self._get_user_input_from_request(request)
        if not user_input: # If no user input could be extracted
             return GuardedRequest(original_request=request, modified_request=request, is_modified=False, applied_guardrails=[self.name])


        render_ctx = PromptRenderCtx(
            driver_type=request.get("driver_type", "openai"),
            model_name=request.get("model", "unknown"),
            request_type="chat",
            variables={"user_input": user_input}
        )

        rendered_prompt = self.template.render(render_ctx)
        system_message: OpenAIMessage = {"role": "system", "content": rendered_prompt}
        
        original_messages = request.get("messages", [])
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
        
        # The apply method expects a dict-like structure for OpenAIChatRequest.
        # We primarily need 'messages'. Other fields like 'model' or 'driver_type'
        # can be added if the apply method evolves to use them directly from here.
        # For now, apply extracts them from the request passed to it.
        return {"messages": messages_for_llm} # type: ignore

    def check(self, conversation: Conversation) -> CheckOutput:
        """
        Checks a conversation for complaints using the bound LLM driver.

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
        guarded_request = self.apply(llm_request_input)
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
                    return CheckOutput(
                        is_triggered=False, errors=llm_errors, raw_llm_response=raw_llm_response_content, metadata=llm_metadata
                    )
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

            complaint_detected = llm_output.get("complaint_detected", False)
            # Ensure boolean, default to False if not a proper boolean
            if not isinstance(complaint_detected, bool):
                self.logger.warning(f"LLM returned non-boolean 'complaint_detected': {complaint_detected}. Defaulting to False.")
                llm_errors.append(f"LLM returned non-boolean 'complaint_detected': {complaint_detected}")
                complaint_detected = False


            category = llm_output.get("category", "N/A")
            summary = llm_output.get("summary", "N/A")
            escalation_needed = llm_output.get("escalation_needed", False)
            if not isinstance(escalation_needed, bool):
                self.logger.warning(f"LLM returned non-boolean 'escalation_needed': {escalation_needed}. Defaulting to False.")
                llm_errors.append(f"LLM returned non-boolean 'escalation_needed': {escalation_needed}")
                escalation_needed = False


            current_details = {
                "category": category,
                "summary": summary,
                "escalation_needed": escalation_needed,
                "parsed_llm_output": llm_output
            }
            llm_metadata["constructed_details_preview"] = str(current_details)[:200]

            return CheckOutput(
                is_triggered=bool(complaint_detected), # Explicitly cast to bool
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

# Register the guardrail with the catalog
GuardrailCatalog.register("complaints_handling", ComplaintsHandlingGuardrail)