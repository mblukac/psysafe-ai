from typing import Dict, Any, List, Optional
import logging

from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.core.models import GuardedRequest, Conversation, Message, CheckOutput
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage
from psysafe.catalog import GuardrailCatalog
from utils.llm_utils import parse_llm_response, LLMResponseParseError


class PiiProtectionGuardrail(PromptGuardrail[OpenAIChatRequest, Any]):
    """
    A guardrail to detect Personally Identifiable Information (PII) in user input.
    It uses a predefined prompt to instruct an LLM to analyze the input and return a JSON object.
    """

    def __init__(self):
        template = PromptTemplate.from_file(
            "psysafe/catalog/pii_protection/prompt.md"
        )
        super().__init__(template=template)
        self.logger = logging.getLogger(__name__)
        if not hasattr(self, 'name'): # Ensure name attribute is set
            self.name = "pii_protection"

    def _extract_user_input_context(self, request: OpenAIChatRequest) -> str:
        """
        Extracts a consolidated string of user messages from the request.
        """
        user_messages_content: List[str] = []
        messages = request.get("messages", [])
        for msg in messages:
            # We are interested in analyzing all text content, regardless of role,
            # as PII could be in assistant responses too if not careful.
            # However, the prompt is geared towards "User Input/Context".
            # For PII detection, it's often best to analyze the most recent user turn,
            # or a configurable window of conversation.
            # For now, let's take all message content.
            content = msg.get("content")
            if isinstance(content, str):
                user_messages_content.append(f"{msg.get('role', 'unknown')}: {content}")
            elif isinstance(content, list): # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        user_messages_content.append(f"{msg.get('role', 'unknown')}: {item.get('text', '')}")
        return "\n".join(user_messages_content)

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        """
        Applies the PII detection analysis prompt to the request.
        This method prepares the request for the LLM call made by the `check` method.
        """
        user_input_context = self._extract_user_input_context(request)

        if not user_input_context:
            return GuardedRequest(
                original_request=request, modified_request=request, is_modified=False, applied_guardrails=[self.name]
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
        modified_messages = [system_message] + original_messages # Prepend system message for analysis
        
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
        return {"messages": messages_for_llm} # type: ignore

    def check(self, conversation: Conversation) -> CheckOutput:
        """
        Checks a conversation for PII using the bound LLM driver.

        Args:
            conversation: The Conversation object to check.

        Returns:
            A CheckOutput object with the PII detection results.
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

            pii_detected = llm_output.get("pii_detected", False)
            if not isinstance(pii_detected, bool):
                self.logger.warning(f"LLM returned non-boolean 'pii_detected': {pii_detected}. Defaulting to False.")
                llm_errors.append(f"LLM returned non-boolean 'pii_detected': {pii_detected}")
                pii_detected = False
            
            pii_types = llm_output.get("pii_types", [])
            pii_details_list = llm_output.get("pii_details", []) # Renamed to avoid conflict
            summary = llm_output.get("summary", "N/A")

            # The guardrail is "triggered" if pii_detected is true.
            is_triggered = bool(pii_detected)

            current_details = {
                "pii_types": pii_types,
                "pii_details": pii_details_list, # Use the renamed variable
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
            self.logger.error(f"LLMResponseParseError in PII check: {e.message}, Raw: {e.raw_response[:200]}", exc_info=True)
            llm_errors.append(f"Failed to parse PII LLM response: {e.message}. Snippet: {e.raw_response[:100] if e.raw_response else 'N/A'}")
            return CheckOutput(
                is_triggered=False, errors=llm_errors, raw_llm_response=e.raw_response,
                metadata={"warning": "Could not parse PII LLM output.", "parser_error_type": type(e).__name__, **llm_metadata}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error processing PII LLM response: {str(e)}", exc_info=True)
            llm_errors.append(f"Unexpected error processing PII LLM response: {str(e)}. Snippet: {raw_llm_response_content[:100] if raw_llm_response_content else 'N/A'}")
            return CheckOutput(
                is_triggered=False, errors=llm_errors, raw_llm_response=raw_llm_response_content,
                metadata={"warning": "Unexpected error during PII LLM response processing.", "error_type": type(e).__name__, **llm_metadata}
            )

GuardrailCatalog.register("pii_protection", PiiProtectionGuardrail)