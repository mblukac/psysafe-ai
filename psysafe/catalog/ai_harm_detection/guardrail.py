from typing import List, Dict, Any, Optional
import json
import logging

from psysafe.catalog.base.llm_guardrail import LLMGuardrail
from psysafe.catalog.ai_harm_detection.config import AiHarmDetectionConfig, HarmClassification, PolicyViolationType
from psysafe.core.types import GuardrailResponse
from psysafe.core.models import Conversation, GuardedRequest
from psysafe.core.exceptions import (
    LLMDriverError,
    ResponseParsingError,
    GuardrailError
)
from psysafe.utils.parsing import ResponseParser
from psysafe.core.template import PromptTemplate, PromptRenderCtx
from psysafe.typing.requests import OpenAIChatRequest
from psysafe.catalog import GuardrailCatalog


class AiHarmDetectionGuardrail(LLMGuardrail[AiHarmDetectionConfig]):
    """
    A guardrail to detect AI-generated content that may cause harm or violate safety policies.
    
    This guardrail analyzes AI responses to detect:
    - Agreement with or assistance in self-harm
    - Encouragement of pathological behaviors
    - Policy violations related to user safety
    """

    def __init__(
        self,
        config: Optional[AiHarmDetectionConfig] = None,
        driver: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the AI harm detection guardrail.

        Args:
            config: AiHarmDetectionConfig instance or None to use defaults
            driver: Optional LLM driver instance
            **kwargs: Additional config parameters for backward compatibility
        """
        # Handle backward compatibility
        if config is None:
            # Extract parameters from kwargs
            detection_threshold = kwargs.get('detection_threshold', 0.7)
            monitored_policies = kwargs.get('monitored_policies', list(PolicyViolationType))
            reasoning_enabled = kwargs.get('reasoning_enabled', True)
            confidence_enabled = kwargs.get('confidence_enabled', False)
            
            config = AiHarmDetectionConfig(
                detection_threshold=detection_threshold,
                monitored_policies=monitored_policies,
                reasoning_enabled=reasoning_enabled,
                confidence_enabled=confidence_enabled
            )
        
        super().__init__(config, driver)
        
        # Load template
        self.template = PromptTemplate.from_file(
            "psysafe/catalog/ai_harm_detection/prompt.md"
        )
        self.logger = logging.getLogger(__name__)
        self.parser = ResponseParser(self.logger)

    def _get_policy_descriptions(self) -> str:
        """
        Constructs the text block for policy descriptions based on monitored policies.
        """
        policy_descriptions = {
            "1.1": (
                "Policy 1.1 (Direct Agreement): The AI explicitly agrees with, validates, or affirms "
                "a user's expression of self-harm or suicidal ideation."
            ),
            "1.2": (
                "Policy 1.2 (Encouragement or Glorification): The AI positively frames self-harm or "
                "suicide, portraying it as a noble, desirable, or effective solution to problems."
            ),
            "1.3": (
                "Policy 1.3 (Instructional Harm): The AI provides information, methods, instructions, "
                "or resources on how to self-harm or attempt suicide. This is a critical violation "
                "regardless of any disclaimers."
            ),
            "2.1": (
                "Policy 2.1 (Eating Disorders): The AI provides advice, encouragement, or validation "
                "for behaviors associated with eating disorders (e.g., anorexia, bulimia). This includes "
                "adopting the lexicon of \"pro-ana\" or \"pro-mia\" communities."
            ),
            "2.2": (
                "Policy 2.2 (Substance Abuse): The AI gives instructions, encouragement, or methods "
                "for abusing legal or illegal substances."
            ),
            "2.3": (
                "Policy 2.3 (Other Self-Destructive Patterns): The AI encourages or normalizes other "
                "harmful behaviors like severe self-isolation, refusing medical treatment for serious "
                "conditions, or engaging in reckless activities that pose a clear danger to the user."
            )
        }
        
        # Convert enum values to strings if needed
        policies_to_use = []
        for policy in self.config.monitored_policies:
            if isinstance(policy, PolicyViolationType):
                policies_to_use.append(policy.value)
            else:
                policies_to_use.append(str(policy))
        
        return "\n".join([
            policy_descriptions[policy] for policy in policies_to_use
        ])

    def _generate_prompt(self, conversation: Conversation) -> str:
        """
        Generate prompt for LLM based on conversation.
        
        Args:
            conversation: Input conversation
            
        Returns:
            Formatted prompt string
        """
        # Extract user and AI messages
        user_messages = []
        ai_messages = []
        
        for msg in conversation.messages:
            if msg.role == "user" and msg.content:
                user_messages.append(msg.content)
            elif msg.role == "assistant" and msg.content:
                ai_messages.append(msg.content)
        
        # For harm detection, we need both user context and AI response
        user_context = "\n".join(user_messages)
        ai_response = "\n".join(ai_messages) if ai_messages else ""
        
        # Get policy descriptions
        policy_descriptions_text = self._get_policy_descriptions()
        
        render_ctx = PromptRenderCtx(
            driver_type="openai",  # Default, could be made configurable
            model_name="unknown",
            request_type="chat",
            variables={
                "user_context": user_context,
                "ai_response": ai_response,
                "policy_descriptions": policy_descriptions_text,
                "reasoning": self.config.reasoning_enabled,
                "confidence": self.config.confidence_enabled,
                "require_step_by_step": self.config.require_step_by_step_reasoning,
                "include_policy_details": self.config.include_policy_details,
            }
        )

        return self.template.render(render_ctx)

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with prompt and return raw response.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Raw LLM response
            
        Raises:
            LLMDriverError: If LLM call fails
        """
        if not hasattr(self.driver, 'send'):
            raise LLMDriverError(
                f"Driver {type(self.driver).__name__} does not support 'send' method",
                guardrail_name=self.__class__.__name__
            )
        
        # Construct request for LLM
        llm_request = {
            "messages": [{"role": "system", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            response = self.driver.send(llm_request)
            
            # Extract content from response
            if response and response.get("choices"):
                first_choice = response["choices"][0]
                if first_choice and first_choice.get("message"):
                    content = first_choice["message"].get("content")
                    if content is not None:  # Allow empty string
                        return content
            
            raise LLMDriverError(
                "Could not extract content from LLM response",
                guardrail_name=self.__class__.__name__,
                context={"raw_response": str(response)}
            )
            
        except Exception as e:
            if isinstance(e, LLMDriverError):
                raise
            raise LLMDriverError(
                f"LLM call failed: {str(e)}",
                guardrail_name=self.__class__.__name__
            )

    def _parse_llm_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract harm detection results.
        
        Args:
            raw_response: Raw LLM response
            
        Returns:
            Parsed response dictionary
            
        Raises:
            ResponseParsingError: If response cannot be parsed
        """
        try:
            # Try to parse as JSON
            response_data = json.loads(raw_response.strip())
            
            # Validate required fields
            if "classification" not in response_data:
                raise ResponseParsingError(
                    "Missing required field 'classification' in LLM response",
                    raw_response=raw_response
                )
            
            # Validate classification value
            classification = response_data["classification"]
            if classification not in [cls.value for cls in HarmClassification]:
                raise ResponseParsingError(
                    f"Invalid classification '{classification}'. Must be one of: {[cls.value for cls in HarmClassification]}",
                    raw_response=raw_response
                )
            
            # Validate policy violation field
            violated_policy_id = response_data.get("violated_policy_id", "N/A")
            if classification == HarmClassification.HARMFUL.value and violated_policy_id == "N/A":
                self.logger.warning(
                    f"HARMFUL classification without policy violation ID: {raw_response[:200]}"
                )
            
            return response_data
            
        except json.JSONDecodeError as e:
            raise ResponseParsingError(
                f"Failed to parse JSON response: {str(e)}",
                raw_response=raw_response
            )
        except Exception as e:
            raise ResponseParsingError(
                f"Unexpected error parsing response: {str(e)}",
                raw_response=raw_response
            )

    def check(self, conversation: Conversation) -> GuardrailResponse:
        """
        Check conversation for AI harm using LLM.
        
        Args:
            conversation: Input conversation
            
        Returns:
            GuardrailResponse with check results
        """
        try:
            # Ensure driver is available
            self._ensure_driver()
            
            # Generate prompt
            prompt = self._generate_prompt(conversation)
            
            # Call LLM
            raw_response = self._call_llm(prompt)
            
            # Parse response
            parsed_response = self._parse_llm_response(raw_response)
            
            # Extract results
            classification = parsed_response.get("classification", HarmClassification.SAFE.value)
            violated_policy_id = parsed_response.get("violated_policy_id", "N/A")
            reasoning = parsed_response.get("reasoning", "")
            
            # Determine if triggered based on classification
            is_triggered = classification in [
                HarmClassification.HARMFUL.value,
                HarmClassification.BORDERLINE.value
            ]
            
            # Calculate risk score based on classification
            risk_score = 0.0
            if classification == HarmClassification.BORDERLINE.value:
                risk_score = self.config.borderline_threshold
            elif classification == HarmClassification.HARMFUL.value:
                risk_score = self.config.harmful_threshold
            
            # Build details
            details = {
                "classification": classification,
                "violated_policy_id": violated_policy_id,
                "reasoning": reasoning,
                "monitored_policies": [policy.value for policy in self.config.monitored_policies],
            }
            
            # Add optional confidence if enabled
            if self.config.confidence_enabled and "confidence_score" in parsed_response:
                details["confidence_score"] = parsed_response["confidence_score"]
            
            response = GuardrailResponse(
                is_triggered=is_triggered,
                risk_score=risk_score,
                details=details,
                raw_llm_response=raw_response,
                metadata={
                    "guardrail_type": "ai_harm_detection",
                    "classification": classification,
                    "policy_violation": violated_policy_id,
                }
            )
            
            return response
            
        except ResponseParsingError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            
            # Apply fail-safe mode
            if self.config.fail_safe_mode:
                return GuardrailResponse(
                    is_triggered=True,  # Fail-safe: assume potential harm
                    risk_score=self.config.borderline_threshold,
                    details={
                        "classification": HarmClassification.BORDERLINE.value,
                        "violated_policy_id": "N/A",
                        "reasoning": "Failed to parse LLM response - defaulting to BORDERLINE for safety",
                    },
                    errors=[f"Failed to parse LLM response: {str(e)}"],
                    raw_llm_response=e.raw_response,
                    metadata={"error_type": "ResponseParsingError", "fail_safe_applied": True}
                )
            else:
                return GuardrailResponse(
                    is_triggered=False,
                    errors=[f"Failed to parse LLM response: {str(e)}"],
                    raw_llm_response=e.raw_response,
                    metadata={"error_type": "ResponseParsingError"}
                )
                
        except LLMDriverError as e:
            self.logger.error(f"LLM driver error: {e}")
            if "No LLM driver configured" in str(e):
                raise
            return GuardrailResponse(
                is_triggered=False,
                errors=[str(e)],
                metadata={"error_type": "LLMDriverError"}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in check: {e}")
            return GuardrailResponse(
                is_triggered=False,
                errors=[f"Unexpected error: {str(e)}"],
                metadata={"error_type": type(e).__name__}
            )

    def apply(self, request: OpenAIChatRequest) -> GuardedRequest[OpenAIChatRequest]:
        """
        Apply the guardrail to an incoming request.

        Args:
            request: The incoming OpenAIChatRequest

        Returns:
            A GuardedRequest object containing the original and modified request
        """
        # Extract messages for context
        user_messages = []
        ai_messages = []
        messages = request.get("messages", [])
        
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                user_messages.append(msg["content"])
            elif msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                ai_messages.append(msg["content"])
        
        user_context = "\n".join(user_messages)
        ai_response = "\n".join(ai_messages)
        
        # Get policy descriptions
        policy_descriptions_text = self._get_policy_descriptions()
        
        render_ctx = PromptRenderCtx(
            driver_type=request.get("driver_type", "openai"),
            model_name=request.get("model", "unknown"),
            request_type="chat",
            variables={
                "user_context": user_context,
                "ai_response": ai_response,
                "policy_descriptions": policy_descriptions_text,
                "reasoning": self.config.reasoning_enabled,
                "confidence": self.config.confidence_enabled,
                "require_step_by_step": self.config.require_step_by_step_reasoning,
                "include_policy_details": self.config.include_policy_details,
            }
        )

        rendered_prompt = self.template.render(render_ctx)

        modified_request = request.copy()
        
        # Ensure 'messages' key exists and is a list
        if "messages" not in modified_request or not isinstance(modified_request.get("messages"), list):
            modified_request["messages"] = []

        modified_request["messages"].insert(
            0, {"role": "system", "content": rendered_prompt}
        )
        
        # Create metadata for GuardedRequest
        metadata: Dict[str, Any] = {
            "guardrail_name": "AiHarmDetectionGuardrail",
            "applied_prompt": rendered_prompt,
            "monitored_policies": [policy.value for policy in self.config.monitored_policies],
            "detection_threshold": self.config.detection_threshold,
            "reasoning_enabled": self.config.reasoning_enabled,
            "confidence_enabled": self.config.confidence_enabled,
        }

        return GuardedRequest(
            original_request=request,
            modified_request=modified_request,
            is_modified=True,
            metadata=metadata,
        )

    def validate(self, response: Any) -> Any:
        """
        Validate a response against the guardrail.
        
        For AI harm detection, we don't perform post-response validation.
        The main logic is in the check method which analyzes conversations.
        
        Args:
            response: The response to validate
            
        Returns:
            ValidationReport indicating the response is valid
        """
        from psysafe.core.models import ValidationReport
        
        return ValidationReport(
            is_valid=True,
            violations=[],
            metadata={"guardrail": "ai_harm_detection"}
        )


# Register the guardrail with the catalog
GuardrailCatalog.register("ai_harm_detection", AiHarmDetectionGuardrail)