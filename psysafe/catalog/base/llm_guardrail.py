from abc import abstractmethod
from typing import TypeVar, Generic, Optional, Any
from psysafe.core.config import GuardrailConfig
from psysafe.core.types import GuardrailResponse
from psysafe.core.models import Conversation
from psysafe.core.exceptions import LLMDriverError
from psysafe.catalog.base.prompt_guardrail import ModernPromptGuardrail

T = TypeVar('T', bound=GuardrailConfig)

class LLMGuardrail(ModernPromptGuardrail[T], Generic[T]):
    """Base class for LLM-based guardrails
    
    This class extends ModernPromptGuardrail to provide additional
    functionality specific to guardrails that interact with LLM drivers.
    """
    
    def __init__(self, config: T, driver: Optional[Any] = None):
        """Initialize LLM guardrail with configuration and optional driver
        
        Args:
            config: Guardrail configuration
            driver: Optional LLM driver instance
        """
        super().__init__(config)
        self.driver = driver
    
    def set_driver(self, driver: Any) -> None:
        """Set or update the LLM driver
        
        Args:
            driver: LLM driver instance
        """
        self.driver = driver
    
    def _ensure_driver(self) -> None:
        """Ensure driver is available before LLM operations"""
        if self.driver is None:
            raise LLMDriverError(
                "No LLM driver configured for this guardrail",
                guardrail_name=self.__class__.__name__
            )
    
    @abstractmethod
    def _generate_prompt(self, conversation: Conversation) -> str:
        """Generate prompt for LLM based on conversation
        
        Args:
            conversation: Input conversation
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt and return raw response
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Raw LLM response
        """
        pass
    
    def check(self, conversation: Conversation) -> GuardrailResponse:
        """Check conversation using LLM
        
        Args:
            conversation: Input conversation
            
        Returns:
            GuardrailResponse with check results
        """
        self._ensure_driver()
        
        try:
            # Generate prompt
            prompt = self._generate_prompt(conversation)
            
            # Call LLM
            raw_response = self._call_llm(prompt)
            
            # Parse response to GuardrailResponse
            response = self.parser.parse_to_model(raw_response, GuardrailResponse)
            
            # Add raw response for debugging
            response.raw_llm_response = raw_response
            
            return response
            
        except Exception as e:
            # Return error response
            return GuardrailResponse(
                is_triggered=False,
                errors=[str(e)],
                metadata={"error_type": type(e).__name__}
            )