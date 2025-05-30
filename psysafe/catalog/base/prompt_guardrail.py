from abc import abstractmethod
from typing import TypeVar, Generic
from psysafe.core.base import GuardrailBase
from psysafe.core.config import GuardrailConfig
from psysafe.core.types import GuardrailResponse
from psysafe.core.models import Conversation
from psysafe.core.exceptions import GuardrailConfigError
from psysafe.utils.parsing import ResponseParser

T = TypeVar('T', bound=GuardrailConfig)

class ModernPromptGuardrail(GuardrailBase, Generic[T]):
    """Modern base class for prompt-based guardrails"""
    
    def __init__(self, config: T):
        self.config = config
        self.parser = ResponseParser()
        self._validate_config()
    
    @abstractmethod
    def check(self, conversation: Conversation) -> GuardrailResponse:
        """Check conversation with typed response"""
        pass
    
    def _validate_config(self) -> None:
        """Validate guardrail configuration"""
        if not isinstance(self.config, GuardrailConfig):
            raise GuardrailConfigError("Invalid configuration type")