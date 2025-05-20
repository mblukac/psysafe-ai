# psysafe/drivers/base.py
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Generic # Removed List, Optional as not used in this ABC

# Assuming RequestT and ResponseT will be imported from psysafe.typing
from psysafe.typing.requests import RequestT
from psysafe.typing.responses import ResponseT

class ChatDriverABC(Generic[RequestT, ResponseT], ABC):
    """Abstract base class for LLM chat drivers."""

    @abstractmethod
    def send(self, request: RequestT) -> ResponseT:
        """Send a request to the LLM and get a single response."""
        pass

    @abstractmethod
    async def stream(self, request: RequestT) -> AsyncIterator[ResponseT]:
        """Send a request to the LLM and stream responses back."""
        # This is an async generator, so it should yield responses.
        # Example:
        # yield response_chunk
        # To make it a valid abstract method that type checkers understand
        # when it's not implemented by a concrete class, we can do this:
        if False: # pragma: no cover
            yield
        # The 'if False' ensures this code is never run but satisfies the type checker
        # that this is indeed an async generator.
        # A concrete implementation would look different.

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the driver (e.g., model name, provider, capabilities)."""
        pass