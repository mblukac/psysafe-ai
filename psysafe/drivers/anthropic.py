# psysafe/drivers/anthropic.py
from typing import Any, AsyncIterator, Dict, List, Optional

from psysafe.drivers.base import ChatDriverABC
from psysafe.typing.requests import AnthropicChatRequest, AnthropicMessage
from psysafe.typing.responses import AnthropicChatResponse, AnthropicStreamEvent

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
except ImportError:
    raise ImportError("Anthropic library is required for AnthropicChatDriver. Please install it (e.g., `pip install anthropic`).")

class AnthropicChatDriver(ChatDriverABC[AnthropicChatRequest, AnthropicChatResponse]):
    """Driver for Anthropic Claude models."""

    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None, **kwargs):
        """
        Initializes the Anthropic Chat Driver.

        Args:
            model: The Anthropic model to use.
            api_key: Optional Anthropic API key. If not provided, anthropic library
                     will look for ANTHROPIC_API_KEY environment variable.
            **kwargs: Additional keyword arguments to pass to the Anthropic client.
        """
        self.model = model
        self.client_kwargs = kwargs
        if api_key:
            self.client_kwargs["api_key"] = api_key
        
        self._client: Optional[Anthropic] = None
        self._async_client: Optional[AsyncAnthropic] = None

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(**self.client_kwargs)
        return self._client

    @property
    def async_client(self) -> AsyncAnthropic:
        if self._async_client is None:
            self._async_client = AsyncAnthropic(**self.client_kwargs)
        return self._async_client

    def send(self, request: AnthropicChatRequest) -> AnthropicChatResponse:
        """
        Send a request to the Anthropic API.
        Request structure example:
        {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello, Claude"}]
        }
        """
        payload = {"model": self.model, **request}
        if "messages" not in payload or not isinstance(payload["messages"], list):
            raise ValueError("Request 'messages' must be a list.")
        if "max_tokens" not in payload: # Anthropic requires max_tokens
            payload["max_tokens"] = 1024 # Default if not provided

        try:
            response = self.client.messages.create(**payload)
            return response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        except Exception as e:
            # print(f"Anthropic API error: {e}") # Replace with proper logging
            raise

    async def stream(self, request: AnthropicChatRequest) -> AsyncIterator[AnthropicStreamEvent]:
        """
        Stream a request to the Anthropic API.
        Yields response event chunks as dictionaries.
        Events can be: message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop.
        """
        payload = {"model": self.model, **request, "stream": True}
        if "messages" not in payload or not isinstance(payload["messages"], list):
            raise ValueError("Request 'messages' must be a list.")
        if "max_tokens" not in payload:
            payload["max_tokens"] = 1024

        try:
            async with self.async_client.messages.stream(**payload) as stream_response:
                async for event in stream_response:
                    # Convert event to dict for consistency
                    # event objects are Pydantic models in anthropic >0.16
                    yield event.model_dump() if hasattr(event, 'model_dump') else dict(event)
        except Exception as e:
            # print(f"Anthropic API streaming error: {e}") # Replace with proper logging
            raise

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "driver_type": "anthropic",
            "model_name": self.model,
            "supports_streaming": True,
            "client_config": self.client_kwargs # Be careful with sensitive info
        }