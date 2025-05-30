# psysafe/drivers/openai.py
from typing import Any, AsyncIterator, Dict, List, Optional # Added List, Optional

# Driver base
from psysafe.drivers.base import ChatDriverABC

# Typing for OpenAI
from psysafe.typing.requests import OpenAIChatRequest, OpenAIMessage # Added OpenAIMessage
from psysafe.typing.responses import OpenAIChatResponse

# OpenAI library - will be a dependency
try:
    import openai
    from openai import OpenAI, AsyncOpenAI # For newer openai versions
except ImportError:
    raise ImportError("OpenAI library is required for OpenAIChatDriver. Please install it (e.g., `pip install openai`).")


class OpenAIChatDriver(ChatDriverABC[OpenAIChatRequest, OpenAIChatResponse]):
    """Driver for OpenAI chat models."""

    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14", api_key: Optional[str] = None, **kwargs):
        """
        Initializes the OpenAI Chat Driver.

        Args:
            model: The OpenAI model to use (e.g., "gpt-4.1-mini-2025-04-14", "gpt-4o").
            api_key: Optional OpenAI API key. If not provided, openai library
                     will look for OPENAI_API_KEY environment variable.
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
        """
        self.model = model
        self.client_kwargs = kwargs
        if api_key:
            self.client_kwargs["api_key"] = api_key
        
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy-loaded synchronous OpenAI client."""
        if self._client is None:
            self._client = OpenAI(**self.client_kwargs)
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy-loaded asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self.client_kwargs)
        return self._async_client

    def send(self, request: OpenAIChatRequest) -> OpenAIChatResponse:
        """
        Send a request to the OpenAI API.
        The request should conform to OpenAIChatRequest structure, typically:
        {
            "messages": [{"role": "user", "content": "Hello!"}],
            "model": "gpt-4.1-mini-2025-04-14", // Optional, can be overridden from driver's default
            ...other_openai_params
        }
        """
        # Ensure model is part of the request, defaulting to driver's model
        payload = {"model": self.model, **request}
        
        # Validate messages format (basic check)
        if not isinstance(payload.get("messages"), list) or \
           not all(isinstance(m, dict) and "role" in m and "content" in m for m in payload["messages"]):
            raise ValueError("Request messages must be a list of dicts with 'role' and 'content' keys.")

        try:
            response = self.client.chat.completions.create(**payload)
            # Convert to dict for consistency if it's not already (Pydantic models from openai >1.0 are objects)
            return response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        except Exception as e:
            # Log error and re-raise or handle appropriately
            raise # Re-raise the exception for now

    async def stream(self, request: OpenAIChatRequest) -> AsyncIterator[OpenAIChatResponse]:
        """
        Stream a request to the OpenAI API.
        Yields response chunks as dictionaries.
        """
        payload = {"model": self.model, **request, "stream": True}
        
        if not isinstance(payload.get("messages"), list) or \
           not all(isinstance(m, dict) and "role" in m and "content" in m for m in payload["messages"]):
            raise ValueError("Request messages must be a list of dicts with 'role' and 'content' keys.")

        try:
            stream_response = await self.async_client.chat.completions.create(**payload)
            async for chunk in stream_response:
                # Convert chunk to dict for consistency
                yield chunk.model_dump() if hasattr(chunk, 'model_dump') else dict(chunk)
        except Exception as e:
            # print(f"OpenAI API streaming error: {e}") # Replace with proper logging
            pass  # Handle the exception gracefully

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the driver."""
        return {
            "driver_type": "openai",
            "model_name": self.model,
            "supports_streaming": True,
            "client_config": self.client_kwargs # Be careful about exposing sensitive info like API keys
        }