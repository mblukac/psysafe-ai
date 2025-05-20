# psysafe/typing/requests.py
from typing import TypeVar, Dict, Any, List

RequestT = TypeVar('RequestT')

# OpenAI specific request type (example structure)
# Based on common usage for openai.chat.completions.create
OpenAIMessage = Dict[str, str] # e.g., {"role": "user", "content": "Hello"}
OpenAIChatRequest = Dict[str, Any] # e.g., {"model": "gpt-3.5-turbo", "messages": List[OpenAIMessage]}
# Anthropic specific request type (example structure)
# Based on common usage for anthropic.messages.create
AnthropicMessage = Dict[str, Any] # e.g., {"role": "user", "content": "Hello"} or {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
AnthropicChatRequest = Dict[str, Any] # e.g., {"model": "claude-2", "max_tokens": 1024, "messages": List[AnthropicMessage]}
# Transformers specific request type (example structure)
# This is a simplified representation; actual inputs can vary greatly.
# We'll assume a text input or a list of messages for chat-like models.
TransformersChatMessage = Dict[str, str] # e.g., {"role": "user", "content": "Hello"}
TransformersChatRequest = Dict[str, Any] # e.g., {"text_input": "Translate to French:", "messages": List[TransformersChatMessage], "generation_kwargs": {"max_new_tokens": 50}}