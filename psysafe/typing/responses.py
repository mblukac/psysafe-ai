# psysafe/typing/responses.py
from typing import Any, TypeVar

ResponseT = TypeVar("ResponseT")

# OpenAI specific response type (example structure)
# Based on common usage for openai.chat.completions.create response object
# This is a simplified version; the actual object is more complex.
# For streaming, chunks will also have a similar structure.
OpenAIChatChoice = dict[
    str, Any,
]  # e.g., {"message": {"role": "assistant", "content": "Hi there!"}, "finish_reason": "stop"}
OpenAIChatResponse = dict[str, Any]  # e.g., {"id": "...", "choices": List[OpenAIChatChoice], "model": "..."}
# Anthropic specific response type (example structure)
# Based on common usage for anthropic.messages.create response object
# This is a simplified version.
AnthropicContentBlock = dict[str, Any]  # e.g., {"type": "text", "text": "Hi there!"}
AnthropicChatResponse = dict[
    str, Any,
]  # e.g., {"id": "...", "content": List[AnthropicContentBlock], "model": "...", "stop_reason": "end_turn"}
# For streaming, chunks can be message_start, content_block_delta, message_delta, message_stop etc.
# We'll simplify the streamed type to be similar to the full response for now.
AnthropicStreamEvent = dict[str, Any]
# Transformers specific response type (example structure)
# Output from pipelines like 'text-generation' or 'conversational'
TransformersChatResponse = dict[str, Any]  # e.g., {"generated_text": "Bonjour!", "conversation_history": [...]}
# Streaming with Transformers is more complex and pipeline-dependent.
# For now, we might not fully support streaming in the same way or make it a simple text stream.
TransformersStreamChunk = dict[str, Any]  # e.g., {"token_text": "Bonjour"} or {"partial_text": "Bonjour"}
