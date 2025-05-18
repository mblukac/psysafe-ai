import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import aisuite as ai
from enum import Enum

# Define paths
ROOT_DIR = Path(__file__).parent.parent
ENV_FILE = ROOT_DIR / ".env"

class LLMModels(Enum):
    GPT4o = 'openai:gpt-4o'
    GPT4oMINI = 'openai:gpt-4o-mini'
    DEEPSEEKR1 = 'deepseek:deepseek-reasoner'

def load_environment() -> None:
    """
    Load environment variables from .env file.
    This should be called before any other functions in this module.
    """
    load_dotenv(ENV_FILE)
    print(f"Loaded environment variables from {ENV_FILE}")


def get_client(additional_configs: Optional[Dict[str, Dict[str, str]]] = None) -> ai.Client:
    """
    Create and configure an aisuite client with API keys from environment variables.
    
    Args:
        additional_configs: Additional provider configurations to pass to the client.
                          Format: {"provider_name": {"key1": "value1", ...}}
    
    Returns:
        Configured aisuite client
    """
    # Create the client
    client = ai.Client()
    
    # Apply any additional configurations
    if additional_configs:
        client.configure(additional_configs)
        
    return client


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    client: Optional[ai.Client] = None,
    **kwargs
) -> Any:
    """
    Call an LLM with the specified model and messages.
    
    Args:
        model: The model identifier in the format "provider:model_name".
               Examples: "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"
        messages: List of message dictionaries with "role" and "content" keys.
        temperature: Sampling temperature. Higher values mean more creative responses.
        client: Optional pre-configured client. If None, a new client will be created.
        **kwargs: Additional keyword arguments to pass to the client.create() method.
    
    Returns:
        The response from the LLM
    """
    if client is None:
        client = get_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        return response
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise


def get_llm_response(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    client: Optional[ai.Client] = None,
    **kwargs
) -> str:
    """
    Get a response from an LLM using a simplified interface.
    
    Args:
        model: The model identifier in the format "provider:model_name"
        prompt: The user prompt to send to the model
        system_prompt: Optional system prompt to set the behavior of the model
        temperature: Sampling temperature. Higher values mean more creative responses.
        client: Optional pre-configured client. If None, a new client will be created.
        **kwargs: Additional keyword arguments to pass to the call_llm() function.
    
    Returns:
        The text response from the LLM
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    response = call_llm(
        model=model,
        messages=messages,
        temperature=temperature,
        client=client,
        **kwargs
    )
    
    return response.choices[0].message.content


def compare_llm_responses(
    prompt: str,
    models: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    client: Optional[ai.Client] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Compare responses from multiple LLMs for the same prompt.
    
    Args:
        prompt: The user prompt to send to the models
        models: List of model identifiers in the format "provider:model_name"
        system_prompt: Optional system prompt to set the behavior of the models
        temperature: Sampling temperature. Higher values mean more creative responses.
        client: Optional pre-configured client. If None, a new client will be created.
        **kwargs: Additional keyword arguments to pass to the get_llm_response() function.
    
    Returns:
        Dictionary mapping model names to their responses
    """
    if client is None:
        client = get_client()
    
    results = {}
    
    for model in models:
        try:
            response = get_llm_response(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                client=client,
                **kwargs
            )
            results[model] = response
        except Exception as e:
            results[model] = f"Error: {str(e)}"
    
    return results


def format_conversation_for_classification(
    conversation: Union[List[Dict[str, str]], List[Tuple[str, str]], str, Any],
    chronological: bool = True,
    include_roles: bool = True,
    format_style: str = "plain",
    max_message_length: Optional[int] = None,
    attribute_prefixes: Optional[Dict[str, str]] = None
) -> str:
    """
    Converts conversation history into a structured string representation for classification.
    
    Args:
        conversation: Conversation history in one of these formats:
            - List of message dictionaries with "role" and "content" keys (OpenAI/Anthropic format)
            - List of tuples of (role, content)
            - String (already formatted conversation)
            - Custom object with a __str__ method
        chronological: If True, formats conversation from oldest to newest (default);
                      If False, formats from newest to oldest
        include_roles: If True, includes role labels in the output
        format_style: Formatting style for the conversation. Options:
            - "plain": Simple text with role prefixes
            - "markdown": Markdown formatting with headers for roles
            - "compact": Minimal formatting to save tokens
        max_message_length: Optional maximum length for any single message
                           (if exceeded, message will be truncated with "...")
        attribute_prefixes: Optional dictionary mapping roles to prefixes
                          Default: {"user": "User: ", "assistant": "Assistant: ", "system": "System: "}
    
    Returns:
        Formatted conversation string ready for classification
    """
    # Set default prefixes if not provided
    if attribute_prefixes is None:
        attribute_prefixes = {
            "user": "User: ",
            "assistant": "Assistant: ",
            "system": "System: ",
            "function": "Function: ",
            "tool": "Tool: ",
            "model": "Model: ",
            "human": "Human: ",
            "ai": "AI: "
        }
    
    # If conversation is already a string, return it as is (with optional truncation)
    if isinstance(conversation, str):
        if max_message_length and len(conversation) > max_message_length:
            return conversation[:max_message_length] + "..."
        return conversation
    
    # Convert conversation to a list of (role, content) tuples for standardization
    messages = []
    
    # Handle different conversation formats
    if isinstance(conversation, list):
        for item in conversation:
            if isinstance(item, dict) and "role" in item and "content" in item:
                # Handle OpenAI/Anthropic format
                role = item.get("role", "unknown")
                content = item.get("content", "")
                messages.append((role, content))
            elif isinstance(item, tuple) and len(item) >= 2:
                # Handle tuple format (role, content)
                role = item[0] if item[0] else "unknown"
                content = item[1] if item[1] else ""
                messages.append((role, content))
            elif hasattr(item, "role") and hasattr(item, "content"):
                # Handle object with role and content attributes
                role = getattr(item, "role", "unknown")
                content = getattr(item, "content", "")
                messages.append((role, content))
            else:
                # Try to convert to string
                try:
                    messages.append(("unknown", str(item)))
                except Exception:
                    messages.append(("unknown", ""))
    else:
        # Try to handle a custom object
        try:
            # Check if the object has a structured conversation representation
            if hasattr(conversation, "messages"):
                items = getattr(conversation, "messages", [])
                for item in items:
                    if hasattr(item, "role") and hasattr(item, "content"):
                        role = getattr(item, "role", "unknown")
                        content = getattr(item, "content", "")
                        messages.append((role, content))
            # Fallback to string representation
            else:
                return str(conversation)
        except Exception:
            # Return empty string if all else fails
            return ""
    
    # Remove empty messages
    messages = [(role, content) for role, content in messages if content]
    
    # Handle empty conversation case
    if not messages:
        return ""
    
    # Reverse order if not chronological
    if not chronological:
        messages = list(reversed(messages))
    
    # Format the conversation based on the requested style
    formatted_lines = []
    
    if format_style == "plain":
        for role, content in messages:
            # Truncate message if needed
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length] + "..."
            
            # Add role prefix if needed
            if include_roles:
                prefix = attribute_prefixes.get(role.lower(), f"{role.capitalize()}: ")
                formatted_lines.append(f"{prefix}{content}")
            else:
                formatted_lines.append(content)
    
    elif format_style == "markdown":
        for role, content in messages:
            # Truncate message if needed
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length] + "..."
            
            # Add role as markdown header if needed
            if include_roles:
                role_header = role.capitalize()
                formatted_lines.append(f"### {role_header}")
                formatted_lines.append(content)
                formatted_lines.append("")  # Add blank line between messages
            else:
                formatted_lines.append(content)
                formatted_lines.append("")  # Add blank line between messages
    
    elif format_style == "compact":
        for role, content in messages:
            # Truncate message if needed
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length] + "..."
            
            # Add minimal role prefix if needed
            if include_roles:
                prefix = role[0].upper() + ": "  # Use first letter of role as prefix
                formatted_lines.append(f"{prefix}{content}")
            else:
                formatted_lines.append(content)
    
    else:  # Default to plain format for unrecognized styles
        for role, content in messages:
            # Truncate message if needed
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length] + "..."
            
            # Add role prefix if needed
            if include_roles:
                prefix = attribute_prefixes.get(role.lower(), f"{role.capitalize()}: ")
                formatted_lines.append(f"{prefix}{content}")
            else:
                formatted_lines.append(content)
    
    # Join the formatted lines with newlines
    formatted_conversation = "\n".join(formatted_lines)
    
    return formatted_conversation