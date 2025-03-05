import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import aisuite as ai

# Define paths
ROOT_DIR = Path(__file__).parent.parent
ENV_FILE = ROOT_DIR / ".env"


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