"""
LLM utility functions for psysafe-ai

This module provides utility functions for interfacing with various LLM providers
through the aisuite library. It handles environment variable management and provides
a simple interface for accessing different LLM models.
"""

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


# Helper functions for specific vulnerability assessment tasks

def analyze_text_vulnerability(
    text: str,
    model: str = "anthropic:claude-3-5-sonnet-20240620",
    indicators: str = "both",
    sensitivity: str = "medium",
    reasoning: bool = True,
    confidence: bool = True,
    client: Optional[ai.Client] = None
) -> Dict[str, Any]:
    """
    Analyze text for vulnerability indicators using an LLM.
    This is a wrapper around the vulnerability detection module.
    
    Args:
        text: The text to analyze
        model: The model identifier in the format "provider:model_name"
        indicators: Which indicators to consider ('external', 'internal', or 'both')
        sensitivity: Detection sensitivity level ('low', 'medium', or 'high')
        reasoning: Whether to include reasoning in the response
        confidence: Whether to include a confidence score in the response
        client: Optional pre-configured client. If None, a new client will be created.
    
    Returns:
        Dictionary with vulnerability analysis results
    """
    from prompts.vulnerability.vulnerability import build_vulnerability_prompt
    from prompts.vulnerability.vulnerability import Sensitivity, VulnerabilityIndicators
    import re
    import xml.etree.ElementTree as ET
    
    # Convert string parameters to appropriate enums
    sensitivity_enum = Sensitivity[sensitivity.upper()]
    indicators_enum = VulnerabilityIndicators[indicators.upper()]
    
    # Build the vulnerability analysis prompt
    prompt = build_vulnerability_prompt(
        user_context=text,
        indicators=indicators_enum,
        sensitivity=sensitivity_enum,
        reasoning=reasoning,
        confidence=confidence
    )
    
    # Call the LLM with the vulnerability prompt
    response = get_llm_response(
        model=model,
        prompt=prompt,
        client=client
    )
    
    # Parse the XML response
    result = {}
    
    try:
        # Quick and dirty XML parsing - you might want to replace this with proper XML handling
        if reasoning:
            analysis_match = re.search(r"<analysis>(.*?)</analysis>", response, re.DOTALL)
            result["analysis"] = analysis_match.group(1).strip() if analysis_match else None
        
        vulnerable_match = re.search(r"<vulnerable>(.*?)</vulnerable>", response, re.DOTALL)
        result["vulnerable"] = vulnerable_match.group(1).strip().lower() == "true" if vulnerable_match else None
        
        if confidence:
            confidence_match = re.search(r"<confidence>(.*?)</confidence>", response, re.DOTALL)
            result["confidence"] = float(confidence_match.group(1).strip()) if confidence_match else None
        
        reason_match = re.search(r"<reason>(.*?)</reason>", response, re.DOTALL)
        result["reason"] = reason_match.group(1).strip() if reason_match else None
    
    except Exception as e:
        result["error"] = f"Failed to parse response: {str(e)}"
        result["raw_response"] = response
    
    return result