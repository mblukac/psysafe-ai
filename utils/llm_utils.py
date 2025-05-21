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
import json
import re
import logging
from xml.etree import ElementTree as ET # For basic XML parsing

# Define a custom exception for parsing errors
class LLMResponseParseError(Exception):
    """Custom exception for errors encountered during LLM response parsing."""
    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.message = message

    def __str__(self):
        return f"{self.message} (Raw response snippet: {self.raw_response[:100] if self.raw_response else 'N/A'})"


def parse_llm_response(
    raw_response: Optional[str], 
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Parses a raw LLM response string into a Python dictionary.

    Attempts parsing in the following order:
    1. Direct JSON parsing.
    2. JSON from Markdown code blocks (e.g., ```json ... ```).
    3. Simple, flat XML-like key-value pairs (e.g., <key1>value1</key1><key2>value2</key2>).

    Args:
        raw_response: The raw string response from the LLM.
        logger: Optional logger instance for logging parsing attempts.

    Returns:
        A Python dictionary if parsing is successful.

    Raises:
        LLMResponseParseError: If all parsing attempts fail.
    """
    if not raw_response or not raw_response.strip():
        if logger:
            logger.warning("Received empty or None raw_response for parsing.")
        raise LLMResponseParseError("Raw response is empty or None.", raw_response=raw_response)

    # Attempt 1: Direct JSON parsing
    try:
        parsed_json = json.loads(raw_response)
        if not isinstance(parsed_json, dict):
            if logger:
                logger.debug(f"Direct JSON parsing resulted in non-dict type: {type(parsed_json)}. Failing.")
            raise LLMResponseParseError(f"Parsed JSON is not a dictionary (got {type(parsed_json)}).", raw_response=raw_response)
        if logger:
            logger.debug("Successfully parsed raw_response as direct JSON.")
        return parsed_json
    except json.JSONDecodeError as e:
        if logger:
            logger.debug(f"Direct JSON parsing failed: {e}. Trying Markdown extraction.")

    # Attempt 2: Extract JSON from Markdown code blocks
    markdown_json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_response, re.IGNORECASE)
    if markdown_json_match:
        json_content = markdown_json_match.group(1).strip()
        try:
            parsed_json = json.loads(json_content)
            if not isinstance(parsed_json, dict):
                if logger:
                    logger.debug(f"Markdown JSON parsing resulted in non-dict type: {type(parsed_json)}. Failing.")
                raise LLMResponseParseError(f"Parsed JSON from Markdown is not a dictionary (got {type(parsed_json)}).", raw_response=raw_response)
            if logger:
                logger.debug("Successfully parsed JSON from Markdown code block.")
            return parsed_json
        except json.JSONDecodeError as e:
            if logger:
                logger.debug(f"JSON parsing from Markdown failed: {e}. Trying XML-like parsing.")
    else:
        if logger:
            logger.debug("No Markdown JSON code block found. Trying XML-like parsing.")
            
    # Attempt 3: Parse simple, flat XML-like key-value pairs
    try:
        xml_to_parse = raw_response.strip()

        if not xml_to_parse.startswith("<"):
            if logger:
                logger.debug(f"String for XML parsing does not start with '<'. Content: {xml_to_parse[:50]}")
            # This is not valid XML, so raise a ParseError that will be caught below
            # and lead to the final LLMResponseParseError
            raise ET.ParseError("String does not start with '<', not valid XML.")
        
        # Heuristic for wrapping: if it's a sequence of tags or doesn't look like a single complete XML doc.
        is_likely_fragment = not (xml_to_parse.startswith("<") and xml_to_parse.endswith(">") and \
                               xml_to_parse.count("<") == (xml_to_parse.count("</") + xml_to_parse.count("/>")))
        
        if (is_likely_fragment and re.match(r"^(<\w+>.*?</\w+>)+$", xml_to_parse, re.DOTALL)) or \
           (re.match(r"^(<\w+>.*?</\w+>)+$", xml_to_parse, re.DOTALL) and not xml_to_parse.startswith("<root>")):
            if not xml_to_parse.startswith("<root>"): 
                 xml_to_parse = f"<root>{xml_to_parse}</root>"
        
        root = None
        try:
            root = ET.fromstring(xml_to_parse)
        except ET.ParseError as e_initial: 
            if not xml_to_parse.startswith("<root>"): 
                xml_to_parse_wrapped = f"<root>{raw_response.strip()}</root>" 
                try:
                    root = ET.fromstring(xml_to_parse_wrapped)
                    xml_to_parse = xml_to_parse_wrapped 
                except ET.ParseError as e_wrapped:
                    if logger:
                        logger.debug(f"XML-like parsing failed after attempting to wrap original response: {e_wrapped}")
                    raise LLMResponseParseError(f"Failed to parse XML-like content: {e_wrapped}", raw_response=raw_response) from e_wrapped
            else: 
                if logger:
                    logger.debug(f"XML-like parsing failed (was already wrapped or initial parse failed): {e_initial}")
                raise e_initial 
        
        if root is None: 
            raise LLMResponseParseError("XML root element could not be determined after parsing attempts.", raw_response=raw_response)

        xml_dict = {}
        elements_to_process = []
        
        # If we wrapped the input with <root>, process the children of this <root>.
        # Determine the base element for processing (either the original root or the single element inside our wrapper)
        current_processing_candidate = root
        if root.tag == 'root' and xml_to_parse.startswith("<root>") and len(list(root)) == 1:
            # If we wrapped a single element, focus on that single_actual_element
            current_processing_candidate = root[0]
        # else, current_processing_candidate remains 'root' (either original root, or our wrapper with multiple children)

        # Decide what to iterate over based on the candidate element's structure
        if current_processing_candidate.text and current_processing_candidate.text.strip() and len(list(current_processing_candidate)) > 0:
            # Candidate has both significant text and children: this is not flat for our simple parser.
            # Process this candidate itself, which will then trigger the "has children" error
            # in the loop below, as it's not a simple key-value.
            elements_to_process = [current_processing_candidate]
        elif len(list(current_processing_candidate)) > 0:
            # Candidate has children but no significant text (or no text at all).
            # Process its children as potential key-value pairs.
            elements_to_process = list(current_processing_candidate)
        else:
            # Candidate is a leaf (no children). Process it as a potential key-value pair.
            elements_to_process = [current_processing_candidate]

        # Note: If current_processing_candidate was 'root' (our parser's wrapper) and it contained
        # multiple children (e.g., <root><k1>v1</k1><k2>v2</k2></root>),
        # the logic above correctly sets elements_to_process = list(root)
        # because root.text is typically None/empty and len(list(root)) > 0.
        
        for element in elements_to_process:
            if element.tag: 
                if element.attrib: 
                    if logger:
                        logger.debug(f"XML element '{element.tag}' has attributes {element.attrib}, failing simple XML parse.")
                    raise LLMResponseParseError(f"XML element '{element.tag}' has attributes, which is not supported by simple parser.", raw_response=raw_response)
                
                if len(list(element)) == 0: 
                    xml_dict[element.tag] = element.text.strip() if element.text else ""
                else: 
                    if logger:
                        logger.warning(f"XML element '{element.tag}' has child elements, which is not flat. Raising error for simple parser.")
                    raise LLMResponseParseError(f"XML element '{element.tag}' has child elements, which is not supported by simple flat parser.", raw_response=raw_response)
            
        if xml_dict:
            if logger:
                logger.debug(f"Successfully parsed simple XML-like response: {xml_dict}")
            return xml_dict
        else:
            if raw_response.strip() and not (root.tag == 'root' and not list(root) and not root.text and not root.attrib and len(elements_to_process)==0) : 
                 if logger:
                    logger.debug("XML parsed to an empty dictionary, but input was not an empty root. Input: " + raw_response[:50])
                 raise LLMResponseParseError("XML parsed to an empty dictionary from non-empty/non-empty-root input.", raw_response=raw_response)
            elif logger: 
                logger.debug("XML input (e.g. <root/>) resulted in an empty dictionary.")
            if not xml_dict and raw_response.strip(): 
                 raise LLMResponseParseError("Failed to extract key-value pairs from XML structure.", raw_response=raw_response)

    except ET.ParseError as e: 
        if logger:
            logger.debug(f"XML-like parsing failed with ET.ParseError: {e}. Raw input: {raw_response[:100]}")
    except LLMResponseParseError: 
        raise
    except Exception as e_gen: 
        if logger:
            logger.error(f"Unexpected error during XML parsing: {e_gen}", exc_info=True)
        raise LLMResponseParseError(f"Unexpected error during XML processing: {e_gen}", raw_response=raw_response) from e_gen

    error_message = "All parsing attempts failed (direct JSON, Markdown JSON, simple XML)."
    if logger:
        logger.error(error_message + f" Raw response: {raw_response[:200]}")
    raise LLMResponseParseError(error_message, raw_response=raw_response)