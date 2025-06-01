import json
import re
import logging
from typing import Dict, Any, Optional, TypeVar
from xml.etree import ElementTree as ET
from pydantic import BaseModel

# Define a custom exception for parsing errors
class LLMResponseParseError(Exception):
    """Custom exception for errors encountered during LLM response parsing."""
    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.raw_response = raw_response

    def __str__(self):
        return f"{self.message}{f'. Raw response snippet: {self.raw_response[:100]}...' if self.raw_response else ''}"

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
    if not logger: # Ensure logger is available
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG) # Default to debug if no logger passed, or configure as needed

    if not raw_response or not raw_response.strip():
        logger.warning("Received empty or None raw_response for parsing.")
        raise LLMResponseParseError("Raw response is empty or None.", raw_response=raw_response)

    # Attempt 1: Direct JSON parsing
    try:
        parsed_json = json.loads(raw_response)
        if not isinstance(parsed_json, dict):
            logger.debug(f"Direct JSON parsing resulted in non-dict type: {type(parsed_json)}. Failing.")
            # This specific message is not directly raised as an exception message to match old tests,
            # but the failure will lead to the generic "All parsing attempts failed" if others fail.
            # To make tests pass that expect "Invalid JSON", we'd need to be more specific here.
            # For now, this path will make it try other methods.
        else: # It is a dict
            logger.debug("Successfully parsed raw_response as direct JSON.")
            return parsed_json
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}. Trying Markdown extraction.")

    # Attempt 2: Extract JSON from Markdown code blocks
    markdown_json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_response, re.IGNORECASE)
    if markdown_json_match:
        json_content = markdown_json_match.group(1).strip()
        try:
            parsed_json = json.loads(json_content)
            if not isinstance(parsed_json, dict):
                logger.debug(f"Markdown JSON parsing resulted in non-dict type: {type(parsed_json)}. Failing.")
                # This will lead to the generic "All parsing attempts failed" if XML also fails.
            else: # It is a dict
                logger.debug("Successfully parsed JSON from Markdown code block.")
                return parsed_json
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing from Markdown failed: {e}. Trying XML-like parsing.")
    else:
        logger.debug("No Markdown JSON code block found. Trying XML-like parsing.")
            
    # Attempt 3: Parse simple, flat XML-like key-value pairs
    try:
        xml_to_parse = raw_response.strip()

        if not xml_to_parse.startswith("<"):
            logger.debug(f"String for XML parsing does not start with '<'. Content: {xml_to_parse[:50]}")
            raise ET.ParseError("String does not start with '<', not valid XML.")
        
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
                    logger.debug(f"XML-like parsing failed after attempting to wrap original response: {e_wrapped}")
                    raise LLMResponseParseError(f"Failed to parse XML-like content: {e_wrapped}", raw_response=raw_response) from e_wrapped
            else: 
                logger.debug(f"XML-like parsing failed (was already wrapped or initial parse failed): {e_initial}")
                raise e_initial 
        
        if root is None: 
            raise LLMResponseParseError("XML root element could not be determined after parsing attempts.", raw_response=raw_response)

        xml_dict = {}
        elements_to_process = []
        
        current_processing_candidate = root
        if root.tag == 'root' and xml_to_parse.startswith("<root>") and len(list(root)) == 1:
            current_processing_candidate = root[0]

        if current_processing_candidate.text and current_processing_candidate.text.strip() and len(list(current_processing_candidate)) > 0:
            elements_to_process = [current_processing_candidate]
        elif len(list(current_processing_candidate)) > 0:
            elements_to_process = list(current_processing_candidate)
        else:
            elements_to_process = [current_processing_candidate]
        
        for element in elements_to_process:
            if element.tag: 
                if element.attrib: 
                    logger.debug(f"XML element '{element.tag}' has attributes {element.attrib}, failing simple XML parse.")
                    raise LLMResponseParseError(f"XML element '{element.tag}' has attributes, which is not supported by simple parser.", raw_response=raw_response)
                
                if len(list(element)) == 0: 
                    xml_dict[element.tag] = element.text.strip() if element.text else ""
                else: 
                    logger.warning(f"XML element '{element.tag}' has child elements, which is not flat. Raising error for simple parser.")
                    raise LLMResponseParseError(f"XML element '{element.tag}' has child elements, which is not supported by simple flat parser.", raw_response=raw_response)
            
        if xml_dict:
            logger.debug(f"Successfully parsed simple XML-like response: {xml_dict}")
            return xml_dict
        else:
            if raw_response.strip() and not (root.tag == 'root' and not list(root) and not root.text and not root.attrib and len(elements_to_process)==0) : 
                 logger.debug("XML parsed to an empty dictionary, but input was not an empty root. Input: " + raw_response[:50])
                 # This case might indicate a valid but empty XML structure that didn't yield key-value pairs.
                 # Depending on strictness, this could be an error or an empty dict.
                 # For now, let it fall through to the final error if no dict is produced.
            elif logger: 
                logger.debug("XML input (e.g. <root/>) resulted in an empty dictionary.")
            # If xml_dict is empty after trying to parse non-empty XML, it's a failure for this strategy.
            if not xml_dict and raw_response.strip(): # Check raw_response again to ensure it wasn't just an empty root tag
                 pass # Let it fall through to the final error

    except ET.ParseError as e: 
        logger.debug(f"XML-like parsing failed with ET.ParseError: {e}. Raw input: {raw_response[:100]}")
    except LLMResponseParseError: # Re-raise if it's one of our specific XML parsing errors
        raise
    except Exception as e_gen: 
        logger.error(f"Unexpected error during XML parsing: {e_gen}", exc_info=True)
        raise LLMResponseParseError(f"Unexpected error during XML processing: {e_gen}", raw_response=raw_response) from e_gen

    # If all attempts fail
    error_message = "All parsing attempts failed (direct JSON, Markdown JSON, simple XML)."
    logger.error(error_message + f" Raw response: {raw_response[:200]}")
    raise LLMResponseParseError(error_message, raw_response=raw_response)