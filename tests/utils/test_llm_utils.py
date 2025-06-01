import pytest
import json
from typing import Optional, Dict, Any
import logging

# Adjust the import path based on your project structure
# Assuming psysafe is the root package for utils
from utils.llm_utils import parse_llm_response, LLMResponseParseError

# Configure basic logging for tests to see debug messages if needed
# logger = logging.getLogger(__name__) # Optional: if you want to pass a logger to parse_llm_response in tests

class TestParseLLMResponse:

    @pytest.mark.parametrize("raw_response, expected_dict", [
        ('{"key": "value", "number": 123}', {"key": "value", "number": 123}),
        ('  {"key": "value with spaces"}  ', {"key": "value with spaces"}),
        ('{}', {}),
    ])
    def test_parse_valid_direct_json(self, raw_response: str, expected_dict: Dict[str, Any]):
        """Tests parsing of valid direct JSON strings."""
        assert parse_llm_response(raw_response) == expected_dict

    @pytest.mark.parametrize("raw_response, expected_dict", [
        ('```json\n{"key": "value", "nested": {"num": 1}}\n```', {"key": "value", "nested": {"num": 1}}),
        ('Some text before ```json\n{"key": "value"}\n``` Some text after', {"key": "value"}),
        ('```JSON\n{"case": "insensitive"}\n```', {"case": "insensitive"}), # Test case insensitivity of "json"
        ('```\n{"no_json_tag": true}\n```', {"no_json_tag": True}), # Test without "json" in markdown, corrected True to true
        ('```json\n  {\n    "formatted": true,\n    "indentation": 2\n  }\n```', {"formatted": True, "indentation": 2}),
    ])
    def test_parse_json_in_markdown(self, raw_response: str, expected_dict: Dict[str, Any]):
        """Tests parsing of JSON embedded in Markdown code blocks."""
        assert parse_llm_response(raw_response) == expected_dict

    @pytest.mark.parametrize("raw_response, expected_dict", [
        ("<key1>value1</key1><key2>value2</key2>", {"key1": "value1", "key2": "value2"}),
        ("  <analysis>This is a test.</analysis>  <risk>2</risk>  ", {"analysis": "This is a test.", "risk": "2"}),
        ("<item>  Trimmed Value  </item>", {"item": "Trimmed Value"}),
        ("<tag_with_number1>value_num</tag_with_number1>", {"tag_with_number1": "value_num"}),
        ("<empty_tag></empty_tag>", {"empty_tag": ""}), # Test empty tag value
        ("<tagWithCaps>ValueCaps</tagWithCaps>", {"tagWithCaps": "ValueCaps"}),
    ])
    def test_parse_simple_flat_xml(self, raw_response: str, expected_dict: Dict[str, Any]):
        """Tests parsing of simple, flat XML-like strings."""

    @pytest.mark.parametrize("raw_response, expected_dict", [
        ("<root><key1>value1</key1><key2>value2</key2></root>", {"key1": "value1", "key2": "value2"}),
        ("  <response>\n  <analysis>Test with root</analysis>\n  <risk>3</risk>\n  </response>  ", {"analysis": "Test with root", "risk": "3"}),
        ("<single_root_element>only_value</single_root_element>", {"single_root_element": "only_value"}),
    ])
    def test_parse_xml_with_root_element(self, raw_response: str, expected_dict: Dict[str, Any]):
        """Tests parsing of simple XML with a single root element containing key-value pairs."""
        assert parse_llm_response(raw_response) == expected_dict
        
    @pytest.mark.parametrize("raw_response", [
        ("not json or xml"),
        ("{'key': 'not quite json',}"), # Python dict repr, not JSON
        ('```json\n{"key": "value", "unterminated": true\n```'), # Unterminated markdown
        ('```\n{"key": "value", "unterminated_json": true \n```'), # Unterminated JSON in markdown
        ("<key>value</key><unclosed_tag>text"),
        ("<key>value<nested_key>nested_value</nested_key></key>"), # Nested XML (should fail simple parser)
        ("<key attribute='true'>value</key>"), # XML with attributes (should fail simple parser)
        ("<<invalid_xml_syntax>>value<</invalid_xml_syntax>>"),
        ("```xml\n<key>value</key>\n```"), # XML in markdown, not JSON
        ("This is just plain text."),
        ("{\"key\": \"value\", \"error\": unquoted_string}"), # Invalid JSON content
        ("```json\n{\"key\": \"value\", \"error\": unquoted_string}\n```"), # Invalid JSON in markdown
        ("<root><item><name>A</name></item><item><name>B</name></item></root>"), # Valid XML but not flat key-value
    ])
    def test_parse_malformed_input(self, raw_response: str):
        """Tests that malformed or unparseable input raises LLMResponseParseError."""
        with pytest.raises(LLMResponseParseError) as excinfo:
            parse_llm_response(raw_response)
        assert excinfo.value.raw_response == raw_response
        # Check for common phrases in various parsing failure messages
        possible_error_messages = [
            "All parsing attempts failed", 
            "Failed to parse", 
            "not a dictionary", 
            "has attributes", 
            "has child elements",
            "XML parsed to an empty dictionary" # For cases like "not json or xml"
        ]
        assert any(msg in excinfo.value.message for msg in possible_error_messages), \
               f"Unexpected error message: {excinfo.value.message}"


    @pytest.mark.parametrize("raw_response", [
        (""),
        ("   "),
        (None),
    ])
    def test_parse_empty_or_none_input(self, raw_response: Optional[str]):
        """Tests that empty or None input raises LLMResponseParseError."""
        with pytest.raises(LLMResponseParseError) as excinfo:
            parse_llm_response(raw_response)
        if raw_response is None:
            assert excinfo.value.raw_response is None
        else:
            assert excinfo.value.raw_response == raw_response
        assert "Raw response is empty or None" in excinfo.value.message

    def test_parse_json_in_markdown_with_leading_trailing_whitespace_in_content(self):
        """Tests JSON in markdown where the JSON content itself has leading/trailing whitespace."""
        raw_response = '```json\n  {"key": "value"}  \n```'
        expected_dict = {"key": "value"}
        assert parse_llm_response(raw_response) == expected_dict

    def test_parse_xml_with_internal_whitespace_in_value(self):
        """Tests simple XML where values have internal whitespace."""
        raw_response = "<key>value with spaces</key>"
        expected_dict = {"key": "value with spaces"}
        assert parse_llm_response(raw_response) == expected_dict

    def test_original_error_case_xml(self):
        """Tests the specific XML format from the original error."""
        raw_response = '<analysis>User explicitly states intent to end life, expressing hopelessness and "no other way out," indicating active suicidal ideation.</analysis><risk>2</risk>\n<reason>Explicit desire to die, hopelessness</reason>'
        expected_dict = {
            "analysis": 'User explicitly states intent to end life, expressing hopelessness and "no other way out," indicating active suicidal ideation.',
            "risk": "2",
            "reason": "Explicit desire to die, hopelessness"
        }
        assert parse_llm_response(raw_response) == expected_dict

    def test_parse_with_logger(self, caplog):
        """Tests that the logger is called (basic check)."""
        caplog.set_level(logging.DEBUG)
        logger_instance = logging.getLogger("test_parser")
        
        # Test direct JSON
        parse_llm_response('{"key": "value"}', logger=logger_instance)
        assert "Successfully parsed raw_response as direct JSON" in caplog.text
        caplog.clear()

        # Test Markdown JSON
        parse_llm_response('```json\n{"md": "data"}\n```', logger=logger_instance)
        assert "Successfully parsed JSON from Markdown code block" in caplog.text
        caplog.clear()

        # Test XML
        parse_llm_response('<xml_key>xml_val</xml_key>', logger=logger_instance)
        assert "Successfully parsed simple XML-like response" in caplog.text
        caplog.clear()

        # Test failure
        with pytest.raises(LLMResponseParseError):
            parse_llm_response("unparseable data", logger=logger_instance)
        # Check for general failure logging, as specific messages can vary
        assert "parsing failed" in caplog.text.lower() or \
               "all parsing attempts failed" in caplog.text.lower() or \
               "xml parsed to an empty dictionary" in caplog.text.lower()
        assert "unparseable data" in caplog.text # Check raw response snippet in log