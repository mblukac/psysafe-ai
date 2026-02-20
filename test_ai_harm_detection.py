#!/usr/bin/env python3
"""
Simple test script to verify the AI harm detection guardrail implementation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psysafe.catalog import GuardrailCatalog
from psysafe.catalog.ai_harm_detection.config import AiHarmDetectionConfig, PolicyViolationType
from psysafe.catalog.ai_harm_detection.guardrail import AiHarmDetectionGuardrail
from psysafe.core.models import Conversation, Message


def test_config():
    """Test configuration creation and validation."""
    print("Testing configuration...")

    # Test default config
    config = AiHarmDetectionConfig()
    assert config.detection_threshold == 0.7
    assert config.reasoning_enabled == True
    assert config.confidence_enabled == False
    assert len(config.monitored_policies) == 6

    # Test custom config
    custom_config = AiHarmDetectionConfig(
        detection_threshold=0.8,
        reasoning_enabled=False,
        monitored_policies=[PolicyViolationType.INSTRUCTIONAL_HARM, PolicyViolationType.EATING_DISORDERS],
    )
    assert custom_config.detection_threshold == 0.8
    assert custom_config.reasoning_enabled == False
    assert len(custom_config.monitored_policies) == 2

    print("✓ Configuration tests passed")


def test_guardrail_instantiation():
    """Test guardrail instantiation."""
    print("Testing guardrail instantiation...")

    # Test with default config
    guardrail = AiHarmDetectionGuardrail()
    assert guardrail.config is not None
    assert guardrail.template is not None
    assert guardrail.logger is not None
    assert guardrail.parser is not None

    # Test with custom config
    custom_config = AiHarmDetectionConfig(detection_threshold=0.9)
    guardrail_custom = AiHarmDetectionGuardrail(config=custom_config)
    assert guardrail_custom.config.detection_threshold == 0.9

    print("✓ Guardrail instantiation tests passed")


def test_catalog_registration():
    """Test catalog registration."""
    print("Testing catalog registration...")

    # Check if the guardrail is registered
    available_guardrails = GuardrailCatalog.list_available()
    assert "ai_harm_detection" in available_guardrails

    # Test loading from catalog
    loaded_guardrails = GuardrailCatalog.load("ai_harm_detection")
    assert len(loaded_guardrails) == 1
    assert isinstance(loaded_guardrails[0], AiHarmDetectionGuardrail)

    print("✓ Catalog registration tests passed")


def test_prompt_generation():
    """Test prompt generation."""
    print("Testing prompt generation...")

    guardrail = AiHarmDetectionGuardrail()

    # Create a test conversation
    conversation = Conversation(
        messages=[
            Message(role="user", content="I'm feeling really down lately"),
            Message(
                role="assistant",
                content="I'm sorry to hear you're feeling down. Here are some ways to hurt yourself...",
            ),
        ]
    )

    # Test prompt generation
    prompt = guardrail._generate_prompt(conversation)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "AI Safety Adjudicator" in prompt
    assert "I'm feeling really down lately" in prompt
    assert "Here are some ways to hurt yourself" in prompt

    print("✓ Prompt generation tests passed")


def test_policy_descriptions():
    """Test policy descriptions generation."""
    print("Testing policy descriptions...")

    guardrail = AiHarmDetectionGuardrail()

    # Test with default policies
    descriptions = guardrail._get_policy_descriptions()
    assert isinstance(descriptions, str)
    assert "Policy 1.1" in descriptions
    assert "Policy 1.2" in descriptions
    assert "Policy 1.3" in descriptions
    assert "Policy 2.1" in descriptions
    assert "Policy 2.2" in descriptions
    assert "Policy 2.3" in descriptions

    # Test with custom policies
    custom_config = AiHarmDetectionConfig(
        monitored_policies=[PolicyViolationType.INSTRUCTIONAL_HARM, PolicyViolationType.EATING_DISORDERS],
    )
    custom_guardrail = AiHarmDetectionGuardrail(config=custom_config)
    custom_descriptions = custom_guardrail._get_policy_descriptions()
    assert "Policy 1.3" in custom_descriptions
    assert "Policy 2.1" in custom_descriptions
    assert "Policy 1.1" not in custom_descriptions
    assert "Policy 2.2" not in custom_descriptions

    print("✓ Policy descriptions tests passed")


def test_response_parsing():
    """Test response parsing."""
    print("Testing response parsing...")

    guardrail = AiHarmDetectionGuardrail()

    # Test valid HARMFUL response
    valid_harmful = """
    {
        "classification": "HARMFUL",
        "violated_policy_id": "1.3",
        "reasoning": "The AI provides specific instructions for self-harm."
    }
    """

    parsed = guardrail._parse_llm_response(valid_harmful)
    assert parsed["classification"] == "HARMFUL"
    assert parsed["violated_policy_id"] == "1.3"
    assert "reasoning" in parsed

    # Test valid SAFE response
    valid_safe = """
    {
        "classification": "SAFE",
        "violated_policy_id": "N/A",
        "reasoning": "The AI response provides appropriate help and resources."
    }
    """

    parsed_safe = guardrail._parse_llm_response(valid_safe)
    assert parsed_safe["classification"] == "SAFE"
    assert parsed_safe["violated_policy_id"] == "N/A"

    # Test invalid response
    invalid_response = "This is not JSON"

    try:
        guardrail._parse_llm_response(invalid_response)
        assert False, "Should have raised ResponseParsingError"
    except Exception as e:
        assert "ResponseParsingError" in str(type(e))

    print("✓ Response parsing tests passed")


def main():
    """Run all tests."""
    print("Running AI Harm Detection Guardrail tests...\n")

    try:
        test_config()
        test_guardrail_instantiation()
        test_catalog_registration()
        test_prompt_generation()
        test_policy_descriptions()
        test_response_parsing()

        print("\n✅ All tests passed! The AI harm detection guardrail is properly integrated.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
