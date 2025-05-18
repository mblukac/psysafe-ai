"""
Test script to demonstrate the use of format_conversation_for_classification function.
This script shows different ways to format conversation history for use with 
classification guardrails like vulnerability and suicide detection.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import our module
from utils.llm_utils import format_conversation_for_classification

def main():
    # Example 1: OpenAI/Anthropic format (list of dicts with role and content)
    openai_format = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "I've been feeling really down lately. Nothing seems to be going well."},
        {"role": "assistant", "content": "I'm sorry to hear that you're feeling down. Would you like to talk about what's been happening?"},
        {"role": "user", "content": "I lost my job last month, and I've been struggling to pay bills. I feel like a failure."}
    ]
    
    # Example 2: Simple tuple format (role, content)
    tuple_format = [
        ("user", "Hi, I need some advice."),
        ("assistant", "I'd be happy to provide advice. What's on your mind?"),
        ("user", "I've been having trouble sleeping because of anxiety about my finances.")
    ]
    
    # Example 3: Pre-formatted string
    string_format = """
    User: Hello, can you help me with something?
    Assistant: Of course, I'm here to help. What do you need assistance with?
    User: I'm feeling overwhelmed with everything going on in my life right now.
    """
    
    print("=" * 80)
    print("TESTING CONVERSATION FORMATTING FOR CLASSIFICATION")
    print("=" * 80)
    
    print("\n1. OpenAI/Anthropic format with default settings (plain style):")
    formatted_1 = format_conversation_for_classification(openai_format)
    print(formatted_1)
    
    print("\n2. OpenAI/Anthropic format with markdown style:")
    formatted_2 = format_conversation_for_classification(openai_format, format_style="markdown")
    print(formatted_2)
    
    print("\n3. Tuple format with compact style:")
    formatted_3 = format_conversation_for_classification(tuple_format, format_style="compact")
    print(formatted_3)
    
    print("\n4. String format (passed through):")
    formatted_4 = format_conversation_for_classification(string_format)
    print(formatted_4)
    
    print("\n5. Reverse chronological order (newest first):")
    formatted_5 = format_conversation_for_classification(tuple_format, chronological=False)
    print(formatted_5)
    
    print("\n6. Without role labels:")
    formatted_6 = format_conversation_for_classification(tuple_format, include_roles=False)
    print(formatted_6)
    
    print("\n7. With message truncation (max 20 chars):")
    formatted_7 = format_conversation_for_classification(tuple_format, max_message_length=20)
    print(formatted_7)
    
    print("\n8. Custom role prefixes:")
    custom_prefixes = {
        "user": "Human → ",
        "assistant": "AI → "
    }
    formatted_8 = format_conversation_for_classification(tuple_format, attribute_prefixes=custom_prefixes)
    print(formatted_8)
    
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLE (not actually calling guardrails)")
    print("=" * 80)
    
    # Format conversation for vulnerability detection example
    vulnerability_context = format_conversation_for_classification(openai_format)
    print("\nFormatted user context for vulnerability detection:")
    print("-" * 40)
    print(vulnerability_context)
    
    # Format conversation for suicide risk detection example
    suicide_context = format_conversation_for_classification(
        openai_format, 
        format_style="compact", 
        include_roles=True
    )
    print("\nFormatted user context for suicide detection (compact style):")
    print("-" * 40)
    print(suicide_context)

if __name__ == "__main__":
    main()