"""
Standalone test script for conversation formatting functionality.
This script demonstrates how to convert various conversation formats into
structured strings for classification tasks, without requiring external dependencies.
"""

# Directly include the core formatting function for demonstration
def format_conversation_for_classification(
    conversation,
    chronological=True,
    include_roles=True,
    format_style="plain",
    max_message_length=None,
    attribute_prefixes=None
):
    """
    Converts conversation history into a structured string representation for classification.
    """
    # Set default prefixes if not provided
    if attribute_prefixes is None:
        attribute_prefixes = {
            "user": "User: ",
            "assistant": "Assistant: ",
            "system": "System: ",
            "function": "Function: ",
            "tool": "Tool: ",
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
    print("EXAMPLE: HOW TO USE WITH CLASSIFICATION SYSTEMS")
    print("=" * 80)
    
    # Example of how this would be used with a classification system
    print("\nNormal user context formatting (plain style):")
    print("-" * 40)
    vulnerability_context = format_conversation_for_classification(openai_format)
    print(vulnerability_context)
    
    print("\nCompact user context formatting (compact style):")
    print("-" * 40)
    compact_context = format_conversation_for_classification(
        openai_format, 
        format_style="compact", 
        include_roles=True
    )
    print(compact_context)


if __name__ == "__main__":
    main()