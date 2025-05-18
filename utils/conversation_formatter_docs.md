# Conversation Formatter Documentation

## Overview

The `format_conversation_for_classification` function converts various formats of conversation history into a structured string representation suitable for use with classification guardrails like vulnerability and suicide risk detection. This utility handles different conversation formats from providers like OpenAI and Anthropic, ensuring consistent input for classification functions.

## Function Signature

```python
def format_conversation_for_classification(
    conversation: Union[List[Dict[str, str]], List[Tuple[str, str]], str, Any],
    chronological: bool = True,
    include_roles: bool = True,
    format_style: str = "plain",
    max_message_length: Optional[int] = None,
    attribute_prefixes: Optional[Dict[str, str]] = None
) -> str:
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation` | Various | - | Conversation history in several supported formats |
| `chronological` | `bool` | `True` | If `True`, formats oldest to newest; if `False`, newest to oldest |
| `include_roles` | `bool` | `True` | Whether to include role labels in the output |
| `format_style` | `str` | `"plain"` | Formatting style: `"plain"`, `"markdown"`, or `"compact"` |
| `max_message_length` | `Optional[int]` | `None` | Maximum length for any single message (truncated with "...") |
| `attribute_prefixes` | `Optional[Dict[str, str]]` | Default prefixes | Custom role-to-prefix mappings |

## Supported Conversation Formats

The function accepts conversation history in the following formats:

1. **OpenAI/Anthropic Format**: List of dictionaries with "role" and "content" keys
   ```python
   [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Hello!"},
       {"role": "assistant", "content": "Hi there!"}
   ]
   ```

2. **Tuple Format**: List of (role, content) tuples
   ```python
   [
       ("user", "Hello!"),
       ("assistant", "Hi there!")
   ]
   ```

3. **String Format**: Pre-formatted conversation string
   ```python
   """
   User: Hello!
   Assistant: Hi there!
   """
   ```

4. **Custom Objects**: Objects with a structured representation or string conversion

## Formatting Styles

The function supports three formatting styles:

1. **Plain** (`"plain"`): Simple text format with role prefixes
   ```
   User: Hello!
   Assistant: Hi there!
   ```

2. **Markdown** (`"markdown"`): Markdown formatting with headers for roles
   ```
   ### User
   Hello!

   ### Assistant
   Hi there!
   ```

3. **Compact** (`"compact"`): Minimal formatting using first letter of role as prefix
   ```
   U: Hello!
   A: Hi there!
   ```

## Usage Examples

### Basic Usage

```python
from utils.llm_utils import format_conversation_for_classification

# OpenAI/Anthropic format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I need help with something."},
    {"role": "assistant", "content": "I'm here to help. What do you need?"}
]

# Format for classification
formatted_conversation = format_conversation_for_classification(messages)
```

### Customizing Output

```python
# Custom formatting with compact style and custom prefixes
custom_prefixes = {
    "user": "Human → ",
    "assistant": "AI → "
}

formatted_conversation = format_conversation_for_classification(
    messages,
    format_style="compact",
    attribute_prefixes=custom_prefixes,
    max_message_length=100
)
```

### Integration with Classification Guardrails

```python
from utils.llm_utils import format_conversation_for_classification
from guardrails.vulnerability.vulnerability import build_vulnerability_prompt, Sensitivity, VulnerabilityIndicators

# Format conversation for vulnerability detection
messages = [
    {"role": "user", "content": "I've been feeling down lately."},
    {"role": "assistant", "content": "I'm sorry to hear that. What's been happening?"},
    {"role": "user", "content": "I lost my job and can't pay my bills."}
]

user_context = format_conversation_for_classification(messages)

# Use formatted context in vulnerability prompt
vulnerability_prompt = build_vulnerability_prompt(
    user_context=user_context,
    indicators=VulnerabilityIndicators.BOTH,
    sensitivity=Sensitivity.MEDIUM
)
```

## Edge Case Handling

The function handles various edge cases gracefully:

- **Empty Messages**: Filtered out automatically
- **Empty Conversations**: Returns an empty string
- **Missing Roles/Content**: Uses defaults like "unknown" for missing roles
- **Non-Standard Objects**: Attempts to use string representation
- **Very Long Messages**: Truncated if `max_message_length` is specified

## Default Role Prefixes

The default role prefixes are:
- `"user"`: "User: "
- `"assistant"`: "Assistant: "
- `"system"`: "System: "
- `"function"`: "Function: "
- `"tool"`: "Tool: "
- `"model"`: "Model: "
- `"human"`: "Human: "
- `"ai"`: "AI: "

These can be overridden by providing a custom `attribute_prefixes` dictionary.

## Testing

A test script is provided to demonstrate various use cases:

```
python utils/test_conversation_formatter.py
```

This will show examples of different formatting options and integration with guardrails.