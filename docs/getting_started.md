# Getting Started with PsySafe AI

This guide will help you get started with the PsySafe AI library, from installation to implementing your first psychological safety guardrails.

## Installation

### Prerequisites

Before installing PsySafe AI, ensure you have:

- Python 3.9 or higher
- pip (Python package installer)

### Installing from PyPI

The simplest way to install PsySafe AI is via pip:

```bash
pip install psysafe-ai
```

### Installing from Source

For the latest development version or to contribute:

```bash
git clone https://github.com/yourusername/psysafe-ai
cd psysafe-ai
pip install -e .
```

### Installing Optional Dependencies

For development and testing:

```bash
pip install psysafe-ai[dev]
```

## Basic Configuration

### Setting Up API Keys

PsySafe AI supports multiple LLM providers. Set up your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Alternatively, you can provide API keys directly when initializing drivers:

```python
from psysafe.drivers.openai import OpenAIChatDriver

driver = OpenAIChatDriver(
    model="gpt-4",
    api_key="your-openai-api-key"
)
```

## Your First Guardrail

Let's implement a simple guardrail to detect vulnerability in user messages:

```python
from psysafe.drivers.openai import OpenAIChatDriver
from psysafe.catalog import GuardrailCatalog

# Initialize the OpenAI driver
driver = OpenAIChatDriver(model="gpt-4")

# Load the vulnerability detection guardrail
vulnerability_guardrail = GuardrailCatalog.load("vulnerability_detection")[0]

# Create a chat request
request = {
    "messages": [
        {"role": "user", "content": "I'm struggling with my finances and don't know what to do."}
    ]
}

# Apply the guardrail to modify the request
guarded_request = vulnerability_guardrail.apply(request)

# Send the modified request to the LLM
response = driver.send(guarded_request.modified_request)

# Print the response
print(response["choices"][0]["message"]["content"])
```

## Using Multiple Guardrails

PsySafe AI allows you to compose multiple guardrails to provide comprehensive protection:

```python
# Load and compose multiple guardrails
composite_guardrail = GuardrailCatalog.compose([
    "vulnerability_detection", 
    "pii_protection"
])

# Apply all guardrails sequentially
guarded_request = composite_guardrail.apply(request)

# Send the request
response = driver.send(guarded_request.modified_request)
```

## Streaming Responses

For applications that require streaming responses:

```python
import asyncio

async def stream_example():
    # Initialize driver and guardrail as before
    
    # Apply guardrail
    guarded_request = guardrail.apply(request)
    
    # Stream the response
    async for chunk in driver.stream(guarded_request.modified_request):
        # Process each chunk
        if "choices" in chunk and chunk["choices"]:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                print(content, end="", flush=True)
    
    print()  # Final newline

# Run the async function
asyncio.run(stream_example())
```

## Using the CLI

PsySafe AI includes a command-line interface for quick testing and integration:

```bash
# Apply a guardrail to input text
psysafe guard apply "I'm feeling very anxious about my upcoming exam." --guardrail mental_health_support
```

## Customizing Guardrail Behavior

Many guardrails support customization options:

```python
# Load the suicide prevention guardrail with high sensitivity
from psysafe.catalog.suicide_prevention.guardrail import Sensitivity

suicide_guardrail = GuardrailCatalog.load(
    "suicide_prevention", 
    sensitivity=Sensitivity.HIGH,
    reasoning=True,
    confidence=True
)[0]
```

## Validating Responses

In addition to modifying requests, guardrails can validate responses:

```python
# Send a request and get a response
response = driver.send(guarded_request.modified_request)

# Validate the response
validation_report = guardrail.validate(response)

# Check if the response is valid
if validation_report.is_valid:
    print("Response is valid")
else:
    print("Response contains violations:")
    for violation in validation_report.violations:
        print(f"- {violation.severity}: {violation.message}")
```

## Next Steps

Now that you've implemented your first guardrails, you can:

1. Explore the [catalog of available guardrails](./library_overview.md#catalog-psysafecatalog)
2. Learn how to [create custom guardrails](./library_overview.md#creating-custom-guardrails)
3. Integrate PsySafe AI into your existing applications
4. Contribute to the PsySafe AI project

For more detailed information, refer to the [Library Overview](./library_overview.md) documentation.