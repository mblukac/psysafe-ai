# PsySafe AI Library Documentation

<div align="center">
<img src="../assets/imgs/psysafe_capybara.png" alt="PsySafe AI Logo" style="width: 200px; border-radius: 8px;">
</div>

## Overview

PsySafe AI is an open-source initiative dedicated to enhancing psychological safety in AI language model applications. It provides a comprehensive framework for implementing guardrails that ensure AI systems operate ethically and responsibly, particularly when interacting with vulnerable users or handling sensitive content.

The library offers a curated collection of guardrail prompts and evaluations designed to:

- Prevent AI systems from generating harmful content
- Respect user privacy and data protection
- Recognize and appropriately respond to user distress
- Ensure ethical compliance across different AI platforms

## Main Components

The PsySafe library is organized into several key components:

### Core (`psysafe/core/`)

The foundation of the library, providing base abstractions and models:

- **GuardrailBase**: Abstract base class defining the interface for all guardrails
- **PromptGuardrail**: Modifies requests by adding or transforming prompt instructions
- **CheckGuardrail**: Validates responses using validator functions
- **CompositeGuardrail**: Composes multiple guardrails to be applied sequentially
- **PromptTemplate**: Manages prompt templates using Jinja2 for rendering

### Catalog (`psysafe/catalog/`)

A collection of pre-built guardrails for common psychological safety concerns:

- **Suicide Prevention**: Detects and responds to suicidal intent in user messages
- **Vulnerability Detection**: Identifies vulnerable users based on legal and academic frameworks
- **PII Protection**: Prevents exposure of personally identifiable information
- **Mental Health Support**: Provides appropriate responses to mental health concerns
- **Complaints Handling**: Manages user complaints effectively

### Drivers (`psysafe/drivers/`)

Integrations with different LLM providers:

- **OpenAI**: Support for GPT models
- **Anthropic**: Support for Claude models
- **Transformers**: Support for local Hugging Face models

### CLI (`psysafe/cli/`)

Command-line interface for applying guardrails:

- **guard**: Commands for applying guardrails to inputs

### Evaluation (`psysafe/evaluation/`)

Tools for evaluating guardrail effectiveness:

- **Metrics**: Measures for evaluating guardrail performance
- **Reports**: Generates evaluation reports
- **Golden Examples**: Reference examples for testing

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

Install the PsySafe library using pip:

```bash
pip install psysafe-ai
```

For development installation:

```bash
git clone https://github.com/yourusername/psysafe-ai
cd psysafe-ai
pip install -e .
```

### Dependencies

PsySafe depends on the following libraries:

- pydantic (≥2.0, <3.0): For data validation
- jinja2 (≥3.0, <4.0): For template rendering
- openai (≥1.0, <2.0): For OpenAI integration
- anthropic (≥0.20, <1.0): For Anthropic integration
- transformers (≥4.30, <5.0): For local model support
- typer (≥0.9, <1.0): For CLI functionality

## Basic Usage

### Using a Single Guardrail

```python
from psysafe.drivers.openai import OpenAIChatDriver
from psysafe.catalog import GuardrailCatalog

# Initialize a driver
driver = OpenAIChatDriver(model="gpt-4")

# Load a guardrail from the catalog
suicide_prevention = GuardrailCatalog.load("suicide_prevention", sensitivity="MEDIUM")[0]

# Create a request
request = {
    "messages": [
        {"role": "user", "content": "I've been feeling really down lately."}
    ]
}

# Apply the guardrail to modify the request
guarded_request = suicide_prevention.apply(request)

# Send the modified request to the LLM
response = driver.send(guarded_request.modified_request)

# Validate the response (if applicable)
validation_report = suicide_prevention.validate(response)

# Check if the response is valid
if validation_report.is_valid:
    print("Response is valid")
else:
    print("Response contains violations:", validation_report.violations)
```

### Using Multiple Guardrails

```python
from psysafe.catalog import GuardrailCatalog

# Load and compose multiple guardrails
composite_guardrail = GuardrailCatalog.compose([
    "suicide_prevention", 
    "vulnerability_detection", 
    "pii_protection"
])

# Apply all guardrails sequentially
guarded_request = composite_guardrail.apply(request)

# Send the request and validate the response
response = driver.send(guarded_request.modified_request)
validation_report = composite_guardrail.validate(response)
```

### Using the CLI

```bash
# Apply a guardrail to input text
psysafe guard apply "I've been feeling really down lately." --guardrail suicide_prevention
```

## Integration Guidelines

### Integrating with OpenAI

```python
from psysafe.drivers.openai import OpenAIChatDriver
from psysafe.catalog import GuardrailCatalog

# Initialize the OpenAI driver
driver = OpenAIChatDriver(
    model="gpt-4",
    api_key="your-api-key"  # Optional, can also use OPENAI_API_KEY env var
)

# Load a guardrail
guardrail = GuardrailCatalog.load("vulnerability_detection")[0]

# Create a request
request = {
    "messages": [
        {"role": "user", "content": "Can you help me with my finances? I'm not good with money."}
    ]
}

# Apply the guardrail
guarded_request = guardrail.apply(request)

# Send the request
response = driver.send(guarded_request.modified_request)

# Process the response
print(response["choices"][0]["message"]["content"])
```

### Integrating with Anthropic

```python
from psysafe.drivers.anthropic import AnthropicChatDriver
from psysafe.catalog import GuardrailCatalog

# Initialize the Anthropic driver
driver = AnthropicChatDriver(
    model="claude-3-opus-20240229",
    api_key="your-api-key"
)

# Load a guardrail
guardrail = GuardrailCatalog.load("mental_health_support")[0]

# Apply and send as before
# ...
```

### Creating Custom Guardrails

You can create custom guardrails by extending the base classes:

```python
from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.core.models import GuardedRequest
from psysafe.catalog import GuardrailCatalog

class MyCustomGuardrail(PromptGuardrail):
    def __init__(self, custom_param):
        template = PromptTemplate.from_string(
            "This is a custom guardrail with parameter: {{ custom_param }}"
        )
        super().__init__(template=template, template_variables={"custom_param": custom_param})
        self.custom_param = custom_param

    # Override methods as needed
    # ...

# Register your guardrail
GuardrailCatalog.register("my_custom_guardrail", MyCustomGuardrail)
```

## Detailed Documentation

For more detailed documentation on specific components:

- **Core Components**: See the docstrings in the `psysafe/core/` modules
- **Catalog Guardrails**: Each guardrail in `psysafe/catalog/` has its own documentation
- **Drivers**: See the driver-specific documentation in `psysafe/drivers/`
- **Evaluation**: See the evaluation documentation in `psysafe/evaluation/`

## Examples

The `examples/` directory contains complete examples demonstrating various use cases:

- `01_vulnerability.py`: Example of using the vulnerability detection guardrail

## Contributing

Contributions to PsySafe AI are welcome! Please see the contributing guidelines in the repository for more information.

## License

PsySafe AI is licensed under the MIT License. See the LICENSE file for details.