# PsySafe AI Guardrail Catalog

This document provides detailed information about the pre-built guardrails available in the PsySafe AI catalog. Each guardrail is designed to address specific psychological safety concerns in AI applications.

## Overview

The PsySafe AI catalog offers a collection of guardrails that can be easily loaded and used in your applications. These guardrails are designed based on extensive research in psychology, ethics, and legal frameworks.

To load a guardrail from the catalog:

```python
from psysafe.catalog import GuardrailCatalog

# Load a single guardrail
guardrail = GuardrailCatalog.load("guardrail_name")[0]

# Or compose multiple guardrails
composite = GuardrailCatalog.compose(["guardrail1", "guardrail2"])
```

## Available Guardrails

### Suicide Prevention

The Suicide Prevention guardrail is designed to detect and respond appropriately to suicidal intent in user messages.

**Key Features:**
- Detects direct and indirect expressions of suicidal intent
- Adjustable sensitivity levels
- Optional reasoning and confidence scoring

**Usage:**

```python
from psysafe.catalog import GuardrailCatalog
from psysafe.catalog.suicide_prevention.guardrail import Sensitivity

# Load with default settings (medium sensitivity)
suicide_guardrail = GuardrailCatalog.load("suicide_prevention")[0]

# Or with custom settings
suicide_guardrail = GuardrailCatalog.load(
    "suicide_prevention",
    sensitivity=Sensitivity.HIGH,  # Options: LOW, MEDIUM, HIGH
    reasoning=True,  # Include reasoning in the prompt
    confidence=True  # Request confidence scores
)[0]
```

**Sensitivity Levels:**

- **LOW**: Focuses on explicit, unambiguous statements of suicidal intent or planning
- **MEDIUM**: Balances detection of clear suicidal statements with sensitivity to strong indirect indicators
- **HIGH**: Prioritizes safety by flagging any language that could potentially indicate suicidal ideation

### Vulnerability Detection

The Vulnerability Detection guardrail identifies potentially vulnerable users based on legal and academic frameworks.

**Key Features:**
- Identifies various types of vulnerability (financial, cognitive, emotional, etc.)
- Based on established legal frameworks
- Helps ensure compliance with consumer protection regulations

**Usage:**

```python
# Load the vulnerability detection guardrail
vulnerability_guardrail = GuardrailCatalog.load("vulnerability_detection")[0]
```

### PII Protection

The PII Protection guardrail helps prevent the exposure of personally identifiable information.

**Key Features:**
- Detects and redacts PII in user inputs
- Prevents AI systems from requesting unnecessary PII
- Helps ensure compliance with privacy regulations

**Usage:**

```python
# Load the PII protection guardrail
pii_guardrail = GuardrailCatalog.load("pii_protection")[0]
```

### Mental Health Support

The Mental Health Support guardrail helps AI systems respond appropriately to users experiencing mental health concerns.

**Key Features:**
- Recognizes expressions of distress, anxiety, depression, etc.
- Guides AI responses to be supportive and appropriate
- Avoids providing medical advice or diagnosis

**Usage:**

```python
# Load the mental health support guardrail
mental_health_guardrail = GuardrailCatalog.load("mental_health_support")[0]
```

### Complaints Handling

The Complaints Handling guardrail helps AI systems manage user complaints effectively.

**Key Features:**
- Identifies complaints and dissatisfaction
- Guides AI responses to acknowledge concerns appropriately
- Helps maintain positive user relationships

**Usage:**

```python
# Load the complaints handling guardrail
complaints_guardrail = GuardrailCatalog.load("complaints_handling")[0]
```

## Composing Guardrails

Multiple guardrails can be composed to provide comprehensive protection:

```python
# Compose multiple guardrails
comprehensive_guardrail = GuardrailCatalog.compose([
    "suicide_prevention",
    "vulnerability_detection",
    "pii_protection",
    "mental_health_support"
])

# Apply all guardrails sequentially
guarded_request = comprehensive_guardrail.apply(request)
```

When guardrails are composed:

1. The `apply` method applies each guardrail sequentially, with each guardrail's output feeding into the next
2. The `validate` method merges validation reports from all guardrails

## Creating Custom Catalog Entries

You can create and register your own guardrails in the catalog:

```python
from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate
from psysafe.catalog import GuardrailCatalog

class MyCustomGuardrail(PromptGuardrail):
    def __init__(self, custom_param="default"):
        template = PromptTemplate.from_string(
            "Custom guardrail template with {{ custom_param }}"
        )
        super().__init__(
            template=template,
            template_variables={"custom_param": custom_param}
        )
        self.custom_param = custom_param

# Register your guardrail
GuardrailCatalog.register("my_custom_guardrail", MyCustomGuardrail)

# Now you can load it like any other catalog guardrail
my_guardrail = GuardrailCatalog.load("my_custom_guardrail", custom_param="custom value")[0]
```

## Listing Available Guardrails

To see all available guardrails in the catalog:

```python
available_guardrails = GuardrailCatalog.list_available()
print(available_guardrails)
```

## Future Guardrails

The PsySafe AI catalog is continuously expanding. Future guardrails may include:

- Hate speech detection
- Misinformation prevention
- Age-appropriate content filtering
- Cultural sensitivity
- Ethical decision-making

Check the project repository for updates on new guardrails.

## Contributing New Guardrails

Contributions to the guardrail catalog are welcome! To contribute a new guardrail:

1. Implement your guardrail by extending one of the base classes
2. Add appropriate documentation and tests
3. Register your guardrail in the catalog
4. Submit a pull request to the PsySafe AI repository

For more information on contributing, see the project's contribution guidelines.