# PsySafe AI Architecture and Design

This document explains the architecture and design principles of the PsySafe AI library, providing insights into how the components work together and how to extend the library.

## Architectural Overview

PsySafe AI follows a modular, extensible architecture designed to provide psychological safety guardrails for AI language model applications. The architecture consists of several key layers:

1. **Core Layer**: Defines the fundamental abstractions and interfaces
2. **Drivers Layer**: Provides integrations with different LLM providers
3. **Catalog Layer**: Offers pre-built guardrails for common use cases
4. **CLI Layer**: Exposes functionality through a command-line interface
5. **Evaluation Layer**: Provides tools for evaluating guardrail effectiveness

Here's a high-level diagram of how these components interact:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Code                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PsySafe Library                          │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│  │  Catalog Layer  │──▶│   Core Layer    │◀──│  Drivers Layer  │ │
│  └─────────────────┘   └────────┬────────┘   └─────────────────┘ │
│           ▲                     │                     ▲           │
│           │                     ▼                     │           │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│  │    CLI Layer    │   │ Evaluation Layer│   │  External LLMs   │ │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Layer

The Core Layer defines the fundamental abstractions and interfaces for the library.

### Key Components

#### GuardrailBase

The `GuardrailBase` abstract class defines the interface for all guardrails:

```python
class GuardrailBase(Generic[RequestT, ResponseT], ABC):
    @abstractmethod
    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """Apply the guardrail to a request."""
        pass

    @abstractmethod
    def validate(self, response: ResponseT) -> ValidationReport:
        """Validate a response against the guardrail."""
        pass

    def bind(self, driver: Any) -> Any:
        """Bind the guardrail to a specific LLM driver instance."""
        return driver
```

This interface ensures that all guardrails can:
- Modify requests before they're sent to the LLM (`apply`)
- Validate responses from the LLM (`validate`)
- Optionally interact with the driver (`bind`)

#### GuardedRequest

The `GuardedRequest` model represents a request that has been processed by a guardrail:

```python
class GuardedRequest(BaseModel, Generic[RequestT]):
    original_request: RequestT
    modified_request: RequestT
    metadata: Dict[str, Any] = {}
```

#### ValidationReport

The `ValidationReport` model represents the result of validating a response:

```python
class ValidationReport(BaseModel):
    is_valid: bool
    violations: List[Violation] = []
    metadata: Dict[str, Any] = {}
```

#### PromptTemplate

The `PromptTemplate` class manages prompt templates using Jinja2:

```python
class PromptTemplate:
    @classmethod
    def from_string(cls, prompt_text: str) -> "PromptTemplate":
        """Creates a PromptTemplate instance from a raw string."""
        pass

    @classmethod
    def from_file(cls, template_file_path: Union[str, Path]) -> "PromptTemplate":
        """Creates a PromptTemplate instance by loading content from a file."""
        pass

    def render(self, context: PromptRenderCtx) -> str:
        """Renders the prompt template with the given context."""
        pass
```

### Guardrail Implementations

The Core Layer provides several implementations of the `GuardrailBase` interface:

#### PromptGuardrail

The `PromptGuardrail` class modifies requests by adding or transforming prompt instructions:

```python
class PromptGuardrail(GuardrailBase[RequestT, ResponseT]):
    def __init__(self, template: PromptTemplate, template_variables: Dict[str, Any] = None):
        """Initializes a PromptGuardrail."""
        pass

    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """Applies the guardrail by rendering the prompt template and modifying the request."""
        pass

    def validate(self, response: ResponseT) -> ValidationReport:
        """PromptGuardrails typically do not validate responses themselves."""
        pass
```

#### CheckGuardrail

The `CheckGuardrail` class validates responses using validator functions:

```python
class CheckGuardrail(GuardrailBase[RequestT, ResponseT]):
    def __init__(self, validators: List[Validator]):
        """Initializes a CheckGuardrail."""
        pass

    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """CheckGuardrails do not modify the request."""
        pass

    def validate(self, response: ResponseT) -> ValidationReport:
        """Validates the response by applying all registered validator functions."""
        pass
```

#### CompositeGuardrail

The `CompositeGuardrail` class composes multiple guardrails:

```python
class CompositeGuardrail(GuardrailBase[RequestT, ResponseT]):
    def __init__(self, guardrails: List[GuardrailBase[RequestT, ResponseT]]):
        """Initializes a CompositeGuardrail."""
        pass

    def apply(self, request: RequestT) -> GuardedRequest[RequestT]:
        """Applies all composed guardrails sequentially to the request."""
        pass

    def validate(self, response: ResponseT) -> ValidationReport:
        """Validates the response by applying all composed guardrails sequentially."""
        pass
```

## Drivers Layer

The Drivers Layer provides integrations with different LLM providers.

### ChatDriverABC

The `ChatDriverABC` abstract class defines the interface for all LLM drivers:

```python
class ChatDriverABC(Generic[RequestT, ResponseT], ABC):
    @abstractmethod
    def send(self, request: RequestT) -> ResponseT:
        """Send a request to the LLM and get a single response."""
        pass

    @abstractmethod
    async def stream(self, request: RequestT) -> AsyncIterator[ResponseT]:
        """Send a request to the LLM and stream responses back."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the driver."""
        pass
```

### Driver Implementations

The library provides implementations for various LLM providers:

- `OpenAIChatDriver`: For OpenAI models (GPT-3.5, GPT-4, etc.)
- `AnthropicChatDriver`: For Anthropic models (Claude, etc.)
- `TransformersChatDriver`: For local Hugging Face models

## Catalog Layer

The Catalog Layer provides a registry of pre-built guardrails.

### GuardrailCatalog

The `GuardrailCatalog` class manages the registry of guardrails:

```python
class GuardrailCatalog:
    @classmethod
    def register(cls, name: str, guardrail_cls: Type[GuardrailBase]) -> None:
        """Registers a guardrail class with a given name."""
        pass

    @classmethod
    def load(cls, names: Union[str, List[str]], **kwargs: Any) -> List[GuardrailBase]:
        """Loads one or more guardrails by name."""
        pass

    @classmethod
    def compose(cls, names: Union[str, List[str]], **kwargs: Any) -> CompositeGuardrail:
        """Loads multiple guardrails by name and returns them wrapped in a CompositeGuardrail."""
        pass

    @classmethod
    def list_available(cls) -> List[str]:
        """Returns a list of names of all registered guardrails."""
        pass
```

## Design Patterns

PsySafe AI employs several design patterns to ensure flexibility and extensibility:

### Strategy Pattern

The `GuardrailBase` interface and its implementations follow the Strategy pattern, allowing different guardrail strategies to be used interchangeably.

### Composite Pattern

The `CompositeGuardrail` class implements the Composite pattern, allowing guardrails to be composed into more complex guardrails.

### Factory Method Pattern

The `GuardrailCatalog.load` and `GuardrailCatalog.compose` methods implement the Factory Method pattern, creating guardrail instances based on their names.

### Template Method Pattern

The `PromptTemplate` class implements the Template Method pattern, providing a framework for rendering prompts with different contexts.

## Extending the Library

### Creating Custom Guardrails

To create a custom guardrail, extend one of the base guardrail classes:

```python
from psysafe.core.prompt import PromptGuardrail
from psysafe.core.template import PromptTemplate

class MyCustomGuardrail(PromptGuardrail):
    def __init__(self, custom_param):
        template = PromptTemplate.from_string(
            "This is a custom guardrail with parameter: {{ custom_param }}"
        )
        super().__init__(template=template, template_variables={"custom_param": custom_param})
        self.custom_param = custom_param

    # Override methods as needed
    # ...
```

### Creating Custom Drivers

To create a custom driver for a new LLM provider, implement the `ChatDriverABC` interface:

```python
from psysafe.drivers.base import ChatDriverABC

class MyCustomDriver(ChatDriverABC):
    def __init__(self, model, api_key=None, **kwargs):
        # Initialize your driver
        pass

    def send(self, request):
        # Implement sending a request to your LLM
        pass

    async def stream(self, request):
        # Implement streaming responses from your LLM
        pass

    def get_metadata(self):
        # Return metadata about your driver
        pass
```

### Creating Custom Validators

To create custom validators for use with `CheckGuardrail`:

```python
from psysafe.core.models import ValidationReport, Violation, ValidationSeverity

def my_custom_validator(response):
    # Implement your validation logic
    if problem_detected(response):
        return ValidationReport(
            is_valid=False,
            violations=[
                Violation(
                    severity=ValidationSeverity.ERROR,
                    code="CUSTOM_VIOLATION",
                    message="A problem was detected in the response",
                    context={"details": "Additional details about the violation"}
                )
            ]
        )
    return ValidationReport(is_valid=True)

# Use with CheckGuardrail
from psysafe.core.check import CheckGuardrail
check_guardrail = CheckGuardrail(validators=[my_custom_validator])
```

## Best Practices

When working with the PsySafe AI architecture:

1. **Separation of Concerns**: Keep guardrails focused on specific safety concerns
2. **Composition over Inheritance**: Use `CompositeGuardrail` to combine functionality
3. **Immutability**: Treat requests and responses as immutable, creating new instances when modifications are needed
4. **Error Handling**: Implement robust error handling in custom guardrails and drivers
5. **Testing**: Write tests for custom guardrails using the evaluation tools

## Future Architecture Directions

The PsySafe AI architecture is designed to evolve with emerging needs:

- **Plugin System**: A more formalized plugin system for extending functionality
- **Middleware Architecture**: Additional middleware layers for cross-cutting concerns
- **Distributed Guardrails**: Support for distributed guardrail execution
- **Feedback Loops**: Mechanisms for learning from guardrail performance in production

For more information on contributing to the architecture, see the project's contribution guidelines.