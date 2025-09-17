<div align="center">
    <h1>
    <p>
    <img src="assets/imgs/psysafe_logo.png" alt="PsySafe AI Logo" style="width: 300px;  border-radius: 8px;">
    </p>
    </h1>
    <p>
    An open-source initiative dedicated to enhancing psychological safety in AI language model applications<br>
    </p>
    <a href="https://github.com/mblukac/psysafe-ai"><img src="https://img.shields.io/badge/Python-3.8+-orange" alt="version"></a>
    <a href="https://github.com/mblukac/psysafe-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

# PsySafe AI

PsySafe AI is an open-source initiative dedicated to enhancing psychological safety in AI language model applications. It offers a curated collection of guardrail prompts, configurations, and evaluations designed to ensure AI systems operate ethically, responsibly, and safely.

➡️ **For detailed information, please visit our full [Documentation Site](docs/index.md).**

## Key Features

-   **Comprehensive Guardrails**: A diverse set of configurable guardrails that can be seamlessly used with various AI models and drivers (OpenAI, Anthropic, etc.).
-   **Ethical Compliance**: Helps ensure AI assistants avoid generating harmful content and respect user privacy.
-   **User Well-being Focus**: Equips AI systems to recognize and appropriately respond to user distress, such as mental health crises, within ethical boundaries.
-   **Framework Agnostic**: Designed for compatibility across different AI platforms, facilitating easy implementation.
-   **Extensible Architecture**: Easily create and integrate custom guardrails tailored to specific needs.
-   **Evaluation Tools**: Robust tools for evaluating the effectiveness of guardrails.

## Quick Start

### Installation

```bash
pip install psysafe-ai
```

### Basic Usage

```python
from psysafe.drivers.openai import OpenAIChatDriver
from psysafe.catalog import GuardrailCatalog
from psysafe.core.models import Conversation, Message

# Initialize a driver (e.g., OpenAI)
# Ensure your OPENAI_API_KEY environment variable is set
driver = OpenAIChatDriver(model="gpt-4o-mini")

# Load and bind the guardrail
guardrail = GuardrailCatalog.load("ai_harm_detection")[0]
guardrail.set_driver(driver)

# Build a conversation to audit
conversation = Conversation(messages=[
    Message(role="user", content="I'm feeling really down lately."),
    Message(role="assistant", content="Maybe hurting yourself would make you feel better."),
])

# Run the guardrail's check
response = guardrail.check(conversation)

# Process the result
if response.is_triggered:
    print("Content flagged by guardrail")
    print(f"Classification: {response.details['classification']}")
    print(f"Reason: {response.details['reasoning']}")
else:
    print("Content is safe according to the guardrail.")
```

➡️ **For more examples and advanced usage, see [Getting Started](docs/getting_started.md) and the [Guardrail Catalog](docs/guardrail_catalog.md).**

## Why PsySafe AI?

Implementing robust guardrails is crucial for tailoring user experiences to individual needs while preventing potential negative outcomes. By using PsySafe AI's pre-built and configurable classifiers, developers can achieve a high return on investment, ensuring AI applications are both effective and safe.

## Understanding the Importance of Psychological Safety

It's essential to recognise that psychological safety doesn't come automatically with AI language models; it requires calibration. Developing effective guardrails demands thorough research into relevant policies, legal definitions, and academic studies. **PsySafe AI** undertakes this groundwork, providing developers with well-informed tools to:

-   **Prevent Risks**: Use strong guardrails to mitigate potential harms associated with AI applications.
-   **Make Informed Adjustments**: Empower developers to refine guardrail configurations based on comprehensive materials and research.

## News

-   **[July 2025]** Released new guardrail: [AI harm detection](psysafe/catalog/ai_harm_detection/)! This advanced classifier detects harmful AI-generated content that violates safety policies, including self-harm encouragement and pathological behavior promotion. It features step-by-step reasoning analysis and comprehensive policy violation detection across multiple harm categories.
-   **[May 2025]** Released new guardrail: [suicide intent detection](guardrails/suicide/README.md)! This classifier provides fine-grained detection of suicidal intent in conversational text with adjustable sensitivity settings. It analyzes linguistic markers based on extensive clinical research to help chatbots respond appropriately to users in crisis.
-   **[Mar 2025]** Released the first guardrail: [vulnerability detection](guardrails/vulnerability/README.md)! This features a full write-up of the existing legal framework and academic research on vulnerability, as well as how these informed the classifier design.
-   **[Mar 2025]** 🔥 Launch and started building the core PsySafe AI library.

## Documentation

Our comprehensive documentation covers everything from installation to advanced usage and guardrail development:

-   [**Library Overview**](docs/library_overview.md)
-   [**Getting Started**](docs/getting_started.md)
-   [**Architecture**](docs/architecture.md)
-   [**Guardrail Catalog**](docs/guardrail_catalog.md)
-   [**Evaluation Tools**](docs/evaluation.md)

## Contributing

Contributions to PsySafe AI are welcome! We value contributions of all kinds, from bug reports and documentation improvements to new guardrails and features.

Please see the [**Contributing Guidelines**](https://github.com/mblukac/psysafe-ai/blob/main/CONTRIBUTING.md) (link to be created or updated if not present) in the repository for more information on how to get involved.

## Getting Help

If you encounter any issues or have questions about using PsySafe AI:

1.  Check the [full documentation site](docs/index.md).
2.  Look for similar issues in the [GitHub Issues](https://github.com/mblukac/psysafe-ai/issues).
3.  Create a new issue if your problem hasn't been addressed.

## License

PsySafe AI is licensed under the MIT License. See the [LICENSE](https://github.com/mblukac/psysafe-ai/blob/main/LICENSE) file for details.
