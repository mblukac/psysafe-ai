# Migration to 0.2

Version 0.2 is a deliberate breaking redesign. The pre-0.2 API mixed prompt construction, provider execution, open-ended model judgments, and application action in objects called guardrails. The new API separates typed observation, deterministic calibration, categorical assessment, and explicit routing.

## Concept mapping

| Pre-0.2 concept | 0.2 replacement |
| --- | --- |
| `GuardrailCatalog.load(...)` | Construct an explicit classifier with an explicit backend |
| `OpenAIChatDriver` / `AnthropicChatDriver` | `OpenAIBackend` / `AnthropicBackend` |
| `TransformersChatDriver` | A project-owned `StructuredBackend` implementation or `CallableBackend` adapter |
| `guardrail.check(...)` / `apply(...)` / `validate(...)` | `classify()` or `observe()` + `calibrate()` |
| `CompositeGuardrail` | `WorkflowGate` / `AsyncWorkflowGate` at one checkpoint |
| `LOW`, `MEDIUM`, `HIGH` | `PRECISE`, `BALANCED`, `PRECAUTIONARY` |
| boolean validity | `Outcome.MATCHED`, `NOT_MATCHED`, or `INDETERMINATE` |
| confidence, severity, risk, or reasoning fields | categorical signals, directness, typed context, and explicit abstention |
| built-in PII “protection” | local value-free locations; application-owned transformation |
| package CLI | application code or a project-specific CLI around the typed API |

The former catalog names map as follows:

| Pre-0.2 catalog name | 0.2 classifier |
| --- | --- |
| `vulnerability_detection` | `VulnerabilitySignalsClassifier` |
| `suicide_prevention` | `SelfHarmClassifier` |
| `ai_harm_detection` | `AssistantHarmClassifier` |
| `mental_health_support` | `DistressSupportClassifier` |
| `complaints_handling` | `ComplaintsClassifier` |
| `pii_protection` | `PIIClassifier` |

Legacy lowercase sensitivity strings can still normalize in core classifier calls during migration (`low` → `precise`, `medium` → `balanced`, `high` → `precautionary`). New code and agent adapters accept only the canonical names. Serialized output is always canonical. Remove aliases once callers have migrated.

## Before

```python
from psysafe.catalog import GuardrailCatalog
from psysafe.core.models import Conversation, Message
from psysafe.drivers.openai import OpenAIChatDriver

driver = OpenAIChatDriver(model="gpt-4o-mini")
guardrail = GuardrailCatalog.load("ai_harm_detection")[0]
guardrail.set_driver(driver)
conversation = Conversation(
    messages=[
        Message(role="user", content=request),
        Message(role="assistant", content=draft_response),
    ]
)
response = guardrail.check(conversation)
```

## After

```python
from psysafe import AssistantHarmClassifier, Conversation, Message, MessageRole, Sensitivity
from psysafe.backends import OpenAIBackend

classifier = AssistantHarmClassifier(
    OpenAIBackend(model="your-structured-output-model")
)
assessment = classifier.classify(
    Conversation(
        messages=(
            Message(role=MessageRole.USER, content=request),
            Message(role=MessageRole.ASSISTANT, content=draft_response),
        )
    ),
    sensitivity=Sensitivity.PRECAUTIONARY,
)
```

The application now owns the action:

```python
from psysafe import Outcome

match assessment.outcome:
    case Outcome.MATCHED:
        adapt_or_review(assessment)
    case Outcome.NOT_MATCHED:
        continue_normally()
    case Outcome.INDETERMINATE:
        use_fallback(assessment.indeterminate_reason)
```

## Composition becomes checkpoint routing

Do not run an undifferentiated list of checks and interpret one boolean. Group classifiers by the role and action boundary they actually govern:

```python
gate = WorkflowGate(
    (classifier_a, classifier_b),
    checkpoint=Checkpoint.INPUT,
    policy=GatePolicy(matched_action=GateAction.REVIEW),
)
decision = gate.evaluate_text(
    request_text,
    artifact_id="request-version-1",
)
```

Create separate gates for task selection, execution, tool input, tool output, and final communication. This makes pre-action checks distinguishable from post-action observation.

## Failure behavior changed

There is no fail-clean option. Provider failures, refusals, invalid structured output, insufficient context, and saturated output become `indeterminate` under the default classifier policy. Gates route indeterminate assessments to a non-allow action. `observe()` raises because no reusable observation exists; use `classify()` when abstention is the desired API.

## Data model changed

- `Conversation` and `Message` are immutable, size-bounded Pydantic models.
- Message content is provider-bound for LLM classifiers; only caller message IDs are replaced.
- Findings cite opaque positions and use finite categorical vocabularies.
- Observation records from storage or queues must be revalidated against the original ordered conversation.
- Results contain no raw provider responses or arbitrary dictionaries.

## Suggested migration order

1. Replace provider drivers and construct classifiers explicitly.
2. Handle all three outcomes before removing old boolean logic.
3. Map legacy sensitivity names to canonical named boundaries and re-evaluate them; do not assume behavioral equivalence from the name alone.
4. Move application actions into `GatePolicy` at explicit checkpoints.
5. Add tuning/holdout golden families and choose boundaries from representative results.
6. Add native agent adapters only after the framework-neutral gates behave correctly.
7. Remove dependencies on old confidence, severity, reasoning, redaction, catalog, composite, and CLI behavior.

Pre-0.2 research and implementation notes were removed from current documentation because they described the retired interface and made claims outside the package's present scope. They remain available in Git history for archaeology, not deployment guidance.
