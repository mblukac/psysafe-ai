# Getting started

## Requirements

PsySafe supports Python 3.10–3.14. The core package includes contracts, local PII detection, workflow gates, and evaluation. LLM-backed classifiers require a provider extra.

```bash
pip install 'psysafe-ai[openai]'
# or
pip install 'psysafe-ai[anthropic]'
```

Configure the provider SDK through its normal environment or client options. Model names are explicit because availability, behavior, price, and structured-output support change over time.

## Create a classifier

```python
from psysafe import Conversation, DistressSupportClassifier, Sensitivity
from psysafe.backends import OpenAIBackend

backend = OpenAIBackend(
    model="your-structured-output-model",
    timeout=20,
)
classifier = DistressSupportClassifier(backend)
conversation = Conversation.from_text(
    "Everything is piling up and I cannot focus on all these steps."
)

assessment = classifier.classify(
    conversation,
    sensitivity=Sensitivity.BALANCED,
)

print(assessment.outcome.value)
print(assessment.signals)
```

Use `AnthropicBackend(model="...")` for Anthropic's structured Messages API. A classifier provider is independent of the application model or agent runtime: an OpenAI agent can use an Anthropic-backed classifier, and vice versa.

## Understand the result

All assessments share these fields:

- `classifier_id` and `policy_version`: the policy identity;
- `sensitivity`: the local boundary applied;
- `outcome`: `matched`, `not_matched`, or `indeterminate`;
- `evidence_directness`: `explicit`, `contextual`, `ambiguous`, or `none`;
- `signals`: categorical labels from the classifier's fixed vocabulary;
- `indeterminate_reason`: a sanitized category when no decision could be made; and
- `metadata`: provider/model labels, not authenticated execution provenance.

Typed classifiers add categorical details such as self-harm context, support adaptations, complaint escalation reasons, or PII locations. No assessment contains per-item confidence, severity, a clinical risk rating, arbitrary reasoning text, or a raw provider response.

Do not reduce an assessment to a boolean unless you have handled abstention:

```python
from psysafe import Outcome

if assessment.outcome is Outcome.MATCHED:
    route_match(assessment)
elif assessment.outcome is Outcome.NOT_MATCHED:
    continue_normally()
else:
    route_unavailable_check(assessment.indeterminate_reason)
```

`assessment.require_match_decision()` raises `IndeterminateAssessmentError` instead of silently treating `indeterminate` as false.

## Observe once, try multiple boundaries

`classify()` is convenient for one boundary. Use `observe()` when policy owners need to compare settings without repeated provider requests:

```python
record = classifier.observe(conversation)

for sensitivity in Sensitivity:
    result = classifier.calibrate(record, sensitivity=sensitivity)
    print(sensitivity.value, result.outcome.value, result.signals)
```

The provider never sees the selected sensitivity. It returns categorical findings with directness labels; calibration filters those findings locally. The boundary is monotonic:

| Sensitivity | Included evidence |
| --- | --- |
| `precise` | explicit |
| `balanced` | explicit, contextual |
| `precautionary` | explicit, contextual, ambiguous |

Choose settings from evaluation evidence and the cost of each downstream action. A broader boundary is not automatically “safer”; it changes both missed matches and unnecessary interventions.

## Classify in context

Construct an immutable `Conversation` when earlier turns matter:

```python
from psysafe import Conversation, Message, MessageRole

conversation = Conversation(
    messages=(
        Message(role=MessageRole.USER, content="I cannot complete this form."),
        Message(role=MessageRole.ASSISTANT, content="Which part is difficult?"),
        Message(
            role=MessageRole.USER,
            content="The instructions are hard to process; can I have more time?",
        ),
    )
)

assessment = classifier.classify_target(
    conversation,
    target_message_index=2,
    sensitivity=Sensitivity.BALANCED,
)
```

Targeted classification can use the full conversation for context while requiring actionable evidence to cite the selected message. Classifiers also enforce the expected evidence role: user-signal classifiers cannot be attached to an assistant-only checkpoint.

## Handle backend failures deliberately

LLM-backed classifiers default to `FailurePolicy.RETURN_INDETERMINATE` for `classify()` and `aclassify()`. No failure becomes `not_matched`.

```python
from psysafe import FailurePolicy, SelfHarmClassifier

strict_classifier = SelfHarmClassifier(
    backend,
    failure_policy=FailurePolicy.RAISE,
)
```

`observe()` and `aobserve()` always raise sanitized backend exceptions when no valid observation exists. Use them where the caller can retry or stop; use `classify()` where a categorical abstention fits the workflow contract. Missing provider extras raise a configuration error, while missing agent-runtime extras fail when the concrete adapter module is imported; neither is disguised as a policy result.

## Protect data

LLM-backed classifiers send `Message.content` to the configured provider. Caller `Message.id` values are replaced by generated positions such as `m0`, but identifiers written in content remain. OpenAI requests set `store=False`; that does not replace reviewing all provider, account, region, retention, and tracing settings.

Minimize context, avoid raw-content logs, and treat stored observation records as sensitive. If a record crosses a storage or queue boundary, call `classifier.validate_record(record, conversation)` before relying on its positional citations.

Continue with [classifier scope](classifiers.md), then choose [workflow integration](workflows.md) or [evaluation](evaluation.md).
