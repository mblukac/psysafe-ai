# PsySafe AI

Calibrated, categorical safety checks for the boundaries where AI systems act.

PsySafe is an alpha Python library for screening observable policy signals in conversations and routing the result at workflow boundaries. It is designed for two common execution shapes:

- sequential workflows such as input → task selection → execution → communication; and
- agent runtimes, with native adapters for the OpenAI Agents SDK and Claude Agent SDK.

Every classifier assessment has one of three outcomes: `matched`, `not_matched`, or `indeterminate`. A workflow gate maps that outcome to an explicit action such as `allow`, `review`, or `block`. PsySafe does not diagnose people, determine urgency, certify safety, or choose an intervention.

## Why categorical calibration

An LLM-backed classifier makes one structured, sensitivity-independent observation. PsySafe then applies a named evidence boundary locally:

- `precise`: explicit evidence only;
- `balanced`: explicit or contextually clear evidence; and
- `precautionary`: explicit, contextual, or plausible-but-ambiguous evidence.

Broader settings include every match from narrower settings. Recalibration does not make another provider request. PsySafe deliberately omits per-item confidence, severity, clinical risk, and numerical ratings; those values should not quietly drive consequential routing without task-specific validation.

## Install

Core contracts, gates, evaluation, and local PII detection:

```bash
pip install psysafe-ai
```

Add the provider used by an LLM-backed classifier:

```bash
pip install 'psysafe-ai[openai]'
pip install 'psysafe-ai[anthropic]'
```

Add an agent runtime adapter independently:

```bash
pip install 'psysafe-ai[openai-agents]'
pip install 'psysafe-ai[claude-agent]'
pip install 'psysafe-ai[agents]'  # both agent SDKs
pip install 'psysafe-ai[all]'     # both providers and both agent SDKs
```

Provider models are always explicit; PsySafe never silently selects a model.

## Classify once, calibrate locally

```python
from psysafe import Conversation, Sensitivity, VulnerabilitySignalsClassifier
from psysafe.backends import OpenAIBackend

classifier = VulnerabilitySignalsClassifier(
    OpenAIBackend(model="your-structured-output-model")
)
conversation = Conversation.from_text(
    "I use a screen reader and need another way to complete this form."
)

record = classifier.observe(conversation)  # one provider request
precise = classifier.calibrate(record, sensitivity=Sensitivity.PRECISE)
precautionary = classifier.recalibrate(
    record,
    sensitivity=Sensitivity.PRECAUTIONARY,
)

print(precise.outcome)
print(precautionary.outcome)
```

Use `classify()` or `aclassify()` when you need one setting only. Their default failure policy returns `indeterminate`; `observe()` and `aobserve()` instead raise a sanitized backend error because no observation exists to calibrate.

## Put checks at action boundaries

```python
from psysafe import (
    Checkpoint,
    GateAction,
    GatePolicy,
    SelfHarmClassifier,
    WorkflowGate,
)
from psysafe.backends import OpenAIBackend

classifier = SelfHarmClassifier(
    OpenAIBackend(model="your-structured-output-model")
)
input_gate = WorkflowGate(
    (classifier,),
    checkpoint=Checkpoint.INPUT,
    policy=GatePolicy(matched_action=GateAction.REVIEW),
)

decision = input_gate.evaluate_text(
    "conversation text",
    artifact_id="request-opaque-version-1",
)

if decision.action is GateAction.ALLOW:
    run_workflow()
elif decision.action is GateAction.REVIEW:
    queue_review()
else:
    stop_workflow()
```

`indeterminate` and independent review signals can never be configured to allow a gate. Use immutable artifact IDs so a decision is consumed only for the exact input version it assessed.

## Included classifiers

| Classifier | What it screens | Important boundary |
| --- | --- | --- |
| `SelfHarmClassifier` | Observable self-harm and suicide signal categories | Routing aid, not clinical risk or urgency assessment |
| `AssistantHarmClassifier` | Assistant encouragement, endorsement, or actionable facilitation of serious self-destructive behavior | Checks assistant messages, not overall response quality |
| `VulnerabilitySignalsClassifier` | Disclosed or observable support-needs drivers | Screens interaction needs; does not label a person |
| `DistressSupportClassifier` | Non-diagnostic distress signals and communication adaptations | Supports response style; does not infer a condition |
| `ComplaintsClassifier` | Complaint categories and independent escalation reasons | Escalation remains an application policy |
| `PIIClassifier` | Local locations of common identifier formats | Locates values; it does not redact them or guarantee exhaustive detection |

Each LLM-backed classifier can export its fixed policy, allowed categorical vocabulary, input contract, named boundaries, and JSON Schema with `export_spec()`.

## Choose an integration path

- [Getting started](https://github.com/mblukac/psysafe-ai/blob/main/docs/getting_started.md) covers providers, conversations, failure handling, and local recalibration.
- [Workflow gates](https://github.com/mblukac/psysafe-ai/blob/main/docs/workflows.md) covers all six sequential checkpoints and both agent SDK adapters.
- [Classifier reference](https://github.com/mblukac/psysafe-ai/blob/main/docs/classifiers.md) documents signal vocabularies and scope.
- [Evaluation](https://github.com/mblukac/psysafe-ai/blob/main/docs/evaluation.md) covers tuning/holdout families, monotonicity, slices, and aggregate metrics.
- [Migration to 0.2](https://github.com/mblukac/psysafe-ai/blob/main/docs/migration.md) maps the removed pre-0.2 API to the categorical design.
- [Architecture](https://github.com/mblukac/psysafe-ai/blob/main/docs/architecture.md) and [design rationale](https://github.com/mblukac/psysafe-ai/blob/main/docs/design_principles.md) explain the trust boundaries and research choices.

Runnable examples live in [`examples/`](https://github.com/mblukac/psysafe-ai/tree/main/examples).

## Safety and privacy

PsySafe is one fallible control in a larger sociotechnical system. Validate each classifier, provider model, sensitivity, language, and routing action on representative data. Keep qualified people responsible for consequential decisions and define an operational fallback for unavailable checks.

LLM-backed classifiers send message content to the configured provider. Caller-supplied `Message.id` values are replaced with positional evidence IDs, but identifiers inside message content are not removed. Minimize input, configure provider retention and tracing deliberately, and do not log raw conversations or model responses by default. The local `PIIClassifier` can help locate some common formats before a provider call, but it is not a complete de-identification system.

Read [SAFETY.md](https://github.com/mblukac/psysafe-ai/blob/main/SAFETY.md) before deployment and report software vulnerabilities through [SECURITY.md](https://github.com/mblukac/psysafe-ai/blob/main/SECURITY.md). PsySafe is available under the [MIT License](https://github.com/mblukac/psysafe-ai/blob/main/LICENSE).
