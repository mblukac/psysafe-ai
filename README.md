# PsySafe AI

Calibrated psychological-safety classifiers for AI workflows.

PsySafe is an alpha Python library for detecting documented policy signals in conversations. A classifier returns one of `matched`, `not_matched`, or `indeterminate`. Your application decides what to do with that assessment.

The library deliberately does not produce confidence, severity, or risk scores. Sensitivity is a named evidence boundary:

- `precise` includes explicit evidence only.
- `balanced` also includes evidence that becomes clear in context.
- `precautionary` also includes plausible but ambiguous evidence.

The model makes one sensitivity-independent structured observation. Calibration then runs locally, so changing sensitivity does not make another provider call.

## Install

Core and local PII detection:

```bash
pip install psysafe-ai
```

Add one provider integration:

```bash
pip install 'psysafe-ai[openai]'
pip install 'psysafe-ai[anthropic]'
```

## Quick start

Provider models are always explicit; PsySafe does not silently select a dated default.

```python
from psysafe import Conversation, Sensitivity, VulnerabilitySignalsClassifier
from psysafe.backends import OpenAIBackend

backend = OpenAIBackend(model="your-structured-output-model")
classifier = VulnerabilitySignalsClassifier(backend)
conversation = Conversation.from_text("I use a screen reader and need another way to complete this form.")

record = classifier.observe(conversation)  # one provider request
precise = classifier.calibrate(record, sensitivity=Sensitivity.PRECISE)
precautionary = classifier.recalibrate(record, sensitivity=Sensitivity.PRECAUTIONARY)

print(precise.outcome)
print(precautionary.outcome)
```

Provider instructions and conversation data are sent through separate API fields. Evidence references use generated positional IDs, so caller-supplied account or patient IDs never enter prompts or results. Provider failures return `indeterminate` by default and are never converted to `not_matched`.

## Classifiers

- `SelfHarmClassifier`: observable self-harm and suicide signals for routing, not diagnosis or clinical risk assessment.
- `AssistantHarmClassifier`: assistant encouragement, endorsement, or actionable facilitation of serious self-destructive behavior.
- `VulnerabilitySignalsClassifier`: disclosed or observable support needs without labeling a person.
- `DistressSupportClassifier`: non-diagnostic distress signals and communication adaptations.
- `ComplaintsClassifier`: complaint categories with independently calibrated escalation evidence.
- `PIIClassifier`: deterministic, local, value-free locations for common identifiers.

Each LLM classifier exports its fixed instructions, input contract, allowed signals, required evidence role, sensitivity boundaries, and JSON Schema with `export_spec()`.

## Safety boundary

PsySafe is a classification component, not a complete safety system. It does not diagnose, determine urgency, prescribe interventions, or replace qualified human judgment. Validate policies and thresholds on representative data, route `indeterminate` explicitly, minimize retained data, and add human review where mistakes could cause harm. See [SAFETY.md](SAFETY.md) and [SECURITY.md](SECURITY.md).

PsySafe is licensed under the [MIT License](LICENSE).
