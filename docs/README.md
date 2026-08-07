# PsySafe documentation

PsySafe provides calibrated, categorical safety checks for the boundaries where AI systems act.

Start with the path that matches your work:

- [Getting started](getting_started.md): install a provider, classify a conversation, recalibrate locally, and handle failures.
- [Classifier reference](classifiers.md): scope, signal vocabularies, evidence roles, and result details.
- [Workflow gates](workflows.md): sequential checkpoints plus OpenAI Agents and Claude Agent SDK integration.
- [Evaluation](evaluation.md): strict golden cases, family-safe splits, named-boundary testing, and aggregate metrics.
- [Architecture](architecture.md): contracts, trust boundaries, data flow, and extension points.
- [Design principles](design_principles.md): why the API uses concise structured prompts, named boundaries, abstention, and boundary-specific evaluation.
- [Migration to 0.2](migration.md): breaking changes from the pre-0.2 guardrail API.

Before deployment, read the repository [safety guidance](../SAFETY.md) and [security policy](../SECURITY.md). Runnable examples are in [`examples/`](../examples/).

## Product contract

The documentation uses these terms consistently:

- A **classifier** screens observable policy evidence. It does not decide what an application should do.
- **Sensitivity** is a named evidence boundary: `precise`, `balanced`, or `precautionary`. It is not severity or urgency.
- An **assessment** is `matched`, `not_matched`, or `indeterminate`.
- A **gate** maps assessments to `allow`, `review`, or `block` at one workflow checkpoint.
- An **evaluation metric** summarizes a dataset. It is not confidence in an individual assessment.

PsySafe is an alpha component, not a complete safety system, clinical instrument, emergency service, redaction system, or compliance certification.
