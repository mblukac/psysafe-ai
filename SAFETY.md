# Safety

PsySafe provides calibrated, categorical checks for observable policy signals at AI workflow boundaries. A result is a fallible software signal, not a diagnosis, clinical risk assessment, urgency determination, intervention recommendation, or statement that a person or interaction is safe.

## Deployment responsibilities

- Define the exact boundary and action each classifier informs. A match has no universal response.
- Validate every classifier, provider model, policy version, language, sensitivity, and action on representative data before deployment and after material changes.
- Measure false negatives, false positives, indeterminate outcomes, intervention rate, and relevant slice performance. Keep related variants in one dataset family so they cannot leak across tuning and holdout splits.
- Route unavailable, refused, malformed, or saturated classifications deliberately. PsySafe gates never allow `indeterminate`; consequential workflows should normally use review or a conservative response.
- Keep qualified people responsible for consequential decisions. Give reviewers enough context to disagree and record overrides without treating model output as ground truth.
- Monitor for distribution shift, prompt injection, subgroup differences, new languages, recurring failure families, and provider behavior changes.

## High-stakes boundaries

Do not use PsySafe as the sole control for emergencies, clinical care, self-harm intervention, eligibility, employment, insurance, credit, law enforcement, or another decision that can materially affect a person's rights, health, or access to services. It is not medical, psychological, legal, or compliance advice and has not been certified as a medical device, emergency service, or assurance standard.

If someone may be in immediate danger, follow the emergency process for your service and location. Do not wait for this package or an AI model to decide whether to seek qualified help.

## Privacy and data flow

LLM-backed classifiers send conversation content to the configured backend. Policy instructions and conversation data use separate provider fields, and caller-supplied message IDs are replaced with opaque positional IDs, but personal information written inside `Message.content` remains in the provider request.

- Minimize the conversation to the context required by the policy.
- Select provider retention, region, encryption, and contractual settings appropriate to your obligations.
- Disable or restrict sensitive tracing and logging. Review agent SDK defaults as well as PsySafe configuration.
- Never put API keys, access tokens, or other secrets in classifier input.
- Treat observation records and evidence locations as sensitive metadata even though they omit raw provider responses.
- Revalidate an observation record against its original ordered conversation after storage, a queue, or another trust boundary.

`PIIClassifier` runs locally and reports value-free offsets for several common formats. It does not redact content, detect every identifier, or prove that text is anonymous. If `locations_truncated` is true, the returned offsets are explicitly incomplete.

## Orchestration boundaries

Place preventive checks before irreversible actions. A tool-output check runs after the tool and cannot undo an external side effect. Agent output guardrails are not streaming content filters; buffer output until the guarded run completes if users must not see rejected content.

Claude Agent SDK hooks do not cover final assistant output. Buffer the run and pass a successful, completed `ResultMessage.result` through an explicit communication gate before display.

Framework adapters cover only the lifecycle events exposed by their SDKs. In particular, OpenAI tool guardrails apply to custom function tools rather than handoffs, hosted tools, MCP tools, or `Agent.as_tool()`. Model input/output and tool payloads may also enter framework traces unless you configure tracing separately.

## License and responsibility

The [MIT License](LICENSE) governs the software. This guidance does not add license restrictions. Deployers remain responsible for system design, notices, consent, testing, human oversight, incident handling, and compliance with applicable law and professional standards.
