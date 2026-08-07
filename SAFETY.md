# Safety

PsySafe provides configurable classifiers for signals that may matter to an
AI application's psychological-safety policy. A classification is a fallible
software signal, not a diagnosis, risk assessment, or determination that a
person or interaction is safe.

## Appropriate use

- Define the policy decision each classifier informs and validate it with
  representative data from the intended language, population, and context.
- Measure false negatives and false positives at every supported sensitivity
  setting before deployment and after material model, prompt, or data changes.
- Route uncertain, unavailable, or malformed classifications to a deliberate
  fallback. For consequential workflows, that normally means review or a
  conservative response rather than silently continuing.
- Keep qualified people responsible for consequential decisions. Give them
  enough context to disagree with a classification and record overrides.
- Monitor for distribution shift, prompt injection, subgroup performance
  differences, and recurring failure modes.

## Boundaries

Do not use PsySafe as the sole control for emergencies, clinical care,
self-harm intervention, eligibility, employment, insurance, credit, law
enforcement, or another decision that can materially affect a person's rights,
health, or access to services. It is not medical, psychological, legal, or
compliance advice and has not been certified as a medical device or emergency
service.

If someone may be in immediate danger, follow the emergency process for your
service and location. Do not wait for this package or an AI model to decide
whether to seek qualified help.

## Privacy

Inputs can contain health information, personal data, or intimate
conversations. Send only the minimum content needed for classification, choose
providers and retention settings appropriate to your obligations, restrict
access to logs and traces, and avoid storing raw inputs or model responses by
default. Never place secrets in classifier inputs.

## License and responsibility

The [MIT License](LICENSE) governs the software. This safety guidance does not
add use restrictions to that license. Deployers remain responsible for their
system design, notices, consent, testing, human oversight, and compliance with
applicable law and professional standards.
