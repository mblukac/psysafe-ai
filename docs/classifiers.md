# Classifier reference

PsySafe classifiers answer narrow questions about observable text. They do not infer a person's diagnosis, stable traits, moral character, urgency, or overall “safety.” Each LLM policy is short, role-scoped, injection-aware, and constrained by a typed structured-output schema.

## Shared behavior

LLM-backed classifiers support:

- `observe()` / `aobserve()` for one sensitivity-independent provider call;
- `calibrate()` / `recalibrate()` for deterministic local boundary changes;
- `classify()` / `aclassify()` for a single check;
- `classify_target()` / `aclassify_target()` for one role-validated message in context; and
- `export_spec()` for the fixed policy, input contract, vocabulary, boundaries, and observation JSON Schema.

Only direct user evidence is actionable for support-needs, distress, and complaint classifiers. Quoted, fictional, and third-party material can provide context but does not label the user.

## `SelfHarmClassifier`

Screens user messages for observable categories:

- `ideation`
- `intent`
- `plan_or_access`
- `preparatory_behavior`
- `suicide_attempt`
- `self_injury_unclear_intent`
- `nonsuicidal_self_injury`

Matched findings retain categorical `subject`, `source_context`, and `timeframe` values. These fields describe language; they are not a clinical assessment, triage level, probability, or urgency decision. Historical, hypothetical, quoted, fictional, and third-party context stays explicit so applications can define appropriate routing.

Evidence role: `user`.

## `AssistantHarmClassifier`

Checks assistant messages for encouragement, endorsement, or actionable instructions in four domains:

- self-harm;
- eating-disorder behavior;
- dangerous substance use; and
- other serious self-destructive behavior.

Signals combine domain and behavior, for example `self_harm.actionable_instructions`. The classifier does not grade general helpfulness, empathy, policy compliance outside this taxonomy, or the user's state.

Evidence role: `assistant`.

## `VulnerabilitySignalsClassifier`

Screens direct user language for disclosed or observable support-needs drivers adapted from the FCA's interaction-oriented categories:

- `health`
- `life_events`
- `resilience`
- `capability`

Findings can suggest categorical communication adaptations such as `clear_language`, `extra_time`, `alternative_channel`, `check_understanding`, `human_support`, and `pause_and_resume`. An adaptation is a routing hint, not proof that it is wanted or appropriate; ask the user where possible.

The classifier screens the interaction for support needs. It must not be used to construct a persistent “vulnerable person” label or infer protected characteristics from writing style.

Evidence role: `user`.

## `DistressSupportClassifier`

Screens direct user language for non-diagnostic response-adaptation signals:

- `overwhelm`
- `panic`
- `grief`
- `loneliness`
- `low_mood`

Typed results can suggest communication adaptations such as calm, clear language or reduced cognitive load. They do not diagnose a condition or estimate its seriousness.

Evidence role: `user`.

## `ComplaintsClassifier`

Categorizes direct user complaints as:

- `service_quality`
- `product_or_outcome`
- `billing_or_payment`
- `access_or_communication`
- `staff_conduct`
- `privacy_or_data`
- `other`

It separately calibrates review signals:

- `explicit_human_request`
- `repeated_unresolved`
- `legal_or_regulatory_concern`
- `safety_or_support_need`

A complaint can be absent while an independent review signal is present. Workflow gates therefore route review signals separately from the main match outcome, and neither review signals nor `indeterminate` can allow a gate.

Evidence role: `user`.

## `PIIClassifier`

Runs locally without an LLM and locates several common identifier formats:

- email addresses;
- payment cards that pass Luhn validation;
- US Social Security numbers;
- phone numbers; and
- IPv4/IPv6 addresses.

Results contain only type, message index, and Python Unicode code-point offsets. They never copy the matched value. Sensitivity changes which ambiguous formats qualify, and compatibility-normalized matching maps locations back to the original text.

This classifier is intentionally not a redactor or de-identification guarantee. Applications must apply and verify their own transformation. When `locations_truncated` is true, the returned offsets are not a complete inventory.

Evidence role: any.

## LLM-backed provider-independent contracts

LLM-backed classifiers expose `ClassifierSpec`, which makes a policy portable across runtimes without exporting execution data:

```python
spec = classifier.export_spec()

print(spec.classifier_id)
print(spec.policy_version)
print(spec.allowed_signals)
print(spec.sensitivity_boundaries)
print(spec.observation_schema)
```

Pin the policy version, provider model, and evaluation dataset together in deployment records. Provider/model metadata in an assessment is descriptive and must not be treated as cryptographic provenance.
