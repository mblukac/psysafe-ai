# Evaluation

PsySafe evaluates categorical behavior across all three named sensitivity boundaries. Aggregate rates describe a dataset; they are not confidence values for individual assessments.

## Golden-case format

Store one strict JSON object per line. Unknown fields, duplicate IDs, oversized inputs, malformed categories, and family leakage are rejected.

```json
{"case_id":"support-explicit-1","family_id":"support-family-1","split":"tuning","slices":["synthetic","en"],"conversation":{"messages":[{"role":"user","content":"I use a screen reader and need an accessible alternative."}]},"expected_boundary":"precise","expected_signals":[{"name":"health","boundary":"precise"}]}
```

Fields:

- `case_id`: an opaque unique case identifier;
- `family_id`: groups paraphrases, translations, adversarial variants, and descendants of one source;
- `split`: `tuning` or `holdout`;
- `slices`: non-sensitive categorical audit labels;
- `conversation`: the classifier input;
- `expected_boundary`: the narrowest expected match (`precise`, `balanced`, `precautionary`), `never`, or `indeterminate`;
- `expected_signals`: optional exact signal expectations, each with its own minimum boundary; and
- `expected_review_boundary` / `expected_review_signals`: optional independent review expectations, used by classifiers such as complaints.

Keep every related variant in the same family and split. `load_golden_cases()` and the runner reject a family that crosses tuning and holdout.

## Run a tuning evaluation

```python
from psysafe import VulnerabilitySignalsClassifier
from psysafe.backends import OpenAIBackend
from psysafe.evaluation import EvaluationSplit, load_golden_cases, run_evaluation

classifier = VulnerabilitySignalsClassifier(
    OpenAIBackend(model="your-structured-output-model")
)
cases = load_golden_cases("golden_cases.jsonl")

report = run_evaluation(
    classifier,
    cases,
    split=EvaluationSplit.TUNING,
)

for boundary in report.by_sensitivity:
    print(boundary.sensitivity.value, boundary.metrics.model_dump())
```

The runner observes each LLM case once and recalibrates it at all three boundaries. It records categorical comparison outcomes, exact signal matches when supplied, and whether broader boundaries preserve narrower matches.

## Interpret the metrics

Reports include confusion counts plus:

- `coverage`: decided binary rows divided by expected binary rows;
- `precision` and `recall` over decided binary rows;
- `effective_recall`, which counts positive abstentions as misses;
- false-positive and false-negative rates;
- `indeterminate_rate`;
- `intervention_rate`, meaning matched or indeterminate;
- exact `signal_match_rate` when signal expectations exist; and
- monotonicity verification/violation counts.

Always inspect counts and denominators. A rate is `None` when its denominator is zero. Choose a named sensitivity from the actual costs of missed matches, unnecessary reviews/blocks, and abstentions at the intended checkpoint. Do not pick a setting from one headline score.

## Use slices without leaking content

Slices should be bounded categorical labels such as language, channel, synthetic/consented source, or an application-relevant scenario family. Compare both errors and intervention burden across slices. Do not put raw text, account IDs, diagnoses, or protected attributes in a slice label.

Evaluation reports omit conversation content and exception text. Case diagnostics contain opaque IDs and categorical results only.

## Keep the holdout sealed

```python
holdout = run_evaluation(
    classifier,
    cases,
    split=EvaluationSplit.HOLDOUT,
)

assert holdout.diagnostics_revealed is False
assert holdout.case_results == ()
```

Holdout reports expose aggregate metrics by default and seal per-case and slice diagnostics. `reveal_holdout_diagnostics=True` is an explicit escape hatch that emits a warning. Once results have influenced prompts, labels, boundaries, routing, model choice, or case selection, that data is tuning data; create a new family-separated holdout.

## Recommended release protocol

1. Define the policy question and downstream action before labeling.
2. Build families spanning clear positives, clear negatives, ambiguous context, quoted/fictional/third-party material, prompt injection, provider refusal, malformed output, language/channel variants, and anticipated deployment slices.
3. Tune prompts, schemas, model choice, and sensitivity on tuning families only.
4. Record classifier policy version, provider model, dataset revision, routing policy, and code commit.
5. Run the sealed holdout once for the release decision.
6. Re-evaluate after any material component changes and monitor production drift with privacy-preserving labels and sampled human review.

For consequential use, supplement automated metrics with domain-expert review, user-impact analysis, abuse testing, operational drills, and a documented fallback. See [design principles](design_principles.md) for the evaluation rationale.
