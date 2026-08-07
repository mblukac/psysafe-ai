# Design principles and research basis

PsySafe is built around a simple claim: a classifier is useful only when its policy question, evidence boundary, workflow location, downstream action, and evaluation are explicit.

## Screen observable policy evidence, not people

Mental-health and support contexts are sociotechnical: model quality is only one part of consent, access, human oversight, escalation operations, and downstream impact. PsySafe therefore returns narrow observable categories and leaves the action to the application. It does not infer diagnoses or stable traits, recommend clinical interventions, or claim that a non-match means safety.

This direction is consistent with [sociotechnical safety evaluation](https://arxiv.org/abs/2310.11986), the [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), and the [WHO summary of a supported 2026 expert workshop on responsible AI for mental health and well-being](https://www.who.int/news/item/20-03-2026-towards-responsible-ai-for-mental-health-and-well-being--experts-chart-a-way-forward).

The support-needs classifier borrows interaction-oriented categories from the UK's [FCA vulnerable-customer guidance](https://www.fca.org.uk/publications/finalised-guidance/guidance-firms-fair-treatment-vulnerable-customers), while deliberately screening disclosed needs rather than assigning a persistent person-level label. The self-harm taxonomy is descriptive only; clinical tools such as [SAMHSA SAFE-T](https://library.samhsa.gov/product/SAFE-T-Pocket-Card-Suicide-Assessment-Five-Step-Evaluation-and-Triage-for-Clinicians/sma09-4432) require qualified assessment and are outside this library.

## Prefer small policies and typed outputs

Modern structured-output APIs can enforce a typed schema directly. PsySafe prompts therefore contain the policy scope, role boundary, signal definitions, evidence rules, injection rule, and output task—without duplicated prose, requested chain-of-thought, a numerical rubric, or examples that accidentally become a second policy.

Provider instructions and untrusted conversation content use separate API fields. Typed outputs, bounded vocabularies, and local validation reduce ambiguity, but do not eliminate model errors or prompt injection. Tests assert both the prompt contract and adversarial response handling.

## Calibrate a named evidence boundary locally

The provider labels each finding as `explicit`, `contextual`, or `ambiguous` without seeing the selected sensitivity. The application then applies a stable, monotonic boundary: `precise`, `balanced`, or `precautionary`.

This is not statistical probability calibration. It is a testable policy boundary that lets teams compare the operational consequences of narrower and broader evidence without making repeated provider calls. The correct boundary depends on the cost of both missed matches and unnecessary interventions at a specific checkpoint.

## Abstain instead of manufacturing certainty

PsySafe uses `indeterminate` when no trustworthy categorical decision exists. A provider refusal, timeout, malformed response, incomplete context, or bounded-output saturation is not evidence for `not_matched`.

The library omits generic per-item self-reported confidence because confidence elicitation and guard-model calibration can change under model choice, prompting, distribution shift, and adversarial inputs. Research such as [Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063) and [guard-model calibration under adversarial conditions](https://arxiv.org/abs/2410.10414) motivates treating this as an empirical, task-specific problem rather than a universally meaningful field. A deployment with a validated probabilistic calibration layer can build it outside PsySafe; it must not silently convert `indeterminate` into clean.

## Evaluate boundaries, families, slices, and actions

Aggregate precision/recall rates are useful dataset statistics, not confidence in one person or message. PsySafe evaluates exact categorical expectations at all named boundaries, checks monotonicity, tracks abstentions and intervention burden, and prevents related families from crossing tuning and holdout splits.

Useful evaluation should include diverse, adversarial, and context-specific cases. The approach is informed by [NIST AI TEVV](https://www.nist.gov/ai-test-evaluation-validation-and-verification-tevv), Google's work on [diverse evaluation data](https://research.google/pubs/the-reasonable-effectiveness-of-diverse-evaluation-data/), the [DICES conversational safety dataset](https://research.google/pubs/dices-dataset-diversity-in-conversational-ai-evaluation-for-safety/), and [automated adversarial discovery for safety classifiers](https://research.google/pubs/automated-adversarial-discovery-for-safety-classifiers/).

Automated metrics are necessary but insufficient for consequential use. Add expert review, affected-user input, operational exercises, privacy review, and monitoring of real downstream actions.

## Check where execution can change course

Sequential systems expose input, task-selection, execution, tool, and communication boundaries. Agent SDKs expose lifecycle guardrails and hooks. PsySafe uses one framework-neutral gate contract across both, then adapts to native SDK objects.

Preventive checks belong before irreversible actions. Post-tool checks can stop subsequent execution but cannot undo the tool. Final-output guardrails are not automatically streaming filters. The OpenAI Agents SDK and Claude Agent SDK expose different coverage, so integrations document rather than hide those gaps. See the official [OpenAI Agents guardrail guide](https://openai.github.io/openai-agents-python/guardrails/) and [Claude Agent SDK hook guide](https://code.claude.com/docs/en/agent-sdk/hooks).

## Keep claims proportional to evidence

PsySafe can screen, abstain, and route. It cannot prevent every harm, establish legal compliance, de-identify arbitrary text, determine that a user is safe, or replace human responsibility. Documentation, code contracts, prompts, examples, tests, and package metadata all use that same boundary.
