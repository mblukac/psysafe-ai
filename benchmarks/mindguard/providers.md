# What the OpenRouter models return

## Additional full-cohort comparators: Clef Flash and Decider Flash

The full-panel runner adds [Clef **Flash**](https://openrouter.ai/cloudflare/clef-flash)
on DeepInfra (only `deepinfra/fp8`; canonical model slug is undated), and
[Perplexity Decider Flash V1.2](https://openrouter.ai/perplexity/pplx-decider-flash-v1.2)
on Perplexity (only `perplexity`; dated model ID in code). No Clef Pro, Workers AI, or fallback.
On 2026-11-20 the routes were listed in the [ZDR catalog](https://openrouter.ai/api/v1/endpoints/zdr)
at $0.019 and $0.01 per million input tokens respectively, with zero output or per-request fee.
Recheck current listings and permissions before spending. Prices and ZDR listing are not permission
to disclose gated research material. Keep optional OpenRouter content logging off.

Dry-run each cohort in the same owner-only private directory. Replace `clef-flash` with
`decider-flash` for the other route:

```sh
D=benchmarks/mindguard/private
python -m benchmarks.mindguard.collect_flash canary --cohort clef-flash --private-dir "$D" --max-requests 2
python -m benchmarks.mindguard.collect_flash batch --cohort clef-flash --private-dir "$D" \
  --annotations "$D/mindguard.parquet" --max-requests 1134
```

After explicit route approval, add `--allow-paid` for the two synthetic long-prefix canaries;
they must return opposite expected choices and bill at least 5,000 input tokens apiece.
The study additionally requires `--allow-gated-provider-processing`, the pinned Parquet hash,
and no state longer in bytes than the synthetic checks. This catches obvious prefix
truncation; it does not prove attention to every token. Each request requires no fallback,
zero data retention, no data collection, and the pinned maximum prices. Keep all private
receipts and reservation ledgers across restarts. Combined route-specific limits are
$0.06 + $0.04 including the synthetic checks. Stop on unverified charges or partial failures:
do not reset ledgers, silently substitute a route, or treat missing scores as negatives.

**Clef serialization and the bounded second synthetic check.** Cloudflare's [published
answer serializer](https://huggingface.co/Cloudflare/clef-flash/blob/main/joint_schema_model.py)
rounds each option independently to four decimal places, after choosing the winner.
Three four-decimal options may sum to 0.9999 or 1.0001. Preserve the values and accept
only the model-specific rounding tolerance, not arbitrary unnormalized scores.
The first Clef synthetic pair produced one valid receipt and one rejected choice; the
original rejected vector was not retained and its exact failure cause is unknown.
For this specific reconciled synthetic-only condition, pass `--clef-canary-v2` to
both Clef commands. The runner checks the failed generation's [billing metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
against its journal, retains both original charges, and requires two new, distinct,
opposite long-tail choices before the study can start. There is no corresponding
retry rule for study failures. A clean replication with two valid original canaries
does not need v2.

## The original three providers

**Verified from public metadata and provider documentation on 2026-11-18; not from a paid probe.**
An unauthenticated GET of the [decision-output catalog](https://openrouter.ai/api/v1/models?output_modalities=decisions)
listed these three snapshots with route `/api/alpha/decisions`. Use the dated canonical slugs, not floating
`latest` aliases. Payload construction, probability parsing, smoke and evaluation never use the network.
Any later inference is isolated in the separately opted-in collector.

| Requested family | Canonical slug | Evidence for what a choice returns |
| --- | --- | --- |
| [Jev 1.14](https://openrouter.ai/typesafe/jev-1.14) | `typesafe/jev-1.14-20261014` | OpenRouter's [recorded Jev response](https://openrouter.ai/blog/insights/what-is-jev/) shows the winner, a map of probabilities keyed by every caller-specified option, and a separate concentration/confidence field. It includes examples of exact zero and one. |
| [Gemini Decision 1](https://openrouter.ai/google/gemini-decision-1) | `google/gemini-decision-1-20261111` | Google [documents normalized probabilities for every choice](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/), as well as the winner. The OpenRouter catalog routes this snapshot to Decisions, not to chat. |
| [Liquid D1 Ultra](https://openrouter.ai/liquid/d1-ultra) | `liquid/d1-ultra-20261015` | Liquid [documents per-label probabilities for choice questions](https://www.liquid.ai/blog/d1-ultra). The OpenRouter catalog lists **expiry 2026-11-24**. Recheck before any future collection and stop if unavailable; D1.2 Ultra is a different model, not a substitute. |

Use one three-way `choice`, not three separate yes/no (`noul`) questions or an ordinal `score`.
The question is under `questions.mindguard`; its criteria define the mutually exclusive options.
The conversation prefix is encoded as JSON text in `state` and must end at the current user message;
no clinician labels or future replies belong in it. The offline request builder contains the fixed rubric.

Read **`answers.mindguard.probabilities.self_harm`** even when the winner is a different option.
The other two probabilities remain in the parsed vector. None is a measured clinician posterior, risk of
future self-harm, model-authored rating, or chat token log probability. The optional auxiliary
`confidence` is not a substitute. The [OpenRouter Decisions guide](https://openrouter.ai/docs/guides/features/decisions)
documents a normalized map for choice; its TypeScript example marks that field optional and a refusal has
no probabilities. Below is a **synthetic format example**, not an observed model result:

```json
{
  "model": "typesafe/jev-1.14-20261014",
  "answers": {
    "mindguard": {
      "type": "choice",
      "choice": "safe",
      "probabilities": {"safe": 0.70, "self_harm": 0.20, "harm_others": 0.10},
      "confidence": 0.55
    }
  }
}
```

The parser requires a matching dated model, non-refused choice, all three finite probabilities in
`[0,1]`, and a consistent winning label (ties allowed). It preserves exact zeros, ones and returned
precision. For a three-way vector reported in whole hundredths, totals of 0.99–1.01 are consistent
with independent rounding; otherwise the sum tolerance is `1e-6`. It does not normalize the values
or impute missing mass. Unknown model, winner-only result, refusal or malformed vector = failed row,
**not** a negative label or a guessed probability. If a later authorized synthetic canary contradicts
this documented schema, keep the response for investigation and stop before sending any dataset text.

## Optional fourth replay comparator: OpenAI Luna

The three original snapshots and their receipt IDs above stay unchanged. Opt in to
`--cohort luna` to collect **only** [GPT-6 Luna Decisions](https://openrouter.ai/openai/gpt-6-luna-decisions),
dated snapshot `openai/gpt-6-luna-decisions-20261006`, using the same three-way rubric.
Luna is a new processor: previous permission for the three-model pilot is not permission
to send this dataset to OpenAI. Before any authorized batch, confirm research-use terms,
disable optional OpenRouter content logging/broadcasts in the account and verify privacy
settings. The request includes `provider.only=["openai"]` and `allow_fallbacks=false`;
the collector separately checks the returned provider is OpenAI. No Azure or silent
fallback. See the [routing controls](https://openrouter.ai/docs/guides/routing/provider-selection).
OpenAI [documents](https://developers.openai.com/api/docs/guides/your-data) no training
by default and up to 30 days of standard abuse-monitoring retention (exceptions apply).
Do not assume this OpenRouter route has contractual zero retention: recheck the
[ZDR catalog](https://openrouter.ai/api/v1/endpoints/zdr) and account settings if required.

On 2026-11-18 the [endpoint catalog](https://openrouter.ai/api/v1/models/openai/gpt-6-luna-decisions/endpoints)
listed US $0.10/M input tokens, $0 output, and a 1,050,000-token context window.
For 1,134 original rated prefixes, serialized state + questions total 6,070,406 UTF-8
bytes. At four bytes per token the no-cache estimate is **$0.152**, excluding
the invented canary. This is not a cap or exact tokenization; the other three
routes billed 1.28–1.60M input tokens on the same text. Verify prices before spending.

Run the following from the checkout, with separate owner-only files in the gitignored
private directory. Without `--allow-paid` this only preflights locally:

```sh
python -m benchmarks.mindguard.collect canary --cohort luna \
  --cache benchmarks/mindguard/private/luna-canary.jsonl --max-requests 1
python -m benchmarks.mindguard.collect batch --cohort luna \
  --annotations benchmarks/mindguard/private/mindguard.parquet \
  --cache benchmarks/mindguard/private/luna-responses.jsonl \
  --canary-cache benchmarks/mindguard/private/luna-canary.jsonl \
  --max-requests 1134 --workers 8
```

For an explicitly authorized run, the canary requires `--allow-paid`; the gated batch
requires both `--allow-paid` and `--allow-gated-provider-processing`. A three-model
canary cache cannot unlock Luna. Luna collection reserves $0.105 per in-flight request
(the full pinned context at the enforced price ceiling) and caps this canary + batch
cache pair at **$0.30** in router credits. The request limits prompt price to $0.10/M,
completion to zero and per-request fees to zero. Actual spending uses `usage.cost`,
not an estimate. OpenRouter must honor the current documented model limits and prices.
A separate private `.attempts.jsonl` journal reserves each request before submission.
Use the same cache pair and journal across invocations; do not delete, copy or reset
them to bypass a stop. This does not impose an account-wide cap on other clients.
Missing/malformed charges or unreconciled attempts block further submissions.
A future failed call writes a private `.failures.jsonl` sidecar with its locally
hashed request ID, a bounded failure category, HTTP status when known, and the
recognized generation ID / numeric reported charge when validated. It never
copies exception text, provider error/refusal text, prompts or answers. Such
metadata is for operator reconciliation; it cannot clear the reservation or
authorize a retry. Missing historical diagnostics cannot be reconstructed locally.
A single operator-authorized recovery is available only for a **single** unscored
attempt after independently reconciling its original charge. Use the same paid Luna
batch command and paths, replacing worker count and cap with
`--workers 1 --max-requests 1 --retry-reconciled-request <local-request-hash>
--reconciled-charge-usd <verified-decimal>`. Do not infer a zero charge from a
missing receipt. The collector preserves the original attempt, durably records the
attestation in a separate owner-only `.reconciliations.jsonl`, counts the original
charge in the $0.30 budget, and only then reserves one retry. It cannot retry a
known refusal, an already-scored request, multiple unresolved attempts, or a
previous retry. A new interruption during recovery stays locked. After success,
resume fresh, untouched requests without the recovery flags.
A failed or refused turn stops collection (in-flight
receipts are drained); it is never marked negative. Reconcile ambiguous billing before
resuming. A failed/partial cohort is **not assessable**, not an estimate of deployment
recall conditional on successful calls. The offline loader `luna.load_luna_scores`
checks one matching receipt per original case before any replay subset is selected.
