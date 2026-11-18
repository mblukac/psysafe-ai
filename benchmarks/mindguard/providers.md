# What the OpenRouter models return

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
