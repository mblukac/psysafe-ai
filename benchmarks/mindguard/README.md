# MindGuard: does a three-model review queue catch more at the same workload?

This is an offline, repository-only research pilot, not a PsySafe classifier or clinical tool. The outcome is a clinician's **self_harm** vote under MindGuard's three-way rubric; `harm_others` is a different concern, not necessarily a safe turn. The comparison does not measure patient outcomes, private clinician beliefs, action preferences, or PsySafe's three evidence boundaries.

For a separate experiment with finite reviewer service times and real arrival intervals, see the [trace-driven staffing replay](staffing.md). Its reference is **majority endorsement of the nominated turn**, not an individual vote.

For time-sensitive human review, start with [When does a safety flag become a timely human review?](reviewer-surges.md). It uses real AI and crisis-service arrival clocks to distinguish missed nominations from tickets lost to queues, and tests whether staffing, sensitivity or dispatch changes that loss.

For the full clinician-rated cohort, start with the [time-sensitive human-review pilot results](full-panel-findings.md). This is a different endpoint: a shared review desk for **either** risk category, timed from the first observed majority endorsement. The [technical draft](timely-human-review-draft.md) describes the study design and the still-proposed customer-support comparison.

The [paired ticket-update experiment](ticket-updates.md) tests what happens when later evidence can change an existing alert at the same FIFO review start. It compares simple and Bayesian rules on all six cached detectors at identical simulated reviewer-minutes.

For **where conversations are lost and which fix buys the most timely catches**, see the [full-cohort loss and equal-work comparison](ticket-updates.md#where-cases-are-lost-and-which-fix-buys-the-most-catches). It separates failure to flag, failure to reach review in time, and a stale ticket at pickup.

## Start without data or credentials

From the repository root, with Python 3.10 or later:

```sh
python -m benchmarks.mindguard smoke
```

The smoke creates 18 invented dialogues / 54 turns and three fake native Decisions responses per turn. It takes the same request, parser, fit, queue, and replay paths as the cached evaluation. JSON output is marked `synthetic_smoke_not_benchmark_results`. The `smoke`, `prepare`, and `evaluate` commands do not send requests, read an API key, or download a dataset.

## Real data path (later, after access and collection approval)

1. Obtain the [official MindGuard test release at the pinned revision](https://huggingface.co/datasets/swordhealth/MindGuard-testset/tree/1c082b67bc0a6574badc3bf4cbe3e12af130f606) under its access conditions (CC BY-NC-SA 4.0). Confirm permission to process the texts on the proposed provider routes before any collection. The Parquet filename contains “train” but this release is the clinician-rated testset, not the separate synthetic training set. Keep records, request files and response caches in private storage, never in git. Use local Parquet (requires an existing `pyarrow` installation) or a JSONL export preserving the official fields and label strings. Verify provenance locally: shape checks alone cannot authenticate a JSONL export.
2. Generate requests **locally**. Use an existing private directory, e.g.:

   ```sh
   python -m benchmarks.mindguard prepare --annotations /private-work/mindguard.parquet --out /private-work/requests.jsonl
   ```

   The new file contains sensitive text and is created owner-only on POSIX; the command refuses to overwrite. The pinned release has 1,134 rated turns and 67 maximal observed dialogue paths. For shared prefixes, the importer requires one observed row per matching path and identical *ordered* clinician votes; ambiguous assignments stop the run. Missing ratings stay missing: a later observed prefix can establish that an earlier user message existed, but does not supply that turn's votes. Neither labels, dialogue identifiers nor future messages are sent in the model payload. The official files were **not** accessed in the smoke.
3. **Collection is deliberately separate.** Consult the [verified wire contract and model pins](providers.md). Before any approved study calls, check availability and test one **invented** canary per model. The following commands are **dry runs** (POSIX); they read no key and send nothing:

   ```sh
   python -m benchmarks.mindguard.collect canary --cache /private-work/canary.jsonl --max-requests 3
   python -m benchmarks.mindguard.collect batch --annotations /private-work/mindguard.parquet --canary-cache /private-work/canary.jsonl --cache /private-work/responses.jsonl --max-requests 3
   ```

   Only after separate authorization, the collector can use `OPENROUTER_API_KEY` with `--allow-paid`. Batch additionally requires successful receipts for all three invented canaries and `--allow-gated-provider-processing`: confirm the dataset terms, account routing, retention and provider policies **before transmission**. Validating a response is not a privacy control. The request cap is per invocation, not a dollar cap. Successful responses are saved incrementally to an owner-only private cache; no automatic retries. If a send times out or validation fails, reconcile billing before resuming. If a provider returns a conflicting ID, refusal, only the winner/confidence, or an incomplete vector, stop before the study; never substitute a model. Full collection requires 3,402 successful study responses, not including canaries. An external cache must likewise use one JSONL envelope per request with the original `request_id`, pinned `model`, and actual API JSON in `response`. The request ID hashes the exact rubric, prefix and model payload; it binds records, not identities or permissions.
4. With an existing complete cache, run:

   Study request IDs use the versioned digest of `["mindguard-case-request-v2", dialogue_id, absolute_user_turn_index, payload]` to distinguish occurrences of identical inputs in different dialogues, including in a shared opening. Old payload-only **study** caches do not match; synthetic canary receipts are unchanged. The release importer preserves a literal empty user message as observed, including in the later history; it never inserts placeholder text or creates a vote for an unannotated turn.

   ```sh
   python -m benchmarks.mindguard evaluate --annotations /private-work/mindguard.parquet --responses /private-work/responses.jsonl --budget 0.10 --folds 5 --seed 2026 --bootstrap 1000
   ```

   Choose the budget **before** inspecting outcomes. Defaults are a pilot protocol, not a clinical operating recommendation. Missing, duplicate, mismatched and extra responses invalidate the comparison. Reports contain counts and fold-level summaries, never dialogue text.

## What is fitted and what is held out?

The pinned release uses `safe`, `unsafe_self_harm_risk`, and `unsafe_harm_to_others` in the individual clinician annotations. The importer maps these exactly to `safe`, `self_harm`, and `harm_others`, respectively, also accepting canonical names in local exports. Unrecognized labels fail the import; no vote is replaced by the majority.

All three providers see the same conversation prefix and one mutually exclusive choice question. Use the native self-harm option probability, **even if the winning label is safe**. The fit target per turn is the fraction of the three independent clinicians who marked self-harm. The binary log loss is `-h log(p) - (1-h) log(1-p)`: equivalent to scoring all three votes. The Brier score is likewise averaged over individual votes; turns have equal weight. Neither metric treats a 2:1 vote as a certain majority label.

Compare calibrated Jev, Gemini, and Liquid individually; the raw arithmetic average (`equal`); Jev + Gemini (`pair`); all three (`stack`); and the best calibrated single selected on inner development folds (`best_single`). Learned methods use logistic regression on native logits, clipping probabilities at 1e-6 for numerical stability. Slopes shrink toward 1 / number of models (penalty coefficient 0.05), intercept toward zero (0.001). No assumption of independent model errors and no hyperparameter sweep. A better log score need not buy another supported review.

Split by **whole dialogue**, keeping dialogues with an identical opening together in both outer and inner folds. Balance majority-positive, minority-only and no-positive dialogues across folds as family size permits. For each outer fold, the best single and the referral thresholds use inner out-of-fold scores from development dialogues only. Refit on the full development fold and evaluate once on its held-out dialogues. Given 67 role-play dialogues and one three-clinician panel, this is internal cross-validation, not independent external validation. See the [MindGuard paper](https://arxiv.org/html/2602.00950v2).

## Read the two workload reports separately

Replay uses **annotated turns only**. Of the 1,156 user positions reconstructed from the maximal observed paths in the pinned release, 22 do not have a rating. A referral at an unrated turn is unassessable, not a false alarm. The main replay does not score or refer at those positions. The `complete_dialogue_live_sensitivity` table restricts the same fitted predictions and frozen thresholds to the 50 dialogues with no such gaps. Bootstrap intervals are conditional on the panel and fitted models, resampling dialogues within each fold, not entire opening families.

- **`fixed`: retrospective queue.** Take each held-out dialogue's highest-scoring turn (earliest if tied), then rank those nominations. Use `--slots K` for an exact total, or round `--budget × total dialogues` once (half up). Apportion that total across held-out folds by size using largest remainders, identically for every method. For example, 10% of the 67 official conversations means 7 reviews per method. Equal dialogue peaks are broken by a label-blind stable key. One catch requires at least two clinicians endorsing the *nominated turn*. An ever-positive dialogue with no supported review is a miss, even if a different turn was reviewed.
- **`live`: chronological replay.** In development, find a threshold on dialogue maxima that targets the same slots without splitting ties or exceeding capacity. Freeze it for this fold, then take only the **first** turn to cross it in each held-out dialogue. An unsupported early alert consumes the one review; it does not earn retrospective credit if a later turn becomes positive. Held-out workload can differ by model: compare actual `reviews`, `catches`, `unsupported`, `premature` and `misses`, not just recall. No claim about the effect of a real intervention follows from replay.
- `eligible_cases` are dialogues with any majority-positive turn. `endorsed_vote_mass` sums the actual vote fractions at reviewed turns, so a minority endorsement is not erased. `stack_minus_best_single` reports count differences and gains/losses on discordant dialogues; uncertainty resamples whole dialogues **paired within each held-out fold**, rebuilding the retrospective queue with frozen predictions and thresholds. These intervals describe case mix conditional on the fits, not variation from refitting or sampling other clinicians.

### Do the three providers miss the same cases?

Read `error_dependence` alongside the ensemble comparison. Among **eligible conversations only**, compare each pair of individually calibrated models under the same queue or live rule: both missed, only the first missed, only the second missed, or neither missed. Report the number all three missed, not just the pairwise correlation (phi). Among conversations with **no majority-endorsed self-harm turn**, report the corresponding overlap in false alerts separately. If either model catches every eligible conversation or misses every one, its miss indicator has no variation and the correlation is undefined, not zero. In live replay, consult the actual per-model review counts before interpreting overlap.

The other comparison is continuous: for each held-out turn subtract the fraction of clinician self-harm votes from each model's self-harm probability. Compare pairwise Pearson correlations of those **signed errors**, for native probabilities and individually calibrated probabilities separately. Agreement on the many safe turns, correlation of the probabilities themselves, and correlation of errors are different things. All three residuals use the same three-vote reference, so shared annotation noise can induce positive correlation. The diagnostics describe errors against this panel; they do not test conditional independence given a latent clinical state. Paired uncertainty resamples dialogues rather than treating adjacent turns as independent.
