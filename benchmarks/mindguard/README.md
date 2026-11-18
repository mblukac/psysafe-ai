# MindGuard: does a three-model review queue catch more at the same workload?

This is an offline, repository-only research pilot, not a PsySafe classifier or clinical tool. The outcome is a clinician's **self_harm** vote under MindGuard's three-way rubric; `harm_others` is a different concern, not necessarily a safe turn. The comparison does not measure patient outcomes, private clinician beliefs, action preferences, or PsySafe's three evidence boundaries.

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

   The new file contains sensitive text and is created owner-only on POSIX; the command refuses to overwrite. It reconstructs unbranched, contiguous dialogue chains by exact message-prefix equality, checks 1,134 turns / 67 chains, and omits votes, majority labels and future messages from the payload. If labels or roles differ, roots collide, or chains cannot be reconstructed, stop and verify the release; do not group by row order. The official files were **not** accessed in the smoke.
3. **Collection is deliberately separate.** Consult the [verified wire contract and model pins](providers.md). Before any approved study calls, check availability and test one **invented** canary per model. The following commands are **dry runs** (POSIX); they read no key and send nothing:

   ```sh
   python -m benchmarks.mindguard.collect canary --cache /private-work/canary.jsonl --max-requests 3
   python -m benchmarks.mindguard.collect batch --annotations /private-work/mindguard.parquet --canary-cache /private-work/canary.jsonl --cache /private-work/responses.jsonl --max-requests 3
   ```

   Only after separate authorization, the collector can use `OPENROUTER_API_KEY` with `--allow-paid`. Batch additionally requires successful receipts for all three invented canaries and `--allow-gated-provider-processing`: confirm the dataset terms, account routing, retention and provider policies **before transmission**. Validating a response is not a privacy control. The request cap is per invocation, not a dollar cap. Successful responses are saved incrementally to an owner-only private cache; no automatic retries. If a send times out or validation fails, reconcile billing before resuming. If a provider returns a conflicting ID, refusal, only the winner/confidence, or an incomplete vector, stop before the study; never substitute a model. Full collection requires 3,402 successful study responses, not including canaries. An external cache must likewise use one JSONL envelope per request with the original `request_id`, pinned `model`, and actual API JSON in `response`. The request ID hashes the exact rubric, prefix and model payload; it binds records, not identities or permissions.
4. With an existing complete cache, run:

   ```sh
   python -m benchmarks.mindguard evaluate --annotations /private-work/mindguard.parquet --responses /private-work/responses.jsonl --budget 0.10 --folds 5 --seed 2026 --bootstrap 1000
   ```

   Choose the budget **before** inspecting outcomes. Defaults are a pilot protocol, not a clinical operating recommendation. Missing, duplicate, mismatched and extra responses invalidate the comparison. Reports contain counts and fold-level summaries, never dialogue text.

## What is fitted and what is held out?

All three providers see the same conversation prefix and one mutually exclusive choice question. Use the native self-harm option probability, **even if the winning label is safe**. The fit target per turn is the fraction of the three independent clinicians who marked self-harm. The binary log loss is `-h log(p) - (1-h) log(1-p)`: equivalent to scoring all three votes. The Brier score is likewise averaged over individual votes; turns have equal weight. Neither metric treats a 2:1 vote as a certain majority label.

Compare calibrated Jev, Gemini, and Liquid individually; the raw arithmetic average (`equal`); Jev + Gemini (`pair`); all three (`stack`); and the best calibrated single selected on inner development folds (`best_single`). Learned methods use logistic regression on native logits, clipping probabilities at 1e-6 for numerical stability. Slopes shrink toward 1 / number of models (penalty coefficient 0.05), intercept toward zero (0.001). No assumption of independent model errors and no hyperparameter sweep. A better log score need not buy another supported review.

Split by **whole dialogue**, stratifying any majority-positive, minority-only and no-positive conversations. For each outer fold, the best single and the referral thresholds use inner out-of-fold scores from development dialogues only. Refit on the full development fold and evaluate once on its held-out dialogues. Given 67 role-play dialogues and one three-clinician panel, this is internal cross-validation, not independent external validation. See the [MindGuard paper](https://arxiv.org/html/2602.00950v2).

## Read the two workload reports separately

- **`fixed`: retrospective queue.** Take each held-out dialogue's highest-scoring turn (earliest if tied), then rank those nominations. Use `--slots K` for an exact total, or round `--budget × total dialogues` once (half up). Apportion that total across held-out folds by size using largest remainders, identically for every method. For example, 10% of the 67 official conversations means 7 reviews per method. Equal dialogue peaks are broken by a label-blind stable key. One catch requires at least two clinicians endorsing the *nominated turn*. An ever-positive dialogue with no supported review is a miss, even if a different turn was reviewed.
- **`live`: chronological replay.** In development, find a threshold on dialogue maxima that targets the same slots without splitting ties or exceeding capacity. Freeze it for this fold, then take only the **first** turn to cross it in each held-out dialogue. An unsupported early alert consumes the one review; it does not earn retrospective credit if a later turn becomes positive. Held-out workload can differ by model: compare actual `reviews`, `catches`, `unsupported`, `premature` and `misses`, not just recall. No claim about the effect of a real intervention follows from replay.
- `eligible_cases` are dialogues with any majority-positive turn. `endorsed_vote_mass` sums the actual vote fractions at reviewed turns, so a minority endorsement is not erased. `stack_minus_best_single` reports count differences and gains/losses on discordant dialogues; uncertainty resamples whole dialogues **paired within each held-out fold**, rebuilding the retrospective queue with frozen predictions and thresholds. These intervals describe case mix conditional on the fits, not variation from refitting or sampling other clinicians.
