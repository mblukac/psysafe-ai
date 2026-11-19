# Timely review under staffing scarcity

This offline experiment couples real model responses and expert role-play annotations to a **separate** public traffic trace. It estimates performance in a specified simulated workflow, not clinical recall, actual harm, or Open Up's operational performance.

See the [four-model results and limits of inference](staffing-findings.md).

## Reproduce

Use the project Python environment, local `pyarrow` for Parquet, the [authorized MindGuard inputs and complete historical cache](README.md), and any private working directory. The simulator and figure generator do not use the network or read API credentials.

```sh
cd benchmarks/mindguard/private
curl --fail --location https://raw.githubusercontent.com/cathug/crisis-des/4fc679321b19bdff91d59f765537c86296cfa1fa/real_interarrivals_nov.csv -o real_interarrivals_nov.csv
cd ../../..
python -m benchmarks.mindguard.staffing \
  --annotations benchmarks/mindguard/private/mindguard.parquet \
  --responses benchmarks/mindguard/private/responses.jsonl \
  --trace benchmarks/mindguard/private/real_interarrivals_nov.csv \
  --staffing 1 2 3 4 6 --placements 8 --seed 2026 \
  --threshold 0.10 --turn-minutes 2 --service-minutes 20 --deadline-minutes 10 \
  > benchmarks/mindguard/private/staffing-results.json
python -m benchmarks.mindguard.staffing_plot \
  --report benchmarks/mindguard/private/staffing-results.json \
  --out benchmarks/mindguard/private/staffing-recall.png
```

Plotting requires optional `matplotlib`; it takes only aggregate JSON. Keep all raw data and model caches outside git. Do not refill missing scores, infer votes for missing turns, or use the last dialogue length as a predictor.

To add OpenAI GPT-6 Luna Decisions, follow the [separate collection, billing and data-permission protocol](providers.md). Add `--luna-responses benchmarks/mindguard/private/luna-responses.jsonl` to the replay only after the entire cohort has validated. An incomplete file fails; omitting the argument runs the original three models with exactly the same placements. The replay never completes a missing cache by making paid calls.

## Observed versus simulated

- **Arrivals are observed.** [Chiang, Law & Yip (2024)](https://formative.jmir.org/2024/1/e46823/) released [9,175 November 2020 visitor interarrival intervals from Hong Kong's Open Up service](https://github.com/cathug/crisis-des/blob/4fc679321b19bdff91d59f765537c86296cfa1fa/real_interarrivals_nov.csv). The input is a single headerless column in minutes, SHA-256 `1414173cc5fbf6d2f50ca1345b38ce6aedd0ec92793af5960a4fb646cb243e29`. Use the cumulative sum of **all** intervals, including a zero gap, without truncating the final arrival or looping the trace.
- **Source dialogues and scores are observed; their pairing to arrival times is simulated.** Of 67 MindGuard paths, 50 (742 turns, 39 opening families) have labels at every observed user position. Eight of the 50 have at least one majority-endorsed self-harm turn. Use only these paths, train/test split by opening family in five folds. Exclude the 17 incomplete paths rather than pretending a missing vote is negative. Apply each fitted policy to its held-out paths. The mixed stream uses out-of-fold predictions, **not** one independent prospectively tested fit.
- **Service is simulated.** Every visitor receives a dialogue, even though not every real visitor became a counseling client. Assign each path 183 or 184 arrivals, randomly permuted across the full trace using a seed, preserving observed within-dialogue score/label trajectories. Hold all eight placements fixed across policies, staffing levels and models. Assume constant reviewer availability, deterministic 20-minute nonpreemptive reviews, 2 minutes between user turns, one ticket per dialogue, queues empty initially, service continues after the last arrival, no feedback into conversations. Provider latency, missed traffic during outages, staff breaks, case-mix changes and abandonment before nomination are not modeled.

## Policies and measures

**Fixed:** estimate the current majority endorsement probability with a per-provider regularized logistic calibration on training data. Nominate the first turn with probability at least 0.10, serve in arrival order. **Paced:** use the same current-turn estimate; estimated congestion raises the cutoff by up to 0.10. Rank pending tickets by their locked metric plus a waiting-time allowance (up to 0.10 within the deadline). **RL stopping:** replace that current-turn estimate with its estimated advantage over waiting; use the identical paced admission and dispatch mechanism. Train the wait value by discounted fitted Q iteration (gamma 0.95), using bins of score and elapsed turn with shrinkage. A terminal dialogue has zero continuation value. The policy sees no future turn, clinical label or remaining dialogue length. RL learns **when to stop in a dialogue**; it is not an optimal controller for the whole queue or a treatment policy.

The first admitted turn is locked; no duplicate tickets, refreshed deadlines or second chances after expiry. At equal timestamps, release servers, observe new scores, dispatch by policy, then expire. A catch requires a reviewer to **start** within the deadline of a nominated, majority-endorsed turn. All ever-endorsed test dialogues enter the recall denominator, including those never flagged. The secondary `recall_by_onset_deadline` additionally requires a start within the same time limit from the **first** endorsed turn. It does not credit an unsupported early alert. Misses separate into no nomination, wrong turn nominated, or supported nomination expiring. No panel endorsement means no credit; it does not establish what a real clinician should do.

The aggregate report includes actual reviews and review-minutes, not only equal staff counts, and two score-only controls: `native_fixed` uses the uncalibrated provider score with FIFO; `matched_fixed` matches the RL policy's *uncongested number of nominated training dialogues* as closely as possible without exceeding it or splitting ties. It is **not** an equal-realized-workload comparison. The 0.10 cutoff is in probability units for fixed/paced but in advantage units for RL. Use `--threshold` for sensitivity analysis, not to select a winner on these same eight cases.

Recall is pooled numerator/denominator over placements. Placement minima/maxima describe only variation in assigning these same cases to these same arrival times. They are not sampling confidence limits; repeated arrivals are not new clinical evidence. Vary service duration, timing, threshold and prevalence assumptions before considering any deployment. Independent real service-time, staffing and adjudicated case data would be required for a staffing recommendation.

## Do the four classifiers miss the same cases?

Run this offline, using the same fold seed and base threshold as the staffing experiment:

```sh
python -m benchmarks.mindguard.staffing_dependence \
  --annotations benchmarks/mindguard/private/mindguard.parquet \
  --responses benchmarks/mindguard/private/responses.jsonl \
  --luna-responses benchmarks/mindguard/private/luna-responses.jsonl \
  --seed 2026 --folds 5 --threshold 0.10 \
  > benchmarks/mindguard/private/staffing-dependence-four.json
```

The unit matters. On all unique source turns we correlate signed forecast errors (score minus the panel majority); the common reference can induce correlation even if the underlying classifiers differ. Among endorsed turns only, we count which pairs miss the same turn at the cutoff. Among ever-endorsed dialogues, we separately count which pairs fail to nominate an endorsed turn on their **first** threshold crossing with unlimited review capacity. This last measure isolates nomination timing from congestion; it is not the six-reviewer replay. None of these analyses pools the repeated traffic placements as new clinical cases.
