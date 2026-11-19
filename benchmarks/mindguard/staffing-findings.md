# Four decision models under reviewer scarcity

**The same classifier can get more cases reviewed on time while sending fewer alerts.** At six concurrent reviewers, GPT-6 Luna Decisions with its native 0.10 cutoff caught 12.5% and started 2,568 reviews per simulated month. Calibrating the score to the panel majority on separate training folds, then applying 0.10, caught 50.0% with 1,468 starts. Under the native cutoff, seven of eight positive source dialogues had already spent their only alert on a turn without majority endorsement. A faster reviewer cannot undo that decision. This is a result about the *first-ticket rule and reference target*, not a judgment that those alerts were clinically inappropriate.

## What the experiment measures

We paired the [observed Open Up arrival trace](https://formative.jmir.org/2024/1/e46823/) (9,175 arrivals) with scored trajectories from the [MindGuard role-play study](https://arxiv.org/html/2602.00950v2) in eight seeded placements. Of 67 source dialogues, 50 were rated at every observed turn; 18 of their 742 turns had majority self-harm endorsement, in eight dialogues from seven distinct opening families. Five folds held out whole opening families. The clock is observed, but which dialogue arrived when, the two-minute turn spacing, the 20-minute service time, and continuous staffing are simulated. No queue decisions or patient outcomes were observed. See [methods, assumptions and reproduction](staffing.md).

For the new comparator we collected 1,134 valid Luna study receipts on the pinned OpenAI-only route; study charges were **$0.1316014** and the synthetic canary **$0.0000230**. One interrupted attempt reconciled to zero additional key usage before its authorized retry. Historical Jev, Gemini and Liquid responses came from cache. No clinician votes were sent. The offline four-model run exactly reproduced the original three-model results across 75 staffing/policy cells.

## Timely recall at the base cutoff

Each cell is a percentage in **fixed / paced / RL stopping** order. A catch requires the first nominated turn to have majority support and review to start within ten minutes of that turn. The three approaches receive identical staffing and arrivals, **not necessarily the same number of reviews**.

| Concurrent reviewers | Jev | Gemini Decision 1 | Liquid D1 Ultra | GPT-6 Luna Decisions |
|---:|---:|---:|---:|---:|
| 1 | 40.4 / 39.8 / 52.3 | 32.1 / 36.7 / 44.3 | 32.4 / 40.7 / 39.3 | 33.4 / 32.5 / 32.4 |
| 2 | 56.8 / 56.4 / 69.9 | 45.4 / 48.2 / 58.6 | 45.4 / 49.8 / 48.6 | 46.0 / 46.0 / 45.4 |
| 3 | 61.4 / 61.2 / 74.0 | 49.1 / 50.0 / 61.8 | 49.0 / 50.4 / 49.9 | 49.3 / 49.3 / 49.1 |
| 4 | 62.4 / 62.3 / 74.9 | 49.9 / 50.2 / 62.4 | 49.9 / 50.3 / 50.0 | 49.9 / 49.9 / 49.9 |
| 6 | 62.5 / 62.5 / 75.0 | 50.0 / 50.0 / 62.5 | 50.0 / 50.0 / 50.0 | 50.0 / 50.0 / 50.0 |

Three checks change the interpretation:

- **Use a strong fixed baseline.** At one reviewer, Jev RL achieved 52.3% with 994 review starts versus 40.4% with 1,063 starts at the default fixed cutoff. A simple cutoff matched to RL's *training* nomination count already reached 50.3% with 985 starts. At six reviewers, that comparator and RL both reached 75.0% with 1,468 starts. For Gemini, the comparator reached 87.5% with 1,468 starts versus RL's 62.5% with 1,284. The last two workloads differ; this is not an equal-workload win.
- **Keep both clocks.** At one reviewer, Jev RL falls from 52.3% to 49.1% when the deadline also runs from the first endorsed turn; Luna RL falls from 32.4% to 30.5%. Waiting for a later, clearer turn is not free. At six reviewers Luna fixed still missed half the source positives: of eight positive dialogues, three first alerted on unsupported turns and one never triggered. None was lost to a full queue.
- **Change the cutoff without tuning on test cases.** At six reviewers, moving the base cutoff from 0.05 to 0.10 to 0.20 gave Luna calibrated fixed recall of 37.5%, 50.0%, 50.0%; RL yielded 50.0%, 50.0%, 0.0% (551 reviews at 0.20). The same numeric cutoff has different meanings: a probability in fixed/paced, and an advantage over waiting in RL. These are sensitivity scenarios, not three independent experiments.

## Are the errors independent?

On the 742 unique held-out source turns, we correlated each model's calibrated probability minus the observed binary majority. Pairwise Pearson correlations follow. Deletion ranges omit one of the 39 opening families at a time, keeping scores fixed; **they are not confidence intervals**.

| Pair | Residual correlation | Family-deletion range | Both miss an endorsed turn (of 18) | Both miss a positive dialogue under uncongested first-ticket rule (of 8) |
|---|---:|---:|---:|---:|
| Jev / Gemini | .888 | .868–.917 | 0 | 3 |
| Jev / Liquid | .879 | .856–.908 | 1 | 3 |
| Gemini / Liquid | .930 | .924–.941 | 0 | 4 |
| Jev / Luna | .581 | .484–.762 | 0 | 3 |
| Gemini / Luna | .669 | .600–.812 | 0 | 3 |
| Liquid / Luna | .616 | .541–.756 | 0 | 3 |

At the calibrated 0.10 cutoff, Gemini missed zero endorsed turns, Jev and Liquid each missed the same one, and Luna missed four different turns. **No endorsed turn was missed by all four, but three of eight positive dialogues were missed by all four when the first alert locked the ticket.** All three Jev dialogue misses were also Luna misses; Luna's more distinct turn-level errors did not rescue a single missed Jev dialogue under that rule. This is not an evaluation of a fitted four-model ensemble. A common panel target and difficult examples can induce residual correlation; these statistics do not identify error independence conditional on true clinical risk.

**Paper implication:** distinguish probability calibration, localization of the first actionable turn, and capacity to review it. Benchmark a learned stopping policy against a calibrated, training-workload-matched threshold, not just a default cutoff. On a new stream, freeze both policies and compare timely supported reviews at matched *realized* review-minutes; audit sampled nonalerts and early alerts, and run a prespecified alternative that permits re-escalation. Repeating these same eight positive paths across a month tests queue mechanics, not generalizability. More independent positive paths and actual service-time measurements are the priority.
