# Time-sensitive human review in AI systems under scarce staffing

**Technical draft for review · Proposed study, not findings**

When an AI system flags a conversation, someone still has to be available to read it. A team may have enough reviewer-hours over a week and still miss the cases that arrive together on Tuesday afternoon. We ask: **When a small team faces a surge, which conversations reach a human in time, which slip through, and would changing the order of review actually help?**

We study this question first in mental-health conversations and then in customer support. In one setting a clinician endorses a risk signal; in the other an annotator identifies a point where a bot should hand the conversation to a person. The two settings need not have the same urgency, deadlines, or consequences. What they share is a decision: a reviewer becomes free while several conversations may need attention. Reinforcement learning (RL) is one way of making that decision. Whether it adds anything over a simple rule is an empirical question.

## The decision and the outcome

Separate three decisions. **Detection** determines whether and when a conversation generates a ticket. **Dispatch** determines which waiting ticket gets the next free reviewer. **Staffing** determines when that reviewer is available. In the main comparison, freeze the detector and staffing so that dispatch policies see exactly the same tickets. Vary detector sensitivity and staffing in separate comparisons. Otherwise a policy that sends fewer alerts, or one that uses more reviewer-hours, can look like a better scheduler.

The primary outcome is **timely review recall**: among conversations with at least one endorsed need, the fraction for which a human starts reviewing a supported turn within the specified window. Count each conversation at most once. Start the clock at its **first observed endorsed turn**, not when the detector eventually raises an alert. A never-flagged case stays in the denominator; an unsupported early flag gets no retrospective credit for a later disclosure. This measures access to review, not whether a reviewer resolved the issue or prevented harm.

For the initial comparison, each conversation can generate one ticket, at the first eligible turn crossing a frozen threshold. The ticket holds that prefix. A reviewer who starts it stays occupied until service finishes. The dispatcher sees scores and messages available by decision time, queue age and current staffing; it does not see clinician votes, true onset, future arrivals or future utterances. The service may expire a ticket based on time since the **alert**; that observable rule is distinct from the **onset** deadline used to score it. Updating an existing ticket when new evidence arrives is a separate comparison, not a change made only for RL.

## Evidence for the two applications

**Mental health.** Use all 67 conversations and 1,134 clinician-rated user turns in the [MindGuard expert release](https://arxiv.org/html/2602.00950v2#S3). Jev, Gemini Decision 1 and Liquid D1 Ultra have cached scores for every rated turn; start the dispatch comparison with Jev and Gemini separately to preserve continuity with the pilot. For a desk covering both risk categories, 17 conversations contain an observed majority-endorsed turn: nine for self-harm (20 turns) and eight for harm to others (22 turns). Report overall access and each category separately. Neither label means depression or distress in general. Preserve all three individual votes for a sensitivity analysis of disagreement.

The [earlier staffing replay](reviewer-surges.md) used only 50 conversations, eight with majority self-harm endorsement. Requiring every position in a conversation to be rated discarded 392 **rated** turns because of 22 **unrated** positions in 17 conversations. Here each entire path enters the workload. Keep the original turn indices and elapsed time; the main replay offers a score at each rated position and makes no claim about a decision at an unscored position. Only one confirmed self-harm case has a gap before its first observed endorsement, exactly one turn earlier. Sixteen other paths could contain self-harm endorsement only in a gap despite having none observed. Bound comparisons using the same possible missing labels for every policy; retain the complete-case analysis as a check. The previous recall estimates are not results for this cohort.

For policy development, the [separate MindGuard training release](https://huggingface.co/datasets/swordhealth/MindGuard-trainingset) offers 644 synthetic conversations, 6,115 labeled user turns and 339 scenarios. It can supply more varied queues than repeating nine self-harm paths. Its judge saw the full conversation; these are development labels, not clinician evidence for when a signal was visible live. Use only the prefix available at each decision. Split by conversation family before resampling, and keep copies on the same side of each split. Freeze policies before scoring held-out families. The expert set has already informed this design; it is an internal benchmark, not untouched external validation.

**Customer support.** [Liu et al., *Time to Transfer*](https://github.com/LauJames/MHCH-DAMI) provide 7,500 real e-commerce bot conversations annotated for turns appropriate for human handoff. Apply the same access metric from the first annotated handoff opportunity, with a support-specific window. The public release provides word IDs and turn features, not reusable raw text or reviewer timestamps: use a predictor compatible with those features. The [ServiceNow incident log](https://doi.org/10.24432/C57S4H) separately records opening times, updates and priorities for 24,918 incidents. Use it to check whether high-priority arrivals cluster in busy periods. It neither links to the chats nor measures active review duration. This is a second test of the allocation method, not a transfer of clinical labels or deadlines into support.

## Replaying steady demand and surges

Resample **whole conversations**, including ordinary ones. Keep their turn order, scores and endorsements together. Assign them to a clock drawn from real arrivals: [BurstGPT](https://arxiv.org/abs/2401.17644) supplies AI request times; [Open Up](https://formative.jmir.org/2024/1/e46823/) supplies crisis-service visitor interarrivals. An AI request is not a conversation or a safety alert. State how many requests are sampled as monitored conversation starts. Timing is observed; the assignment of review need to those times is simulated.

| Comparison | What changes | What stays fixed |
|---|---|---|
| Steady vs bursty arrivals | Evenly space the same starts, then restore their observed times | Number of conversations, paths, horizon and service requirements |
| Steady vs bursty need | Move more need-positive paths into selected windows, swapping whole paths | Exact arrival times, path counts and overall prevalence |
| Combined surge | Concentrate need in the busy windows of the bursty clock | Same underlying case set as the steady control |
| Higher sustained load | Increase monitored arrivals per hour | Case mix, service-time distribution and deadline; do not speed up the reviewers |

Start with one, two and three reviewers **simultaneously on duty**, plus an immediate-service control. For comparability, the mental-health pilot reference is ten minutes to *start*, twenty minutes of service, and two minutes between user turns. These are scenario values, not measured clinical requirements. Vary service time, including variation across tickets, and vary the deadline. Use identical service draws for the same ticket across dispatch policies. Separately compare flat coverage with more shift overlap during *forecast* peaks at equal scheduled staff-hours; record actual busy minutes too. A retrospective peak label may describe where misses occurred, but it cannot guide the simulated live dispatcher.

## What better dispatch could recover

[Adusumilli, Geiecke and Schilter](https://arxiv.org/html/1904.01047v5) frame allocation as a trade-off between acting now and preserving capacity for another arrival. They estimate gains from treatment using outcome data. Here capacity returns when service finishes, tickets can wait, and the measurable reward is a timely supported review. We borrow the allocation principle, not a causal estimate of benefit or a guarantee about RL. [Lykouris and Weng](https://arxiv.org/abs/2402.12237) already distinguish admission and scheduling in congested human–AI systems. Our focus is what happens between the first evidence of need in a conversation and the start of human review.

Consider one free reviewer and two tickets. A has a 95% chance of warranting review and 15 minutes left; B has a 60% chance and two minutes left. Each review occupies ten minutes. Taking A first gives 0.95 expected timely warranted reviews: B expires. Taking B first gives **1.55**, because A can still start ten minutes later. The example assumes known time windows to isolate the decision. In the replay, true onset is hidden. A practical comparator should estimate **what a ticket loses by waiting one more service interval**, not simply sort by its risk score.

Before fitting RL, compute three quantities on each identical stream at a fixed alert rule and deadline:

- **U:** recall if every ticket starts immediately. It measures what the detector makes possible.
- **O(c):** best feasible recall with the specified staffing schedule *c*, allowing a hindsight scheduler to know labels, arrivals and service times. It must use the same tickets, release times, service durations and expiry rules; its allowed schedules must include every live policy's schedule.
- **Rπ(c):** recall achieved by a policy π using only information available at dispatch.

For the fixed-prefix endpoint, a delay cannot turn an unsupported or already-late ticket into a catch. On the same positive-case denominator, the missed fraction therefore separates as:

$$
1-R_\pi(c) = (1-U) + \bigl(U-O(c)\bigr) + \bigl(O(c)-R_\pi(c)\bigr).
$$

The first term is missed or mistimed detection. The second is loss **no ordering could avoid** with those reviewers and tickets. The third is the most that better scheduling *and better information together* could recover. It is not a promised RL gain. With enough free reviewers, all dispatch rules start the same tickets immediately. Conversely, an overloaded queue can have many expirations but few meaningful choices if no reviewer becomes free before the deadlines.

## Proposed pilot and criteria for comparison

After review, first reproduce the all-67 observed-turn workload and calculate the immediate-service and hindsight bounds. Verify the no-contention tie and check the hindsight calculation on small queues with known optima before using it to interpret a policy difference. Then compare first-in first-out, highest calibrated risk, earliest estimated deadline and estimated loss from waiting. Pilot RL as a **dispatch-only** policy against the strongest development-selected simple rule. Give both the same information, tickets and staffing. Set alert sensitivity on development data; repeat the paired comparison at different sensitivities rather than tuning a threshold separately for RL on the evaluation cases.

The main display should show **timely recall against actual reviewer-minutes**, with staffing and demand regime labeled. Beside it, show the three sources of missed reviews. Report peak and off-peak recall, ticket expirations, unsupported starts, and how often a free reviewer actually had to choose between tickets. Keep comparisons paired by conversation assignment and arrival block. Report unique positive conversations and opening families alongside resampled volume; vary families and time blocks separately to distinguish uncertainty about cases from sensitivity to traffic. Repeating a role-play improves our understanding of queues, not the size of the clinical evidence base.

**Decision for review:** Does this define the right success criterion: more warranted conversations reaching a human *in time*, at comparable reviewer effort? If so, the first pilot tests where capacity actually forces a choice. A finding that ordinary dispatch suffices, that shift coverage matters more, or that the detector missed the case before it could enter the queue would each answer the paper's question.
