# Suicide Intent Classification in Therapy Chat Interactions  

by the **PsySafe AI Research Team**  

---

## Executive Summary  

Suicide claims more than 700 000 lives every year, yet most people who die by suicide signal their distress long before the final act.[^1]  
These signals are woven into everyday language: abrupt farewells, absolutist phrases that leave no room for hope, quiet admissions of unbearable pain.  
An effective therapy-chatbot must therefore be attuned not only to stark declarations—*“I’m going to kill myself”*—but also to whispered hints and coded metaphors.

This report distils the academic consensus on the **linguistic signature of suicidal intent** and shows how those insights translate into a fine-grained, tunable classifier.  
The aim is to underpin PsySafe AI’s next guardrail: a suicide-intent detector whose *sensitivity* can be calibrated to the demands of any deployment, from a gently supportive self-help bot to a high-stakes crisis-chat service.  

---

## Linguistic Markers of Suicidal Intent  

### Direct Declarations  
The clearest—and thankfully the rarest—indicator is an explicit wish or plan to die.  
Phrases such as *“I want to die,”* *“I’m going to end it,”* or method-specific statements (“I’ve counted out the pills”) leave little ambiguity.  
Any classifier, regardless of sensitivity setting, must treat such utterances as **imminent, high-priority risk**.

### Indirect Allusions  
Many individuals couch the same intention in softer terms, for example:  

* “I just want to sleep forever.”  
* “You won’t have to worry about me much longer.”  

Interpreting these lines demands contextual awareness—the difference between hyperbole and hidden intent often lies in what preceded the remark and how the speaker’s tone has evolved.

### Hopelessness & Absolutist Thinking  
Across dozens of longitudinal studies, **hopelessness** stands out as a robust predictor of suicidal behaviour.[^2]  
Linguistically, hopelessness surfaces in absolutist terms: *always, never, nothing, completely.*  
A large corpus analysis showed suicidal-ideation forums using significantly more of these words than either depression or anxiety forums.[^3]

### Burdensomeness & Social Withdrawal  
According to the **Interpersonal Theory of Suicide**, a lethal desire blossoms when two cognitions converge:[^4]

1. *“I am a burden.”*  
2. *“I don’t belong here.”*

Sentences like *“Everyone would be better off without me”* or *“No one would notice if I disappeared”* reveal these cognitions and reliably raise risk scores.

### Preoccupation with Death & Farewells  
Repetitive musings on death (*“What happens after we die?”*) or sudden farewell messages (*“Thank you for everything—goodbye.”*) often appear in last-month timelines of those who later attempt suicide.[^5]

### Planning Language  
Mentions of means, timing, or preparations (writing a will, deleting social-media accounts) indicate escalating intent.  
Crisis Text Line’s triage algorithm, trained on over 200 million messages, prioritises texters describing such plans and flags **86 %** of people at severe imminent risk within their opening texts.[^6]

| **Cue Category**   | **Illustrative Phrases**              | **Indicative Meaning**   |
|--------------------|---------------------------------------|--------------------------|
| Direct ideation    | “I want to die tonight.”              | Overt intent             |
| Indirect allusion  | “I wish I could disappear.”           | Veiled desire            |
| Hopelessness       | “Nothing will ever change.”           | No future orientation    |
| Absolutist wording | “Always alone, never good enough.”    | Black-and-white thought  |
| Burdensomeness     | “I’m just a problem for everyone.”    | Perceived burden         |
| Social withdrawal  | “Nobody cares if I’m here.”           | Thwarted belonging       |
| Death focus        | “If I weren’t alive tomorrow …”       | Rumination on mortality  |
| Farewell signals   | “Promise me you’ll remember me.”      | Possible final goodbye   |
| Planning details   | “I counted the pills—just enough.”    | Means and immediacy      |

---

## Ambiguity & Uncertainty in Suicidal Language  

Human speech is rarely binary.  
One user’s *“I’m done”* may herald a suicide attempt, while another’s signals simple exasperation.  
**Uncertainty is therefore intrinsic** to risk detection.  

Research shows that large language models (LLMs) mirror this ambiguity: ChatGPT-3.5 systematically underrated high-risk vignettes, whereas GPT-4 tended to over-estimate severity, aligning more closely with clinician judgments but still not perfectly.[^7]

For a safety-critical chatbot, the aim is *not* to seek impossible certainty; it is to surface the model’s confidence and allow designers to set the trip-wire.  
A probability-bearing output—*“0.82 likelihood of suicidal intent”*—lends itself to clear governance:  

* **Below threshold** → empathic inquiries.  
* **Above threshold** → escalation or human review.  

**Contextual memory** further refines the signal: the same sentence uttered after ten minutes of hopeless rumination deserves a sharper reading than if voiced in a light-hearted rant.

---

## Fine-Grained Risk Detection & Adjustable Sensitivity  

PsySafe AI’s classifier is conceived as a **multi-level detector** rather than a blunt labeler.  
Internally it scores each message on constituent dimensions—hopelessness, burdensomeness, death preoccupation, planning cues.  
These scores accumulate into a continuous risk index that updates with every user turn.

Developers (or clinicians) can **tune the sensitivity knob**:

| Sensitivity | Threshold Logic (illustrative)                                        | Typical Action                               |
|-------------|-----------------------------------------------------------------------|----------------------------------------------|
| **Low**     | Alert only if *explicit* ideation or planning statements exceed 0.9.  | Silent logging or gentle check-in            |
| **Medium**  | Trigger when any *two* moderate-risk cues combine to reach 0.6.       | Offer coping strategies; assess safety       |
| **High**    | Flag on *any* single cue above 0.3, including vague despair.          | Prioritise conversation; escalate if needed  |

Such calibration reconciles two competing truths: **missing a genuine plea is unacceptable**, yet inundating users with crisis scripts erodes trust.  
Pilot deployments that started with high sensitivity then ratcheted down as false positives stabilised illustrate the pragmatic path.[^8]

---

## Conclusion  

Language is the earliest seismograph of suicidal crisis.  
By focusing on linguistic features that research has consistently tied to self-harm—overt statements, hopeless or absolutist cognition, burdensomeness, withdrawals, morbid farewells, and evidence of planning—an LLM-based classifier can listen with clinically informed ears.  

Exposing its confidence, retaining conversational context, and allowing explicit threshold control turns that classifier into a **flexible guardrail**: gentle where it may, unflinching where it must.  
In doing so, therapy chatbots built on PsySafe AI can ensure that no cry—however muted—passes unnoticed.

---

### Key References & Literature  

[^1]: World Health Organization. *Preventing Suicide: A Global Imperative.* WHO Press, 2014.  
[^2]: Beck, A. T. et al. “Relationship Between Hopelessness and Ultimate Suicide.” *American Journal of Psychiatry* 147, no. 2 (1990): 190-195.  
[^3]: Al-Mosaiwi, M., & Johnstone, T. “In an Absolute State: Elevated Use of Absolutist Words Is a Marker Specific to Anxiety, Depression, and Suicidal Ideation.” *Clinical Psychological Science* 6, no. 4 (2018): 529-545.  
[^4]: Joiner, T. *Why People Die by Suicide.* Harvard University Press, 2005.  
[^5]: O’Connor, R. C., & Nock, M. K. “The Psychology of Suicidal Behaviour.” *The Lancet Psychiatry* 1, no. 1 (2014): 73-85.  
[^6]: Crisis Text Line. “Detecting Crisis: An AI Solution.” Blog post, 2022.  
[^7]: Levi-Belz, Y., Elyoseph, Z., & Gamliel, I. “Suicide Risk Assessments Through the Eyes of ChatGPT-3.5 Versus ChatGPT-4.” *JMIR Mental Health* 10 (2023): e51232.  
[^8]: Fernandes, A. C. et al. “Identifying Suicide Ideation and Suicidal Attempts in a Psychiatric Research Database Using Natural Language Processing.” *Scientific Reports* 8 (2018): 7426.  
