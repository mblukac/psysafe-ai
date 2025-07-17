# AI Response Harm Detection Guardrail: A Scientific and Legal Framework

<p align="center">
by <strong>mblukac</strong><br/>
Version: <strong>1.0</strong> • July 17, 2025
</p>
<br>

<p align="center">
  <!-- Replace the image path below with one that exists in your repo -->
  <img src="../../assets/imgs/guardrails/ai_response_harm_guardrail.jpeg" alt="AI Response Harm Detection Guardrail" width="30%" />
</p>
<br>

---

## 1. Introduction: The Need for AI Response Evaluation

Most AI safety research focuses on **input moderation**—validating and sanitizing user prompts to prevent misuse. While essential, this approach is incomplete. It fails to address a critical vulnerability: an AI's *own generated output*. An AI can be manipulated through sophisticated adversarial attacks to produce harmful content, even in response to seemingly benign prompts.

This document outlines the scientific, legal, and technical framework for a different kind of safeguard: an **AI Response Guardrail**. This system does **not** evaluate the user's input; it evaluates the **AI's output before it is delivered to the user**. Its purpose is to detect and block responses that agree with or encourage self-harm and other pathological behaviors, ensuring the AI does not become an unintentional vector for harm. This is the first guardrail of its kind in the *psysafe* project, establishing a critical layer of defense focused on the model's own behavior.

> [!NOTE]
> **Scope:** Output-side safety layer. The guardrail inspects *model responses* just prior to user delivery. It can run inline, async, or batch depending on architecture.

---

## 2. The Scientific and Clinical Basis for Harm Detection

To build an effective detection model, we must first define **"harm"** based on established clinical and psychological best practices. The guardrail's logic is rooted in standards of care developed by mental health professionals for interacting with individuals in distress.

### 2.1 Responding to Self-Harm and Suicidal Ideation

An AI's response to a user expressing suicidal ideation must align with principles established by organizations such as the **National Alliance on Mental Illness (NAMI)** and the **National Institute of Mental Health (NIMH)**. The primary goals are to **de-escalate the immediate crisis** and **connect the user to professional human help**. [^nami] [^nimh]

#### Key Principles Encoded in Our Detection Rubric

- **Non-Judgmental Stance**  
  Responses must be calm, supportive, and empathetic. Shock, pity, criticism, or moralizing can induce shame and shut down communication. The guardrail flags outputs that judge, trivialize, or dismiss user pain. [^nami]

- **Direct & Unambiguous Communication**  
  Asking directly about suicide *does not* increase risk; it signals seriousness and care. Safe responses include questions like: *"Are you thinking about killing yourself?"* to assess risk. [^nimh]

- **Effective Hand-off to Professional Help**  
  An AI is not a therapist. Its role is a **bridge** to durable, human-centered support systems. A response is unsafe if it engages in prolonged therapeutic conversation *without* promptly providing crisis resources (e.g., the **988 Suicide & Crisis Lifeline** in the U.S.). [^nimh] [^lifeline]

- **Safety Planning & Means Reduction**  
  While an AI cannot physically intervene, it can guide the user toward creating a safety plan and emphasize reducing access to lethal means. **Any response that provides instructions on methods of self-harm is a critical failure.** [^nimh]

> [!IMPORTANT]
> Always offer crisis support resources appropriate to the user's region when self-harm intent is suspected. Where geolocation is unknown, provide globally recognized emergency guidance and a prompt to seek local emergency services.

---

### 2.2 Encouragement of Pathological Behavior

The guardrail also detects **encouragement of other self-destructive patterns**, such as eating disorders or substance abuse. The core principle: **refuse to reinforce or operationalize harmful behaviors**.

#### Tone & Demand Avoidance

Research into conditions such as **Pathological Demand Avoidance (PDA)** shows that authoritative commands often backfire, escalating resistance. Instead of *"You must stop starving yourself,"* a collaborative approach—*"Some people find it helpful to talk to a support line. Would you be open to that?"*—is more effective. Tone and framing are evaluated to ensure empowerment over coercion. [^pda]

#### Harmful Community Lexicons

Online communities built around pathology (e.g., *"pro-ana"* for anorexia) evolve coded languages. In attempting to be "helpful," an AI might mirror these terms—thereby validating and normalizing harmful practices. The guardrail flags usage of such in-group, harm-reinforcing lexicons. [^kolla] [^ghosh2024]

#### Subtle Linguistic & Symbolic Cues

Self-harm or disordered behavior intent may surface in **pronoun shifts, emotional tone patterns, metaphor clusters, numeric coding, or emoji combinations**. Detection must extend beyond literal phrasing to context, pattern, and symbol use. [^ghosh2025]

---

## 3. The Legal and Ethical Imperative

Developing a robust AI response guardrail is not just best practice—it is a **legal and ethical necessity**. A global regulatory trend is emerging that holds AI providers accountable for harms arising from model-generated content.

### 3.1 Provider Liability (United States)

Section 230 of the Communications Decency Act protects platforms from liability for *third-party* content—but **does not shield content generated by the platform's own AI**. This opens the door to negligence and product liability claims where providers fail to implement reasonable safeguards against foreseeable harms (e.g., self-harm encouragement). Providers aware that their AI can be manipulated into producing harmful advice, yet failing to mitigate the risk, could face litigation under traditional tort standards.

### 3.2 The EU AI Act (High-Risk Classification & Penalties)

The **EU AI Act** establishes a risk-tiered compliance regime. AI systems that interact with vulnerable populations or impact health are often classified as **High-Risk**, triggering obligations for **risk assessments, data governance, logging, transparency, and human oversight**. Non-compliance can result in penalties of **up to €35 million or 7% of global annual turnover**, whichever is higher (subject to final legislative calibration). Extraterritorial reach applies to providers serving EU users. [^euai]

### 3.3 EU Product Liability Directive (Software & AI)

Updates to the EU Product Liability framework explicitly cover software and AI systems. It establishes **strict liability**—damage caused by a defective product need not prove negligence. A **rebuttable presumption of defectiveness** may apply in complex AI systems when a claimant cannot reasonably access technical evidence, shifting the burden toward providers. [^eupld]

### 3.4 Industry Standard of Care

Leading AI organizations—including **OpenAI**, **Google**, and **Anthropic**—have published safety charters, policies, and technical mitigations focused on risk assessment, red teaming, and multi-layered safety controls. The convergence of these approaches is increasingly cited as a **de facto industry "standard of care"**. Falling materially below such norms may be introduced as evidence in negligence or product-defect litigation. [^openai] [^googleai] [^anthropic]

---

## 4. Technical Methodology: LLM-as-a-Judge

Given the scale of AI interactions, **manual human review of every output is infeasible**. Instead, we employ the **LLM-as-a-Judge** paradigm—now widely used in evaluation pipelines—to automatically review model outputs for safety and policy compliance.

### 4.1 How It Works

A dedicated **"judge" model** receives:

1. The **candidate AI response** (about to be shown to a user).
2. A structured **evaluation rubric** (policy-aligned, safety-specific).
3. Optional **context metadata** (conversation history, model ID, region, user risk tier).

The judge applies the rubric and returns a structured label. In this framework we use a **direct categorical scoring** scheme:

- `SAFE`
- `BORDERLINE`
- `HARMFUL`

This simplifies downstream handling logic (pass-through, soft intercept, hard block / escalation).

### 4.2 Mitigating Judge Vulnerabilities

LLM judges are not infallible; they inherit biases and failure modes. A defensible detection system must explicitly mitigate these.

#### Apology Bias

LLM judges may rate responses as “safer” when they begin with an apology (e.g., *"I'm sorry, but..."*) even if the content that follows is unsafe. Reported skews can be extreme in preference modeling studies. Our rubric instructs the judge to **ignore surface politeness markers** and evaluate substantive content. [^judgebias]

#### Data Drift

Classifiers trained solely on *human-written* content often degrade when scoring *machine-generated* language. Continuous dataset curation must include both human- and model-generated examples of harmful content to maintain calibration. [^llmasajudge_survey]

#### Lack of True Understanding

Judges may rely on shallow correlations rather than conceptual safety reasoning. We increase robustness by decomposing evaluation into **concrete, factual sub-questions** (e.g., “Does this response provide instructions for self-harm?”) instead of broad prompts like “Is this safe?” [^llmasajudge_survey]

#### Programmatic Judging (PAJAMA & Related Approaches)

Emerging work generates **explicit judging code** (e.g., Python or JSON-path checks) from policy specs, producing more transparent, auditable, and cost-efficient evaluation cascades. The **PAJAMA** framework and related programmatic judging research represent promising upgrade paths for future guardrail versions. [^pajama]

---

## 5. References

[^nami]: National Alliance on Mental Illness (NAMI). (2023). *How to Respond to Self-Harm.*  
[^nimh]: National Institute of Mental Health (NIMH). Various publications including *5 Action Steps to Help Someone Having Thoughts of Suicide* (2024) and *Ask Suicide-Screening Questions (ASQ) Toolkit* (2022).  
[^lifeline]: **988 Suicide & Crisis Lifeline** (U.S.): <https://988lifeline.org/>  
[^pda]: National Autistic Society. (2024). *Demand Avoidance.*  
[^kolla]: Kolla, S., et al. (2024). *Safe Spaces or Toxic Places? Content Moderation and Social Dynamics of Online Eating Disorder Communities.* arXiv.  
[^ghosh2024]: Ghosh, S., et al. (2024). *Linguistic Markers of Suicidal Ideation and Support in Online Communities.* arXiv.  
[^ghosh2025]: Ghosh, S., et al. (2025). *Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Differentiation and Emoji Interpretation.* arXiv.  
[^euai]: European Commission. (2024). *Regulatory Framework Proposal on Artificial Intelligence;* artificialintelligenceact.eu (2024) *Article 99: Penalties.*  
[^eupld]: European Parliament. (2023). *EU AI Liability Rules.* EPRS; Dentons. (2025). *Challenges in Establishing Liability for AI-Driven Products.*  
[^openai]: OpenAI. (2024). *OpenAI Safety Practices;* OpenAI Charter (2018).  
[^googleai]: Google. (2023). *Google AI Principles;* Google AI Principles Progress Update.  
[^anthropic]: Anthropic. (n.d.; 2024). *Anthropic Research Principles; Anthropic Safety and Security Info Sheet.*  
[^judgebias]: Observed in preference and safety-judging research; see Li et al. (2024) *LLM-as-a-Judge: A Systematic Survey* and Liusie et al. (2025) *On the Reliability of LLM-as-a-Judge for Safety Evaluation.*  
[^llmasajudge_survey]: Li, M., et al. (2024). *LLM-as-a-Judge: A Systematic Survey.* arXiv.  
[^pajama]: Jiang, Z., et al. (2024). *PAJAMA: Programmatic and Modular LLM-Based Evaluation.* arXiv.
