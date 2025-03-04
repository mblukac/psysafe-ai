<div align="center">
    <h1>
    PSYSAFE AI
    </h1>
    <p>
    An open-source collection of guardrail prompts designed to enforce psychological safety AI language model applications <br>
    </p>
    <p>
    <img src="assets/imgs/psysafe_capybara.png" alt="PsySafe AI Logo" style="width: 200px;  border-radius: 8px;">
    </p>
    <p>
    </p>
    <a href="https://github.com/mblukac/psysafe-ai"><img src="https://img.shields.io/badge/Python-3.8+-orange" alt="version"></a>
    <a href="https://github.com/mblukac/psysafe-ai"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

# Description
PsySafe AI is an open-source collection of guardrail prompts designed to enforce psychological safety in AI language model applications. Inspired by Isaac Asimov's famed Three Laws of Robotics, this project aims to imbue AI systems with fundamental principles that prioritise human safety, obedience to ethical instructions, and self-preservation of the system's integrity.

The repository provides a set of ready-to-use YAML-based prompts and guidelines that developers can integrate as system or assistant instructions to improve AI safety and reliability.Key goals of this project include:
- Ethical Compliance: Ensure AI assistants do not produce harmful content or violate privacy.
- User Well-being: Recognize user distress (e.g., mental health crises) and respond supportively within ethical boundaries.
- Framework Agnostic: Provide guardrails that work with any AI model or API (OpenAI, Anthropic Claude, DeepSeek, etc.) with minimal setup.

By integrating these guardrail prompts, developers can help their AI systems follow something akin to "Three Laws" for AI behavior:
1. Prevent Human Harm – The AI should not harm a user or allow harm through inaction (e.g., intervene if a user is in crisis).
2. Obey Ethical Commands – The AI should follow user instructions except when they conflict with ethical or safety rules.
3. Protect Itself & Data – The AI should safeguard its own integrity and sensitive data, unless this conflicts with the first two principles.