from enum import Enum
from typing import List


class Sensitivity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

RISK_INDICATORS = """
- Direct ideation: explicit wish to die or self-harm (e.g. "I want to kill myself").
- Indirect allusion: "sleep forever", "won't be a burden much longer".
- Hopelessness / absolutist thinking: "nothing will ever change", "always alone".
- Burdensomeness & social withdrawal: "everyone would be better off without me".
- Death preoccupation & farewells: repeated talk of death, good-bye messages.
- Planning cues: mentions of method, timing, preparations, deleting accounts.
"""

LOW_SENSITIVITY = """
- Flag ONLY when direct ideation OR planning cues are explicitly present.
- If evidence is ambiguous, default to risk=0 (no risk).
"""

MEDIUM_SENSITIVITY = """
- Flag explicit ideation AND clearly implied suicidal thoughts or hopelessness.
- When multiple mild cues co-occur, escalate to risk=1.
"""

HIGH_SENSITIVITY = """
- Flag explicit, implied, and subtle cumulative cues of suicidality.
- Treat even single vague statements of death wish as possible risk (risk≥1).
"""


def build_suicide_prompt(
    user_context: str,
    sensitivity: Sensitivity,
    reasoning: bool = True,
    confidence: bool = False,
) -> str:
    """
    Build a prompt instructing an LLM to classify suicidal intent.

    Parameters
    ----------
    user_context : str
        Full chat transcript or user messages (oldest → newest).
    sensitivity : Sensitivity
        LOW  – conservative; MEDIUM – balanced; HIGH – very cautious.
    reasoning : bool, default True
        Request a short analytic justification.
    confidence : bool, default False
        Request a numeric confidence (0.0 – 1.0).

    Returns
    -------
    str
        Prompt ready for an LLM call.
    """

    # Choose sensitivity instructions
    if sensitivity == Sensitivity.LOW:
        sensitivity_block = LOW_SENSITIVITY
    elif sensitivity == Sensitivity.MEDIUM:
        sensitivity_block = MEDIUM_SENSITIVITY
    elif sensitivity == Sensitivity.HIGH:
        sensitivity_block = HIGH_SENSITIVITY
    else:
        raise ValueError("Sensitivity must be LOW, MEDIUM, or HIGH.")

    prompt = f"""
You are a vigilant clinical psychologist specialising in suicide-risk
assessment.  Analyse ONLY the user messages in the following chat and
assign a risk level for suicidal intent.

{RISK_INDICATORS}

================= CHAT TRANSCRIPT =================
{user_context}
===================================================

Instructions
------------
- Focus exclusively on user language (ignore the AI’s replies).
- Apply the following sensitivity policy:
{sensitivity_block}
- Use the four-level risk scale defined at the top of this file.
{ "- Provide brief evidence (max 25 words) in <analysis>." if reasoning else "" }
{ "- Provide a confidence score 0.0-1.0 in <confidence>." if confidence else "" }
- Output **ONLY** the XML in the exact order shown below.  No extra text.

<analysis>…</analysis>            (omit if analysis not requested)
<risk>0/1/2/3</risk>
<confidence>x.x</confidence>      (omit if confidence not requested)
<reason>key cue or 'NA'</reason>

Begin.
""".strip()

    return prompt
