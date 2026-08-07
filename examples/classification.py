"""Make one structured observation and recalibrate it locally."""

from __future__ import annotations

import os

from psysafe import Conversation, DistressSupportClassifier, Sensitivity
from psysafe.backends import OpenAIBackend


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} before running this example.")
    return value


def main() -> None:
    backend = OpenAIBackend(model=required_environment("PSYSAFE_CLASSIFIER_MODEL"))
    classifier = DistressSupportClassifier(backend)
    conversation = Conversation.from_text(
        "I have felt overwhelmed lately and need the next steps explained clearly.",
    )

    # This is the only provider request. Sensitivity is intentionally absent.
    record = classifier.observe(conversation)

    for sensitivity in (
        Sensitivity.PRECISE,
        Sensitivity.BALANCED,
        Sensitivity.PRECAUTIONARY,
    ):
        assessment = classifier.calibrate(record, sensitivity=sensitivity)
        print(
            {
                "sensitivity": assessment.sensitivity.value,
                "outcome": assessment.outcome.value,
                "signals": assessment.signals,
                "evidence_directness": assessment.evidence_directness.value,
            },
        )


if __name__ == "__main__":
    main()
