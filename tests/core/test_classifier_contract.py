import asyncio

import pytest

from psysafe.core.classifier import (
    AsyncClassifier,
    Classifier,
    IndeterminateAssessmentError,
)
from psysafe.core.contracts import (
    Assessment,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    Outcome,
    Sensitivity,
)


class ExampleClassifier:
    classifier_id = "example"
    policy_version = "1.0.0"

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
        )


class ExampleAsyncClassifier:
    classifier_id = "example_async"
    policy_version = "1.0.0"

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=("example_signal",),
        )


def test_classifier_protocols_are_runtime_checkable() -> None:
    assert isinstance(ExampleClassifier(), Classifier)
    assert isinstance(ExampleAsyncClassifier(), AsyncClassifier)

    result = asyncio.run(ExampleAsyncClassifier().aclassify(Conversation.from_text("hello")))
    assert result.require_match_decision() is True


def test_indeterminate_cannot_be_coerced_to_a_negative_decision() -> None:
    result = Assessment.indeterminate(
        classifier_id="example",
        policy_version="1.0.0",
        sensitivity=Sensitivity.BALANCED,
        reason=IndeterminateReason.PROVIDER_ERROR,
    )

    assert result.outcome is Outcome.INDETERMINATE
    with pytest.raises(IndeterminateAssessmentError):
        result.require_match_decision()
