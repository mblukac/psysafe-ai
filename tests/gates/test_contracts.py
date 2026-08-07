from __future__ import annotations

import pytest
from pydantic import ValidationError

from psysafe.core.contracts import EvidenceDirectness, IndeterminateReason, Outcome, Sensitivity
from psysafe.gates import (
    Checkpoint,
    ClassifierPolicyOverride,
    GateAction,
    GateAssessment,
    GateDecision,
    GatePolicy,
)


def _matched_gate_assessment(*, review_signals: tuple[str, ...] = ()) -> GateAssessment:
    return GateAssessment(
        classifier_id="assistant_harm",
        policy_version="2026.08.1",
        sensitivity=Sensitivity.BALANCED,
        outcome=Outcome.MATCHED,
        evidence_directness=EvidenceDirectness.EXPLICIT,
        signals=("unsafe_instruction",),
        review_signals=review_signals,
    )


def test_checkpoints_cover_sequential_and_tool_execution_boundaries() -> None:
    assert tuple(checkpoint.value for checkpoint in Checkpoint) == (
        "input",
        "task_selection",
        "execution",
        "tool_input",
        "tool_output",
        "communication",
    )


def test_gate_assessment_is_routing_only_and_strict() -> None:
    assessment = _matched_gate_assessment(review_signals=("human_review",))

    assert assessment.model_dump() == {
        "classifier_id": "assistant_harm",
        "policy_version": "2026.08.1",
        "sensitivity": Sensitivity.BALANCED,
        "outcome": Outcome.MATCHED,
        "evidence_directness": EvidenceDirectness.EXPLICIT,
        "signals": ("unsafe_instruction",),
        "indeterminate_reason": None,
        "review_signals": ("human_review",),
    }
    with pytest.raises(ValidationError):
        GateAssessment.model_validate({**assessment.model_dump(), "raw_input": "private"})
    with pytest.raises(ValidationError):
        assessment.model_copy(update={"classifier_id": "changed"}, deep=True).model_validate(
            {**assessment.model_dump(), "review_signals": ("not a safe label",)},
        )


def test_gate_decision_rejects_fail_open_states() -> None:
    with pytest.raises(ValidationError, match="review signals cannot allow"):
        GateDecision(
            checkpoint=Checkpoint.COMMUNICATION,
            artifact_id="communication:v1",
            target_message_index=0,
            action=GateAction.ALLOW,
            assessments=(_matched_gate_assessment(review_signals=("human_review",)),),
        )

    indeterminate = GateAssessment(
        classifier_id="distress",
        policy_version="2026.08.1",
        outcome=Outcome.INDETERMINATE,
        indeterminate_reason=IndeterminateReason.TIMEOUT,
    )
    with pytest.raises(ValidationError, match="indeterminate assessment cannot allow"):
        GateDecision(
            checkpoint=Checkpoint.INPUT,
            artifact_id="input:v1",
            target_message_index=0,
            action=GateAction.ALLOW,
            assessments=(indeterminate,),
        )

    with pytest.raises(ValidationError, match="unbound gate decision cannot allow"):
        GateDecision(
            checkpoint=Checkpoint.INPUT,
            artifact_id="input:v1",
            target_message_index=None,
            action=GateAction.ALLOW,
            assessments=(
                GateAssessment(
                    classifier_id="distress",
                    policy_version="2026.08.1",
                    outcome=Outcome.NOT_MATCHED,
                ),
            ),
        )


def test_gate_decision_requires_unique_classifier_ids() -> None:
    assessment = _matched_gate_assessment()
    with pytest.raises(ValidationError, match="unique classifier IDs"):
        GateDecision(
            checkpoint=Checkpoint.INPUT,
            artifact_id="input:v1",
            target_message_index=0,
            action=GateAction.REVIEW,
            assessments=(assessment, assessment),
        )


def test_gate_decision_requires_safe_artifact_correlation_fields() -> None:
    payload = {
        "checkpoint": Checkpoint.INPUT,
        "target_message_index": 0,
        "action": GateAction.REVIEW,
        "assessments": (_matched_gate_assessment(),),
    }

    with pytest.raises(ValidationError):
        GateDecision.model_validate(payload)
    with pytest.raises(ValidationError):
        GateDecision.model_validate({**payload, "artifact_id": "raw artifact text is not opaque"})
    with pytest.raises(ValidationError):
        GateDecision.model_validate(
            {**payload, "artifact_id": "input:v1", "target_message_index": 128},
        )


def test_policy_requires_explicit_match_action_and_fail_safe_fallbacks() -> None:
    with pytest.raises(ValidationError):
        GatePolicy.model_validate({})
    with pytest.raises(ValidationError, match="cannot allow"):
        GatePolicy(matched_action=GateAction.REVIEW, indeterminate_action=GateAction.ALLOW)
    with pytest.raises(ValidationError, match="cannot allow"):
        GatePolicy(matched_action=GateAction.REVIEW, review_signal_action=GateAction.ALLOW)

    policy = GatePolicy(matched_action=GateAction.BLOCK)
    assert policy.action_for(Outcome.MATCHED) is GateAction.BLOCK
    assert policy.action_for(Outcome.NOT_MATCHED) is GateAction.ALLOW
    assert policy.action_for(Outcome.INDETERMINATE) is GateAction.REVIEW


def test_classifier_overrides_are_bounded_unique_and_fail_safe() -> None:
    override = ClassifierPolicyOverride(
        classifier_id="distress",
        matched_action=GateAction.REVIEW,
        indeterminate_action=GateAction.BLOCK,
    )
    policy = GatePolicy(
        matched_action=GateAction.BLOCK,
        overrides=(override,),
    )

    assert policy.action_for(Outcome.MATCHED, classifier_id="distress") is GateAction.REVIEW
    assert policy.action_for(Outcome.MATCHED, classifier_id="assistant_harm") is GateAction.BLOCK
    assert policy.action_for(Outcome.INDETERMINATE, classifier_id="distress") is GateAction.BLOCK
    assert policy.review_action_for("distress") is GateAction.REVIEW
    with pytest.raises(ValidationError, match="at least one"):
        ClassifierPolicyOverride(classifier_id="distress")
    with pytest.raises(ValidationError, match="cannot allow"):
        ClassifierPolicyOverride(
            classifier_id="distress",
            review_signal_action=GateAction.ALLOW,
        )
    with pytest.raises(ValidationError, match="unique classifier IDs"):
        GatePolicy(matched_action=GateAction.BLOCK, overrides=(override, override))
