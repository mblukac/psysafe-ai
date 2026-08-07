import pytest
from pydantic import ValidationError

from psysafe.core.contracts import (
    MAX_CONVERSATION_CONTENT_CHARS,
    MAX_CONVERSATION_MESSAGES,
    MAX_MESSAGE_CONTENT_CHARS,
    Assessment,
    AssessmentMetadata,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    Message,
    MessageRole,
    Outcome,
    Sensitivity,
)


def test_message_validates_role_and_content() -> None:
    message = Message(id="turn:1", role="user", content="Keep my whitespace. ")

    assert message.id == "turn:1"
    assert message.role is MessageRole.USER
    assert message.content == "Keep my whitespace. "

    with pytest.raises(ValidationError):
        Message(role="observer", content="hello")
    with pytest.raises(ValidationError):
        Message(role="user", content="  \n")
    with pytest.raises(ValidationError):
        Message(role="user", content="x" * (MAX_MESSAGE_CONTENT_CHARS + 1))
    with pytest.raises(ValidationError):
        Message(id="raw text is not an identifier", role="user", content="hello")
    with pytest.raises(ValidationError):
        Message(role="user", content="hello", raw_response="provider output")


def test_conversation_requires_messages_and_enforces_both_size_limits() -> None:
    conversation = Conversation.from_text("hello")
    assert conversation.messages == (Message(role=MessageRole.USER, content="hello"),)

    with pytest.raises(ValidationError):
        Conversation(messages=[])
    with pytest.raises(ValidationError):
        Conversation(messages=[Message(role="user", content="hello")], api_key="secret")
    with pytest.raises(ValidationError):
        Conversation(
            messages=[Message(role="user", content="x")] * (MAX_CONVERSATION_MESSAGES + 1),
        )

    chunk = "x" * MAX_MESSAGE_CONTENT_CHARS
    too_large = MAX_CONVERSATION_CONTENT_CHARS // MAX_MESSAGE_CONTENT_CHARS + 1
    with pytest.raises(ValidationError):
        Conversation(messages=[Message(role="user", content=chunk)] * too_large)
    with pytest.raises(ValidationError):
        Conversation(
            messages=[
                Message(id="duplicate", role="user", content="first"),
                Message(id="duplicate", role="assistant", content="second"),
            ],
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("precise", Sensitivity.PRECISE),
        ("balanced", Sensitivity.BALANCED),
        ("precautionary", Sensitivity.PRECAUTIONARY),
        ("low", Sensitivity.PRECISE),
        ("medium", Sensitivity.BALANCED),
        ("high", Sensitivity.PRECAUTIONARY),
    ],
)
def test_sensitivity_accepts_legacy_aliases(value: str, expected: Sensitivity) -> None:
    assert Sensitivity(value) is expected

    assessment = Assessment(
        classifier_id="example",
        policy_version="1.0.0",
        sensitivity=value,
        outcome="not_matched",
    )
    assert assessment.sensitivity is expected
    assert assessment.model_dump(mode="json")["sensitivity"] == expected.value


def test_assessment_is_categorical_and_contains_only_safe_provenance() -> None:
    assessment = Assessment(
        classifier_id="suicidal_language",
        policy_version="2026.08.1",
        sensitivity=Sensitivity.PRECAUTIONARY,
        outcome=Outcome.MATCHED,
        evidence_directness=EvidenceDirectness.EXPLICIT,
        signals=("direct_self_harm_statement",),
        metadata=AssessmentMetadata(provider="openai", model="gpt-example"),
    )

    serialized = assessment.model_dump(mode="json")
    assert serialized == {
        "classifier_id": "suicidal_language",
        "policy_version": "2026.08.1",
        "sensitivity": "precautionary",
        "outcome": "matched",
        "evidence_directness": "explicit",
        "signals": ["direct_self_harm_statement"],
        "indeterminate_reason": None,
        "metadata": {"provider": "openai", "model": "gpt-example"},
    }
    forbidden_names = {
        "api_key",
        "confidence",
        "confidence_score",
        "raw_content",
        "raw_response",
        "risk_score",
    }
    assert forbidden_names.isdisjoint(Assessment.model_fields)

    with pytest.raises(ValidationError):
        AssessmentMetadata(provider="openai", api_key="sk-secret")
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="1.0.0",
            outcome="matched",
            raw_response="secret model output",
        )


def test_assessment_requires_a_bounded_policy_version() -> None:
    with pytest.raises(ValidationError):
        Assessment(classifier_id="example", outcome="not_matched")
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="version with spaces",
            outcome="not_matched",
        )


def test_assessment_outcome_is_consistent_with_asserted_evidence() -> None:
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="1.0.0",
            outcome="matched",
        )
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="1.0.0",
            outcome="not_matched",
            evidence_directness="ambiguous",
            signals=("possible_signal",),
        )


def test_indeterminate_assessment_is_an_explicit_non_decision() -> None:
    assessment = Assessment.indeterminate(
        classifier_id="example",
        policy_version="1.0.0",
        sensitivity=Sensitivity.BALANCED,
        reason=IndeterminateReason.INVALID_RESPONSE,
    )

    assert assessment.outcome is Outcome.INDETERMINATE
    assert assessment.indeterminate_reason is IndeterminateReason.INVALID_RESPONSE
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="1.0.0",
            outcome=Outcome.INDETERMINATE,
        )
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="1.0.0",
            outcome=Outcome.NOT_MATCHED,
            indeterminate_reason=IndeterminateReason.PROVIDER_ERROR,
        )


def test_indeterminate_reasons_cover_provider_failure_modes() -> None:
    assert {
        IndeterminateReason.REFUSED,
        IndeterminateReason.INVALID_RESPONSE,
        IndeterminateReason.PROVIDER_ERROR,
        IndeterminateReason.TIMEOUT,
    }.issubset(set(IndeterminateReason))


def test_signal_labels_cannot_smuggle_raw_content() -> None:
    with pytest.raises(ValidationError):
        Assessment(
            classifier_id="example",
            policy_version="1.0.0",
            outcome="matched",
            evidence_directness="explicit",
            signals=("User said: this is raw content",),
        )
