from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import cast

import pytest

from psysafe.backends.base import BackendConfigurationError, CallableBackend
from psysafe.classifiers.complaints import (
    ComplaintEscalation,
    ComplaintsClassifier,
    ComplaintsObservation,
    EscalationReason,
)
from psysafe.classifiers.context import EvidenceSubject, SourceContext
from psysafe.core.classifier import ClassificationError
from psysafe.core.contracts import (
    Assessment,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    Message,
    MessageRole,
    Outcome,
    Sensitivity,
)
from psysafe.gates import (
    MAX_GATE_CLASSIFIERS,
    AsyncWorkflowGate,
    Checkpoint,
    ClassifierPolicyOverride,
    GateAction,
    GateAssessment,
    GateDecision,
    GatePolicy,
    WorkflowGate,
)

_ARTIFACT_ID = "artifact:v1"


def _assessment(
    classifier_id: str,
    *,
    outcome: Outcome,
    sensitivity: Sensitivity = Sensitivity.BALANCED,
    reason: IndeterminateReason | None = None,
) -> Assessment:
    if outcome is Outcome.MATCHED:
        return Assessment(
            classifier_id=classifier_id,
            policy_version="2026.08.1",
            sensitivity=sensitivity,
            outcome=outcome,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=("policy_match",),
        )
    if outcome is Outcome.INDETERMINATE:
        return Assessment.indeterminate(
            classifier_id=classifier_id,
            policy_version="2026.08.1",
            sensitivity=sensitivity,
            reason=reason or IndeterminateReason.INTERNAL_ERROR,
        )
    return Assessment(
        classifier_id=classifier_id,
        policy_version="2026.08.1",
        sensitivity=sensitivity,
        outcome=outcome,
    )


class StaticClassifier:
    def __init__(
        self,
        assessment: Assessment,
        *,
        evidence_role: MessageRole | None = None,
        allowed_signals: tuple[str, ...] = ("policy_match",),
        allowed_review_signals: tuple[str, ...] = (),
    ) -> None:
        self._assessment = assessment
        self._evidence_role = evidence_role
        self._allowed_signals = allowed_signals
        self._allowed_review_signals = allowed_review_signals

    @property
    def classifier_id(self) -> str:
        return self._assessment.classifier_id

    @property
    def policy_version(self) -> str:
        return self._assessment.policy_version

    @property
    def evidence_role(self) -> MessageRole | None:
        return self._evidence_role

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return self._allowed_signals

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return self._allowed_review_signals

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        if sensitivity is not self._assessment.sensitivity:
            return self._assessment.model_copy(update={"sensitivity": sensitivity})
        return self._assessment

    def classify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del target_message_index
        return self.classify(conversation, sensitivity=sensitivity)


class CallableClassifier:
    def __init__(
        self,
        classifier_id: str,
        handler: Callable[[Conversation, int, Sensitivity], Assessment],
        *,
        evidence_role: MessageRole | None = None,
        allowed_signals: tuple[str, ...] = ("policy_match",),
        allowed_review_signals: tuple[str, ...] = (),
    ) -> None:
        self._classifier_id = classifier_id
        self._handler = handler
        self._evidence_role = evidence_role
        self._allowed_signals = allowed_signals
        self._allowed_review_signals = allowed_review_signals
        self.call_count = 0

    @property
    def classifier_id(self) -> str:
        return self._classifier_id

    @property
    def policy_version(self) -> str:
        return "2026.08.1"

    @property
    def evidence_role(self) -> MessageRole | None:
        return self._evidence_role

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return self._allowed_signals

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return self._allowed_review_signals

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return self.classify_target(
            conversation,
            target_message_index=len(conversation.messages) - 1,
            sensitivity=sensitivity,
        )

    def classify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        self.call_count += 1
        return self._handler(conversation, target_message_index, sensitivity)


class RaisingClassifier:
    def __init__(self, classifier_id: str, error_factory: Callable[[], Exception]) -> None:
        self._classifier_id = classifier_id
        self._error_factory = error_factory

    @property
    def classifier_id(self) -> str:
        return self._classifier_id

    @property
    def policy_version(self) -> str:
        return "2026.08.1"

    @property
    def evidence_role(self) -> MessageRole | None:
        return None

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return ("policy_match",)

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return ()

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return self.classify_target(
            conversation,
            target_message_index=len(conversation.messages) - 1,
            sensitivity=sensitivity,
        )

    def classify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, target_message_index, sensitivity
        raise self._error_factory()


class StaticAsyncClassifier:
    def __init__(
        self,
        assessment: Assessment,
        handler: Callable[[], Awaitable[None]] | None = None,
        *,
        evidence_role: MessageRole | None = None,
        allowed_signals: tuple[str, ...] = ("policy_match",),
        allowed_review_signals: tuple[str, ...] = (),
    ) -> None:
        self._assessment = assessment
        self._handler = handler
        self._evidence_role = evidence_role
        self._allowed_signals = allowed_signals
        self._allowed_review_signals = allowed_review_signals

    @property
    def classifier_id(self) -> str:
        return self._assessment.classifier_id

    @property
    def policy_version(self) -> str:
        return self._assessment.policy_version

    @property
    def evidence_role(self) -> MessageRole | None:
        return self._evidence_role

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return self._allowed_signals

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return self._allowed_review_signals

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return await self.aclassify_target(
            conversation,
            target_message_index=len(conversation.messages) - 1,
            sensitivity=sensitivity,
        )

    async def aclassify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, target_message_index
        if self._handler is not None:
            await self._handler()
        if sensitivity is not self._assessment.sensitivity:
            return self._assessment.model_copy(update={"sensitivity": sensitivity})
        return self._assessment


class RaisingAsyncClassifier:
    def __init__(self, classifier_id: str, error_factory: Callable[[], Exception]) -> None:
        self._classifier_id = classifier_id
        self._error_factory = error_factory

    @property
    def classifier_id(self) -> str:
        return self._classifier_id

    @property
    def policy_version(self) -> str:
        return "2026.08.1"

    @property
    def evidence_role(self) -> MessageRole | None:
        return None

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return ("policy_match",)

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return ()

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return await self.aclassify_target(
            conversation,
            target_message_index=len(conversation.messages) - 1,
            sensitivity=sensitivity,
        )

    async def aclassify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, target_message_index, sensitivity
        raise self._error_factory()


def _gate_traceback_locals(error: BaseException) -> str:
    locals_repr: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("psysafe/gates/workflow.py"):
            locals_repr.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(locals_repr)


def test_sync_gate_resolves_precedence_and_per_classifier_overrides() -> None:
    gate = WorkflowGate(
        (
            StaticClassifier(_assessment("distress", outcome=Outcome.MATCHED)),
            StaticClassifier(_assessment("assistant_harm", outcome=Outcome.MATCHED)),
            StaticClassifier(_assessment("pii", outcome=Outcome.NOT_MATCHED)),
        ),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(
            matched_action=GateAction.REVIEW,
            overrides=(
                ClassifierPolicyOverride(
                    classifier_id="assistant_harm",
                    matched_action=GateAction.BLOCK,
                ),
            ),
        ),
    )

    decision = gate.evaluate_text("A bounded stage target.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.BLOCK
    assert decision.classifier_ids == ("distress", "assistant_harm", "pii")
    assert tuple(assessment.outcome for assessment in decision.assessments) == (
        Outcome.MATCHED,
        Outcome.MATCHED,
        Outcome.NOT_MATCHED,
    )


def test_gate_rejects_policy_override_for_unconfigured_classifier() -> None:
    with pytest.raises(ValueError, match="configured classifier IDs"):
        WorkflowGate(
            (StaticClassifier(_assessment("distress", outcome=Outcome.NOT_MATCHED)),),
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(
                matched_action=GateAction.REVIEW,
                overrides=(
                    ClassifierPolicyOverride(
                        classifier_id="assistant_harm",
                        matched_action=GateAction.BLOCK,
                    ),
                ),
            ),
        )


def test_gate_requires_a_finite_classifier_signal_vocabulary() -> None:
    classifier = StaticClassifier(
        _assessment("distress", outcome=Outcome.NOT_MATCHED),
        allowed_signals=(),
    )

    with pytest.raises(ValueError, match="non-empty tuple"):
        WorkflowGate(
            (classifier,),
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )


def test_conversation_evaluation_binds_only_the_selected_role_checked_target() -> None:
    context_sizes: list[int] = []

    def classify_target(
        conversation: Conversation,
        target_message_index: int,
        sensitivity: Sensitivity,
    ) -> Assessment:
        del sensitivity
        context_sizes.append(len(conversation.messages))
        matched = "old-danger" in conversation.messages[target_message_index].content
        return _assessment("target_check", outcome=Outcome.MATCHED if matched else Outcome.NOT_MATCHED)

    classifier = CallableClassifier("target_check", classify_target)
    gate = WorkflowGate(
        (classifier,),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )
    conversation = Conversation(
        messages=(
            Message(id="private:old", role=MessageRole.USER, content="old-danger"),
            Message(id="private:reply", role=MessageRole.ASSISTANT, content="intermediate"),
            Message(id="private:new", role=MessageRole.USER, content="current-safe-target"),
        ),
    )

    latest = gate.evaluate(conversation, artifact_id="input:current:v1")
    explicitly_selected = gate.evaluate(
        conversation,
        artifact_id="input:old:v1",
        target_message_index=0,
    )

    assert latest.action is GateAction.ALLOW
    assert latest.artifact_id == "input:current:v1"
    assert latest.target_message_index == 2
    assert explicitly_selected.action is GateAction.BLOCK
    assert explicitly_selected.target_message_index == 0
    assert classifier.call_count == 2
    assert context_sizes == [3, 3]


def test_role_or_target_mismatch_is_indeterminate_without_classifier_execution() -> None:
    classifier = CallableClassifier(
        "target_check",
        lambda _conversation, _target, _sensitivity: _assessment(
            "target_check",
            outcome=Outcome.NOT_MATCHED,
        ),
    )
    gate = WorkflowGate(
        (classifier,),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )
    wrong_role = Conversation(
        messages=(Message(role=MessageRole.ASSISTANT, content="Not an input-stage target."),),
    )

    role_decision = gate.evaluate(wrong_role, artifact_id=_ARTIFACT_ID)
    index_decision = gate.evaluate(
        wrong_role,
        artifact_id=_ARTIFACT_ID,
        target_message_index=4,
    )

    assert role_decision.action is GateAction.REVIEW
    assert index_decision.action is GateAction.REVIEW
    assert all(
        decision.assessments[0].indeterminate_reason is IndeterminateReason.INSUFFICIENT_INPUT
        for decision in (role_decision, index_decision)
    )
    assert classifier.call_count == 0


def test_classifier_evidence_role_must_match_bound_checkpoint() -> None:
    classifier = StaticClassifier(
        _assessment("distress", outcome=Outcome.NOT_MATCHED),
        evidence_role=MessageRole.USER,
    )

    with pytest.raises(ValueError, match="evidence_role"):
        WorkflowGate(
            (classifier,),
            checkpoint=Checkpoint.COMMUNICATION,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )


@pytest.mark.parametrize(
    ("checkpoint", "required_role"),
    [
        (Checkpoint.INPUT, MessageRole.USER),
        (Checkpoint.TASK_SELECTION, MessageRole.ASSISTANT),
        (Checkpoint.EXECUTION, MessageRole.ASSISTANT),
        (Checkpoint.TOOL_INPUT, MessageRole.ASSISTANT),
        (Checkpoint.TOOL_OUTPUT, MessageRole.TOOL),
        (Checkpoint.COMMUNICATION, MessageRole.ASSISTANT),
    ],
)
def test_text_convenience_enforces_checkpoint_role(
    checkpoint: Checkpoint,
    required_role: MessageRole,
) -> None:
    def classify_role(
        conversation: Conversation,
        target_message_index: int,
        sensitivity: Sensitivity,
    ) -> Assessment:
        del sensitivity
        role_matches = (
            len(conversation.messages) == 1
            and target_message_index == 0
            and conversation.messages[0].role is required_role
        )
        return _assessment("role_check", outcome=Outcome.NOT_MATCHED if role_matches else Outcome.MATCHED)

    gate = WorkflowGate(
        (CallableClassifier("role_check", classify_role),),
        checkpoint=checkpoint,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )

    assert gate.checkpoint is checkpoint
    assert gate.evaluate_text("One target.", artifact_id=_ARTIFACT_ID).action is GateAction.ALLOW


def test_invalid_text_is_a_fail_safe_decision_without_raw_data() -> None:
    gate = WorkflowGate(
        (StaticClassifier(_assessment("distress", outcome=Outcome.NOT_MATCHED)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )

    decision = gate.evaluate_text(" \n", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.target_message_index is None
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INSUFFICIENT_INPUT
    assert "messages" not in decision.model_dump_json()


def test_unknown_classifier_failures_become_indeterminate_and_never_allow() -> None:
    gate = WorkflowGate(
        (RaisingClassifier("distress", lambda: RuntimeError("raw provider failure")),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text("Sensitive workflow input.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INTERNAL_ERROR
    assert "raw provider failure" not in decision.model_dump_json()
    assert "Sensitive workflow input" not in decision.model_dump_json()


def test_sanitized_classification_error_preserves_only_categorical_reason() -> None:
    gate = WorkflowGate(
        (
            RaisingClassifier(
                "distress",
                lambda: ClassificationError(
                    classifier_id="distress",
                    policy_version="2026.08.1",
                    reason=IndeterminateReason.TIMEOUT,
                ),
            ),
        ),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW, indeterminate_action=GateAction.BLOCK),
    )

    decision = gate.evaluate_text("Sensitive workflow input.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.BLOCK
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.TIMEOUT


def test_subclass_fields_are_not_retained_in_gate_decision() -> None:
    class DataBearingAssessment(Assessment):
        raw_input: str
        provider_payload: dict[str, str]

    assessment = DataBearingAssessment(
        classifier_id="custom",
        policy_version="2026.08.1",
        outcome=Outcome.MATCHED,
        evidence_directness=EvidenceDirectness.EXPLICIT,
        signals=("policy_match",),
        raw_input="private user content",
        provider_payload={"response": "private provider data"},
    )
    gate = WorkflowGate(
        (StaticClassifier(assessment),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )

    decision = gate.evaluate_text("private user content", artifact_id=_ARTIFACT_ID)
    serialized = decision.model_dump_json()

    assert type(decision.assessments[0]) is GateAssessment
    assert "raw_input" not in serialized
    assert "provider_payload" not in serialized
    assert "private user content" not in serialized
    assert "private provider data" not in serialized


def test_escalation_only_complaint_forces_review_and_survives_json_round_trip() -> None:
    escalation = ComplaintEscalation(
        signal=EscalationReason.EXPLICIT_HUMAN_REQUEST,
        directness=EvidenceDirectness.EXPLICIT,
        message_ids=("m0",),
        subject=EvidenceSubject.USER,
        source_context=SourceContext.DIRECT,
    )

    def complaint_observation(**_: object) -> ComplaintsObservation:
        return ComplaintsObservation(
            findings=(),
            escalations=(escalation,),
            insufficient_context=False,
        )

    gate = WorkflowGate(
        (ComplaintsClassifier(CallableBackend(complaint_observation)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text(
        "Please get me a manager.",
        artifact_id="complaint:v1",
        sensitivity=Sensitivity.PRECISE,
    )
    serialized = decision.model_dump_json()
    restored = GateDecision.model_validate_json(serialized)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].outcome is Outcome.NOT_MATCHED
    assert decision.assessments[0].review_signals == ("explicit_human_request",)
    assert restored == decision
    assert "escalations" not in serialized
    assert "message_ids" not in serialized
    assert "Please get me a manager" not in serialized


def test_custom_categorical_review_signal_cannot_fail_open() -> None:
    class CustomRoutingAssessment(Assessment):
        review_signals: tuple[str, ...]

    assessment = CustomRoutingAssessment(
        classifier_id="custom_router",
        policy_version="2026.08.1",
        outcome=Outcome.NOT_MATCHED,
        review_signals=("manual_approval",),
    )
    gate = WorkflowGate(
        (
            StaticClassifier(
                assessment,
                allowed_review_signals=("manual_approval",),
            ),
        ),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text(
        "A target requiring manual approval.",
        artifact_id=_ARTIFACT_ID,
    )

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].review_signals == ("manual_approval",)


def test_invalid_custom_review_signal_becomes_indeterminate_without_leaking() -> None:
    class UnvalidatedRoutingAssessment(Assessment):
        review_signals: tuple[str, ...]

    assessment = UnvalidatedRoutingAssessment.model_construct(
        classifier_id="custom_router",
        policy_version="2026.08.1",
        sensitivity=Sensitivity.BALANCED,
        outcome=Outcome.NOT_MATCHED,
        evidence_directness=EvidenceDirectness.NONE,
        signals=(),
        indeterminate_reason=None,
        review_signals=("private routing reason with spaces",),
    )
    gate = WorkflowGate(
        (
            StaticClassifier(
                assessment,
                allowed_review_signals=("manual_approval",),
            ),
        ),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text("Private workflow input.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].outcome is Outcome.INDETERMINATE
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INVALID_RESPONSE
    assert "private routing reason" not in decision.model_dump_json()


def test_unconfigured_base_signal_becomes_indeterminate_without_leaking() -> None:
    assessment = Assessment(
        classifier_id="custom_router",
        policy_version="2026.08.1",
        outcome=Outcome.MATCHED,
        evidence_directness=EvidenceDirectness.EXPLICIT,
        signals=("private_alice",),
    )
    gate = WorkflowGate(
        (StaticClassifier(assessment, allowed_signals=("policy_match",)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text("Private workflow input.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].outcome is Outcome.INDETERMINATE
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INVALID_RESPONSE
    assert "private_alice" not in decision.model_dump_json()


def test_untrusted_signal_iterable_is_rejected_without_iteration() -> None:
    class TrackingIterable:
        def __init__(self) -> None:
            self.iterated = False

        def __iter__(self) -> Iterator[str]:
            self.iterated = True
            return iter(("policy_match",))

    raw_signals = TrackingIterable()
    assessment = Assessment.model_construct(
        classifier_id="custom_router",
        policy_version="2026.08.1",
        sensitivity=Sensitivity.BALANCED,
        outcome=Outcome.MATCHED,
        evidence_directness=EvidenceDirectness.EXPLICIT,
        signals=cast(tuple[str, ...], raw_signals),
        indeterminate_reason=None,
    )
    gate = WorkflowGate(
        (StaticClassifier(assessment),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text("Private workflow input.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INVALID_RESPONSE
    assert raw_signals.iterated is False


def test_malicious_review_signal_property_becomes_indeterminate() -> None:
    class ExplosiveReviewAssessment(Assessment):
        @property
        def review_signals(self) -> tuple[str, ...]:
            raise RuntimeError("private property failure")

    assessment = ExplosiveReviewAssessment(
        classifier_id="custom_router",
        policy_version="2026.08.1",
        outcome=Outcome.NOT_MATCHED,
    )
    gate = WorkflowGate(
        (
            StaticClassifier(
                assessment,
                allowed_review_signals=("manual_approval",),
            ),
        ),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = gate.evaluate_text("Private workflow input.", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INVALID_RESPONSE
    assert "private property failure" not in decision.model_dump_json()


def test_classifier_iterables_are_bounded_before_binding_properties_are_read() -> None:
    yielded = 0

    def unbounded_classifiers() -> Iterator[StaticClassifier]:
        nonlocal yielded
        while True:
            yielded += 1
            yield StaticClassifier(_assessment(f"classifier_{yielded}", outcome=Outcome.NOT_MATCHED))

    with pytest.raises(ValueError, match=f"at most {MAX_GATE_CLASSIFIERS}"):
        WorkflowGate(
            unbounded_classifiers(),
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )

    assert yielded == MAX_GATE_CLASSIFIERS + 1


def test_list_subclass_cannot_bypass_bounded_classifier_consumption() -> None:
    class HostileList(list[StaticClassifier]):
        def __init__(self) -> None:
            super().__init__()
            self.yielded = 0

        def __len__(self) -> int:
            raise AssertionError("subclass length must not be trusted")

        def __iter__(self) -> Iterator[StaticClassifier]:
            while True:
                self.yielded += 1
                yield StaticClassifier(_assessment(f"hostile_{self.yielded}", outcome=Outcome.NOT_MATCHED))

    classifiers = HostileList()

    with pytest.raises(ValueError, match=f"at most {MAX_GATE_CLASSIFIERS}"):
        WorkflowGate(
            classifiers,
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )

    assert classifiers.yielded == MAX_GATE_CLASSIFIERS + 1


def test_hostile_tuple_and_string_subclasses_are_not_iterated_or_hashed() -> None:
    class HostileTuple(tuple[str, ...]):
        iterated = False

        def __iter__(self) -> Iterator[str]:
            type(self).iterated = True
            raise AssertionError("hostile tuple must not be iterated")

    class HostileString(str):
        hashed = False

        def __hash__(self) -> int:
            type(self).hashed = True
            raise AssertionError("hostile string must not be hashed")

    with pytest.raises(ValueError, match="allowed_signals"):
        WorkflowGate(
            (
                StaticClassifier(
                    _assessment("hostile_config", outcome=Outcome.NOT_MATCHED),
                    allowed_signals=cast(tuple[str, ...], HostileTuple(("policy_match",))),
                ),
            ),
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )
    assert HostileTuple.iterated is False

    raw_signals = (HostileString("policy_match"),)
    malformed = Assessment.model_construct(
        classifier_id="hostile_result",
        policy_version="2026.08.1",
        sensitivity=Sensitivity.BALANCED,
        outcome=Outcome.MATCHED,
        evidence_directness=EvidenceDirectness.EXPLICIT,
        signals=raw_signals,
        indeterminate_reason=None,
    )
    decision = WorkflowGate(
        (StaticClassifier(malformed),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    ).evaluate_text("private", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].outcome is Outcome.INDETERMINATE
    assert HostileString.hashed is False


def test_sensitive_string_subclasses_are_rejected_without_execution_or_trace_leaks() -> None:
    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("PRIVATE STRING ACCESSOR")

        __hash__ = str.__hash__

    gate = WorkflowGate(
        (StaticClassifier(_assessment("distress", outcome=Outcome.NOT_MATCHED)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )
    raw_text = "PRIVATE NORMALIZATION INPUT"

    with pytest.raises(ValueError, match="invalid workflow gate sensitivity") as sensitivity_error:
        gate.evaluate_text(
            raw_text,
            artifact_id=_ARTIFACT_ID,
            sensitivity=cast(Sensitivity, HostileString("balanced")),
        )
    with pytest.raises(ValueError, match="invalid workflow gate artifact_id") as artifact_error:
        gate.evaluate_text(
            raw_text,
            artifact_id=cast(str, HostileString("artifact:v1")),
        )

    for error in (sensitivity_error.value, artifact_error.value):
        locals_text = _gate_traceback_locals(error)
        assert raw_text not in locals_text
        assert "PRIVATE STRING ACCESSOR" not in locals_text


@pytest.mark.parametrize("use_async", [False, True])
def test_cancellation_from_result_sanitization_drops_gate_input(use_async: bool) -> None:
    class CancellingReviewAssessment(Assessment):
        @property
        def review_signals(self) -> tuple[str, ...]:
            raise asyncio.CancelledError("PRIVATE RESULT CANCELLATION")

    assessment = CancellingReviewAssessment(
        classifier_id="cancel_result",
        policy_version="2026.08.1",
        outcome=Outcome.NOT_MATCHED,
    )
    raw_text = "PRIVATE CANCELLED GATE INPUT"

    if use_async:
        gate = AsyncWorkflowGate(
            (StaticAsyncClassifier(assessment, allowed_review_signals=("manual_review",)),),
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )

        async def run() -> None:
            await gate.aevaluate_text(raw_text, artifact_id=_ARTIFACT_ID)

        with pytest.raises(asyncio.CancelledError) as caught:
            asyncio.run(run())
    else:
        gate = WorkflowGate(
            (StaticClassifier(assessment, allowed_review_signals=("manual_review",)),),
            checkpoint=Checkpoint.INPUT,
            policy=GatePolicy(matched_action=GateAction.REVIEW),
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            gate.evaluate_text(raw_text, artifact_id=_ARTIFACT_ID)

    locals_text = _gate_traceback_locals(caught.value)
    assert raw_text not in locals_text
    assert "PRIVATE RESULT CANCELLATION" not in locals_text


@pytest.mark.parametrize("extra", ["openai", "anthropic"])
def test_configuration_error_is_actionable_and_traceback_does_not_retain_sync_input(extra: str) -> None:
    gate = WorkflowGate(
        (RaisingClassifier("distress", lambda: BackendConfigurationError(extra)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )
    raw_text = "traceback-private-sync"

    with pytest.raises(BackendConfigurationError) as captured:
        gate.evaluate_text(raw_text, artifact_id=_ARTIFACT_ID)

    assert captured.value.extra == extra
    assert raw_text not in _gate_traceback_locals(captured.value)
    assert captured.value.__cause__ is None


def test_invalid_artifact_id_traceback_does_not_retain_input() -> None:
    gate = WorkflowGate(
        (StaticClassifier(_assessment("distress", outcome=Outcome.NOT_MATCHED)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )
    raw_text = "traceback-private-artifact-input"
    invalid_artifact_id = "private invalid artifact id"

    with pytest.raises(ValueError, match="invalid workflow gate artifact_id") as captured:
        gate.evaluate_text(raw_text, artifact_id=invalid_artifact_id)

    traceback_locals = _gate_traceback_locals(captured.value)
    assert raw_text not in traceback_locals
    assert invalid_artifact_id not in traceback_locals


def test_invalid_enum_tracebacks_do_not_retain_conversation() -> None:
    gate = WorkflowGate(
        (StaticClassifier(_assessment("distress", outcome=Outcome.NOT_MATCHED)),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )
    raw_text = "traceback-private-conversation"
    invalid_field = "private-invalid-sensitivity"
    conversation = Conversation(messages=(Message(role=MessageRole.USER, content=raw_text),))

    with pytest.raises(ValueError, match="invalid workflow gate sensitivity") as captured:
        gate.evaluate(
            conversation,
            artifact_id=_ARTIFACT_ID,
            sensitivity=cast(Sensitivity, invalid_field),
        )

    traceback_locals = _gate_traceback_locals(captured.value)
    assert raw_text not in traceback_locals
    assert invalid_field not in traceback_locals


@pytest.mark.asyncio
async def test_async_gate_is_concurrent_but_preserves_configuration_order() -> None:
    started = 0
    all_started = asyncio.Event()

    async def barrier() -> None:
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        await all_started.wait()

    gate = AsyncWorkflowGate(
        (
            StaticAsyncClassifier(_assessment("first", outcome=Outcome.NOT_MATCHED), barrier),
            StaticAsyncClassifier(_assessment("second", outcome=Outcome.MATCHED), barrier),
        ),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )

    decision = await asyncio.wait_for(
        gate.aevaluate_text("A bounded stage target.", artifact_id=_ARTIFACT_ID),
        timeout=1,
    )

    assert started == 2
    assert decision.classifier_ids == ("first", "second")
    assert decision.action is GateAction.BLOCK


@pytest.mark.asyncio
async def test_async_unknown_failure_is_indeterminate() -> None:
    gate = AsyncWorkflowGate(
        (RaisingAsyncClassifier("distress", lambda: RuntimeError("private async failure")),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.ALLOW),
    )

    decision = await gate.aevaluate_text("private async input", artifact_id=_ARTIFACT_ID)

    assert decision.action is GateAction.REVIEW
    assert decision.assessments[0].indeterminate_reason is IndeterminateReason.INTERNAL_ERROR
    assert "private" not in decision.model_dump_json()


@pytest.mark.asyncio
async def test_configuration_error_traceback_does_not_retain_async_input() -> None:
    gate = AsyncWorkflowGate(
        (RaisingAsyncClassifier("distress", lambda: BackendConfigurationError("openai")),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )
    raw_text = "traceback-private-async"

    with pytest.raises(BackendConfigurationError) as captured:
        await gate.aevaluate_text(raw_text, artifact_id=_ARTIFACT_ID)

    assert captured.value.extra == "openai"
    assert raw_text not in _gate_traceback_locals(captured.value)
    assert captured.value.__cause__ is None
