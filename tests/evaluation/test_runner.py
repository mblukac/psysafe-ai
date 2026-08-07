"""Synthetic contract tests; they make no classifier-quality claim."""

import asyncio
from enum import Enum

import pytest
from pydantic import Field, ValidationError

from psysafe.backends import CallableBackend
from psysafe.backends.base import BackendError
from psysafe.classifiers.base import Finding, Observation, ObservationRecord, PolicyClassifier
from psysafe.classifiers.complaints import (
    ComplaintCategory,
    ComplaintEscalation,
    ComplaintFinding,
    ComplaintsClassifier,
    ComplaintsObservation,
    EscalationReason,
)
from psysafe.classifiers.context import EvidenceSubject, SourceContext
from psysafe.classifiers.pii import PIIClassifier
from psysafe.classifiers.prompting import PromptSpec
from psysafe.core.classifier import ClassificationError
from psysafe.core.contracts import (
    Assessment,
    AssessmentMetadata,
    Conversation,
    EvidenceDirectness,
    IndeterminateReason,
    Outcome,
    Sensitivity,
)
from psysafe.evaluation import (
    MAX_GOLDEN_CASES,
    BoundaryOutcome,
    CaseOutcome,
    EvaluationReport,
    EvaluationRunError,
    EvaluationRunner,
    EvaluationRunReason,
    EvaluationSplit,
    ExpectedBoundary,
    ExpectedSignal,
    GoldenCase,
    HoldoutDiagnosticsWarning,
    MonotonicityStatus,
    run_evaluation,
)

ObservationModel = Observation[Finding]


def _signal(name: str, boundary: ExpectedBoundary) -> ExpectedSignal:
    return ExpectedSignal(name=name, boundary=boundary)


def _case(
    *,
    case_id: str = "synthetic-1",
    family_id: str = "family-1",
    split: EvaluationSplit = EvaluationSplit.TUNING,
    content: str = "Synthetic contract fixture.",
    expected_boundary: ExpectedBoundary = ExpectedBoundary.NEVER,
    expected_signals: tuple[ExpectedSignal, ...] | None = None,
    expected_review_boundary: ExpectedBoundary | None = None,
    expected_review_signals: tuple[ExpectedSignal, ...] | None = None,
    slices: tuple[str, ...] = ("synthetic",),
) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        family_id=family_id,
        split=split,
        slices=slices,
        conversation=Conversation.from_text(content),
        expected_boundary=expected_boundary,
        expected_signals=expected_signals,
        expected_review_boundary=expected_review_boundary,
        expected_review_signals=expected_review_signals,
    )


def _policy(
    backend: CallableBackend,
    *,
    allowed_signals: frozenset[str] = frozenset({"test_signal"}),
) -> PolicyClassifier[Observation[Finding]]:
    return PolicyClassifier(
        classifier_id="synthetic_policy",
        policy_version="1",
        prompt=PromptSpec(instructions="Classify the synthetic contract fixture."),
        backend=backend,
        observation_model=ObservationModel,
        allowed_signals=allowed_signals,
    )


def _finding(
    signal: str = "test_signal",
    directness: EvidenceDirectness = EvidenceDirectness.CONTEXTUAL,
) -> Finding:
    return Finding(signal=signal, directness=directness, message_ids=("m0",))


def _evaluation_traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if "/psysafe/evaluation/" in traceback.tb_frame.f_code.co_filename:
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


def test_policy_runner_reuses_one_observation_and_reports_documented_metrics() -> None:
    backend = CallableBackend(
        lambda **_: ObservationModel(findings=(_finding(),), insufficient_context=False),
    )
    golden_case = _case(
        expected_boundary=ExpectedBoundary.BALANCED,
        expected_signals=(_signal("test_signal", ExpectedBoundary.BALANCED),),
    )

    report = run_evaluation(_policy(backend), (golden_case,), split=EvaluationSplit.TUNING)

    assert backend.call_count == 1
    assert report.case_count == 1
    assert report.classifier_id == "synthetic_policy"
    assert report.policy_version == "1"
    assert report.diagnostics_revealed is True
    result = report.case_results[0]
    assert result.classifier_id == report.classifier_id
    assert result.policy_version == report.policy_version
    assert result.outcome is CaseOutcome.PASSED
    assert result.monotonicity is MonotonicityStatus.VERIFIED
    assert tuple(boundary.outcome for boundary in result.boundaries) == (
        BoundaryOutcome.TRUE_NEGATIVE,
        BoundaryOutcome.TRUE_POSITIVE,
        BoundaryOutcome.TRUE_POSITIVE,
    )
    assert result.boundaries[0].expected_signals == ()
    assert result.boundaries[1].expected_signals == ("test_signal",)
    assert report.metrics.true_positive == 2
    assert report.metrics.true_negative == 1
    assert report.metrics.coverage == 1.0
    assert report.metrics.precision == 1.0
    assert report.metrics.recall == 1.0
    assert report.metrics.effective_recall == 1.0
    assert report.metrics.false_positive_rate == 0.0
    assert report.metrics.false_negative_rate == 0.0
    assert report.metrics.indeterminate_rate == 0.0
    assert report.metrics.intervention_rate == pytest.approx(2 / 3)
    assert report.metrics.signal_matches == 2
    assert report.metrics.signal_match_rate == 1.0
    assert report.slices[0].slice == "synthetic"
    assert "Synthetic contract fixture" not in report.model_dump_json()
    assert "confidence" not in result.model_dump_json()
    assert "score" not in result.model_dump_json()


def test_signal_expectations_can_expand_at_broader_boundaries() -> None:
    backend = CallableBackend(
        lambda **_: ObservationModel(
            findings=(
                _finding("direct_signal", EvidenceDirectness.EXPLICIT),
                _finding("context_signal", EvidenceDirectness.CONTEXTUAL),
            ),
            insufficient_context=False,
        ),
    )
    golden_case = _case(
        expected_boundary=ExpectedBoundary.PRECISE,
        expected_signals=(
            _signal("direct_signal", ExpectedBoundary.PRECISE),
            _signal("context_signal", ExpectedBoundary.BALANCED),
        ),
    )

    report = run_evaluation(
        _policy(backend, allowed_signals=frozenset({"direct_signal", "context_signal"})),
        (golden_case,),
        split=EvaluationSplit.TUNING,
    )

    result = report.case_results[0]
    assert result.outcome is CaseOutcome.PASSED
    assert tuple(boundary.actual_signals for boundary in result.boundaries) == (
        ("direct_signal",),
        ("direct_signal", "context_signal"),
        ("direct_signal", "context_signal"),
    )
    assert report.metrics.signal_matches == 3


def test_one_policy_backend_failure_becomes_three_indeterminates_without_retry() -> None:
    backend = CallableBackend(lambda **_: (_ for _ in ()).throw(RuntimeError("PRIVATE PROVIDER BODY")))
    golden_case = _case(
        expected_boundary=ExpectedBoundary.PRECISE,
        expected_signals=(_signal("test_signal", ExpectedBoundary.PRECISE),),
    )

    report = EvaluationRunner(_policy(backend), split=EvaluationSplit.TUNING).run((golden_case,))

    assert backend.call_count == 1
    result = report.case_results[0]
    assert result.outcome is CaseOutcome.UNEXPECTED_INDETERMINATE
    assert result.monotonicity is MonotonicityStatus.INDETERMINATE
    assert {boundary.actual_outcome for boundary in result.boundaries} == {Outcome.INDETERMINATE}
    assert {boundary.indeterminate_reason for boundary in result.boundaries} == {
        IndeterminateReason.PROVIDER_ERROR,
    }
    assert report.metrics.coverage == 0.0
    assert report.metrics.recall is None
    assert report.metrics.effective_recall == 0.0
    assert report.metrics.positive_indeterminate == 3
    assert report.metrics.indeterminate_rate == 1.0
    assert report.metrics.intervention_rate == 1.0
    assert "PRIVATE" not in report.model_dump_json()


class _CalibrationFailurePolicy(PolicyClassifier[Observation[Finding]]):
    def __init__(self, backend: CallableBackend) -> None:
        super().__init__(
            classifier_id="calibration_failure",
            policy_version="1",
            prompt=PromptSpec(instructions="Synthetic contract fixture."),
            backend=backend,
            observation_model=ObservationModel,
            allowed_signals=frozenset({"test_signal"}),
        )
        self.calibration_calls = 0

    def calibrate(
        self,
        record: ObservationRecord[Observation[Finding]],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        self.calibration_calls += 1
        if sensitivity is Sensitivity.BALANCED:
            raise RuntimeError("PRIVATE CALIBRATION STATE")
        return super().calibrate(record, sensitivity=sensitivity)


def test_policy_calibration_failure_stops_without_observing_again() -> None:
    backend = CallableBackend(
        lambda **_: ObservationModel(
            findings=(_finding(directness=EvidenceDirectness.EXPLICIT),),
            insufficient_context=False,
        ),
    )
    classifier = _CalibrationFailurePolicy(backend)
    golden_case = _case(
        expected_boundary=ExpectedBoundary.PRECISE,
        expected_signals=(_signal("test_signal", ExpectedBoundary.PRECISE),),
    )

    report = run_evaluation(classifier, (golden_case,), split=EvaluationSplit.TUNING)

    assert backend.call_count == 1
    assert classifier.calibration_calls == 2
    assert tuple(boundary.actual_outcome for boundary in report.case_results[0].boundaries) == (
        Outcome.MATCHED,
        Outcome.INDETERMINATE,
        Outcome.INDETERMINATE,
    )
    assert report.case_results[0].boundaries[1].indeterminate_reason is IndeterminateReason.INTERNAL_ERROR
    assert "PRIVATE" not in report.model_dump_json()


class _CountingCalibrationPolicy(_CalibrationFailurePolicy):
    def calibrate(
        self,
        record: ObservationRecord[Observation[Finding]],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        self.calibration_calls += 1
        return PolicyClassifier.calibrate(self, record, sensitivity=sensitivity)


def test_policy_runner_calibrates_every_boundary_from_one_indeterminate_observation() -> None:
    backend = CallableBackend(
        lambda **_: ObservationModel(findings=(), insufficient_context=True),
    )
    classifier = _CountingCalibrationPolicy(backend)

    report = run_evaluation(
        classifier,
        (_case(expected_boundary=ExpectedBoundary.INDETERMINATE),),
        split=EvaluationSplit.TUNING,
    )

    assert backend.call_count == 1
    assert classifier.calibration_calls == 3
    assert {boundary.actual_outcome for boundary in report.case_results[0].boundaries} == {
        Outcome.INDETERMINATE,
    }
    assert report.case_results[0].outcome is CaseOutcome.EXPECTED_INDETERMINATE


class _CancellingOrdinaryClassifier:
    classifier_id = "cancelling_ordinary"
    policy_version = "2026.08.2"
    allowed_signals: tuple[str, ...] = ()
    allowed_review_signals: tuple[str, ...] = ()

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, sensitivity
        raise asyncio.CancelledError("PRIVATE CANCELLATION BODY")


class _CancellingPolicy(PolicyClassifier[Observation[Finding]]):
    def __init__(self, backend: CallableBackend, *, phase: str) -> None:
        super().__init__(
            classifier_id="cancelling_policy",
            policy_version="2026.08.2",
            prompt=PromptSpec(instructions="Synthetic cancellation fixture."),
            backend=backend,
            observation_model=ObservationModel,
            allowed_signals=frozenset({"test_signal"}),
        )
        self._phase = phase

    def observe(self, conversation: Conversation) -> ObservationRecord[Observation[Finding]]:
        if self._phase == "observe":
            raise asyncio.CancelledError("PRIVATE CANCELLATION BODY")
        return super().observe(conversation)

    def calibrate(
        self,
        record: ObservationRecord[Observation[Finding]],
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        if self._phase == "calibrate":
            raise asyncio.CancelledError("PRIVATE CANCELLATION BODY")
        return super().calibrate(record, sensitivity=sensitivity)


def _cancelling_classifier(phase: str):
    if phase == "ordinary":
        return _CancellingOrdinaryClassifier()
    backend = CallableBackend(
        lambda **_: ObservationModel(
            findings=(_finding(directness=EvidenceDirectness.EXPLICIT),),
            insufficient_context=False,
        ),
    )
    return _CancellingPolicy(backend, phase=phase)


def _invoke_cancelling_run(classifier, golden_case: GoldenCase, *, direct_runner: bool) -> None:
    if direct_runner:
        EvaluationRunner(classifier, split=EvaluationSplit.TUNING).run((golden_case,))
    else:
        run_evaluation(classifier, (golden_case,), split=EvaluationSplit.TUNING)


@pytest.mark.parametrize("phase", ["ordinary", "observe", "calibrate"])
@pytest.mark.parametrize("direct_runner", [False, True])
def test_runner_resanitizes_cancellation_without_retaining_cases(phase: str, direct_runner: bool) -> None:
    golden_case = _case(content="PRIVATE CANCELLATION CONVERSATION")
    classifier = _cancelling_classifier(phase)

    with pytest.raises(asyncio.CancelledError) as caught:
        _invoke_cancelling_run(classifier, golden_case, direct_runner=direct_runner)

    assert str(caught.value) == ""
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    library_locals = _evaluation_traceback_locals(caught.value)
    assert "PRIVATE CANCELLATION BODY" not in library_locals
    assert "PRIVATE CANCELLATION CONVERSATION" not in library_locals


class _CountingPII:
    classifier_id = PIIClassifier.classifier_id
    policy_version = PIIClassifier.policy_version

    def __init__(self) -> None:
        self.calls = 0
        self._classifier = PIIClassifier()

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return self._classifier.allowed_signals

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return self._classifier.allowed_review_signals

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        self.calls += 1
        return self._classifier.classify(conversation, sensitivity=sensitivity)


def test_ordinary_deterministic_classifier_runs_once_per_sensitivity() -> None:
    classifier = _CountingPII()
    golden_case = _case(
        content="Contact synthetic@example.test",
        expected_boundary=ExpectedBoundary.PRECISE,
        expected_signals=(_signal("email_address", ExpectedBoundary.PRECISE),),
    )

    report = run_evaluation(classifier, (golden_case,), split=EvaluationSplit.TUNING)

    assert classifier.calls == 3
    assert report.case_results[0].outcome is CaseOutcome.PASSED
    assert report.metrics.true_positive == 3
    assert report.metrics.signal_matches == 3
    assert "synthetic@example.test" not in report.model_dump_json()


class _ExplodingClassifier:
    classifier_id = "synthetic_exploding"
    policy_version = "1"
    allowed_signals: tuple[str, ...] = ()
    allowed_review_signals: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.calls = 0

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, sensitivity
        self.calls += 1
        raise RuntimeError("PRIVATE UNKNOWN CLASSIFIER BODY")


def test_unknown_ordinary_failure_is_categorical_and_stops_retrying() -> None:
    classifier = _ExplodingClassifier()

    report = run_evaluation(classifier, (_case(),), split=EvaluationSplit.TUNING)

    assert classifier.calls == 1
    assert {boundary.actual_outcome for boundary in report.case_results[0].boundaries} == {
        Outcome.INDETERMINATE,
    }
    assert {boundary.indeterminate_reason for boundary in report.case_results[0].boundaries} == {
        IndeterminateReason.INTERNAL_ERROR,
    }
    assert "PRIVATE" not in report.model_dump_json()


class _MutableIdentityClassifier:
    allowed_signals: tuple[str, ...] = ()
    allowed_review_signals: tuple[str, ...] = ()

    def __init__(self, *, explode: bool = False) -> None:
        self.classifier_id_reads = 0
        self.policy_version_reads = 0
        self.calls = 0
        self._explode = explode

    @property
    def classifier_id(self) -> str:
        self.classifier_id_reads += 1
        return "snapshot_classifier" if self.classifier_id_reads == 1 else "mutated_classifier"

    @property
    def policy_version(self) -> str:
        self.policy_version_reads += 1
        return "2026.08.2" if self.policy_version_reads == 1 else "mutated"

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        self.calls += 1
        if self._explode:
            raise RuntimeError("PRIVATE MUTABLE CLASSIFIER FAILURE")
        return Assessment(
            classifier_id="snapshot_classifier",
            policy_version="2026.08.2",
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
        )


@pytest.mark.parametrize("explode", [False, True])
def test_runner_snapshots_policy_identity_once_for_results_and_failures(explode: bool) -> None:
    classifier = _MutableIdentityClassifier(explode=explode)

    report = run_evaluation(classifier, (_case(),), split=EvaluationSplit.TUNING)

    assert classifier.classifier_id_reads == 1
    assert classifier.policy_version_reads == 1
    assert report.classifier_id == "snapshot_classifier"
    assert report.policy_version == "2026.08.2"
    assert report.case_results[0].classifier_id == report.classifier_id
    assert report.case_results[0].policy_version == report.policy_version
    assert {boundary.actual_outcome for boundary in report.case_results[0].boundaries} == {
        Outcome.INDETERMINATE if explode else Outcome.NOT_MATCHED,
    }


class _HostileClassificationError(ClassificationError):
    def __init__(self) -> None:
        RuntimeError.__init__(self, "PRIVATE HOSTILE ERROR")

    @property
    def reason(self) -> IndeterminateReason:
        raise RuntimeError("PRIVATE HOSTILE REASON")


class _HostileBackendError(BackendError):
    def __init__(self) -> None:
        RuntimeError.__init__(self, "PRIVATE HOSTILE ERROR")

    @property
    def reason(self) -> IndeterminateReason:
        raise RuntimeError("PRIVATE HOSTILE REASON")


class _TypedErrorClassifier(_ExplodingClassifier):
    def __init__(self, error_type: type[Exception]) -> None:
        super().__init__()
        self._error_type = error_type

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, sensitivity
        self.calls += 1
        raise self._error_type()


@pytest.mark.parametrize(
    ("error_type", "expected_reason"),
    [
        (_HostileClassificationError, IndeterminateReason.INTERNAL_ERROR),
        (_HostileBackendError, IndeterminateReason.PROVIDER_ERROR),
    ],
)
def test_untrusted_exception_reason_accessors_are_never_trusted(error_type, expected_reason) -> None:
    classifier = _TypedErrorClassifier(error_type)

    report = run_evaluation(classifier, (_case(),), split=EvaluationSplit.TUNING)

    assert classifier.calls == 1
    assert {boundary.indeterminate_reason for boundary in report.case_results[0].boundaries} == {
        expected_reason,
    }
    assert "PRIVATE" not in report.model_dump_json()


class _NonMonotonicClassifier:
    classifier_id = "synthetic_nonmonotonic"
    policy_version = "1"
    allowed_signals = ("test_signal",)
    allowed_review_signals: tuple[str, ...] = ()

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        if sensitivity is Sensitivity.BALANCED:
            return Assessment(
                classifier_id=self.classifier_id,
                policy_version=self.policy_version,
                sensitivity=sensitivity,
                outcome=Outcome.NOT_MATCHED,
            )
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=("test_signal",),
        )


def test_runner_explicitly_reports_nonmonotonic_classifier_behavior() -> None:
    report = run_evaluation(
        _NonMonotonicClassifier(),
        (_case(expected_boundary=ExpectedBoundary.PRECISE),),
        split=EvaluationSplit.TUNING,
    )

    assert report.case_results[0].monotonicity is MonotonicityStatus.VIOLATED
    assert report.case_results[0].outcome is CaseOutcome.MONOTONICITY_VIOLATION
    assert report.monotonicity_violations == 1


class _SignalDropClassifier:
    classifier_id = "synthetic_signal_drop"
    policy_version = "1"
    allowed_signals = ("signal_a", "signal_b")
    allowed_review_signals: tuple[str, ...] = ()

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        signals = {
            Sensitivity.PRECISE: ("signal_a",),
            Sensitivity.BALANCED: ("signal_a", "signal_b"),
            Sensitivity.PRECAUTIONARY: ("signal_b",),
        }[sensitivity]
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=signals,
        )


def test_monotonicity_requires_broader_boundaries_to_preserve_signal_sets() -> None:
    report = run_evaluation(
        _SignalDropClassifier(),
        (_case(expected_boundary=ExpectedBoundary.PRECISE),),
        split=EvaluationSplit.TUNING,
    )

    assert report.case_results[0].monotonicity is MonotonicityStatus.VIOLATED
    assert report.case_results[0].outcome is CaseOutcome.MONOTONICITY_VIOLATION


class _ReviewAssessment(Assessment):
    review_signals: tuple[str, ...] = Field(default_factory=tuple)


class _IndependentReviewClassifier:
    classifier_id = "synthetic_review"
    policy_version = "1"
    allowed_signals: tuple[str, ...] = ()
    allowed_review_signals = ("human_review",)

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        review_signals = () if sensitivity is Sensitivity.PRECISE else ("human_review",)
        return _ReviewAssessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
            review_signals=review_signals,
        )


def test_independent_review_boundary_evaluates_escalation_only_cases() -> None:
    golden_case = _case(
        expected_boundary=ExpectedBoundary.NEVER,
        expected_review_boundary=ExpectedBoundary.BALANCED,
        expected_review_signals=(_signal("human_review", ExpectedBoundary.BALANCED),),
    )

    report = run_evaluation(
        _IndependentReviewClassifier(),
        (golden_case,),
        split=EvaluationSplit.TUNING,
    )

    result = report.case_results[0]
    assert result.outcome is CaseOutcome.PASSED
    assert result.review_monotonicity is MonotonicityStatus.VERIFIED
    assert report.metrics.true_negative == 3
    assert report.review_metrics is not None
    assert report.review_metrics.true_negative == 1
    assert report.review_metrics.true_positive == 2
    assert report.review_metrics.signal_matches == 2


class _ReviewSignalDropClassifier(_IndependentReviewClassifier):
    allowed_review_signals = ("review_a", "review_b")

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        review_signals = {
            Sensitivity.PRECISE: ("review_a",),
            Sensitivity.BALANCED: ("review_a", "review_b"),
            Sensitivity.PRECAUTIONARY: ("review_b",),
        }[sensitivity]
        return _ReviewAssessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
            review_signals=review_signals,
        )


def test_review_monotonicity_also_preserves_signal_sets() -> None:
    report = run_evaluation(
        _ReviewSignalDropClassifier(),
        (
            _case(
                expected_review_boundary=ExpectedBoundary.PRECISE,
            ),
        ),
        split=EvaluationSplit.TUNING,
    )

    assert report.case_results[0].review_monotonicity is MonotonicityStatus.VIOLATED
    assert report.case_results[0].outcome is CaseOutcome.MONOTONICITY_VIOLATION
    assert report.review_monotonicity_violations == 1


def test_complaint_escalation_reasons_are_extracted_categorically() -> None:
    escalation = ComplaintEscalation(
        signal=EscalationReason.LEGAL_OR_REGULATORY_CONCERN,
        directness=EvidenceDirectness.CONTEXTUAL,
        message_ids=("m0",),
        subject=EvidenceSubject.USER,
        source_context=SourceContext.DIRECT,
    )
    finding = ComplaintFinding(
        signal=ComplaintCategory.BILLING_OR_PAYMENT,
        directness=EvidenceDirectness.EXPLICIT,
        message_ids=("m0",),
        subject=EvidenceSubject.USER,
        source_context=SourceContext.DIRECT,
    )
    backend = CallableBackend(
        lambda **_: ComplaintsObservation(
            findings=(finding,),
            escalations=(escalation,),
            insufficient_context=False,
        ),
    )
    golden_case = _case(
        expected_boundary=ExpectedBoundary.PRECISE,
        expected_signals=(_signal(ComplaintCategory.BILLING_OR_PAYMENT.value, ExpectedBoundary.PRECISE),),
        expected_review_boundary=ExpectedBoundary.BALANCED,
        expected_review_signals=(
            _signal(EscalationReason.LEGAL_OR_REGULATORY_CONCERN.value, ExpectedBoundary.BALANCED),
        ),
    )

    report = run_evaluation(
        ComplaintsClassifier(backend),
        (golden_case,),
        split=EvaluationSplit.TUNING,
    )

    assert backend.call_count == 1
    result = report.case_results[0]
    assert result.outcome is CaseOutcome.PASSED
    assert result.boundaries[0].actual_review_signals == ()
    assert result.boundaries[1].actual_review_signals == (EscalationReason.LEGAL_OR_REGULATORY_CONCERN.value,)


def test_review_vocabulary_requires_complete_positive_and_negative_labels() -> None:
    classifier = _IndependentReviewClassifier()
    labeled = _case(case_id="labeled", family_id="labeled-family", expected_review_boundary=ExpectedBoundary.NEVER)
    unlabeled = _case(case_id="unlabeled", family_id="unlabeled-family")

    for cases in ((unlabeled,), (labeled, unlabeled)):
        with pytest.raises(EvaluationRunError) as caught:
            run_evaluation(classifier, cases, split=EvaluationSplit.TUNING)
        assert caught.value.reason is EvaluationRunReason.INCOMPLETE_REVIEW_LABELS

    with pytest.raises(EvaluationRunError) as missing_vocabulary:
        run_evaluation(
            _CountingPII(),
            (labeled,),
            split=EvaluationSplit.TUNING,
        )
    assert missing_vocabulary.value.reason is EvaluationRunReason.MISSING_REVIEW_VOCABULARY


def test_holdout_reports_are_aggregate_only_unless_explicitly_revealed() -> None:
    golden_case = _case(
        case_id="private-holdout-case",
        family_id="private-holdout-family",
        split=EvaluationSplit.HOLDOUT,
        content="PRIVATE HOLDOUT CONVERSATION",
    )
    classifier = _CountingPII()

    sealed = run_evaluation(classifier, (golden_case,), split=EvaluationSplit.HOLDOUT)
    with pytest.warns(HoldoutDiagnosticsWarning):
        revealed = run_evaluation(
            classifier,
            (golden_case,),
            split=EvaluationSplit.HOLDOUT,
            reveal_holdout_diagnostics=True,
        )

    assert sealed.diagnostics_revealed is False
    assert sealed.case_results == ()
    assert sealed.slices == ()
    assert "private-holdout-case" not in sealed.model_dump_json()
    assert "private-holdout-family" not in sealed.model_dump_json()
    assert "PRIVATE HOLDOUT CONVERSATION" not in sealed.model_dump_json()
    assert revealed.diagnostics_revealed is True
    assert revealed.case_results[0].case_id == "private-holdout-case"
    assert revealed.slices[0].slice == "synthetic"


def test_holdout_reveal_flag_rejects_truthy_non_booleans() -> None:
    with pytest.raises(EvaluationRunError) as caught:
        run_evaluation(
            _CountingPII(),
            (_case(split=EvaluationSplit.HOLDOUT),),
            split=EvaluationSplit.HOLDOUT,
            reveal_holdout_diagnostics="false",  # type: ignore[arg-type]
        )

    assert caught.value.reason is EvaluationRunReason.INVALID_ASSESSMENT


def test_runner_constructor_sanitizes_invalid_private_configuration() -> None:
    with pytest.raises(EvaluationRunError) as caught:
        EvaluationRunner(
            _CountingPII(),
            split=EvaluationSplit.TUNING,
            allowed_signals=("PRIVATE CONFIG VALUE",),
        )

    assert caught.value.reason is EvaluationRunReason.INVALID_ASSESSMENT
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "PRIVATE CONFIG VALUE" not in _evaluation_traceback_locals(caught.value)


class _HostileCaseList(list[GoldenCase]):
    iterated = False

    def __iter__(self):
        type(self).iterated = True
        return super().__iter__()


def test_runner_bounds_direct_case_containers_before_iteration() -> None:
    golden_case = _case(content="PRIVATE BOUNDED CASE")
    hostile = _HostileCaseList([golden_case])
    oversized = [golden_case] * (MAX_GOLDEN_CASES + 1)

    with pytest.raises(EvaluationRunError) as hostile_error:
        run_evaluation(_CountingPII(), hostile, split=EvaluationSplit.TUNING)
    with pytest.raises(EvaluationRunError) as oversized_error:
        run_evaluation(_CountingPII(), oversized, split=EvaluationSplit.TUNING)

    assert hostile_error.value.reason is EvaluationRunReason.INVALID_DATASET
    assert oversized_error.value.reason is EvaluationRunReason.TOO_MANY_CASES
    assert _HostileCaseList.iterated is False
    for caught in (hostile_error.value, oversized_error.value):
        assert "PRIVATE BOUNDED CASE" not in _evaluation_traceback_locals(caught)


def test_report_contract_rejects_diagnostics_flag_mismatch() -> None:
    report = run_evaluation(_CountingPII(), (_case(),), split=EvaluationSplit.TUNING)
    invalid = report.model_dump()
    invalid["diagnostics_revealed"] = False

    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(invalid)

    sealed_with_slice = report.model_dump()
    sealed_with_slice["diagnostics_revealed"] = False
    sealed_with_slice["case_results"] = []
    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(sealed_with_slice)

    mismatched_policy = report.model_dump()
    mismatched_policy["case_results"][0]["policy_version"] = "other"
    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(mismatched_policy)


def test_runner_selects_one_split_and_sanitizes_family_leakage_errors() -> None:
    classifier = _CountingPII()
    tuning = _case(case_id="tuning-1", family_id="tuning-family")
    holdout = _case(
        case_id="holdout-1",
        family_id="holdout-family",
        split=EvaluationSplit.HOLDOUT,
    )

    with pytest.warns(HoldoutDiagnosticsWarning):
        report = run_evaluation(
            classifier,
            (tuning, holdout),
            split=EvaluationSplit.HOLDOUT,
            reveal_holdout_diagnostics=True,
        )

    assert report.case_count == 1
    assert report.case_results[0].case_id == "holdout-1"
    assert classifier.calls == 3

    leaked = _case(
        case_id="private-leaked-id",
        family_id="tuning-family",
        split=EvaluationSplit.HOLDOUT,
        content="PRIVATE LEAKED CONVERSATION",
    )
    with pytest.raises(EvaluationRunError) as caught:
        run_evaluation(classifier, (tuning, leaked), split=EvaluationSplit.HOLDOUT)

    assert caught.value.reason is EvaluationRunReason.FAMILY_SPLIT_CONFLICT
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    library_locals = _evaluation_traceback_locals(caught.value)
    assert "PRIVATE LEAKED CONVERSATION" not in library_locals
    assert "private-leaked-id" not in library_locals


class _UnknownSignalClassifier:
    classifier_id = "unknown_signal"
    policy_version = "1"
    allowed_signals = ("public_signal",)
    allowed_review_signals: tuple[str, ...] = ()

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=("private_signal",),
        )


def test_runner_validation_errors_drop_cases_and_untrusted_outputs_from_tracebacks() -> None:
    golden_case = _case(content="PRIVATE VALIDATION CONVERSATION")

    with pytest.raises(EvaluationRunError) as public_error:
        run_evaluation(_UnknownSignalClassifier(), (golden_case,), split=EvaluationSplit.TUNING)
    with pytest.raises(EvaluationRunError) as direct_error:
        EvaluationRunner(_UnknownSignalClassifier(), split=EvaluationSplit.TUNING).run((golden_case,))

    for caught in (public_error.value, direct_error.value):
        assert caught.reason is EvaluationRunReason.INVALID_ASSESSMENT
        assert str(caught) == "evaluation could not run (invalid_assessment)"
        assert caught.__cause__ is None
        assert caught.__context__ is None
        library_locals = _evaluation_traceback_locals(caught)
        assert "PRIVATE VALIDATION CONVERSATION" not in library_locals
        assert "private_signal" not in library_locals


class _UnsafeReviewSignal(str, Enum):
    VALUE = "Unsafe review text"


class _UnsafeReviewAssessment(Assessment):
    review_signals: tuple[_UnsafeReviewSignal, ...] = (_UnsafeReviewSignal.VALUE,)


class _UnsafeAssessmentClassifier:
    classifier_id = "unsafe_assessment"
    policy_version = "1"
    allowed_signals: tuple[str, ...] = ()
    allowed_review_signals = ("safe_review",)

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        return _UnsafeReviewAssessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
        )


def test_runner_rejects_non_categorical_extracted_review_values() -> None:
    with pytest.raises(EvaluationRunError) as caught:
        run_evaluation(
            _UnsafeAssessmentClassifier(),
            (_case(expected_review_boundary=ExpectedBoundary.NEVER),),
            split=EvaluationSplit.TUNING,
        )

    assert caught.value.reason is EvaluationRunReason.INVALID_ASSESSMENT


class _HostileSignalTuple(tuple[str, ...]):
    iterated = False

    def __iter__(self):
        type(self).iterated = True
        return super().__iter__()


class _ConstructedAssessmentClassifier:
    classifier_id = "constructed_assessment"
    policy_version = "1"
    allowed_signals = ("safe_signal",)
    allowed_review_signals: tuple[str, ...] = ()

    def __init__(self, signals: object) -> None:
        self._signals = signals

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        return Assessment.model_construct(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=self._signals,
            indeterminate_reason=None,
            metadata=AssessmentMetadata(),
        )


@pytest.mark.parametrize("signals", [("safe_signal",), _HostileSignalTuple(("safe_signal",))])
def test_runner_revalidates_constructed_assessment_state_without_iterating_hostile_values(signals: object) -> None:
    _HostileSignalTuple.iterated = False

    with pytest.raises(EvaluationRunError) as caught:
        run_evaluation(
            _ConstructedAssessmentClassifier(signals),
            (_case(),),
            split=EvaluationSplit.TUNING,
        )

    assert caught.value.reason is EvaluationRunReason.INVALID_ASSESSMENT
    assert _HostileSignalTuple.iterated is False


class _DuplicateReviewAssessment(Assessment):
    @property
    def review_signals(self) -> object:
        return ("safe_review", "safe_review")


class _OversizedReviewAssessment(Assessment):
    @property
    def review_signals(self) -> object:
        return ("safe_review",) * 65


class _HostileReviewList(list[str]):
    iterated = False

    def __iter__(self):
        type(self).iterated = True
        return super().__iter__()


class _HostileContainerReviewAssessment(Assessment):
    @property
    def review_signals(self) -> object:
        return _HostileReviewList(["safe_review"])


class _ReviewContainerClassifier(_UnsafeAssessmentClassifier):
    def __init__(self, assessment_type: type[Assessment]) -> None:
        self._assessment_type = assessment_type

    def classify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation
        return self._assessment_type(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
        )


@pytest.mark.parametrize(
    "assessment_type",
    [
        _DuplicateReviewAssessment,
        _OversizedReviewAssessment,
        _HostileContainerReviewAssessment,
    ],
)
def test_runner_rejects_duplicate_oversized_and_hostile_review_containers(assessment_type) -> None:
    _HostileReviewList.iterated = False

    with pytest.raises(EvaluationRunError) as caught:
        run_evaluation(
            _ReviewContainerClassifier(assessment_type),
            (_case(expected_review_boundary=ExpectedBoundary.NEVER),),
            split=EvaluationSplit.TUNING,
        )

    assert caught.value.reason is EvaluationRunReason.INVALID_ASSESSMENT
    assert _HostileReviewList.iterated is False
