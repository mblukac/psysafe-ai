"""Evaluation runner for calibrated and ordinary classifiers."""

from __future__ import annotations

import asyncio
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn, Protocol, cast, runtime_checkable

from psysafe.backends.base import BackendConfigurationError, BackendError, _fresh_configuration_error
from psysafe.classifiers.base import PolicyClassifier
from psysafe.core.classifier import ClassificationError, Classifier
from psysafe.core.contracts import (
    MAX_ASSESSMENT_SIGNALS,
    Assessment,
    AssessmentMetadata,
    EvidenceDirectness,
    IndeterminateReason,
    Outcome,
    Sensitivity,
)
from psysafe.evaluation.loader import MAX_GOLDEN_CASES, GoldenCaseLoadError, audit_split_families
from psysafe.evaluation.metrics import MetricRow, summarize
from psysafe.evaluation.models import (
    BoundaryOutcome,
    BoundaryResult,
    CaseOutcome,
    CaseResult,
    EvaluationReport,
    EvaluationSplit,
    GoldenCase,
    MetricSummary,
    MonotonicityStatus,
    SensitivityReport,
    SliceReport,
)

_SENSITIVITIES = (Sensitivity.PRECISE, Sensitivity.BALANCED, Sensitivity.PRECAUTIONARY)
_SAFE_SIGNAL = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")
_SAFE_POLICY_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


@dataclass(frozen=True, slots=True)
class _AssessmentSnapshot:
    """Detached categorical fields used after the classifier boundary."""

    assessment: Assessment
    review_signals: tuple[str, ...] = ()


@runtime_checkable
class _HasReviewSignals(Protocol):
    @property
    def review_signals(self) -> object: ...


@runtime_checkable
class _HasEscalationReasons(Protocol):
    @property
    def escalation_reasons(self) -> object: ...


@runtime_checkable
class _HasAllowedSignals(Protocol):
    @property
    def allowed_signals(self) -> object: ...


@runtime_checkable
class _HasAllowedReviewSignals(Protocol):
    @property
    def allowed_review_signals(self) -> object: ...


class EvaluationRunReason(str, Enum):
    """Safe categories for local runner failures."""

    EMPTY_DATASET = "empty_dataset"
    EMPTY_SPLIT = "empty_split"
    DUPLICATE_CASE_ID = "duplicate_case_id"
    FAMILY_SPLIT_CONFLICT = "family_split_conflict"
    INCOMPLETE_REVIEW_LABELS = "incomplete_review_labels"
    MISSING_REVIEW_VOCABULARY = "missing_review_vocabulary"
    INVALID_EXPECTATION = "invalid_expectation"
    INVALID_ASSESSMENT = "invalid_assessment"
    INVALID_DATASET = "invalid_dataset"
    TOO_MANY_CASES = "too_many_cases"


class EvaluationRunError(ValueError):
    """A fixed-message run error without case content or provider text."""

    def __init__(self, reason: EvaluationRunReason) -> None:
        self.reason = reason
        super().__init__(f"evaluation could not run ({reason.value})")


class HoldoutDiagnosticsWarning(UserWarning):
    """Revealed case diagnostics invalidate repeated unbiased holdout use."""


def _raise_run_error(reason: EvaluationRunReason) -> NoReturn:
    raise EvaluationRunError(reason) from None


def _raise_cancelled() -> NoReturn:
    raise asyncio.CancelledError from None


def _categorical_signal(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if type(value) is not str or _SAFE_SIGNAL.fullmatch(value) is None:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    return value


def _policy_identity(classifier_id: object, policy_version: object) -> tuple[str, str]:
    if type(classifier_id) is not str or _SAFE_SIGNAL.fullmatch(classifier_id) is None:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    if type(policy_version) is not str or _SAFE_POLICY_VERSION.fullmatch(policy_version) is None:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    return classifier_id, policy_version


def _snapshot_vocabulary(raw_values: object) -> frozenset[str]:
    """Copy a bounded trusted vocabulary before processing any cases."""

    if type(raw_values) not in {tuple, list, set, frozenset}:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    values = cast(tuple[object, ...] | list[object] | set[object] | frozenset[object], raw_values)
    if len(values) > MAX_ASSESSMENT_SIGNALS:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    normalized = tuple(_categorical_signal(value) for value in values)
    if len(normalized) != len(set(normalized)):
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    return frozenset(normalized)


def _snapshot_cases(raw_cases: object) -> tuple[GoldenCase, ...]:
    """Bound and detach an exact built-in case container before iteration."""

    if type(raw_cases) not in {tuple, list}:
        _raise_run_error(EvaluationRunReason.INVALID_DATASET)
    cases = cast(tuple[object, ...] | list[object], raw_cases)
    if len(cases) > MAX_GOLDEN_CASES:
        _raise_run_error(EvaluationRunReason.TOO_MANY_CASES)
    if any(type(golden_case) is not GoldenCase for golden_case in cases):
        _raise_run_error(EvaluationRunReason.INVALID_DATASET)
    return cast(tuple[GoldenCase, ...], tuple(cases))


def _safe_error_reason(error: object, fallback: IndeterminateReason) -> IndeterminateReason:
    """Read an untrusted exception category without trusting its accessor."""

    try:
        reason = error.reason  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - exception subclasses are an integration boundary.
        return fallback
    return reason if isinstance(reason, IndeterminateReason) else fallback


def _safe_run_reason(error: EvaluationRunError) -> EvaluationRunReason:
    # Only this module's exact fixed-state exception is trusted. A classifier
    # can otherwise raise a hostile subclass with an accessor that executes.
    return error.reason if type(error) is EvaluationRunError else EvaluationRunReason.INVALID_ASSESSMENT


class EvaluationRunner:
    """Evaluate exactly one tuning or holdout split.

    Tuning reports include per-case diagnostics. Holdout diagnostics are sealed
    by default; opting into ``reveal_holdout_diagnostics`` enables adaptive
    inspection and repeated use invalidates the holdout as an unbiased set.
    Signal vocabularies are snapshotted from classifier configuration before
    any case runs, or from explicitly supplied trusted configuration.
    """

    def __init__(
        self,
        classifier: Classifier,
        *,
        split: EvaluationSplit,
        allowed_signals: Sequence[str] | None = None,
        allowed_review_signals: Sequence[str] | None = None,
        reveal_holdout_diagnostics: bool = False,
    ) -> None:
        failure: EvaluationRunReason | None = None
        cancelled = False
        normalized_split: EvaluationSplit | None = None
        signal_vocabulary: frozenset[str] = frozenset()
        review_vocabulary: frozenset[str] = frozenset()
        classifier_id = ""
        policy_version = ""
        configured_signals: object = allowed_signals
        configured_review: object = allowed_review_signals
        try:
            normalized_split = EvaluationSplit(split)
            classifier_id, policy_version = _policy_identity(
                classifier.classifier_id,
                classifier.policy_version,
            )
            if configured_signals is None:
                configured_signals = classifier.allowed_signals if isinstance(classifier, _HasAllowedSignals) else ()
            if configured_review is None:
                configured_review = (
                    classifier.allowed_review_signals if isinstance(classifier, _HasAllowedReviewSignals) else ()
                )
            signal_vocabulary = _snapshot_vocabulary(configured_signals)
            review_vocabulary = _snapshot_vocabulary(configured_review)
            if not isinstance(reveal_holdout_diagnostics, bool):
                _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
        except asyncio.CancelledError:
            cancelled = True
        except EvaluationRunError as error:
            failure = _safe_run_reason(error)
        except Exception:  # noqa: BLE001 - classifier configuration is an integration boundary.
            failure = EvaluationRunReason.INVALID_ASSESSMENT
        if cancelled or failure is not None or normalized_split is None:
            safe_cancelled = cancelled
            safe_failure = failure or EvaluationRunReason.INVALID_ASSESSMENT
            del (
                self,
                classifier,
                split,
                allowed_signals,
                allowed_review_signals,
                reveal_holdout_diagnostics,
                normalized_split,
                signal_vocabulary,
                review_vocabulary,
                classifier_id,
                policy_version,
                configured_signals,
                configured_review,
                cancelled,
            )
            if safe_cancelled:
                _raise_cancelled()
            _raise_run_error(safe_failure)
        self._classifier = classifier
        self._classifier_id = classifier_id
        self._policy_version = policy_version
        self._split = normalized_split
        self._allowed_signals = signal_vocabulary
        self._allowed_review_signals = review_vocabulary
        self._reveal_holdout_diagnostics = reveal_holdout_diagnostics
        if normalized_split is EvaluationSplit.HOLDOUT and reveal_holdout_diagnostics:
            warnings.warn(
                "revealing case diagnostics enables adaptive inspection; repeated use invalidates this holdout",
                HoldoutDiagnosticsWarning,
                stacklevel=2,
            )

    @property
    def split(self) -> EvaluationSplit:
        return self._split

    def run(self, cases: Sequence[GoldenCase]) -> EvaluationReport:
        failure: EvaluationRunReason | None = None
        configuration_extra: str | None = None
        cancelled = False
        bounded_cases: tuple[GoldenCase, ...] = ()
        selected: tuple[GoldenCase, ...] = ()
        results: tuple[CaseResult, ...] = ()
        report: EvaluationReport | None = None
        try:
            bounded_cases = _snapshot_cases(cases)
            selected = self._select_cases(bounded_cases)
            results = tuple(self._run_case(case) for case in selected)
            reveal_diagnostics = self.split is EvaluationSplit.TUNING or self._reveal_holdout_diagnostics
            # Report construction needs only categorical CaseResults. Release
            # raw inputs before crossing another validation boundary.
            selected = ()
            bounded_cases = ()
            cases = ()
            report = _build_report(
                self.split,
                self._classifier_id,
                self._policy_version,
                results,
                reveal_diagnostics=reveal_diagnostics,
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as error:
            fresh_error = _fresh_configuration_error(error)
            if fresh_error is None:
                failure = EvaluationRunReason.INVALID_ASSESSMENT
            else:
                configuration_extra = fresh_error.extra
        except EvaluationRunError as error:
            failure = _safe_run_reason(error)
        except Exception:  # noqa: BLE001 - sanitize every runner boundary.
            failure = EvaluationRunReason.INVALID_ASSESSMENT
        if cancelled:
            del self, cases, bounded_cases, selected, results, report, failure, configuration_extra
            _raise_cancelled()
        if configuration_extra is not None:
            safe_extra = configuration_extra
            del self, cases, bounded_cases, selected, results, report, failure, configuration_extra
            raise BackendConfigurationError(safe_extra) from None
        if failure is not None:
            safe_failure = failure
            del self, cases, bounded_cases, selected, results, report, configuration_extra
            _raise_run_error(safe_failure)
        if report is None:
            del self, cases, bounded_cases, selected, results
            _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
        return report

    def _select_cases(self, cases: Sequence[GoldenCase]) -> tuple[GoldenCase, ...]:
        if not cases:
            _raise_run_error(EvaluationRunReason.EMPTY_DATASET)
        try:
            audit_split_families(cases)
        except GoldenCaseLoadError:
            _raise_run_error(EvaluationRunReason.FAMILY_SPLIT_CONFLICT)
        case_ids: set[str] = set()
        selected: list[GoldenCase] = []
        for golden_case in cases:
            if golden_case.case_id in case_ids:
                _raise_run_error(EvaluationRunReason.DUPLICATE_CASE_ID)
            case_ids.add(golden_case.case_id)
            if golden_case.split is self.split:
                selected.append(golden_case)
        if not selected:
            _raise_run_error(EvaluationRunReason.EMPTY_SPLIT)
        review_labels = tuple(case.expected_review_boundary is not None for case in selected)
        if self._allowed_review_signals and not all(review_labels):
            _raise_run_error(EvaluationRunReason.INCOMPLETE_REVIEW_LABELS)
        if not self._allowed_review_signals and any(review_labels):
            _raise_run_error(EvaluationRunReason.MISSING_REVIEW_VOCABULARY)
        for golden_case in selected:
            if (
                golden_case.expected_signals is not None
                and not {signal.name for signal in golden_case.expected_signals} <= self._allowed_signals
            ):
                _raise_run_error(EvaluationRunReason.INVALID_EXPECTATION)
            if (
                golden_case.expected_review_signals is not None
                and not {signal.name for signal in golden_case.expected_review_signals} <= self._allowed_review_signals
            ):
                _raise_run_error(EvaluationRunReason.INVALID_EXPECTATION)
        return tuple(selected)

    def _run_case(self, golden_case: GoldenCase) -> CaseResult:
        assessments = self._assess(golden_case)
        boundaries = tuple(
            _boundary_result(
                golden_case,
                sensitivity,
                assessments[sensitivity],
            )
            for sensitivity in _SENSITIVITIES
        )
        typed_boundaries = (boundaries[0], boundaries[1], boundaries[2])
        monotonicity = _verify_monotonicity(
            (
                typed_boundaries[0].actual_outcome,
                typed_boundaries[1].actual_outcome,
                typed_boundaries[2].actual_outcome,
            ),
            (
                typed_boundaries[0].actual_signals,
                typed_boundaries[1].actual_signals,
                typed_boundaries[2].actual_signals,
            ),
        )
        review_monotonicity = (
            _verify_monotonicity(
                (
                    typed_boundaries[0].actual_review_outcome,
                    typed_boundaries[1].actual_review_outcome,
                    typed_boundaries[2].actual_review_outcome,
                ),
                (
                    typed_boundaries[0].actual_review_signals,
                    typed_boundaries[1].actual_review_signals,
                    typed_boundaries[2].actual_review_signals,
                ),
            )
            if golden_case.expected_review_boundary is not None
            else None
        )
        return CaseResult(
            classifier_id=self._classifier_id,
            policy_version=self._policy_version,
            case_id=golden_case.case_id,
            family_id=golden_case.family_id,
            split=golden_case.split,
            slices=golden_case.slices,
            expected_boundary=golden_case.expected_boundary,
            expected_review_boundary=golden_case.expected_review_boundary,
            outcome=_case_outcome(typed_boundaries, monotonicity, review_monotonicity),
            monotonicity=monotonicity,
            review_monotonicity=review_monotonicity,
            boundaries=typed_boundaries,
        )

    def _assess(self, golden_case: GoldenCase) -> dict[Sensitivity, _AssessmentSnapshot]:
        if isinstance(self._classifier, PolicyClassifier):
            return self._assess_policy(self._classifier, golden_case)
        return self._assess_ordinary(golden_case)

    def _assess_policy(
        self,
        classifier: PolicyClassifier[Any],
        golden_case: GoldenCase,
    ) -> dict[Sensitivity, _AssessmentSnapshot]:
        metadata = AssessmentMetadata()
        try:
            record = classifier.observe(golden_case.conversation)
        except BackendConfigurationError as error:
            fresh_error = _fresh_configuration_error(error)
            if fresh_error is not None:
                raise fresh_error from None
            return _failure_assessments(
                self._classifier_id,
                self._policy_version,
                IndeterminateReason.INTERNAL_ERROR,
                metadata,
            )
        except BackendError as error:
            return _failure_assessments(
                self._classifier_id,
                self._policy_version,
                _safe_error_reason(error, IndeterminateReason.PROVIDER_ERROR),
                metadata,
            )
        except TimeoutError:
            return _failure_assessments(
                self._classifier_id,
                self._policy_version,
                IndeterminateReason.TIMEOUT,
                metadata,
            )
        except Exception:  # noqa: BLE001 - untrusted classifier boundary.
            return _failure_assessments(
                self._classifier_id,
                self._policy_version,
                IndeterminateReason.INTERNAL_ERROR,
                metadata,
            )
        assessments: dict[Sensitivity, _AssessmentSnapshot] = {}
        for index, sensitivity in enumerate(_SENSITIVITIES):
            try:
                assessment = classifier.calibrate(record, sensitivity=sensitivity)
            except ClassificationError as error:
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        _safe_error_reason(error, IndeterminateReason.INTERNAL_ERROR),
                        metadata,
                    ),
                )
                break
            except BackendConfigurationError as error:
                fresh_error = _fresh_configuration_error(error)
                if fresh_error is not None:
                    raise fresh_error from None
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        IndeterminateReason.INTERNAL_ERROR,
                        metadata,
                    ),
                )
                break
            except BackendError as error:
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        _safe_error_reason(error, IndeterminateReason.PROVIDER_ERROR),
                        metadata,
                    ),
                )
                break
            except TimeoutError:
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        IndeterminateReason.TIMEOUT,
                        metadata,
                    ),
                )
                break
            except Exception:  # noqa: BLE001 - untrusted classifier boundary.
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        IndeterminateReason.INTERNAL_ERROR,
                        metadata,
                    ),
                )
                break
            snapshot = _snapshot_assessment(
                self._classifier_id,
                self._policy_version,
                sensitivity,
                assessment,
                self._allowed_signals,
                self._allowed_review_signals,
            )
            assessments[sensitivity] = snapshot
            metadata = snapshot.assessment.metadata
        return assessments

    def _assess_ordinary(self, golden_case: GoldenCase) -> dict[Sensitivity, _AssessmentSnapshot]:
        assessments: dict[Sensitivity, _AssessmentSnapshot] = {}
        metadata = AssessmentMetadata()
        for index, sensitivity in enumerate(_SENSITIVITIES):
            try:
                assessment = self._classifier.classify(golden_case.conversation, sensitivity=sensitivity)
            except ClassificationError as error:
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        _safe_error_reason(error, IndeterminateReason.INTERNAL_ERROR),
                        metadata,
                    ),
                )
                break
            except BackendConfigurationError as error:
                fresh_error = _fresh_configuration_error(error)
                if fresh_error is not None:
                    raise fresh_error from None
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        IndeterminateReason.INTERNAL_ERROR,
                        metadata,
                    ),
                )
                break
            except BackendError as error:
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        _safe_error_reason(error, IndeterminateReason.PROVIDER_ERROR),
                        metadata,
                    ),
                )
                break
            except TimeoutError:
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        IndeterminateReason.TIMEOUT,
                        metadata,
                    ),
                )
                break
            except Exception:  # noqa: BLE001 - untrusted classifier boundary.
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index:],
                        IndeterminateReason.INTERNAL_ERROR,
                        metadata,
                    ),
                )
                break
            snapshot = _snapshot_assessment(
                self._classifier_id,
                self._policy_version,
                sensitivity,
                assessment,
                self._allowed_signals,
                self._allowed_review_signals,
            )
            assessments[sensitivity] = snapshot
            metadata = snapshot.assessment.metadata
            if snapshot.assessment.outcome is Outcome.INDETERMINATE:
                reason = snapshot.assessment.indeterminate_reason or IndeterminateReason.INTERNAL_ERROR
                assessments.update(
                    _remaining_failure_assessments(
                        self._classifier_id,
                        self._policy_version,
                        _SENSITIVITIES[index + 1 :],
                        reason,
                        metadata,
                    ),
                )
                break
        return assessments


def _failure_assessments(
    classifier_id: str,
    policy_version: str,
    reason: IndeterminateReason,
    metadata: AssessmentMetadata,
) -> dict[Sensitivity, _AssessmentSnapshot]:
    return _remaining_failure_assessments(
        classifier_id,
        policy_version,
        _SENSITIVITIES,
        reason,
        metadata,
    )


def _remaining_failure_assessments(
    classifier_id: str,
    policy_version: str,
    sensitivities: Sequence[Sensitivity],
    reason: IndeterminateReason,
    metadata: AssessmentMetadata,
) -> dict[Sensitivity, _AssessmentSnapshot]:
    return {
        sensitivity: _AssessmentSnapshot(
            Assessment.indeterminate(
                classifier_id=classifier_id,
                policy_version=policy_version,
                sensitivity=sensitivity,
                reason=reason,
                metadata=metadata,
            ),
        )
        for sensitivity in sensitivities
    }


def _snapshot_metadata(value: object) -> AssessmentMetadata:
    """Copy only validated, non-sensitive execution labels."""

    if not isinstance(value, AssessmentMetadata):
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or set(state) != {"provider", "model"}:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    provider = state["provider"]
    model = state["model"]
    if (provider is not None and type(provider) is not str) or (model is not None and type(model) is not str):
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    try:
        return AssessmentMetadata(provider=provider, model=model)
    except Exception:  # noqa: BLE001 - convert malformed model state to a fixed error.
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)


def _snapshot_assessment(
    classifier_id: str,
    policy_version: str,
    sensitivity: Sensitivity,
    assessment: object,
    allowed_signals: frozenset[str],
    allowed_review_signals: frozenset[str],
) -> _AssessmentSnapshot:
    """Validate once and detach output before metrics consume it."""

    if not isinstance(assessment, Assessment):
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    state = object.__getattribute__(assessment, "__dict__")
    required = {
        "classifier_id",
        "policy_version",
        "sensitivity",
        "outcome",
        "evidence_directness",
        "signals",
        "indeterminate_reason",
        "metadata",
    }
    if type(state) is not dict or not required <= set(state):
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    observed_id, observed_version = _policy_identity(state["classifier_id"], state["policy_version"])
    raw_sensitivity = state["sensitivity"]
    outcome = state["outcome"]
    evidence_directness = state["evidence_directness"]
    reason = state["indeterminate_reason"]
    raw_signals = state["signals"]
    if (
        observed_id != classifier_id
        or observed_version != policy_version
        or type(raw_sensitivity) is not Sensitivity
        or raw_sensitivity is not sensitivity
        or type(outcome) is not Outcome
        or type(evidence_directness) is not EvidenceDirectness
        or (reason is not None and type(reason) is not IndeterminateReason)
        or type(raw_signals) is not tuple
        or len(raw_signals) > MAX_ASSESSMENT_SIGNALS
    ):
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    signals = tuple(_categorical_signal(value) for value in raw_signals)
    if len(signals) != len(set(signals)) or not set(signals) <= allowed_signals:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    metadata = _snapshot_metadata(state["metadata"])
    review_signals = _review_signals(assessment, allowed_review_signals)
    try:
        safe_assessment = Assessment(
            classifier_id=observed_id,
            policy_version=observed_version,
            sensitivity=raw_sensitivity,
            outcome=outcome,
            evidence_directness=evidence_directness,
            signals=signals,
            indeterminate_reason=reason,
            metadata=metadata,
        )
    except Exception:  # noqa: BLE001 - convert malformed model state to a fixed error.
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    return _AssessmentSnapshot(safe_assessment, review_signals)


def _review_signals(
    assessment: Assessment,
    allowed_review_signals: frozenset[str],
) -> tuple[str, ...]:
    raw_values: object = ()
    if isinstance(assessment, _HasReviewSignals):
        raw_values = assessment.review_signals
    elif isinstance(assessment, _HasEscalationReasons):
        raw_values = assessment.escalation_reasons
    if type(raw_values) not in {tuple, list, set, frozenset}:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    values = cast(tuple[object, ...] | list[object] | set[object] | frozenset[object], raw_values)
    if len(values) > MAX_ASSESSMENT_SIGNALS:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    normalized = tuple(_categorical_signal(value) for value in values)
    if len(normalized) != len(set(normalized)) or not set(normalized) <= allowed_review_signals:
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    return normalized


def _compare(
    expected: Outcome,
    actual: Outcome,
    expected_signals: tuple[str, ...] | None,
    actual_signals: tuple[str, ...],
) -> BoundaryOutcome:
    if expected is Outcome.INDETERMINATE:
        return (
            BoundaryOutcome.EXPECTED_INDETERMINATE
            if actual is Outcome.INDETERMINATE
            else BoundaryOutcome.UNEXPECTED_DECISION
        )
    if actual is Outcome.INDETERMINATE:
        return BoundaryOutcome.UNEXPECTED_INDETERMINATE
    if expected is Outcome.MATCHED:
        if actual is Outcome.NOT_MATCHED:
            return BoundaryOutcome.FALSE_NEGATIVE
        if expected_signals is not None and set(expected_signals) != set(actual_signals):
            return BoundaryOutcome.SIGNAL_MISMATCH
        return BoundaryOutcome.TRUE_POSITIVE
    return BoundaryOutcome.FALSE_POSITIVE if actual is Outcome.MATCHED else BoundaryOutcome.TRUE_NEGATIVE


def _boundary_result(
    golden_case: GoldenCase,
    sensitivity: Sensitivity,
    snapshot: _AssessmentSnapshot,
) -> BoundaryResult:
    assessment = snapshot.assessment
    expected = golden_case.expected_boundary.expected_outcome(sensitivity)
    expected_signals = golden_case.signals_at(sensitivity)
    actual_review_signals = snapshot.review_signals
    actual_review = (
        Outcome.INDETERMINATE
        if assessment.outcome is Outcome.INDETERMINATE
        else Outcome.MATCHED if actual_review_signals else Outcome.NOT_MATCHED
    )
    expected_review = (
        golden_case.expected_review_boundary.expected_outcome(sensitivity)
        if golden_case.expected_review_boundary is not None
        else None
    )
    expected_review_signals = golden_case.signals_at(sensitivity, review=True)
    return BoundaryResult(
        sensitivity=sensitivity,
        expected_outcome=expected,
        actual_outcome=assessment.outcome,
        outcome=_compare(expected, assessment.outcome, expected_signals, assessment.signals),
        expected_signals=expected_signals,
        actual_signals=assessment.signals,
        indeterminate_reason=assessment.indeterminate_reason,
        expected_review_outcome=expected_review,
        actual_review_outcome=actual_review,
        review_outcome=(
            _compare(
                expected_review,
                actual_review,
                expected_review_signals,
                actual_review_signals,
            )
            if expected_review is not None
            else None
        ),
        expected_review_signals=expected_review_signals,
        actual_review_signals=actual_review_signals,
    )


def _verify_monotonicity(
    outcomes: tuple[Outcome, Outcome, Outcome],
    signal_sets: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> MonotonicityStatus:
    decided = tuple(outcome is Outcome.MATCHED for outcome in outcomes)
    for earlier_index in range(len(outcomes) - 1):
        if outcomes[earlier_index] is not Outcome.MATCHED:
            continue
        if any(later is Outcome.NOT_MATCHED for later in outcomes[earlier_index + 1 :]):
            return MonotonicityStatus.VIOLATED
    for earlier_index in range(len(signal_sets) - 1):
        for later_index in range(earlier_index + 1, len(signal_sets)):
            if Outcome.INDETERMINATE in {outcomes[earlier_index], outcomes[later_index]}:
                continue
            if not set(signal_sets[earlier_index]) <= set(signal_sets[later_index]):
                return MonotonicityStatus.VIOLATED
    if any(outcome is Outcome.INDETERMINATE for outcome in outcomes):
        return MonotonicityStatus.INDETERMINATE
    return MonotonicityStatus.VERIFIED if decided == tuple(sorted(decided)) else MonotonicityStatus.VIOLATED


def _case_outcome(
    boundaries: tuple[BoundaryResult, BoundaryResult, BoundaryResult],
    monotonicity: MonotonicityStatus,
    review_monotonicity: MonotonicityStatus | None,
) -> CaseOutcome:
    if monotonicity is MonotonicityStatus.VIOLATED or review_monotonicity is MonotonicityStatus.VIOLATED:
        return CaseOutcome.MONOTONICITY_VIOLATION
    acceptable = {
        BoundaryOutcome.TRUE_POSITIVE,
        BoundaryOutcome.TRUE_NEGATIVE,
        BoundaryOutcome.EXPECTED_INDETERMINATE,
    }
    outcomes = [boundary.outcome for boundary in boundaries]
    outcomes.extend(boundary.review_outcome for boundary in boundaries if boundary.review_outcome is not None)
    failures = {outcome for outcome in outcomes if outcome not in acceptable}
    if not failures:
        if all(boundary.outcome is BoundaryOutcome.EXPECTED_INDETERMINATE for boundary in boundaries):
            return CaseOutcome.EXPECTED_INDETERMINATE
        return CaseOutcome.PASSED
    mapping = {
        BoundaryOutcome.FALSE_POSITIVE: CaseOutcome.FALSE_POSITIVE,
        BoundaryOutcome.FALSE_NEGATIVE: CaseOutcome.FALSE_NEGATIVE,
        BoundaryOutcome.UNEXPECTED_INDETERMINATE: CaseOutcome.UNEXPECTED_INDETERMINATE,
        BoundaryOutcome.UNEXPECTED_DECISION: CaseOutcome.UNEXPECTED_DECISION,
        BoundaryOutcome.SIGNAL_MISMATCH: CaseOutcome.SIGNAL_MISMATCH,
    }
    if len(failures) == 1:
        return mapping[next(iter(failures))]
    return CaseOutcome.MULTIPLE_FAILURES


def _metric_rows(
    results: Sequence[CaseResult],
    *,
    sensitivity: Sensitivity | None = None,
    review: bool = False,
) -> tuple[MetricRow, ...]:
    rows: list[MetricRow] = []
    for result in results:
        for boundary in result.boundaries:
            if sensitivity is not None and boundary.sensitivity is not sensitivity:
                continue
            if review:
                if boundary.expected_review_outcome is None:
                    continue
                rows.append(
                    MetricRow(
                        expected=boundary.expected_review_outcome,
                        actual=boundary.actual_review_outcome,
                        expected_signals=boundary.expected_review_signals,
                        actual_signals=boundary.actual_review_signals,
                    ),
                )
            else:
                rows.append(
                    MetricRow(
                        expected=boundary.expected_outcome,
                        actual=boundary.actual_outcome,
                        expected_signals=boundary.expected_signals,
                        actual_signals=boundary.actual_signals,
                    ),
                )
    return tuple(rows)


def _optional_review_metrics(
    results: Sequence[CaseResult],
    *,
    sensitivity: Sensitivity | None = None,
) -> MetricSummary | None:
    rows = _metric_rows(results, sensitivity=sensitivity, review=True)
    return summarize(rows) if rows else None


def _sensitivity_reports(
    results: Sequence[CaseResult],
) -> tuple[SensitivityReport, SensitivityReport, SensitivityReport]:
    reports = tuple(
        SensitivityReport(
            sensitivity=sensitivity,
            metrics=summarize(_metric_rows(results, sensitivity=sensitivity)),
            review_metrics=_optional_review_metrics(results, sensitivity=sensitivity),
        )
        for sensitivity in _SENSITIVITIES
    )
    return reports[0], reports[1], reports[2]


def _build_report(
    split: EvaluationSplit,
    classifier_id: str,
    policy_version: str,
    results: tuple[CaseResult, ...],
    *,
    reveal_diagnostics: bool,
) -> EvaluationReport:
    slices = sorted({slice_name for result in results for slice_name in result.slices}) if reveal_diagnostics else []
    slice_reports: list[SliceReport] = []
    for slice_name in slices:
        selected = tuple(result for result in results if slice_name in result.slices)
        slice_reports.append(
            SliceReport(
                slice=slice_name,
                case_count=len(selected),
                metrics=summarize(_metric_rows(selected)),
                review_metrics=_optional_review_metrics(selected),
                by_sensitivity=_sensitivity_reports(selected),
            ),
        )
    return EvaluationReport(
        classifier_id=classifier_id,
        policy_version=policy_version,
        split=split,
        case_count=len(results),
        diagnostics_revealed=reveal_diagnostics,
        case_results=results if reveal_diagnostics else (),
        metrics=summarize(_metric_rows(results)),
        review_metrics=_optional_review_metrics(results),
        by_sensitivity=_sensitivity_reports(results),
        slices=tuple(slice_reports),
        monotonicity_verified=sum(result.monotonicity is MonotonicityStatus.VERIFIED for result in results),
        monotonicity_violations=sum(result.monotonicity is MonotonicityStatus.VIOLATED for result in results),
        monotonicity_indeterminate=sum(result.monotonicity is MonotonicityStatus.INDETERMINATE for result in results),
        review_monotonicity_violations=sum(
            result.review_monotonicity is MonotonicityStatus.VIOLATED for result in results
        ),
    )


def run_evaluation(
    classifier: Classifier,
    cases: Sequence[GoldenCase],
    *,
    split: EvaluationSplit,
    allowed_signals: Sequence[str] | None = None,
    allowed_review_signals: Sequence[str] | None = None,
    reveal_holdout_diagnostics: bool = False,
) -> EvaluationReport:
    """Evaluate one split with categorical output and sealed holdout details.

    Setting ``reveal_holdout_diagnostics`` to true exposes per-case holdout
    results. Repeated inspection then invalidates that split as an unbiased
    holdout. Vocabulary overrides must come from trusted configuration.
    """

    failure: EvaluationRunReason | None = None
    configuration_extra: str | None = None
    cancelled = False
    report: EvaluationReport | None = None
    runner: EvaluationRunner | None = None
    try:
        runner = EvaluationRunner(
            classifier,
            split=split,
            allowed_signals=allowed_signals,
            allowed_review_signals=allowed_review_signals,
            reveal_holdout_diagnostics=reveal_holdout_diagnostics,
        )
        report = runner.run(cases)
    except asyncio.CancelledError:
        cancelled = True
    except BackendConfigurationError as error:
        fresh_error = _fresh_configuration_error(error)
        if fresh_error is None:
            failure = EvaluationRunReason.INVALID_ASSESSMENT
        else:
            configuration_extra = fresh_error.extra
    except EvaluationRunError as error:
        failure = _safe_run_reason(error)
    except Exception:  # noqa: BLE001 - sanitize every public runner boundary.
        failure = EvaluationRunReason.INVALID_ASSESSMENT
    if cancelled:
        del (
            classifier,
            cases,
            split,
            allowed_signals,
            allowed_review_signals,
            reveal_holdout_diagnostics,
            runner,
            report,
            failure,
            configuration_extra,
        )
        _raise_cancelled()
    if configuration_extra is not None:
        safe_extra = configuration_extra
        del (
            classifier,
            cases,
            split,
            allowed_signals,
            allowed_review_signals,
            reveal_holdout_diagnostics,
            runner,
            report,
            failure,
            configuration_extra,
        )
        raise BackendConfigurationError(safe_extra) from None
    if failure is not None:
        safe_failure = failure
        del (
            classifier,
            cases,
            split,
            allowed_signals,
            allowed_review_signals,
            reveal_holdout_diagnostics,
            runner,
            report,
            configuration_extra,
        )
        _raise_run_error(safe_failure)
    if report is None:
        del classifier, cases, split, allowed_signals, allowed_review_signals, reveal_holdout_diagnostics, runner
        _raise_run_error(EvaluationRunReason.INVALID_ASSESSMENT)
    return report


__all__ = [
    "EvaluationRunError",
    "EvaluationRunReason",
    "EvaluationRunner",
    "HoldoutDiagnosticsWarning",
    "run_evaluation",
]
