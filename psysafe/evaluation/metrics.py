"""Aggregate categorical evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from psysafe.core.contracts import Outcome
from psysafe.evaluation.models import MetricSummary


@dataclass(frozen=True, slots=True)
class MetricRow:
    """Internal value-free row used to aggregate a report."""

    expected: Outcome
    actual: Outcome
    expected_signals: tuple[str, ...] | None
    actual_signals: tuple[str, ...]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: Iterable[MetricRow]) -> MetricSummary:
    """Aggregate rows using the denominators documented by ``MetricSummary``."""

    values = tuple(rows)
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    positive_indeterminate = 0
    negative_indeterminate = 0
    expected_indeterminate = 0
    unexpected_decision = 0
    actual_matched = 0
    actual_not_matched = 0
    actual_indeterminate = 0
    signal_expectations = 0
    signal_matches = 0

    for row in values:
        if row.actual is Outcome.MATCHED:
            actual_matched += 1
        elif row.actual is Outcome.NOT_MATCHED:
            actual_not_matched += 1
        else:
            actual_indeterminate += 1

        if row.expected is Outcome.INDETERMINATE:
            expected_indeterminate += 1
            if row.actual is not Outcome.INDETERMINATE:
                unexpected_decision += 1
        elif row.expected is Outcome.MATCHED:
            if row.actual is Outcome.MATCHED:
                true_positive += 1
            elif row.actual is Outcome.NOT_MATCHED:
                false_negative += 1
            else:
                positive_indeterminate += 1
        elif row.actual is Outcome.MATCHED:
            false_positive += 1
        elif row.actual is Outcome.NOT_MATCHED:
            true_negative += 1
        else:
            negative_indeterminate += 1

        if row.expected is Outcome.MATCHED and row.expected_signals is not None:
            signal_expectations += 1
            if row.actual is Outcome.MATCHED and set(row.expected_signals) == set(row.actual_signals):
                signal_matches += 1

    expected_binary = len(values) - expected_indeterminate
    decided_binary = true_positive + true_negative + false_positive + false_negative
    signal_mismatches = signal_expectations - signal_matches
    return MetricSummary(
        evaluations=len(values),
        expected_binary=expected_binary,
        expected_indeterminate=expected_indeterminate,
        decided_binary=decided_binary,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        positive_indeterminate=positive_indeterminate,
        negative_indeterminate=negative_indeterminate,
        unexpected_decision=unexpected_decision,
        actual_matched=actual_matched,
        actual_not_matched=actual_not_matched,
        actual_indeterminate=actual_indeterminate,
        signal_expectations=signal_expectations,
        signal_matches=signal_matches,
        signal_mismatches=signal_mismatches,
        coverage=_ratio(decided_binary, expected_binary),
        precision=_ratio(true_positive, true_positive + false_positive),
        recall=_ratio(true_positive, true_positive + false_negative),
        effective_recall=_ratio(
            true_positive,
            true_positive + false_negative + positive_indeterminate,
        ),
        false_positive_rate=_ratio(false_positive, false_positive + true_negative),
        false_negative_rate=_ratio(false_negative, false_negative + true_positive),
        indeterminate_rate=_ratio(actual_indeterminate, len(values)),
        intervention_rate=_ratio(actual_matched + actual_indeterminate, len(values)),
        signal_match_rate=_ratio(signal_matches, signal_expectations),
    )


__all__ = ["MetricRow", "summarize"]
