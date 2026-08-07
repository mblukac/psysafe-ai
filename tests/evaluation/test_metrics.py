"""Synthetic metric-contract tests; they make no classifier-quality claim."""

import pytest

from psysafe.core.contracts import Outcome
from psysafe.evaluation.metrics import MetricRow, summarize


def _row(
    expected: Outcome,
    actual: Outcome,
    *,
    expected_signals: tuple[str, ...] | None = None,
    actual_signals: tuple[str, ...] = (),
) -> MetricRow:
    return MetricRow(
        expected=expected,
        actual=actual,
        expected_signals=expected_signals,
        actual_signals=actual_signals,
    )


def test_metric_denominators_distinguish_selective_and_effective_recall() -> None:
    summary = summarize(
        (
            _row(Outcome.MATCHED, Outcome.MATCHED, expected_signals=("signal_a",), actual_signals=("signal_a",)),
            _row(Outcome.MATCHED, Outcome.NOT_MATCHED, expected_signals=("signal_a",)),
            # Even equal signal sets cannot be a match when the decision is
            # indeterminate (relevant to independent review-signal models).
            _row(
                Outcome.MATCHED,
                Outcome.INDETERMINATE,
                expected_signals=("signal_a",),
                actual_signals=("signal_a",),
            ),
            _row(Outcome.NOT_MATCHED, Outcome.MATCHED, actual_signals=("signal_a",)),
            _row(Outcome.NOT_MATCHED, Outcome.NOT_MATCHED),
            _row(Outcome.NOT_MATCHED, Outcome.INDETERMINATE),
            _row(Outcome.INDETERMINATE, Outcome.INDETERMINATE),
            _row(Outcome.INDETERMINATE, Outcome.MATCHED, actual_signals=("signal_a",)),
        ),
    )

    assert summary.evaluations == 8
    assert summary.expected_binary == 6
    assert summary.expected_indeterminate == 2
    assert summary.decided_binary == 4
    assert summary.true_positive == 1
    assert summary.true_negative == 1
    assert summary.false_positive == 1
    assert summary.false_negative == 1
    assert summary.positive_indeterminate == 1
    assert summary.negative_indeterminate == 1
    assert summary.unexpected_decision == 1
    assert summary.actual_matched == 3
    assert summary.actual_not_matched == 2
    assert summary.actual_indeterminate == 3
    assert summary.coverage == pytest.approx(4 / 6)
    assert summary.precision == 0.5
    assert summary.recall == 0.5
    assert summary.effective_recall == pytest.approx(1 / 3)
    assert summary.false_positive_rate == 0.5
    assert summary.false_negative_rate == 0.5
    assert summary.indeterminate_rate == pytest.approx(3 / 8)
    assert summary.intervention_rate == pytest.approx(6 / 8)
    assert summary.signal_expectations == 3
    assert summary.signal_matches == 1
    assert summary.signal_mismatches == 2
    assert summary.signal_match_rate == pytest.approx(1 / 3)


def test_metric_rates_are_none_when_their_denominators_are_undefined() -> None:
    summary = summarize((_row(Outcome.INDETERMINATE, Outcome.INDETERMINATE),))

    assert summary.coverage is None
    assert summary.precision is None
    assert summary.recall is None
    assert summary.effective_recall is None
    assert summary.false_positive_rate is None
    assert summary.false_negative_rate is None
    assert summary.signal_match_rate is None
    assert summary.indeterminate_rate == 1.0
    assert summary.intervention_rate == 1.0
