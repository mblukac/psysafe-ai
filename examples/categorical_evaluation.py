"""Evaluate named sensitivity boundaries with categorical golden cases."""

from __future__ import annotations

from pathlib import Path

from psysafe import PIIClassifier
from psysafe.evaluation import EvaluationSplit, load_golden_cases, run_evaluation


def main() -> None:
    cases = load_golden_cases(Path(__file__).with_name("golden_cases.jsonl"))
    report = run_evaluation(
        PIIClassifier(),
        cases,
        split=EvaluationSplit.TUNING,
    )

    for result in report.case_results:
        print(
            {
                "case": result.case_id,
                "outcome": result.outcome.value,
                "monotonicity": result.monotonicity.value,
            },
        )


if __name__ == "__main__":
    main()
