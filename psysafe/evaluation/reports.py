# psysafe/evaluation/reports.py
from typing import List
from psysafe.evaluation.models import EvaluationResult

def generate_summary_report(results: List[EvaluationResult]) -> str:
    """Generates a simple text summary of evaluation results."""
    if not results:
        return "No evaluation results to report."

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    summary = f"Evaluation Summary:\n"
    summary += f"--------------------\n"
    summary += f"Total Tests: {len(results)}\n"
    summary += f"Passed: {passed_count}\n"
    summary += f"Failed: {failed_count}\n\n"

    summary += "Detailed Results:\n"
    for res in results:
        summary += (
            f"- Test Case: {res.test_case_id} (Guardrail: {res.guardrail_name}) - "
            f"{'PASSED' if res.passed else 'FAILED'}\n"
        )
        if res.details:
            summary += f"  Details: {res.details}\n"
        for metric in res.metrics:
            summary += f"  Metric: {metric.metric_name} = {metric.value}\n"
    return summary

# Future: Add functions for HTML reports, JSON reports, etc.