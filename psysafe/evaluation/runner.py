# psysafe/evaluation/runner.py
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from psysafe.catalog import GuardrailCatalog
from psysafe.drivers.base import ChatDriverABC
from psysafe.evaluation.metrics import placeholder_accuracy_metric  # Example metric
from psysafe.evaluation.models import EvaluationResult, MetricResult, TestCase  # Corrected import


class EvaluationRunner:
    def __init__(self, driver: ChatDriverABC, catalog: type[GuardrailCatalog]):
        self.driver = driver
        self.catalog = catalog

    def load_test_cases(self, file_path: str | Path) -> list[TestCase]:
        """Loads test cases from a JSONL file."""
        test_cases = []
        with Path(file_path).open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    test_cases.append(TestCase(**data))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line in {file_path}: {e} - Line: {line.strip()}")
                except Exception as e:  # Catch Pydantic validation errors etc.
                    print(f"Skipping invalid test case in {file_path}: {e} - Line: {line.strip()}")
        return test_cases

    def run_evaluation(
        self,
        guardrail_name: str,
        test_cases: list[TestCase],
        guardrail_init_kwargs: dict[str, Any] | None = None,
    ) -> list[EvaluationResult]:
        """
        Runs evaluation for a given guardrail against a list of test cases.
        (Placeholder implementation).
        """
        results: list[EvaluationResult] = []
        if guardrail_init_kwargs is None:
            guardrail_init_kwargs = {}

        try:
            # Load the guardrail instance
            # Assuming load returns a list, take the first one for a single guardrail eval
            guardrail_instances = self.catalog.load(guardrail_name, **guardrail_init_kwargs)
            if not guardrail_instances:
                raise ValueError(f"Could not load guardrail: {guardrail_name}")
            guardrail = guardrail_instances[0]
        except Exception as e:
            print(f"Failed to load guardrail {guardrail_name}: {e}")
            # Return empty results or results indicating failure to load
            for tc in test_cases:
                results.append(
                    EvaluationResult(
                        guardrail_name=guardrail_name,
                        test_case_id=tc.id,
                        passed=False,
                        details=f"Failed to load guardrail: {e}",
                    ),
                )
            return results

        for tc in test_cases:
            print(f"Running test case: {tc.id} for guardrail: {guardrail_name}")
            passed_test = False
            eval_details = ""
            metrics_results: list[MetricResult] = []
            actual_report = None

            try:
                # 1. Apply guardrail to input request (input transformation part)
                # The input_request in TestCase should be compatible with the driver
                guardrail.apply(tc.input_request)

                # 2. Send modified request through the driver to get an LLM response
                # This is a simplification. The actual LLM call might be mocked or
                # a dummy response might be used if the guardrail primarily modifies requests
                # or if the test case is focused on input transformation.
                # For guardrails that validate LLM output, an actual/mocked LLM call is needed.

                # Let's assume for now the guardrail's main job is input transformation
                # or the test case provides an expected LLM output to validate against.
                # If tc.expected_outcome has a simulated LLM response, use that.
                # Otherwise, this part needs a call to self.driver.send(guarded_request_obj.modified_request)

                # For now, let's simulate a dummy LLM response if not focused on actual LLM call
                simulated_llm_response_content = (
                    tc.expected_outcome.get("simulated_llm_response_content", "Placeholder LLM response.")
                    if tc.expected_outcome
                    else "Placeholder LLM response."
                )

                # Construct a dummy response object that matches what the guardrail.validate() expects.
                # This needs to align with the ResponseT of the guardrail.
                # If guardrail expects OpenAIChatResponse:
                dummy_llm_response_for_validation = {
                    "choices": [{"message": {"role": "assistant", "content": simulated_llm_response_content}}],
                    # Add other fields if the specific guardrail's validate method depends on them
                }

                # 3. Validate the (dummy or actual) LLM response
                actual_report = guardrail.validate(dummy_llm_response_for_validation)  # type: ignore

                # 4. Compare with expected outcome / validation report
                if tc.expected_validation_report:
                    # Basic check:
                    passed_test = tc.expected_validation_report.is_valid == actual_report.is_valid
                    # Add metric calculation
                    metrics_results.append(
                        placeholder_accuracy_metric(tc.expected_validation_report, actual_report),
                    )
                    if not passed_test:
                        eval_details = (
                            f"Expected is_valid={tc.expected_validation_report.is_valid}, got {actual_report.is_valid}"
                        )
                    # More detailed comparison of violations could be added here.
                else:
                    # If no expected_validation_report, consider test passed if no errors.
                    # Or, this means the test case is not designed for validation outcome.
                    passed_test = True  # Placeholder logic
                    eval_details = "No expected_validation_report provided for comparison."

            except Exception as e:
                passed_test = False
                eval_details = f"Error during evaluation: {e}"
                print(f"Error evaluating test case {tc.id}: {e}")

            results.append(
                EvaluationResult(
                    guardrail_name=guardrail_name,
                    test_case_id=tc.id,
                    passed=passed_test,
                    actual_validation_report=actual_report,
                    metrics=metrics_results,
                    details=eval_details,
                ),
            )
        return results


# Example usage (for testing the runner itself, not part of SDK usage)
if __name__ == "__main__":
    # This is a very basic example and assumes certain files/structures exist
    # It also needs a dummy driver and catalog setup.
    print("EvaluationRunner basic structure. Needs proper setup for example run.")

    # Create a dummy driver
    class DummyDriver(ChatDriverABC):
        def send(self, request: Any) -> Any:
            return {"choices": [{"message": {"content": "dummy response"}}]}

        async def stream(self, request: Any) -> AsyncIterator[Any]:
            yield {"choices": [{"message": {"content": "dummy stream"}}]}

        def get_metadata(self) -> dict[str, Any]:
            return {"type": "dummy"}
