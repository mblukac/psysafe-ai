# Contributing to PsySafe-AI

PsySafe-AI provides calibrated, categorical safety checks for the boundaries where AI systems act. Contributions
should keep that contract clear, testable, and appropriately limited.

## Before you start

- Search existing issues before opening a new one. For larger behavior or API changes, start with a feature request.
- Never paste real user content, raw prompts, model responses, logs, traces, credentials, identifiers, or health data
  into an issue, pull request, test, or fixture. Use synthetic, minimized examples.
- Report security vulnerabilities through the private process in [SECURITY.md](SECURITY.md), not a public issue.

## Design contract

Classifier and workflow changes should preserve these principles:

- Report observable policy signals about content or actions, not diagnoses or traits about people.
- Return categorical outcomes: `matched`, `not_matched`, or `indeterminate`.
- Calibrate with the named `precise`, `balanced`, and `precautionary` sensitivity boundaries.
- Do not add per-item confidence scores, numerical ratings, clinical risk estimates, or hidden chain-of-thought.
- Keep detection separate from the application's `allow`, `review`, or `block` decision.
- Treat `indeterminate` as requiring review or blocking; it must not silently become `allow`.
- Do not claim that the package prevents harm, makes clinical decisions, or establishes legal or regulatory compliance.

## Development

Install uv 0.12.2, then install the locked project and development dependencies:

```bash
uv sync --locked --all-extras --group dev
uv run pre-commit install
```

Before submitting a pull request, run:

```bash
uv run pytest --cov=psysafe --cov-report=term-missing --cov-fail-under=85
uv run ruff check .
uv run black --check .
uv run mypy psysafe examples
uv run python examples/sequential_gate.py
uv run python examples/categorical_evaluation.py
uv run pre-commit run --all-files
uv lock --check
uv build --no-sources --build-constraint build-constraints.txt --require-hashes
```

Add focused tests for behavior changes. Changes to classifiers, policies, prompts, or calibration also need evaluation
on representative tuning and held-out cases. Keep related case families in one split, cover relevant slices, check
`indeterminate` behavior, and verify that sensitivity changes are monotonic. Report aggregate evaluation results only;
do not include sensitive examples.

Update user-facing documentation and migration guidance when behavior, boundaries, dependencies, or public APIs
change. Keep commits focused and explain any compatibility or fail-safe implications in the pull request.
