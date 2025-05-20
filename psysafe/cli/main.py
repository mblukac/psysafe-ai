# psysafe/cli/main.py
import typer
from typing import Optional
from typing_extensions import Annotated
from psysafe.cli import guard # Import the guard subcommand module

# It's good practice to initialize the catalog somewhere central if guardrails
# need to be available upon CLI startup, or ensure guardrail modules are imported.
# For now, guardrail modules register themselves upon import.
# We might need to explicitly import catalog modules if they aren't automatically.
# Example: from psysafe.catalog.vulnerability_detection import guardrail as vd_guardrail_module
# This ensures VulnerabilityDetectionGuardrail is registered.
# A better way would be a plugin system or explicit registration call.
# For now, direct imports in guard.py or here might be needed if catalog is empty.

# To ensure guardrails are registered when CLI starts, we can import their modules.
# This is a simple way; a plugin system (e.g., entry_points) is more robust for extensibility.
try:
    from psysafe.catalog.vulnerability_detection.guardrail import VulnerabilityDetectionGuardrail
    from psysafe.catalog.suicide_prevention.guardrail import SuicidePreventionGuardrail
    # These imports ensure the @GuardrailCatalog.register lines in those files are executed.
except ImportError as e:
    print(f"Warning: Could not pre-load all catalog guardrails for CLI: {e}")


app = typer.Typer(
    name="psysafe",
    help="PsySafe AI SDK - Command Line Interface for managing and applying psychological safety guardrails.",
    add_completion=False # Can be enabled if desired
)

# Add subcommands
app.add_typer(guard.app, name="guard")

# Create a new Typer app for test commands
test_app = typer.Typer(help="Commands for running evaluations and tests.")

@test_app.command("run")
def run_tests(
    guardrail_name: Annotated[str, typer.Option(help="Name of the guardrail to test (or 'all').")] = "all",
    test_file: Annotated[Optional[str], typer.Option(help="Path to a specific JSONL test case file.")] = None,
    # Add more options like output directory for reports, etc.
):
    """
    Runs evaluations for specified guardrails using golden examples.
    (This is a placeholder - actual implementation will use EvaluationRunner)
    """
    print(f"Placeholder: Running tests for guardrail: '{guardrail_name}'")
    if test_file:
        print(f"Using test file: {test_file}")
    # Future:
    # 1. Initialize a driver (e.g., OpenAIChatDriver or a DummyDriver for some tests)
    # 2. Initialize EvaluationRunner(driver=driver, catalog=GuardrailCatalog)
    # 3. If guardrail_name is 'all', iterate GuardrailCatalog.list_available()
    # 4. For each guardrail:
    #    a. Construct path to its examples.jsonl (e.g., psysafe/catalog/{name}/examples.jsonl) or use test_file
    #    b. Load test cases: runner.load_test_cases(...)
    #    c. Run evaluation: runner.run_evaluation(guardrail_name, test_cases, guardrail_init_kwargs={...})
    #       (Need a way to get default/test-specific init_kwargs for each guardrail)
    #    d. Generate and print/save report: reports.generate_summary_report(...)
    print("CLI 'test run' command needs full implementation.")

app.add_typer(test_app, name="test")
# Future: app.add_typer(report_app, name="report")


@app.command()
def version():
    """
    Displays the version of the PsySafe AI SDK.
    """
    # Placeholder for version. This would typically come from __version__ in psysafe/__init__.py
    try:
        from psysafe import __version__ as sdk_version
        print(f"PsySafe AI SDK version: {sdk_version}")
    except ImportError:
        print("PsySafe AI SDK version: (unknown - __version__ not found)")

if __name__ == "__main__":
    app()