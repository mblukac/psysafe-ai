"""Route immutable artifacts through a sequential workflow."""

from __future__ import annotations

from psysafe import (
    Checkpoint,
    GateAction,
    GatePolicy,
    PIIClassifier,
    Sensitivity,
    WorkflowGate,
)


def gate_for(checkpoint: Checkpoint) -> WorkflowGate:
    return WorkflowGate(
        (PIIClassifier(),),
        checkpoint=checkpoint,
        policy=GatePolicy(
            matched_action=GateAction.BLOCK,
            indeterminate_action=GateAction.REVIEW,
        ),
    )


def require_allow(
    gate: WorkflowGate,
    artifact: str,
    *,
    artifact_id: str,
) -> str:
    """Check the exact artifact version that the next stage will consume."""

    decision = gate.evaluate_text(
        artifact,
        artifact_id=artifact_id,
        sensitivity=Sensitivity.BALANCED,
    )
    print({"checkpoint": decision.checkpoint.value, "action": decision.action.value})

    if decision.action is GateAction.REVIEW:
        raise SystemExit("Route this artifact to the application's review path.")
    if decision.action is GateAction.BLOCK:
        raise SystemExit("Stop this workflow under the configured routing policy.")
    return artifact


def select_task(user_input: str) -> str:
    del user_input
    return "summarize_release_notes"


def build_execution_plan(task: str) -> str:
    return f"Read the supplied release notes and execute task {task}."


def execute(plan: str) -> str:
    del plan
    return "The release notes describe calibrated safety checks for workflow boundaries."


def compose_communication(result: str) -> str:
    return f"Summary: {result}"


def main() -> None:
    # Artifact IDs identify exact immutable versions; consume each decision
    # immediately so a different value cannot be substituted after checking.
    user_input = require_allow(
        gate_for(Checkpoint.INPUT),
        "Summarize the supplied release notes.",
        artifact_id="request:input:v1",
    )
    task = require_allow(
        gate_for(Checkpoint.TASK_SELECTION),
        select_task(user_input),
        artifact_id="request:task-selection:v1",
    )
    plan = require_allow(
        gate_for(Checkpoint.EXECUTION),
        build_execution_plan(task),
        artifact_id="request:execution-plan:v1",
    )
    result = execute(plan)
    communication = require_allow(
        gate_for(Checkpoint.COMMUNICATION),
        compose_communication(result),
        artifact_id="request:communication:v1",
    )
    print(communication)


if __name__ == "__main__":
    main()
