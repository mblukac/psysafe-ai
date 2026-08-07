"""Attach PsySafe gates to native OpenAI Agents SDK boundaries."""

from __future__ import annotations

import asyncio
import os

from agents import Agent, RunConfig, Runner, function_tool

from psysafe import AsyncWorkflowGate, Checkpoint, GateAction, GatePolicy, PIIClassifier, Sensitivity
from psysafe.integrations.openai_agents import (
    openai_input_guardrail,
    openai_output_guardrail,
    openai_tool_input_guardrail,
    openai_tool_output_guardrail,
)


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} before running this example.")
    return value


def gate_for(checkpoint: Checkpoint) -> AsyncWorkflowGate:
    return AsyncWorkflowGate(
        (PIIClassifier(),),
        checkpoint=checkpoint,
        policy=GatePolicy(
            matched_action=GateAction.BLOCK,
            indeterminate_action=GateAction.REVIEW,
        ),
    )


async def main() -> None:
    sensitivity = Sensitivity.BALANCED
    input_check = openai_input_guardrail(
        gate_for(Checkpoint.INPUT),
        sensitivity=sensitivity,
    )
    communication_check = openai_output_guardrail(
        gate_for(Checkpoint.COMMUNICATION),
        sensitivity=sensitivity,
    )
    tool_input_check = openai_tool_input_guardrail(
        gate_for(Checkpoint.TOOL_INPUT),
        sensitivity=sensitivity,
    )
    tool_output_check = openai_tool_output_guardrail(
        gate_for(Checkpoint.TOOL_OUTPUT),
        sensitivity=sensitivity,
    )

    @function_tool(
        tool_input_guardrails=[tool_input_check],
        tool_output_guardrails=[tool_output_check],
    )
    def lookup_term(term: str) -> str:
        """Return a definition from a small, side-effect-free glossary."""

        glossary = {
            "calibration": "Applying an explicit policy boundary to a structured observation.",
            "indeterminate": "A failed or incomplete check that requires a non-allow route.",
        }
        return glossary.get(term, "No glossary entry is available.")

    agent = Agent(
        name="Safety-aware explainer",
        model=required_environment("OPENAI_AGENT_MODEL"),
        instructions="Use the glossary tool, then explain the selected concept plainly.",
        tools=[lookup_term],
        input_guardrails=[input_check],
        output_guardrails=[communication_check],
    )

    # Keep sensitive trace payloads disabled. Output guardrails are not
    # streaming filters, so expose content only after this guarded run returns.
    result = await Runner.run(
        agent,
        "Explain calibration.",
        run_config=RunConfig(trace_include_sensitive_data=False),
    )
    print(result.final_output)

    # Tool output checks happen after execution and cannot undo side effects.
    # Use tool input checks for prevention. These native tool guardrails cover
    # custom function tools, not hosted tools, MCP, handoffs, or Agent.as_tool().


if __name__ == "__main__":
    asyncio.run(main())
