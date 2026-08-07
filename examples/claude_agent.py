"""Attach PsySafe gates to native Claude Agent SDK hooks."""

from __future__ import annotations

import asyncio
import os
import secrets

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from psysafe import AsyncWorkflowGate, Checkpoint, GateAction, GatePolicy, PIIClassifier, Sensitivity
from psysafe.integrations.claude_agent import claude_agent_hooks


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
    communication_gate = gate_for(Checkpoint.COMMUNICATION)
    hooks = claude_agent_hooks(
        user_prompt_submit_gate=gate_for(Checkpoint.INPUT),
        pre_tool_use_gate=gate_for(Checkpoint.TOOL_INPUT),
        post_tool_use_gate=gate_for(Checkpoint.TOOL_OUTPUT),
        sensitivity=sensitivity,
    )
    options = ClaudeAgentOptions(
        model=required_environment("CLAUDE_AGENT_MODEL"),
        hooks=hooks,
        tools=[],
        strict_mcp_config=True,
    )

    async for message in query(
        prompt="Explain why an indeterminate safety check needs an explicit route.",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            if (
                message.is_error
                or message.subtype != "success"
                or message.terminal_reason not in (None, "completed")
                or message.result is None
            ):
                raise RuntimeError("Claude Agent run did not produce a completed final result.")
            # Claude hooks do not cover final assistant output. Ignore
            # intermediate AssistantMessage events and expose only a final
            # result that passes an explicit communication gate.
            decision = await communication_gate.aevaluate_text(
                message.result,
                artifact_id=f"claude-result:{secrets.token_hex(16)}",
                sensitivity=sensitivity,
            )
            if not decision.is_allowed:
                raise RuntimeError(f"PsySafe routed final output to {decision.action.value}; do not display it.")
            print(message.result)

    # This minimal run disables tools. When adding them, keep Claude's normal
    # permission flow enabled: an ALLOW from the pre-tool hook does not grant
    # permission. A post-tool hook can stop subsequent work but cannot undo an
    # external side effect, so put preventive policy at PreToolUse.
    # The SDK hooks also have no final-output boundary; buffer the run and gate
    # ResultMessage.result before display, as above.


if __name__ == "__main__":
    asyncio.run(main())
