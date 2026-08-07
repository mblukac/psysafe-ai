# Workflow gates

Safety checks are most useful at the boundary immediately before an application acts. PsySafe supports framework-neutral sequential orchestration and native agent lifecycle integration with the same categorical gate contract.

## Sequential orchestration

A common workflow is:

```text
input → task selection → execution → communication
                         ↘ tool input → tool → tool output ↗
```

PsySafe represents those locations with six checkpoints:

| Checkpoint | Typical artifact | Expected role |
| --- | --- | --- |
| `INPUT` | user request | user |
| `TASK_SELECTION` | selected task or route | assistant |
| `EXECUTION` | plan or executable step | assistant |
| `TOOL_INPUT` | tool name and arguments | assistant |
| `TOOL_OUTPUT` | tool result | tool |
| `COMMUNICATION` | final user-facing response | assistant |

Create a separate gate for each checkpoint. The classifier must support the checkpoint's evidence role.

```python
from psysafe import (
    Checkpoint,
    GateAction,
    GatePolicy,
    PIIClassifier,
    Sensitivity,
    WorkflowGate,
)

communication_gate = WorkflowGate(
    (PIIClassifier(),),
    checkpoint=Checkpoint.COMMUNICATION,
    policy=GatePolicy(matched_action=GateAction.BLOCK),
)

decision = communication_gate.evaluate_text(
    draft_response,
    artifact_id="response-version-018",
    sensitivity=Sensitivity.BALANCED,
)

if not decision.is_allowed:
    route_non_allow(decision.action, decision.assessments)
```

An artifact ID is an opaque identifier for the exact immutable version being checked. Generate a new one after any mutation. Never approve a changed tool payload or response with an earlier decision.

### Policy behavior

`GatePolicy(matched_action=...)` makes the action for a match explicit. `not_matched` allows. `indeterminate` and independent review signals default to review and cannot be configured to allow.

Use `ClassifierPolicyOverride` when classifiers at one checkpoint need different actions. The strongest result wins: `block` outranks `review`, which outranks `allow`.

Use `AsyncWorkflowGate` and `await gate.aevaluate_text(...)` in asynchronous systems. Its classifiers run concurrently while returned assessments remain in configured order.

## OpenAI Agents SDK

Install the adapter and whichever backend your classifier uses:

```bash
pip install 'psysafe-ai[openai,openai-agents]'
```

The factories return native SDK guardrail objects:

```python
from agents import Agent, RunConfig, Runner

from psysafe import AsyncWorkflowGate, Checkpoint, GateAction, GatePolicy, PIIClassifier
from psysafe.integrations.openai_agents import (
    openai_input_guardrail,
    openai_output_guardrail,
)

def agent_gate(checkpoint: Checkpoint) -> AsyncWorkflowGate:
    return AsyncWorkflowGate(
        (PIIClassifier(),),
        checkpoint=checkpoint,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )

input_gate = agent_gate(Checkpoint.INPUT)
communication_gate = agent_gate(Checkpoint.COMMUNICATION)

agent = Agent(
    name="Support agent",
    instructions="Help the user clearly and concisely.",
    input_guardrails=[openai_input_guardrail(input_gate)],
    output_guardrails=[openai_output_guardrail(communication_gate)],
)

result = await Runner.run(
    agent,
    user_input,
    run_config=RunConfig(trace_include_sensitive_data=False),
)
```

Input checks are configured to finish before model execution. OpenAI input guardrails run only on the first agent in a chain; output guardrails run only on the agent producing the final output. They do not inspect intermediate handoff content.

For custom function tools, attach `openai_tool_input_guardrail(tool_input_gate)` and `openai_tool_output_guardrail(tool_output_gate)` through the SDK's tool guardrail parameters. SDK tool guardrails do not cover handoffs, hosted tools, MCP tools, or `Agent.as_tool()`; wrap those boundaries separately.

Output guardrails are not streaming filters. If rejected content must never reach a user, buffer streamed deltas and reveal them only after the guarded run succeeds. Tool-output checks run after execution and cannot undo a side effect.

The adapter rejects unsupported structured media rather than silently dropping it. Convert media to a trusted, bounded text representation before the gate if your policy needs to assess it.

See [`examples/openai_agents.py`](../examples/openai_agents.py) and the official [OpenAI Agents guardrail documentation](https://openai.github.io/openai-agents-python/guardrails/).

## Claude Agent SDK

Install the adapter and whichever backend your classifier uses:

```bash
pip install 'psysafe-ai[anthropic,claude-agent]'
```

Build a native hook mapping:

```python
import secrets

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from psysafe import AsyncWorkflowGate, Checkpoint, GateAction, GatePolicy, PIIClassifier
from psysafe.integrations.claude_agent import claude_agent_hooks

def agent_gate(checkpoint: Checkpoint) -> AsyncWorkflowGate:
    return AsyncWorkflowGate(
        (PIIClassifier(),),
        checkpoint=checkpoint,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )

communication_gate = agent_gate(Checkpoint.COMMUNICATION)
options = ClaudeAgentOptions(
    hooks=claude_agent_hooks(
        user_prompt_submit_gate=agent_gate(Checkpoint.INPUT),
        pre_tool_use_gate=agent_gate(Checkpoint.TOOL_INPUT),
        post_tool_use_gate=agent_gate(Checkpoint.TOOL_OUTPUT),
    )
)

async for message in query(prompt=user_input, options=options):
    # Do not expose intermediate AssistantMessage content.
    if isinstance(message, ResultMessage):
        if (
            message.is_error
            or message.subtype != "success"
            or message.terminal_reason not in (None, "completed")
            or message.result is None
        ):
            route_agent_failure()
            continue
        decision = await communication_gate.aevaluate_text(
            message.result,
            artifact_id=f"claude-result:{secrets.token_hex(16)}",
        )
        if decision.is_allowed:
            consume(message.result)
        else:
            route_non_allow(decision.action, decision.assessments)
```

The mapping uses `UserPromptSubmit`, `PreToolUse`, and `PostToolUse`. A rejected user prompt is suppressed. At `PreToolUse`, review requests normal permission confirmation and block denies; an allow result preserves the SDK's ordinary permission flow. At `PostToolUse`, review or block stops subsequent agent execution but cannot reverse an action already performed.

Claude Agent SDK hooks do not provide a final-assistant-output guardrail. Do not display intermediate `AssistantMessage` events; gate `ResultMessage.result` explicitly before display, as above. This is not a streaming filter, so buffer content until the communication decision allows it.

See [`examples/claude_agent.py`](../examples/claude_agent.py) and the official [Claude Agent SDK hooks documentation](https://code.claude.com/docs/en/agent-sdk/hooks).

## Data, errors, and tracing

Agent adapters canonicalize bounded JSON-compatible values and bind tool decisions to the exact call ID and payload. Unsupported objects, cycles, excessive inputs, or malformed SDK values fail closed with a fixed exception that does not echo data. The OpenAI adapter also rejects recognized structured-media items; convert media to a trusted, bounded text representation before either adapter if your policy needs to assess it.

The canonicalized text is still sent to the gate's classifier backend. Configure both the classifier provider and agent framework for appropriate retention and tracing, avoid content-bearing artifact IDs, and do not persist raw gate inputs or framework traces by default.

Always test the exact lifecycle boundaries, SDK version, provider model, tools, handoffs, streaming behavior, and non-allow exception path used in production.
