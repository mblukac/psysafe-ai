# PsySafe examples

These examples show where to place calibrated, categorical checks in two common
runtime shapes: a sequential workflow and an agent loop. PsySafe returns
`matched`, `not_matched`, or `indeterminate`; your application maps those
outcomes to `allow`, `review`, or `block`.

The examples do not produce per-item confidence, severity, urgency, or risk
ratings. The three sensitivity settings are named policy boundaries:
`precise`, `balanced`, and `precautionary`.

## Setup

Install the core package and only the provider/runtime extras you use:

```console
pip install -e '.[openai]'
pip install -e '.[openai-agents]'
pip install -e '.[claude-agent]'
```

Provider-backed examples read model names from environment variables instead
of baking a model choice into application code:

- `classification.py`: `OPENAI_API_KEY` and `PSYSAFE_CLASSIFIER_MODEL`
- `openai_agents.py`: `OPENAI_API_KEY` and `OPENAI_AGENT_MODEL`
- `claude_agent.py`: Claude Agent SDK authentication and `CLAUDE_AGENT_MODEL`

Run an example from the repository root:

```console
python examples/classification.py
python examples/sequential_gate.py
python examples/openai_agents.py
python examples/claude_agent.py
python examples/categorical_evaluation.py
```

## What each example demonstrates

- [`classification.py`](classification.py) makes one structured model
  observation, then applies all three sensitivity boundaries locally.
- [`sequential_gate.py`](sequential_gate.py) checks immutable artifacts at
  input, task-selection, execution-plan, and communication boundaries.
- [`openai_agents.py`](openai_agents.py) attaches native OpenAI Agents SDK
  input, output, and custom function-tool guardrails.
- [`claude_agent.py`](claude_agent.py) attaches native Claude Agent SDK
  `UserPromptSubmit`, `PreToolUse`, and `PostToolUse` hooks, then explicitly
  gates a completed `ResultMessage.result` before display.
- [`categorical_evaluation.py`](categorical_evaluation.py) runs a tuning split
  from strict JSONL cases and prints only categorical case results.

Use the same sensitivity-independent observation when comparing sensitivity
boundaries; recalibration does not call the provider again. Treat
`indeterminate` as a failed or incomplete check, never as a clean result.

Agent boundary details matter. OpenAI output guardrails are final-output checks,
not streaming filters, and Claude's Python hooks do not cover final assistant
output. Buffer content until the guarded OpenAI run or an explicit Claude
`ResultMessage.result` communication check allows display. OpenAI custom tool
guardrails do not cover hosted tools, MCP tools, handoffs, or agents exposed as
tools. In either SDK, a post-tool check can halt what happens next but cannot
undo a side effect that already occurred. Put preventive checks before execution
and keep each SDK's normal permission controls enabled.
