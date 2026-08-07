"""OpenAI Agents SDK guardrails backed by calibrated PsySafe gates.

The factories in this module intentionally return the SDK's native guardrail
objects.  A gate decision is the only value exposed as ``output_info``; raw
workflow artifacts are never copied into guardrail results or artifact IDs.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
from typing import Any, NoReturn

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.tool_context import ToolContext

from psysafe.core.contracts import Sensitivity
from psysafe.gates import AsyncWorkflowGate, Checkpoint, GateAction, GateDecision
from psysafe.integrations._serialization import IntegrationInputError, canonical_json

_MAX_TOOL_CALL_ID_CHARS = 4_096
_MAX_TOOL_NAME_CHARS = 1_024
_UNSUPPORTED_MODALITY_TYPES = frozenset(
    {
        "audio",
        "computer_screenshot",
        "file",
        "image",
        "image_generation_call",
        "input_audio",
        "input_file",
        "input_image",
    },
)
_NAMED_SENSITIVITIES = {
    Sensitivity.PRECISE.value: Sensitivity.PRECISE,
    Sensitivity.BALANCED.value: Sensitivity.BALANCED,
    Sensitivity.PRECAUTIONARY.value: Sensitivity.PRECAUTIONARY,
}


class _SerializationFailure:
    """Non-data-bearing marker returned after a serialization failure."""

    __slots__ = ()


class _Cancellation:
    """Non-data-bearing marker used to clear payloads before cancellation."""

    __slots__ = ()


_SERIALIZATION_FAILURE = _SerializationFailure()
_CANCELLATION = _Cancellation()


class _ToolArtifactIds:
    """Produce opaque exact-artifact IDs without exposing the HMAC key."""

    __slots__ = ("__secret", "_prefix")

    def __init__(self, prefix: str) -> None:
        self.__secret = secrets.token_bytes(32)
        self._prefix = prefix

    def from_call(self, call_id: object, payload: object) -> str | _SerializationFailure:
        if type(call_id) is not str or not 1 <= len(call_id) <= _MAX_TOOL_CALL_ID_CHARS or type(payload) is not str:
            return _SERIALIZATION_FAILURE
        try:
            encoded_call_id = call_id.encode("utf-8", errors="strict")
            encoded_payload = payload.encode("utf-8", errors="strict")
        except UnicodeError:
            return _SERIALIZATION_FAILURE
        # The length prefix prevents ambiguous concatenations even when a call
        # ID itself contains a NUL or another delimiter-like character.
        material = len(encoded_call_id).to_bytes(8, "big") + encoded_call_id + encoded_payload
        digest = hmac.new(self.__secret, material, hashlib.sha256).hexdigest()
        return f"{self._prefix}:{digest}"

    def __repr__(self) -> str:
        return "<opaque tool artifact ID factory>"


def _validated_gate(gate: object, expected_checkpoint: Checkpoint) -> AsyncWorkflowGate:
    if type(gate) is not AsyncWorkflowGate:
        raise TypeError("gate must be an AsyncWorkflowGate")
    if gate.checkpoint is not expected_checkpoint:
        raise ValueError(f"gate checkpoint must be {expected_checkpoint.value!r}")
    return gate


def _validated_sensitivity(value: object) -> Sensitivity:
    if type(value) is Sensitivity:
        return value
    if type(value) is str:
        normalized = _NAMED_SENSITIVITIES.get(value)
        if normalized is not None:
            return normalized
    raise ValueError("sensitivity must be precise, balanced, or precautionary")


def _contains_unsupported_modality(value: object) -> bool:
    """Recognize structured media references without dispatching custom code."""

    if type(value) is dict:
        concrete = value
        modality = dict.get(concrete, "type")
        if type(modality) is str and modality in _UNSUPPORTED_MODALITY_TYPES:
            return True
        return any(_contains_unsupported_modality(item) for item in dict.values(dict.copy(concrete)))
    if type(value) is list:
        return any(_contains_unsupported_modality(item) for item in value.copy())
    if type(value) is tuple:
        return any(_contains_unsupported_modality(item) for item in value)
    return False


def _serialized_text(value: object) -> str | _SerializationFailure | _Cancellation:
    if type(value) is str:
        return value
    try:
        rendered = canonical_json(value)
        if _contains_unsupported_modality(value):
            return _SERIALIZATION_FAILURE
        return rendered
    except asyncio.CancelledError:
        return _CANCELLATION
    except Exception:  # noqa: BLE001 - the integration boundary must replace data-bearing errors.
        return _SERIALIZATION_FAILURE


def _tool_text(
    tool_name: object,
    tool_input: object,
    *,
    tool_output: object | None = None,
    include_output: bool,
) -> str | _SerializationFailure | _Cancellation:
    if type(tool_name) is not str or not 1 <= len(tool_name) <= _MAX_TOOL_NAME_CHARS:
        return _SERIALIZATION_FAILURE
    envelope: dict[str, object] = {
        "tool_input": tool_input,
        "tool_name": tool_name,
    }
    if include_output:
        envelope["tool_output"] = tool_output
    try:
        rendered = canonical_json(envelope)
        if _contains_unsupported_modality(envelope):
            return _SERIALIZATION_FAILURE
        return rendered
    except asyncio.CancelledError:
        return _CANCELLATION
    except Exception:  # noqa: BLE001 - the integration boundary must replace data-bearing errors.
        return _SERIALIZATION_FAILURE


def _fresh_artifact_id(prefix: str) -> str:
    """Create a content-independent correlation ID for one guardrail invocation."""

    return f"{prefix}:{secrets.token_hex(16)}"


def _raise_input_error() -> NoReturn:
    raise IntegrationInputError() from None


def _agent_result(decision: GateDecision) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(
        output_info=decision,
        tripwire_triggered=decision.action is not GateAction.ALLOW,
    )


def _tool_result(decision: GateDecision) -> ToolGuardrailFunctionOutput:
    if decision.action is GateAction.ALLOW:
        return ToolGuardrailFunctionOutput.allow(output_info=decision)
    return ToolGuardrailFunctionOutput.raise_exception(output_info=decision)


def openai_input_guardrail(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> InputGuardrail[Any]:
    """Create a blocking guardrail for the first agent's input.

    OpenAI Agents SDK input guardrails run only on the first agent in a chain.
    This factory sets ``run_in_parallel=False`` so the check finishes before
    model execution or downstream side effects can begin.
    """

    checked_gate = _validated_gate(gate, Checkpoint.INPUT)
    checked_sensitivity = _validated_sensitivity(sensitivity)

    @input_guardrail(name="psysafe_input", run_in_parallel=False)
    async def guardrail(
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        agent_input: str | list[TResponseInputItem],
    ) -> GuardrailFunctionOutput:
        del context, agent
        text = _serialized_text(agent_input)
        if isinstance(text, _Cancellation):
            del agent_input, text
            raise asyncio.CancelledError() from None
        if isinstance(text, _SerializationFailure):
            del agent_input, text
            _raise_input_error()
        try:
            decision = await checked_gate.aevaluate_text(
                text,
                artifact_id=_fresh_artifact_id("oai-input"),
                sensitivity=checked_sensitivity,
            )
        except BaseException as caught:
            del agent_input, text
            if isinstance(caught, asyncio.CancelledError):
                del caught
                raise asyncio.CancelledError() from None
            raise
        return _agent_result(decision)

    return guardrail


def openai_output_guardrail(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> OutputGuardrail[Any]:
    """Create a guardrail for the final-producing agent's communication.

    OpenAI Agents SDK output guardrails run only for the agent that produces
    the final output; they do not check intermediate handoff outputs. This is
    not a streaming filter: when the check is preventive, buffer streamed
    deltas and expose content only after the guarded run completes successfully.
    """

    checked_gate = _validated_gate(gate, Checkpoint.COMMUNICATION)
    checked_sensitivity = _validated_sensitivity(sensitivity)

    @output_guardrail(name="psysafe_communication")
    async def guardrail(
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        agent_output: Any,
    ) -> GuardrailFunctionOutput:
        del context, agent
        text = _serialized_text(agent_output)
        if isinstance(text, _Cancellation):
            del agent_output, text
            raise asyncio.CancelledError() from None
        if isinstance(text, _SerializationFailure):
            del agent_output, text
            _raise_input_error()
        try:
            decision = await checked_gate.aevaluate_text(
                text,
                artifact_id=_fresh_artifact_id("oai-output"),
                sensitivity=checked_sensitivity,
            )
        except BaseException as caught:
            del agent_output, text
            if isinstance(caught, asyncio.CancelledError):
                del caught
                raise asyncio.CancelledError() from None
            raise
        return _agent_result(decision)

    return guardrail


def openai_tool_input_guardrail(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> ToolInputGuardrail[Any]:
    """Create a pre-execution guardrail for custom function-tool arguments.

    SDK tool guardrails cover custom function tools only. They do not wrap
    handoffs, hosted tools, MCP tools, or ``Agent.as_tool()``.
    """

    checked_gate = _validated_gate(gate, Checkpoint.TOOL_INPUT)
    checked_sensitivity = _validated_sensitivity(sensitivity)
    artifact_ids = _ToolArtifactIds("oai-tool-input")

    @tool_input_guardrail(name="psysafe_tool_input")
    async def guardrail(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        if type(data) is not ToolInputGuardrailData:
            del data
            _raise_input_error()
        context = data.context
        if type(context) is not ToolContext:
            del data, context
            _raise_input_error()
        call_id = context.tool_call_id
        tool_name = context.tool_name
        tool_input = context.tool_arguments
        del data, context
        text = _tool_text(tool_name, tool_input, include_output=False)
        if isinstance(text, _Cancellation):
            del call_id, tool_name, tool_input, text
            raise asyncio.CancelledError() from None
        if isinstance(text, _SerializationFailure):
            del call_id, tool_name, tool_input, text
            _raise_input_error()
        artifact_id = artifact_ids.from_call(call_id, text)
        if isinstance(artifact_id, _SerializationFailure):
            del call_id, tool_name, tool_input, text, artifact_id
            _raise_input_error()
        del call_id, tool_name, tool_input
        try:
            decision = await checked_gate.aevaluate_text(
                text,
                artifact_id=artifact_id,
                sensitivity=checked_sensitivity,
            )
        except BaseException as caught:
            del text, artifact_id
            if isinstance(caught, asyncio.CancelledError):
                del caught
                raise asyncio.CancelledError() from None
            raise
        return _tool_result(decision)

    return guardrail


def openai_tool_output_guardrail(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> ToolOutputGuardrail[Any]:
    """Create a post-execution guardrail for custom function-tool output.

    SDK tool guardrails cover custom function tools only. They do not wrap
    handoffs, hosted tools, MCP tools, or ``Agent.as_tool()``. A non-allow
    decision halts the run, but cannot undo a side effect that already occurred.
    """

    checked_gate = _validated_gate(gate, Checkpoint.TOOL_OUTPUT)
    checked_sensitivity = _validated_sensitivity(sensitivity)
    artifact_ids = _ToolArtifactIds("oai-tool-output")

    @tool_output_guardrail(name="psysafe_tool_output")
    async def guardrail(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        if type(data) is not ToolOutputGuardrailData:
            del data
            _raise_input_error()
        context = data.context
        if type(context) is not ToolContext:
            del data, context
            _raise_input_error()
        call_id = context.tool_call_id
        tool_name = context.tool_name
        tool_input = context.tool_arguments
        tool_output = data.output
        del data, context
        text = _tool_text(
            tool_name,
            tool_input,
            tool_output=tool_output,
            include_output=True,
        )
        if isinstance(text, _Cancellation):
            del call_id, tool_name, tool_input, tool_output, text
            raise asyncio.CancelledError() from None
        if isinstance(text, _SerializationFailure):
            del call_id, tool_name, tool_input, tool_output, text
            _raise_input_error()
        artifact_id = artifact_ids.from_call(call_id, text)
        if isinstance(artifact_id, _SerializationFailure):
            del call_id, tool_name, tool_input, tool_output, text, artifact_id
            _raise_input_error()
        del call_id, tool_name, tool_input, tool_output
        try:
            decision = await checked_gate.aevaluate_text(
                text,
                artifact_id=artifact_id,
                sensitivity=checked_sensitivity,
            )
        except BaseException as caught:
            del text, artifact_id
            if isinstance(caught, asyncio.CancelledError):
                del caught
                raise asyncio.CancelledError() from None
            raise
        return _tool_result(decision)

    return guardrail


__all__ = [
    "openai_input_guardrail",
    "openai_output_guardrail",
    "openai_tool_input_guardrail",
    "openai_tool_output_guardrail",
]
