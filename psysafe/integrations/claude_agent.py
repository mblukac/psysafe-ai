"""Claude Agent SDK hooks for PsySafe workflow gates.

The hooks translate Claude Agent SDK lifecycle boundaries into PsySafe's
categorical gate decisions.  They never expose classifier details, source
content, or tool-use identifiers in hook output.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
from typing import NoReturn, cast

from claude_agent_sdk import (
    HookCallback,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
)
from claude_agent_sdk.types import (
    HookEvent,
    PreToolUseHookSpecificOutput,
    SyncHookJSONOutput,
)

from psysafe.backends.base import BackendConfigurationError, _fresh_configuration_error
from psysafe.core.contracts import Sensitivity
from psysafe.gates import AsyncWorkflowGate, Checkpoint, GateAction, GateDecision
from psysafe.integrations._serialization import IntegrationInputError, canonical_json

_MAX_TOOL_USE_ID_CHARS = 1_024
_MAX_TOOL_NAME_CHARS = 256
_NAMED_SENSITIVITIES = {
    Sensitivity.PRECISE.value: Sensitivity.PRECISE,
    Sensitivity.BALANCED.value: Sensitivity.BALANCED,
    Sensitivity.PRECAUTIONARY.value: Sensitivity.PRECAUTIONARY,
}

_PROMPT_REVIEW_REASON = "PsySafe requires human review before this prompt can continue."
_PROMPT_BLOCK_REASON = "PsySafe blocked this prompt under the configured safety policy."
_TOOL_REVIEW_REASON = "PsySafe requires human review before this tool call can continue."
_TOOL_BLOCK_REASON = "PsySafe blocked this tool call under the configured safety policy."
_OUTPUT_REVIEW_REASON = "PsySafe stopped the run for human review after the tool completed."
_OUTPUT_BLOCK_REASON = "PsySafe stopped the run after the tool output matched the configured safety policy."


class _ToolArtifactIds:
    """Create keyed exact-artifact correlations without exposing inputs."""

    __slots__ = ("__secret", "_prefix")

    def __init__(self, prefix: str) -> None:
        self.__secret = secrets.token_bytes(32)
        self._prefix = prefix

    def from_tool_artifact(self, tool_use_id: str, payload: str) -> str:
        try:
            identifier_bytes = tool_use_id.encode("utf-8")
            payload_bytes = payload.encode("utf-8")
        except (UnicodeError, ValueError):
            _raise_input_error()
        # Length prefixes make the two variable-length fields unambiguous.
        material = (
            len(identifier_bytes).to_bytes(8, "big")
            + identifier_bytes
            + len(payload_bytes).to_bytes(8, "big")
            + payload_bytes
        )
        digest = hmac.new(self.__secret, material, hashlib.sha256).hexdigest()
        return f"{self._prefix}:{digest}"

    def __repr__(self) -> str:
        return "<opaque tool artifact ID factory>"


def _raise_input_error() -> NoReturn:
    """Raise a fresh fixed error from a frame with no integration input."""

    raise IntegrationInputError from None


def _raise_cancelled() -> NoReturn:
    """Raise fresh cancellation from a frame with no integration input."""

    raise asyncio.CancelledError from None


def _raise_configuration_error(error: BackendConfigurationError) -> NoReturn:
    """Raise an actionable provider error without retaining the hook callback."""

    raise error from None


def _raise_runtime_error() -> NoReturn:
    """Collapse an unexpected gate failure to a fixed, data-free exception."""

    raise RuntimeError("PsySafe Claude Agent hook evaluation failed") from None


def _validated_sensitivity(value: object) -> Sensitivity:
    if type(value) is Sensitivity:
        return value
    if type(value) is str:
        normalized = _NAMED_SENSITIVITIES.get(value)
        if normalized is not None:
            return normalized
    raise ValueError("sensitivity must be precise, balanced, or precautionary")


def _validated_gate(gate: object, checkpoint: Checkpoint) -> AsyncWorkflowGate:
    # Exact gates keep callback dispatch on the audited library implementation;
    # a subclass could override ``checkpoint`` or ``aevaluate_text`` and retain
    # the raw SDK payload.
    if type(gate) is not AsyncWorkflowGate:
        raise TypeError("Claude Agent hooks require an AsyncWorkflowGate")
    actual_checkpoint = gate.checkpoint
    if type(actual_checkpoint) is not Checkpoint or actual_checkpoint is not checkpoint:
        raise ValueError(f"Claude Agent hook requires the {checkpoint.value!r} checkpoint")
    return gate


def _hook_data(value: HookInput, event: HookEvent) -> dict[str, object]:
    if type(value) is not dict:
        _raise_input_error()
    data = cast(dict[str, object], value)
    raw_event = data.get("hook_event_name")
    if type(raw_event) is not str or raw_event != event:
        _raise_input_error()
    return data


def _user_prompt(value: HookInput, callback_tool_use_id: object) -> str:
    data = _hook_data(value, "UserPromptSubmit")
    prompt = data.get("prompt")
    if type(prompt) is not str or callback_tool_use_id is not None:
        _raise_input_error()
    return prompt


def _tool_payload(
    value: HookInput,
    callback_tool_use_id: object,
    *,
    event: HookEvent,
    include_response: bool,
) -> tuple[str, str]:
    data = _hook_data(value, event)
    tool_name = data.get("tool_name")
    tool_input = data.get("tool_input")
    embedded_tool_use_id = data.get("tool_use_id")
    if type(tool_name) is not str or not 1 <= len(tool_name) <= _MAX_TOOL_NAME_CHARS or type(tool_input) is not dict:
        _raise_input_error()
    if (
        type(embedded_tool_use_id) is not str
        or not embedded_tool_use_id
        or len(embedded_tool_use_id) > _MAX_TOOL_USE_ID_CHARS
        or type(callback_tool_use_id) is not str
        or callback_tool_use_id != embedded_tool_use_id
    ):
        _raise_input_error()

    payload: dict[str, object] = {
        "tool_input": tool_input,
        "tool_name": tool_name,
    }
    if include_response:
        if "tool_response" not in data:
            _raise_input_error()
        payload["tool_response"] = data["tool_response"]
    return canonical_json(payload), embedded_tool_use_id


def _prompt_artifact_id() -> str:
    return f"claude-prompt:{secrets.token_hex(16)}"


def _prompt_output(action: GateAction) -> SyncHookJSONOutput:
    if action is GateAction.ALLOW:
        return {}
    reason = _PROMPT_REVIEW_REASON if action is GateAction.REVIEW else _PROMPT_BLOCK_REASON
    # ``suppressOriginalPrompt`` was added to the CLI hook contract before it
    # appeared in every supported Python SDK TypedDict. Unknown keys pass
    # through the SDK converter unchanged, so this remains compatible with the
    # earliest supported 0.2.x release and prevents rejected content appearing
    # in the user-visible block message.
    return cast(
        SyncHookJSONOutput,
        {
            "continue_": False,
            "stopReason": reason,
            "decision": "block",
            "reason": reason,
            "suppressOriginalPrompt": True,
        },
    )


def _pre_tool_output(action: GateAction) -> SyncHookJSONOutput:
    if action is GateAction.ALLOW:
        # An explicit allow would skip Claude's normal permission flow.
        return {}
    if action is GateAction.REVIEW:
        specific: PreToolUseHookSpecificOutput = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": _TOOL_REVIEW_REASON,
        }
    else:
        specific = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": _TOOL_BLOCK_REASON,
        }
    return {"hookSpecificOutput": specific}


def _post_tool_output(action: GateAction) -> SyncHookJSONOutput:
    if action is GateAction.ALLOW:
        return {}
    reason = _OUTPUT_REVIEW_REASON if action is GateAction.REVIEW else _OUTPUT_BLOCK_REASON
    return {"continue_": False, "stopReason": reason}


def claude_user_prompt_submit_hook(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> HookCallback:
    """Create a ``UserPromptSubmit`` hook backed by an INPUT gate."""

    bound_gate = _validated_gate(gate, Checkpoint.INPUT)
    bound_sensitivity = _validated_sensitivity(sensitivity)

    async def hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        prompt: str | None = None
        artifact_id: str | None = None
        decision: GateDecision | None = None
        invalid_input = False
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        runtime_failure = False
        try:
            prompt = _user_prompt(input_data, tool_use_id)
            artifact_id = _prompt_artifact_id()
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - SDK values and serializer behavior are untrusted.
            invalid_input = True
        if invalid_input or cancelled:
            del input_data, tool_use_id, context, prompt, artifact_id, decision
            if cancelled:
                _raise_cancelled()
            _raise_input_error()
        if prompt is None or artifact_id is None:
            del input_data, tool_use_id, context, prompt, artifact_id, decision
            _raise_runtime_error()
        try:
            decision = await bound_gate.aevaluate_text(
                prompt,
                artifact_id=artifact_id,
                sensitivity=bound_sensitivity,
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as error:
            configuration_failure = _fresh_configuration_error(error)
            runtime_failure = configuration_failure is None
        except Exception:  # noqa: BLE001 - gate execution is a data-bearing boundary.
            runtime_failure = True
        if cancelled or configuration_failure is not None or runtime_failure:
            del input_data, tool_use_id, context, prompt, artifact_id, decision
            if configuration_failure is not None:
                _raise_configuration_error(configuration_failure)
            if runtime_failure:
                _raise_runtime_error()
            _raise_cancelled()
        if type(decision) is not GateDecision:
            del input_data, tool_use_id, context, prompt, artifact_id, decision
            _raise_runtime_error()
        action = decision.action
        del input_data, tool_use_id, context, prompt, artifact_id, decision
        return _prompt_output(action)

    return hook


def claude_pre_tool_use_hook(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> HookCallback:
    """Create a ``PreToolUse`` hook backed by a TOOL_INPUT gate."""

    bound_gate = _validated_gate(gate, Checkpoint.TOOL_INPUT)
    bound_sensitivity = _validated_sensitivity(sensitivity)
    artifact_ids = _ToolArtifactIds("claude-tool-input")

    async def hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        payload: str | None = None
        embedded_tool_use_id: str | None = None
        artifact_id: str | None = None
        decision: GateDecision | None = None
        invalid_input = False
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        runtime_failure = False
        try:
            payload, embedded_tool_use_id = _tool_payload(
                input_data,
                tool_use_id,
                event="PreToolUse",
                include_response=False,
            )
            artifact_id = artifact_ids.from_tool_artifact(embedded_tool_use_id, payload)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - SDK values and serializer behavior are untrusted.
            invalid_input = True
        if invalid_input or cancelled:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            if cancelled:
                _raise_cancelled()
            _raise_input_error()
        if payload is None or artifact_id is None:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            _raise_runtime_error()
        try:
            decision = await bound_gate.aevaluate_text(
                payload,
                artifact_id=artifact_id,
                sensitivity=bound_sensitivity,
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as error:
            configuration_failure = _fresh_configuration_error(error)
            runtime_failure = configuration_failure is None
        except Exception:  # noqa: BLE001 - gate execution is a data-bearing boundary.
            runtime_failure = True
        if cancelled or configuration_failure is not None or runtime_failure:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            if configuration_failure is not None:
                _raise_configuration_error(configuration_failure)
            if runtime_failure:
                _raise_runtime_error()
            _raise_cancelled()
        if type(decision) is not GateDecision:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            _raise_runtime_error()
        action = decision.action
        del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
        return _pre_tool_output(action)

    return hook


def claude_post_tool_use_hook(
    gate: AsyncWorkflowGate,
    *,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> HookCallback:
    """Create a ``PostToolUse`` hook backed by a TOOL_OUTPUT gate.

    A post-tool hook runs after the tool has executed.  REVIEW and BLOCK stop
    subsequent agent execution, but cannot undo an already completed external
    side effect.  Applications should use ``PreToolUse`` for preventive checks.
    """

    bound_gate = _validated_gate(gate, Checkpoint.TOOL_OUTPUT)
    bound_sensitivity = _validated_sensitivity(sensitivity)
    artifact_ids = _ToolArtifactIds("claude-tool-output")

    async def hook(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        payload: str | None = None
        embedded_tool_use_id: str | None = None
        artifact_id: str | None = None
        decision: GateDecision | None = None
        invalid_input = False
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        runtime_failure = False
        try:
            payload, embedded_tool_use_id = _tool_payload(
                input_data,
                tool_use_id,
                event="PostToolUse",
                include_response=True,
            )
            artifact_id = artifact_ids.from_tool_artifact(embedded_tool_use_id, payload)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - SDK values and serializer behavior are untrusted.
            invalid_input = True
        if invalid_input or cancelled:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            if cancelled:
                _raise_cancelled()
            _raise_input_error()
        if payload is None or artifact_id is None:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            _raise_runtime_error()
        try:
            decision = await bound_gate.aevaluate_text(
                payload,
                artifact_id=artifact_id,
                sensitivity=bound_sensitivity,
            )
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as error:
            configuration_failure = _fresh_configuration_error(error)
            runtime_failure = configuration_failure is None
        except Exception:  # noqa: BLE001 - gate execution is a data-bearing boundary.
            runtime_failure = True
        if cancelled or configuration_failure is not None or runtime_failure:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            if configuration_failure is not None:
                _raise_configuration_error(configuration_failure)
            if runtime_failure:
                _raise_runtime_error()
            _raise_cancelled()
        if type(decision) is not GateDecision:
            del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
            _raise_runtime_error()
        action = decision.action
        del input_data, tool_use_id, context, payload, embedded_tool_use_id, artifact_id, decision
        return _post_tool_output(action)

    return hook


def claude_agent_hooks(
    *,
    user_prompt_submit_gate: AsyncWorkflowGate | None = None,
    pre_tool_use_gate: AsyncWorkflowGate | None = None,
    post_tool_use_gate: AsyncWorkflowGate | None = None,
    sensitivity: Sensitivity | str = Sensitivity.BALANCED,
) -> dict[HookEvent, list[HookMatcher]]:
    """Build a hook mapping accepted by ``ClaudeAgentOptions(hooks=...)``."""

    bound_sensitivity = _validated_sensitivity(sensitivity)
    hooks: dict[HookEvent, list[HookMatcher]] = {}
    if user_prompt_submit_gate is not None:
        hooks["UserPromptSubmit"] = [
            HookMatcher(
                hooks=[
                    claude_user_prompt_submit_hook(
                        user_prompt_submit_gate,
                        sensitivity=bound_sensitivity,
                    ),
                ],
            ),
        ]
    if pre_tool_use_gate is not None:
        hooks["PreToolUse"] = [
            HookMatcher(
                hooks=[
                    claude_pre_tool_use_hook(
                        pre_tool_use_gate,
                        sensitivity=bound_sensitivity,
                    ),
                ],
            ),
        ]
    if post_tool_use_gate is not None:
        hooks["PostToolUse"] = [
            HookMatcher(
                hooks=[
                    claude_post_tool_use_hook(
                        post_tool_use_gate,
                        sensitivity=bound_sensitivity,
                    ),
                ],
            ),
        ]
    if not hooks:
        raise ValueError("at least one Claude Agent gate is required")
    return hooks


__all__ = [
    "claude_agent_hooks",
    "claude_post_tool_use_hook",
    "claude_pre_tool_use_hook",
    "claude_user_prompt_submit_hook",
]
