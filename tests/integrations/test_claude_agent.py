from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookCallback,
    HookContext,
    HookInput,
    HookMatcher,
    PostToolUseHookInput,
    PreToolUseHookInput,
    UserPromptSubmitHookInput,
)

from psysafe.backends.base import BackendConfigurationError
from psysafe.core.contracts import (
    Assessment,
    Conversation,
    EvidenceDirectness,
    MessageRole,
    Outcome,
    Sensitivity,
)
from psysafe.gates import AsyncWorkflowGate, Checkpoint, GateAction, GatePolicy
from psysafe.integrations._serialization import IntegrationInputError
from psysafe.integrations.claude_agent import (
    claude_agent_hooks,
    claude_post_tool_use_hook,
    claude_pre_tool_use_hook,
    claude_user_prompt_submit_hook,
)

_CONTEXT = HookContext(signal=None)
_PROMPT_REVIEW_REASON = "PsySafe requires human review before this prompt can continue."
_PROMPT_BLOCK_REASON = "PsySafe blocked this prompt under the configured safety policy."
_TOOL_REVIEW_REASON = "PsySafe requires human review before this tool call can continue."
_TOOL_BLOCK_REASON = "PsySafe blocked this tool call under the configured safety policy."
_OUTPUT_REVIEW_REASON = "PsySafe stopped the run for human review after the tool completed."
_OUTPUT_BLOCK_REASON = "PsySafe stopped the run after the tool output matched the configured safety policy."

_ROLE_FOR_CHECKPOINT = {
    Checkpoint.INPUT: MessageRole.USER,
    Checkpoint.TOOL_INPUT: MessageRole.ASSISTANT,
    Checkpoint.TOOL_OUTPUT: MessageRole.TOOL,
}


class _StaticAsyncClassifier:
    classifier_id = "claude_hook_fixture"
    policy_version = "2026.08.2"
    allowed_signals = ("policy_match",)
    allowed_review_signals: tuple[str, ...] = ()

    def __init__(self, *, matched: bool, evidence_role: MessageRole) -> None:
        self.evidence_role = evidence_role
        self._matched = matched
        self.seen_texts: list[str] = []
        self.seen_sensitivities: list[Sensitivity] = []

    def _assessment(self, sensitivity: Sensitivity) -> Assessment:
        if self._matched:
            return Assessment(
                classifier_id=self.classifier_id,
                policy_version=self.policy_version,
                sensitivity=sensitivity,
                outcome=Outcome.MATCHED,
                evidence_directness=EvidenceDirectness.EXPLICIT,
                signals=("policy_match",),
            )
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.NOT_MATCHED,
        )

    async def aclassify(
        self,
        conversation: Conversation,
        *,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        return await self.aclassify_target(
            conversation,
            target_message_index=len(conversation.messages) - 1,
            sensitivity=sensitivity,
        )

    async def aclassify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        self.seen_texts.append(conversation.messages[target_message_index].content)
        self.seen_sensitivities.append(sensitivity)
        return self._assessment(sensitivity)


def _gate(
    checkpoint: Checkpoint,
    action: GateAction,
) -> tuple[AsyncWorkflowGate, _StaticAsyncClassifier]:
    classifier = _StaticAsyncClassifier(
        matched=action is not GateAction.ALLOW,
        evidence_role=_ROLE_FOR_CHECKPOINT[checkpoint],
    )
    return (
        AsyncWorkflowGate(
            (classifier,),
            checkpoint=checkpoint,
            policy=GatePolicy(matched_action=action),
        ),
        classifier,
    )


def _prompt_input(prompt: str) -> UserPromptSubmitHookInput:
    return UserPromptSubmitHookInput(
        session_id="session-redacted",
        transcript_path="/workspace/transcript-redacted",
        cwd="/workspace/project-redacted",
        hook_event_name="UserPromptSubmit",
        prompt=prompt,
    )


def _pre_tool_input(
    tool_input: dict[str, object],
    *,
    tool_use_id: str = "toolu_private_123",
) -> PreToolUseHookInput:
    return PreToolUseHookInput(
        session_id="session-redacted",
        transcript_path="/workspace/transcript-redacted",
        cwd="/workspace/project-redacted",
        hook_event_name="PreToolUse",
        tool_name="SendMessage",
        tool_input=tool_input,
        tool_use_id=tool_use_id,
    )


def _post_tool_input(
    tool_input: dict[str, object],
    tool_response: object,
    *,
    tool_use_id: str = "toolu_private_123",
) -> PostToolUseHookInput:
    return PostToolUseHookInput(
        session_id="session-redacted",
        transcript_path="/workspace/transcript-redacted",
        cwd="/workspace/project-redacted",
        hook_event_name="PostToolUse",
        tool_name="SendMessage",
        tool_input=tool_input,
        tool_response=tool_response,
        tool_use_id=tool_use_id,
    )


async def _invoke(
    hook: HookCallback,
    input_data: HookInput,
    tool_use_id: str | None,
):
    return await hook(input_data, tool_use_id, _CONTEXT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (GateAction.ALLOW, {}),
        (
            GateAction.REVIEW,
            {
                "continue_": False,
                "stopReason": _PROMPT_REVIEW_REASON,
                "decision": "block",
                "reason": _PROMPT_REVIEW_REASON,
                "suppressOriginalPrompt": True,
            },
        ),
        (
            GateAction.BLOCK,
            {
                "continue_": False,
                "stopReason": _PROMPT_BLOCK_REASON,
                "decision": "block",
                "reason": _PROMPT_BLOCK_REASON,
                "suppressOriginalPrompt": True,
            },
        ),
    ],
)
async def test_user_prompt_submit_maps_gate_actions_to_real_hook_output(
    action: GateAction,
    expected: dict[str, object],
) -> None:
    gate, classifier = _gate(Checkpoint.INPUT, action)
    hook = claude_user_prompt_submit_hook(gate, sensitivity="precautionary")
    prompt = "Please help me draft a difficult message."

    output = await _invoke(hook, _prompt_input(prompt), None)

    assert output == expected
    assert classifier.seen_texts == [prompt]
    assert classifier.seen_sensitivities == [Sensitivity.PRECAUTIONARY]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (GateAction.ALLOW, {}),
        (
            GateAction.REVIEW,
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": _TOOL_REVIEW_REASON,
                },
            },
        ),
        (
            GateAction.BLOCK,
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": _TOOL_BLOCK_REASON,
                },
            },
        ),
    ],
)
async def test_pre_tool_use_preserves_normal_permissions_and_maps_non_allow_actions(
    action: GateAction,
    expected: dict[str, object],
) -> None:
    gate, classifier = _gate(Checkpoint.TOOL_INPUT, action)
    hook = claude_pre_tool_use_hook(gate)
    input_data = _pre_tool_input({"recipient": "team", "body": ("hello", True)})

    output = await _invoke(hook, input_data, input_data["tool_use_id"])

    assert output == expected
    assert json.loads(classifier.seen_texts[0]) == {
        "tool_input": {"body": ["hello", True], "recipient": "team"},
        "tool_name": "SendMessage",
    }
    assert input_data["tool_use_id"] not in classifier.seen_texts[0]
    if action is GateAction.ALLOW:
        assert "permissionDecision" not in str(output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (GateAction.ALLOW, {}),
        (GateAction.REVIEW, {"continue_": False, "stopReason": _OUTPUT_REVIEW_REASON}),
        (GateAction.BLOCK, {"continue_": False, "stopReason": _OUTPUT_BLOCK_REASON}),
    ],
)
async def test_post_tool_use_stops_only_subsequent_execution_for_non_allow_actions(
    action: GateAction,
    expected: dict[str, object],
) -> None:
    gate, classifier = _gate(Checkpoint.TOOL_OUTPUT, action)
    hook = claude_post_tool_use_hook(gate)
    input_data = _post_tool_input(
        {"recipient": "team"},
        {"delivered": True, "receipt": ["accepted", 42]},
    )

    output = await _invoke(hook, input_data, input_data["tool_use_id"])

    assert output == expected
    assert json.loads(classifier.seen_texts[0]) == {
        "tool_input": {"recipient": "team"},
        "tool_name": "SendMessage",
        "tool_response": {"delivered": True, "receipt": ["accepted", 42]},
    }


def test_factories_require_their_exact_checkpoint_and_named_sensitivity() -> None:
    input_gate, _ = _gate(Checkpoint.INPUT, GateAction.ALLOW)
    tool_input_gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
    tool_output_gate, _ = _gate(Checkpoint.TOOL_OUTPUT, GateAction.ALLOW)

    with pytest.raises(ValueError, match="'input' checkpoint"):
        claude_user_prompt_submit_hook(tool_input_gate)
    with pytest.raises(ValueError, match="'tool_input' checkpoint"):
        claude_pre_tool_use_hook(tool_output_gate)
    with pytest.raises(ValueError, match="'tool_output' checkpoint"):
        claude_post_tool_use_hook(input_gate)
    claude_user_prompt_submit_hook(input_gate, sensitivity="precise")

    class SensitivityString(str):
        pass

    for invalid_sensitivity in (
        "low",
        "medium",
        "high",
        "Precise",
        " precise",
        1,
        SensitivityString("precise"),
    ):
        with pytest.raises(ValueError, match="precise, balanced, or precautionary"):
            claude_user_prompt_submit_hook(
                input_gate,
                sensitivity=cast(Sensitivity, invalid_sensitivity),
            )
    with pytest.raises(TypeError, match="AsyncWorkflowGate"):
        claude_user_prompt_submit_hook(cast(AsyncWorkflowGate, object()))

    class UnsafeGate(AsyncWorkflowGate):
        pass

    classifier = _StaticAsyncClassifier(matched=False, evidence_role=MessageRole.USER)
    unsafe_gate = UnsafeGate(
        (classifier,),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.REVIEW),
    )
    with pytest.raises(TypeError, match="AsyncWorkflowGate"):
        claude_user_prompt_submit_hook(unsafe_gate)


def test_composer_returns_real_hook_matchers_suitable_for_options() -> None:
    input_gate, _ = _gate(Checkpoint.INPUT, GateAction.ALLOW)
    tool_input_gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
    tool_output_gate, _ = _gate(Checkpoint.TOOL_OUTPUT, GateAction.ALLOW)

    hooks = claude_agent_hooks(
        user_prompt_submit_gate=input_gate,
        pre_tool_use_gate=tool_input_gate,
        post_tool_use_gate=tool_output_gate,
        sensitivity="precise",
    )
    options = ClaudeAgentOptions(hooks=hooks)

    assert options.hooks is hooks
    assert set(hooks) == {"UserPromptSubmit", "PreToolUse", "PostToolUse"}
    for matchers in hooks.values():
        assert len(matchers) == 1
        assert isinstance(matchers[0], HookMatcher)
        assert matchers[0].matcher is None
        assert len(matchers[0].hooks) == 1
        assert callable(matchers[0].hooks[0])

    with pytest.raises(ValueError, match="at least one"):
        claude_agent_hooks()


@pytest.mark.asyncio
async def test_prompt_artifact_ids_are_fresh_opaque_and_never_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_prompt = "PRIVATE PROMPT CONTENT"
    gate, _ = _gate(Checkpoint.INPUT, GateAction.ALLOW)
    artifact_ids: list[str] = []
    original = AsyncWorkflowGate.aevaluate_text

    async def record_artifact(
        self: AsyncWorkflowGate,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ):
        artifact_ids.append(artifact_id)
        return await original(self, text, artifact_id=artifact_id, sensitivity=sensitivity)

    monkeypatch.setattr(AsyncWorkflowGate, "aevaluate_text", record_artifact)
    hook = claude_user_prompt_submit_hook(gate)

    first = await _invoke(hook, _prompt_input(raw_prompt), None)
    second = await _invoke(hook, _prompt_input(raw_prompt), None)

    assert first == second == {}
    assert len(set(artifact_ids)) == 2
    assert all(value.startswith("claude-prompt:") for value in artifact_ids)
    assert all(raw_prompt not in value for value in artifact_ids)
    assert not any(value in str(first) for value in artifact_ids)


@pytest.mark.asyncio
async def test_tool_artifact_ids_are_keyed_to_exact_artifacts_and_factory_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
    second_gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
    artifacts_by_gate: dict[int, list[str]] = {id(first_gate): [], id(second_gate): []}
    original = AsyncWorkflowGate.aevaluate_text

    async def record_artifact(
        self: AsyncWorkflowGate,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ):
        artifacts_by_gate[id(self)].append(artifact_id)
        return await original(self, text, artifact_id=artifact_id, sensitivity=sensitivity)

    monkeypatch.setattr(AsyncWorkflowGate, "aevaluate_text", record_artifact)
    first_hook = claude_pre_tool_use_hook(first_gate)
    second_hook = claude_pre_tool_use_hook(second_gate)
    first_id = "toolu_PRIVATE_FIRST"
    second_id = "toolu_PRIVATE_SECOND"

    await _invoke(first_hook, _pre_tool_input({"body": "PRIVATE A"}, tool_use_id=first_id), first_id)
    await _invoke(first_hook, _pre_tool_input({"body": "PRIVATE A"}, tool_use_id=first_id), first_id)
    await _invoke(first_hook, _pre_tool_input({"body": "PRIVATE B"}, tool_use_id=first_id), first_id)
    await _invoke(first_hook, _pre_tool_input({"body": "PRIVATE A"}, tool_use_id=second_id), second_id)
    output = await _invoke(second_hook, _pre_tool_input({"body": "PRIVATE A"}, tool_use_id=first_id), first_id)

    first_artifacts = artifacts_by_gate[id(first_gate)]
    assert first_artifacts[0] == first_artifacts[1]
    assert first_artifacts[0] != first_artifacts[2]
    assert first_artifacts[0] != first_artifacts[3]
    assert first_artifacts[0] != artifacts_by_gate[id(second_gate)][0]
    assert all(value.startswith("claude-tool-input:") for value in first_artifacts)
    assert all(first_id not in value and second_id not in value for value in first_artifacts)
    assert all("PRIVATE" not in value for value in first_artifacts)
    assert output == {}
    assert first_id not in str(output)


@pytest.mark.asyncio
async def test_post_tool_artifact_id_changes_with_the_exact_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, _ = _gate(Checkpoint.TOOL_OUTPUT, GateAction.ALLOW)
    artifact_ids: list[str] = []
    original = AsyncWorkflowGate.aevaluate_text

    async def record_artifact(
        self: AsyncWorkflowGate,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ):
        artifact_ids.append(artifact_id)
        return await original(self, text, artifact_id=artifact_id, sensitivity=sensitivity)

    monkeypatch.setattr(AsyncWorkflowGate, "aevaluate_text", record_artifact)
    hook = claude_post_tool_use_hook(gate)
    tool_use_id = "toolu_PRIVATE_OUTPUT"

    await _invoke(hook, _post_tool_input({}, {"result": "A"}, tool_use_id=tool_use_id), tool_use_id)
    await _invoke(hook, _post_tool_input({}, {"result": "B"}, tool_use_id=tool_use_id), tool_use_id)

    assert len(artifact_ids) == 2
    assert artifact_ids[0] != artifact_ids[1]
    assert all(tool_use_id not in value and "result" not in value for value in artifact_ids)


def _integration_traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename
        if "/psysafe/integrations/" in filename:
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {"unsafe": {1, 2}},
        {1: "non-string-key"},
        {"unsafe": "\ud800"},
    ],
)
async def test_tool_serialization_rejection_is_fixed_and_does_not_retain_input(
    invalid: dict[object, object],
) -> None:
    raw_marker = "PRIVATE UNSERIALIZABLE TOOL INPUT"
    invalid["raw"] = raw_marker
    gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
    hook = claude_pre_tool_use_hook(gate)
    input_data = _pre_tool_input(cast(dict[str, object], invalid))

    with pytest.raises(IntegrationInputError) as caught:
        await _invoke(hook, input_data, input_data["tool_use_id"])

    assert str(caught.value) == "integration input is invalid"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert raw_marker not in _integration_traceback_locals(caught.value)


@pytest.mark.asyncio
async def test_post_tool_serialization_rejects_unsupported_response_without_retention() -> None:
    raw_marker = "PRIVATE UNSERIALIZABLE TOOL RESPONSE"
    gate, _ = _gate(Checkpoint.TOOL_OUTPUT, GateAction.ALLOW)
    hook = claude_post_tool_use_hook(gate)
    input_data = _post_tool_input({}, {"raw": raw_marker, "unsafe": object()})

    with pytest.raises(IntegrationInputError) as caught:
        await _invoke(hook, input_data, input_data["tool_use_id"])

    assert raw_marker not in _integration_traceback_locals(caught.value)


@pytest.mark.asyncio
async def test_invalid_event_or_tool_use_id_is_rejected_without_echoing_values() -> None:
    raw_marker = "PRIVATE MISMATCHED IDENTIFIER"
    gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
    hook = claude_pre_tool_use_hook(gate)
    input_data = _pre_tool_input({"body": raw_marker}, tool_use_id="toolu_expected")

    with pytest.raises(IntegrationInputError) as caught:
        await _invoke(hook, cast(HookInput, input_data), raw_marker)

    assert raw_marker not in str(caught.value)
    assert raw_marker not in _integration_traceback_locals(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_kind", ["prompt", "pre_tool", "post_tool"])
async def test_cancellation_is_fresh_and_does_not_retain_hook_input(
    monkeypatch: pytest.MonkeyPatch,
    hook_kind: str,
) -> None:
    raw_marker = "PRIVATE CANCELLED HOOK INPUT"
    tool_use_id: str | None = None
    if hook_kind == "prompt":
        gate, _ = _gate(Checkpoint.INPUT, GateAction.ALLOW)
        hook = claude_user_prompt_submit_hook(gate)
        input_data: HookInput = _prompt_input(raw_marker)
    elif hook_kind == "pre_tool":
        gate, _ = _gate(Checkpoint.TOOL_INPUT, GateAction.ALLOW)
        hook = claude_pre_tool_use_hook(gate)
        pre_input = _pre_tool_input({"body": raw_marker})
        input_data = pre_input
        tool_use_id = pre_input["tool_use_id"]
    else:
        gate, _ = _gate(Checkpoint.TOOL_OUTPUT, GateAction.ALLOW)
        hook = claude_post_tool_use_hook(gate)
        post_input = _post_tool_input({"body": raw_marker}, {"result": raw_marker})
        input_data = post_input
        tool_use_id = post_input["tool_use_id"]

    async def cancel(
        self: AsyncWorkflowGate,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ):
        del self, artifact_id, sensitivity
        raise asyncio.CancelledError(f"PRIVATE CANCELLATION: {text}")

    monkeypatch.setattr(AsyncWorkflowGate, "aevaluate_text", cancel)

    with pytest.raises(asyncio.CancelledError) as caught:
        await _invoke(hook, input_data, tool_use_id)

    assert str(caught.value) == ""
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert raw_marker not in _integration_traceback_locals(caught.value)
    assert "PRIVATE CANCELLATION" not in _integration_traceback_locals(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["configuration", "unexpected"])
async def test_gate_failures_are_resanitized_without_retaining_hook_input(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    raw_marker = "PRIVATE FAILED HOOK INPUT"
    gate, _ = _gate(Checkpoint.INPUT, GateAction.ALLOW)
    hook = claude_user_prompt_submit_hook(gate)

    async def fail(
        self: AsyncWorkflowGate,
        text: str,
        *,
        artifact_id: str,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ):
        del self, artifact_id, sensitivity
        if failure_kind == "configuration":
            raise BackendConfigurationError("anthropic")
        raise RuntimeError(f"PRIVATE GATE FAILURE: {text}")

    monkeypatch.setattr(AsyncWorkflowGate, "aevaluate_text", fail)
    expected_error = BackendConfigurationError if failure_kind == "configuration" else RuntimeError

    with pytest.raises(expected_error) as caught:
        await _invoke(hook, _prompt_input(raw_marker), None)

    if failure_kind == "unexpected":
        assert str(caught.value) == "PsySafe Claude Agent hook evaluation failed"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert raw_marker not in _integration_traceback_locals(caught.value)
    assert "PRIVATE GATE FAILURE" not in _integration_traceback_locals(caught.value)
