from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from agents import (
    Agent,
    FunctionTool,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
    function_tool,
)
from agents.tool_context import ToolContext

from psysafe.backends.base import BackendConfigurationError
from psysafe.core.contracts import (
    Assessment,
    Conversation,
    EvidenceDirectness,
    MessageRole,
    Outcome,
    Sensitivity,
)
from psysafe.gates import AsyncWorkflowGate, Checkpoint, GateAction, GateDecision, GatePolicy
from psysafe.integrations._serialization import IntegrationInputError
from psysafe.integrations.openai_agents import (
    openai_input_guardrail,
    openai_output_guardrail,
    openai_tool_input_guardrail,
    openai_tool_output_guardrail,
)


class RecordingAsyncClassifier:
    def __init__(self, *, cancel: bool = False) -> None:
        self.cancel = cancel
        self.seen: list[tuple[str, Sensitivity]] = []

    @property
    def classifier_id(self) -> str:
        return "integration_test"

    @property
    def policy_version(self) -> str:
        return "2026.08.2"

    @property
    def evidence_role(self) -> MessageRole | None:
        return None

    @property
    def allowed_signals(self) -> tuple[str, ...]:
        return ("policy_match",)

    @property
    def allowed_review_signals(self) -> tuple[str, ...]:
        return ()

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
        if self.cancel:
            await asyncio.sleep(0)
            raise asyncio.CancelledError("classifier-private-cancellation")
        self.seen.append((conversation.messages[target_message_index].content, sensitivity))
        return Assessment(
            classifier_id=self.classifier_id,
            policy_version=self.policy_version,
            sensitivity=sensitivity,
            outcome=Outcome.MATCHED,
            evidence_directness=EvidenceDirectness.EXPLICIT,
            signals=("policy_match",),
        )


class ConfigurationFailureClassifier(RecordingAsyncClassifier):
    async def aclassify_target(
        self,
        conversation: Conversation,
        *,
        target_message_index: int,
        sensitivity: Sensitivity = Sensitivity.BALANCED,
    ) -> Assessment:
        del conversation, target_message_index, sensitivity
        raise BackendConfigurationError("openai")


def _gate(
    checkpoint: Checkpoint,
    action: GateAction = GateAction.ALLOW,
    *,
    cancel: bool = False,
) -> tuple[AsyncWorkflowGate, RecordingAsyncClassifier]:
    classifier = RecordingAsyncClassifier(cancel=cancel)
    return (
        AsyncWorkflowGate(
            (classifier,),
            checkpoint=checkpoint,
            policy=GatePolicy(matched_action=action),
        ),
        classifier,
    )


def _agent() -> Agent[None]:
    return Agent(name="integration-test")


def _context() -> RunContextWrapper[None]:
    return RunContextWrapper(context=None)


def _tool_context(
    *,
    call_id: object = "call-1",
    arguments: object = "safe arguments",
    tool_name: str = "custom_function",
) -> ToolContext[None]:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_arguments=arguments,
    )


def _traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("psysafe/integrations/openai_agents.py"):
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


def test_factories_return_native_sdk_objects_with_blocking_input() -> None:
    input_gate, _ = _gate(Checkpoint.INPUT)
    output_gate, _ = _gate(Checkpoint.COMMUNICATION)
    tool_input_gate, _ = _gate(Checkpoint.TOOL_INPUT)
    tool_output_gate, _ = _gate(Checkpoint.TOOL_OUTPUT)

    input_check = openai_input_guardrail(input_gate)
    output_check = openai_output_guardrail(output_gate)
    tool_input_check = openai_tool_input_guardrail(tool_input_gate)
    tool_output_check = openai_tool_output_guardrail(tool_output_gate)

    assert isinstance(input_check, InputGuardrail)
    assert input_check.get_name() == "psysafe_input"
    assert input_check.run_in_parallel is False
    assert isinstance(output_check, OutputGuardrail)
    assert output_check.get_name() == "psysafe_communication"
    assert isinstance(tool_input_check, ToolInputGuardrail)
    assert tool_input_check.get_name() == "psysafe_tool_input"
    assert isinstance(tool_output_check, ToolOutputGuardrail)
    assert tool_output_check.get_name() == "psysafe_tool_output"

    attached_agent = Agent(
        name="guarded",
        input_guardrails=[input_check],
        output_guardrails=[output_check],
    )

    @function_tool(
        tool_input_guardrails=[tool_input_check],
        tool_output_guardrails=[tool_output_check],
    )
    def custom_function(value: str) -> str:
        return value

    assert attached_agent.input_guardrails == [input_check]
    assert attached_agent.output_guardrails == [output_check]
    assert isinstance(custom_function, FunctionTool)
    assert custom_function.tool_input_guardrails == [tool_input_check]
    assert custom_function.tool_output_guardrails == [tool_output_check]


@pytest.mark.parametrize("action", [GateAction.ALLOW, GateAction.REVIEW, GateAction.BLOCK])
async def test_agent_guardrails_map_allow_review_and_block_to_sdk_tripwires(action: GateAction) -> None:
    input_gate, input_classifier = _gate(Checkpoint.INPUT, action)
    output_gate, output_classifier = _gate(Checkpoint.COMMUNICATION, action)
    input_check = openai_input_guardrail(input_gate, sensitivity="precautionary")
    output_check = openai_output_guardrail(output_gate, sensitivity=Sensitivity.PRECAUTIONARY)

    input_result = await input_check.run(_agent(), "direct input", _context())
    output_result = await output_check.run(_context(), _agent(), "direct output")

    assert input_result.output.tripwire_triggered is (action is not GateAction.ALLOW)
    assert output_result.output.tripwire_triggered is (action is not GateAction.ALLOW)
    assert isinstance(input_result.output.output_info, GateDecision)
    assert isinstance(output_result.output.output_info, GateDecision)
    assert input_result.output.output_info.action is action
    assert output_result.output.output_info.action is action
    assert input_classifier.seen == [("direct input", Sensitivity.PRECAUTIONARY)]
    assert output_classifier.seen == [("direct output", Sensitivity.PRECAUTIONARY)]


@pytest.mark.parametrize("action", [GateAction.ALLOW, GateAction.REVIEW, GateAction.BLOCK])
async def test_tool_guardrails_allow_only_allow_decisions(action: GateAction) -> None:
    input_gate, input_classifier = _gate(Checkpoint.TOOL_INPUT, action)
    output_gate, output_classifier = _gate(Checkpoint.TOOL_OUTPUT, action)
    input_check = openai_tool_input_guardrail(input_gate)
    output_check = openai_tool_output_guardrail(output_gate)
    tool_context = _tool_context()

    input_result = await input_check.run(ToolInputGuardrailData(context=tool_context, agent=_agent()))
    output_result = await output_check.run(
        ToolOutputGuardrailData(context=tool_context, agent=_agent(), output="safe output"),
    )

    expected_behavior = "allow" if action is GateAction.ALLOW else "raise_exception"
    assert input_result.behavior == {"type": expected_behavior}
    assert output_result.behavior == {"type": expected_behavior}
    assert isinstance(input_result.output_info, GateDecision)
    assert isinstance(output_result.output_info, GateDecision)
    assert input_result.output_info.action is action
    assert output_result.output_info.action is action
    assert json.loads(input_classifier.seen[0][0]) == {
        "tool_input": "safe arguments",
        "tool_name": "custom_function",
    }
    assert input_classifier.seen[0][1] is Sensitivity.BALANCED
    assert json.loads(output_classifier.seen[0][0]) == {
        "tool_input": "safe arguments",
        "tool_name": "custom_function",
        "tool_output": "safe output",
    }
    assert output_classifier.seen[0][1] is Sensitivity.BALANCED


async def test_non_string_sdk_values_use_strict_canonical_json() -> None:
    input_gate, input_classifier = _gate(Checkpoint.INPUT)
    output_gate, output_classifier = _gate(Checkpoint.COMMUNICATION)
    tool_input_gate, tool_input_classifier = _gate(Checkpoint.TOOL_INPUT)
    tool_output_gate, tool_output_classifier = _gate(Checkpoint.TOOL_OUTPUT)

    input_check = openai_input_guardrail(input_gate)
    output_check = openai_output_guardrail(output_gate)
    tool_input_check = openai_tool_input_guardrail(tool_input_gate)
    tool_output_check = openai_tool_output_guardrail(tool_output_gate)

    await input_check.run(
        _agent(),
        [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        _context(),
    )
    await output_check.run(_context(), _agent(), {"b": 2, "a": 1})
    await tool_input_check.run(
        ToolInputGuardrailData(
            context=_tool_context(arguments={"b": 2, "a": 1}),
            agent=_agent(),
        ),
    )
    await tool_output_check.run(
        ToolOutputGuardrailData(
            context=_tool_context(),
            agent=_agent(),
            output=("one", 2, True),
        ),
    )

    assert input_classifier.seen[0][0] == '[{"content":[{"text":"hello","type":"input_text"}],"role":"user"}]'
    assert output_classifier.seen[0][0] == '{"a":1,"b":2}'
    assert json.loads(tool_input_classifier.seen[0][0]) == {
        "tool_input": {"a": 1, "b": 2},
        "tool_name": "custom_function",
    }
    assert json.loads(tool_output_classifier.seen[0][0]) == {
        "tool_input": "safe arguments",
        "tool_name": "custom_function",
        "tool_output": ["one", 2, True],
    }


@pytest.mark.parametrize(
    ("modality", "reference_field"),
    [("input_image", "image_url"), ("input_file", "file_id")],
)
async def test_input_guardrail_rejects_opaque_media_without_classifying_references(
    modality: str,
    reference_field: str,
) -> None:
    private_reference = f"private-{modality}-reference"
    gate, classifier = _gate(Checkpoint.INPUT)
    check = openai_input_guardrail(gate)
    agent_input = [
        {
            "role": "user",
            "content": [{"type": modality, reference_field: private_reference}],
        },
    ]

    with pytest.raises(IntegrationInputError) as caught:
        await check.run(_agent(), agent_input, _context())  # type: ignore[arg-type]

    assert classifier.seen == []
    assert str(caught.value) == "integration input is invalid"
    assert private_reference not in _traceback_locals(caught.value)


@pytest.mark.parametrize(
    ("modality", "reference_field"),
    [("image", "image_url"), ("file", "file_id")],
)
async def test_tool_output_guardrail_rejects_opaque_media_without_classifying_references(
    modality: str,
    reference_field: str,
) -> None:
    private_reference = f"private-{modality}-tool-output"
    gate, classifier = _gate(Checkpoint.TOOL_OUTPUT)
    check = openai_tool_output_guardrail(gate)

    with pytest.raises(IntegrationInputError) as caught:
        await check.run(
            ToolOutputGuardrailData(
                context=_tool_context(),
                agent=_agent(),
                output={"type": modality, reference_field: private_reference},
            ),
        )

    assert classifier.seen == []
    assert str(caught.value) == "integration input is invalid"
    assert private_reference not in _traceback_locals(caught.value)


@pytest.mark.parametrize(
    ("factory", "checkpoint"),
    [
        (openai_input_guardrail, Checkpoint.INPUT),
        (openai_output_guardrail, Checkpoint.COMMUNICATION),
        (openai_tool_input_guardrail, Checkpoint.TOOL_INPUT),
        (openai_tool_output_guardrail, Checkpoint.TOOL_OUTPUT),
    ],
)
def test_factories_require_the_exact_checkpoint(
    factory: Callable[..., object],
    checkpoint: Checkpoint,
) -> None:
    wrong_checkpoint = Checkpoint.COMMUNICATION if checkpoint is not Checkpoint.COMMUNICATION else Checkpoint.INPUT
    gate, _ = _gate(wrong_checkpoint)

    with pytest.raises(ValueError, match=checkpoint.value):
        factory(gate)

    with pytest.raises(TypeError, match="AsyncWorkflowGate"):
        factory(object())


def test_factory_rejects_gate_subclasses_without_dispatching_overrides() -> None:
    class HostileGate(AsyncWorkflowGate):
        @property
        def checkpoint(self) -> Checkpoint:
            raise AssertionError("overridden checkpoint must not run")

        async def aevaluate_text(
            self,
            text: str,
            *,
            artifact_id: str,
            sensitivity: Sensitivity = Sensitivity.BALANCED,
        ) -> GateDecision:
            raise AssertionError(f"overridden evaluation must not receive {text!r} or {artifact_id!r}")

    hostile = object.__new__(HostileGate)

    with pytest.raises(TypeError, match="AsyncWorkflowGate"):
        openai_input_guardrail(hostile)


@pytest.mark.parametrize(
    "invalid",
    ["low", "medium", "high", "BALANCED", " balanced ", 1, object()],
)
def test_sensitivity_rejects_legacy_or_non_named_boundaries(invalid: object) -> None:
    gate, _ = _gate(Checkpoint.INPUT)

    with pytest.raises(ValueError, match="precise, balanced, or precautionary"):
        openai_input_guardrail(gate, sensitivity=invalid)  # type: ignore[arg-type]


def test_sensitivity_rejects_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    gate, _ = _gate(Checkpoint.INPUT)
    with pytest.raises(ValueError, match="precise, balanced, or precautionary"):
        openai_input_guardrail(gate, sensitivity=StringSubclass("balanced"))


async def test_agent_artifact_ids_are_fresh_opaque_and_content_independent() -> None:
    private_content = "customer-secret-content"
    input_gate, _ = _gate(Checkpoint.INPUT)
    output_gate, _ = _gate(Checkpoint.COMMUNICATION)
    input_check = openai_input_guardrail(input_gate)
    output_check = openai_output_guardrail(output_gate)

    first_input = (await input_check.run(_agent(), private_content, _context())).output.output_info
    second_input = (await input_check.run(_agent(), private_content, _context())).output.output_info
    first_output = (await output_check.run(_context(), _agent(), private_content)).output.output_info
    second_output = (await output_check.run(_context(), _agent(), private_content)).output.output_info

    ids = (
        first_input.artifact_id,
        second_input.artifact_id,
        first_output.artifact_id,
        second_output.artifact_id,
    )
    assert len(set(ids)) == len(ids)
    assert all(private_content not in artifact_id for artifact_id in ids)
    assert private_content not in repr(first_input)
    assert private_content not in repr(first_output)


async def test_tool_input_artifact_ids_bind_the_call_and_exact_payload() -> None:
    raw_call_id = "private-call-id-123"
    private_arguments = "private tool arguments"
    gate, _ = _gate(Checkpoint.TOOL_INPUT)
    check = openai_tool_input_guardrail(gate)
    independent_check = openai_tool_input_guardrail(gate)

    async def evaluate(
        guardrail: ToolInputGuardrail[Any],
        call_id: str,
        *,
        arguments: object = private_arguments,
        tool_name: str = "custom_function",
    ) -> GateDecision:
        result = await guardrail.run(
            ToolInputGuardrailData(
                context=_tool_context(call_id=call_id, arguments=arguments, tool_name=tool_name),
                agent=_agent(),
            ),
        )
        assert isinstance(result.output_info, GateDecision)
        return result.output_info

    first = await evaluate(check, raw_call_id)
    repeated = await evaluate(check, raw_call_id)
    changed_arguments = await evaluate(check, raw_call_id, arguments="changed private arguments")
    changed_tool = await evaluate(check, raw_call_id, tool_name="destructive_function")
    other_call = await evaluate(check, "another-private-call-id")
    other_factory = await evaluate(independent_check, raw_call_id)

    assert first.artifact_id == repeated.artifact_id
    assert first.artifact_id != changed_arguments.artifact_id
    assert first.artifact_id != changed_tool.artifact_id
    assert first.artifact_id != other_call.artifact_id
    assert first.artifact_id != other_factory.artifact_id
    for decision in (first, repeated, changed_arguments, changed_tool, other_call, other_factory):
        assert raw_call_id not in decision.artifact_id
        assert private_arguments not in decision.artifact_id


async def test_tool_output_artifact_ids_bind_input_and_output_versions() -> None:
    raw_call_id = "private-output-call-id"
    gate, _ = _gate(Checkpoint.TOOL_OUTPUT)
    check = openai_tool_output_guardrail(gate)

    async def evaluate(*, arguments: object, output: object) -> GateDecision:
        result = await check.run(
            ToolOutputGuardrailData(
                context=_tool_context(call_id=raw_call_id, arguments=arguments),
                agent=_agent(),
                output=output,
            ),
        )
        assert isinstance(result.output_info, GateDecision)
        return result.output_info

    first = await evaluate(arguments={"recipient": "private-a"}, output={"status": "private-done"})
    repeated = await evaluate(arguments={"recipient": "private-a"}, output={"status": "private-done"})
    changed_input = await evaluate(arguments={"recipient": "private-b"}, output={"status": "private-done"})
    changed_output = await evaluate(arguments={"recipient": "private-a"}, output={"status": "private-failed"})

    assert first.artifact_id == repeated.artifact_id
    assert first.artifact_id != changed_input.artifact_id
    assert first.artifact_id != changed_output.artifact_id
    assert all(raw_call_id not in decision.artifact_id for decision in (first, repeated, changed_input, changed_output))


@pytest.mark.parametrize("invalid_call_id", ["", "x" * 4_097, "\ud800", 123, object()])
async def test_tool_call_ids_are_strictly_validated_without_echo(invalid_call_id: object) -> None:
    gate, _ = _gate(Checkpoint.TOOL_INPUT)
    check = openai_tool_input_guardrail(gate)

    with pytest.raises(IntegrationInputError) as caught:
        await check.run(
            ToolInputGuardrailData(
                context=_tool_context(call_id=invalid_call_id, arguments="private arguments"),
                agent=_agent(),
            ),
        )

    assert str(caught.value) == "integration input is invalid"
    assert "private arguments" not in _traceback_locals(caught.value)
    assert repr(invalid_call_id) not in str(caught.value)


async def test_unsupported_values_raise_fresh_data_free_integration_errors() -> None:
    private_value = object()
    input_gate, _ = _gate(Checkpoint.INPUT)
    output_gate, _ = _gate(Checkpoint.COMMUNICATION)
    tool_input_gate, _ = _gate(Checkpoint.TOOL_INPUT)
    tool_output_gate, _ = _gate(Checkpoint.TOOL_OUTPUT)
    operations: tuple[Callable[[], Awaitable[object]], ...] = (
        lambda: openai_input_guardrail(input_gate).run(_agent(), private_value, _context()),  # type: ignore[arg-type]
        lambda: openai_output_guardrail(output_gate).run(_context(), _agent(), private_value),
        lambda: openai_tool_input_guardrail(tool_input_gate).run(
            ToolInputGuardrailData(
                context=_tool_context(arguments=private_value),
                agent=_agent(),
            ),
        ),
        lambda: openai_tool_output_guardrail(tool_output_gate).run(
            ToolOutputGuardrailData(
                context=_tool_context(),
                agent=_agent(),
                output=private_value,
            ),
        ),
    )

    for operation in operations:
        with pytest.raises(IntegrationInputError) as caught:
            await operation()
        assert str(caught.value) == "integration input is invalid"
        assert caught.value.__cause__ is None


async def test_tool_guardrails_reject_non_sdk_data_objects_before_dispatch() -> None:
    input_gate, _ = _gate(Checkpoint.TOOL_INPUT)
    output_gate, _ = _gate(Checkpoint.TOOL_OUTPUT)

    with pytest.raises(IntegrationInputError):
        await openai_tool_input_guardrail(input_gate).run(object())  # type: ignore[arg-type]
    with pytest.raises(IntegrationInputError):
        await openai_tool_output_guardrail(output_gate).run(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("checkpoint", "operation"),
    [
        (
            Checkpoint.INPUT,
            lambda check, secret: check.run(_agent(), secret, _context()),
        ),
        (
            Checkpoint.COMMUNICATION,
            lambda check, secret: check.run(_context(), _agent(), secret),
        ),
        (
            Checkpoint.TOOL_INPUT,
            lambda check, secret: check.run(
                ToolInputGuardrailData(
                    context=_tool_context(arguments=secret),
                    agent=_agent(),
                ),
            ),
        ),
        (
            Checkpoint.TOOL_OUTPUT,
            lambda check, secret: check.run(
                ToolOutputGuardrailData(
                    context=_tool_context(),
                    agent=_agent(),
                    output=secret,
                ),
            ),
        ),
    ],
)
async def test_cancellation_is_fresh_and_does_not_retain_payloads(
    checkpoint: Checkpoint,
    operation: Callable[[Any, str], Awaitable[object]],
) -> None:
    secret = f"private-{checkpoint.value}-payload"
    gate, _ = _gate(checkpoint, cancel=True)
    factories = {
        Checkpoint.INPUT: openai_input_guardrail,
        Checkpoint.COMMUNICATION: openai_output_guardrail,
        Checkpoint.TOOL_INPUT: openai_tool_input_guardrail,
        Checkpoint.TOOL_OUTPUT: openai_tool_output_guardrail,
    }
    check = factories[checkpoint](gate)

    with pytest.raises(asyncio.CancelledError) as caught:
        await operation(check, secret)

    assert caught.value.args == ()
    assert caught.value.__cause__ is None
    assert secret not in _traceback_locals(caught.value)


async def test_sanitized_gate_errors_do_not_retain_raw_sdk_input() -> None:
    private_input = "private-provider-error-input"
    gate = AsyncWorkflowGate(
        (ConfigurationFailureClassifier(),),
        checkpoint=Checkpoint.INPUT,
        policy=GatePolicy(matched_action=GateAction.BLOCK),
    )
    check = openai_input_guardrail(gate)

    with pytest.raises(BackendConfigurationError) as caught:
        await check.run(_agent(), private_input, _context())

    assert str(caught.value) == "provider support requires `pip install 'psysafe-ai[openai]'`"
    assert private_input not in _traceback_locals(caught.value)
