from types import SimpleNamespace

import pytest
from pydantic import BaseModel, model_validator

from psysafe.backends import (
    AnthropicBackend,
    BackendInvalidResponseError,
    BackendProviderError,
    BackendRefusalError,
    BackendTimeoutError,
    OpenAIBackend,
)
from psysafe.classifiers.base import Finding, Observation
from psysafe.core.contracts import EvidenceDirectness

ObservationModel = Observation[Finding]


def _library_traceback_locals(error: BaseException) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if "/psysafe/" in traceback.tb_frame.f_code.co_filename:
            values.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(values)


def _observation() -> Observation[Finding]:
    return ObservationModel(
        findings=(
            Finding(
                signal="test_signal",
                directness=EvidenceDirectness.EXPLICIT,
                message_ids=("m0",),
            ),
        ),
        insufficient_context=False,
    )


class _SyncEndpoint:
    def __init__(self, response=None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _AsyncEndpoint(_SyncEndpoint):
    async def parse(self, **kwargs: object):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def test_openai_uses_responses_parse_with_separate_instructions_and_input() -> None:
    endpoint = _SyncEndpoint(SimpleNamespace(output_parsed=_observation(), output=[], status="completed"))
    client = SimpleNamespace(responses=endpoint, api_key="must-not-leak")
    backend = OpenAIBackend(model="explicit-openai-model", client=client)

    result = backend.complete(
        instructions="fixed policy",
        input_text='{"messages":[{"content":"untrusted"}]}',
        output_type=ObservationModel,
    )

    assert result == _observation()
    assert endpoint.calls == [
        {
            "model": "explicit-openai-model",
            "instructions": "fixed policy",
            "input": '{"messages":[{"content":"untrusted"}]}',
            "text_format": ObservationModel,
            "store": False,
        },
    ]
    assert backend.provider == "openai"
    assert backend.model == "explicit-openai-model"
    assert "must-not-leak" not in repr(backend)


@pytest.mark.asyncio
async def test_openai_async_request_matches_sync_shape() -> None:
    endpoint = _AsyncEndpoint(SimpleNamespace(output_parsed=_observation(), output=[], status="completed"))
    client = SimpleNamespace(responses=endpoint)
    backend = OpenAIBackend(model="explicit-openai-model", async_client=client)

    result = await backend.acomplete(
        instructions="fixed policy",
        input_text="untrusted JSON",
        output_type=ObservationModel,
    )

    assert result == _observation()
    assert endpoint.calls[0]["instructions"] == "fixed policy"
    assert endpoint.calls[0]["input"] == "untrusted JSON"
    assert endpoint.calls[0]["text_format"] is ObservationModel


def test_anthropic_uses_messages_parse_and_top_level_system() -> None:
    endpoint = _SyncEndpoint(SimpleNamespace(parsed_output=_observation(), content=[], stop_reason="end_turn"))
    client = SimpleNamespace(messages=endpoint, api_key="must-not-leak")
    backend = AnthropicBackend(model="explicit-anthropic-model", client=client)

    result = backend.complete(
        instructions="fixed policy",
        input_text="untrusted JSON",
        output_type=ObservationModel,
    )

    assert result == _observation()
    assert endpoint.calls == [
        {
            "model": "explicit-anthropic-model",
            "max_tokens": 16384,
            "system": "fixed policy",
            "messages": [{"role": "user", "content": "untrusted JSON"}],
            "output_format": ObservationModel,
        },
    ]
    assert backend.provider == "anthropic"
    assert backend.model == "explicit-anthropic-model"
    assert "must-not-leak" not in repr(backend)


@pytest.mark.asyncio
async def test_anthropic_async_request_matches_sync_shape() -> None:
    endpoint = _AsyncEndpoint(SimpleNamespace(parsed_output=_observation(), content=[], stop_reason="end_turn"))
    client = SimpleNamespace(messages=endpoint)
    backend = AnthropicBackend(model="explicit-anthropic-model", async_client=client)

    result = await backend.acomplete(
        instructions="fixed policy",
        input_text="untrusted JSON",
        output_type=ObservationModel,
    )

    assert result == _observation()
    assert endpoint.calls[0]["system"] == "fixed policy"
    assert endpoint.calls[0]["messages"] == [{"role": "user", "content": "untrusted JSON"}]
    assert endpoint.calls[0]["output_format"] is ObservationModel


@pytest.mark.parametrize(
    ("backend_type", "response"),
    [
        (
            OpenAIBackend,
            SimpleNamespace(
                output_parsed=_observation(),
                output=(),
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            ),
        ),
        (
            AnthropicBackend,
            SimpleNamespace(
                parsed_output=_observation(),
                content=(),
                stop_reason="max_tokens",
            ),
        ),
        (
            AnthropicBackend,
            SimpleNamespace(
                parsed_output=_observation(),
                content=(),
                stop_reason="model_context_window_exceeded",
            ),
        ),
    ],
)
def test_static_provider_parsers_reject_declared_truncation(backend_type, response) -> None:
    with pytest.raises(BackendInvalidResponseError):
        backend_type._parsed(response, ObservationModel)


@pytest.mark.parametrize("status", [None, "failed", "cancelled", "in_progress", "queued"])
def test_openai_parser_rejects_parsed_nonterminal_or_failed_responses(status: str | None) -> None:
    response = SimpleNamespace(output_parsed=_observation(), output=(), status=status)

    with pytest.raises(BackendInvalidResponseError):
        OpenAIBackend._parsed(response, ObservationModel)


@pytest.mark.parametrize("stop_reason", [None, "pause_turn", "tool_use", "stop_sequence"])
def test_anthropic_parser_requires_a_complete_structured_turn(stop_reason: str | None) -> None:
    response = SimpleNamespace(parsed_output=_observation(), content=(), stop_reason=stop_reason)

    with pytest.raises(BackendInvalidResponseError):
        AnthropicBackend._parsed(response, ObservationModel)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_sync_provider_backends_reject_parsed_but_truncated_output(provider: str) -> None:
    if provider == "openai":
        response = SimpleNamespace(
            output_parsed=_observation(),
            output=(),
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        backend = OpenAIBackend(
            model="model",
            client=SimpleNamespace(responses=_SyncEndpoint(response)),
        )
    else:
        response = SimpleNamespace(
            parsed_output=_observation(),
            content=(),
            stop_reason="max_tokens",
        )
        backend = AnthropicBackend(
            model="model",
            client=SimpleNamespace(messages=_SyncEndpoint(response)),
        )

    with pytest.raises(BackendInvalidResponseError):
        backend.complete(instructions="fixed", input_text="private", output_type=ObservationModel)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_async_provider_backends_reject_parsed_but_truncated_output(provider: str) -> None:
    if provider == "openai":
        response = SimpleNamespace(
            output_parsed=_observation(),
            output=(),
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        backend = OpenAIBackend(
            model="model",
            async_client=SimpleNamespace(responses=_AsyncEndpoint(response)),
        )
    else:
        response = SimpleNamespace(
            parsed_output=_observation(),
            content=(),
            stop_reason="max_tokens",
        )
        backend = AnthropicBackend(
            model="model",
            async_client=SimpleNamespace(messages=_AsyncEndpoint(response)),
        )

    with pytest.raises(BackendInvalidResponseError):
        await backend.acomplete(instructions="fixed", input_text="private", output_type=ObservationModel)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_sync_provider_backends_reject_other_incomplete_states(provider: str) -> None:
    if provider == "openai":
        response = SimpleNamespace(output_parsed=_observation(), output=(), status="failed")
        backend = OpenAIBackend(model="model", client=SimpleNamespace(responses=_SyncEndpoint(response)))
    else:
        response = SimpleNamespace(parsed_output=_observation(), content=(), stop_reason="tool_use")
        backend = AnthropicBackend(model="model", client=SimpleNamespace(messages=_SyncEndpoint(response)))

    with pytest.raises(BackendInvalidResponseError):
        backend.complete(instructions="fixed", input_text="private", output_type=ObservationModel)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_async_provider_backends_reject_other_incomplete_states(provider: str) -> None:
    if provider == "openai":
        response = SimpleNamespace(output_parsed=_observation(), output=(), status="queued")
        backend = OpenAIBackend(model="model", async_client=SimpleNamespace(responses=_AsyncEndpoint(response)))
    else:
        response = SimpleNamespace(parsed_output=_observation(), content=(), stop_reason="pause_turn")
        backend = AnthropicBackend(model="model", async_client=SimpleNamespace(messages=_AsyncEndpoint(response)))

    with pytest.raises(BackendInvalidResponseError):
        await backend.acomplete(instructions="fixed", input_text="private", output_type=ObservationModel)


@pytest.mark.parametrize(
    ("backend", "error_type"),
    [
        (
            OpenAIBackend(
                model="model",
                client=SimpleNamespace(
                    responses=_SyncEndpoint(
                        SimpleNamespace(
                            output_parsed=None,
                            output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="raw")])],
                        ),
                    ),
                ),
            ),
            BackendRefusalError,
        ),
        (
            AnthropicBackend(
                model="model",
                client=SimpleNamespace(
                    messages=_SyncEndpoint(
                        SimpleNamespace(parsed_output=None, content=[], stop_reason="refusal"),
                    ),
                ),
            ),
            BackendRefusalError,
        ),
        (
            OpenAIBackend(
                model="model",
                client=SimpleNamespace(responses=_SyncEndpoint(SimpleNamespace(output_parsed=None, output=[]))),
            ),
            BackendInvalidResponseError,
        ),
        (
            OpenAIBackend(
                model="model",
                client=SimpleNamespace(responses=_SyncEndpoint(error=TimeoutError("raw request"))),
            ),
            BackendTimeoutError,
        ),
        (
            AnthropicBackend(
                model="model",
                client=SimpleNamespace(messages=_SyncEndpoint(error=RuntimeError("api_key=secret"))),
            ),
            BackendProviderError,
        ),
    ],
)
def test_provider_failures_are_sanitized_without_original_exception(backend, error_type) -> None:
    with pytest.raises(error_type) as caught:
        backend.complete(instructions="fixed", input_text="private", output_type=ObservationModel)

    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_provider_backends_require_an_explicit_nonblank_model() -> None:
    with pytest.raises(ValueError, match="model must contain"):
        OpenAIBackend(model="  ")
    with pytest.raises(ValueError, match="model must contain"):
        AnthropicBackend(model="  ")


def test_callable_provider_identifier_fits_assessment_metadata_contract() -> None:
    from psysafe.backends import CallableBackend

    with pytest.raises(ValueError, match="provider must contain between 1 and 80"):
        CallableBackend(lambda **_: _observation(), provider="p" * 81)


class _ExplodingOutput(BaseModel):
    @model_validator(mode="after")
    def explode(self):
        raise RuntimeError("SECRET VALIDATOR INPUT")


def test_callable_backend_sanitizes_unexpected_schema_validator_failures() -> None:
    from psysafe.backends import CallableBackend

    backend = CallableBackend(lambda **_: {})

    with pytest.raises(BackendInvalidResponseError) as caught:
        backend.complete(instructions="fixed", input_text="PRIVATE INPUT", output_type=_ExplodingOutput)

    assert "SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    library_locals = _library_traceback_locals(caught.value)
    assert "PRIVATE INPUT" not in library_locals
    assert "SECRET VALIDATOR INPUT" not in library_locals


class _ExplodingOpenAIResponse:
    output = ()
    status = "completed"

    @property
    def output_parsed(self):
        raise RuntimeError("SECRET RESPONSE PROPERTY")


def test_provider_postprocessing_is_inside_the_sanitized_boundary() -> None:
    backend = OpenAIBackend(
        model="model",
        client=SimpleNamespace(responses=_SyncEndpoint(_ExplodingOpenAIResponse())),
    )

    with pytest.raises(BackendProviderError) as caught:
        backend.complete(instructions="fixed", input_text="PRIVATE INPUT", output_type=ObservationModel)

    assert "SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    library_locals = _library_traceback_locals(caught.value)
    assert "PRIVATE INPUT" not in library_locals
    assert "SECRET RESPONSE PROPERTY" not in library_locals
