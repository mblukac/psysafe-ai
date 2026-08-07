"""Anthropic Messages API backend using native Pydantic structured output."""

from __future__ import annotations

import asyncio
from typing import Any

from psysafe.backends._errors import sanitized_backend_error
from psysafe.backends.base import (
    BackendConfigurationError,
    BackendError,
    BackendInvalidResponseError,
    BackendProviderError,
    BackendRefusalError,
    OutputT,
    _fresh_backend_error,
    _fresh_configuration_error,
    _raise_backend_error,
    _raise_cancelled,
    _raise_configuration_error,
    _safe_identifier,
)


class AnthropicBackend:
    """Classify with ``messages.parse`` and an explicit model."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        async_client: Any | None = None,
        max_tokens: int = 16384,
        timeout: float | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self._model = _safe_identifier(model, field="model")
        self._client = client
        self._async_client = async_client
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> OutputT:
        failure: BackendError | None = None
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        response: object | None = None
        result: OutputT | None = None
        try:
            response = self._sync_client().messages.parse(
                **self._request(
                    instructions=instructions,
                    input_text=input_text,
                    output_type=output_type,
                ),
            )
            result = self._parsed(response, output_type)
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
            if configuration_failure is None:
                failure = BackendProviderError()
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        # Provider SDK exceptions are intentionally collapsed to safe categories.
        except Exception as caught:  # noqa: BLE001
            failure = sanitized_backend_error(caught)
        if cancelled:
            del self, instructions, input_text, output_type, response, result, failure, configuration_failure
            _raise_cancelled()
        if configuration_failure is not None:
            del self, instructions, input_text, output_type, response, result, failure
            _raise_configuration_error(configuration_failure)
        if failure is not None:
            del self, instructions, input_text, output_type, response, result, configuration_failure
            _raise_backend_error(failure)
        if result is None:
            failure = BackendInvalidResponseError()
            del self, instructions, input_text, output_type, response, result, configuration_failure
            _raise_backend_error(failure)
        return result

    async def acomplete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> OutputT:
        failure: BackendError | None = None
        cancelled = False
        configuration_failure: BackendConfigurationError | None = None
        response: object | None = None
        result: OutputT | None = None
        try:
            response = await self._async_anthropic_client().messages.parse(
                **self._request(
                    instructions=instructions,
                    input_text=input_text,
                    output_type=output_type,
                ),
            )
            result = self._parsed(response, output_type)
        except asyncio.CancelledError:
            cancelled = True
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
            if configuration_failure is None:
                failure = BackendProviderError()
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        # Provider SDK exceptions are intentionally collapsed to safe categories.
        except Exception as caught:  # noqa: BLE001
            failure = sanitized_backend_error(caught)
        if cancelled:
            del self, instructions, input_text, output_type, response, result, failure, configuration_failure
            _raise_cancelled()
        if configuration_failure is not None:
            del self, instructions, input_text, output_type, response, result, failure
            _raise_configuration_error(configuration_failure)
        if failure is not None:
            del self, instructions, input_text, output_type, response, result, configuration_failure
            _raise_backend_error(failure)
        if result is None:
            failure = BackendInvalidResponseError()
            del self, instructions, input_text, output_type, response, result, configuration_failure
            _raise_backend_error(failure)
        return result

    def _request(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": instructions,
            "messages": [{"role": "user", "content": input_text}],
            "output_format": output_type,
        }
        if self._timeout is not None:
            request["timeout"] = self._timeout
        return request

    @staticmethod
    def _parsed(response: object | None, output_type: type[OutputT]) -> OutputT:
        if _anthropic_refused(response):
            raise BackendRefusalError from None
        if response is None or getattr(response, "stop_reason", None) != "end_turn":
            raise BackendInvalidResponseError from None
        parsed = getattr(response, "parsed_output", None)
        if not isinstance(parsed, output_type):
            raise BackendInvalidResponseError from None
        return parsed

    def _sync_client(self) -> Any:
        if self._client is None:
            self._client = _create_anthropic_client(asynchronous=False)
        return self._client

    def _async_anthropic_client(self) -> Any:
        if self._async_client is None:
            self._async_client = _create_anthropic_client(asynchronous=True)
        return self._async_client


def _create_anthropic_client(*, asynchronous: bool) -> Any:
    unavailable = False
    client_type: Any | None = None
    try:
        if asynchronous:
            from anthropic import AsyncAnthropic

            client_type = AsyncAnthropic
        else:
            from anthropic import Anthropic

            client_type = Anthropic
    except (ImportError, AttributeError):
        unavailable = True
    if unavailable or client_type is None:
        raise BackendConfigurationError("anthropic") from None
    return client_type()


def _anthropic_refused(response: object | None) -> bool:
    if response is None:
        return False
    if getattr(response, "stop_reason", None) == "refusal":
        return True
    return any(getattr(block, "type", None) == "refusal" for block in (getattr(response, "content", ()) or ()))


__all__ = ["AnthropicBackend"]
