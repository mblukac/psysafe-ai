"""OpenAI Responses API backend using native Pydantic structured output."""

from __future__ import annotations

from typing import Any

from psysafe.backends._errors import sanitized_backend_error
from psysafe.backends.base import (
    BackendConfigurationError,
    BackendError,
    BackendInvalidResponseError,
    BackendRefusalError,
    OutputT,
    _fresh_backend_error,
    _fresh_configuration_error,
    _raise_backend_error,
    _raise_configuration_error,
    _safe_identifier,
)


class OpenAIBackend:
    """Classify with ``responses.parse`` and an explicit model.

    ``instructions`` and untrusted ``input`` remain distinct API fields, and
    Responses storage is disabled for every request made by this adapter.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        async_client: Any | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self._model = _safe_identifier(model, field="model")
        self._client = client
        self._async_client = async_client
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout

    @property
    def provider(self) -> str:
        return "openai"

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
        configuration_failure: BackendConfigurationError | None = None
        response: object | None = None
        result: OutputT | None = None
        try:
            response = self._sync_client().responses.parse(
                **self._request(
                    instructions=instructions,
                    input_text=input_text,
                    output_type=output_type,
                ),
            )
            result = self._parsed(response, output_type)
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        # Provider SDK exceptions are intentionally collapsed to safe categories.
        except Exception as caught:  # noqa: BLE001
            failure = sanitized_backend_error(caught)
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
        configuration_failure: BackendConfigurationError | None = None
        response: object | None = None
        result: OutputT | None = None
        try:
            response = await self._async_openai_client().responses.parse(
                **self._request(
                    instructions=instructions,
                    input_text=input_text,
                    output_type=output_type,
                ),
            )
            result = self._parsed(response, output_type)
        except BackendConfigurationError as caught:
            configuration_failure = _fresh_configuration_error(caught)
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        # Provider SDK exceptions are intentionally collapsed to safe categories.
        except Exception as caught:  # noqa: BLE001
            failure = sanitized_backend_error(caught)
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
            "instructions": instructions,
            "input": input_text,
            "text_format": output_type,
            "store": False,
        }
        if self._max_output_tokens is not None:
            request["max_output_tokens"] = self._max_output_tokens
        if self._timeout is not None:
            request["timeout"] = self._timeout
        return request

    @staticmethod
    def _parsed(response: object | None, output_type: type[OutputT]) -> OutputT:
        if _openai_refused(response):
            raise BackendRefusalError from None
        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, output_type):
            raise BackendInvalidResponseError from None
        return parsed

    def _sync_client(self) -> Any:
        if self._client is None:
            self._client = _create_openai_client(asynchronous=False)
        return self._client

    def _async_openai_client(self) -> Any:
        if self._async_client is None:
            self._async_client = _create_openai_client(asynchronous=True)
        return self._async_client


def _create_openai_client(*, asynchronous: bool) -> Any:
    unavailable = False
    client_type: Any | None = None
    try:
        if asynchronous:
            from openai import AsyncOpenAI

            client_type = AsyncOpenAI
        else:
            from openai import OpenAI

            client_type = OpenAI
    except (ImportError, AttributeError):
        unavailable = True
    if unavailable or client_type is None:
        raise BackendConfigurationError("openai") from None
    return client_type()


def _openai_refused(response: object | None) -> bool:
    if response is None:
        return False
    for output_item in getattr(response, "output", ()) or ():
        for content_item in getattr(output_item, "content", ()) or ():
            if getattr(content_item, "type", None) == "refusal":
                return True
    incomplete_details = getattr(response, "incomplete_details", None)
    return getattr(incomplete_details, "reason", None) == "content_filter"


__all__ = ["OpenAIBackend"]
