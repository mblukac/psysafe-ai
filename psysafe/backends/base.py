"""Provider-neutral structured-classification backends.

Backends receive fixed policy instructions and untrusted input as separate
arguments.  They return a validated Pydantic model or raise a sanitized error;
raw provider responses and exception text never become part of the public
contract.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import NoReturn, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from psysafe.core.contracts import IndeterminateReason

OutputT = TypeVar("OutputT", bound=BaseModel)


class BackendError(RuntimeError):
    """A sanitized structured-backend failure."""

    reason = IndeterminateReason.PROVIDER_ERROR

    def __init__(self) -> None:
        super().__init__(f"structured classification failed ({self.reason.value})")


class BackendRefusalError(BackendError):
    """The provider refused to perform the classification."""

    reason = IndeterminateReason.REFUSED


class BackendTimeoutError(BackendError):
    """The provider did not complete the classification in time."""

    reason = IndeterminateReason.TIMEOUT


class BackendInvalidResponseError(BackendError):
    """The provider response did not satisfy the requested schema."""

    reason = IndeterminateReason.INVALID_RESPONSE


class BackendProviderError(BackendError):
    """The provider failed for a reason safe to expose only categorically."""

    reason = IndeterminateReason.PROVIDER_ERROR


class BackendConfigurationError(RuntimeError):
    """A safe, actionable missing-provider configuration error."""

    def __init__(self, extra: str) -> None:
        if extra not in {"anthropic", "openai"}:
            raise ValueError("unknown provider extra")
        self.extra = extra
        super().__init__(f"provider support requires `pip install 'psysafe-ai[{extra}]'`")


def _fresh_backend_error(error: BackendError) -> BackendError:
    """Copy only a known categorical reason, dropping causes and custom state."""

    error_types: dict[IndeterminateReason, type[BackendError]] = {
        IndeterminateReason.REFUSED: BackendRefusalError,
        IndeterminateReason.TIMEOUT: BackendTimeoutError,
        IndeterminateReason.INVALID_RESPONSE: BackendInvalidResponseError,
        IndeterminateReason.PROVIDER_ERROR: BackendProviderError,
    }
    return error_types.get(error.reason, BackendProviderError)()


def _fresh_configuration_error(error: BackendConfigurationError) -> BackendConfigurationError:
    return BackendConfigurationError(error.extra)


def _raise_backend_error(error: BackendError) -> NoReturn:
    """Raise from a frame containing categorical state only."""

    raise _fresh_backend_error(error) from None


def _raise_configuration_error(error: BackendConfigurationError) -> NoReturn:
    """Raise an actionable fixed message without retaining integration frames."""

    raise _fresh_configuration_error(error) from None


@runtime_checkable
class StructuredBackend(Protocol):
    """Sync and async backend contract for provider-native structured output."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> OutputT: ...

    async def acomplete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> OutputT: ...


StructuredHandler = Callable[..., BaseModel | Mapping[str, object]]
AsyncStructuredHandler = Callable[..., Awaitable[BaseModel | Mapping[str, object]]]


def _validated_output(value: object, output_type: type[OutputT]) -> OutputT:
    """Validate callable output without including invalid values in errors."""

    if isinstance(value, output_type):
        return value
    if isinstance(value, Mapping):
        invalid = False
        try:
            return output_type.model_validate(value)
        except Exception:  # noqa: BLE001 - schema validators are an untrusted boundary.
            invalid = True
        if invalid:
            raise BackendInvalidResponseError from None
    raise BackendInvalidResponseError from None


class CallableBackend:
    """Deterministic adapter for local classifiers, examples, and tests.

    The backend never retains instructions, input text, outputs, or exceptions.
    ``call_count`` is intentionally the only execution state it exposes.
    """

    def __init__(
        self,
        handler: StructuredHandler,
        *,
        async_handler: AsyncStructuredHandler | None = None,
        provider: str = "callable",
        model: str = "deterministic",
    ) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        if async_handler is not None and not callable(async_handler):
            raise TypeError("async_handler must be callable")
        self._handler = handler
        self._async_handler = async_handler
        self._provider = _safe_identifier(provider, field="provider", max_length=80)
        self._model = _safe_identifier(model, field="model")
        self._call_count = 0

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def call_count(self) -> int:
        """Number of sync and async completions, without request retention."""

        return self._call_count

    def complete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> OutputT:
        self._call_count += 1
        failure: BackendError | None = None
        value: object | None = None
        result: OutputT | None = None
        try:
            value = self._handler(
                instructions=instructions,
                input_text=input_text,
                output_type=output_type,
            )
            result = _validated_output(value, output_type)
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        except TimeoutError:
            failure = BackendTimeoutError()
        except ValidationError:
            failure = BackendInvalidResponseError()
        # A caller-supplied handler is a provider boundary; never expose its exception.
        except Exception:  # noqa: BLE001
            failure = BackendProviderError()
        if failure is not None:
            del self, instructions, input_text, output_type, value, result
            _raise_backend_error(failure)
        if result is None:
            failure = BackendInvalidResponseError()
            del self, instructions, input_text, output_type, value, result
            _raise_backend_error(failure)
        return result

    async def acomplete(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type: type[OutputT],
    ) -> OutputT:
        self._call_count += 1
        failure: BackendError | None = None
        value: object | None = None
        result: OutputT | None = None
        try:
            if self._async_handler is None:
                value = self._handler(
                    instructions=instructions,
                    input_text=input_text,
                    output_type=output_type,
                )
            else:
                value = await self._async_handler(
                    instructions=instructions,
                    input_text=input_text,
                    output_type=output_type,
                )
            result = _validated_output(value, output_type)
        except BackendError as caught:
            failure = _fresh_backend_error(caught)
        except TimeoutError:
            failure = BackendTimeoutError()
        except ValidationError:
            failure = BackendInvalidResponseError()
        # A caller-supplied handler is a provider boundary; never expose its exception.
        except Exception:  # noqa: BLE001
            failure = BackendProviderError()
        if failure is not None:
            del self, instructions, input_text, output_type, value, result
            _raise_backend_error(failure)
        if result is None:
            failure = BackendInvalidResponseError()
            del self, instructions, input_text, output_type, value, result
            _raise_backend_error(failure)
        return result


def _safe_identifier(value: str, *, field: str, max_length: int = 160) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise ValueError(f"{field} must contain between 1 and {max_length} characters")
    if any(character in normalized for character in ("\n", "\r", "\x00")):
        raise ValueError(f"{field} must be a single-line identifier")
    return normalized


__all__ = [
    "AsyncStructuredHandler",
    "BackendConfigurationError",
    "BackendError",
    "BackendInvalidResponseError",
    "BackendProviderError",
    "BackendRefusalError",
    "BackendTimeoutError",
    "CallableBackend",
    "OutputT",
    "StructuredBackend",
    "StructuredHandler",
]
