"""Internal provider-exception categorization without exception retention."""

from __future__ import annotations

from psysafe.backends.base import (
    BackendError,
    BackendInvalidResponseError,
    BackendProviderError,
    BackendRefusalError,
    BackendTimeoutError,
    _fresh_backend_error,
)


def sanitized_backend_error(error: BaseException) -> BackendError:
    """Map provider exceptions by type name, discarding their sensitive text."""

    if isinstance(error, BackendError):
        return _fresh_backend_error(error)
    if isinstance(error, TimeoutError):
        return BackendTimeoutError()

    type_name = type(error).__name__.lower()
    if "timeout" in type_name:
        return BackendTimeoutError()
    if "refusal" in type_name or "contentfilter" in type_name or "content_filter" in type_name:
        return BackendRefusalError()
    if any(
        marker in type_name
        for marker in (
            "json",
            "lengthfinish",
            "parse",
            "schema",
            "validation",
        )
    ):
        return BackendInvalidResponseError()
    return BackendProviderError()


__all__ = ["sanitized_backend_error"]
