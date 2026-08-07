"""Strict serialization boundary shared by optional agent integrations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import NoReturn, cast

MAX_NESTING_DEPTH = 32
"""Maximum number of parent/child edges below the root value."""

MAX_TOTAL_NODES = 8_192
"""Maximum number of values and object keys in one input tree."""

MAX_CONTAINER_ITEMS = 512
"""Maximum number of items in any one list, tuple, or object."""

MAX_TOTAL_CONTAINER_ITEMS = 4_096
"""Maximum number of items across all containers in one input tree."""

MAX_STRING_LENGTH = 16_384
"""Maximum number of Unicode code points in a string value."""

MAX_KEY_LENGTH = 256
"""Maximum number of Unicode code points in an object key."""

MAX_OUTPUT_LENGTH = 65_536
"""Maximum rendered size, in both characters and UTF-8 bytes."""

MAX_INTEGER_BITS = 4_096
"""Maximum integer magnitude, measured using ``int.bit_length``."""

_INPUT_ERROR_MESSAGE = "integration input is invalid"


class IntegrationInputError(ValueError):
    """Report an unsafe integration value without echoing its contents."""

    def __init__(self) -> None:
        super().__init__(_INPUT_ERROR_MESSAGE)


class _SerializationError(Exception):
    """Internal data-free control-flow exception."""


@dataclass(slots=True)
class _TraversalState:
    nodes: int = 0
    container_items: int = 0
    active_container_ids: set[int] = field(default_factory=set)


def _invalid() -> NoReturn:
    raise _SerializationError


def _add_node(state: _TraversalState) -> None:
    state.nodes += 1
    if state.nodes > MAX_TOTAL_NODES:
        _invalid()


def _add_container_items(state: _TraversalState, count: int) -> None:
    if count > MAX_CONTAINER_ITEMS:
        _invalid()
    state.container_items += count
    if state.container_items > MAX_TOTAL_CONTAINER_ITEMS:
        _invalid()


def _contains_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in text)


def _copy_sequence(value: list[object] | tuple[object, ...], depth: int, state: _TraversalState) -> list[object]:
    container_id = id(value)
    if container_id in state.active_container_ids:
        _invalid()

    items: list[object] | tuple[object, ...]
    if type(value) is list:
        items = value.copy()
    else:
        items = cast(tuple[object, ...], value)
    _add_container_items(state, len(items))

    state.active_container_ids.add(container_id)
    try:
        return [_copy_value(item, depth + 1, state) for item in items]
    finally:
        state.active_container_ids.remove(container_id)


def _copy_mapping(value: dict[str, object], depth: int, state: _TraversalState) -> dict[str, object]:
    container_id = id(value)
    if container_id in state.active_container_ids:
        _invalid()

    items = tuple(dict.items(dict.copy(value)))
    _add_container_items(state, len(items))

    copied: dict[str, object] = {}
    state.active_container_ids.add(container_id)
    try:
        for key, item in items:
            if type(key) is not str or len(key) > MAX_KEY_LENGTH or _contains_surrogate(key):
                _invalid()
            _add_node(state)
            copied[key] = _copy_value(item, depth + 1, state)
    finally:
        state.active_container_ids.remove(container_id)
    return copied


def _copy_value(value: object, depth: int, state: _TraversalState) -> object:
    if depth > MAX_NESTING_DEPTH:
        _invalid()
    _add_node(state)

    value_type = type(value)
    if value is None or value_type is bool:
        return value
    if value_type is int:
        integer = cast(int, value)
        if int.bit_length(integer) > MAX_INTEGER_BITS:
            _invalid()
        return integer
    if value_type is float:
        number = cast(float, value)
        if not math.isfinite(number):
            _invalid()
        return number
    if value_type is str:
        text = cast(str, value)
        if len(text) > MAX_STRING_LENGTH or _contains_surrogate(text):
            _invalid()
        return text
    if value_type is list or value_type is tuple:
        return _copy_sequence(cast(list[object] | tuple[object, ...], value), depth, state)
    if value_type is dict:
        return _copy_mapping(cast(dict[str, object], value), depth, state)
    _invalid()


def _attempt_render(value: object) -> str | None:
    try:
        rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        encoded_length = len(rendered.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError):
        value = None
        return None
    if len(rendered) > MAX_OUTPUT_LENGTH or encoded_length > MAX_OUTPUT_LENGTH:
        return None
    return rendered


def _render(value: object) -> str:
    copied = _copy_value(value, 0, _TraversalState())
    rendered = _attempt_render(copied)
    if rendered is None:
        copied = None
        _invalid()
    return rendered


def canonical_json(value: object) -> str:
    """Render a bounded tree of exact built-in JSON values deterministically.

    Tuples are accepted as immutable array inputs. All subclasses and arbitrary
    objects are rejected before any of their methods can be dispatched.
    """

    try:
        return _render(value)
    except _SerializationError as failure:
        failure.__traceback__ = None
        value = None
        raise IntegrationInputError() from None
