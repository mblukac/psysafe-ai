from __future__ import annotations

import math
from collections.abc import Iterator
from types import TracebackType

import pytest

from psysafe.integrations._serialization import (
    MAX_CONTAINER_ITEMS,
    MAX_INTEGER_BITS,
    MAX_KEY_LENGTH,
    MAX_NESTING_DEPTH,
    MAX_OUTPUT_LENGTH,
    MAX_STRING_LENGTH,
    MAX_TOTAL_CONTAINER_ITEMS,
    MAX_TOTAL_NODES,
    IntegrationInputError,
    canonical_json,
)


def test_canonical_json_is_compact_sorted_and_preserves_unicode() -> None:
    value = {
        "z": (True, None, 3, 1.25),
        "a": {"snowman": "☃", "accent": "é"},
    }

    assert canonical_json(value) == '{"a":{"accent":"é","snowman":"☃"},"z":[true,null,3,1.25]}'
    assert canonical_json(value) == canonical_json(value)


@pytest.mark.parametrize("value", [None, False, True, 0, -(2**128), 0.0, -0.0, 1.5, "", [], (), {}])
def test_exact_builtin_values_are_accepted(value: object) -> None:
    assert isinstance(canonical_json(value), str)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(value: float) -> None:
    with pytest.raises(IntegrationInputError, match=r"^integration input is invalid$"):
        canonical_json(value)


def test_dict_keys_must_be_exact_strings() -> None:
    with pytest.raises(IntegrationInputError):
        canonical_json({1: "not a string key"})


class _HostileObject:
    calls = 0

    @property
    def payload(self) -> str:
        type(self).calls += 1
        raise AssertionError("property access must not run")

    def __iter__(self) -> Iterator[object]:
        type(self).calls += 1
        raise AssertionError("iteration must not run")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("repr must not run")

    def __str__(self) -> str:
        type(self).calls += 1
        raise AssertionError("str must not run")


class _HostileList(list[object]):
    calls = 0

    def __iter__(self) -> Iterator[object]:
        type(self).calls += 1
        raise AssertionError("iteration must not run")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("repr must not run")


class _HostileString(str):
    calls = 0

    def __str__(self) -> str:
        type(self).calls += 1
        raise AssertionError("str must not run")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("repr must not run")


class _HostileTuple(tuple[object, ...]):
    calls = 0

    def __iter__(self) -> Iterator[object]:
        type(self).calls += 1
        raise AssertionError("iteration must not run")


class _HostileDict(dict[str, object]):
    calls = 0

    def __iter__(self) -> Iterator[str]:
        type(self).calls += 1
        raise AssertionError("iteration must not run")


class _HostileInt(int):
    calls = 0

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("repr must not run")


class _HostileFloat(float):
    calls = 0

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("repr must not run")


_HOSTILE_TYPES = (_HostileObject, _HostileList, _HostileString, _HostileTuple, _HostileDict, _HostileInt, _HostileFloat)


@pytest.mark.parametrize(
    "value",
    [
        _HostileObject(),
        _HostileList(["private"]),
        _HostileString("private"),
        _HostileTuple(("private",)),
        _HostileDict({"payload": "private"}),
        _HostileInt(7),
        _HostileFloat(1.5),
    ],
    ids=lambda value: type(value).__name__,
)
def test_hostile_objects_and_builtin_subclasses_are_rejected_without_dispatch(value: object) -> None:
    for hostile_type in _HOSTILE_TYPES:
        hostile_type.calls = 0

    with pytest.raises(IntegrationInputError):
        canonical_json(value)

    assert all(hostile_type.calls == 0 for hostile_type in _HOSTILE_TYPES)


def test_hostile_nested_value_is_not_dispatched() -> None:
    hostile = _HostileObject()
    _HostileObject.calls = 0

    with pytest.raises(IntegrationInputError):
        canonical_json({"safe-key": [hostile]})

    assert _HostileObject.calls == 0


def test_string_subclass_key_is_rejected_without_dispatch() -> None:
    key = _HostileString("private")
    value = {key: None}
    _HostileString.calls = 0

    with pytest.raises(IntegrationInputError):
        canonical_json(value)

    assert _HostileString.calls == 0


def test_direct_and_indirect_cycles_are_rejected() -> None:
    direct: list[object] = []
    direct.append(direct)
    with pytest.raises(IntegrationInputError):
        canonical_json(direct)

    left: list[object] = []
    right: dict[str, object] = {"left": left}
    left.append(right)
    with pytest.raises(IntegrationInputError):
        canonical_json(right)


def test_shared_acyclic_container_is_accepted() -> None:
    shared: list[object] = ["value"]
    assert canonical_json([shared, shared]) == '[["value"],["value"]]'


def _nested_list(edges: int) -> object:
    value: object = None
    for _ in range(edges):
        value = [value]
    return value


def test_nesting_depth_boundary_is_exact() -> None:
    assert canonical_json(_nested_list(MAX_NESTING_DEPTH))
    with pytest.raises(IntegrationInputError):
        canonical_json(_nested_list(MAX_NESTING_DEPTH + 1))


def test_per_container_item_boundary_is_exact() -> None:
    assert canonical_json([None] * MAX_CONTAINER_ITEMS)
    with pytest.raises(IntegrationInputError):
        canonical_json([None] * (MAX_CONTAINER_ITEMS + 1))


def _aggregate_item_tree(total_items: int) -> list[object]:
    groups: list[object] = []
    remaining = total_items
    while remaining > 0:
        size = min(MAX_CONTAINER_ITEMS, remaining - 1) if remaining > 1 else 0
        groups.append([None] * size)
        remaining -= size + 1
    return groups


def test_total_container_item_boundary_is_exact() -> None:
    assert canonical_json(_aggregate_item_tree(MAX_TOTAL_CONTAINER_ITEMS))
    with pytest.raises(IntegrationInputError):
        canonical_json(_aggregate_item_tree(MAX_TOTAL_CONTAINER_ITEMS + 1))


def _dict_chain(total_entries: int) -> dict[str, object]:
    tail: dict[str, object] | None = None
    remaining = total_entries
    while remaining:
        item_count = min(MAX_CONTAINER_ITEMS, remaining)
        current: dict[str, object] = {}
        first_leaf = 0
        if tail is not None:
            current["next"] = tail
            first_leaf = 1
        for index in range(first_leaf, item_count):
            current[f"k{index}"] = None
        tail = current
        remaining -= item_count
    assert tail is not None
    return tail


def test_total_node_boundary_is_exact() -> None:
    assert MAX_TOTAL_NODES == 2 * MAX_TOTAL_CONTAINER_ITEMS
    assert canonical_json([_dict_chain(MAX_TOTAL_CONTAINER_ITEMS - 1)])
    with pytest.raises(IntegrationInputError):
        canonical_json(_dict_chain(MAX_TOTAL_CONTAINER_ITEMS))


def test_string_and_key_length_boundaries_are_exact() -> None:
    assert canonical_json("x" * MAX_STRING_LENGTH)
    with pytest.raises(IntegrationInputError):
        canonical_json("x" * (MAX_STRING_LENGTH + 1))

    assert canonical_json({"k" * MAX_KEY_LENGTH: None})
    with pytest.raises(IntegrationInputError):
        canonical_json({"k" * (MAX_KEY_LENGTH + 1): None})


def test_integer_magnitude_boundary_is_exact() -> None:
    assert canonical_json(1 << (MAX_INTEGER_BITS - 1))
    with pytest.raises(IntegrationInputError):
        canonical_json(1 << MAX_INTEGER_BITS)


def _ascii_output_at_length(length: int) -> list[str]:
    overhead = 2 + 3 + (4 * 2)
    payload_length = length - overhead
    full_strings = ["x" * MAX_STRING_LENGTH] * 3
    full_strings.append("x" * (payload_length - 3 * MAX_STRING_LENGTH))
    return full_strings


def test_rendered_output_length_boundary_is_exact() -> None:
    at_limit = _ascii_output_at_length(MAX_OUTPUT_LENGTH)
    assert len(canonical_json(at_limit)) == MAX_OUTPUT_LENGTH

    over_limit = list(at_limit)
    over_limit[-1] += "x"
    with pytest.raises(IntegrationInputError):
        canonical_json(over_limit)


def test_utf8_byte_length_is_bounded_even_when_character_length_is_smaller() -> None:
    with pytest.raises(IntegrationInputError):
        canonical_json(["💣" * MAX_STRING_LENGTH] * 2)


@pytest.mark.parametrize("value", ["\ud800", {"\udfff": None}, {"payload": "\ud800"}])
def test_unpaired_surrogates_are_sanitized(value: object) -> None:
    with pytest.raises(IntegrationInputError, match=r"^integration input is invalid$") as captured:
        canonical_json(value)

    assert captured.value.__context__ is not None
    assert captured.value.__context__.__traceback__ is None
    assert captured.value.__context__.__context__ is None


def _integration_traceback_frames(traceback: TracebackType | None) -> list[TracebackType]:
    frames: list[TracebackType] = []
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == "psysafe.integrations._serialization":
            frames.append(traceback)
        traceback = traceback.tb_next
    return frames


def test_failure_traceback_does_not_retain_raw_input() -> None:
    private_marker = "private-customer-prompt-0c6ce5"
    raw: dict[str, object] = {"payload": private_marker}
    raw["cycle"] = raw

    with pytest.raises(IntegrationInputError) as captured:
        canonical_json(raw)

    error = captured.value
    frames = _integration_traceback_frames(error.__traceback__)
    assert frames
    for traceback in frames:
        assert all(local is not raw for local in traceback.tb_frame.f_locals.values())
        assert private_marker not in traceback.tb_frame.f_locals.values()
    assert str(error) == "integration input is invalid"
    assert private_marker not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is not None
    assert error.__context__.__traceback__ is None


def test_all_invalid_inputs_use_the_same_fresh_exception() -> None:
    errors: list[IntegrationInputError] = []
    for value in (object(), float("nan"), {"too-long": "x" * (MAX_STRING_LENGTH + 1)}):
        try:
            canonical_json(value)
        except IntegrationInputError as error:
            errors.append(error)

    assert len(errors) == 3
    assert len({id(error) for error in errors}) == 3
    assert {error.args for error in errors} == {("integration input is invalid",)}
