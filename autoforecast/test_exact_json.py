"""Tests for unambiguous JSON loading at evidence boundaries."""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast._exact_json import ExactJSONError, load_exact_json, loads_exact_json


def test_loads_exact_json_accepts_nested_unique_objects() -> None:
    assert loads_exact_json('{"outer":{"value":1},"items":[{"value":2}]}') == {
        "outer": {"value": 1},
        "items": [{"value": 2}],
    }


def test_loads_exact_json_rejects_duplicate_keys_at_every_depth() -> None:
    cases = [
        ('{"value":1,"value":2}', "value"),
        ('{"outer":{"value":1,"value":2}}', "value"),
        ('[{"value":1,"value":2}]', "value"),
    ]
    for payload, key in cases:
        try:
            loads_exact_json(payload)
        except ExactJSONError as error:
            assert str(error) == f"duplicate JSON object key {key!r}"
            assert error.kind == "duplicate"
        else:
            raise AssertionError(f"duplicate key {key!r} was accepted in {payload}")


def test_loads_exact_json_wraps_decode_and_unicode_failures() -> None:
    try:
        loads_exact_json('{"value":')
    except ExactJSONError as malformed:
        assert malformed.kind == "decode"
        assert malformed.cause_type_name == "JSONDecodeError"
    else:
        raise AssertionError("malformed JSON was accepted")

    try:
        loads_exact_json(b'"\xff"')
    except ExactJSONError as invalid_unicode:
        assert invalid_unicode.kind == "decode"
        assert invalid_unicode.cause_type_name == "UnicodeDecodeError"
    else:
        raise AssertionError("invalid Unicode was accepted as JSON")


def test_loads_exact_json_rejects_non_standard_and_overflowed_non_finite_numbers() -> None:
    for value in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
        try:
            loads_exact_json(f'{{"value":{value}}}')
        except ExactJSONError as error:
            assert error.kind == "decode"
            assert str(error) == f"non-finite JSON number {value!r}"
        else:
            raise AssertionError(f"non-standard JSON number {value} was accepted")


def test_load_exact_json_wraps_read_failures() -> None:
    with tempfile.TemporaryDirectory() as raw:
        missing = pathlib.Path(raw) / "missing.json"
        try:
            load_exact_json(missing)
        except ExactJSONError as error:
            assert error.kind == "read"
            assert error.cause_type_name == "FileNotFoundError"
        else:
            raise AssertionError("a missing JSON file was accepted")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
