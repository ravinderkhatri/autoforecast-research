"""Tests for unambiguous YAML loading at immutable specification boundaries."""

from __future__ import annotations

import pytest

from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml


def test_exact_yaml_accepts_unique_nested_mappings() -> None:
    assert loads_exact_yaml("outer:\n  value: 1\nitems: [{value: 2}]\n") == {
        "outer": {"value": 1},
        "items": [{"value": 2}],
    }


@pytest.mark.parametrize(
    "payload",
    (
        "value: 1\nvalue: 2\n",
        "outer:\n  value: 1\n  value: 2\n",
        "items: [{value: 1, value: 2}]\n",
    ),
)
def test_exact_yaml_rejects_duplicate_keys_at_every_depth(payload: str) -> None:
    with pytest.raises(ExactYAMLError, match="duplicate YAML mapping key 'value'"):
        loads_exact_yaml(payload)


@pytest.mark.parametrize("payload", ("value: &shared [1]\ncopy: *shared\n", "a: &a [*a]\n"))
def test_exact_yaml_rejects_anchors_and_aliases(payload: str) -> None:
    with pytest.raises(ExactYAMLError, match="anchors and aliases"):
        loads_exact_yaml(payload)


@pytest.mark.parametrize("format,payload", [
    ("json", '{"outer":{"x":1,"x":2}}'),
    ("json", '{"x":NaN}'), ("json", '{"x":Infinity}'), ("json", '{"x":1e999}'),
    ("yaml", "x: .inf"), ("yaml", "x: .nan"),
    ("yaml", "true: x"), ("yaml", "1: x"), ("yaml", "x: 2024-01-01"),
    ("yaml", "x: !!python/object/apply:os.system ['false']"),
    ("yaml", "x: &value 1\ny: *value"), ("yaml", "x: {y: 1, y: 2}"),
])
def test_registration_codecs_reject_ambiguous_or_non_json_values(format, payload):
    from autoforecast._exact_yaml import loads_record

    with pytest.raises(ValueError):
        loads_record(payload, format=format)


def test_record_encoding_preserves_scalar_types_and_logical_identity(tmp_path):
    from autoforecast._exact_yaml import loads_record, read_record, record_bytes
    from autoforecast.benchmark import digest

    value = {"unicode": "温度", "integer": 1, "float": 1.0, "boolean": True,
             "null": None, "date": "2024-01-01", "numeric_key": {"1": "one"}}
    for suffix, format in (("json", "json"), ("yaml", "yaml"), ("yml", "yaml")):
        payload = record_bytes(value, format=format)
        path = tmp_path / ("config." + suffix)
        path.write_bytes(payload)
        decoded = read_record(path)
        assert decoded == value and digest(decoded) == digest(value)
        assert type(decoded["integer"]) is int
        assert type(decoded["float"]) is float
        assert type(decoded["boolean"]) is bool
        assert type(decoded["date"]) is str
        assert loads_record(payload, format=format) == decoded
    assert digest({"x": 1}) != digest({"x": 1.0}) != digest({"x": True})
    shared = {"version": 2}
    repeated = {"primary": shared, "audit": shared}
    assert loads_record(record_bytes(repeated, format="yaml"), format="yaml") == repeated
