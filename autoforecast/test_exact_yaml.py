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
