"""Strict YAML decoding for immutable request and campaign specifications."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Hashable


class ExactYAMLError(ValueError):
    """A YAML document is malformed or contains an ambiguous mapping."""


def loads_exact_yaml(payload: str | bytes) -> object:
    """Safely decode YAML while rejecting aliases and duplicate keys at every depth."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ExactYAMLError("strict YAML decoding requires PyYAML") from exc

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    try:
        if any(
            isinstance(token, yaml.tokens.AliasToken | yaml.tokens.AnchorToken)
            for token in yaml.scan(payload)
        ):
            raise ExactYAMLError("YAML anchors and aliases are not permitted in governed input")
    except ExactYAMLError:
        raise
    except yaml.YAMLError as exc:
        raise ExactYAMLError(str(exc)) from exc

    def construct_mapping(loader, node, deep=False):  # noqa: ANN001, ANN202
        if not isinstance(node, yaml.MappingNode):
            raise ExactYAMLError("expected a YAML mapping node")
        loader.flatten_mapping(node)
        result: dict[Hashable, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ExactYAMLError("YAML mapping keys must be hashable")
            if key in result:
                raise ExactYAMLError(f"duplicate YAML mapping key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        return yaml.load(payload, Loader=UniqueKeySafeLoader)
    except ExactYAMLError:
        raise
    except yaml.YAMLError as exc:
        raise ExactYAMLError(str(exc)) from exc


def loads_record(payload: str | bytes, *, format: str) -> object:
    """Decode registration data with identical unambiguous JSON scalar rules."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON mapping key {key!r}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    if format == "json":
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    elif format == "yaml":
        value = loads_exact_yaml(payload)
    else:
        raise ValueError("record format must be json or yaml")

    def check(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("record mapping keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("record numbers must be finite")
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("records require JSON scalar types; quote ISO dates")
    check(value)
    return value


def read_record(path: pathlib.Path) -> object:
    """Read a supported user-authored registration configuration."""
    formats = {".json": "json", ".yaml": "yaml", ".yml": "yaml"}
    if path.suffix.lower() not in formats:
        raise ValueError("configuration extension must be .json, .yaml or .yml")
    return loads_record(path.read_bytes(), format=formats[path.suffix.lower()])


def record_bytes(value: object, *, format: str) -> bytes:
    """Encode one validated record; logical digests continue to use canonical JSON."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    loads_record(canonical, format="json")
    if format == "json":
        return (canonical + "\n").encode()
    if format != "yaml":
        raise ValueError("record format must be json or yaml")
    import yaml

    # Expand shared Python containers so the writer never emits YAML aliases.
    return yaml.safe_dump(json.loads(canonical), sort_keys=True, allow_unicode=True).encode()


__all__ = ["ExactYAMLError", "loads_exact_yaml", "loads_record", "read_record", "record_bytes"]
