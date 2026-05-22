"""Strict YAML decoding for immutable request and campaign specifications."""

from __future__ import annotations

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


__all__ = ["ExactYAMLError", "loads_exact_yaml"]
