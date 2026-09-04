"""One JSON Schema shared by configuration tools and registration field checks."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator, validators


@lru_cache(maxsize=1)
def _schema():
    return json.loads(files("autoforecast").joinpath("study_config.schema.json").read_text())


def study_config_schema() -> dict:
    """Return an independent copy of the packaged registration schema."""
    return json.loads(json.dumps(_schema()))


@lru_cache(maxsize=8)
def _validator(section):
    schema = _schema()
    if section is not None:
        schema = {"$ref": "#/$defs/" + section, "$defs": schema["$defs"]}
    checker = Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda _checker, item: type(item) is int,
    )
    return validators.extend(Draft202012Validator, type_checker=checker)(schema)


def validate_shape(value, section=None):
    """Check field names/types without including private setting values in errors."""
    validator = _validator(section)
    error = next(validator.iter_errors(value), None)
    if error is None:
        return
    # Choose diagnostics from the form identified by the caller's field names.
    # Never include its values, which may contain connection details.
    while error.validator == "oneOf" and isinstance(error.instance, dict):
        choices = []
        for index, branch in enumerate(error.validator_value):
            properties = branch.get("properties", {})
            if any("const" in rule and key in error.instance
                   and error.instance[key] != rule["const"]
                   for key, rule in properties.items()):
                continue
            choices.append((len(set(properties) & set(error.instance)), index))
        best = max((score for score, _ in choices), default=0)
        matches = [index for score, index in choices if score == best]
        if not best or len(matches) != 1:
            break
        error = next(child for child in error.context if child.schema_path[0] == matches[0])
    field = ".".join(str(part) for part in error.absolute_path) or section or "study config"
    if error.validator == "type":
        names = set(error.absolute_path)
        if names & {"training_start", "final_train_cutoff", "validation_origins",
                    "origin", "start"}:
            raise ValueError(f"{field}: use canonical ISO YYYY-MM-DD dates, or "
                             "YYYY-MM-DDTHH:MM:SS timestamps for hourly studies")
        if names & {"validation_horizon", "final_test_horizon", "horizon"}:
            raise ValueError(f"{field}: boundaries require a positive integer horizon")
    if error.validator == "required":
        missing = sorted(set(error.validator_value) - set(error.instance))
        if missing == ["references"]:
            raise ValueError("references: WRMSSE requires configured scoring references; "
                             "supply weight/scale settings or a reference path")
        raise ValueError(f"{field}.{missing[0]}: required; add this field using the schema example")
    if error.validator == "additionalProperties":
        unknown = sorted(set(error.instance) - set(error.schema.get("properties", {})))
        raise ValueError(f"{field}: unknown fields {unknown}; use the published field names")
    if error.validator == "not":
        raise ValueError(f"{field}: conflicting settings; check the allowed combinations "
                         "in study_config.schema.json")
    if error.validator == "oneOf":
        groups = [", ".join(branch.get("required", ())) for branch in error.validator_value]
        raise ValueError(f"{field}: choose exactly one supported form: {'; or '.join(groups)}")
    raise ValueError(f"{field}: invalid {error.validator}; check its type and allowed values "
                     "in study_config.schema.json")
