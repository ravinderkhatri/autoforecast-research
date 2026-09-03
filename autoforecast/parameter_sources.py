"""Governed resolution of defaults, manual values, and HPO selections.

Human-authored YAML chooses exactly one source.  This module turns that choice into one complete,
finite JSON record which the generic runner can verify without learning campaign semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
from typing import cast

from mlops.provenance import (
    ProvenanceError,
    sanitize_repository_origin,
    validate_data_reference,
)
from mlops.run_bundle import load_completed_run_bundle
from spec.resolved import ResolvedBuild, ResolveError

from autoforecast._exact_json import ExactJSONError, load_exact_json, loads_exact_json
from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast._numeric import is_finite_real
from autoforecast.adapter_contracts import descriptor
from autoforecast.experiments import (
    ExperimentError,
    load_completed_attempt,
    load_study_selection,
    require_heldout_evaluation_approval,
)
from autoforecast.frequency import (
    FrequencyError,
    frequency_is_supported,
    parse_frequency,
    parse_origin,
    period_points,
)
from autoforecast.hpo import HPOError, HPOStudySpec, load_hpo_study
from autoforecast.model_catalog import Candidate
from autoforecast.model_execution import (
    ExecutionPlanError,
    resolve_revision_rolling_config,
    revision_required_history,
    validate_revision_parameter_groups,
)
from autoforecast.model_names import REVISION_VARIANTS
from autoforecast.precision_policy import effective_precision
from autoforecast.project import installed_library_identity, installed_mode
from autoforecast.tuning import (
    PARAMETER_EXECUTION_COMPATIBILITY,
    TuningError,
    loads_tuning_spec,
    materialize_manual_parameters,
    materialize_runtime_parameters,
    materialize_trial_parameters,
)
from autoforecast.tuning_support import tuning_support
from provenance.lineage import LineageError, directory_digest

__all__ = [
    "PARAMETER_EXECUTION_COMPATIBILITY",
    "PARAMETER_RESOLUTION_SCHEMA",
    "ParameterSourceError",
    "default_parameter_resolution",
    "load_parameter_resolution",
    "manual_parameter_resolution",
    "parameter_resolution_from_selection",
    "parameter_resolution_sha256",
    "validate_parameter_source_capsule",
    "validate_parameter_resolution",
]

PARAMETER_RESOLUTION_SCHEMA = "autoforecast.parameter-resolution.v1"
_KINDS = frozenset({"defaults", "manual", "tune", "reuse_selection"})
_SHA_CHARS = frozenset("0123456789abcdef")
_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,127}$")


class ParameterSourceError(ValueError):
    """A parameter source is ambiguous, incompatible, mutable, or malformed."""


def _canonical(record: dict[str, object]) -> bytes:
    try:
        return (json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise ParameterSourceError(f"parameter resolution is not finite JSON: {error}") from error


def parameter_resolution_sha256(record: dict[str, object]) -> str:
    """Return the stable identity used by campaign and run evidence."""
    return hashlib.sha256(_canonical(record)).hexdigest()


def validate_parameter_source_capsule(
    bundle_path: str | pathlib.Path,
    *,
    expected_parameter_resolution_sha256: str,
) -> dict[str, object]:
    """Verify the portable HPO evidence retained inside a completed run bundle."""
    root = _without_links(bundle_path, "run bundle").resolve()
    try:
        load_completed_run_bundle(root)
    except ProvenanceError as error:
        raise ParameterSourceError(
            f"parameter source capsule bundle is invalid: {error}"
        ) from error
    capsule = _exact_record(
        root / "provenance" / "parameter-source" / "capsule.json",
        "parameter source capsule",
    )
    required = {
        "schema",
        "parameter_resolution_sha256",
        "study_id",
        "study_config",
        "study_config_sha256",
        "selection",
        "selection_sha256",
        "selected_trial_id",
        "selected_attempt_id",
        "selected_attempt_manifest_sha256",
        "selected_attempt",
        "artifacts",
    }
    artifacts = capsule.get("artifacts")
    if (
        set(capsule) != required
        or capsule.get("schema") != "autoforecast.parameter-source-capsule.v2"
        or not _is_sha256(expected_parameter_resolution_sha256)
        or capsule.get("parameter_resolution_sha256") != expected_parameter_resolution_sha256
        or type(artifacts) is not list
        or not artifacts
        or type(capsule.get("study_id")) is not str
        or type(capsule.get("selected_trial_id")) is not str
        or type(capsule.get("selected_attempt_id")) is not str
        or any(
            not _is_sha256(capsule.get(field))
            for field in (
                "study_config_sha256",
                "selection_sha256",
                "selected_attempt_manifest_sha256",
            )
        )
    ):
        raise ParameterSourceError("parameter source capsule identity is invalid")
    retained: set[str] = set()
    for item in artifacts:
        if (
            type(item) is not dict
            or set(item) != {"path", "sha256", "bytes"}
            or not _is_relative_source_path(item.get("path"))
            or not cast(str, item["path"]).startswith("provenance/parameter-source/")
            or cast(str, item["path"]) in retained
            or not _is_sha256(item.get("sha256"))
            or type(item.get("bytes")) is not int
            or cast(int, item["bytes"]) < 0
        ):
            raise ParameterSourceError("parameter source capsule artifact entry is invalid")
        relative = cast(str, item["path"])
        retained.add(relative)
        target = _without_links(root / relative, "parameter source capsule artifact")
        if (
            not target.is_file()
            or target.stat().st_size != item["bytes"]
            or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ParameterSourceError("parameter source capsule artifact bytes differ")
    for field, required_suffix in (
        ("study_config", None),
        ("selection", "selection.json"),
        ("selected_attempt", None),
    ):
        relative = capsule.get(field)
        if (
            not _is_relative_source_path(relative)
            or required_suffix is not None
            and pathlib.PurePosixPath(cast(str, relative)).name != required_suffix
        ):
            raise ParameterSourceError(f"parameter source capsule {field} path is invalid")
        target = _without_links(root / cast(str, relative), f"parameter source capsule {field}")
        if field == "selected_attempt":
            if not target.is_dir() or not any(
                path == cast(str, relative) or path.startswith(f"{relative}/") for path in retained
            ):
                raise ParameterSourceError("parameter source capsule selected attempt is missing")
        elif cast(str, relative) not in retained or not target.is_file():
            raise ParameterSourceError(f"parameter source capsule {field} is missing")
    capsule_root = root / "provenance" / "parameter-source"
    actual_retained = {
        path.relative_to(root).as_posix()
        for path in capsule_root.rglob("*")
        if path.is_file() and path.name != "capsule.json"
    }
    if actual_retained != retained:
        raise ParameterSourceError("parameter source capsule has undeclared or missing evidence")

    resolution = _exact_record(
        root / "metadata" / "parameter-resolution.json", "retained parameter resolution"
    )
    if parameter_resolution_sha256(resolution) != expected_parameter_resolution_sha256:
        raise ParameterSourceError("retained parameter resolution identity differs from capsule")
    source = resolution.get("source")
    if (
        type(source) is not dict
        or type(source.get("kind")) is not str
        or source.get("kind") not in {"tune", "reuse_selection"}
    ):
        raise ParameterSourceError("retained parameter resolution is not an HPO source")
    selection_path = root / cast(str, capsule["selection"])
    config_path = root / cast(str, capsule["study_config"])
    attempt_path = root / cast(str, capsule["selected_attempt"])
    try:
        selection = _exact_record(selection_path, "retained HPO selection")
        metadata, completion = load_completed_attempt(attempt_path)
    except (ExperimentError, ExactJSONError) as error:
        raise ParameterSourceError(f"retained HPO evidence is invalid: {error}") from error
    candidates = selection.get("candidates")
    selected = selection.get("selected")
    winner = selected[0] if isinstance(selected, list) and selected else None
    if (
        hashlib.sha256(selection_path.read_bytes()).hexdigest() != capsule["selection_sha256"]
        or hashlib.sha256(config_path.read_bytes()).hexdigest() != capsule["study_config_sha256"]
        or not _has_exact_positive_ranks(candidates)
        or not _has_exact_positive_ranks(selected)
        or type(winner) is not dict
        or not _is_rank_one(winner.get("rank"))
        or winner.get("trial_id") != capsule["selected_trial_id"]
        or winner.get("attempt_id") != capsule["selected_attempt_id"]
        or winner.get("artifact_manifest_sha256") != capsule["selected_attempt_manifest_sha256"]
        or completion.get("artifact_manifest_sha256") != capsule["selected_attempt_manifest_sha256"]
        or metadata.get("study_id") != capsule["study_id"]
        or metadata.get("trial_id") != capsule["selected_trial_id"]
        or metadata.get("attempt_id") != capsule["selected_attempt_id"]
    ):
        raise ParameterSourceError("retained HPO selection and attempt disagree")
    cross_links = {
        "study_id": "study_id",
        "study_config_sha256": "study_config_sha256",
        "selection_sha256": "selection_sha256",
        "selected_trial_id": "selected_trial_id",
        "selected_attempt_id": "selected_attempt_id",
        "selected_attempt_manifest_sha256": "selected_attempt_manifest_sha256",
    }
    if any(
        source.get(source_key) != capsule.get(capsule_key)
        for source_key, capsule_key in cross_links.items()
    ):
        raise ParameterSourceError("parameter source capsule differs from its resolution")
    _validate_retained_attempt_data_references(attempt_path)
    return capsule


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _is_commit(value: object) -> bool:
    return type(value) is str and len(value) in {40, 64} and set(value) <= _SHA_CHARS


def _is_rank_one(value: object) -> bool:
    return type(value) is int and value == 1


def _has_exact_positive_ranks(value: object) -> bool:
    return (
        type(value) is list
        and bool(value)
        and all(
            type(candidate) is dict and type(rank := candidate.get("rank")) is int and rank > 0
            for candidate in value
        )
    )


def _is_absolute_artifact_path(value: object, *, name: str | None = None) -> bool:
    if type(value) is not str:
        return False
    path = pathlib.PurePath(value)
    return path.is_absolute() and ".." not in path.parts and (name is None or path.name == name)


def _is_relative_source_path(value: object) -> bool:
    if type(value) is not str:
        return False
    path = pathlib.PurePosixPath(value)
    return value not in {"", "."} and not path.is_absolute() and ".." not in path.parts


def _without_links(path: str | pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))
    cursor = lexical
    while True:
        if cursor.is_symlink() and (cursor.as_posix(), cursor.resolve().as_posix()) not in {
            ("/var", "/private/var"),
            ("/tmp", "/private/tmp"),
        }:
            raise ParameterSourceError(f"{label} may not traverse a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return lexical


def _flatten(grouped: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        f"{namespace}.{name}": value
        for namespace, values in grouped.items()
        for name, value in values.items()
    }


def _same_exact(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left, allow_nan=False, sort_keys=True, separators=(",", ":")
        ) == json.dumps(right, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False


def _require_safe_data_reference(reference: object, label: str) -> str:
    """Reject secret-bearing references before retaining or replaying HPO evidence."""
    try:
        return validate_data_reference(reference)
    except ProvenanceError as error:
        raise ParameterSourceError(f"{label} is unsafe: {error}") from error


def _validate_retained_attempt_data_references(attempt: pathlib.Path) -> None:
    """Recheck data references copied into a portable HPO evidence capsule."""
    governed = _exact_record(
        attempt / "evaluation" / "governed-run.json", "retained governed-run record"
    )
    run_id = governed.get("run_id")
    bundle_relative = governed.get("source_bundle")
    if type(run_id) is not str or not _is_relative_source_path(bundle_relative):
        raise ParameterSourceError("retained governed-run identity is invalid")
    parts = pathlib.PurePosixPath(cast(str, bundle_relative)).parts
    if len(parts) != 5 or parts[:2] != ("source-run", run_id):
        raise ParameterSourceError("retained governed-run source bundle is invalid")
    source_bundle = _without_links(attempt.joinpath(*parts), "retained source bundle").resolve()
    if not source_bundle.is_relative_to(attempt.resolve()):
        raise ParameterSourceError("retained source bundle escapes its HPO attempt")
    data = _exact_record(
        source_bundle / "provenance" / "data.json", "retained source data identity"
    )
    _require_safe_data_reference(data.get("reference"), "retained source data reference")
    score = _exact_record(
        attempt / "source-run" / run_id / "evaluation" / "post-training-evaluation.json",
        "retained source score input",
    )
    score_input = score.get("score_input")
    if type(score_input) is not dict:
        raise ParameterSourceError("retained source score input is invalid")
    _require_safe_data_reference(
        score_input.get("reference"), "retained source score data reference"
    )


def _execution_plan_matches(
    plan: dict[str, object], candidate: Candidate, expected: dict[str, object]
) -> bool:
    effective = plan.get("effective_parameters")
    adapter_kwargs = plan.get("adapter_kwargs")
    readiness = plan.get("readiness_settings")
    if (
        type(effective) is not dict
        or type(adapter_kwargs) is not dict
        or type(readiness) is not dict
    ):
        return False
    controls = {"data.horizon", "training.epochs", "training.seed"}
    if candidate.adapter == "mft":
        controls.add("resolved.build_hash")
    if candidate.adapter in {
        "sft",
        "hmfnet",
        "pact",
    }:
        controls.update({"implementation.config_sha256", "implementation.variant"})
    if set(effective) != set(expected) | controls or any(
        not _same_exact(effective.get(name), value) for name, value in expected.items()
    ):
        return False
    if (
        type(effective.get("data.horizon")) is not int
        or cast(int, effective["data.horizon"]) < 1
        or type(effective.get("training.epochs")) is not int
        or cast(int, effective["training.epochs"]) < 1
        or type(effective.get("training.seed")) is not int
        or not 0 <= cast(int, effective["training.seed"]) <= 2**32 - 1
    ):
        return False
    if candidate.adapter in {"timesfm_hnb", "timesfm_v8"}:
        expected_kwargs = {
            **{name: expected[f"model.{name}"] for name in (
                "context_days", "group_size", "max_variates", "head_width")},
            "learning_rate": expected["training.learning_rate"],
            "mean_loss_weight": expected["training.mean_loss_weight"],
            "train_windows": expected["training.train_windows"],
            "train_stride": expected["training.train_stride"],
            "max_epochs": effective["training.epochs"], "seed": effective["training.seed"],
        }
        return readiness == {} and _same_exact(adapter_kwargs, expected_kwargs)
    if candidate.adapter == "tft":
        expected_kwargs = {
            "max_epochs": effective["training.epochs"],
            "hidden_size": expected["model.hidden_size"],
            "attention_head_size": expected["model.attention_head_size"],
            "dropout": expected["model.dropout"],
            "learning_rate": expected["training.learning_rate"],
            "seed": effective["training.seed"],
        }
        return readiness == {} and _same_exact(adapter_kwargs, expected_kwargs)
    if candidate.adapter == "shallow_global":
        expected_kwargs = {
            "num_leaves": expected["model.num_leaves"],
            "min_data_in_leaf": expected["model.min_data_in_leaf"],
            "num_boost_round": expected["model.num_boost_round"],
            "learning_rate": expected["training.learning_rate"],
            "seed": effective["training.seed"],
        }
        return readiness == {} and _same_exact(adapter_kwargs, expected_kwargs)
    if candidate.adapter == "mft":
        expected_keys = {
            "encoder_days",
            "trunk_stages",
            "decoder_layers",
            "history_attention_layers",
            "commercial_attention_layers",
            "event_kernel_layers",
            "epochs",
            "seed",
            "lr",
            "resolved_build",
        }
        resolved_build = adapter_kwargs.get("resolved_build")
        try:
            resolved_build_hash = (
                ResolvedBuild.from_record(resolved_build).build_hash
                if type(resolved_build) is dict
                else None
            )
        except (ResolveError, TypeError, ValueError):
            return False
        return (
            set(adapter_kwargs) == expected_keys
            and _same_exact(readiness, {"encoder_days": expected["model.encoder_days"]})
            and _same_exact(adapter_kwargs.get("encoder_days"), expected["model.encoder_days"])
            and _same_exact(adapter_kwargs.get("trunk_stages"), expected["model.trunk_stages"])
            and _same_exact(adapter_kwargs.get("decoder_layers"), expected["model.decoder_layers"])
            and _same_exact(
                adapter_kwargs.get("history_attention_layers"),
                expected["model.history_attention_layers"],
            )
            and _same_exact(
                adapter_kwargs.get("commercial_attention_layers"),
                expected["model.commercial_attention_layers"],
            )
            and _same_exact(
                adapter_kwargs.get("event_kernel_layers"),
                expected["model.event_kernel_layers"],
            )
            and _same_exact(adapter_kwargs.get("epochs"), effective["training.epochs"])
            and _same_exact(adapter_kwargs.get("seed"), effective["training.seed"])
            and _same_exact(adapter_kwargs.get("lr"), expected["training.learning_rate"])
            and resolved_build_hash == effective.get("resolved.build_hash")
        )
    if candidate.adapter in {
        "sft",
        "hmfnet",
        "pact",
    }:
        rolling = adapter_kwargs.get("rolling_config")
        expected_variant = REVISION_VARIANTS[candidate.adapter]
        if type(rolling) is not dict:
            return False
        try:
            expected_rolling = resolve_revision_rolling_config(
                candidate,
                model_parameters={
                    name.removeprefix("model."): value
                    for name, value in expected.items()
                    if name.startswith("model.")
                },
                training_parameters={
                    name.removeprefix("training."): value
                    for name, value in expected.items()
                    if name.startswith("training.")
                },
                epochs=cast(int, effective["training.epochs"]),
                seed=cast(int, effective["training.seed"]),
            )
        except (ExecutionPlanError, KeyError):
            return False
        return (
            set(adapter_kwargs) == {"rolling_config"}
            and _same_exact(
                readiness,
                {
                    "required_history": revision_required_history(
                        expected_rolling, cast(int, effective["data.horizon"])
                    )
                },
            )
            and _same_exact(rolling, expected_rolling)
            and effective.get("implementation.variant") == expected_variant
            and effective.get("implementation.config_sha256")
            == candidate.implementation_config_sha256
        )
    return False


def _training_controls_are_valid(record: dict[str, object]) -> bool:
    expected = {
        "schema",
        "maximum_epochs",
        "patience",
        "device",
        "batch_size",
        "gradient_microbatch_size",
        "early_stopping_absolute_min_delta",
        "early_stopping_relative_min_delta",
    }
    maximum_epochs = record.get("maximum_epochs")
    patience = record.get("patience")
    batch_size = record.get("batch_size")
    microbatch = record.get("gradient_microbatch_size")
    device = record.get("device")
    absolute_delta = record.get("early_stopping_absolute_min_delta")
    relative_delta = record.get("early_stopping_relative_min_delta")
    schema = record.get("schema")
    precision_valid = (
        schema == "autoforecast.training-controls.v2"
        and set(record) == expected | {"requested_precision", "effective_precision"}
        and record.get("requested_precision") in ("32-true", "16-mixed")
        and record.get("effective_precision") in (
            record.get("requested_precision"), "native"
        )
    )
    return (
        precision_valid
        and all(
            type(value) is int and value > 0 for value in (maximum_epochs, patience, batch_size)
        )
        and (
            microbatch is None
            or type(microbatch) is int
            and 0 < microbatch <= cast(int, batch_size)
        )
        and (device is None or type(device) is str and device in {"cpu", "cuda", "mps"})
        and is_finite_real(absolute_delta)
        and float(absolute_delta) >= 0
        and is_finite_real(relative_delta)
        and float(relative_delta) >= 0
    )


def _training_controls_match_candidate(record: dict[str, object], candidate: Candidate) -> bool:
    if not _training_controls_are_valid(record):
        return False
    return record["effective_precision"] == effective_precision(
        candidate.adapter, cast(str, record["requested_precision"])
    )


def _spec(candidate: Candidate):
    support = tuning_support(candidate.adapter)
    if not support.supported:
        return None
    try:
        spec = loads_tuning_spec(candidate.source_bytes, source_path=candidate.source_path)
    except TuningError as error:
        raise ParameterSourceError(str(error)) from error
    if (
        spec.model_id != candidate.id
        or spec.adapter != candidate.adapter
        or spec.source_sha256 != candidate.source_sha256
    ):
        raise ParameterSourceError("tuning surface does not match its catalog candidate")
    return spec


def _record(
    candidate: Candidate,
    *,
    source: dict[str, object],
    effective_parameters: dict[str, object],
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": PARAMETER_RESOLUTION_SCHEMA,
        "compatibility": PARAMETER_EXECUTION_COMPATIBILITY,
        "model_id": candidate.id,
        "adapter": candidate.adapter,
        "model_profile_sha256": candidate.source_sha256,
        "source": source,
        "effective_parameters": effective_parameters,
    }
    return validate_parameter_resolution(record, candidate)


def default_parameter_resolution(candidate: Candidate) -> dict[str, object]:
    """Freeze profile defaults, or an empty map for an adapter with no parameter surface."""
    spec = _spec(candidate)
    values = {} if spec is None else _flatten(spec.defaults())
    return _record(candidate, source={"kind": "defaults"}, effective_parameters=values)


def manual_parameter_resolution(
    candidate: Candidate, overrides: dict[str, object]
) -> dict[str, object]:
    """Resolve partial user values over profile defaults without applying HPO-only bounds."""
    spec = _spec(candidate)
    if spec is None:
        raise ParameterSourceError(
            f"model {candidate.id!r} has no typed manual-parameter execution surface"
        )
    try:
        grouped = materialize_manual_parameters(spec, overrides)
    except TuningError as error:
        raise ParameterSourceError(str(error)) from error
    return _record(
        candidate,
        source={"kind": "manual", "base": "profile_defaults", "overrides": overrides},
        effective_parameters=_flatten(grouped),
    )


def _exact_record(path: pathlib.Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ParameterSourceError(f"{label} must be a regular file")
    try:
        value = load_exact_json(path)
    except ExactJSONError as error:
        raise ParameterSourceError(f"{label} is invalid: {error}") from error
    if type(value) is not dict:
        raise ParameterSourceError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _selected_source_commit(
    attempt: pathlib.Path,
    candidate: Candidate,
    study: HPOStudySpec,
    *,
    project_root: pathlib.Path,
    expected_parameters: dict[str, object],
    expected_protocol: dict[str, object] | None,
    attempt_completion: dict[str, object],
) -> tuple[str | None, dict[str, object]]:
    """Verify the selected attempt came from one complete governed runner bundle."""
    evaluation_files = {
        item.relative_to(attempt).as_posix()
        for item in (attempt / "evaluation").rglob("*")
        if item.is_file()
    }
    if evaluation_files != {
        "evaluation/governed-run.json",
        "evaluation/validation.json",
    }:
        raise ParameterSourceError(
            "selected builtin HPO attempt has unexpected evaluation evidence"
        )
    governed = _exact_record(
        attempt / "evaluation" / "governed-run.json", "selected governed-run record"
    )
    governed_keys = {
        "schema",
        "run_id",
        "source_bundle",
        "metric",
        "training_data_fingerprint",
        "validation_data_fingerprint",
        "source_data_fingerprint",
        "test_metrics",
    }
    study_evaluation = governed.get("study_evaluation")
    if "study_evaluation" in governed:
        governed_keys.add("study_evaluation")
        if type(study_evaluation) is not dict:
            raise ParameterSourceError("selected HPO study evaluation is invalid")
    run_id = governed.get("run_id")
    bundle_relative = governed.get("source_bundle")
    metric = governed.get("metric")
    if (
        set(governed) != governed_keys
        or governed.get("schema") != "autoforecast.hpo-governed-run.v1"
        or type(run_id) is not str
        or _RUN_ID.fullmatch(run_id) is None
        or not _is_relative_source_path(bundle_relative)
        or type(metric) is not str
        or not metric
        or not _is_sha256(governed.get("training_data_fingerprint"))
        or not _is_sha256(governed.get("validation_data_fingerprint"))
        or governed.get("source_data_fingerprint") != study.data["fingerprint"]
        or governed.get("test_metrics") != "not_evaluated"
    ):
        raise ParameterSourceError("selected governed-run record is incompatible")
    assert isinstance(bundle_relative, str)
    bundle_parts = pathlib.PurePosixPath(bundle_relative).parts
    if (
        len(bundle_parts) != 5
        or bundle_parts[:3] != ("source-run", run_id, "attempts")
        or bundle_parts[3] != candidate.id
    ):
        raise ParameterSourceError("selected governed-run source bundle path is incompatible")
    bundle = _without_links(attempt.joinpath(*bundle_parts), "selected source bundle").resolve()
    if not bundle.is_relative_to(attempt):
        raise ParameterSourceError("selected source bundle escapes its HPO attempt")
    try:
        completion = load_completed_run_bundle(bundle)
    except (OSError, ProvenanceError) as error:
        raise ParameterSourceError(f"selected source bundle is invalid: {error}") from error
    if completion.get("status") != "completed" or completion.get("run_id") != bundle_parts[-1]:
        raise ParameterSourceError("selected source bundle did not complete successfully")

    validation = _exact_record(
        attempt / "evaluation" / "validation.json", "selected validation record"
    )
    validation_metrics = validation.get("metrics")
    if (
        set(validation) != {"schema", "objective", "value", "metrics", "test_metrics"}
        or validation.get("schema") != "autoforecast.hpo-validation.v1"
        or validation.get("objective") != "validation/objective"
        or validation.get("test_metrics") != "not_evaluated"
        or not _same_exact(
            validation_metrics,
            {f"validation/{metric}": validation.get("value")},
        )
    ):
        raise ParameterSourceError("selected validation record is incompatible")

    run_definition = _exact_record(bundle / "metadata" / "run.json", "source run definition")
    resolved = _exact_record(bundle / "metadata" / "resolved-run.json", "resolved source run")
    plan = _exact_record(bundle / "metadata" / "execution-plan.json", "source execution plan")
    training_controls = _exact_record(
        bundle / "metadata" / "training-controls.json", "source training controls"
    )
    resolved_fields = {
        "schema", "model_id", "adapter", "origin", "horizon", "settings", "runtime"
    }
    if (
        set(run_definition) != {"schema", "run_id", "task", "started_at_utc"}
        or run_definition.get("schema") != "mlops.run-definition.v1"
        or run_definition.get("run_id") != bundle_parts[-1]
        or run_definition.get("task") != "forecast"
        or set(resolved) not in (resolved_fields, resolved_fields | {"frequency"})
        or resolved.get("schema") != "autoforecast.resolved-run.v1"
        or resolved.get("model_id") != candidate.id
        or resolved.get("adapter") != candidate.adapter
        or resolved.get("runtime") != candidate.runtime
        or type(resolved.get("origin")) is not str
        or type(resolved.get("horizon")) is not int
        or cast(int, resolved["horizon"]) < 1
        or type(resolved.get("settings")) is not list
        or set(plan) != {"schema", "readiness_settings", "adapter_kwargs", "effective_parameters"}
        or plan.get("schema") != "autoforecast.execution-plan.v1"
        or not _execution_plan_matches(plan, candidate, expected_parameters)
    ):
        raise ParameterSourceError("selected source execution plan differs from HPO parameters")

    data = _exact_record(bundle / "provenance" / "data.json", "selected source data identity")
    _require_safe_data_reference(data.get("reference"), "selected source data reference")
    runtime = _exact_record(
        bundle / "provenance" / "runtime.json", "selected source runtime identity"
    )
    provenance = _exact_record(
        bundle / "provenance" / "manifest.json", "selected source provenance manifest"
    )
    source = _exact_record(bundle / "provenance" / "source.json", "selected source identity")
    library_source = source.get("schema") == "autoforecast.installed-library.v1"
    if library_source and (
        not installed_mode() or not _same_exact(source, installed_library_identity().as_record())
    ):
        raise ParameterSourceError("selected installed library differs from the active library")
    configuration = provenance.get("configuration")
    model_configuration = (
        configuration.get("model_profile") if type(configuration) is dict else None
    )
    implementation_configuration = (
        configuration.get("model_implementation_config")
        if type(configuration) is dict
        else None
    )
    model_configuration_path = (
        model_configuration.get("path") if type(model_configuration) is dict else None
    )
    runtime_lockfile = runtime.get("lockfile")
    runtime_lockfile_sha = runtime.get("lockfile_sha256")
    if (
        data.get("schema") != "mlops.data-snapshot.v1"
        or data.get("id") != f"{study.data['id']}-hpo-training"
        or data.get("fingerprint") != governed["training_data_fingerprint"]
        or set(runtime)
        != {
            "schema",
            "python",
            "implementation",
            "platform",
            "lockfile",
            "lockfile_sha256",
            "image_digest",
        } | ({"installed_packages"} if library_source else set())
        or runtime.get("schema") != (
            "mlops.runtime-identity.v2" if library_source else "mlops.runtime-identity.v1"
        )
        or (library_source and (
            type(runtime.get("installed_packages")) is not list
            or not runtime["installed_packages"]
            or any(type(row) is not list or len(row) != 2
                   or any(type(value) is not str or not value for value in row)
                   for row in cast(list, runtime["installed_packages"]))
        ))
        or (not library_source and (
            not _is_relative_source_path(runtime_lockfile) or not _is_sha256(runtime_lockfile_sha)
        ))
        or (library_source and (runtime_lockfile is not None or runtime_lockfile_sha is not None))
        or provenance.get("schema") != "mlops.run-provenance.v1"
        or provenance.get("data") != data
        or provenance.get("runtime") != runtime
        or type(model_configuration) is not dict
        or not _is_relative_source_path(model_configuration_path)
        or model_configuration.get("sha256") != candidate.source_sha256
        or (candidate.implementation_config_path is None)
        != (implementation_configuration is None)
    ):
        raise ParameterSourceError("selected source data/configuration provenance is incompatible")
    assert isinstance(model_configuration_path, str)
    retained_profile = _without_links(
        bundle / model_configuration_path, "selected model profile copy"
    )
    if (
        not retained_profile.is_file()
        or hashlib.sha256(retained_profile.read_bytes()).hexdigest() != candidate.source_sha256
    ):
        raise ParameterSourceError("selected source model profile copy is incompatible")
    if candidate.implementation_config_path is not None:
        if type(implementation_configuration) is not dict:
            raise ParameterSourceError("selected source implementation provenance is missing")
        implementation_relative = implementation_configuration.get("path")
        expected_retained_path = (
            "provenance/config/model_implementation_config"
            f"{candidate.implementation_config_path.suffix or '.txt'}"
        )
        if (
            not _is_relative_source_path(implementation_relative)
            or implementation_configuration.get("sha256")
            != candidate.implementation_config_sha256
            or type(implementation_relative) is not str
            or implementation_relative != expected_retained_path
        ):
            raise ParameterSourceError(
                "selected source implementation provenance is incompatible"
            )
        retained_implementation = _without_links(
            bundle / implementation_relative,
            "selected model implementation config copy",
        )
        if (
            not retained_implementation.is_file()
            or hashlib.sha256(retained_implementation.read_bytes()).hexdigest()
            != candidate.implementation_config_sha256
        ):
            raise ParameterSourceError(
                "selected source implementation config copy is incompatible"
            )
    retained_lock = _without_links(
        bundle / "provenance" / "runtime" / "uv.lock", "selected runtime lockfile copy"
    )
    if not library_source and (
        not retained_lock.is_file()
        or hashlib.sha256(retained_lock.read_bytes()).hexdigest() != runtime_lockfile_sha
    ):
        raise ParameterSourceError("selected source runtime lockfile copy is incompatible")
    dataset_configuration = (
        configuration.get("dataset_contract") if type(configuration) is dict else None
    )
    run_configuration = configuration.get("run_request") if type(configuration) is dict else None
    if (
        type(dataset_configuration) is not dict
        or not _is_relative_source_path(dataset_configuration.get("path"))
        or not _is_sha256(dataset_configuration.get("sha256"))
        or type(run_configuration) is not dict
        or not _is_relative_source_path(run_configuration.get("path"))
        or not _is_sha256(run_configuration.get("sha256"))
    ):
        raise ParameterSourceError("selected source protocol configuration is incomplete")
    retained_dataset = _without_links(
        bundle / cast(str, dataset_configuration["path"]),
        "selected source dataset contract",
    )
    retained_request = _without_links(
        bundle / cast(str, run_configuration["path"]), "selected source run request"
    )
    try:
        dataset_bytes = retained_dataset.read_bytes()
        request_bytes = retained_request.read_bytes()
        if (
            hashlib.sha256(dataset_bytes).hexdigest() != dataset_configuration["sha256"]
            or hashlib.sha256(request_bytes).hexdigest() != run_configuration["sha256"]
        ):
            raise ParameterSourceError(
                "selected source protocol configuration bytes are incompatible"
            )
        request_document = loads_exact_yaml(request_bytes)
        request_run = request_document.get("run") if type(request_document) is dict else None
    except (ExactYAMLError, OSError) as error:
        raise ParameterSourceError("selected source run request is invalid") from error
    if type(request_run) is not dict:
        raise ParameterSourceError("selected source run request is invalid")
    from autoforecast.data_admission import _load_dataset_contract_bytes

    retained_contract = _load_dataset_contract_bytes(dataset_bytes, retained_dataset)
    if (retained_contract.required_series_count is None) != (study_evaluation is None):
        raise ParameterSourceError("selected HPO study evaluation does not match its dataset")

    forecast = _exact_record(bundle / "evaluation" / "forecast.json", "selected source forecast")
    model_artifacts = _exact_record(
        bundle / "provenance" / "model-artifacts.json", "selected model artifact manifest"
    )
    files = model_artifacts.get("files")
    completion_summary = completion.get("summary")
    expected_validation_cells = (
        cast(int, study.data["validation_series"]) * cast(int, resolved.get("horizon"))
        if type(resolved.get("horizon")) is int
        else -1
    )
    if (
        model_artifacts.get("schema") != "autoforecast.model-artifacts.v1"
        or type(files) is not list
        or not files
        or set(forecast)
        != {
            "adapter",
            "path",
            "origin",
            "horizon",
            "n_series",
            "disposition",
            "target_estimand",
            "frequency",
            "n_emitted",
            "n_declined",
        }
        or forecast.get("adapter") != candidate.adapter
        or forecast.get("origin") != resolved.get("origin")
        or any(
            type(forecast.get(field)) is not int
            for field in ("horizon", "n_series", "n_emitted", "n_declined")
        )
        or forecast.get("horizon") != resolved.get("horizon")
        or forecast.get("n_series") != study.data["validation_series"]
        or forecast.get("n_emitted") != expected_validation_cells
        or forecast.get("n_declined") != 0
        or type(completion_summary) is not dict
        or completion_summary.get("forecast_path") != forecast.get("path")
    ):
        raise ParameterSourceError("selected source forecast/model evidence is incomplete")
    try:
        cadence = parse_frequency(cast(str, forecast["frequency"]))
        resolved_cadence = parse_frequency(cast(str, resolved.get("frequency", "D")))
        origin = parse_origin(cast(str, resolved["origin"]), cadence)
        target_start, target_end = (
            parse_origin(item, cadence)
            for item in cast(list[str], study.data["validation_target_date_range"])
        )
        expected = period_points(origin, cadence, cast(int, resolved["horizon"]))
    except (FrequencyError, TypeError, ValueError) as error:
        raise ParameterSourceError("selected source temporal evidence is invalid") from error
    if (
        resolved_cadence.code != cadence.code
        or not frequency_is_supported(
            cadence,
            descriptor(candidate.adapter).supported_frequency_families,
            supports_strided_daily=descriptor(candidate.adapter).supports_strided_daily,
        )
        or expected[0] != target_start
        or expected[-1] != target_end
        or cast(dict[str, object], plan["effective_parameters"])["data.horizon"]
        != resolved["horizon"]
    ):
        raise ParameterSourceError("selected source temporal evidence differs from the HPO split")
    data_check = _exact_record(bundle / "checks" / "data.json", "selected source data check")
    facts = data_check.get("facts")
    if (
        type(facts) is not dict
        or facts.get("as_of") != resolved["origin"]
        or facts.get("data_range") != study.data["training_date_range"]
        or facts.get("data_range") != study.data["validation_date_range"]
        or type(facts.get("n_series")) is not int
        or facts.get("n_series") != study.data["training_series"]
        or facts.get("data_fingerprint") != governed["training_data_fingerprint"]
    ):
        raise ParameterSourceError("selected source data check differs from the HPO split")
    model_paths: set[str] = set()
    for item in files:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise ParameterSourceError("selected model artifact manifest is invalid")
        relative, expected_sha = item.get("path"), item.get("sha256")
        if not _is_relative_source_path(relative) or not _is_sha256(expected_sha):
            raise ParameterSourceError("selected model artifact manifest is invalid")
        assert isinstance(relative, str) and isinstance(expected_sha, str)
        if relative in model_paths:
            raise ParameterSourceError("selected model artifact manifest contains duplicates")
        model_paths.add(relative)
        nested_artifact = _without_links(bundle / relative, "selected model artifact")
        copied_artifact = _without_links(attempt / relative, "copied HPO model artifact")
        if (
            not relative.startswith("model/")
            or not nested_artifact.is_file()
            or not copied_artifact.is_file()
            or hashlib.sha256(nested_artifact.read_bytes()).hexdigest() != expected_sha
            or hashlib.sha256(copied_artifact.read_bytes()).hexdigest() != expected_sha
        ):
            raise ParameterSourceError("selected model artifact bytes are incompatible")

    comparison = _exact_record(
        attempt / "source-run" / cast(str, run_id) / "evaluation" / "candidate-comparison.json",
        "selected source comparison",
    )
    comparison_candidates = comparison.get("candidates")
    comparison_candidate = (
        next(
            (
                item
                for item in comparison_candidates
                if type(item) is dict and item.get("model_id") == candidate.id
            ),
            None,
        )
        if type(comparison_candidates) is list
        else None
    )
    score = comparison_candidate.get("score") if type(comparison_candidate) is dict else None
    metric_value = score.get(metric) if type(score) is dict else None
    ranking = comparison.get("ranking")
    ranking_record = ranking if type(ranking) is dict else {}
    validation_value = validation.get("value")
    score_evidence = _exact_record(
        attempt / "source-run" / cast(str, run_id) / "evaluation" / "post-training-evaluation.json",
        "selected source score input",
    )
    score_input = score_evidence.get("score_input")
    if type(score_input) is dict:
        _require_safe_data_reference(
            score_input.get("reference"), "selected source score data reference"
        )
    forecast_artifacts = score_evidence.get("forecast_artifacts")
    try:
        forecast_artifact = _without_links(
            cast(str, forecast.get("path")), "selected source forecast artifact"
        ).resolve()
        expected_forecast_artifact = (bundle / "evaluation" / "forecast").resolve()
        forecast_digest = directory_digest(forecast_artifact, suffixes=(".parquet",))
    except (LineageError, OSError, TypeError, ValueError) as error:
        raise ParameterSourceError("selected source forecast artifact is invalid") from error
    fixed = comparison.get("evaluation_contract")
    if study_evaluation is not None:
        from autoforecast.hpo_bridge import _require_exact_validation_population

        if type(fixed) is not dict:
            raise ParameterSourceError("selected HPO study evaluation is missing")
        try:
            _require_exact_validation_population(
                comparison, forecast, model_id=candidate.id, metric=cast(str, metric),
                validation_series=cast(int, study.data["validation_series"]),
                horizon=cast(int, resolved["horizon"]), fixed_contract=fixed,
            )
        except (HPOError, ValueError) as error:
            raise ParameterSourceError(f"selected HPO study score is invalid: {error}") from error
        expected_study = {
            "study_id": fixed["study_id"], "boundary": fixed["boundary"],
            "evaluation_contract_sha256": hashlib.sha256(_canonical(fixed)).hexdigest(),
            "metric_policy": fixed["metric_policy"],
        }
        score_shape_valid = (
            _same_exact(study_evaluation, expected_study)
            and fixed["boundary"] == (
                f"tuning/{resolved['origin']}/{resolved['horizon']}/{cadence.code}"
            )
            and fixed["actuals"]["sha256"] == governed["source_data_fingerprint"]
            and fixed["actuals"]["sha256"] == governed["validation_data_fingerprint"]
            and _same_exact(metric_value, validation_value)
        )
    elif metric == "wrmsse":
        per_level = score.get("per_level") if type(score) is dict else None
        scored_by_level = score.get("n_series_scored") if type(score) is dict else None
        dropped_by_level = score.get("n_series_dropped") if type(score) is dict else None
        score_shape_valid = (
            comparison.get("comparison_kind") == "m5_full_grid_wrmsse"
            and type(score) is dict
            and set(score) == {"wrmsse", "per_level", "n_series_scored", "n_series_dropped"}
            and type(per_level) is dict
            and bool(per_level)
            and type(scored_by_level) is dict
            and type(dropped_by_level) is dict
            and set(per_level) == set(scored_by_level) == set(dropped_by_level)
            and all(is_finite_real(value) and float(value) >= 0 for value in per_level.values())
            and all(type(value) is int and value > 0 for value in scored_by_level.values())
            and all(type(value) is int and value == 0 for value in dropped_by_level.values())
            and _same_exact(score.get("wrmsse"), validation_value)
            and type(ranking_record.get("n_common_cells")) is int
            and ranking_record.get("n_common_cells") == expected_validation_cells
        )
    else:
        score_shape_valid = (
            comparison.get("comparison_kind") == "exploratory_common_cells"
            and type(score) is dict
            and set(score)
            == {
                "bias",
                "mae",
                "n_dropped_ineligible",
                "n_dropped_no_data",
                "n_scored",
                "n_zero_filled",
                "rmse",
                "wape",
            }
            and _same_exact(metric_value, validation_value)
            and type(ranking_record.get("n_common_cells")) is int
            and ranking_record.get("n_common_cells") == expected_validation_cells
            and type(score.get("n_scored")) is int
            and score.get("n_scored") == expected_validation_cells
            and type(score.get("n_dropped_ineligible")) is int
            and score.get("n_dropped_ineligible") == 0
            and type(score.get("n_dropped_no_data")) is int
            and score.get("n_dropped_no_data") == 0
            and type(zero_filled := score.get("n_zero_filled")) is int
            and 0 <= zero_filled <= expected_validation_cells
        )
    if (
        set(comparison)
        != {
            "schema",
            "comparison_kind",
            "metric",
            "missing_actual",
            "candidates",
            "ranking",
            "comparison_error",
        } | ({"evaluation_contract"} if study_evaluation is not None else set())
        or comparison.get("schema") != "autoforecast.run-candidate-comparison.v1"
        or not score_shape_valid
        or comparison.get("comparison_error") is not None
        or comparison.get("metric") != metric
        or comparison_candidates != [comparison_candidate]
        or type(comparison_candidate) is not dict
        or set(comparison_candidate) != {"model_id", "status", "score", "detail"}
        or comparison_candidate.get("status") != "scored"
        or comparison_candidate.get("detail") != ""
        or type(ranking) is not dict
        or set(ranking) != {"metric", "n_common_cells", "order"}
        or ranking.get("metric") != metric
        or ranking.get("order") != [candidate.id]
        or set(score_evidence)
        != {
            "schema",
            "status",
            "evidence_role",
            "score_input",
            "forecast_artifacts",
            "comparison",
        }
        or score_evidence.get("schema") != "autoforecast.post-training-evaluation.v1"
        or score_evidence.get("status") != "scored"
        or score_evidence.get("evidence_role") != "hpo_validation_selection"
        or type(score_input) is not dict
        or score_input.get("fingerprint") != governed["validation_data_fingerprint"]
        or type(forecast_artifacts) is not dict
        or set(forecast_artifacts) != {candidate.id}
        or forecast_artifact != expected_forecast_artifact
        or forecast_artifacts.get(candidate.id) != forecast_digest
        or attempt_completion.get("summary")
        != {
            "selection_metric": study.objective,
            "validation_objective": validation_value,
            "test_metrics": "not_evaluated",
        }
    ):
        raise ParameterSourceError("selected source score differs from completed HPO evidence")

    if not library_source and set(source) != {
        "schema",
        "repository_root",
        "commit",
        "tree",
        "origin_url",
        "relevant_paths",
    }:
        raise ParameterSourceError("selected source identity is incomplete")
    if provenance.get("source") != source:
        raise ParameterSourceError("selected source provenance manifest disagrees with source")
    commit = None
    if not library_source:
        commit, tree = source.get("commit"), source.get("tree")
        relevant_paths = source.get("relevant_paths")
        repository_root = source.get("repository_root")
        origin_url = source.get("origin_url")
        if type(relevant_paths) is not list or any(
            type(item) is not str or not item for item in relevant_paths
        ):
            raise ParameterSourceError("selected source identity is incompatible")
        if origin_url is not None and (
            type(origin_url) is not str or sanitize_repository_origin(origin_url) != origin_url
        ):
            raise ParameterSourceError("selected source identity is incompatible")
        if (
            source.get("schema") != "mlops.source-identity.v1"
            or not _is_commit(commit)
            or not _is_commit(tree)
            or type(repository_root) is not str
            or not pathlib.Path(repository_root).is_absolute()
            or not {
                "autoforecast",
                "forecast",
                "mlops",
                "provenance",
                "scripts",
                "spec",
                "configs",
                "pyproject.toml",
                "uv.lock",
            }.issubset(set(relevant_paths))
        ):
            raise ParameterSourceError("selected source identity is incompatible")
        assert isinstance(commit, str) and isinstance(tree, str)
        commit_check = subprocess.run(
            ["git", "-C", str(project_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        tree_check = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", f"{commit}^{{tree}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_check.returncode or tree_check.returncode or tree_check.stdout.strip() != tree:
            raise ParameterSourceError("selected source Git commit/tree is not retained locally")
        source_configs = [
            (candidate.source_path, candidate.source_sha256),
            (study.source_path, study.source_sha256),
            (project_root / cast(str, runtime_lockfile), cast(str, runtime_lockfile_sha)),
        ]
        if candidate.implementation_config_path is not None:
            assert candidate.implementation_config_sha256 is not None
            source_configs.append(
                (
                    candidate.implementation_config_path,
                    candidate.implementation_config_sha256,
                )
            )
        for source_path, expected_sha in source_configs:
            try:
                relative = source_path.resolve().relative_to(project_root).as_posix()
            except ValueError as error:
                raise ParameterSourceError(
                    "selected source configuration is outside the project"
                ) from error
            blob = subprocess.run(
                ["git", "-C", str(project_root), "show", f"{commit}:{relative}"],
                capture_output=True,
                check=False,
            )
            if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != expected_sha:
                raise ParameterSourceError(
                    "selected source commit does not contain the declared model/study configuration"
                )
    observed_protocol = {
        "metric": metric,
        "missing_actual": comparison.get("missing_actual"),
        "selection_data_fingerprint": governed.get("source_data_fingerprint"),
        "dataset_contract_sha256": dataset_configuration.get("sha256"),
        "features": request_run.get("features"),
        "training_seed": cast(dict[str, object], plan["effective_parameters"]).get("training.seed"),
        "maximum_epochs": training_controls.get("maximum_epochs"),
        "patience": training_controls.get("patience"),
        "device": training_controls.get("device"),
        "batch_size": training_controls.get("batch_size"),
        "gradient_microbatch_size": training_controls.get("gradient_microbatch_size"),
        "early_stopping_absolute_min_delta": training_controls.get(
            "early_stopping_absolute_min_delta"
        ),
        "early_stopping_relative_min_delta": training_controls.get(
            "early_stopping_relative_min_delta"
        ),
        "precision": training_controls.get("effective_precision"),
        **({"study_evaluation": study_evaluation} if study_evaluation is not None else {}),
        **({"library": source} if library_source else {}),
    }
    if (
        not _training_controls_match_candidate(training_controls, candidate)
        or expected_protocol is not None
        and not _same_exact(observed_protocol, expected_protocol)
    ):
        raise ParameterSourceError("selected HPO protocol differs from the campaign protocol")
    return commit, observed_protocol


def parameter_resolution_from_selection(
    candidate: Candidate,
    study: HPOStudySpec,
    selection_path: str | pathlib.Path,
    *,
    kind: str,
    project_root: str | pathlib.Path,
    expected_protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    """Verify a sealed HPO winner and materialize its complete parameter resolution."""
    if kind not in {"tune", "reuse_selection"}:
        raise ParameterSourceError("HPO selection kind must be tune or reuse_selection")
    root = _without_links(project_root, "project root").resolve()
    selection = _without_links(selection_path, "HPO selection").resolve()
    if not selection.is_file() or selection.name != "selection.json":
        raise ParameterSourceError("HPO selection must be a regular selection.json")
    study_path = _without_links(study.source_path, "HPO study config").resolve()
    if not study_path.is_file() or not study_path.is_relative_to(root):
        raise ParameterSourceError("HPO study config must remain below the project root")
    if study.evaluator != "builtin":
        raise ParameterSourceError("campaign parameter selection requires evaluator: builtin")
    if (
        study.tuning.model_id != candidate.id
        or study.tuning.adapter != candidate.adapter
        or study.tuning.source_sha256 != candidate.source_sha256
    ):
        raise ParameterSourceError("HPO study model profile differs from the campaign candidate")
    if selection.parent.name != study.id:
        raise ParameterSourceError("HPO selection path does not match its declared study ID")
    try:
        selection_bytes = selection.read_bytes()
        sealed = load_study_selection(selection.parent)
    except (ExperimentError, ExactJSONError) as error:
        raise ParameterSourceError(f"HPO selection is invalid: {error}") from error
    selected = sealed.get("selected")
    if not isinstance(selected, list) or not selected or not isinstance(selected[0], dict):
        raise ParameterSourceError("HPO selection has no rank-one candidate")
    winner = cast(dict[str, object], selected[0])
    trial_id, attempt_id = winner.get("trial_id"), winner.get("attempt_id")
    if (
        not isinstance(trial_id, str)
        or not isinstance(attempt_id, str)
        or not _is_rank_one(winner.get("rank"))
        or sealed.get("direction") != study.direction
        or sealed.get("comparability_fingerprint") != study.comparability_fingerprint
    ):
        raise ParameterSourceError("HPO rank-one candidate identity is invalid")
    try:
        approved = require_heldout_evaluation_approval(
            selection.parent, trial_id=trial_id, attempt_id=attempt_id
        )
        relative = approved.get("attempt_path")
        if not isinstance(relative, str) or approved != winner:
            raise ParameterSourceError("selected HPO attempt path is invalid")
        attempt = _without_links(selection.parent / relative, "selected HPO attempt").resolve()
        metadata, completion = load_completed_attempt(attempt)
    except (ExperimentError, ExactJSONError) as error:
        raise ParameterSourceError(f"selected HPO attempt is invalid: {error}") from error
    model = metadata.get("model")
    selected_study = metadata.get("study")
    expected_study: dict[str, object] = {
        "config_sha256": study.source_sha256,
        "objective": study.objective,
        "direction": study.direction,
        "backend": study.backend,
        "top_k": study.top_k,
    }
    parameters = metadata.get("hyperparameters")
    expected_data = {"name": study.data["id"], **study.data}
    if (
        completion.get("status") != "completed"
        or completion.get("artifact_manifest_sha256") != winner.get("artifact_manifest_sha256")
        or metadata.get("purpose") != "hpo_validation_selection"
        or metadata.get("study_id") != study.id
        or metadata.get("trial_id") != trial_id
        or metadata.get("attempt_id") != attempt_id
        or metadata.get("comparability_fingerprint") != study.comparability_fingerprint
        or model
        != {
            "id": candidate.id,
            "adapter": candidate.adapter,
            "config_sha256": candidate.source_sha256,
        }
        or selected_study != expected_study
        or sealed.get("top_k") != study.top_k
        or metadata.get("data") != expected_data
        or metadata.get("parameter_execution_compatibility") != PARAMETER_EXECUTION_COMPATIBILITY
        or type(parameters) is not dict
    ):
        raise ParameterSourceError("selected HPO metadata is incompatible with this model/study")
    try:
        effective = _flatten(
            materialize_trial_parameters(study.tuning, cast(dict[str, object], parameters))
        )
    except TuningError as error:
        raise ParameterSourceError(f"selected HPO parameters are invalid: {error}") from error
    if selection.read_bytes() != selection_bytes:
        raise ParameterSourceError("HPO selection changed while it was being verified")
    selected_source_commit, observed_protocol = _selected_source_commit(
        attempt,
        candidate,
        study,
        project_root=root,
        expected_parameters=effective,
        expected_protocol=expected_protocol,
        attempt_completion=completion,
    )
    source = {
        "kind": kind,
        "study_id": study.id,
        "study_config_path": study_path.relative_to(root).as_posix(),
        "study_config_sha256": study.source_sha256,
        "selection_path": str(selection),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "selected_trial_id": trial_id,
        "selected_attempt_id": attempt_id,
        "selected_attempt_path": str(attempt),
        "selected_attempt_manifest_sha256": completion.get("artifact_manifest_sha256"),
        "protocol": observed_protocol,
        "selected_source_commit": selected_source_commit,
    }
    return _record(candidate, source=source, effective_parameters=effective)


def validate_parameter_resolution(
    record: dict[str, object],
    candidate: Candidate,
    *,
    project_root: str | pathlib.Path | None = None,
    verify_external: bool = False,
) -> dict[str, object]:
    """Validate a frozen resolution and optionally replay its external HPO provenance."""
    required = {
        "schema",
        "compatibility",
        "model_id",
        "adapter",
        "model_profile_sha256",
        "source",
        "effective_parameters",
    }
    if type(record) is not dict or set(record) != required:
        raise ParameterSourceError(f"parameter resolution must contain exactly {sorted(required)}")
    if (
        record["schema"] != PARAMETER_RESOLUTION_SCHEMA
        or record["compatibility"] != PARAMETER_EXECUTION_COMPATIBILITY
        or record["model_id"] != candidate.id
        or record["adapter"] != candidate.adapter
        or record["model_profile_sha256"] != candidate.source_sha256
    ):
        raise ParameterSourceError("parameter resolution identity differs from its model profile")
    source = record["source"]
    values = record["effective_parameters"]
    if type(source) is not dict or type(values) is not dict:
        raise ParameterSourceError("parameter resolution source/values must be mappings")
    kind = source.get("kind")
    if type(kind) is not str or kind not in _KINDS:
        raise ParameterSourceError("parameter resolution source kind is unsupported")
    spec = _spec(candidate)
    if kind == "defaults":
        if set(source) != {"kind"}:
            raise ParameterSourceError("defaults source has inactive fields")
        expected = {} if spec is None else _flatten(spec.defaults())
    elif kind == "manual":
        if (
            set(source) != {"kind", "base", "overrides"}
            or source.get("base") != "profile_defaults"
            or type(source.get("overrides")) is not dict
        ):
            raise ParameterSourceError("manual source requires only base and overrides")
        if spec is None:
            raise ParameterSourceError("manual source requires a typed parameter surface")
        try:
            expected = _flatten(
                materialize_manual_parameters(spec, cast(dict[str, object], source["overrides"]))
            )
        except TuningError as error:
            raise ParameterSourceError(str(error)) from error
    else:
        hpo_fields = {
            "kind",
            "study_id",
            "study_config_path",
            "study_config_sha256",
            "selection_path",
            "selection_sha256",
            "selected_trial_id",
            "selected_attempt_id",
            "selected_attempt_path",
            "selected_attempt_manifest_sha256",
            "protocol",
            "selected_source_commit",
        }
        if (
            set(source) != hpo_fields
            or any(
                not _is_sha256(source.get(name))
                for name in (
                    "study_config_sha256",
                    "selection_sha256",
                    "selected_attempt_manifest_sha256",
                )
            )
            or (
                type(source.get("study_id")) is not str
                or _ID.fullmatch(cast(str, source["study_id"])) is None
                or not _is_relative_source_path(source.get("study_config_path"))
                or not _is_absolute_artifact_path(
                    source.get("selection_path"), name="selection.json"
                )
                or not _is_absolute_artifact_path(source.get("selected_attempt_path"))
                or type(source.get("selected_trial_id")) is not str
                or _ID.fullmatch(cast(str, source["selected_trial_id"])) is None
                or type(source.get("selected_attempt_id")) is not str
                or _ID.fullmatch(cast(str, source["selected_attempt_id"])) is None
                or type(source.get("protocol")) is not dict
                or (
                    not _is_commit(source.get("selected_source_commit"))
                    and not (
                        source.get("selected_source_commit") is None
                        and type(source.get("protocol")) is dict
                        and type(source["protocol"].get("library")) is dict
                        and source["protocol"]["library"].get("schema")
                        == "autoforecast.installed-library.v1"
                    )
                )
            )
        ):
            raise ParameterSourceError("HPO parameter source provenance is incomplete")
        if spec is None:
            raise ParameterSourceError("HPO source requires a typed parameter surface")
        try:
            expected = _flatten(
                materialize_runtime_parameters(spec, cast(dict[str, object], values))
            )
        except TuningError as error:
            raise ParameterSourceError(str(error)) from error
        if verify_external:
            if project_root is None:
                raise ParameterSourceError("project_root is required to verify HPO provenance")
            study_path = pathlib.Path(project_root) / cast(str, source["study_config_path"])
            try:
                study = load_hpo_study(study_path, project_root=project_root)
            except (HPOError, TuningError) as error:
                raise ParameterSourceError(f"HPO study cannot be replayed: {error}") from error
            replayed = parameter_resolution_from_selection(
                candidate,
                study,
                cast(str, source["selection_path"]),
                kind=cast(str, kind),
                project_root=project_root,
                expected_protocol=cast(dict[str, object], source["protocol"]),
            )
            if _canonical(replayed) != _canonical(record):
                raise ParameterSourceError("HPO parameter resolution differs from sealed evidence")
    if _canonical(cast(dict[str, object], values)) != _canonical(expected):
        raise ParameterSourceError("effective parameters differ from their declared source")
    if candidate.adapter in {
        "sft",
        "hmfnet",
        "pact",
    }:
        try:
            model_parameters = {
                name.removeprefix("model."): value
                for name, value in expected.items()
                if name.startswith("model.")
            }
            training_parameters = {
                name.removeprefix("training."): value
                for name, value in expected.items()
                if name.startswith("training.")
            }
            model_parameters, training_parameters = validate_revision_parameter_groups(
                candidate,
                model_parameters,
                training_parameters,
            )
            resolve_revision_rolling_config(
                candidate,
                model_parameters=model_parameters,
                training_parameters=training_parameters,
                epochs=1,
                seed=0,
            )
        except ExecutionPlanError as error:
            raise ParameterSourceError(
                f"revision parameter source cannot form a valid execution recipe: {error}"
            ) from error
    return record


def load_parameter_resolution(
    path: str | pathlib.Path,
    expected_sha256: str,
    candidate: Candidate,
    *,
    project_root: str | pathlib.Path,
) -> dict[str, object]:
    """Load one immutable JSON resolution and replay all applicable provenance."""
    source = _without_links(path, "parameter resolution").resolve()
    if not source.is_file() or not _is_sha256(expected_sha256):
        raise ParameterSourceError("parameter resolution path or SHA-256 is invalid")
    try:
        raw = source.read_bytes()
        record = loads_exact_json(raw)
    except (OSError, ExactJSONError) as error:
        raise ParameterSourceError(f"parameter resolution JSON is invalid: {error}") from error
    if (
        type(record) is not dict
        or raw != _canonical(cast(dict[str, object], record))
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ParameterSourceError("parameter resolution bytes do not match its identity")
    return validate_parameter_resolution(
        cast(dict[str, object], record),
        candidate,
        project_root=project_root,
        verify_external=True,
    )
