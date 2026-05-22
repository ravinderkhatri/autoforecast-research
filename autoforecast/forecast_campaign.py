"""Model/data-agnostic forecasting campaigns on the durable local supervisor.

The high-level request freezes data, model, temporal, initialization, and budget identities. It
then compiles those identities into ordinary :mod:`autoforecast.campaign` stages. Training and
selection never receive the separate final-score input; only the score stage does.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import pathlib
import pickle
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from mlops._safe_files import (
    SafeFileError,
    _create_fresh_directory,
    publish_bytes_set,
)
from mlops.provenance import ProvenanceError, capture_source_identity, validate_data_reference

from autoforecast import resources
from autoforecast._exact_json import ExactJSONError, load_exact_json, loads_exact_json
from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast._numeric import is_finite_real
from autoforecast.adapters import validate_execution_controls
from autoforecast.benchmarks.policy import is_m5_daily
from autoforecast.campaign import CampaignError, CampaignStore
from autoforecast.data_admission import load_dataset_contract
from autoforecast.hpo import HPOError, HPOStudySpec, load_hpo_study, run_hpo_study
from autoforecast.hpo_bridge import ForecastTrialBridge
from autoforecast.model_catalog import Candidate, discover_candidates
from autoforecast.panel import Forecast, period_days
from autoforecast.parameter_sources import (
    ParameterSourceError,
    _training_controls_are_valid,
    default_parameter_resolution,
    manual_parameter_resolution,
    parameter_resolution_from_selection,
    parameter_resolution_sha256,
    validate_parameter_resolution,
    validate_parameter_source_capsule,
)
from autoforecast.precision_policy import effective_precision
from autoforecast.project import (
    installed_library_identity,
    installed_mode,
    project_arguments,
    project_entrypoint,
    project_root,
)
from autoforecast.revision_config import verify_validation_ledger
from autoforecast.run_evaluation import compare_run_forecasts
from autoforecast.train_workflow import load_run_request
from autoforecast.training_lease import inherited_training_lease_pass_fds
from autoforecast.training_progress import PipelineProgress
from provenance.lineage import directory_content_digest, directory_digest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUEST_SCHEMA = "autoforecast.forecast-campaign-request.v4"
RESOLVED_SCHEMA = "autoforecast.forecast-campaign-resolved.v4"
INSTALLED_RESOLVED_SCHEMA = "autoforecast.forecast-campaign-resolved.v5"
LEGACY_RESOLVED_SCHEMA = "autoforecast.forecast-campaign-resolved.v3"
SCORE_INPUT_SCHEMA = "autoforecast.forecast-campaign-score-input.v1"
CHILD_RESULT_SCHEMA = "autoforecast.forecast-campaign-child-result.v3"
CHILD_RESULT_PRECISION_SCHEMA = "autoforecast.forecast-campaign-child-result.v4"
SELECTION_LOCK_SCHEMA = "autoforecast.forecast-campaign-selection-lock.v4"
FINAL_SCORE_SCHEMA = "autoforecast.forecast-campaign-final-score.v3"
REPLAY_SCHEMA = "autoforecast.forecast-campaign-replay.v3"
VALIDATION_SUMMARY_SCHEMA = "autoforecast.forecast-campaign-validation-summary.v2"
REPORT_SCHEMA = "autoforecast.forecast-campaign-report.v2"

_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
_SHA_CHARS = frozenset("0123456789abcdef")
_ARMS = ("fresh", "full_state_continuation", "weights_only_finetune")
_FIXED_BUDGET_ADAPTERS = frozenset(
    {"sft", "hmfnet", "pact",
     "tft_q", "tft_hnb", }
)
_ROLLING_BUDGET_ADAPTERS = _FIXED_BUDGET_ADAPTERS - {
    "tft_q",
    "tft_hnb",

}
_LEGACY_TIE_BREAK = ("mean_score", "score_stddev", "model_id", "initialization")
_TIE_BREAK = (
    "mean_score",
    "score_stddev",
    "model_id",
    "variant_id",
    "initialization",
)

InitializationArm = Literal["fresh", "full_state_continuation", "weights_only_finetune"]
ParameterMode = Literal["defaults", "manual", "tune", "reuse_selection"]
CampaignPurpose = Literal[
    "forecast_campaign",
    "single_model_benchmark",
    "research_campaign",
    "capacity_diagnostic",
]


class ForecastCampaignError(RuntimeError):
    """A high-level campaign request or one of its evidence artifacts is invalid."""


def _root() -> pathlib.Path:
    return project_root(ROOT)


def _json(value: object) -> str:
    try:
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as error:
        raise ForecastCampaignError(f"campaign evidence is not finite JSON: {error}") from error


def _metric_summary_banner(metric: str, validation: float, test: float) -> str:
    """Render the final validation/test metric pair as one prominent terminal line."""
    name = metric.upper()
    body = (
        f" FINAL {name} SUMMARY | Validation {name}: {validation:.6f} | "
        f"Test {name}: {test:.6f} "
    )
    columns = shutil.get_terminal_size(fallback=(120, 24)).columns
    width = max(len(body) + 12, min(max(columns, 80), 160))
    return body.center(width, "#")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _exact(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        got = sorted(value) if type(value) is dict else type(value).__name__
        raise ForecastCampaignError(f"{label} requires exactly {sorted(keys)}, got {got}")
    return cast(dict[str, object], value)


def _same_exact(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        observed_dict = cast(dict[object, object], observed)
        return observed_dict.keys() == expected.keys() and all(
            _same_exact(observed_dict[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        observed_list = cast(list[object], observed)
        return len(observed_list) == len(expected) and all(
            _same_exact(left, right) for left, right in zip(observed_list, expected, strict=True)
        )
    return observed == expected


def _identifier(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not 2 <= len(value) <= 64
        or value[0] not in "abcdefghijklmnopqrstuvwxyz"
        or any(char not in _ID_CHARS for char in value)
    ):
        raise ForecastCampaignError(f"{label} must be a lowercase campaign identifier")
    return value


def _date(value: object, label: str) -> dt.date:
    if type(value) is not str:
        raise ForecastCampaignError(f"{label} must be an ISO date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as error:
        raise ForecastCampaignError(f"{label} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ForecastCampaignError(f"{label} must be a canonical ISO date")
    return parsed


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ForecastCampaignError(f"{label} must be an integer >= {minimum}")
    return value


def _nonnegative_number(value: object, label: str) -> float:
    if not is_finite_real(value) or float(value) < 0:
        raise ForecastCampaignError(f"{label} must be a finite non-negative number")
    return float(value)


def _path_without_links(path: str | pathlib.Path, label: str) -> pathlib.Path:
    """Return an absolute lexical path after rejecting every symlink component."""
    lexical = pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))
    cursor = lexical
    while True:
        if cursor.is_symlink():
            alias = (
                unicodedata.normalize("NFC", cursor.as_posix()).casefold(),
                unicodedata.normalize("NFC", cursor.resolve().as_posix()).casefold(),
            )
            if not (
                sys.platform == "darwin" and alias in {
                    ("/var", "/private/var"), ("/tmp", "/private/tmp")
                }
            ):
                raise ForecastCampaignError(f"{label} may not traverse a symlink: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return lexical


def _run_group_path(
    attempt_dir: pathlib.Path, value: str | pathlib.Path, label: str
) -> pathlib.Path:
    """Bind one governed run group to the current stage attempt's run directory."""
    run_root = _path_without_links(attempt_dir / "run", f"{label} root")
    run_path = _path_without_links(value, label)
    if run_path.parent != run_root or not run_path.is_dir():
        raise ForecastCampaignError(f"{label} is outside the current stage attempt")
    return run_path


def _published_child_run_groups(run_root: pathlib.Path) -> list[pathlib.Path]:
    """Return child-published run groups, excluding its private worker scratch directory."""
    return sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and path.name != ".autoforecast-worker-scratch"
    )


def _run_checkpoint_path(
    run_path: pathlib.Path,
    model_id: str,
    value: str | pathlib.Path,
) -> pathlib.Path:
    """Bind checkpoint evidence to the unique model checkpoint in this run group."""
    checkpoint_path = _path_without_links(value, "child checkpoint")
    expected = sorted((run_path / "attempts" / model_id).glob("*/model/model.pt"))
    if expected != [checkpoint_path] or not checkpoint_path.is_file():
        raise ForecastCampaignError("child checkpoint is outside the governed child run")
    return checkpoint_path


def _yaml_load(path: pathlib.Path) -> tuple[dict[str, object], bytes]:
    path = _path_without_links(path, "request")
    if not path.is_file():
        raise ForecastCampaignError(f"request must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = loads_exact_yaml(raw)
    except ExactYAMLError as error:
        raise ForecastCampaignError(f"invalid YAML in {path}: {error}") from error
    if type(value) is not dict:
        raise ForecastCampaignError("forecast campaign request must be a mapping")
    return cast(dict[str, object], value), raw


@dataclass(frozen=True)
class DataInput:
    """One declared source whose concrete bytes are supplied as a runtime binding."""

    snapshot_id: str | None
    reference: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class TrainingBudget:
    """Comparable optimizer and stopping budget for one run class."""

    maximum_epochs: int
    patience: int
    timeout_seconds: int
    max_attempts: int
    absolute_min_delta: float
    relative_min_delta: float

    def as_record(self) -> dict[str, object]:
        return {
            "maximum_epochs": self.maximum_epochs,
            "patience": self.patience,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "early_stopping_absolute_min_delta": self.absolute_min_delta,
            "early_stopping_relative_min_delta": self.relative_min_delta,
        }


@dataclass(frozen=True)
class ComputeBudget:
    """Per-stage admission and training batch controls."""

    device: Literal["cpu", "cuda", "mps"]
    batch_size: int
    gradient_microbatch_size: int | None
    memory_gib: float
    min_free_disk_gib: float
    precision: Literal["32-true", "16-mixed"] = "32-true"

    def resources(self) -> dict[str, object]:
        return {
            "device": self.device,
            "memory_gib": self.memory_gib,
            "min_free_disk_gib": self.min_free_disk_gib,
        }


@dataclass(frozen=True)
class ParameterSourceRequest:
    """Exactly one human-selected parameter source for one campaign model."""

    mode: ParameterMode
    base: str | None = None
    values: dict[str, object] | None = None
    study_config: pathlib.Path | None = None
    selection: pathlib.Path | None = None


@dataclass(frozen=True)
class ForecastCampaignRequest:
    """Strict user intent before local paths are content-hashed and resolved."""

    schema_version: int
    name: str
    purpose: CampaignPurpose
    contract: pathlib.Path
    template: pathlib.Path | None
    selection_data: DataInput
    final_training_data: DataInput
    final_score_data: DataInput
    catalog: pathlib.Path
    model_ids: tuple[str, ...]
    variant_ids: tuple[str | None, ...]
    candidate_ids: tuple[str | None, ...]
    parameter_sources: tuple[ParameterSourceRequest, ...]
    training_start: dt.date | None
    initialization_origin: dt.date | None
    validation_origins: tuple[dt.date, ...]
    validation_horizon: int
    final_train_cutoff: dt.date
    final_test_horizon: int
    seeds: tuple[int, ...]
    initialization_arms: tuple[InitializationArm, ...]
    metric: Literal["rmse", "mae", "wape", "wrmsse"]
    missing_actual: Literal["sparse_zero", "no_data"]
    evidence_role: str
    initialization_budget: TrainingBudget
    validation_budget: TrainingBudget
    final_fit_budget: TrainingBudget
    compute: ComputeBudget
    mlflow_enabled: bool
    source_path: pathlib.Path
    source_bytes: bytes


@dataclass(frozen=True)
class CompiledForecastCampaign:
    """Immutable resolved inputs plus the executor-neutral P0 plan."""

    directory: pathlib.Path
    plan_path: pathlib.Path
    resolved_path: pathlib.Path
    score_input_path: pathlib.Path
    preflight_lines: tuple[str, ...]


def _candidate_id(model_id: str, variant_id: str) -> str:
    """Return the stable campaign-arm identity without changing the catalog model identity."""
    return model_id if variant_id == "default" else f"@{len(model_id)}:{model_id}:{variant_id}"


def _data_input(value: object, label: str, *, schema_version: int) -> DataInput:
    if schema_version >= 4:
        if type(value) is not dict:
            raise ForecastCampaignError(f"{label} must be a mapping")
        body = _exact(value, {"source_uri"} | ({"sha256"} if "sha256" in value else set()), label)
        snapshot = None
        reference = body["source_uri"]
        reference_label = f"{label}.source_uri"
        expected_sha256 = body.get("sha256")
        if expected_sha256 is not None and not _is_sha256(expected_sha256):
            raise ForecastCampaignError(f"{label}.sha256 must be a lowercase SHA-256")
    else:
        body = _exact(value, {"snapshot_id", "reference"}, label)
        snapshot = _identifier(body["snapshot_id"], f"{label}.snapshot_id")
        reference = body["reference"]
        reference_label = f"{label}.reference"
        expected_sha256 = None
    try:
        stable_reference = validate_data_reference(reference)
    except ProvenanceError as error:
        raise ForecastCampaignError(f"{reference_label} is unsafe: {error}") from error
    return DataInput(snapshot, stable_reference, cast(str | None, expected_sha256))


def _request_path(value: object, label: str) -> pathlib.Path:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ForecastCampaignError(f"{label} must be a non-blank path string")
    return pathlib.Path(value)


def _parameter_source(value: object, label: str, *, schema_version: int) -> ParameterSourceRequest:
    if type(value) is not dict:
        raise ForecastCampaignError(f"{label} must be a mapping")
    field = "mode" if schema_version >= 4 else "source"
    mode = value.get(field)
    if mode == "defaults":
        _exact(value, {field}, label)
        return ParameterSourceRequest("defaults")
    if mode == "manual":
        keys = {field, "values"} if schema_version >= 4 else {field, "base", "values"}
        body = _exact(value, keys, label)
        if schema_version < 4 and body["base"] != "profile_defaults":
            raise ForecastCampaignError(f"{label} manual source requires base: profile_defaults")
        if type(body["values"]) is not dict:
            raise ForecastCampaignError(f"{label} manual mode requires a values mapping")
        try:
            _json(body["values"])
        except ForecastCampaignError as error:
            raise ForecastCampaignError(f"{label}.values is invalid: {error}") from error
        return ParameterSourceRequest(
            "manual",
            base="profile_defaults",
            values=cast(dict[str, object], body["values"]),
        )
    if mode == "tune":
        body = _exact(value, {field, "study_config"}, label)
        return ParameterSourceRequest(
            "tune", study_config=_request_path(body["study_config"], f"{label}.study_config")
        )
    reuse_mode = "reuse_tuned" if schema_version >= 4 else "reuse_selection"
    if mode == reuse_mode:
        body = _exact(value, {field, "study_config", "selection"}, label)
        return ParameterSourceRequest(
            "reuse_selection",
            study_config=_request_path(body["study_config"], f"{label}.study_config"),
            selection=_request_path(body["selection"], f"{label}.selection"),
        )
    raise ForecastCampaignError(f"{label}.{field} must be defaults, manual, tune, or {reuse_mode}")


def _budget(value: object, label: str) -> TrainingBudget:
    body = _exact(
        value,
        {
            "maximum_epochs",
            "patience",
            "timeout_seconds",
            "max_attempts",
            "early_stopping_absolute_min_delta",
            "early_stopping_relative_min_delta",
        },
        label,
    )
    return TrainingBudget(
        _positive_int(body["maximum_epochs"], f"{label}.maximum_epochs"),
        _positive_int(body["patience"], f"{label}.patience"),
        _positive_int(body["timeout_seconds"], f"{label}.timeout_seconds", minimum=2),
        _positive_int(body["max_attempts"], f"{label}.max_attempts"),
        _nonnegative_number(
            body["early_stopping_absolute_min_delta"],
            f"{label}.early_stopping_absolute_min_delta",
        ),
        _nonnegative_number(
            body["early_stopping_relative_min_delta"],
            f"{label}.early_stopping_relative_min_delta",
        ),
    )


def load_forecast_campaign_request(path: str | pathlib.Path) -> ForecastCampaignRequest:
    """Parse one exact, model/data-agnostic forecasting request."""
    source = _path_without_links(path, "request")
    value, raw = _yaml_load(source)
    top = _exact(value, {"schema_version", "forecast_campaign"}, "request")
    schema_version = top["schema_version"]
    if type(schema_version) is not int or schema_version not in {1, 2, 3, 4, 5, 6}:
        raise ForecastCampaignError(
            "forecast campaign schema_version must be integer 1, 2, 3, 4, 5 or 6"
        )
    body_keys = {
        "name",
        "dataset",
        "models",
        "temporal",
        "seeds",
        "initialization_arms",
        "selection",
        "final_evidence_role",
        "budgets",
        "compute",
        "tracking",
    }
    if schema_version >= 5:
        body_keys.add("purpose")
    body = _exact(
        top["forecast_campaign"],
        body_keys,
        "forecast_campaign",
    )
    raw_dataset = body["dataset"]
    if type(raw_dataset) is not dict:
        raise ForecastCampaignError("forecast_campaign.dataset must be a mapping")
    dataset_keys = {"contract", "selection", "final_training", "final_score"}
    if schema_version < 4 or "run_template" in raw_dataset:
        dataset_keys.add("run_template")
    dataset = _exact(raw_dataset, dataset_keys, "forecast_campaign.dataset")
    model_keys = {"catalog", "ids"} if schema_version == 1 else {"catalog", "candidates"}
    models = _exact(body["models"], model_keys, "forecast_campaign.models")
    temporal = _exact(
        body["temporal"],
        {
            "training_start",
            "initialization_origin",
            "validation_origins",
            "validation_horizon",
            "final_train_cutoff",
            "final_test_horizon",
        },
        "forecast_campaign.temporal",
    )
    selection = _exact(
        body["selection"],
        {"metric", "missing_actual", "direction", "tie_break"},
        "forecast_campaign.selection",
    )
    if type(selection["metric"]) is not str or selection["metric"] not in {
        "rmse",
        "mae",
        "wape",
        "wrmsse",
    }:
        raise ForecastCampaignError("selection.metric is unsupported")
    if type(selection["missing_actual"]) is not str or selection["missing_actual"] not in {
        "sparse_zero",
        "no_data",
    }:
        raise ForecastCampaignError("selection.missing_actual is unsupported")
    if selection["direction"] != "minimize":
        raise ForecastCampaignError("forecast error metrics must use selection.direction=minimize")
    tie_break = selection["tie_break"]
    expected_tie_break = _TIE_BREAK if schema_version >= 3 else _LEGACY_TIE_BREAK
    if type(tie_break) is not list or tuple(tie_break) != expected_tie_break:
        raise ForecastCampaignError(
            f"selection.tie_break must be exactly {list(expected_tie_break)}"
        )
    if schema_version == 1:
        ids = models["ids"]
        if type(ids) is not list or not ids:
            raise ForecastCampaignError("models.ids must be a non-empty list")
        model_ids = tuple(_identifier(item, "models.ids item") for item in ids)
        variant_ids = tuple("default" for _ in model_ids)
        parameter_sources = tuple(ParameterSourceRequest("defaults") for _ in model_ids)
    else:
        candidates = models["candidates"]
        if type(candidates) is not list or not candidates:
            raise ForecastCampaignError("models.candidates must be a non-empty list")
        if schema_version >= 4:
            candidate_keys = {"model_id", "parameters"}
        elif schema_version == 3:
            candidate_keys = {"model_id", "variant_id", "parameters"}
        else:
            candidate_keys = {"id", "parameters"}
        parsed_candidates = [
            _exact(item, candidate_keys, f"models.candidates[{index}]")
            for index, item in enumerate(candidates)
        ]
        model_ids = tuple(
            _identifier(
                item["model_id"] if schema_version >= 3 else item["id"],
                f"models.candidates[{index}].model_id"
                if schema_version >= 3
                else f"models.candidates[{index}].id",
            )
            for index, item in enumerate(parsed_candidates)
        )
        variant_ids = tuple(
            (
                _identifier(item["variant_id"], f"models.candidates[{index}].variant_id")
                if schema_version == 3
                else None
                if schema_version >= 4
                else "default"
            )
            for index, item in enumerate(parsed_candidates)
        )
        parameter_sources = tuple(
            _parameter_source(
                item["parameters"],
                f"models.candidates[{index}].parameters",
                schema_version=schema_version,
            )
            for index, item in enumerate(parsed_candidates)
        )
    candidate_ids = tuple(
        None if variant_id is None else _candidate_id(model_id, variant_id)
        for model_id, variant_id in zip(model_ids, variant_ids, strict=True)
    )
    if schema_version < 4 and len(set(candidate_ids)) != len(candidate_ids):
        raise ForecastCampaignError(
            "models must not repeat the same (model_id, variant_id) candidate"
        )
    raw_origins = temporal["validation_origins"]
    if type(raw_origins) is not list or not raw_origins:
        raise ForecastCampaignError("temporal.validation_origins must be a non-empty list")
    origins = tuple(_date(item, "validation origin") for item in raw_origins)
    if tuple(sorted(set(origins))) != origins:
        raise ForecastCampaignError("validation origins must be unique and strictly increasing")
    training_start = (
        None
        if temporal["training_start"] is None
        else _date(temporal["training_start"], "temporal.training_start")
    )
    initialization_origin = (
        None
        if temporal["initialization_origin"] is None
        else _date(temporal["initialization_origin"], "temporal.initialization_origin")
    )
    validation_horizon = _positive_int(
        temporal["validation_horizon"], "temporal.validation_horizon"
    )
    final_cutoff = _date(temporal["final_train_cutoff"], "temporal.final_train_cutoff")
    final_horizon = _positive_int(temporal["final_test_horizon"], "temporal.final_test_horizon")
    if training_start is not None and training_start >= origins[0]:
        raise ForecastCampaignError("training_start must precede every validation origin")
    if initialization_origin is not None and not (
        (training_start is None or training_start <= initialization_origin)
        and initialization_origin < origins[0]
    ):
        raise ForecastCampaignError(
            "initialization_origin must follow start and precede validation"
        )
    if any(origin + dt.timedelta(days=validation_horizon) > final_cutoff for origin in origins):
        raise ForecastCampaignError("validation windows must end on or before final_train_cutoff")
    seeds = body["seeds"]
    if (
        type(seeds) is not list
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ForecastCampaignError("seeds must be unique non-negative integers")
    arms = body["initialization_arms"]
    if (
        type(arms) is not list
        or not arms
        or any(type(arm) is not str or arm not in _ARMS for arm in arms)
        or len(set(arms)) != len(arms)
        or "fresh" not in arms
    ):
        raise ForecastCampaignError(
            "initialization_arms must be unique supported arms and include fresh"
        )
    budgets = _exact(body["budgets"], {"initialization", "validation", "final_fit"}, "budgets")
    compute_keys = {
        "device", "batch_size", "gradient_microbatch_size", "memory_gib", "min_free_disk_gib"
    }
    raw_compute = body["compute"]
    if schema_version >= 6 and type(raw_compute) is dict and "precision" in raw_compute:
        compute_keys.add("precision")
    compute = _exact(raw_compute, compute_keys, "compute")
    precision = compute.get("precision", "16-mixed" if schema_version >= 6 else "32-true")
    if precision not in ("16-mixed", "32-true"):
        raise ForecastCampaignError("compute.precision must be 16-mixed or 32-true")
    if type(compute["device"]) is not str or compute["device"] not in {
        "cpu",
        "cuda",
        "mps",
    }:
        raise ForecastCampaignError("compute.device is unsupported")
    microbatch = compute["gradient_microbatch_size"]
    if microbatch is not None:
        microbatch = _positive_int(microbatch, "compute.gradient_microbatch_size")
    batch_size = _positive_int(compute["batch_size"], "compute.batch_size")
    if microbatch is not None and microbatch > batch_size:
        raise ForecastCampaignError("gradient microbatch cannot exceed logical batch size")
    tracking = _exact(body["tracking"], {"mlflow"}, "tracking")
    if type(tracking["mlflow"]) is not bool:
        raise ForecastCampaignError("tracking.mlflow must be boolean")
    evidence_role = _identifier(body["final_evidence_role"], "final_evidence_role")
    purpose = "forecast_campaign" if schema_version < 5 else body["purpose"]
    if type(purpose) is not str or purpose not in {
        "forecast_campaign",
        "single_model_benchmark",
        "research_campaign",
        "capacity_diagnostic",
    }:
        raise ForecastCampaignError("forecast_campaign.purpose is unsupported")
    if purpose == "single_model_benchmark":
        if len(model_ids) != 1:
            raise ForecastCampaignError(
                "single_model_benchmark requires exactly one model candidate"
            )
        if len(seeds) != 1:
            raise ForecastCampaignError("single_model_benchmark requires exactly one seed")
        if len(origins) != 1:
            raise ForecastCampaignError(
                "single_model_benchmark requires exactly one validation origin"
            )
        if tuple(arms) != ("fresh",):
            raise ForecastCampaignError(
                "single_model_benchmark requires initialization_arms=[fresh]"
            )
    if purpose == "research_campaign":
        if len(model_ids) != 1:
            raise ForecastCampaignError("research_campaign requires exactly one model candidate")
        if len(seeds) != 3:
            raise ForecastCampaignError("research_campaign requires exactly three seeds")
        if len(origins) < 2:
            raise ForecastCampaignError(
                "research_campaign requires at least two validation origins"
            )
        if tuple(arms) != ("fresh",):
            raise ForecastCampaignError("research_campaign requires initialization_arms=[fresh]")
    return ForecastCampaignRequest(
        schema_version,
        _identifier(body["name"], "forecast_campaign.name"),
        cast(CampaignPurpose, purpose),
        _request_path(dataset["contract"], "forecast_campaign.dataset.contract"),
        (
            None
            if "run_template" not in dataset
            else _request_path(dataset["run_template"], "forecast_campaign.dataset.run_template")
        ),
        _data_input(dataset["selection"], "dataset.selection", schema_version=schema_version),
        _data_input(
            dataset["final_training"],
            "dataset.final_training",
            schema_version=schema_version,
        ),
        _data_input(dataset["final_score"], "dataset.final_score", schema_version=schema_version),
        _request_path(models["catalog"], "forecast_campaign.models.catalog"),
        model_ids,
        variant_ids,
        candidate_ids,
        parameter_sources,
        training_start,
        initialization_origin,
        origins,
        validation_horizon,
        final_cutoff,
        final_horizon,
        tuple(cast(list[int], seeds)),
        tuple(cast(list[InitializationArm], arms)),
        cast(Literal["rmse", "mae", "wape", "wrmsse"], selection["metric"]),
        cast(Literal["sparse_zero", "no_data"], selection["missing_actual"]),
        evidence_role,
        _budget(budgets["initialization"], "budgets.initialization"),
        _budget(budgets["validation"], "budgets.validation"),
        _budget(budgets["final_fit"], "budgets.final_fit"),
        ComputeBudget(
            cast(Literal["cpu", "cuda", "mps"], compute["device"]),
            batch_size,
            cast(int | None, microbatch),
            _positive_memory(compute["memory_gib"]),
            _nonnegative_number(compute["min_free_disk_gib"], "compute.min_free_disk_gib"),
            cast(Literal["32-true", "16-mixed"], precision),
        ),
        cast(bool, tracking["mlflow"]),
        source,
        raw,
    )


def initialization_capability(candidate: Candidate, arm: InitializationArm) -> dict[str, object]:
    """Declare campaign-level initialization support without silently falling back."""
    if arm == "fresh":
        return {"status": "supported", "reason": "adapter supports a new governed run"}
    if arm == "weights_only_finetune" and candidate.adapter == "mft":
        return {
            "status": "supported",
            "reason": (
                "generic runner accepts a hash-verified inference checkpoint and resets "
                "the optimizer"
            ),
        }
    if arm == "full_state_continuation":
        return {
            "status": "not_compatible",
            "reason": (
                "generic campaign runner has no prospective full-state continuation API; "
                "same-run fault recovery is a different lifecycle"
            ),
        }
    return {
        "status": "not_compatible",
        "reason": f"adapter {candidate.adapter} exposes no weights-only continuation input",
    }


def _positive_memory(value: object) -> float:
    memory = _nonnegative_number(value, "compute.memory_gib")
    if memory == 0:
        raise ForecastCampaignError("compute.memory_gib must be strictly positive")
    return memory


def _source_path(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = _path_without_links(_root() / path if not path.is_absolute() else path, label)
    resolved = lexical.resolve()
    if not resolved.is_relative_to(_root().resolve()) or not resolved.is_file():
        raise ForecastCampaignError(f"{label} must be a regular source file below {_root()}")
    return resolved


def _data_identity(
    value: DataInput, runtime_path: str | pathlib.Path, label: str
) -> dict[str, object]:
    path = _path_without_links(runtime_path, label).resolve()
    if not (path.is_file() or path.is_dir()):
        raise ForecastCampaignError(f"{label} must be a regular Parquet file or directory")
    try:
        digest = directory_digest(path, suffixes=(".parquet",))
    except Exception as error:
        raise ForecastCampaignError(f"cannot fingerprint {label}: {error}") from error
    if value.expected_sha256 is not None and digest != value.expected_sha256:
        raise ForecastCampaignError(f"{label} differs from its pinned SHA-256")
    role = label.removesuffix(" data").replace(" ", "-")
    return {
        "path": str(path),
        "snapshot_id": value.snapshot_id or f"{role}-{digest[:16]}",
        "reference": value.reference,
        "sha256": digest,
    }


def _git_source_identity(request_path: pathlib.Path) -> dict[str, object]:
    if installed_mode():
        if not request_path.is_relative_to(_root()):
            raise ForecastCampaignError("campaign request must be below the experiment project")
        return {"git_commit": None, "request_path": request_path.relative_to(_root()).as_posix(),
                "library": installed_library_identity().as_record()}
    if not request_path.is_relative_to(_root()):
        raise ForecastCampaignError("forecast campaign request must be committed source input")
    relative = request_path.relative_to(_root()).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ForecastCampaignError("forecast campaign request must be committed source input")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise ForecastCampaignError("source checkout must be clean before campaign compilation")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_root(), capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"git_commit": commit, "request_path": relative}


def _write_immutable_set(directory: pathlib.Path, payloads: dict[str, bytes]) -> None:
    directory = _path_without_links(directory, "compiled artifact directory")
    try:
        publish_bytes_set(directory, payloads)
    except SafeFileError as error:
        raise ForecastCampaignError(
            f"could not publish immutable compiled artifacts: {directory}"
        ) from error


def _child_stage_id(model_index: int, arm: str, seed_index: int, origin_index: int) -> str:
    arm_code = {"fresh": "f", "weights_only_finetune": "w", "full_state_continuation": "s"}[arm]
    return f"v{model_index}{arm_code}s{seed_index}o{origin_index}"


def _parent_stage_id(model_index: int, seed_index: int) -> str:
    return f"p{model_index:02d}w{seed_index:02d}"


def _domain_command(
    action: str, resolved: pathlib.Path, resolved_sha: str, *args: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "autoforecast.forecast_campaign",
        action,
        "--resolved",
        str(resolved),
        "--resolved-sha256",
        resolved_sha,
        *args,
        *project_arguments(),
    ]


def _stage(
    stage_id: str,
    command: list[str],
    *,
    needs: list[str],
    timeout: int,
    attempts: int,
    outputs: list[str],
    resources_record: dict[str, object],
) -> dict[str, object]:
    return {
        "id": stage_id,
        "needs": needs,
        "command": command,
        "timeout_seconds": timeout,
        "max_attempts": attempts,
        "retry_mode": "restart",
        "required_outputs": outputs,
        "resources": resources_record,
    }


def _campaign_hpo_study(
    source: ParameterSourceRequest,
    candidate: Candidate,
    *,
    selection_data: dict[str, object],
    validation_origins: tuple[dt.date, ...],
    step_days: int,
    require_data_id: bool,
) -> HPOStudySpec:
    """Load one inner-validation study and prove it cannot consume outer/test evidence."""
    if source.study_config is None:
        raise ForecastCampaignError("HPO parameter mode requires a study config")
    study_path = _source_path(source.study_config, "HPO study config")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", study_path.relative_to(_root()).as_posix()],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ForecastCampaignError("HPO study config must be committed source input")
    try:
        study = load_hpo_study(study_path, project_root=_root())
    except (HPOError, ValueError) as error:
        raise ForecastCampaignError(f"invalid campaign HPO study: {error}") from error
    if study.evaluator != "builtin":
        raise ForecastCampaignError("campaign HPO requires study.evaluator: builtin")
    if study.direction != "minimize":
        raise ForecastCampaignError("forecast campaign HPO must minimize validation error")
    if (
        study.tuning.model_id != candidate.id
        or study.tuning.adapter != candidate.adapter
        or study.tuning.source_sha256 != candidate.source_sha256
    ):
        raise ForecastCampaignError("HPO study model profile differs from its campaign model")
    if study.data.get("fingerprint") != selection_data["sha256"] or (
        require_data_id and study.data.get("id") != selection_data["snapshot_id"]
    ):
        raise ForecastCampaignError("HPO study data identity differs from campaign selection data")
    target = study.data.get("validation_target_date_range")
    if type(target) is not list or len(target) != 2:
        raise ForecastCampaignError("HPO validation target range is invalid")
    start, end = (_date(value, "HPO validation target") for value in target)
    span = (end - start).days
    if span < 0 or span % step_days:
        raise ForecastCampaignError("HPO validation target range does not match dataset cadence")
    # HPO is an inner selection problem. Its last target must precede the first outer target,
    # which starts one data period after the first campaign validation origin.
    if end > validation_origins[0]:
        raise ForecastCampaignError(
            "HPO validation targets must end on or before the first outer validation origin"
        )
    return study


def _study_record(study: HPOStudySpec) -> dict[str, object]:
    return {
        "id": study.id,
        "path": study.source_path.relative_to(_root()).as_posix(),
        "sha256": study.source_sha256,
        "max_trials": study.max_trials,
    }


def _external_parameter_mode(mode: ParameterMode) -> str:
    return "reuse_tuned" if mode == "reuse_selection" else mode


def _variant_spec_identity(
    candidate: Candidate,
    mode: ParameterMode,
    resolution: dict[str, object] | None,
    study: dict[str, object] | None,
    expected_protocol: dict[str, object],
) -> tuple[str, str]:
    """Identify the frozen candidate intent without requiring a user-authored label."""
    if mode == "tune":
        assert study is not None and resolution is None
        parameters: dict[str, object] = {
            "study_sha256": study["sha256"],
            "protocol": expected_protocol,
        }
    elif mode == "reuse_selection":
        assert study is not None and resolution is not None
        source = cast(dict[str, object], resolution["source"])
        parameters = {
            "study_sha256": study["sha256"],
            "selection_sha256": source["selection_sha256"],
            "selected_trial_id": source["selected_trial_id"],
            "selected_attempt_id": source["selected_attempt_id"],
            "selected_attempt_manifest_sha256": source["selected_attempt_manifest_sha256"],
            "selected_source_commit": source["selected_source_commit"],
            "protocol": source["protocol"],
            "effective_parameters": resolution["effective_parameters"],
        }
    else:
        assert study is None and resolution is not None
        parameters = {
            "effective_parameters": resolution["effective_parameters"],
            **(
                {"protocol": {"precision": expected_protocol["precision"]}}
                if "precision" in expected_protocol else {}
            ),
        }
    record = {
        "schema": "autoforecast.variant-spec.v1",
        "model_id": candidate.id,
        "model_profile_sha256": candidate.source_sha256,
        "implementation_config_sha256": candidate.implementation_config_sha256,
        "mode": _external_parameter_mode(mode),
        "parameters": parameters,
    }
    digest = _sha_bytes(_json(record).encode())
    return f"{_external_parameter_mode(mode)}-{digest[:16]}", digest


def _preflight_lines(
    profiles: list[dict[str, object]],
    data: dict[str, dict[str, object]],
) -> tuple[str, ...]:
    messages = [
        (
            f"data.{role}: source_uri={record['reference']} "
            f"snapshot_id={record['snapshot_id']} content_sha256={record['sha256']}"
        )
        for role, record in data.items()
    ]
    descriptions = {
        "defaults": "HPO off; using catalog profile defaults",
        "manual": "HPO off; using explicit values over catalog profile defaults",
        "tune": "HPO on; tuning with the frozen study",
        "reuse_selection": "HPO off; reusing a verified tuned selection",
    }
    messages.extend(
        (
            f"model={profile['model_id']} variant_id={profile['variant_id']} "
            f"parameter_mode={_external_parameter_mode(cast(ParameterMode, profile['mode']))} "
            f"({descriptions[cast(str, profile['mode'])]})"
        )
        for profile in profiles
    )
    return tuple(messages)


def _parameter_profiles(
    request: ForecastCampaignRequest,
    candidates: dict[str, Candidate],
    selection_data: dict[str, object],
    *,
    step_days: int,
    dataset_contract_sha256: str,
    features: dict[str, object] | None,
) -> list[dict[str, object]]:
    expected_protocol = {
        "metric": request.metric,
        "missing_actual": request.missing_actual,
        "selection_data_fingerprint": selection_data["sha256"],
        "dataset_contract_sha256": dataset_contract_sha256,
        "features": features,
        "training_seed": 0,
        "maximum_epochs": request.validation_budget.maximum_epochs,
        "patience": request.validation_budget.patience,
        "device": request.compute.device,
        "batch_size": request.compute.batch_size,
        "gradient_microbatch_size": request.compute.gradient_microbatch_size,
        "early_stopping_absolute_min_delta": request.validation_budget.absolute_min_delta,
        "early_stopping_relative_min_delta": request.validation_budget.relative_min_delta,
    }
    profiles: list[dict[str, object]] = []
    for index, (declared_candidate_id, model_id, declared_variant_id, source) in enumerate(
        zip(
            request.candidate_ids,
            request.model_ids,
            request.variant_ids,
            request.parameter_sources,
            strict=True,
        )
    ):
        candidate = candidates[model_id]
        candidate_protocol = {
            **expected_protocol,
            **(
                {"precision": effective_precision(candidate.adapter, request.compute.precision)}
                if request.schema_version >= 6 else {}
            ),
        }
        study: HPOStudySpec | None = None
        study_record: dict[str, object] | None = None
        tune_stage: str | None = None
        try:
            if source.mode == "defaults":
                resolution = default_parameter_resolution(candidate)
            elif source.mode == "manual":
                resolution = manual_parameter_resolution(
                    candidate, cast(dict[str, object], source.values)
                )
            else:
                study = _campaign_hpo_study(
                    source,
                    candidate,
                    selection_data=selection_data,
                    validation_origins=request.validation_origins,
                    step_days=step_days,
                    require_data_id=request.schema_version < 4,
                )
                study_record = _study_record(study)
                if source.mode == "tune":
                    resolution = None
                    tune_stage = f"t{index:02d}"
                else:
                    selection = cast(pathlib.Path, source.selection)
                    selection_path = _path_without_links(
                        _root() / selection if not selection.is_absolute() else selection,
                        "reused HPO selection",
                    )
                    resolution = parameter_resolution_from_selection(
                        candidate,
                        study,
                        selection_path,
                        kind="reuse_selection",
                        project_root=_root(),
                        expected_protocol=candidate_protocol,
                    )
        except ParameterSourceError as error:
            raise ForecastCampaignError(
                f"parameter source for {model_id!r} is invalid: {error}"
            ) from error
        generated_variant_id, variant_spec_sha256 = _variant_spec_identity(
            candidate,
            source.mode,
            resolution,
            study_record,
            candidate_protocol,
        )
        variant_id = declared_variant_id or generated_variant_id
        candidate_id = declared_candidate_id or _candidate_id(model_id, variant_id)
        profiles.append(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "variant_spec_sha256": variant_spec_sha256,
                "mode": source.mode,
                "study": study_record,
                "tune_stage": tune_stage,
                "resolution": resolution,
            }
        )
    candidate_ids = [cast(str, profile["candidate_id"]) for profile in profiles]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ForecastCampaignError(
            "models must not repeat the same resolved model and parameter specification"
        )
    return profiles


def _require_separate_long_revision_runs(
    adapter_names: tuple[str, ...], epoch_ceilings: tuple[int, ...]
) -> None:
    """Keep full SFT and HMFNet runs sequential and in distinct immutable campaign roots."""
    if (
        {"sft", "hmfnet"}.issubset(adapter_names)
        and any(ceiling > 2 for ceiling in epoch_ceilings)
    ):
        raise ForecastCampaignError(
            "long SFT and HMFNet campaigns require separate requests and output roots"
        )


def compile_forecast_campaign(
    request_path: str | pathlib.Path,
    *,
    selection_data_path: str | pathlib.Path,
    final_training_data_path: str | pathlib.Path,
    final_score_data_path: str | pathlib.Path,
    compiled_root: str | pathlib.Path | None = None,
    benchmark_path: pathlib.Path | None = None,
) -> CompiledForecastCampaign:
    """Resolve and freeze a high-level request into idempotent durable stage commands."""
    request = load_forecast_campaign_request(request_path)
    contract = _source_path(request.contract, "dataset contract")
    template_path = (
        None if request.template is None else _source_path(request.template, "run template")
    )
    catalog_lexical = _path_without_links(
        _root() / request.catalog if not request.catalog.is_absolute() else request.catalog,
        "model catalog",
    )
    catalog = catalog_lexical.resolve()
    if not catalog.is_relative_to(_root().resolve()) or not catalog.is_dir():
        raise ForecastCampaignError("model catalog must be a source directory below the repository")
    if (
        is_m5_daily(contract)
        and request.final_train_cutoff == dt.date(2016, 5, 22)
        and request.final_test_horizon == 28
        and request.evidence_role != "archived_private_replay_not_blind"
    ):
        raise ForecastCampaignError(
            "the revealed M5 private window must use final_evidence_role="
            "archived_private_replay_not_blind"
        )
    try:
        contract_definition = load_dataset_contract(contract)
        selection_panel = contract_definition.panel(selection_data_path)
        step_days = period_days(selection_panel.frequency)
    except Exception as error:
        raise ForecastCampaignError(
            f"dataset contract cannot define campaign cadence: {error}"
        ) from error
    if any(
        origin + dt.timedelta(days=request.validation_horizon * step_days)
        > request.final_train_cutoff
        for origin in request.validation_origins
    ):
        raise ForecastCampaignError(
            "validation windows must end on or before final_train_cutoff at dataset frequency"
        )
    if template_path is None:
        calendar_events: tuple[tuple[object, ...], ...] = ()
        event_max_support = 30
    else:
        template = load_run_request(template_path)
        if (_root() / template.dataset_contract).resolve() != contract:
            raise ForecastCampaignError("run template and campaign dataset contracts differ")
        if (_root() / template.model_catalog).resolve() != catalog:
            raise ForecastCampaignError("run template and campaign model catalogs differ")
        calendar_events = cast(tuple[tuple[object, ...], ...], template.calendar_events)
        event_max_support = template.event_max_support
    candidates = {candidate.id: candidate for candidate in discover_candidates(catalog)}
    unknown = sorted(set(request.model_ids) - set(candidates))
    if unknown:
        raise ForecastCampaignError(f"campaign model IDs are absent from catalog: {unknown}")
    if (
        request.purpose == "single_model_benchmark"
        and any(
            candidates[model_id].adapter in _FIXED_BUDGET_ADAPTERS
            for model_id in request.model_ids
        )
        and request.metric not in {"rmse", "wrmsse"}
    ):
        raise ForecastCampaignError(
            "fixed-budget benchmark selection requires RMSE or WRMSSE"
        )
    _require_separate_long_revision_runs(
        tuple(candidates[model_id].adapter for model_id in request.model_ids),
        tuple(
            budget.maximum_epochs for budget in (
                request.initialization_budget,
                request.validation_budget,
                request.final_fit_budget,
            )
        ),
    )
    source = _git_source_identity(request.source_path)
    if (
        request.compute.precision == "16-mixed"
        and "weights_only_finetune" in request.initialization_arms
        and any(
            candidates[model_id].adapter == "mft"
            for model_id in request.model_ids
        )
    ):
        raise ForecastCampaignError(
            "MFT weights-only continuation requires compute.precision: 32-true; "
            "use a fresh-only campaign for mixed precision"
        )
    if any(candidates[model_id].adapter in {"tft_q", "tft_hnb"} for model_id in request.model_ids):
        if len(request.seeds) != 1 or len(request.validation_origins) != 1:
            raise ForecastCampaignError(
                "TFT-Q/TFT-HNB checkpoint selection/refit requires one seed "
                "and one validation origin"
            )
        if (
            contract != (_root() / "configs/datasets/m5_daily.yaml").resolve()
            or request.metric != "wrmsse"
            or request.validation_horizon != 28
            or request.final_test_horizon != 28
            or any(
                budget.patience != 10
                for budget in (
                    request.initialization_budget,
                    request.validation_budget,
                    request.final_fit_budget,
                )
            )
            or any(
                budget.relative_min_delta != 0
                for budget in (
                    request.initialization_budget,
                    request.validation_budget,
                    request.final_fit_budget,
                )
            )
            or any(
                source.expected_sha256 is None
                for source in (
                    request.selection_data,
                    request.final_training_data,
                    request.final_score_data,
                )
            )
        ):
            raise ForecastCampaignError(
                "TFT-Q/TFT-HNB M5 requires the M5 daily contract, WRMSSE, 28-day horizons, "
                "patience 10, zero relative min-delta, and pinned data-role digests"
            )
    if request.metric == "wrmsse" and selection_panel.price_col is None:
        raise ForecastCampaignError(
            "WRMSSE selection requires a declared price column before training"
        )
    for model_id in dict.fromkeys(request.model_ids):
        candidate = candidates[model_id]
        if (request.purpose == "capacity_diagnostic"
                and candidate.adapter in {"tft_q", "tft_hnb", }):
            raise ForecastCampaignError(
                "TFT-Q/TFT-HNB capacity diagnostics require a validation window "
                "and are unsupported"
            )
        if request.metric == "wrmsse" and candidate.adapter in _ROLLING_BUDGET_ADAPTERS:
            try:
                recipe = loads_exact_json(candidate.implementation_config_bytes or b"")
            except ExactJSONError as error:
                raise ForecastCampaignError(
                    f"campaign model {model_id!r} has invalid hierarchy weighting controls"
                ) from error
            if type(recipe) is not dict or recipe.get("hierarchy_weighting") == "uniform":
                raise ForecastCampaignError(
                    f"campaign model {model_id!r} cannot select revenue-weighted WRMSSE "
                    "with uniform hierarchy weighting"
                )
        try:
            validate_execution_controls(
                candidate.adapter,
                device=request.compute.device,
                batch_size=request.compute.batch_size,
                gradient_microbatch_size=request.compute.gradient_microbatch_size,
                has_generic_events=bool(calendar_events),
                precision=request.compute.precision,
            )
        except ValueError as error:
            raise ForecastCampaignError(
                f"campaign controls are not executable by {model_id!r}: {error}"
            ) from error
        if candidate.adapter in {
            "sft",
            "hmfnet",
            "pact",
        }:
            try:
                default_parameter_resolution(candidate)
            except ParameterSourceError as error:
                raise ForecastCampaignError(
                    f"campaign model {model_id!r} has no valid baseline execution recipe: {error}"
                ) from error
    selection_data = _data_identity(request.selection_data, selection_data_path, "selection data")
    final_training = _data_identity(
        request.final_training_data, final_training_data_path, "final training data"
    )
    final_score = _data_identity(
        request.final_score_data, final_score_data_path, "final score data"
    )
    if (
        pathlib.Path(cast(str, final_score["path"]))
        in {
            pathlib.Path(cast(str, selection_data["path"])),
            pathlib.Path(cast(str, final_training["path"])),
        }
        or final_score["sha256"]
        in {
            selection_data["sha256"],
            final_training["sha256"],
        }
        or directory_content_digest(cast(str, final_score["path"]), suffixes=(".parquet",))
        in {
            directory_content_digest(cast(str, selection_data["path"]), suffixes=(".parquet",)),
            directory_content_digest(cast(str, final_training["path"]), suffixes=(".parquet",)),
        }
    ):
        raise ForecastCampaignError(
            "final score labels must use a separate, content-distinct input from selection and "
            "final training"
        )
    template_features: dict[str, object] | None = (
        None
        if not calendar_events
        else {
            "calendar_events": {
                "max_support": event_max_support,
                "events": [list(value) for value in calendar_events],
            }
        }
    )
    parameter_profiles = _parameter_profiles(
        request,
        candidates,
        selection_data,
        step_days=step_days,
        dataset_contract_sha256=_sha_file(contract),
        features=template_features,
    )
    capability: list[dict[str, object]] = []
    children: list[dict[str, object]] = []
    parent_ids: dict[tuple[str, int], str] = {}
    for model_index, profile in enumerate(parameter_profiles):
        candidate_id = cast(str, profile["candidate_id"])
        model_id = cast(str, profile["model_id"])
        variant_id = cast(str, profile["variant_id"])
        candidate = candidates[model_id]
        for arm in request.initialization_arms:
            report = initialization_capability(candidate, arm)
            capability.append(
                {
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "variant_id": variant_id,
                    "adapter": candidate.adapter,
                    "initialization": arm,
                    **report,
                }
            )
            if report["status"] != "supported":
                continue
            if arm == "weights_only_finetune":
                if request.initialization_origin is None:
                    raise ForecastCampaignError(
                        "a compatible weights_only_finetune arm requires initialization_origin"
                    )
                for seed_index, seed in enumerate(request.seeds):
                    parent_id = _parent_stage_id(model_index, seed_index)
                    parent_ids[(candidate_id, seed)] = parent_id
                    children.append(
                        {
                            "id": parent_id,
                            "kind": "initialization_parent",
                            "candidate_id": candidate_id,
                            "model_id": model_id,
                            "variant_id": variant_id,
                            "adapter": candidate.adapter,
                            "parameter_mode": profile["mode"],
                            "initialization": "fresh_parent",
                            "seed": seed,
                            "origin": request.initialization_origin.isoformat(),
                            "horizon": request.validation_horizon,
                            "parent_stage": None,
                            "budget": request.initialization_budget.as_record(),
                        }
                    )
            for seed_index, seed in enumerate(request.seeds):
                for origin_index, origin in enumerate(request.validation_origins):
                    children.append(
                        {
                            "id": _child_stage_id(model_index, arm, seed_index, origin_index),
                            "kind": "validation",
                            "candidate_id": candidate_id,
                            "model_id": model_id,
                            "variant_id": variant_id,
                            "adapter": candidate.adapter,
                            "parameter_mode": profile["mode"],
                            "initialization": arm,
                            "seed": seed,
                            "origin": origin.isoformat(),
                            "horizon": request.validation_horizon,
                            "parent_stage": parent_ids.get((candidate_id, seed))
                            if arm != "fresh"
                            else None,
                            "budget": request.validation_budget.as_record(),
                        }
                    )
    request_sha = _sha_bytes(request.source_bytes)
    resolved: dict[str, object] = {
        "schema": INSTALLED_RESOLVED_SCHEMA if installed_mode() else RESOLVED_SCHEMA,
        "request_schema_version": request.schema_version,
        "name": request.name,
        "purpose": request.purpose,
        "source": {
            **source,
            "request_sha256": request_sha,
            "dataset_contract": contract.relative_to(_root()).as_posix(),
            "dataset_contract_sha256": _sha_file(contract),
            "run_template": (
                None if template_path is None else template_path.relative_to(_root()).as_posix()
            ),
            "run_template_sha256": None if template_path is None else _sha_file(template_path),
            "model_catalog": catalog.relative_to(_root()).as_posix(),
            "model_profiles": [
                {
                    "model_id": model_id,
                    "path": candidates[model_id].source_path.relative_to(_root()).as_posix(),
                    "sha256": candidates[model_id].source_sha256,
                    "implementation_config": (
                        None
                        if candidates[model_id].implementation_config_path is None
                        else cast(
                            pathlib.Path,
                            candidates[model_id].implementation_config_path,
                        )
                        .relative_to(_root())
                        .as_posix()
                    ),
                    "implementation_config_sha256": candidates[
                        model_id
                    ].implementation_config_sha256,
                }
                for model_id in dict.fromkeys(request.model_ids)
            ],
        },
        "data": {"selection": selection_data, "final_training": final_training},
        "models": [
            {
                "candidate_id": profile["candidate_id"],
                "model_id": profile["model_id"],
                "variant_id": profile["variant_id"],
            }
            for profile in parameter_profiles
        ],
        "parameter_profiles": parameter_profiles,
        "temporal": {
            "training_start": None
            if request.training_start is None
            else request.training_start.isoformat(),
            "initialization_origin": (
                None
                if request.initialization_origin is None
                else request.initialization_origin.isoformat()
            ),
            "validation_origins": [value.isoformat() for value in request.validation_origins],
            "validation_horizon": request.validation_horizon,
            "final_train_cutoff": request.final_train_cutoff.isoformat(),
            "final_test_horizon": request.final_test_horizon,
            "step_days": step_days,
        },
        "seeds": list(request.seeds),
        "initialization_capabilities": capability,
        "children": children,
        "selection": {
            "metric": request.metric,
            "missing_actual": request.missing_actual,
            "direction": "minimize",
            "tie_break": list(_TIE_BREAK),
        },
        "budgets": {
            "initialization": request.initialization_budget.as_record(),
            "validation": request.validation_budget.as_record(),
            "final_fit": request.final_fit_budget.as_record(),
        },
        "compute": {
            "device": request.compute.device,
            "batch_size": request.compute.batch_size,
            "gradient_microbatch_size": request.compute.gradient_microbatch_size,
            "memory_gib": request.compute.memory_gib,
            "min_free_disk_gib": request.compute.min_free_disk_gib,
            **({"precision": request.compute.precision} if request.schema_version >= 6 else {}),
        },
        "tracking": {"mlflow": request.mlflow_enabled},
        "template_features": {
            "calendar_events": [list(value) for value in calendar_events],
            "event_max_support": event_max_support,
        },
    }
    if benchmark_path is not None:
        from autoforecast.benchmark import _evaluation_contract_from_record, verify_benchmark
        benchmark_sha256 = _sha_file(benchmark_path)
        benchmark_path, benchmark = verify_benchmark(benchmark_path, benchmark_sha256)
        recipe = benchmark["recipe"]
        if (recipe["validation_origins"] != cast(dict, resolved["temporal"])["validation_origins"]
            or recipe["validation_horizon"] != request.validation_horizon
            or recipe["final_train_cutoff"] != request.final_train_cutoff.isoformat()
            or recipe["final_test_horizon"] != request.final_test_horizon
            or recipe["training_start"] != cast(dict, resolved["temporal"])["training_start"]
            or contract != (benchmark_path.parent / "dataset.yaml").resolve()
            or any(record["sha256"] != benchmark["artifacts"][role] for role, record in (
                ("selection", selection_data), ("final_training", final_training),
                ("final_score", final_score),
            ))):
            raise ForecastCampaignError("campaign inputs differ from frozen benchmark")
        contracts = {}
        boundaries = [
            ("validation", o.isoformat(), request.validation_horizon)
            for o in request.validation_origins
        ] + [("test", request.final_train_cutoff.isoformat(), request.final_test_horizon)]
        for stage, origin, span in boundaries:
            value = _evaluation_contract_from_record(
                benchmark_path, benchmark, stage=stage, origin=origin, horizon=span,
                metric=request.metric
            )
            contracts[value["boundary"]] = value
        resolved["benchmark"] = {"id": benchmark["benchmark_id"], "path": str(benchmark_path),
                                 "sha256": benchmark_sha256, "evaluations": contracts}
    elif installed_mode():
        raise ForecastCampaignError("installed campaigns require a frozen benchmark")
    resolved_payload = _json(resolved).encode()
    resolved_sha = _sha_bytes(resolved_payload)
    score_input = {
        "schema": SCORE_INPUT_SCHEMA,
        "campaign_request_sha256": request_sha,
        "resolved_sha256": resolved_sha,
        "final_score_data": final_score,
        "metric": request.metric,
        "missing_actual": request.missing_actual,
        "evidence_role": request.evidence_role,
        "origin": request.final_train_cutoff.isoformat(),
        "horizon": request.final_test_horizon,
    }
    score_payload = _json(score_input).encode()
    score_sha = _sha_bytes(score_payload)
    artifact_set_sha = _sha_bytes(
        _json(
            {
                "request_sha256": request_sha,
                "resolved_sha256": resolved_sha,
                "score_input_sha256": score_sha,
            }
        ).encode()
    )
    target_root = _path_without_links(
        pathlib.Path(compiled_root)
        if compiled_root is not None
        else _root() / ".autoforecast" / "forecast-campaign-specs",
        "compiled campaign root",
    )
    directory = target_root / f"{request.name}-{artifact_set_sha[:16]}"
    try:
        _create_fresh_directory(directory)
    except SafeFileError as error:
        if "already exists" not in str(error):
            raise ForecastCampaignError(
                f"could not create compiled campaign directory: {directory}"
            ) from error
        checked_directory = _path_without_links(directory, "compiled campaign directory")
        if not checked_directory.is_dir():
            raise ForecastCampaignError(
                f"compiled campaign directory is invalid: {directory}"
            ) from error
    resolved_path = directory / "resolved.json"
    score_path = directory / "score-input.json"
    request_copy = directory / "request-source.yaml"
    plan_path = directory / "campaign-plan.yaml"
    if request.purpose == "capacity_diagnostic" and (
        len(children) != 1
        or children[0]["kind"] != "validation"
        or children[0]["parent_stage"] is not None
        or any(profile["tune_stage"] is not None for profile in parameter_profiles)
    ):
        raise ForecastCampaignError(
            "capacity diagnostic requires exactly one untuned root validation child"
        )
    stages: list[dict[str, object]] = []
    for profile in parameter_profiles:
        tune_stage = profile["tune_stage"]
        if tune_stage is None:
            continue
        study_record = cast(dict[str, object], profile["study"])
        timeout = (
            request.validation_budget.timeout_seconds * cast(int, study_record["max_trials"]) + 300
        )
        stages.append(
            _stage(
                cast(str, tune_stage),
                _domain_command(
                    "tune-parameters",
                    resolved_path,
                    resolved_sha,
                    "--model-id",
                    cast(str, profile["model_id"]),
                    "--variant-id",
                    cast(str, profile["variant_id"]),
                    "--campaign-root",
                    "{campaign_root}",
                    "--attempt-dir",
                    "{attempt_dir}",
                ),
                needs=[],
                timeout=timeout,
                # A fresh backend restart would reset max_trials and exceed the paid-compute
                # budget. Until backend-native resume is implemented, fail closed after one try.
                attempts=1,
                outputs=[
                    "parameter-resolution.json",
                    "hpo",
                    "hpo-training-panel",
                    "hpo-validation-panel",
                    "hpo-run-request.yaml",
                ],
                resources_record=request.compute.resources(),
            )
        )
    validation_ids: list[str] = []
    for child in children:
        child_id = cast(str, child["id"])
        budget = cast(dict[str, object], child["budget"])
        profile = next(
            item for item in parameter_profiles if item["candidate_id"] == child["candidate_id"]
        )
        needs = [] if profile["tune_stage"] is None else [cast(str, profile["tune_stage"])]
        if child["parent_stage"] is not None:
            needs.append(cast(str, child["parent_stage"]))
        required_outputs = [
            "child-result.json",
            "run",
            "training-panel",
            "child-request.yaml",
            "parameter-resolution.json",
        ]
        if child["kind"] == "validation" and request.purpose != "capacity_diagnostic":
            required_outputs.append("evaluation-panel")
        stages.append(
            _stage(
                child_id,
                _domain_command(
                    "run-child",
                    resolved_path,
                    resolved_sha,
                    "--child-id",
                    child_id,
                    "--campaign-root",
                    "{campaign_root}",
                    "--attempt-dir",
                    "{attempt_dir}",
                ),
                needs=needs,
                timeout=cast(int, budget["timeout_seconds"]),
                attempts=cast(int, budget["max_attempts"]),
                outputs=required_outputs,
                resources_record=request.compute.resources(),
            )
        )
        if child["kind"] == "validation" and request.purpose != "capacity_diagnostic":
            validation_ids.append(child_id)
    if request.purpose != "capacity_diagnostic":
        stages.append(
            _stage(
                "aggregate_selection",
                _domain_command(
                    "aggregate-selection",
                    resolved_path,
                    resolved_sha,
                    "--campaign-root",
                    "{campaign_root}",
                    "--attempt-dir",
                    "{attempt_dir}",
                ),
                needs=validation_ids,
                timeout=300,
                attempts=1,
                outputs=["selection-lock.json", "validation-summary.json"],
                resources_record={"device": "cpu", "memory_gib": 0.5, "min_free_disk_gib": 0.1},
            )
        )
        stages.append(
            _stage(
                "final_fit",
                _domain_command(
                    "run-final-fit",
                    resolved_path,
                    resolved_sha,
                    "--campaign-root",
                    "{campaign_root}",
                    "--attempt-dir",
                    "{attempt_dir}",
                ),
                needs=["aggregate_selection"],
                timeout=request.final_fit_budget.timeout_seconds,
                attempts=request.final_fit_budget.max_attempts,
                outputs=[
                    "child-result.json",
                    "run",
                    "training-panel",
                    "child-request.yaml",
                    "parameter-resolution.json",
                ],
                resources_record=request.compute.resources(),
            )
        )
        stages.append(
            _stage(
                "final_score",
                _domain_command(
                    "score-final",
                    resolved_path,
                    resolved_sha,
                    "--score-input",
                    str(score_path),
                    "--score-input-sha256",
                    score_sha,
                    "--campaign-root",
                    "{campaign_root}",
                    "--attempt-dir",
                    "{attempt_dir}",
                ),
                needs=["final_fit"],
                timeout=max(300, request.validation_budget.timeout_seconds),
                attempts=1,
                outputs=["final-score.json", "evaluation", "scoring-inputs"],
                resources_record={
                    "device": "cpu",
                    "memory_gib": request.compute.memory_gib,
                    "min_free_disk_gib": request.compute.min_free_disk_gib,
                },
            )
        )
        stages.append(
            _stage(
                "replay_report",
                _domain_command(
                    "replay-report",
                    resolved_path,
                    resolved_sha,
                    "--campaign-root",
                    "{campaign_root}",
                    "--attempt-dir",
                    "{attempt_dir}",
                ),
                needs=["final_score"],
                timeout=300,
                attempts=1,
                outputs=["campaign-report.json", "replay-verification.json", "report.html"],
                resources_record={"device": "cpu", "memory_gib": 0.5, "min_free_disk_gib": 0.1},
            )
        )
    try:
        import yaml

        plan_payload = yaml.safe_dump(
            {"schema_version": 1, "campaign": {"name": request.name, "stages": stages}},
            sort_keys=False,
        ).encode()
    except Exception as error:  # pragma: no cover - dependency already used by request parser
        raise ForecastCampaignError(f"cannot serialize campaign plan: {error}") from error
    _write_immutable_set(
        directory,
        {
            request_copy.name: request.source_bytes,
            resolved_path.name: resolved_payload,
            score_path.name: score_payload,
            plan_path.name: plan_payload,
        },
    )
    preflight = _preflight_lines(
        parameter_profiles,
        {
            "selection": selection_data,
            "final_training": final_training,
            "final_score": final_score,
        },
    )
    return CompiledForecastCampaign(directory, plan_path, resolved_path, score_path, preflight)


def _load_exact(
    path: pathlib.Path, expected_sha: str, schema: str | tuple[str, ...]
) -> dict[str, object]:
    if len(expected_sha) != 64 or any(char not in _SHA_CHARS for char in expected_sha):
        raise ForecastCampaignError("expected artifact SHA-256 is invalid")
    if path.is_symlink() or not path.is_file():
        raise ForecastCampaignError(f"immutable artifact identity differs: {path}")
    try:
        raw = path.read_bytes()
        value = loads_exact_json(raw)
    except (OSError, ExactJSONError) as error:
        raise ForecastCampaignError(f"invalid exact JSON in {path}: {error}") from error
    if _sha_bytes(raw) != expected_sha:
        raise ForecastCampaignError(f"immutable artifact identity differs: {path}")
    schemas = (schema,) if isinstance(schema, str) else schema
    if type(value) is not dict or value.get("schema") not in schemas:
        raise ForecastCampaignError(f"artifact schema differs: {path}")
    return cast(dict[str, object], value)


def _load_hashed_json(path: pathlib.Path, expected_sha: str, label: str) -> dict[str, object]:
    """Parse exactly the bytes whose digest was authenticated by the caller."""
    if not _is_sha256(expected_sha) or path.is_symlink() or not path.is_file():
        raise ForecastCampaignError(f"{label} identity differs: {path}")
    try:
        raw = path.read_bytes()
        value = loads_exact_json(raw)
    except (OSError, ExactJSONError) as error:
        raise ForecastCampaignError(f"invalid {label}: {error}") from error
    if _sha_bytes(raw) != expected_sha or type(value) is not dict:
        raise ForecastCampaignError(f"{label} identity differs: {path}")
    return cast(dict[str, object], value)


def _load_json_with_identity(path: pathlib.Path, label: str) -> tuple[dict[str, object], str]:
    """Read a newly-created JSON artifact once and return that exact byte identity."""
    if path.is_symlink() or not path.is_file():
        raise ForecastCampaignError(f"{label} is missing or linked: {path}")
    try:
        raw = path.read_bytes()
        value = loads_exact_json(raw)
    except (OSError, ExactJSONError) as error:
        raise ForecastCampaignError(f"invalid {label}: {error}") from error
    if type(value) is not dict:
        raise ForecastCampaignError(f"{label} must be a JSON object")
    return cast(dict[str, object], value), _sha_bytes(raw)


def _verify_resolved_source(resolved: dict[str, object]) -> None:
    """Refuse to execute a compiled campaign from changed code or model configuration."""
    source = _exact(
        resolved.get("source"),
        {
            "git_commit",
            "request_path",
            "request_sha256",
            "dataset_contract",
            "dataset_contract_sha256",
            "run_template",
            "run_template_sha256",
            "model_catalog",
            "model_profiles",
        } | ({"library"} if resolved.get("schema") == INSTALLED_RESOLVED_SCHEMA else set()),
        "resolved source",
    )
    if "library" in source:
        if not installed_mode() or source["library"] != installed_library_identity().as_record():
            raise ForecastCampaignError("installed campaign library changed after compilation")
        if source["git_commit"] is not None:
            raise ForecastCampaignError("installed campaign cannot claim a checkout commit")
    else:
        expected_commit = source["git_commit"]
        if (
            type(expected_commit) is not str
            or len(expected_commit) not in {40, 64}
            or any(char not in _SHA_CHARS for char in expected_commit)
        ):
            raise ForecastCampaignError("resolved source git_commit is invalid")
        try:
            current = capture_source_identity(_root())
        except ProvenanceError as error:
            raise ForecastCampaignError(f"cannot verify clean campaign source: {error}") from error
        if current.commit != expected_commit:
            raise ForecastCampaignError(
                "campaign source commit changed after compilation: "
                f"expected {expected_commit}, observed {current.commit}"
            )

    for path_key, sha_key, label in (
        ("request_path", "request_sha256", "campaign request"),
        ("dataset_contract", "dataset_contract_sha256", "dataset contract"),
        ("run_template", "run_template_sha256", "run template"),
    ):
        relative = source[path_key]
        expected_sha = source[sha_key]
        if label == "run template" and relative is None and expected_sha is None:
            continue
        if type(relative) is not str or type(expected_sha) is not str:
            raise ForecastCampaignError(f"resolved {label} identity is invalid")
        path = _source_path(pathlib.Path(relative), label)
        if not _is_sha256(expected_sha) or _sha_file(path) != expected_sha:
            raise ForecastCampaignError(f"resolved {label} bytes changed after compilation")

    catalog_value = source["model_catalog"]
    if type(catalog_value) is not str:
        raise ForecastCampaignError("resolved model catalog path is invalid")
    catalog_lexical = _path_without_links(_root() / catalog_value, "model catalog")
    catalog = catalog_lexical.resolve()
    if not catalog.is_relative_to(_root().resolve()) or not catalog.is_dir():
        raise ForecastCampaignError("resolved model catalog must remain below the repository")

    models = resolved.get("models")
    profiles = source["model_profiles"]
    if type(models) is not list or not models:
        raise ForecastCampaignError("resolved models are invalid")
    model_records: list[dict[str, object]] = []
    for index, raw_model in enumerate(models):
        model = _exact(
            raw_model,
            {"candidate_id", "model_id", "variant_id"},
            f"resolved model {index}",
        )
        candidate_id, model_id, variant_id = (
            model["candidate_id"],
            model["model_id"],
            model["variant_id"],
        )
        if (
            type(candidate_id) is not str
            or type(model_id) is not str
            or type(variant_id) is not str
            or candidate_id != _candidate_id(model_id, variant_id)
        ):
            raise ForecastCampaignError(f"resolved model {index} identity is invalid")
        model_records.append(model)
    if len({cast(str, item["candidate_id"]) for item in model_records}) != len(model_records):
        raise ForecastCampaignError("resolved candidate identities are not unique")
    expected_model_ids = list(dict.fromkeys(cast(str, item["model_id"]) for item in model_records))
    if type(profiles) is not list or len(profiles) != len(expected_model_ids):
        raise ForecastCampaignError("resolved model profile identities are incomplete")
    profile_ids: list[str] = []
    for index, raw_profile in enumerate(profiles):
        profile = _exact(
            raw_profile,
            {
                "model_id",
                "path",
                "sha256",
                "implementation_config",
                "implementation_config_sha256",
            },
            f"resolved model profile {index}",
        )
        model_id = profile["model_id"]
        relative = profile["path"]
        expected_sha = profile["sha256"]
        if type(model_id) is not str or type(relative) is not str or type(expected_sha) is not str:
            raise ForecastCampaignError(f"resolved model profile {index} identity is invalid")
        path = _source_path(pathlib.Path(relative), f"model profile {model_id}")
        if not path.is_relative_to(catalog):
            raise ForecastCampaignError(f"model profile {model_id} is outside the model catalog")
        if not _is_sha256(expected_sha) or _sha_file(path) != expected_sha:
            raise ForecastCampaignError(f"model profile {model_id} bytes changed after compilation")
        implementation_relative = profile["implementation_config"]
        implementation_sha = profile["implementation_config_sha256"]
        if (implementation_relative is None) != (implementation_sha is None):
            raise ForecastCampaignError(
                f"model profile {model_id} implementation identity is incomplete"
            )
        if implementation_relative is not None:
            if type(implementation_relative) is not str or not _is_sha256(implementation_sha):
                raise ForecastCampaignError(
                    f"model profile {model_id} implementation identity is invalid"
                )
            implementation_path = _source_path(
                pathlib.Path(implementation_relative),
                f"model implementation config {model_id}",
            )
            if _sha_file(implementation_path) != implementation_sha:
                raise ForecastCampaignError(
                    f"model implementation config {model_id} bytes changed after compilation"
                )
        profile_ids.append(model_id)
    if profile_ids != expected_model_ids or len(set(profile_ids)) != len(profile_ids):
        raise ForecastCampaignError("resolved model profiles do not exactly match selected models")


def _resolved_catalog_candidates(resolved: dict[str, object]) -> dict[str, Candidate]:
    """Rediscover catalog bytes only when they still match the compiled source identities."""
    source = cast(dict[str, object], resolved["source"])
    catalog = (_root() / cast(str, source["model_catalog"])).resolve()
    discovered = {candidate.id: candidate for candidate in discover_candidates(catalog)}
    profiles = cast(list[dict[str, object]], source["model_profiles"])
    candidates: dict[str, Candidate] = {}
    for profile in profiles:
        model_id = cast(str, profile["model_id"])
        candidate = discovered.get(model_id)
        if candidate is None:
            raise ForecastCampaignError(f"compiled campaign model disappeared: {model_id!r}")
        try:
            profile_path = candidate.source_path.relative_to(_root()).as_posix()
            implementation_path = (
                None
                if candidate.implementation_config_path is None
                else candidate.implementation_config_path.relative_to(_root()).as_posix()
            )
        except ValueError as error:
            raise ForecastCampaignError(
                f"catalog candidate {model_id!r} escaped the repository"
            ) from error
        if (
            profile_path != profile["path"]
            or candidate.source_sha256 != profile["sha256"]
            or implementation_path != profile["implementation_config"]
            or candidate.implementation_config_sha256 != profile["implementation_config_sha256"]
        ):
            raise ForecastCampaignError(
                f"catalog candidate {model_id!r} changed after campaign compilation"
            )
        candidates[model_id] = candidate
    resolved_models = cast(list[dict[str, object]], resolved["models"])
    expected = {cast(str, item["model_id"]) for item in resolved_models}
    if set(candidates) != expected:
        raise ForecastCampaignError("resolved catalog candidates do not match selected models")
    return candidates


def _verify_resolved_parameter_profiles(resolved: dict[str, object]) -> None:
    resolved_schema = resolved.get("schema")
    if type(resolved_schema) is not str or resolved_schema not in {
        LEGACY_RESOLVED_SCHEMA,
        RESOLVED_SCHEMA,
        INSTALLED_RESOLVED_SCHEMA,
    }:
        raise ForecastCampaignError("resolved artifact schema is invalid")
    if resolved_schema != LEGACY_RESOLVED_SCHEMA and "request_schema_version" not in resolved:
        raise ForecastCampaignError("resolved request schema version is missing")
    if resolved_schema == LEGACY_RESOLVED_SCHEMA and "request_schema_version" in resolved:
        raise ForecastCampaignError("legacy resolved artifact must not have request_schema_version")
    request_schema_version = resolved.get("request_schema_version", 3)
    if type(request_schema_version) is not int or request_schema_version not in {1, 2, 3, 4, 5, 6}:
        raise ForecastCampaignError("resolved request schema version is invalid")
    purpose = resolved.get("purpose", "forecast_campaign")
    if type(purpose) is not str or purpose not in {
        "forecast_campaign",
        "single_model_benchmark",
        "research_campaign",
        "capacity_diagnostic",
    }:
        raise ForecastCampaignError("resolved campaign purpose is invalid")
    if request_schema_version >= 5 and "purpose" not in resolved:
        raise ForecastCampaignError("resolved campaign purpose is missing")
    candidates = _resolved_catalog_candidates(resolved)
    models = cast(list[dict[str, object]], resolved.get("models"))
    profiles = resolved.get("parameter_profiles")
    if type(profiles) is not list or len(profiles) != len(models):
        raise ForecastCampaignError("resolved parameter profiles are incomplete")
    data = cast(dict[str, object], resolved["data"])
    temporal = cast(dict[str, object], resolved["temporal"])
    origins = tuple(
        _date(value, "resolved validation origin")
        for value in cast(list[object], temporal["validation_origins"])
    )
    step_days = _positive_int(temporal.get("step_days"), "resolved temporal.step_days")
    source = cast(dict[str, object], resolved["source"])
    selection = cast(dict[str, object], resolved["selection"])
    budgets = cast(dict[str, object], resolved["budgets"])
    validation_budget = cast(dict[str, object], budgets["validation"])
    resolved_compute_keys = {
        "device", "batch_size", "gradient_microbatch_size", "memory_gib", "min_free_disk_gib"
    }
    if request_schema_version >= 6:
        resolved_compute_keys.add("precision")
    compute = _exact(resolved["compute"], resolved_compute_keys, "resolved compute")
    if request_schema_version >= 6 and compute["precision"] not in ("16-mixed", "32-true"):
        raise ForecastCampaignError("resolved compute precision is invalid")
    expected_protocol = {
        "metric": selection["metric"],
        "missing_actual": selection["missing_actual"],
        "selection_data_fingerprint": cast(dict[str, object], data["selection"])["sha256"],
        "dataset_contract_sha256": source["dataset_contract_sha256"],
        "features": _features(resolved),
        "training_seed": 0,
        "maximum_epochs": validation_budget["maximum_epochs"],
        "patience": validation_budget["patience"],
        "device": compute["device"],
        "batch_size": compute["batch_size"],
        "gradient_microbatch_size": compute["gradient_microbatch_size"],
        "early_stopping_absolute_min_delta": validation_budget["early_stopping_absolute_min_delta"],
        "early_stopping_relative_min_delta": validation_budget["early_stopping_relative_min_delta"],
    }
    profile_ids: list[str] = []
    for index, raw in enumerate(profiles):
        expected_keys = {
            "candidate_id",
            "model_id",
            "variant_id",
            "mode",
            "study",
            "tune_stage",
            "resolution",
        }
        has_variant_spec = resolved_schema != LEGACY_RESOLVED_SCHEMA
        if has_variant_spec:
            expected_keys.add("variant_spec_sha256")
        profile = _exact(
            raw,
            expected_keys,
            f"resolved parameter profile {index}",
        )
        candidate_id = profile["candidate_id"]
        model_id = profile["model_id"]
        variant_id = profile["variant_id"]
        mode = profile["mode"]
        variant_spec_sha256 = profile.get("variant_spec_sha256")
        if (
            type(candidate_id) is not str
            or type(model_id) is not str
            or type(variant_id) is not str
            or (has_variant_spec and not _is_sha256(variant_spec_sha256))
            or candidate_id != models[index]["candidate_id"]
            or model_id != models[index]["model_id"]
            or variant_id != models[index]["variant_id"]
            or candidate_id != _candidate_id(model_id, variant_id)
            or model_id not in candidates
            or mode
            not in {
                "defaults",
                "manual",
                "tune",
                "reuse_selection",
            }
        ):
            raise ForecastCampaignError(f"resolved parameter profile {index} identity is invalid")
        candidate = candidates[cast(str, model_id)]
        candidate_protocol = {
            **expected_protocol,
            **(
                {"precision": effective_precision(
                    candidate.adapter, cast(str, compute.get("precision", "32-true"))
                )}
                if request_schema_version >= 6 else {}
            ),
        }
        study_value = profile["study"]
        if mode in {"tune", "reuse_selection"}:
            study_record = _exact(
                study_value,
                {"id", "path", "sha256", "max_trials"},
                f"resolved parameter profile {index} study",
            )
            if type(study_record["path"]) is not str:
                raise ForecastCampaignError("resolved HPO study path is invalid")
            study = _campaign_hpo_study(
                ParameterSourceRequest(
                    cast(ParameterMode, mode),
                    study_config=pathlib.Path(cast(str, study_record["path"])),
                ),
                candidate,
                selection_data=cast(dict[str, object], data["selection"]),
                validation_origins=origins,
                step_days=step_days,
                require_data_id=request_schema_version < 4,
            )
            if not _same_exact(study_record, _study_record(study)):
                raise ForecastCampaignError("resolved HPO study identity changed")
        elif study_value is not None:
            raise ForecastCampaignError("non-HPO parameter mode has inactive study fields")
        resolution = profile["resolution"]
        validated_resolution: dict[str, object] | None = None
        try:
            if mode == "tune":
                if resolution is not None or profile["tune_stage"] != f"t{index:02d}":
                    raise ForecastCampaignError("tune mode has invalid stage/resolution fields")
            else:
                if profile["tune_stage"] is not None or type(resolution) is not dict:
                    raise ForecastCampaignError("static parameter mode has invalid stage fields")
                validated = validate_parameter_resolution(
                    cast(dict[str, object], resolution),
                    candidate,
                    project_root=_root(),
                    verify_external=mode == "reuse_selection",
                )
                if cast(dict[str, object], validated["source"])["kind"] != mode:
                    raise ForecastCampaignError("parameter resolution mode differs")
                validated_resolution = validated
        except ParameterSourceError as error:
            raise ForecastCampaignError(
                f"resolved parameter profile {model_id!r} is invalid: {error}"
            ) from error
        generated_variant_id, generated_spec_sha256 = _variant_spec_identity(
            candidate,
            cast(ParameterMode, mode),
            validated_resolution,
            cast(dict[str, object] | None, study_value),
            candidate_protocol,
        )
        if (has_variant_spec and generated_spec_sha256 != variant_spec_sha256) or (
            request_schema_version >= 4 and generated_variant_id != variant_id
        ):
            raise ForecastCampaignError(
                f"resolved parameter profile {model_id!r} variant specification differs"
            )
        profile_ids.append(cast(str, candidate_id))
    if profile_ids != [cast(str, item["candidate_id"]) for item in models]:
        raise ForecastCampaignError("resolved parameter profile order differs from models")


def _fixed_evaluation(resolved: dict, stage: str, origin: str, horizon: int) -> dict | None:
    if "benchmark" not in resolved:
        return None
    from autoforecast.benchmark import boundary, verify_benchmark
    benchmark = resolved["benchmark"]
    verify_benchmark(pathlib.Path(benchmark["path"]), benchmark["sha256"])
    return benchmark["evaluations"][boundary(stage, origin, horizon)]


def _load_resolved(path: pathlib.Path, expected_sha: str) -> dict[str, object]:
    resolved = _load_exact(
        path, expected_sha, (LEGACY_RESOLVED_SCHEMA, RESOLVED_SCHEMA, INSTALLED_RESOLVED_SCHEMA)
    )
    _verify_resolved_source(resolved)
    _verify_resolved_parameter_profiles(resolved)
    return resolved


def _completed_output(campaign_root: pathlib.Path, stage_id: str, relative: str) -> pathlib.Path:
    store = CampaignStore.from_root(campaign_root)
    state = store.state()
    item = next(
        (item for item in cast(list[dict[str, object]], state["stages"]) if item["id"] == stage_id),
        None,
    )
    if item is None or item["status"] != "completed":
        raise ForecastCampaignError(f"required upstream stage is not complete: {stage_id}")
    path = (
        campaign_root
        / "stages"
        / stage_id
        / f"attempt-{cast(int, item['attempts']):04d}"
        / relative
    )
    if path.is_symlink() or not path.exists():
        raise ForecastCampaignError(f"required upstream output is missing: {stage_id}/{relative}")
    return path


def _verify_data(record: dict[str, object], label: str) -> pathlib.Path:
    path = pathlib.Path(cast(str, record["path"]))
    observed = directory_digest(path, suffixes=(".parquet",))
    if observed != record["sha256"]:
        raise ForecastCampaignError(f"{label} bytes changed after campaign compilation")
    return path


def _snapshot_parquet_directory(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    expected_sha256: str,
    label: str,
) -> pathlib.Path:
    """Copy authenticated Parquet bytes into a stage-owned immutable input boundary."""
    if destination.exists():
        raise ForecastCampaignError(f"{label} snapshot destination already exists")
    if directory_digest(source, suffixes=(".parquet",)) != expected_sha256:
        raise ForecastCampaignError(f"{label} changed before its scoring snapshot")
    descendants = () if source.is_file() else tuple(source.rglob("*"))
    if source.is_symlink() or any(item.is_symlink() for item in descendants):
        raise ForecastCampaignError(f"{label} may not contain symlinks")
    destination.mkdir()
    parquet_files = (
        ((pathlib.Path(source.name), source),)
        if source.is_file()
        else tuple(
            (item.relative_to(source), item)
            for item in descendants
            if item.is_file() and item.suffix == ".parquet"
        )
    )
    for relative, parquet in parquet_files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parquet, target, follow_symlinks=False)
    copied = directory_digest(destination, suffixes=(".parquet",))
    if (
        copied != expected_sha256
        or directory_digest(source, suffixes=(".parquet",)) != expected_sha256
    ):
        shutil.rmtree(destination, ignore_errors=True)
        raise ForecastCampaignError(f"{label} changed while its scoring snapshot was copied")
    return destination


def _materialize_training_panel(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    contract_path: pathlib.Path,
    origin: str,
    horizon: int,
    training_start: str | None,
    expected_source_sha256: str,
) -> str:
    """Stream a chronological panel and physically mask future labels when features need rows."""
    if directory_digest(source, suffixes=(".parquet",)) != expected_source_sha256:
        raise ForecastCampaignError("training source changed before materialization")
    contract = load_dataset_contract(contract_path)
    panel = contract.panel(source).validate()
    cutoff = dt.date.fromisoformat(origin)
    end = cutoff + dt.timedelta(days=horizon * period_days(panel.frequency))
    dataset = pads.dataset(source, format="parquet", partitioning="hive")
    date_field = pc.field(panel.time_col)
    condition = date_field <= pa.scalar(end if panel.known_future else cutoff)
    if training_start is not None:
        condition = condition & (date_field >= pa.scalar(dt.date.fromisoformat(training_start)))
    if panel.known_future and panel.exposure_col is None:
        raise ForecastCampaignError(
            "physical future-label masking requires an exposure column when known-future rows "
            "must remain in the training panel"
        )
    destination.mkdir()
    output = destination / "part-00000.parquet"
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in dataset.scanner(filter=condition, batch_size=131072).to_batches():
            if panel.known_future:
                # Arrow generates compute kernels at runtime; its stubs omit these exports.
                future = pc.greater(  # pyright: ignore[reportAttributeAccessIssue]
                    batch[panel.time_col], pa.scalar(cutoff)
                )
                target_index = batch.schema.get_field_index(panel.target_col)
                zero = pa.scalar(0, type=batch[target_index].type)
                batch = batch.set_column(
                    target_index,
                    panel.target_col,
                    pc.if_else(future, zero, batch[target_index]),  # pyright: ignore[reportAttributeAccessIssue]
                )
                exposure_index = batch.schema.get_field_index(cast(str, panel.exposure_col))
                hidden = pa.scalar("unobserved", type=batch[exposure_index].type)
                batch = batch.set_column(
                    exposure_index,
                    cast(str, panel.exposure_col),
                    pc.if_else(future, hidden, batch[exposure_index]),  # pyright: ignore[reportAttributeAccessIssue]
                )
            if writer is None:
                writer = pq.ParquetWriter(output, batch.schema)
            cast(pq.ParquetWriter, writer).write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or not output.is_file():
        shutil.rmtree(destination)
        raise ForecastCampaignError("chronological training panel is empty")
    if directory_digest(source, suffixes=(".parquet",)) != expected_source_sha256:
        shutil.rmtree(destination)
        raise ForecastCampaignError("training source changed during materialization")
    return directory_digest(destination, suffixes=(".parquet",))


def _panel_range_facts(
    source: pathlib.Path,
    *,
    contract_path: pathlib.Path,
    start: str | None = None,
    end: str | None = None,
) -> tuple[str, str, int]:
    """Return exact date bounds and series cardinality for one declared panel slice."""
    panel = load_dataset_contract(contract_path).panel(source).validate()
    dataset = pads.dataset(source, format="parquet", partitioning="hive")
    condition = None
    if start is not None:
        condition = pc.field(panel.time_col) >= pa.scalar(dt.date.fromisoformat(start))
    if end is not None:
        upper = pc.field(panel.time_col) <= pa.scalar(dt.date.fromisoformat(end))
        condition = upper if condition is None else condition & upper
    first: dt.date | None = None
    last: dt.date | None = None
    series: set[object] = set()
    for batch in dataset.scanner(
        columns=[panel.series_col, panel.time_col], filter=condition, batch_size=131072
    ).to_batches():
        for raw in batch[panel.time_col].to_pylist():
            day = raw.date() if isinstance(raw, dt.datetime) else raw
            if isinstance(day, str):
                day = dt.date.fromisoformat(day)
            if not isinstance(day, dt.date):
                raise ForecastCampaignError("HPO panel contains a non-date time value")
            first = day if first is None or day < first else first
            last = day if last is None or day > last else last
        series.update(item for item in batch[panel.series_col].to_pylist() if item is not None)
    if first is None or last is None or not series:
        raise ForecastCampaignError("HPO declared panel slice is empty")
    return first.isoformat(), last.isoformat(), len(series)


def _materialize_panel_range(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    contract_path: pathlib.Path,
    start: str | None,
    end: str,
    expected_source_sha256: str,
) -> str:
    """Copy one governed evaluation range, including metric history but no later labels."""
    if directory_digest(source, suffixes=(".parquet",)) != expected_source_sha256:
        raise ForecastCampaignError("HPO source changed before validation materialization")
    panel = load_dataset_contract(contract_path).panel(source).validate()
    dataset = pads.dataset(source, format="parquet", partitioning="hive")
    condition = pc.field(panel.time_col) <= pa.scalar(dt.date.fromisoformat(end))
    if start is not None:
        condition = condition & (
            pc.field(panel.time_col) >= pa.scalar(dt.date.fromisoformat(start))
        )
    destination.mkdir()
    output = destination / "part-00000.parquet"
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in dataset.scanner(filter=condition, batch_size=131072).to_batches():
            if writer is None:
                writer = pq.ParquetWriter(output, batch.schema)
            cast(pq.ParquetWriter, writer).write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or not output.is_file():
        shutil.rmtree(destination)
        raise ForecastCampaignError("evaluation panel is empty")
    if directory_digest(source, suffixes=(".parquet",)) != expected_source_sha256:
        shutil.rmtree(destination)
        raise ForecastCampaignError("source changed during evaluation materialization")
    return directory_digest(destination, suffixes=(".parquet",))


def _features(resolved: dict[str, object]) -> dict[str, object] | None:
    feature_record = cast(dict[str, object], resolved["template_features"])
    events = cast(list[list[object]], feature_record["calendar_events"])
    if not events:
        return None
    return {
        "calendar_events": {
            "max_support": feature_record["event_max_support"],
            "events": events,
        }
    }


def _parameter_profile(resolved: dict[str, object], candidate_id: str) -> dict[str, object]:
    matches = [
        item
        for item in cast(list[dict[str, object]], resolved["parameter_profiles"])
        if item["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ForecastCampaignError(
            f"no unique parameter profile for campaign candidate {candidate_id!r}"
        )
    return matches[0]


def _campaign_candidate(resolved: dict[str, object], model_id: str) -> Candidate:
    candidate = _resolved_catalog_candidates(resolved).get(model_id)
    if candidate is None:
        raise ForecastCampaignError(f"no unique catalog candidate for model {model_id!r}")
    return candidate


def _validated_parameter_record(
    record: object,
    candidate: Candidate,
    *,
    expected_mode: object,
) -> dict[str, object]:
    if type(record) is not dict:
        raise ForecastCampaignError("parameter resolution must be a JSON object")
    try:
        validated = validate_parameter_resolution(
            cast(dict[str, object], record),
            candidate,
            project_root=_root(),
            verify_external=expected_mode in {"tune", "reuse_selection"},
        )
    except ParameterSourceError as error:
        raise ForecastCampaignError(f"parameter resolution is invalid: {error}") from error
    if cast(dict[str, object], validated["source"]).get("kind") != expected_mode:
        raise ForecastCampaignError("parameter resolution source mode differs from campaign")
    return validated


def _child_parameter_resolution(
    resolved: dict[str, object], candidate_id: str, campaign_root: pathlib.Path
) -> dict[str, object]:
    profile = _parameter_profile(resolved, candidate_id)
    model_id = cast(str, profile["model_id"])
    candidate = _campaign_candidate(resolved, model_id)
    if profile["mode"] == "tune":
        path = _completed_output(
            campaign_root, cast(str, profile["tune_stage"]), "parameter-resolution.json"
        )
        try:
            value = load_exact_json(path)
        except ExactJSONError as error:
            raise ForecastCampaignError(
                f"tuned parameter resolution is invalid: {error}"
            ) from error
        record = _validated_parameter_record(value, candidate, expected_mode=profile["mode"])
        study = cast(dict[str, object], profile["study"])
        record_source = cast(dict[str, object], record["source"])
        if (
            record_source["study_id"] != study["id"]
            or record_source["study_config_sha256"] != study["sha256"]
        ):
            raise ForecastCampaignError("tuned parameters differ from the configured study")
        return record
    return _validated_parameter_record(
        profile["resolution"], candidate, expected_mode=profile["mode"]
    )


def _write_hpo_request(
    path: pathlib.Path,
    resolved: dict[str, object],
    *,
    model_id: str,
    origin: dt.date,
    horizon: int,
) -> None:
    source = cast(dict[str, object], resolved["source"])
    selection = cast(dict[str, object], resolved["selection"])
    run: dict[str, object] = {
        "name": f"hpo-{model_id}"[:64],
        "dataset_contract": source["dataset_contract"],
        "model_catalog": source["model_catalog"],
        "models": [model_id],
        "origin": origin.isoformat(),
        "horizon": horizon,
        "tracking": {"mlflow": {"enabled": False}},
        "evaluation": {
            "missing_actual": selection["missing_actual"],
            "metric": selection["metric"],
            "input_mode": "post_training_score_only",
            "evidence_role": "hpo_validation_selection",
        },
    }
    features = _features(resolved)
    if features is not None:
        run["features"] = features
    import yaml

    schema_version = 3 if cast(int, resolved.get("request_schema_version", 3)) >= 6 else 2
    path.write_text(
        yaml.safe_dump(
            {"schema_version": schema_version, "run": run},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def run_tune_parameters(
    resolved_path: pathlib.Path,
    resolved_sha: str,
    *,
    model_id: str,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
    variant_id: str = "default",
) -> None:
    """Run inner-validation HPO as a durable dependency and seal its rank-one values."""
    del campaign_root  # The supervisor owns this stage; HPO evidence stays below its attempt.
    resolved = _load_resolved(resolved_path, resolved_sha)
    candidate_id = _candidate_id(model_id, variant_id)
    profile = _parameter_profile(resolved, candidate_id)
    if profile["mode"] != "tune" or profile["resolution"] is not None:
        raise ForecastCampaignError(
            f"campaign candidate {candidate_id!r} has no pending tune stage"
        )
    candidate = _campaign_candidate(resolved, model_id)
    study_record = cast(dict[str, object], profile["study"])
    try:
        study = load_hpo_study(_root() / cast(str, study_record["path"]), project_root=_root())
    except HPOError as error:
        raise ForecastCampaignError(f"cannot load frozen HPO study: {error}") from error
    if not _same_exact(_study_record(study), study_record):
        raise ForecastCampaignError("HPO study changed after campaign compilation")
    target = cast(list[str], study.data["validation_target_date_range"])
    start, end = (dt.date.fromisoformat(value) for value in target)
    step_days = cast(int, cast(dict[str, object], resolved["temporal"])["step_days"])
    span = (end - start).days
    horizon = span // step_days + 1
    origin = start - dt.timedelta(days=step_days)
    request_path = attempt_dir / "hpo-run-request.yaml"
    _write_hpo_request(request_path, resolved, model_id=model_id, origin=origin, horizon=horizon)
    data_record = cast(dict[str, object], cast(dict[str, object], resolved["data"])["selection"])
    data_path = _verify_data(data_record, "HPO selection data")
    contract_path = _root() / cast(
        str, cast(dict[str, object], resolved["source"])["dataset_contract"]
    )
    declared_training = cast(list[str], study.data["training_date_range"])
    declared_validation_history = cast(list[str], study.data["validation_date_range"])
    declared_observed = cast(list[str], study.data["observed_date_range"])
    declared_test = cast(list[str], study.data["test_date_range"])
    declared_test_target = cast(list[str], study.data["test_target_date_range"])
    if declared_training != declared_validation_history:
        raise ForecastCampaignError(
            "builtin HPO executes one shared history; training_date_range and "
            "validation_date_range must match"
        )
    observed_facts = _panel_range_facts(data_path, contract_path=contract_path)
    training_facts = _panel_range_facts(
        data_path,
        contract_path=contract_path,
        start=declared_training[0],
        end=declared_training[1],
    )
    validation_facts = _panel_range_facts(
        data_path, contract_path=contract_path, start=target[0], end=target[1]
    )
    test_facts = _panel_range_facts(
        data_path, contract_path=contract_path, start=declared_test[0], end=declared_test[1]
    )
    test_target_facts = _panel_range_facts(
        data_path,
        contract_path=contract_path,
        start=declared_test_target[0],
        end=declared_test_target[1],
    )
    if (
        list(observed_facts[:2]) != declared_observed
        or list(training_facts[:2]) != declared_training
        or training_facts[2] != study.data["training_series"]
        or list(validation_facts[:2]) != target
        or validation_facts[2] != study.data["validation_series"]
        or list(test_facts[:2]) != declared_test
        or list(test_target_facts[:2]) != declared_test_target
    ):
        raise ForecastCampaignError("actual HPO panel facts differ from the frozen study split")
    training_path = attempt_dir / "hpo-training-panel"
    training_fingerprint = _materialize_training_panel(
        data_path,
        training_path,
        contract_path=contract_path,
        origin=origin.isoformat(),
        horizon=horizon,
        training_start=declared_training[0],
        expected_source_sha256=cast(str, data_record["sha256"]),
    )
    validation_path = attempt_dir / "hpo-validation-panel"
    validation_fingerprint = _materialize_panel_range(
        data_path,
        validation_path,
        contract_path=contract_path,
        # Point metrics use only target dates, but WRMSSE also needs pre-origin demand history
        # for scale and revenue weights. This panel remains evaluation-only and ends at the
        # validation target boundary, so final/test labels are still physically absent.
        start=declared_validation_history[0],
        end=target[1],
        expected_source_sha256=cast(str, data_record["sha256"]),
    )
    budget = cast(dict[str, object], cast(dict[str, object], resolved["budgets"])["validation"])
    compute = cast(dict[str, object], resolved["compute"])
    # The bridge names study-owned slices with the study label; their fingerprints remain proof.
    study_data_id = cast(str, study.data["id"])
    bridge = ForecastTrialBridge(
        request_path=request_path,
        data_path=training_path,
        data_snapshot_id=f"{study_data_id}-hpo-training",
        data_reference=(
            f"{str(data_record['reference']).rstrip('/')}"
            f"/__autoforecast_slice__/hpo-training-through-{origin.isoformat()}"
        ),
        epochs=cast(int, budget["maximum_epochs"]),
        timeout_seconds=cast(int, budget["timeout_seconds"]),
        project_root=_root(),
        candidate=candidate,
        patience=cast(int, budget["patience"]),
        device=cast(str, compute["device"]),
        precision=cast(str, compute.get("precision", "32-true")),
        batch_size=cast(int, compute["batch_size"]),
        gradient_microbatch_size=cast(int | None, compute["gradient_microbatch_size"]),
        early_stopping_absolute_min_delta=cast(float, budget["early_stopping_absolute_min_delta"]),
        early_stopping_relative_min_delta=cast(float, budget["early_stopping_relative_min_delta"]),
        training_data_fingerprint=training_fingerprint,
        evaluation_data_path=validation_path,
        evaluation_data_snapshot_id=f"{study_data_id}-hpo-validation",
        evaluation_data_reference=(
            f"{str(data_record['reference']).rstrip('/')}"
            f"/__autoforecast_slice__/hpo-validation-"
            f"{declared_validation_history[0]}-to-{target[1]}"
        ),
        evaluation_data_fingerprint=validation_fingerprint,
        source_data_path=data_path,
        source_data_fingerprint=cast(str, data_record["sha256"]),
    )
    hpo_root = attempt_dir / "hpo"
    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_URI")
        if cast(dict[str, object], resolved["tracking"])["mlflow"]
        else None
    )
    try:
        run_hpo_study(
            study,
            hpo_root,
            bridge,
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment=os.environ.get(
                "AUTOFORECAST_MLFLOW_HPO_EXPERIMENT", "AutoForecast HPO"
            ),
        )
        selection_path = hpo_root / "studies" / study.id / "selection.json"
        record = parameter_resolution_from_selection(
            candidate,
            study,
            selection_path,
            kind="tune",
            project_root=_root(),
            expected_protocol={
                "metric": cast(dict[str, object], resolved["selection"])["metric"],
                "missing_actual": cast(dict[str, object], resolved["selection"])["missing_actual"],
                "selection_data_fingerprint": data_record["sha256"],
                "dataset_contract_sha256": cast(dict[str, object], resolved["source"])[
                    "dataset_contract_sha256"
                ],
                "features": _features(resolved),
                "training_seed": 0,
                "maximum_epochs": cast(dict[str, object], budget)["maximum_epochs"],
                "patience": cast(dict[str, object], budget)["patience"],
                "device": compute["device"],
                "batch_size": compute["batch_size"],
                "gradient_microbatch_size": compute["gradient_microbatch_size"],
                "early_stopping_absolute_min_delta": cast(dict[str, object], budget)[
                    "early_stopping_absolute_min_delta"
                ],
                "early_stopping_relative_min_delta": cast(dict[str, object], budget)[
                    "early_stopping_relative_min_delta"
                ],
                **(
                    {"precision": effective_precision(
                        candidate.adapter, cast(str, compute["precision"])
                    )}
                    if cast(int, resolved.get("request_schema_version", 3)) >= 6 else {}
                ),
            },
        )
    except (HPOError, ParameterSourceError, ValueError) as error:
        raise ForecastCampaignError(f"HPO parameter stage failed: {error}") from error
    (attempt_dir / "parameter-resolution.json").write_text(_json(record), encoding="utf-8")


def _write_child_request(
    path: pathlib.Path,
    resolved: dict[str, object],
    child: dict[str, object],
    *,
    evaluation: bool,
) -> None:
    source = cast(dict[str, object], resolved["source"])
    run: dict[str, object] = {
        "name": cast(str, child["id"]),
        "dataset_contract": source["dataset_contract"],
        "model_catalog": source["model_catalog"],
        "models": [child["model_id"]],
        "origin": child["origin"],
        "horizon": child["horizon"],
        "tracking": {
            "mlflow": {
                "enabled": cast(dict[str, object], resolved["tracking"])["mlflow"]
                if resolved.get("purpose") != "capacity_diagnostic"
                else False
            }
        },
    }
    feature = _features(resolved)
    if feature is not None:
        run["features"] = feature
    version = 1
    if evaluation:
        version = 2
        selection = cast(dict[str, object], resolved["selection"])
        run["evaluation"] = {
            "missing_actual": selection["missing_actual"],
            "metric": selection["metric"],
            "input_mode": "post_training_score_only",
            "evidence_role": "rolling_validation",
        }
    if cast(int, resolved.get("request_schema_version", 3)) >= 6:
        version = 3
    import yaml

    path.write_text(
        yaml.safe_dump({"schema_version": version, "run": run}, sort_keys=False),
        encoding="utf-8",
    )


def _peak_rss(process: subprocess.Popen[bytes]) -> int | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is present in governed runtime
        process.wait()
        return None
    peak = 0
    try:
        root = psutil.Process(process.pid)
    except psutil.Error:
        process.wait()
        return peak
    while process.poll() is None:
        with contextlib.suppress(psutil.Error):
            processes = [root, *root.children(recursive=True)]
            peak = max(peak, sum(item.memory_info().rss for item in processes))
        time.sleep(0.1)
    with contextlib.suppress(psutil.Error):
        peak = max(peak, root.memory_info().rss)
    return peak


def _score_value(comparison: dict[str, object], metric: str, model_id: str) -> float:
    if type(comparison) is not dict:
        raise ForecastCampaignError("child comparison must be a mapping")
    candidates_value = comparison.get("candidates")
    if (
        type(candidates_value) is not list
        or len(candidates_value) != 1
        or type(candidates_value[0]) is not dict
        or candidates_value[0].get("model_id") != model_id
    ):
        raise ForecastCampaignError("child comparison does not contain exactly its declared model")
    candidates = cast(list[dict[str, object]], candidates_value)
    candidate = candidates[0]
    if candidate.get("status") != "scored" or type(candidate.get("score")) is not dict:
        raise ForecastCampaignError(f"child model was not scored: {candidate.get('detail', '')}")
    scores = cast(dict[str, object], candidate["score"])
    raw = scores.get("wrmsse" if metric == "wrmsse" else metric)
    if not is_finite_real(raw) or float(raw) < 0:
        raise ForecastCampaignError("child score is absent or non-finite")
    return float(raw)


_CHILD_RESULT_FIELDS = {
    "schema",
    "status",
    "kind",
    "child_id",
    "candidate_id",
    "model_id",
    "variant_id",
    "adapter",
    "parameter_mode",
    "parameter_resolution",
    "parameter_resolution_sha256",
    "initialization",
    "seed",
    "origin",
    "horizon",
    "budget",
    "training_data_sha256",
    "parent",
    "run_path",
    "run_summary_sha256",
    "forecast_record_path",
    "forecast_record_sha256",
    "forecast_artifact_sha256",
    "runtime_seconds",
    "peak_rss_bytes",
    "replay",
    "checkpoint",
    "score",
    "comparison_sha256",
    "selection_lock_sha256",
}
_CHILD_RESULT_PRECISION_FIELDS = _CHILD_RESULT_FIELDS | {"training_controls_sha256"}


def _rolling_metrics_path(run_path: pathlib.Path, model_id: str) -> pathlib.Path:
    matches = sorted((run_path / "attempts" / model_id).glob("*/metrics/epochs.jsonl"))
    if len(matches) != 1:
        raise ForecastCampaignError("revision child has no unique training metrics journal")
    path = _path_without_links(str(matches[0]), "revision training metrics journal")
    if not path.is_file():
        raise ForecastCampaignError("revision training metrics journal is missing")
    return path


def _rolling_metrics_rows(result: dict[str, object]) -> list[dict[str, object]]:
    """Read exactly the update journal whose digest is in the sealed child result."""
    run_path = result.get("run_path")
    model_id = result.get("model_id")
    metrics_sha = result.get("training_metrics_sha256")
    if type(run_path) is not str or type(model_id) is not str or not _is_sha256(metrics_sha):
        raise ForecastCampaignError("revision training metrics identity is missing")
    metrics_path = _rolling_metrics_path(pathlib.Path(run_path), model_id)
    try:
        raw = metrics_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metrics_sha:
            raise ForecastCampaignError("revision training metrics changed")
        rows = [loads_exact_json(line) for line in raw.splitlines()]
    except (OSError, ExactJSONError) as error:
        raise ForecastCampaignError("revision training metrics are invalid") from error
    if not rows or any(type(row) is not dict for row in rows):
        raise ForecastCampaignError("revision training metrics are invalid")
    return cast(list[dict[str, object]], rows)


def _rolling_selection_objective_masks(
    resolved: dict[str, object], result: dict[str, object], origins: tuple[str, ...],
) -> dict[str, object]:
    """Rebuild likelihood support from the authenticated, physically masked training panel."""
    import numpy as np

    from autoforecast.pact_features import build_future_segments

    run_path = result.get("run_path")
    digest = result.get("training_data_sha256")
    if type(run_path) is not str or not _is_sha256(digest):
        raise ForecastCampaignError("revision training panel identity is missing")
    panel_path = _path_without_links(
        pathlib.Path(run_path).parent.parent / "training-panel", "revision training panel"
    )
    if directory_digest(panel_path, suffixes=(".parquet",)) != digest:
        raise ForecastCampaignError("revision training panel changed")
    source = cast(dict[str, object], resolved["source"])
    contract_path = _root() / cast(str, source["dataset_contract"])
    panel = load_dataset_contract(contract_path).panel(panel_path).validate()
    dataset = pads.dataset(panel_path, format="parquet", partitioning="hive")
    time_field = pc.field(panel.time_col)
    earliest: dt.date | None = None
    for batch in dataset.scanner(columns=[panel.time_col], batch_size=131072).to_batches():
        # This public kernel is generated dynamically by Arrow.
        minimum = pc.min(batch[panel.time_col]).as_py()  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(minimum, dt.datetime):
            minimum = minimum.date()
        if minimum is not None:
            if type(minimum) is not dt.date:
                raise ForecastCampaignError("revision training panel dates are invalid")
            earliest = minimum if earliest is None else min(earliest, minimum)
    if earliest is None:
        raise ForecastCampaignError("revision training panel is empty")
    first = dataset.to_table(
        columns=[panel.series_col], filter=time_field == pa.scalar(earliest)
    ).sort_by([(panel.series_col, "ascending")])
    ids = first[panel.series_col].to_pylist()
    if not ids or len(ids) != len(set(ids)) or any(type(item) is not str for item in ids):
        raise ForecastCampaignError("revision training series order is invalid")
    row_by_id = {series_id: index for index, series_id in enumerate(ids)}
    n_series = len(ids)
    horizon = result.get("horizon")
    if type(horizon) is not int or horizon < 1 or not origins:
        raise ForecastCampaignError("revision training windows are invalid")
    stride = period_days(panel.frequency)
    masks: dict[str, object] = {}
    for origin in origins:
        try:
            cutoff = dt.date.fromisoformat(origin)
        except (TypeError, ValueError) as error:
            raise ForecastCampaignError("revision journal origin is invalid") from error
        if cutoff.isoformat() != origin:
            raise ForecastCampaignError("revision journal origin is invalid")
        end = cutoff + dt.timedelta(days=horizon * stride)
        states = np.full((n_series, horizon), 2, dtype=np.int8)
        seen = np.zeros((n_series, horizon), dtype=bool)
        columns = [panel.series_col, panel.time_col]
        if panel.exposure_col:
            columns.append(panel.exposure_col)
        condition = (time_field > pa.scalar(cutoff)) & (time_field <= pa.scalar(end))
        for batch in dataset.scanner(
            columns=columns, filter=condition, batch_size=131072
        ).to_batches():
            for index, (series_id, raw_day) in enumerate(zip(
                batch[panel.series_col].to_pylist(), batch[panel.time_col].to_pylist(),
                strict=True,
            )):
                day = raw_day.date() if isinstance(raw_day, dt.datetime) else raw_day
                offset = (day - cutoff).days if isinstance(day, dt.date) else -1
                position = offset // stride - 1
                if (
                    series_id not in row_by_id or offset <= 0 or offset % stride
                    or not 0 <= position < horizon
                    or seen[row_by_id[series_id], position]
                ):
                    raise ForecastCampaignError("revision training window grid is invalid")
                row = row_by_id[series_id]
                state = (
                    batch[panel.exposure_col][index].as_py()
                    if panel.exposure_col else "exact"
                )
                if state not in {"exact", "lower_bound", "unobserved"}:
                    raise ForecastCampaignError("revision training observation state is invalid")
                states[row, position] = {"exact": 0, "lower_bound": 1,
                                         "unobserved": 2}[state]
                seen[row, position] = True
        if not seen.all():
            raise ForecastCampaignError("revision training window grid is incomplete")
        if result.get("adapter") == "pact":
            masks[origin] = build_future_segments(
                np.empty((n_series, horizon, 0), dtype=np.float32),
                origin=cutoff,
                target=np.zeros((n_series, horizon), dtype=np.float32),
                physical_state=states,
                target_estimand=panel.target_estimand,
            )["objective_day_mask"]
        else:
            if np.any(states == 1):
                raise ForecastCampaignError("revision training has unsupported lower bounds")
            masks[origin] = states == 0
    if directory_digest(panel_path, suffixes=(".parquet",)) != digest:
        raise ForecastCampaignError("revision training panel changed during verification")
    return masks


def _governed_rolling_selection_origins(
    resolved: dict[str, object], result: dict[str, object], *,
    training_key: str = "selection_training_origins",
) -> tuple[str, ...]:
    """Derive every training origin from governed parameters, independent of the journal."""
    training = _revision_training_record(result)
    if training is None:
        raise ForecastCampaignError("governed rolling training evidence is missing")
    parameters = result.get("parameter_resolution")
    effective = parameters.get("effective_parameters") if type(parameters) is dict else None
    config = training.get("config")
    temporal = resolved.get("temporal")
    windows = effective.get("training.train_windows") if type(effective) is dict else None
    stride = effective.get("training.train_stride") if type(effective) is dict else None
    encoder = effective.get("model.encoder_days") if type(effective) is dict else None
    step_days = temporal.get("step_days") if type(temporal) is dict else None
    horizon = result.get("horizon")
    origin = result.get("origin")
    if (
        type(windows) is not int or windows < 1
        or type(stride) is not int or stride < 1
        or type(encoder) is not int or encoder < 1
        or type(step_days) is not int or step_days < 1
        or type(horizon) is not int or horizon < 1
        or type(origin) is not str or type(config) is not dict
        or config.get("train_windows") != windows
        or config.get("train_stride") != stride
        or config.get("encoder_days") != encoder
    ):
        raise ForecastCampaignError("governed rolling training-origin controls differ")
    try:
        cutoff = dt.date.fromisoformat(origin)
        origins = tuple(
            (cutoff - dt.timedelta(days=(horizon + index * stride) * step_days)).isoformat()
            for index in range(windows)
        )
    except (ValueError, OverflowError) as error:
        raise ForecastCampaignError("governed rolling training origins are invalid") from error
    if origin != cutoff.isoformat() or not _same_exact(
        training.get(training_key), origins
    ):
        raise ForecastCampaignError("checkpoint training origins differ from governed schedule")
    return origins


def _verify_rolling_selection_journal(
    result: dict[str, object], metric: str, batch_size: int,
    objective_masks: dict[str, object], *, expected_validation_every: int = 1,
) -> None:
    """Bind the checkpoint's selection history to the independently sealed update journal."""
    training = _revision_training_record(result)
    if training is None:
        raise ForecastCampaignError("revision selection has no training evidence")
    selection_rows = _rolling_metrics_rows(result)
    if any(row.get("phase") != "selection" for row in selection_rows):
        raise ForecastCampaignError("revision selection journal has another training phase")
    import numpy as np

    from autoforecast.rolling_training import (
        replay_sampler_journal,
        verified_validation_summary,
    )

    steps = training.get("steps_per_epoch")
    ceiling = training.get("epochs_per_phase")
    config = training.get("config")
    validation_every = (
        config.get("validation_every_epochs") if type(config) is dict else None
    )
    if type(steps) is not int or steps < 1 or type(ceiling) is not int or not selection_rows:
        raise ForecastCampaignError("revision journal epoch cadence is invalid")
    if (
        type(expected_validation_every) is not int or expected_validation_every < 1
        or validation_every != expected_validation_every
    ):
        raise ForecastCampaignError("revision journal validation cadence is invalid")
    journal_origins = [row.get("origin") for row in selection_rows]
    if (
        not objective_masks or any(type(origin) is not str for origin in journal_origins)
        or not set(journal_origins) <= set(objective_masks)
        or not _same_exact(training.get("selection_training_origins"),
                           tuple(objective_masks))
    ):
        raise ForecastCampaignError("revision journal objective origins differ")
    trainable = np.stack([
        np.asarray(mask).any(1) for mask in objective_masks.values()
    ]).any(0)
    if not trainable.any():
        raise ForecastCampaignError("revision journal has no trainable series")
    seed = result.get("seed")
    config = training.get("config")
    if (
        type(seed) is not int or seed < 0 or type(config) is not dict
        or config.get("seed") != seed or config.get("factor_rank") != 0
        or config.get("batch_size") != batch_size
    ):
        raise ForecastCampaignError("revision journal sampler controls differ")
    try:
        replay = replay_sampler_journal(
            selection_rows, phase="selection",
            objective_masks={key: np.asarray(value) for key, value in objective_masks.items()},
            blocks=[np.asarray([index], dtype=np.int64) for index in np.flatnonzero(trainable)],
            batch_size=batch_size, seed=seed, epochs=ceiling,
            hierarchy_every=1, hierarchy_only_enabled=False,
        )
    except ValueError as error:
        raise ForecastCampaignError(f"revision journal sampler is invalid: {error}") from error
    if replay["steps_per_epoch"] != steps:
        raise ForecastCampaignError("revision journal sampler epoch cadence differs")
    seen = np.zeros(len(trainable), dtype=bool)
    visits = np.zeros(len(trainable), dtype=np.int64)
    proposals = np.zeros(len(trainable), dtype=np.int64)
    epoch_visits = np.zeros(len(trainable), dtype=np.int64)
    epoch_proposals = np.zeros(len(trainable), dtype=np.int64)
    epoch_summaries: list[dict[str, int]] = []
    for row in selection_rows:
        origin = cast(str, row["origin"])
        mask = np.asarray(objective_masks[origin])
        sampled = row.get("sampled_series_indices")
        if (
            mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[0] != len(trainable)
            or type(sampled) is not list
            or any(type(index) is not int or not 0 <= index < len(trainable)
                   for index in sampled)
            or len(set(sampled)) != len(sampled)
            or (sampled and not trainable[sampled].all())
        ):
            raise ForecastCampaignError("revision journal sampled-series evidence is invalid")
        if not sampled and (
            row.get("origin_sampling") != "hierarchy_only"
            or origin != next(iter(objective_masks))
            or row.get("likelihood_series_in_batch") != 0
            or row.get("likelihood_informative_series_in_batch") != 0
        ):
            raise ForecastCampaignError("revision journal hierarchy-only evidence is invalid")
        observed = mask[sampled].any(1)
        if (
            row.get("likelihood_series_in_batch") != len(sampled)
            or row.get("likelihood_informative_series_in_batch") != int(observed.sum())
        ):
            raise ForecastCampaignError("revision journal informative coverage is invalid")
        indices = np.asarray(sampled, dtype=np.int64)
        proposals[indices] += 1
        visits[indices[observed]] += 1
        epoch_proposals[indices] += 1
        epoch_visits[indices[observed]] += 1
        seen[indices[observed]] = True
        update = row.get("update")
        if type(update) is not int:
            raise ForecastCampaignError("revision journal update is invalid")
        if update % steps == 0:
            summary = {
                "epoch": update // steps,
                "proposed_series": int((epoch_proposals[trainable] > 0).sum()),
                "minimum_row_proposals": int(epoch_proposals[trainable].min()),
                "maximum_row_proposals": int(epoch_proposals[trainable].max()),
                "informative_series": int((epoch_visits[trainable] > 0).sum()),
                "minimum_informative_visits": int(epoch_visits[trainable].min()),
                "maximum_informative_visits": int(epoch_visits[trainable].max()),
            }
            if not _same_exact(row.get("epoch_coverage"), summary):
                raise ForecastCampaignError("revision journal epoch coverage is invalid")
            epoch_summaries.append(summary)
            epoch_visits.fill(0)
            epoch_proposals.fill(0)
        slot = row.get("validation_slot")
        if slot is not None and (
            type(slot) is not dict
            or slot.get("covered_series") != int(seen[trainable].sum())
            or slot.get("trainable_series") != int(trainable.sum())
            or slot.get("evaluated") != bool(seen[trainable].all())
        ):
            raise ForecastCampaignError("revision journal validation coverage is invalid")
    slots = [row["validation_slot"] for row in selection_rows if "validation_slot" in row]
    evaluations = []
    validation_origins = training.get("validation_origins")
    if not isinstance(validation_origins, tuple | list) or any(
        type(origin) is not str for origin in validation_origins
    ):
        raise ForecastCampaignError("revision journal validation origins are invalid")
    for row in selection_rows:
        if "validation_summary" not in row:
            if "validation" in row:
                raise ForecastCampaignError("revision journal has orphan validation details")
            continue
        summary = row["validation_summary"]
        if (
            type(summary) is not dict or summary.get("metric_name") != metric
            or type(row.get("update")) is not int
        ):
            raise ForecastCampaignError("revision journal validation summary is invalid")
        try:
            verified_metric = verified_validation_summary(
                row, tuple(validation_origins), metric
            )
        except ValueError as error:
            raise ForecastCampaignError(
                f"revision journal validation summary is invalid: {error}"
            ) from error
        evaluations.append({"update": row["update"], f"mean_{metric}": verified_metric})
    last_update = len(selection_rows)
    epoch_updates = [row.get("update") for row in selection_rows if "epoch_coverage" in row]
    if (
        [row.get("update") for row in selection_rows] != list(range(1, last_update + 1))
        or any(row.get("optimizer_steps") != row.get("update") for row in selection_rows)
        or epoch_updates != list(range(steps, last_update + 1, steps))
        or not _same_exact(training.get("validation_slots"), slots)
        or not _same_exact(training.get("validation"), evaluations)
    ):
        raise ForecastCampaignError("revision selection differs from its committed journal")
    forecast_path = result.get("forecast_record_path")
    forecast_sha = result.get("forecast_record_sha256")
    record = (
        _load_hashed_json(
            _path_without_links(forecast_path, "revision forecast record"),
            cast(str, forecast_sha), "revision forecast record",
        )
        if type(forecast_path) is str and _is_sha256(forecast_sha) else None
    )
    total = record.get("n_series") if type(record) is dict else None
    excluded = training.get("excluded_series_count")
    coverage = training.get("phase_coverage")
    selection_coverage = coverage.get("selection") if type(coverage) is dict else None
    expected_coverage = {
        "epochs": epoch_summaries,
        "series_seen": int(seen[trainable].sum()),
        "informative_series_visits": int(visits.sum()),
        "minimum_visits_per_series": int(visits[trainable].min()),
        "maximum_visits_per_series": int(visits[trainable].max()),
        "minimum_row_proposals": int(proposals[trainable].min()),
        "maximum_row_proposals": int(proposals[trainable].max()),
    }
    if (
        type(total) is not int or type(excluded) is not int
        or not 0 <= excluded < total
        or type(batch_size) is not int or batch_size < 1
        or not _same_exact(selection_coverage, expected_coverage)
        or len(trainable) != total or int(trainable.sum()) != total - excluded
        or not seen[trainable].all()
        or steps != (total - excluded + batch_size - 1) // batch_size
    ):
        raise ForecastCampaignError(
            "revision journal trainable population or epoch cadence is invalid"
        )
    try:
        verify_validation_ledger(
            slots, evaluations, metric=metric,
            validation_interval_updates=steps * cast(int, validation_every),
            budget_updates=ceiling * steps, last_update=last_update,
            trainable_series=total - excluded,
        )
    except ValueError as error:
        raise ForecastCampaignError(
            f"revision journal validation ledger is invalid: {error}"
        ) from error


def _verify_rolling_refit_journal(
    result: dict[str, object], expected: int, *,
    batch_size: int | None = None, resolved: dict[str, object] | None = None,
) -> None:
    """Require one journaled successful optimizer step per selected refit update."""
    rows = _rolling_metrics_rows(result)
    if len(rows) != expected or any(
        row.get("phase") != "refit"
        or type(row.get("update")) is not int or row["update"] != index
        or type(row.get("optimizer_steps")) is not int
        or row["optimizer_steps"] != index
        or any(key in row for key in ("validation", "validation_slot", "validation_summary"))
        for index, row in enumerate(rows, start=1)
    ):
        raise ForecastCampaignError("final refit journal differs from selected optimizer updates")
    if resolved is None:
        return
    import numpy as np

    from autoforecast.rolling_training import replay_sampler_journal

    training = _revision_training_record(result)
    if training is None:
        raise ForecastCampaignError("final refit training evidence is missing")
    config = training.get("config")
    seed = result.get("seed")
    epochs = training.get("epochs_per_phase")
    if (
        type(batch_size) is not int or batch_size < 1
        or type(seed) is not int or seed < 0
        or type(epochs) is not int or epochs < 1
        or type(config) is not dict or config.get("seed") != seed
        or config.get("factor_rank") != 0 or config.get("batch_size") != batch_size
    ):
        raise ForecastCampaignError("final refit sampler controls differ")
    origins = _governed_rolling_selection_origins(
        resolved, result, training_key="refit_training_origins"
    )
    objective_masks = _rolling_selection_objective_masks(resolved, result, origins)
    trainable = np.stack([
        np.asarray(mask).any(1) for mask in objective_masks.values()
    ]).any(0)
    try:
        replay = replay_sampler_journal(
            rows, phase="refit",
            objective_masks={key: np.asarray(value) for key, value in objective_masks.items()},
            blocks=[np.asarray([index], dtype=np.int64) for index in np.flatnonzero(trainable)],
            batch_size=batch_size, seed=seed, epochs=epochs,
            hierarchy_every=1, hierarchy_only_enabled=False,
        )
    except ValueError as error:
        raise ForecastCampaignError(f"final refit sampler is invalid: {error}") from error
    phase_coverage = training.get("phase_coverage")
    if (
        training.get("steps_per_epoch") != replay["steps_per_epoch"]
        or training.get("refit_series_count") != int(trainable.sum())
        or not cast(np.ndarray, replay["seen"])[trainable].all()
        or type(phase_coverage) is not dict
        or not _same_exact(phase_coverage.get("refit"), replay["phase_summary"])
    ):
        raise ForecastCampaignError("final refit sampler population or cadence differs")


_FORECAST_RECORD_FIELDS = {
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


def _verified_forecast(
    record_path: pathlib.Path,
    *,
    expected_sha256: str | None = None,
    expected_adapter: str,
    expected_origin: str,
    expected_horizon: int,
) -> tuple[Forecast, str]:
    if record_path.is_symlink() or not record_path.is_file():
        raise ForecastCampaignError("child forecast record is missing or linked")
    if expected_sha256 is None:
        record, record_sha256 = _load_json_with_identity(record_path, "child forecast record")
    else:
        record = _load_hashed_json(record_path, expected_sha256, "child forecast record")
        record_sha256 = expected_sha256
    if type(record) is not dict or set(record) != _FORECAST_RECORD_FIELDS:
        raise ForecastCampaignError("child forecast record fields differ")
    forecast = Forecast.from_record(cast(dict[str, object], record)).validate()
    if (
        forecast.adapter != expected_adapter
        or forecast.origin != expected_origin
        or forecast.horizon != expected_horizon
        or record["n_emitted"] != forecast.n_emitted
        or record["n_declined"] != forecast.n_declined
    ):
        raise ForecastCampaignError("child forecast identity differs from its resolved request")
    return forecast, record_sha256


def _parent_checkpoint_identity(
    parent_stage: str, parent_result: dict[str, object]
) -> dict[str, object]:
    """Return the exact checkpoint identity authenticated by a governed parent result."""
    checkpoint = parent_result.get("checkpoint")
    if type(checkpoint) is not dict or set(checkpoint) != {"path", "sha256"}:
        raise ForecastCampaignError(
            "initialization parent did not publish one authenticated checkpoint"
        )
    checkpoint_path = _path_without_links(
        cast(str, checkpoint["path"]), "initialization parent checkpoint"
    )
    if not _is_sha256(checkpoint["sha256"]) or _sha_file(checkpoint_path) != checkpoint["sha256"]:
        raise ForecastCampaignError("initialization parent checkpoint bytes changed")
    return {"stage_id": parent_stage, **checkpoint}


def _revision_training_record(result: dict[str, object]) -> dict[str, object] | None:
    """Read training metadata from an already authenticated revision checkpoint."""
    if result.get("adapter") not in _FIXED_BUDGET_ADAPTERS:
        return None
    checkpoint = result.get("checkpoint")
    if type(checkpoint) is not dict or set(checkpoint) != {"path", "sha256"}:
        raise ForecastCampaignError("revision child did not publish one checkpoint")
    path = _path_without_links(cast(str, checkpoint["path"]), "revision child checkpoint")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ForecastCampaignError("revision child training evidence is unreadable") from error
    if hashlib.sha256(raw).hexdigest() != checkpoint["sha256"]:
        raise ForecastCampaignError("revision selection checkpoint changed")
    try:
        import torch

        saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        training = saved["training"]
    except (
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        pickle.UnpicklingError,
    ) as error:
        raise ForecastCampaignError("revision child training evidence is unreadable") from error
    if type(saved) is not dict or type(training) is not dict:
        raise ForecastCampaignError("revision child training evidence is invalid")
    budget = result.get("budget")
    ceiling = budget.get("maximum_epochs") if type(budget) is dict else None
    steps = training.get("steps_per_epoch")
    selected = training.get("selected_updates")
    recorded_ceiling = (
        training.get("maximum_epochs")
        if result.get("adapter") in {"tft_q", "tft_hnb", }
        else training.get("epochs_per_phase")
    )
    config = training.get("config")
    contract = training.get("training_contract")
    fixed_refit = type(contract) is dict and contract.get("mode") == "fixed_refit"
    if (
        type(ceiling) is not int
        or type(recorded_ceiling) is not int
        or recorded_ceiling != ceiling
        or (result.get("adapter") not in {"tft_q", "tft_hnb", } and (
            type(config) is not dict or type(config.get("epochs")) is not int
            or config.get("epochs") != ceiling
        ))
        or type(steps) is not int or steps < 1
        or type(selected) is not int or selected < 1
        or (not fixed_refit and selected > ceiling * steps)
    ):
        raise ForecastCampaignError("revision ceiling and selected update evidence disagree")
    return cast(dict[str, object], training)


def _selected_update_budget(
    result: dict[str, object], expected_metric: str | None = None,
    batch_size: int | None = None,
    expected_loss_kind: str | None = None,
    *, expected_validation_every: int = 1,
) -> int | None:
    """Read the selected optimizer frontier from fixed-holdout training evidence."""
    training = _revision_training_record(result)
    if training is None:
        return None
    selected = training.get("selected_updates")
    contract = training.get("training_contract")
    validation = training.get("validation")
    origin = result.get("origin")
    if (
        type(selected) is not int
        or selected < 1
        or type(origin) is not str
        or type(contract) is not dict
        or contract.get("mode") != "fixed_validation"
        or contract.get("fixed_update_budget") is not None
        or training.get("refit_optimizer_steps") is not None
        or type(validation) is not list
        or not validation
    ):
        raise ForecastCampaignError("revision selection budget evidence is invalid")
    budget = result.get("budget")
    steps = training.get("steps_per_epoch")
    ceiling = budget.get("maximum_epochs") if type(budget) is dict else None
    patience = budget.get("patience") if type(budget) is dict else None
    absolute_delta = (
        budget.get("early_stopping_absolute_min_delta") if type(budget) is dict else None
    )
    relative_delta = (
        budget.get("early_stopping_relative_min_delta") if type(budget) is dict else None
    )
    if (
        type(ceiling) is not int or type(steps) is not int
        or type(patience) is not int or patience < 1
        or not is_finite_real(absolute_delta) or absolute_delta < 0
        or not is_finite_real(relative_delta) or relative_delta < 0
    ):
        raise ForecastCampaignError("revision selection budget controls are invalid")
    if result.get("adapter") in {"tft_q", "tft_hnb", }:
        import torch

        if expected_loss_kind is None:
            parameters = result.get("parameter_resolution")
            effective = parameters.get("effective_parameters") if type(parameters) is dict else None
            expected_loss_kind = (
                effective.get("model.loss_kind") if type(effective) is dict else None
            )
        record_path = result.get("forecast_record_path")
        record_sha = result.get("forecast_record_sha256")
        record = (
            _load_hashed_json(
                _path_without_links(record_path, "TFT-Q/TFT-HNB forecast record"),
                cast(str, record_sha),
                "TFT-Q/TFT-HNB forecast record",
            )
            if type(record_path) is str and _is_sha256(record_sha)
            else None
        )
        series_count = record.get("n_series") if type(record) is dict else None
        examples = series_count
        if (
            type(batch_size) is not int
            or batch_size < 1
            or type(series_count) is not int
            or (series_count < 1)
            or (training.get("n_series") != series_count)
            or (type(examples) is not int)
            or (examples < 1)
            or (steps != (examples + batch_size - 1) // batch_size)
            or (len(validation) > ceiling)
            or (type(training.get("early_stopping_patience")) is not int)
            or (training.get("early_stopping_patience") != patience)
            or (not is_finite_real(training.get("early_stopping_min_delta")))
            or (training.get("early_stopping_min_delta") != absolute_delta)
            or (relative_delta != 0)
            or (expected_metric is not None and expected_metric != "wrmsse")
            or (expected_loss_kind not in {"quantile", "v3_hurdle_nb"})
            or (not _same_exact(training.get("validation"), training.get("epoch_log")))
        ):
            raise ForecastCampaignError(
                "TFT-Q/TFT-HNB selection controls disagree with the budget"
            )
        selected_epoch = training.get("selected_epoch")
        epochs: list[int] = []
        updates: list[int] = []
        scores: list[torch.Tensor] = []
        loss_dtype = torch.float32 if expected_loss_kind == "quantile" else torch.float64
        for row in validation:
            if (
                type(row) is not dict
                or type(row.get("epoch")) is not int
                or type(row.get("updates")) is not int
                or type(row.get("val_loss")) is not float
                or not is_finite_real(row.get("val_loss"))
                or row["val_loss"] < 0
                or type(row.get("val_loss_dtype")) is not str
                or row["val_loss_dtype"] != str(loss_dtype)
            ):
                raise ForecastCampaignError(
                    "TFT-Q/TFT-HNB selection validation history is invalid"
                )
            typed_loss = torch.tensor(row["val_loss"], dtype=loss_dtype)
            if not bool(torch.isfinite(typed_loss)) or float(typed_loss) != row["val_loss"]:
                raise ForecastCampaignError(
                    "TFT-Q/TFT-HNB validation loss is not representable"
                )
            epochs.append(row["epoch"])
            updates.append(row["updates"])
            scores.append(typed_loss)
        best_index = min(range(len(scores)), key=lambda index: float(scores[index]))
        best_stop = torch.tensor(math.inf, dtype=loss_dtype)
        stale = 0
        stopped_early = False
        for index, current in enumerate(scores):
            # Lightning compares tensors after applying min_delta in the tensor dtype.
            if bool(current + absolute_delta < best_stop):
                best_stop, stale = current, 0
            else:
                stale += 1
            if stale >= patience:
                stopped_early = index == len(scores) - 1
                if not stopped_early:
                    raise ForecastCampaignError(
                        "TFT-Q/TFT-HNB selection continued after early stopping"
                    )
        witnessed = (
            training.get("validation_origin") == origin
            and type(selected_epoch) is int
            and epochs == list(range(1, len(epochs) + 1))
            and len(epochs) <= ceiling
            and updates == [epoch * steps for epoch in epochs]
            and (len(epochs) == ceiling or stopped_early)
            and epochs[best_index] == selected_epoch
            and updates[best_index] == selected
        )
    else:
        metric = training.get("selection_metric")
        validation_origins = training.get("validation_origins")
        validation_slots = training.get("validation_slots")
        config = training.get("config")
        validation_every = config.get("validation_every_epochs") if type(config) is dict else None
        if (
            type(metric) is not str
            or metric not in {"rmse", "wrmsse"}
            or (expected_metric is not None and metric != expected_metric)
            or type(config) is not dict
            or config.get("selection_metric") not in (metric, "auto")
            or type(config.get("patience")) is not int
            or config.get("patience") != patience
            or not is_finite_real(config.get("early_stopping_absolute_min_delta"))
            or config.get("early_stopping_absolute_min_delta") != absolute_delta
            or not is_finite_real(config.get("early_stopping_relative_min_delta"))
            or config.get("early_stopping_relative_min_delta") != relative_delta
            or type(validation_every) is not int or validation_every < 1
            or validation_every != expected_validation_every
            or type(validation_slots) is not list or not validation_slots
        ):
            raise ForecastCampaignError("revision selection metric controls are invalid")
        best_score = math.inf
        best_update = 0
        last_update = 0
        stale = 0
        stopped_early = False
        cadence = steps * validation_every
        budget_updates = ceiling * steps
        due_updates = list(range(cadence, budget_updates + 1, cadence))
        if not due_updates or due_updates[-1] != budget_updates:
            due_updates.append(budget_updates)
        slot_updates: list[int] = []
        evaluated_updates: list[int] = []
        covered_before = 0
        trainable_count = None
        for slot in validation_slots:
            if type(slot) is not dict:
                raise ForecastCampaignError("revision validation slot evidence is invalid")
            slot_update = slot.get("update")
            covered = slot.get("covered_series")
            trainable = slot.get("trainable_series")
            evaluated = slot.get("evaluated")
            if (
                type(slot_update) is not int
                or type(covered) is not int or type(trainable) is not int
                or type(evaluated) is not bool
                or trainable < 1 or not 0 <= covered <= trainable
                or covered < covered_before
                or (trainable_count is not None and trainable != trainable_count)
                or evaluated != (covered == trainable)
            ):
                raise ForecastCampaignError("revision validation slot evidence is invalid")
            slot_updates.append(slot_update)
            if evaluated:
                evaluated_updates.append(slot_update)
            covered_before, trainable_count = covered, trainable
        if slot_updates != due_updates[:len(slot_updates)]:
            raise ForecastCampaignError("revision validation slot cadence is invalid")
        seen_updates: list[int] = []
        for index, row in enumerate(validation):
            score = row.get(f"mean_{metric}") if type(row) is dict else None
            update = row.get("update") if type(row) is dict else None
            if (
                type(row) is not dict
                or type(update) is not int or update <= last_update
                or type(score) is not float
                or not is_finite_real(score) or score < 0
            ):
                raise ForecastCampaignError("revision selection validation history is invalid")
            if not math.isfinite(best_score) or score < best_score - max(
                absolute_delta, abs(best_score) * relative_delta
            ):
                best_score, best_update = float(score), update
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                stopped_early = index == len(validation) - 1
                if not stopped_early:
                    raise ForecastCampaignError("revision selection continued after early stopping")
            last_update = update
            seen_updates.append(update)
        witnessed = (
            contract.get("validation_origin") == origin
            and (validation_origins == (origin,) or validation_origins == [origin])
            and seen_updates == evaluated_updates
            and slot_updates[-1] == last_update
            and (last_update == budget_updates or stopped_early)
            and best_update == selected
        )
    if not witnessed:
        raise ForecastCampaignError("revision selected updates lack a validation witness")
    return selected


def _verify_fixed_refit_budget(
    result: dict[str, object], expected: int, *, batch_size: int | None = None,
    resolved: dict[str, object] | None = None,
) -> None:
    """Fail closed if a final worker did not honor the sealed update frontier."""
    training = _revision_training_record(result)
    if training is None:
        raise ForecastCampaignError("final refit adapter cannot prove the sealed update budget")
    contract = training.get("training_contract")
    validation_origins = training.get("validation_origins")
    if (
        training.get("selected_updates") != expected
        or type(training.get("refit_optimizer_steps")) is not int
        or training.get("refit_optimizer_steps") != expected
        or type(contract) is not dict
        or contract.get("mode") != "fixed_refit"
        or contract.get("fixed_update_budget") != expected
        or (
            result.get("adapter") in _ROLLING_BUDGET_ADAPTERS
            and training.get("budget_policy") != "sealed_fixed_update_budget"
        )
        or training.get("validation") != []
        or (
            result.get("adapter") in {"tft_q", "tft_hnb", }
            and training.get("validation_origin") is not None
        )
        or (
            result.get("adapter") not in {"tft_q", "tft_hnb", }
            and (
                contract.get("validation_origin") is not None
                or type(validation_origins) not in (list, tuple)
                or bool(validation_origins)
                or training.get("validation_slots") != []
            )
        )
    ):
        raise ForecastCampaignError("final refit did not honor the sealed update budget")
    if result.get("adapter") in _ROLLING_BUDGET_ADAPTERS:
        _verify_rolling_refit_journal(
            result, expected, batch_size=batch_size, resolved=resolved
        )
    elif result.get("adapter") in {"tft_q", "tft_hnb", }:
        record_path = result.get("forecast_record_path")
        record_sha = result.get("forecast_record_sha256")
        record = (
            _load_hashed_json(
                _path_without_links(record_path, "TFT-Q/TFT-HNB final forecast record"),
                cast(str, record_sha), "TFT-Q/TFT-HNB final forecast record",
            )
            if type(record_path) is str and _is_sha256(record_sha) else None
        )
        count = record.get("n_series") if type(record) is dict else None
        examples = count
        steps = training.get("steps_per_epoch")
        epoch_log = training.get("epoch_log")
        if (
            type(batch_size) is not int
            or batch_size < 1
            or type(count) is not int
            or (count < 1)
            or (type(examples) is not int)
            or (examples < 1)
            or (type(steps) is not int)
            or (steps < 1)
            or (type(epoch_log) is not list)
        ):
            raise ForecastCampaignError(
                "TFT-Q/TFT-HNB final refit epoch telemetry is missing"
            )
        epochs = math.ceil(expected / steps)
        if (
            training.get("n_series") != count
            or steps != (examples + batch_size - 1) // batch_size
            or expected < steps
            or (len(epoch_log) != epochs)
            or (training.get("selected_epoch") != epochs)
            or any(
                (
                    type(row) is not dict
                    or set(row)
                    != {
                        "epoch",
                        "updates",
                        "train_loss",
                        "val_loss",
                        "val_loss_dtype",
                        "elapsed_seconds",
                    }
                    or row.get("epoch") != index
                    or (row.get("updates") != min(index * steps, expected))
                    or (type(row.get("train_loss")) is not float)
                    or (not is_finite_real(row.get("train_loss")))
                    or (row.get("val_loss") is not None)
                    or (row.get("val_loss_dtype") is not None)
                    or (type(row.get("elapsed_seconds")) is not float)
                    or (not is_finite_real(row.get("elapsed_seconds")))
                    or (cast(float, row["elapsed_seconds"]) < 0)
                    for index, row in enumerate(epoch_log, start=1)
                )
            )
        ):
            raise ForecastCampaignError(
                "TFT-Q/TFT-HNB final refit epoch telemetry differs "
                "from selected updates"
            )


def _verify_child_precision_controls(
    run_path: pathlib.Path,
    resolved: dict[str, object],
    child: dict[str, object],
    expected_sha256: object,
) -> None:
    """Bind a schema-v6 child to the precision and controls actually sealed by its run."""
    control_paths = sorted(
        (run_path / "attempts" / cast(str, child["model_id"])).glob(
            "*/metadata/training-controls.json"
        )
    )
    controls = (
        _load_hashed_json(control_paths[0], cast(str, expected_sha256), "child training controls")
        if len(control_paths) == 1 and _is_sha256(expected_sha256) else None
    )
    compute = cast(dict[str, object], resolved["compute"])
    budget = cast(dict[str, object], child["budget"])
    requested = compute["precision"]
    expected = {
        "schema": "autoforecast.training-controls.v2",
        "maximum_epochs": budget["maximum_epochs"],
        "patience": budget["patience"],
        "device": compute["device"],
        "batch_size": compute["batch_size"],
        "gradient_microbatch_size": compute["gradient_microbatch_size"],
        "early_stopping_absolute_min_delta": budget["early_stopping_absolute_min_delta"],
        "early_stopping_relative_min_delta": budget["early_stopping_relative_min_delta"],
        "requested_precision": requested,
        "effective_precision": effective_precision(
            cast(str, child["adapter"]), cast(str, requested)
        ),
    }
    if (
        type(controls) is not dict
        or not _training_controls_are_valid(controls)
        or not _same_exact(controls, expected)
    ):
        raise ForecastCampaignError(
            f"child training precision evidence differs: {child['id']}"
        )


def _verify_revision_checkpoint_binding(
    saved: object, resolved: dict[str, object], child: dict[str, object],
    parameters: dict[str, object], training_path: pathlib.Path,
    training_digest: str,
) -> None:
    """Bind the model's executed recipe to the separately governed parameter resolution."""
    if type(saved) is not dict or type(saved.get("training")) is not dict:
        raise ForecastCampaignError("revision checkpoint has no training record")
    training = cast(dict[str, object], saved["training"])
    effective = parameters.get("effective_parameters")
    if type(effective) is not dict:
        raise ForecastCampaignError("revision checkpoint has no governed parameters")
    budget = cast(dict[str, object], child["budget"])
    compute = cast(dict[str, object], resolved["compute"])
    metric = cast(dict[str, object], resolved["selection"])["metric"]
    mode = (
        "rolling_backtest" if resolved.get("purpose") == "capacity_diagnostic"
        else "fixed_validation" if child["kind"] == "validation" else "fixed_refit"
    )
    if child["adapter"] in _ROLLING_BUDGET_ADAPTERS:
        from autoforecast.model_execution import (
            REVISION_MODEL_PARAMETERS,
            REVISION_TRAINING_PARAMETERS,
            ExecutionPlanError,
            resolve_revision_rolling_config,
        )
        from autoforecast.rolling_training import RollingConfig

        candidate = _campaign_candidate(resolved, cast(str, child["model_id"]))
        try:
            recipe = resolve_revision_rolling_config(
                candidate,
                model_parameters={
                    name: effective[f"model.{name}"]
                    for name in REVISION_MODEL_PARAMETERS[cast(str, child["adapter"])]
                },
                training_parameters={
                    name: effective[f"training.{name}"]
                    for name in REVISION_TRAINING_PARAMETERS
                },
                epochs=cast(int, budget["maximum_epochs"]),
                seed=cast(int, child["seed"]),
            )
            expected = asdict(RollingConfig(**{
                **recipe,
                "device": compute["device"],
                "batch_size": compute["batch_size"],
                "patience": budget["patience"],
                "event_max_support": cast(dict[str, object], resolved["template_features"])[
                    "event_max_support"
                ],
                "early_stopping_absolute_min_delta": float(
                    cast(float, budget["early_stopping_absolute_min_delta"])
                ),
                "early_stopping_relative_min_delta": float(
                    cast(float, budget["early_stopping_relative_min_delta"])
                ),
                **({"selection_metric": metric}
                   if mode == "fixed_validation" and metric in {"rmse", "wrmsse"} else {}),
            }))
        except (ExecutionPlanError, KeyError, TypeError, ValueError) as error:
            raise ForecastCampaignError(
                "governed rolling checkpoint recipe cannot be reconstructed"
            ) from error
        binding = saved.get("binding")
        training_contract = training.get("training_contract")
        precision = effective_precision(
            cast(str, child["adapter"]), cast(str, compute.get("precision", "32-true"))
        )
        if (
            type(binding) is not dict
            or type(training_contract) is not dict
            or not _same_exact(training.get("config"), expected)
            or not _same_exact(binding.get("config"), json.loads(_json(expected)))
            or binding.get("origin") != child["origin"]
            or binding.get("horizon") != child["horizon"]
            or not _same_exact(binding.get("training_contract"), training_contract)
            or training_contract.get("mode") != mode
            or training.get("precision", "32-true") != precision
            or binding.get("precision", "32-true") != precision
        ):
            raise ForecastCampaignError(
                "rolling checkpoint recipe differs from governed parameters"
            )
        return
    if child["adapter"] not in {"tft_q", "tft_hnb"}:
        return
    from autoforecast.adapters.tft_q import LOSS_VARIANTS, QUANTILES

    panel = load_dataset_contract(
        _root() / cast(str, cast(dict[str, object], resolved["source"])["dataset_contract"])
    ).panel(training_path)
    loss_kind = effective.get("model.loss_kind")
    if type(loss_kind) is not str or loss_kind not in LOSS_VARIANTS:
        raise ForecastCampaignError("TFT-Q/TFT-HNB checkpoint has an invalid governed loss")
    names = (
        "hidden_size", "hidden_continuous_size", "attention_head_size", "dropout",
        "max_encoder_length", "min_encoder_length", "lag_short_days", "lag_long_days",
        "output_size",
    )
    values = {name: effective.get(f"model.{name}") for name in names}
    precision = effective_precision(cast(str, child["adapter"]),
                                    cast(str, compute.get("precision", "32-true")))
    contract = training.get("training_contract")
    fixed_budget = contract.get("fixed_update_budget") if type(contract) is dict else None
    identity = {
        "panel": asdict(panel), "origin": child["origin"], "horizon": child["horizon"],
        "seed": child["seed"], "batch_size": compute["batch_size"],
        "early_stopping_patience": budget["patience"],
        "early_stopping_min_delta": budget["early_stopping_absolute_min_delta"],
        "requested_device": compute["device"], "resolved_device": compute["device"],
        **values, "learning_rate": effective.get("training.learning_rate"),
        "loss_kind": loss_kind, "precision": precision, "training_mode": mode,
        "fixed_update_budget": fixed_budget, "max_epochs": budget["maximum_epochs"],
        "panel_sha256": training_digest,
    }
    profile = {
        "variant": LOSS_VARIANTS[loss_kind], "loss_kind": loss_kind,
        "point_prediction": "median" if loss_kind == "quantile" else "mean",
        "target_normalization": "identity",
        "cold_start": "source_unobserved_rows_zero_with_encoder_mask_v1",
        "mean_loss_weight": 0.2 if loss_kind == "v3_hurdle_nb" else None,
        "quantiles": list(QUANTILES) if loss_kind == "quantile" else None,
        **{name: values[name] for name in (
            "hidden_size", "hidden_continuous_size", "attention_head_size", "dropout",
            "max_encoder_length", "min_encoder_length", "output_size",
        )},
        "learning_rate": effective.get("training.learning_rate"), "precision": precision,
        "lags": [values["lag_short_days"], values["lag_long_days"]],
    }
    if (
        saved.get("schema") != "autoforecast.tft-v6-inference-state.v2"
        or not _same_exact(saved.get("run_identity"), identity)
        or not _same_exact(saved.get("profile"), profile)
        or type(contract) is not dict or contract.get("mode") != mode
    ):
        raise ForecastCampaignError(
            "TFT-Q/TFT-HNB checkpoint recipe differs from governed parameters"
        )


def _validated_child_result(
    resolved: dict[str, object],
    child: dict[str, object],
    path: pathlib.Path,
    *,
    require_score: bool,
) -> dict[str, object]:
    try:
        value = load_exact_json(path)
    except ExactJSONError as error:
        raise ForecastCampaignError(f"invalid child result: {error}") from error
    precision_aware = cast(int, resolved.get("request_schema_version", 3)) >= 6
    expected_fields = (
        _CHILD_RESULT_PRECISION_FIELDS if precision_aware else _CHILD_RESULT_FIELDS
    )
    if type(value) is dict and value.get("adapter") in _ROLLING_BUDGET_ADAPTERS:
        expected_fields = expected_fields | {"training_metrics_sha256"}
    if type(value) is not dict or set(value) != expected_fields:
        raise ForecastCampaignError("child result fields differ")
    result = cast(dict[str, object], value)
    identity_fields = (
        "kind",
        "child_id",
        "candidate_id",
        "model_id",
        "variant_id",
        "adapter",
        "parameter_mode",
        "initialization",
        "seed",
        "origin",
        "horizon",
        "budget",
    )
    if (
        result["schema"] != (
            CHILD_RESULT_PRECISION_SCHEMA if precision_aware else CHILD_RESULT_SCHEMA
        )
        or result["status"] != "completed"
        or any(
            result[field] != child[field if field != "child_id" else "id"]
            for field in identity_fields
        )
        or not _is_sha256(result["training_data_sha256"])
    ):
        raise ForecastCampaignError(f"child result identity/status differs: {child['id']}")
    stages_directory = path.parent.parent.parent
    if stages_directory.name != "stages":
        raise ForecastCampaignError("child result is outside a campaign stage attempt")
    campaign_root = stages_directory.parent
    expected_parameters = _child_parameter_resolution(
        resolved, cast(str, child["candidate_id"]), campaign_root
    )
    observed_parameters = _validated_parameter_record(
        result["parameter_resolution"],
        _campaign_candidate(resolved, cast(str, child["model_id"])),
        expected_mode=child["parameter_mode"],
    )
    parameter_path = path.parent / "parameter-resolution.json"
    parameter_sha = cast(str, result["parameter_resolution_sha256"])
    parameter_file = (
        _load_hashed_json(parameter_path, parameter_sha, "child parameter evidence")
        if _is_sha256(parameter_sha)
        else None
    )
    if (
        not _same_exact(observed_parameters, expected_parameters)
        or parameter_file is None
        or parameter_sha != parameter_resolution_sha256(observed_parameters)
        or not _same_exact(parameter_file, observed_parameters)
    ):
        raise ForecastCampaignError(f"child parameter evidence differs: {child['id']}")
    training_path = path.parent / "training-panel"
    if directory_digest(training_path, suffixes=(".parquet",)) != result["training_data_sha256"]:
        raise ForecastCampaignError(f"child training materialization changed: {child['id']}")
    runtime = result["runtime_seconds"]
    rss = result["peak_rss_bytes"]
    if (
        not is_finite_real(runtime)
        or float(runtime) < 0
        or (rss is not None and (type(rss) is not int or rss < 0))
    ):
        raise ForecastCampaignError(f"child runtime evidence is invalid: {child['id']}")
    replay = result["replay"]
    if replay != {
        "status": "verified_artifact_bytes",
        "scope": "sealed forecast manifest and Parquet; not an inference re-execution",
    }:
        raise ForecastCampaignError(f"child replay evidence is invalid: {child['id']}")
    run_path = _run_group_path(path.parent, cast(str, result["run_path"]), "child run path")
    if result["adapter"] in _ROLLING_BUDGET_ADAPTERS:
        metrics_sha = result["training_metrics_sha256"]
        if (
            not _is_sha256(metrics_sha)
            or _sha_file(_rolling_metrics_path(run_path, cast(str, child["model_id"])))
            != metrics_sha
        ):
            raise ForecastCampaignError("revision child training metrics changed")
    summary_path = run_path / "run-summary.json"
    if (
        not _is_sha256(result["run_summary_sha256"])
        or _sha_file(summary_path) != result["run_summary_sha256"]
    ):
        raise ForecastCampaignError(f"child run summary changed: {child['id']}")
    run_parameter_paths = sorted(
        (run_path / "attempts" / cast(str, child["model_id"])).glob(
            "*/metadata/parameter-resolution.json"
        )
    )
    run_parameter = (
        _load_hashed_json(run_parameter_paths[0], parameter_sha, "child run parameter evidence")
        if len(run_parameter_paths) == 1
        else None
    )
    if run_parameter is None or not _same_exact(run_parameter, observed_parameters):
        raise ForecastCampaignError(f"child run parameter evidence differs: {child['id']}")
    if precision_aware:
        _verify_child_precision_controls(
            run_path, resolved, child, result["training_controls_sha256"]
        )
    if child["parameter_mode"] in {"tune", "reuse_selection"}:
        try:
            validate_parameter_source_capsule(
                run_parameter_paths[0].parent.parent,
                expected_parameter_resolution_sha256=parameter_sha,
            )
        except ParameterSourceError as error:
            raise ForecastCampaignError(
                f"child HPO evidence capsule differs: {child['id']}: {error}"
            ) from error
    forecast_path = _path_without_links(
        cast(str, result["forecast_record_path"]), "child forecast record"
    )
    expected_records = sorted(
        (run_path / "attempts" / cast(str, child["model_id"])).glob("*/evaluation/forecast.json")
    )
    if expected_records != [forecast_path] or not _is_sha256(result["forecast_record_sha256"]):
        raise ForecastCampaignError(f"child forecast record location differs: {child['id']}")
    forecast_sha = result["forecast_record_sha256"]
    if not _is_sha256(forecast_sha):
        raise ForecastCampaignError(f"child forecast record changed: {child['id']}")
    forecast, _ = _verified_forecast(
        forecast_path,
        expected_sha256=cast(str, forecast_sha),
        expected_adapter=cast(str, child["adapter"]),
        expected_origin=cast(str, child["origin"]),
        expected_horizon=cast(int, child["horizon"]),
    )
    expected_artifact = forecast_path.parent / "forecast"
    if pathlib.Path(forecast.path).resolve() != expected_artifact.resolve():
        raise ForecastCampaignError(f"child forecast artifact location differs: {child['id']}")
    if (
        not _is_sha256(result["forecast_artifact_sha256"])
        or directory_digest(expected_artifact, suffixes=(".parquet",))
        != result["forecast_artifact_sha256"]
    ):
        raise ForecastCampaignError(f"child forecast artifact changed: {child['id']}")
    checkpoint = result["checkpoint"]
    if checkpoint is not None:
        if type(checkpoint) is not dict or set(checkpoint) != {"path", "sha256"}:
            raise ForecastCampaignError(f"child checkpoint evidence is invalid: {child['id']}")
        checkpoint_path = _run_checkpoint_path(
            run_path,
            cast(str, child["model_id"]),
            cast(str, checkpoint["path"]),
        )
        if not _is_sha256(checkpoint["sha256"]):
            raise ForecastCampaignError(f"child checkpoint changed: {child['id']}")
        try:
            raw_checkpoint = checkpoint_path.read_bytes()
        except OSError as error:
            raise ForecastCampaignError(f"child checkpoint is unreadable: {child['id']}") from error
        if hashlib.sha256(raw_checkpoint).hexdigest() != checkpoint["sha256"]:
            raise ForecastCampaignError(f"child checkpoint changed: {child['id']}")
        if result["adapter"] in _FIXED_BUDGET_ADAPTERS:
            try:
                import torch

                saved_checkpoint = torch.load(
                    io.BytesIO(raw_checkpoint), map_location="cpu", weights_only=True
                )
            except (
                ImportError, OSError, RuntimeError, ValueError, pickle.UnpicklingError
            ) as error:
                raise ForecastCampaignError(
                    f"child checkpoint is invalid: {child['id']}"
                ) from error
            _verify_revision_checkpoint_binding(
                saved_checkpoint, resolved, child, observed_parameters, training_path,
                cast(str, result["training_data_sha256"]),
            )
    elif result["adapter"] in _FIXED_BUDGET_ADAPTERS:
        raise ForecastCampaignError(f"revision child checkpoint is missing: {child['id']}")
    elif list((run_path / "attempts" / cast(str, child["model_id"])).glob("*/model/model.pt")):
        raise ForecastCampaignError(f"child checkpoint evidence is missing: {child['id']}")
    score = result["score"]
    comparison_sha = result["comparison_sha256"]
    if require_score:
        if not is_finite_real(score) or float(score) < 0 or not _is_sha256(comparison_sha):
            raise ForecastCampaignError(f"required validation child is unscored: {child['id']}")
        comparison_path = run_path / "evaluation" / "candidate-comparison.json"
        if _sha_file(comparison_path) != comparison_sha:
            raise ForecastCampaignError(f"child comparison changed: {child['id']}")
        fixed = _fixed_evaluation(resolved, "validation",
            cast(str, child["origin"]), cast(int, child["horizon"]),
        )
        if fixed is not None:
            comparison = cast(dict, load_exact_json(comparison_path))
            if (comparison.get("evaluation_contract") != fixed
                    or comparison.get("comparison_kind") != "fixed_benchmark"):
                raise ForecastCampaignError("child did not score its frozen benchmark grid/metric")
            metric = cast(dict, resolved["selection"])["metric"]
            if _score_value(comparison, metric, cast(str, child["model_id"])) != score:
                raise ForecastCampaignError("child score differs from verified comparison")
    elif score is not None or comparison_sha is not None:
        raise ForecastCampaignError(
            f"non-validation child unexpectedly contains a score: {child['id']}"
        )
    parent = result["parent"]
    parent_stage = child.get("parent_stage")
    if parent_stage is None:
        if parent is not None:
            raise ForecastCampaignError(f"fresh child unexpectedly has a parent: {child['id']}")
    else:
        if parent_stage == child["id"]:
            raise ForecastCampaignError(f"child cannot name itself as its parent: {child['id']}")
        try:
            parent_child = next(
                item
                for item in cast(list[dict[str, object]], resolved["children"])
                if item["id"] == parent_stage
            )
        except StopIteration as error:
            raise ForecastCampaignError(
                f"child parent stage is not declared: {child['id']}"
            ) from error
        parent_result_path = _completed_output(
            campaign_root, cast(str, parent_stage), "child-result.json"
        )
        parent_result = _validated_child_result(
            resolved,
            parent_child,
            parent_result_path,
            require_score=parent_child["kind"] == "validation",
        )
        expected_parent = _parent_checkpoint_identity(cast(str, parent_stage), parent_result)
        if not _same_exact(parent, expected_parent):
            raise ForecastCampaignError(
                f"child parent evidence differs from its governed parent: {child['id']}"
            )
    selection_lock_sha = result["selection_lock_sha256"]
    if child["kind"] == "final_fit":
        if selection_lock_sha is not None and not _is_sha256(selection_lock_sha):
            raise ForecastCampaignError("final fit selection-lock identity is invalid")
    elif selection_lock_sha is not None:
        raise ForecastCampaignError(f"non-final child has selection authorization: {child['id']}")
    return result


def _run_generic_child(
    resolved: dict[str, object],
    child: dict[str, object],
    *,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
    evaluate: bool,
    fixed_update_budget: int | None = None,
) -> dict[str, object]:
    candidates = _resolved_catalog_candidates(resolved)
    _require_separate_long_revision_runs(
        tuple(
            candidates[cast(str, model["model_id"])].adapter
            for model in cast(list[dict[str, object]], resolved["models"])
        ),
        tuple(
            cast(int, cast(dict[str, object], budget)["maximum_epochs"])
            for budget in cast(dict[str, dict[str, object]], resolved["budgets"]).values()
        ),
    )
    source_record = cast(dict[str, object], resolved["source"])
    data = cast(dict[str, object], resolved["data"])
    data_record = cast(dict[str, object], data["selection"])
    contract_path = _root() / cast(str, source_record["dataset_contract"])
    training_path = attempt_dir / "training-panel"
    input_progress = PipelineProgress(
        "Governed data preparation", 2 if evaluate else 1, phase_key="input"
    )
    with input_progress.step("Authenticate and materialize the training slice"):
        source_path = _verify_data(data_record, "selection data")
        training_digest = _materialize_training_panel(
            source_path,
            training_path,
            contract_path=contract_path,
            origin=cast(str, child["origin"]),
            horizon=cast(int, child["horizon"]),
            training_start=cast(
                str | None, cast(dict[str, object], resolved["temporal"])["training_start"]
            ),
            expected_source_sha256=cast(str, data_record["sha256"]),
        )
    request_path = attempt_dir / "child-request.yaml"
    _write_child_request(request_path, resolved, child, evaluation=evaluate)
    parameter_resolution = _child_parameter_resolution(
        resolved, cast(str, child["candidate_id"]), campaign_root
    )
    parameter_path = attempt_dir / "parameter-resolution.json"
    parameter_path.write_text(_json(parameter_resolution), encoding="utf-8")
    parameter_sha = parameter_resolution_sha256(parameter_resolution)
    run_root = attempt_dir / "run"
    budget = cast(dict[str, object], child["budget"])
    compute = cast(dict[str, object], resolved["compute"])
    command = [
        sys.executable,
        "-m", "scripts.run_autoforecast",
        str(request_path),
        "--data-path",
        str(training_path),
        "--data-snapshot-id",
        f"{data_record['snapshot_id']}-{child['origin']}",
        "--data-reference",
        f"{str(data_record['reference']).rstrip('/')}"
        f"/__autoforecast_slice__/through-{child['origin']}",
        "--output-root",
        str(run_root),
        "--epochs",
        str(budget["maximum_epochs"]),
        "--seed",
        str(child["seed"]),
        "--timeout-seconds",
        str(budget["timeout_seconds"]),
        "--patience",
        str(budget["patience"]),
        "--device",
        str(compute["device"]),
        "--precision",
        str(compute.get("precision", "32-true")),
        "--batch-size",
        str(compute["batch_size"]),
        "--early-stopping-absolute-min-delta",
        str(budget["early_stopping_absolute_min_delta"]),
        "--early-stopping-relative-min-delta",
        str(budget["early_stopping_relative_min_delta"]),
        "--no-notifications",
        "--parameter-resolution",
        str(parameter_path),
        "--parameter-resolution-sha256",
        parameter_sha,
    ]
    if evaluate and "benchmark" in resolved:
        fixed = _fixed_evaluation(resolved, "validation",
            cast(str, child["origin"]), cast(int, child["horizon"]),
        )
        contract_file = attempt_dir / "evaluation-contract.json"
        contract_file.write_text(_json(fixed))
        command.extend(["--evaluation-contract", str(contract_file),
                        "--evaluation-contract-sha256", _sha_file(contract_file)])
    command.extend(project_arguments())
    if compute["gradient_microbatch_size"] is not None:
        command.extend(["--gradient-microbatch-size", str(compute["gradient_microbatch_size"])])
    fixed_budget_adapter = child["adapter"] in _FIXED_BUDGET_ADAPTERS
    if child["kind"] == "validation" and fixed_budget_adapter and evaluate:
        command.extend(["--training-mode", "fixed_validation"])
    elif child["kind"] == "final_fit" and fixed_budget_adapter:
        if type(fixed_update_budget) is not int or fixed_update_budget < 1:
            raise ForecastCampaignError(
                "fixed-budget final fit requires the locked positive optimizer-update budget"
            )
        command.extend(
            [
                "--training-mode",
                "fixed_refit",
                "--fixed-update-budget",
                str(fixed_update_budget),
            ]
        )
    if evaluate:
        # Selection training ends at the public cutoff, but its validation labels live in the
        # separately authenticated final-training snapshot.  Reusing ``source_path`` here
        # silently produced 28 null target days and then retrained the same doomed child.
        with input_progress.step("Authenticate and materialize the validation slice"):
            evaluation_record = cast(dict[str, object], data["final_training"])
            evaluation_source_path = _verify_data(evaluation_record, "final training data")
            temporal = cast(dict[str, object], resolved["temporal"])
            target_end = dt.date.fromisoformat(cast(str, child["origin"])) + dt.timedelta(
                days=cast(int, child["horizon"]) * cast(int, temporal["step_days"])
            )
            evaluation_path = attempt_dir / "evaluation-panel"
            _materialize_panel_range(
                evaluation_source_path,
                evaluation_path,
                contract_path=contract_path,
                start=cast(str | None, temporal["training_start"]),
                end=target_end.isoformat(),
                expected_source_sha256=cast(str, evaluation_record["sha256"]),
            )
        command.extend(
            [
                "--evaluation-data-path",
                str(evaluation_path),
                "--evaluation-data-snapshot-id",
                f"{evaluation_record['snapshot_id']}-{child['id']}-evaluation",
                "--evaluation-data-reference",
                f"{str(evaluation_record['reference']).rstrip('/')}"
                f"/__autoforecast_slice__/through-{target_end.isoformat()}",
            ]
        )
    parent_stage = child.get("parent_stage")
    parent_identity = None
    if parent_stage is not None:
        parent_result_path = _completed_output(
            campaign_root, cast(str, parent_stage), "child-result.json"
        )
        parent_child = next(
            item
            for item in cast(list[dict[str, object]], resolved["children"])
            if item["id"] == parent_stage
        )
        parent_result = _validated_child_result(
            resolved,
            parent_child,
            parent_result_path,
            require_score=False,
        )
        parent_identity = _parent_checkpoint_identity(cast(str, parent_stage), parent_result)
        checkpoint = {
            "path": parent_identity["path"],
            "sha256": parent_identity["sha256"],
        }
        checkpoint_path = pathlib.Path(cast(str, checkpoint["path"]))
        command.extend(
            [
                "--parent-checkpoint",
                str(checkpoint_path),
                "--parent-checkpoint-sha256",
                cast(str, checkpoint["sha256"]),
            ]
        )
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=_root(), pass_fds=inherited_training_lease_pass_fds()
    )
    peak = _peak_rss(process)
    return_code = process.wait()
    duration = time.monotonic() - started
    if return_code != 0:
        raise ForecastCampaignError(f"governed child run exited with status {return_code}")
    runs = _published_child_run_groups(run_root)
    if len(runs) != 1:
        raise ForecastCampaignError("governed child did not publish exactly one run group")
    run = _run_group_path(attempt_dir, runs[0], "governed child run path")
    summary_path = run / "run-summary.json"
    summary = json.loads(summary_path.read_text())
    outcomes = summary.get("models")
    if (
        type(outcomes) is not list
        or len(outcomes) != 1
        or outcomes[0].get("model_id") != child["model_id"]
        or outcomes[0].get("status") != "completed"
    ):
        raise ForecastCampaignError("governed child summary is not one completed model")
    forecast_paths = list(
        (run / "attempts" / cast(str, child["model_id"])).glob("*/evaluation/forecast.json")
    )
    if len(forecast_paths) != 1:
        raise ForecastCampaignError("governed child has no unique forecast record")
    forecast, forecast_record_sha = _verified_forecast(
        forecast_paths[0],
        expected_adapter=cast(str, child["adapter"]),
        expected_origin=cast(str, child["origin"]),
        expected_horizon=cast(int, child["horizon"]),
    )
    if pathlib.Path(forecast.path).resolve() != (forecast_paths[0].parent / "forecast").resolve():
        raise ForecastCampaignError("governed child forecast artifact escaped its run bundle")
    run_parameter_paths = list(
        (run / "attempts" / cast(str, child["model_id"])).glob(
            "*/metadata/parameter-resolution.json"
        )
    )
    run_parameter = (
        _load_hashed_json(run_parameter_paths[0], parameter_sha, "governed child parameters")
        if len(run_parameter_paths) == 1
        else None
    )
    if run_parameter is None or not _same_exact(run_parameter, parameter_resolution):
        raise ForecastCampaignError("governed child did not retain its parameter resolution")
    result: dict[str, object] = {
        "schema": (
            CHILD_RESULT_PRECISION_SCHEMA
            if cast(int, resolved.get("request_schema_version", 3)) >= 6 else CHILD_RESULT_SCHEMA
        ),
        "status": "completed",
        "kind": child["kind"],
        "child_id": child["id"],
        "candidate_id": child["candidate_id"],
        "model_id": child["model_id"],
        "variant_id": child["variant_id"],
        "adapter": child["adapter"],
        "parameter_mode": child["parameter_mode"],
        "parameter_resolution": parameter_resolution,
        "parameter_resolution_sha256": parameter_sha,
        "initialization": child["initialization"],
        "seed": child["seed"],
        "origin": child["origin"],
        "horizon": child["horizon"],
        "budget": budget,
        "training_data_sha256": training_digest,
        "parent": parent_identity,
        "run_path": str(run),
        "run_summary_sha256": _sha_file(summary_path),
        "forecast_record_path": str(forecast_paths[0]),
        "forecast_record_sha256": forecast_record_sha,
        "forecast_artifact_sha256": directory_digest(forecast.path, suffixes=(".parquet",)),
        "runtime_seconds": duration,
        "peak_rss_bytes": peak,
        "replay": {
            "status": "verified_artifact_bytes",
            "scope": "sealed forecast manifest and Parquet; not an inference re-execution",
        },
        "checkpoint": None,
        "score": None,
        "comparison_sha256": None,
        "selection_lock_sha256": None,
    }
    if child["adapter"] in _ROLLING_BUDGET_ADAPTERS:
        result["training_metrics_sha256"] = _sha_file(
            _rolling_metrics_path(run, cast(str, child["model_id"]))
        )
    if cast(int, resolved.get("request_schema_version", 3)) >= 6:
        control_paths = list(
            (run / "attempts" / cast(str, child["model_id"])).glob(
                "*/metadata/training-controls.json"
            )
        )
        if len(control_paths) != 1:
            raise ForecastCampaignError("governed child has no unique training controls")
        result["training_controls_sha256"] = _sha_file(control_paths[0])
    checkpoints = list((run / "attempts" / cast(str, child["model_id"])).glob("*/model/model.pt"))
    if checkpoints:
        if len(checkpoints) != 1:
            raise ForecastCampaignError("governed child has multiple model checkpoints")
        result["checkpoint"] = {"path": str(checkpoints[0]), "sha256": _sha_file(checkpoints[0])}
    if evaluate:
        comparison_path = run / "evaluation/candidate-comparison.json"
        comparison = cast(dict[str, object], json.loads(comparison_path.read_text()))
        metric = cast(str, cast(dict[str, object], resolved["selection"])["metric"])
        result["score"] = _score_value(comparison, metric, cast(str, child["model_id"]))
        result["comparison_sha256"] = _sha_file(comparison_path)
    result_path = attempt_dir / "child-result.json"
    result_path.write_text(_json(result), encoding="utf-8")
    return _validated_child_result(
        resolved,
        child,
        result_path,
        require_score=evaluate,
    )


def run_child(
    resolved_path: pathlib.Path,
    resolved_sha: str,
    *,
    child_id: str,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
) -> None:
    """Execute one pre-resolved initialization-parent or validation child."""
    resolved = _load_resolved(resolved_path, resolved_sha)
    matches = [
        item
        for item in cast(list[dict[str, object]], resolved["children"])
        if item["id"] == child_id
    ]
    if len(matches) != 1:
        raise ForecastCampaignError(f"resolved campaign has no unique child {child_id!r}")
    child = matches[0]
    _run_generic_child(
        resolved,
        child,
        campaign_root=campaign_root,
        attempt_dir=attempt_dir,
        evaluate=(
            child["kind"] == "validation" and resolved.get("purpose") != "capacity_diagnostic"
        ),
    )


def _selection_statistics(
    resolved: dict[str, object],
    results: list[dict[str, object]],
    campaign_root: pathlib.Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(
            (cast(str, result["candidate_id"]), cast(str, result["initialization"])), []
        ).append(result)
    expected = len(cast(list[int], resolved["seeds"])) * len(
        cast(list[str], cast(dict[str, object], resolved["temporal"])["validation_origins"])
    )
    candidates: list[dict[str, object]] = []
    for (candidate_id, initialization), items in sorted(grouped.items()):
        model_id = cast(str, items[0]["model_id"])
        variant_id = cast(str, items[0]["variant_id"])
        if len(items) != expected:
            raise ForecastCampaignError(
                f"candidate {candidate_id}/{initialization} has {len(items)} folds; "
                f"expected {expected}"
            )
        scores = [float(cast(float | int, item["score"])) for item in items]
        runtimes = [float(cast(float | int, item["runtime_seconds"])) for item in items]
        rss = [
            cast(int, item["peak_rss_bytes"])
            for item in items
            if item["peak_rss_bytes"] is not None
        ]
        parameter_hashes = {cast(str, item["parameter_resolution_sha256"]) for item in items}
        if len(parameter_hashes) != 1:
            raise ForecastCampaignError(
                f"candidate {candidate_id}/{initialization} used different parameter resolutions"
            )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "initialization": initialization,
                "parameter_mode": items[0]["parameter_mode"],
                "parameter_resolution": items[0]["parameter_resolution"],
                "parameter_resolution_sha256": next(iter(parameter_hashes)),
                "n_children": len(items),
                "scores": scores,
                "mean_score": statistics.fmean(scores),
                "median_score": statistics.median(scores),
                "score_stddev": statistics.pstdev(scores),
                "runtime_seconds_total": sum(runtimes),
                "runtime_seconds_mean": statistics.fmean(runtimes),
                "peak_rss_bytes_max": max(rss) if rss else None,
                "failed_children": 0,
                "replay_statuses": [
                    cast(dict[str, object], item["replay"])["status"] for item in items
                ],
                "child_result_sha256": [
                    _sha_file(
                        _completed_output(
                            campaign_root, cast(str, item["child_id"]), "child-result.json"
                        )
                    )
                    for item in items
                ],
            }
        )
    if not candidates:
        raise ForecastCampaignError("no compatible validation candidate was produced")
    fresh_scores = {
        (item["candidate_id"], item["seed"], item["origin"]): float(
            cast(float | int, item["score"])
        )
        for item in results
        if item["initialization"] == "fresh"
    }
    paired: list[dict[str, object]] = []
    for (candidate_id, initialization), items in sorted(grouped.items()):
        if initialization == "fresh":
            continue
        model_id = cast(str, items[0]["model_id"])
        variant_id = cast(str, items[0]["variant_id"])
        deltas: list[float] = []
        pairs: list[dict[str, object]] = []
        for item in sorted(items, key=lambda value: (value["seed"], value["origin"])):
            key = (candidate_id, item["seed"], item["origin"])
            if key not in fresh_scores:
                raise ForecastCampaignError(
                    f"warm candidate {candidate_id}/{initialization} has no paired fresh child "
                    f"{key}"
                )
            delta = float(cast(float | int, item["score"])) - fresh_scores[key]
            deltas.append(delta)
            pairs.append(
                {
                    "seed": item["seed"],
                    "origin": item["origin"],
                    "warm_minus_fresh": delta,
                }
            )
        paired.append(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                "variant_id": variant_id,
                "initialization": initialization,
                "pairs": pairs,
                "mean_delta": statistics.fmean(deltas),
                "median_delta": statistics.median(deltas),
                "delta_stddev": statistics.pstdev(deltas),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["mean_score"],
            item["score_stddev"],
            item["model_id"],
            item["variant_id"],
            item["initialization"],
        ),
    )
    return candidates, paired, ranked


def _locked_selected_update_budget(
    resolved: dict[str, object],
    chosen: dict[str, object],
    results: list[dict[str, object]],
) -> int | None:
    """Lock a validation-selected optimizer frontier for trainable revision models."""
    matches = [
        result
        for result in results
        if result.get("candidate_id") == chosen["candidate_id"]
        and result.get("initialization") == chosen["initialization"]
    ]
    if resolved.get("purpose") == "single_model_benchmark" and len(matches) != 1:
        raise ForecastCampaignError("fixed-budget selection has no unique child budget")
    adapters = {result.get("adapter") for result in matches}
    if not adapters:
        children = cast(list[dict[str, object]], resolved.get("children", []))
        if any(
            child.get("kind") == "validation"
            and child.get("candidate_id") == chosen["candidate_id"]
            and child.get("initialization") == chosen["initialization"]
            and child.get("adapter") in _FIXED_BUDGET_ADAPTERS
            for child in children
        ):
            raise ForecastCampaignError("fixed-budget selection has no child budget")
        return None
    if not adapters <= _FIXED_BUDGET_ADAPTERS:
        return None
    if len(adapters) != 1 or (
        resolved.get("purpose") == "single_model_benchmark" and len(matches) != 1
    ) or (bool(adapters & {"tft_q", "tft_hnb", }) and len(matches) != 1):
        raise ForecastCampaignError("fixed-budget selection has no unique child budget")
    selection = resolved.get("selection")
    expected_metric = selection.get("metric") if type(selection) is dict else None
    if type(expected_metric) is not str:
        raise ForecastCampaignError("fixed-budget selection metric is missing")
    # MAE/WAPE rank the externally scored candidates; the rolling trainer still
    # selects its optimizer frontier on RMSE when no RMSE/WRMSSE metric was requested.
    rolling_frontier_metric = (
        expected_metric if expected_metric in {"rmse", "wrmsse"} else "rmse"
    )
    compute = resolved.get("compute")
    batch_size = compute.get("batch_size") if type(compute) is dict else None
    selected_updates: list[int] = []
    for result in matches:
        parameters = result.get("parameter_resolution")
        effective = parameters.get("effective_parameters") if type(parameters) is dict else None
        loss_kind = effective.get("model.loss_kind") if type(effective) is dict else None
        is_rolling = result.get("adapter") in _ROLLING_BUDGET_ADAPTERS
        validation_every = 1
        if is_rolling:
            model_id = result.get("model_id")
            if type(model_id) is not str:
                raise ForecastCampaignError("rolling model identity is missing")
            candidate = _campaign_candidate(resolved, model_id)
            try:
                raw_config = (
                    loads_exact_json(candidate.implementation_config_bytes)
                    if candidate.implementation_config_bytes is not None else None
                )
            except ExactJSONError as error:
                raise ForecastCampaignError(
                    "governed rolling validation config is invalid"
                ) from error
            if type(raw_config) is not dict:
                raise ForecastCampaignError("governed rolling validation config is missing")
            validation_every = (
                effective.get(
                    "training.validation_every_epochs", raw_config.get("validation_every_epochs")
                )
                if type(effective) is dict else raw_config.get("validation_every_epochs")
            )
            if type(validation_every) is not int or validation_every < 1:
                raise ForecastCampaignError("governed rolling validation cadence is invalid")
        selected = (
            _selected_update_budget(
                result, rolling_frontier_metric, batch_size, loss_kind,
                expected_validation_every=validation_every,
            )
            if is_rolling else _selected_update_budget(
                result, expected_metric, batch_size, loss_kind
            )
        )
        if type(selected) is not int or selected < 1:
            raise ForecastCampaignError("fixed-budget selection has no update witness")
        if is_rolling:
            origins = _governed_rolling_selection_origins(resolved, result)
            masks = _rolling_selection_objective_masks(resolved, result, origins)
            _verify_rolling_selection_journal(
                result, rolling_frontier_metric, cast(int, batch_size), masks,
                expected_validation_every=validation_every,
            )
        selected_updates.append(selected)
    # Multi-origin research validation can select different frontiers. Use the lower
    # median so one extreme run cannot inflate the final training cost.
    return statistics.median_low(selected_updates)


def aggregate_selection(
    resolved_path: pathlib.Path,
    resolved_sha: str,
    *,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
) -> None:
    """Require every compatible validation child, aggregate paired evidence, and seal selection."""
    resolved = _load_resolved(resolved_path, resolved_sha)
    if resolved.get("purpose") == "capacity_diagnostic":
        raise ForecastCampaignError("capacity diagnostic children may not enter model selection")
    validations = [
        item
        for item in cast(list[dict[str, object]], resolved["children"])
        if item["kind"] == "validation"
    ]
    results: list[dict[str, object]] = []
    for child in validations:
        path = _completed_output(campaign_root, cast(str, child["id"]), "child-result.json")
        results.append(
            _validated_child_result(
                resolved,
                child,
                path,
                require_score=True,
            )
        )
    candidates, paired, ranked = _selection_statistics(resolved, results, campaign_root)
    chosen = ranked[0]
    selected_update_budget = _locked_selected_update_budget(resolved, chosen, results)
    summary = {
        "schema": VALIDATION_SUMMARY_SCHEMA,
        "resolved_sha256": resolved_sha,
        "selection": resolved["selection"],
        "capabilities": resolved["initialization_capabilities"],
        "candidates": candidates,
        "paired_initialization_deltas": paired,
        "ranking": [
            {
                "candidate_id": item["candidate_id"],
                "model_id": item["model_id"],
                "variant_id": item["variant_id"],
                "initialization": item["initialization"],
            }
            for item in ranked
        ],
    }
    summary_path = attempt_dir / "validation-summary.json"
    summary_path.write_text(_json(summary), encoding="utf-8")
    lock = {
        "schema": SELECTION_LOCK_SCHEMA,
        "status": "locked",
        "resolved_sha256": resolved_sha,
        "source": resolved["source"],
        "selection_data": cast(dict[str, object], resolved["data"])["selection"],
        **({"benchmark": resolved["benchmark"]} if "benchmark" in resolved else {}),
        "metric": cast(dict[str, object], resolved["selection"])["metric"],
        "direction": "minimize",
        "tie_break": list(_TIE_BREAK),
        "validation_origins": cast(dict[str, object], resolved["temporal"])["validation_origins"],
        "validation_horizon": cast(dict[str, object], resolved["temporal"])["validation_horizon"],
        "validation_summary_sha256": _sha_file(summary_path),
        "chosen": {
            "candidate_id": chosen["candidate_id"],
            "model_id": chosen["model_id"],
            "variant_id": chosen["variant_id"],
            "initialization": chosen["initialization"],
            "mean_score": chosen["mean_score"],
            "score_stddev": chosen["score_stddev"],
            "parameter_mode": chosen["parameter_mode"],
            "parameter_resolution": chosen["parameter_resolution"],
            "parameter_resolution_sha256": chosen["parameter_resolution_sha256"],
            "final_seed": cast(list[int], resolved["seeds"])[0],
            "final_fit_budget": cast(dict[str, object], resolved["budgets"])["final_fit"],
            "selected_update_budget": selected_update_budget,
        },
    }
    (attempt_dir / "selection-lock.json").write_text(_json(lock), encoding="utf-8")


def _validated_selection_evidence(
    resolved: dict[str, object],
    resolved_sha: str,
    campaign_root: pathlib.Path,
) -> tuple[dict[str, object], dict[str, object], pathlib.Path, pathlib.Path]:
    summary_path = _completed_output(
        campaign_root, "aggregate_selection", "validation-summary.json"
    )
    lock_path = _completed_output(campaign_root, "aggregate_selection", "selection-lock.json")
    validations = [
        item
        for item in cast(list[dict[str, object]], resolved["children"])
        if item["kind"] == "validation"
    ]
    results = [
        _validated_child_result(
            resolved,
            child,
            _completed_output(campaign_root, cast(str, child["id"]), "child-result.json"),
            require_score=True,
        )
        for child in validations
    ]
    candidates, paired, ranked = _selection_statistics(resolved, results, campaign_root)
    expected_summary = {
        "schema": VALIDATION_SUMMARY_SCHEMA,
        "resolved_sha256": resolved_sha,
        "selection": resolved["selection"],
        "capabilities": resolved["initialization_capabilities"],
        "candidates": candidates,
        "paired_initialization_deltas": paired,
        "ranking": [
            {
                "candidate_id": item["candidate_id"],
                "model_id": item["model_id"],
                "variant_id": item["variant_id"],
                "initialization": item["initialization"],
            }
            for item in ranked
        ],
    }
    try:
        summary = load_exact_json(summary_path)
        lock = load_exact_json(lock_path)
    except ExactJSONError as error:
        raise ForecastCampaignError(f"selection evidence is invalid JSON: {error}") from error
    if not _same_exact(summary, expected_summary):
        raise ForecastCampaignError("validation summary differs from recomputed child evidence")
    chosen = ranked[0]
    selected_update_budget = _locked_selected_update_budget(resolved, chosen, results)
    expected_lock = {
        "schema": SELECTION_LOCK_SCHEMA,
        "status": "locked",
        "resolved_sha256": resolved_sha,
        "source": resolved["source"],
        "selection_data": cast(dict[str, object], resolved["data"])["selection"],
        **({"benchmark": resolved["benchmark"]} if "benchmark" in resolved else {}),
        "metric": cast(dict[str, object], resolved["selection"])["metric"],
        "direction": "minimize",
        "tie_break": list(_TIE_BREAK),
        "validation_origins": cast(dict[str, object], resolved["temporal"])["validation_origins"],
        "validation_horizon": cast(dict[str, object], resolved["temporal"])["validation_horizon"],
        "validation_summary_sha256": _sha_file(summary_path),
        "chosen": {
            "candidate_id": chosen["candidate_id"],
            "model_id": chosen["model_id"],
            "variant_id": chosen["variant_id"],
            "initialization": chosen["initialization"],
            "mean_score": chosen["mean_score"],
            "score_stddev": chosen["score_stddev"],
            "parameter_mode": chosen.get("parameter_mode"),
            "parameter_resolution": chosen.get("parameter_resolution"),
            "parameter_resolution_sha256": chosen.get("parameter_resolution_sha256"),
            "final_seed": cast(list[int], resolved["seeds"])[0],
            "final_fit_budget": cast(dict[str, object], resolved["budgets"])["final_fit"],
            "selected_update_budget": selected_update_budget,
        },
    }
    if not _same_exact(lock, expected_lock):
        raise ForecastCampaignError("selection lock differs from recomputed validation evidence")
    return (
        cast(dict[str, object], lock),
        cast(dict[str, object], summary),
        lock_path,
        summary_path,
    )


def run_final_fit(
    resolved_path: pathlib.Path,
    resolved_sha: str,
    *,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
) -> None:
    """Fit the sealed winner on all declared pre-test training data, without final labels."""
    resolved = _load_resolved(resolved_path, resolved_sha)
    if resolved.get("purpose") == "capacity_diagnostic":
        raise ForecastCampaignError("capacity diagnostics have no final fit")
    lock, _summary, lock_path, _summary_path = _validated_selection_evidence(
        resolved, resolved_sha, campaign_root
    )
    chosen = cast(dict[str, object], lock["chosen"])
    candidate_id = cast(str, chosen["candidate_id"])
    model_id = cast(str, chosen["model_id"])
    initialization = cast(str, chosen["initialization"])
    candidate = next(
        item
        for item in cast(list[dict[str, object]], resolved["initialization_capabilities"])
        if item["candidate_id"] == candidate_id and item["initialization"] == initialization
    )
    parent_stage = None
    if initialization == "weights_only_finetune":
        parent_stage = next(
            item["id"]
            for item in cast(list[dict[str, object]], resolved["children"])
            if item["kind"] == "initialization_parent"
            and item["candidate_id"] == candidate_id
            and item["seed"] == chosen["final_seed"]
        )
    child = {
        "id": "final_fit",
        "kind": "final_fit",
        "candidate_id": candidate_id,
        "model_id": model_id,
        "variant_id": chosen["variant_id"],
        "adapter": candidate["adapter"],
        "parameter_mode": _parameter_profile(resolved, candidate_id)["mode"],
        "initialization": initialization,
        "seed": chosen["final_seed"],
        "origin": cast(dict[str, object], resolved["temporal"])["final_train_cutoff"],
        "horizon": cast(dict[str, object], resolved["temporal"])["final_test_horizon"],
        "parent_stage": parent_stage,
        "budget": chosen["final_fit_budget"],
    }
    data = cast(dict[str, object], resolved["data"])
    selection = data["selection"]
    data["selection"] = data["final_training"]
    try:
        result = _run_generic_child(
            resolved,
            child,
            campaign_root=campaign_root,
            attempt_dir=attempt_dir,
            evaluate=False,
            fixed_update_budget=cast(int | None, chosen.get("selected_update_budget")),
        )
    finally:
        data["selection"] = selection
    if result.get("parameter_resolution_sha256") != chosen.get("parameter_resolution_sha256"):
        raise ForecastCampaignError("final fit parameter identity differs from selection lock")
    selected_update_budget = chosen.get("selected_update_budget")
    if selected_update_budget is not None:
        if type(selected_update_budget) is not int or selected_update_budget < 1:
            raise ForecastCampaignError("selection lock has an invalid update budget")
        compute = cast(dict[str, object], resolved["compute"])
        _verify_fixed_refit_budget(
            result, selected_update_budget, batch_size=cast(int, compute["batch_size"]),
            resolved=resolved,
        )
    result["selection_lock_sha256"] = _sha_file(lock_path)
    (attempt_dir / "child-result.json").write_text(_json(result), encoding="utf-8")


def _resolved_final_child(
    resolved: dict[str, object], lock: dict[str, object]
) -> dict[str, object]:
    chosen = cast(dict[str, object], lock["chosen"])
    temporal = cast(dict[str, object], resolved["temporal"])
    capability = next(
        item
        for item in cast(list[dict[str, object]], resolved["initialization_capabilities"])
        if item["candidate_id"] == chosen["candidate_id"]
        and item["initialization"] == chosen["initialization"]
        and item["status"] == "supported"
    )
    parent_stage = None
    if chosen["initialization"] == "weights_only_finetune":
        parent_stage = next(
            item["id"]
            for item in cast(list[dict[str, object]], resolved["children"])
            if item["kind"] == "initialization_parent"
            and item["candidate_id"] == chosen["candidate_id"]
            and item["seed"] == chosen["final_seed"]
        )
    return {
        "id": "final_fit",
        "kind": "final_fit",
        "candidate_id": chosen["candidate_id"],
        "model_id": chosen["model_id"],
        "variant_id": chosen["variant_id"],
        "adapter": capability["adapter"],
        "parameter_mode": _parameter_profile(resolved, cast(str, chosen["candidate_id"]))["mode"],
        "initialization": chosen["initialization"],
        "seed": chosen["final_seed"],
        "origin": temporal["final_train_cutoff"],
        "horizon": temporal["final_test_horizon"],
        "parent_stage": parent_stage,
        "budget": chosen["final_fit_budget"],
    }


def _validated_locked_final_result(
    resolved: dict[str, object], lock: dict[str, object],
    lock_path: pathlib.Path, result_path: pathlib.Path,
) -> dict[str, object]:
    """Recheck the sealed refit frontier at every final-result consumption boundary."""
    result = _validated_child_result(
        resolved, _resolved_final_child(resolved, lock), result_path,
        require_score=False,
    )
    if result.get("selection_lock_sha256") != _sha_file(lock_path):
        raise ForecastCampaignError("final fit was not authorized by the current selection lock")
    selected = cast(dict[str, object], lock["chosen"]).get("selected_update_budget")
    if selected is not None:
        if type(selected) is not int or selected < 1:
            raise ForecastCampaignError("selection lock has an invalid update budget")
        compute = cast(dict[str, object], resolved["compute"])
        _verify_fixed_refit_budget(
            result, selected, batch_size=cast(int, compute["batch_size"]),
            resolved=resolved,
        )
    return result


def score_final(
    resolved_path: pathlib.Path,
    resolved_sha: str,
    score_input_path: pathlib.Path,
    score_input_sha: str,
    *,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
) -> None:
    """Open the separate final labels only after selection and final fit are sealed."""
    resolved = _load_resolved(resolved_path, resolved_sha)
    if resolved.get("purpose") == "capacity_diagnostic":
        raise ForecastCampaignError("capacity diagnostics have no final score")
    score_input = _load_exact(score_input_path, score_input_sha, SCORE_INPUT_SCHEMA)
    expected_score_fields = {
        "schema",
        "campaign_request_sha256",
        "resolved_sha256",
        "final_score_data",
        "metric",
        "missing_actual",
        "evidence_role",
        "origin",
        "horizon",
    }
    score_data = _exact(
        score_input.get("final_score_data"),
        {"path", "snapshot_id", "reference", "sha256"},
        "final score data",
    )
    temporal = cast(dict[str, object], resolved["temporal"])
    selection = cast(dict[str, object], resolved["selection"])
    source = cast(dict[str, object], resolved["source"])
    if (
        set(score_input) != expected_score_fields
        or score_input["resolved_sha256"] != resolved_sha
        or score_input["campaign_request_sha256"] != source["request_sha256"]
        or score_input["metric"] != selection["metric"]
        or score_input["missing_actual"] != selection["missing_actual"]
        or type(score_input["evidence_role"]) is not str
        or not cast(str, score_input["evidence_role"]).strip()
        or score_input["origin"] != temporal["final_train_cutoff"]
        or score_input["horizon"] != temporal["final_test_horizon"]
        or not _is_sha256(score_data["sha256"])
        or type(score_data["path"]) is not str
        or type(score_data["snapshot_id"]) is not str
        or not cast(str, score_data["snapshot_id"]).strip()
        or type(score_data["reference"]) is not str
        or not cast(str, score_data["reference"]).strip()
        or score_data["sha256"]
        in {
            cast(dict[str, object], cast(dict[str, object], resolved["data"])["selection"])[
                "sha256"
            ],
            cast(dict[str, object], cast(dict[str, object], resolved["data"])["final_training"])[
                "sha256"
            ],
        }
    ):
        raise ForecastCampaignError("final score input is not the separate resolved test boundary")
    lock, _summary, lock_path, _summary_path = _validated_selection_evidence(
        resolved, resolved_sha, campaign_root
    )
    final_result_path = _completed_output(campaign_root, "final_fit", "child-result.json")
    final_result = _validated_locked_final_result(
        resolved, lock, lock_path, final_result_path
    )
    forecast_record_path = pathlib.Path(cast(str, final_result["forecast_record_path"]))
    forecast_record_sha = final_result["forecast_record_sha256"]
    if not _is_sha256(forecast_record_sha):
        raise ForecastCampaignError("sealed final forecast record identity is invalid")
    forecast, _ = _verified_forecast(
        forecast_record_path,
        expected_sha256=cast(str, forecast_record_sha),
        expected_adapter=cast(str, final_result["adapter"]),
        expected_origin=cast(str, score_input["origin"]),
        expected_horizon=cast(int, score_input["horizon"]),
    )
    input_progress = PipelineProgress("Sealed test input verification", 2, phase_key="input")
    with input_progress.step("Authenticate the separate test-label snapshot"):
        score_path = _verify_data(score_data, "final score data")
    input_root = attempt_dir / "scoring-inputs"
    input_root.mkdir()
    with input_progress.step("Snapshot the sealed forecast and test labels"):
        forecast_snapshot = _snapshot_parquet_directory(
            pathlib.Path(forecast.path),
            input_root / "forecast",
            expected_sha256=cast(str, final_result["forecast_artifact_sha256"]),
            label="sealed final forecast",
        )
        score_snapshot = _snapshot_parquet_directory(
            score_path,
            input_root / "actuals",
            expected_sha256=cast(str, score_data["sha256"]),
            label="final score data",
        )
    forecast = replace(forecast, path=str(forecast_snapshot))
    panel = load_dataset_contract(_root() / cast(str, source["dataset_contract"])).panel(
        score_snapshot
    )
    profile = resources.forecast_scoring_profile(
        cast(str, score_input["metric"]), (cast(str, final_result["adapter"]),)
    )
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    score_progress = PipelineProgress("Final test scoring", 2, phase_key="score")
    with score_progress.step("Start the bounded Spark scoring runtime"):
        spark = resources.acquire_local_runtime(
            SparkSession, SparkContext, profile=profile,
            app_name="autoforecast_campaign_final_score",
        )
        spark.sparkContext.setLogLevel("ERROR")
    try:
        with score_progress.step("Calculate and publish test WRMSSE", spark=spark):
            score_progress.operation("Join the sealed forecast to test labels and calculate WRMSSE")
            comparison_path = compare_run_forecasts(
                panel,
                {cast(str, final_result["model_id"]): forecast},
                spark,
                missing_actual=cast(Any, score_input["missing_actual"]),
                metric=cast(Any, score_input["metric"]),
                output_dir=attempt_dir / "evaluation",
                fixed_contract=_fixed_evaluation(resolved, "test",
                    cast(str, score_input["origin"]), cast(int, score_input["horizon"])),
            )
    finally:
        with contextlib.suppress(Exception):
            spark.stop()
    comparison = cast(dict[str, object], json.loads(comparison_path.read_text()))
    score = _score_value(
        comparison, cast(str, score_input["metric"]), cast(str, final_result["model_id"])
    )
    if (
        directory_digest(score_snapshot, suffixes=(".parquet",)) != score_data["sha256"]
        or directory_digest(forecast_snapshot, suffixes=(".parquet",))
        != final_result["forecast_artifact_sha256"]
    ):
        shutil.rmtree(attempt_dir / "evaluation", ignore_errors=True)
        raise ForecastCampaignError("stage-owned scoring inputs changed while evaluation ran")
    record = {
        "schema": FINAL_SCORE_SCHEMA,
        "status": "scored",
        "resolved_sha256": resolved_sha,
        "selection_lock_sha256": _sha_file(lock_path),
        "final_fit_result_sha256": _sha_file(final_result_path),
        "score_input_sha256": score_input_sha,
        "score_data": {
            "snapshot_id": score_data["snapshot_id"],
            "reference": score_data["reference"],
            "sha256": score_data["sha256"],
        },
        "forecast_input_artifact_sha256": final_result["forecast_artifact_sha256"],
        "score_input_artifact_sha256": score_data["sha256"],
        "evidence_role": score_input["evidence_role"],
        "candidate_id": final_result["candidate_id"],
        "model_id": final_result["model_id"],
        "variant_id": final_result["variant_id"],
        "parameter_resolution_sha256": final_result["parameter_resolution_sha256"],
        "initialization": final_result["initialization"],
        "origin": score_input["origin"],
        "horizon": score_input["horizon"],
        "metric": score_input["metric"],
        "score": score,
        "comparison_sha256": _sha_file(comparison_path),
    }
    (attempt_dir / "final-score.json").write_text(_json(record), encoding="utf-8")


def replay_report(
    resolved_path: pathlib.Path,
    resolved_sha: str,
    *,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
) -> None:
    """Re-hash all sealed evidence and publish the human and machine campaign report."""
    resolved = _load_resolved(resolved_path, resolved_sha)
    lock, summary, lock_path, summary_path = _validated_selection_evidence(
        resolved, resolved_sha, campaign_root
    )
    final_fit_path = _completed_output(campaign_root, "final_fit", "child-result.json")
    final_score_path = _completed_output(campaign_root, "final_score", "final-score.json")
    final_fit = _validated_locked_final_result(resolved, lock, lock_path, final_fit_path)
    try:
        final_score_value = load_exact_json(final_score_path)
    except ExactJSONError as error:
        raise ForecastCampaignError(f"final score record is invalid JSON: {error}") from error
    final_score_fields = {
        "schema",
        "status",
        "resolved_sha256",
        "selection_lock_sha256",
        "final_fit_result_sha256",
        "score_input_sha256",
        "score_data",
        "forecast_input_artifact_sha256",
        "score_input_artifact_sha256",
        "evidence_role",
        "candidate_id",
        "model_id",
        "variant_id",
        "parameter_resolution_sha256",
        "initialization",
        "origin",
        "horizon",
        "metric",
        "score",
        "comparison_sha256",
    }
    if type(final_score_value) is not dict or set(final_score_value) != final_score_fields:
        raise ForecastCampaignError("final score record fields differ")
    final_score = cast(dict[str, object], final_score_value)
    score_input_path = resolved_path.parent / "score-input.json"
    score_input_sha = final_score["score_input_sha256"]
    if not _is_sha256(score_input_sha):
        raise ForecastCampaignError("final score input identity is invalid")
    score_input = _load_exact(score_input_path, cast(str, score_input_sha), SCORE_INPUT_SCHEMA)
    score_data = _exact(
        score_input.get("final_score_data"),
        {"path", "snapshot_id", "reference", "sha256"},
        "final score replay data",
    )
    reported_score_data = {
        "snapshot_id": score_data["snapshot_id"],
        "reference": score_data["reference"],
        "sha256": score_data["sha256"],
    }
    numeric_score = final_score["score"]
    comparison_path = final_score_path.parent / "evaluation" / "candidate-comparison.json"
    retained_forecast = final_score_path.parent / "scoring-inputs" / "forecast"
    retained_actuals = final_score_path.parent / "scoring-inputs" / "actuals"
    if (
        final_score["schema"] != FINAL_SCORE_SCHEMA
        or final_score["status"] != "scored"
        or final_score["resolved_sha256"] != resolved_sha
        or final_score["selection_lock_sha256"] != _sha_file(lock_path)
        or final_score["final_fit_result_sha256"] != _sha_file(final_fit_path)
        or not _same_exact(final_score["score_data"], reported_score_data)
        or final_score["forecast_input_artifact_sha256"] != final_fit["forecast_artifact_sha256"]
        or final_score["score_input_artifact_sha256"] != score_data["sha256"]
        or final_score["evidence_role"] != score_input["evidence_role"]
        or final_score["candidate_id"] != final_fit["candidate_id"]
        or final_score["model_id"] != final_fit["model_id"]
        or final_score["variant_id"] != final_fit["variant_id"]
        or final_score["parameter_resolution_sha256"] != final_fit["parameter_resolution_sha256"]
        or final_score["initialization"] != final_fit["initialization"]
        or final_score["origin"] != score_input["origin"]
        or final_score["horizon"] != score_input["horizon"]
        or final_score["metric"] != score_input["metric"]
        or not is_finite_real(numeric_score)
        or float(numeric_score) < 0
        or not _is_sha256(final_score["comparison_sha256"])
        or _sha_file(comparison_path) != final_score["comparison_sha256"]
        or directory_digest(retained_forecast, suffixes=(".parquet",))
        != final_score["forecast_input_artifact_sha256"]
        or directory_digest(retained_actuals, suffixes=(".parquet",))
        != final_score["score_input_artifact_sha256"]
    ):
        raise ForecastCampaignError("final score record differs from its sealed evidence")
    comparison = cast(dict[str, object], load_exact_json(comparison_path))
    if _score_value(
        comparison, cast(str, final_score["metric"]), cast(str, final_score["model_id"])
    ) != float(numeric_score):
        raise ForecastCampaignError("final score value differs from its candidate comparison")
    forecast_path = pathlib.Path(cast(str, final_fit["forecast_record_path"]))
    checks = {
        "selection_lock": lock.get("resolved_sha256") == resolved_sha,
        "validation_summary": lock.get("validation_summary_sha256") == _sha_file(summary_path),
        "final_fit_authorization": final_fit.get("selection_lock_sha256") == _sha_file(lock_path),
        "parameter_identity": final_fit.get("parameter_resolution_sha256")
        == cast(dict[str, object], lock["chosen"]).get("parameter_resolution_sha256"),
        "final_score_authorization": final_score.get("selection_lock_sha256")
        == _sha_file(lock_path),
        "final_forecast": final_fit.get("forecast_record_sha256") == _sha_file(forecast_path),
        "final_score_result": final_score.get("final_fit_result_sha256")
        == _sha_file(final_fit_path),
    }
    if not all(checks.values()):
        raise ForecastCampaignError(f"campaign replay verification failed: {checks}")
    replay = {
        "schema": REPLAY_SCHEMA,
        "status": "verified",
        "scope": "journal, selection, final-fit authorization, sealed forecast, and score lineage",
        "checks": checks,
    }
    (attempt_dir / "replay-verification.json").write_text(_json(replay), encoding="utf-8")
    report = {
        "schema": REPORT_SCHEMA,
        "campaign": resolved["name"],
        "resolved_sha256": resolved_sha,
        "source": resolved["source"],
        "data": {
            "selection": cast(dict[str, object], resolved["data"])["selection"],
            "final_training": cast(dict[str, object], resolved["data"])["final_training"],
            "final_score": final_score["score_data"],
        },
        "temporal": resolved["temporal"],
        "parameter_profiles": resolved["parameter_profiles"],
        "capabilities": resolved["initialization_capabilities"],
        "validation": summary,
        "selection_lock_sha256": _sha_file(lock_path),
        "chosen": lock["chosen"],
        "final_fit": final_fit,
        "final_score": final_score,
        "replay": replay,
    }
    (attempt_dir / "campaign-report.json").write_text(_json(report), encoding="utf-8")
    chosen = cast(dict[str, object], lock["chosen"])
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['model_id']))}</td>"
        f"<td>{html.escape(str(item['variant_id']))}</td>"
        f"<td>{html.escape(str(item['initialization']))}</td>"
        f"<td>{html.escape(str(item['mean_score']))}</td>"
        f"<td>{html.escape(str(item['score_stddev']))}</td></tr>"
        for item in cast(list[dict[str, object]], summary["candidates"])
    )
    (attempt_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Forecast campaign</title>"
        "<style>body{font:14px system-ui;margin:32px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccd3df;padding:8px;text-align:left}</style>"
        f"<h1>{html.escape(cast(str, resolved['name']))}</h1>"
        f"<p>Selected <b>{html.escape(str(chosen['model_id']))}</b> variant "
        f"<b>{html.escape(str(chosen['variant_id']))}</b> with "
        f"{html.escape(str(chosen['initialization']))}. Final "
        f"{html.escape(str(final_score['metric']))}: "
        f"<b>{html.escape(str(final_score['score']))}</b>.</p>"
        "<p>MLflow and TensorBoard are optional read-only views; the files linked by this "
        "campaign journal are authoritative.</p>"
        "<table><tr><th>Model</th><th>Variant</th><th>Initialization</th>"
        "<th>Mean validation</th>"
        f"<th>Stddev</th></tr>{rows}</table>",
        encoding="utf-8",
    )
    print(
        _metric_summary_banner(
            cast(str, final_score["metric"]),
            float(cast(float | int, chosen["mean_score"])),
            float(cast(float | int, final_score["score"])),
        ),
        file=sys.stderr,
        flush=True,
    )


@project_entrypoint
def _main() -> int:
    parser = argparse.ArgumentParser(description="Internal stages for a forecast campaign.")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in (
        "tune-parameters",
        "run-child",
        "aggregate-selection",
        "run-final-fit",
        "score-final",
        "replay-report",
    ):
        command = commands.add_parser(action)
        command.add_argument("--resolved", type=pathlib.Path, required=True)
        command.add_argument("--resolved-sha256", required=True)
        command.add_argument("--campaign-root", type=pathlib.Path, required=True)
        command.add_argument("--attempt-dir", type=pathlib.Path, required=True)
        if action == "run-child":
            command.add_argument("--child-id", required=True)
        if action == "tune-parameters":
            command.add_argument("--model-id", required=True)
            command.add_argument("--variant-id", required=True)
        if action == "score-final":
            command.add_argument("--score-input", type=pathlib.Path, required=True)
            command.add_argument("--score-input-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.action == "tune-parameters":
            run_tune_parameters(
                args.resolved,
                args.resolved_sha256,
                model_id=args.model_id,
                variant_id=args.variant_id,
                campaign_root=args.campaign_root.resolve(),
                attempt_dir=args.attempt_dir.resolve(),
            )
        elif args.action == "run-child":
            run_child(
                args.resolved,
                args.resolved_sha256,
                child_id=args.child_id,
                campaign_root=args.campaign_root.resolve(),
                attempt_dir=args.attempt_dir.resolve(),
            )
        elif args.action == "aggregate-selection":
            aggregate_selection(
                args.resolved,
                args.resolved_sha256,
                campaign_root=args.campaign_root.resolve(),
                attempt_dir=args.attempt_dir.resolve(),
            )
        elif args.action == "run-final-fit":
            run_final_fit(
                args.resolved,
                args.resolved_sha256,
                campaign_root=args.campaign_root.resolve(),
                attempt_dir=args.attempt_dir.resolve(),
            )
        elif args.action == "score-final":
            score_final(
                args.resolved,
                args.resolved_sha256,
                args.score_input,
                args.score_input_sha256,
                campaign_root=args.campaign_root.resolve(),
                attempt_dir=args.attempt_dir.resolve(),
            )
        else:
            replay_report(
                args.resolved,
                args.resolved_sha256,
                campaign_root=args.campaign_root.resolve(),
                attempt_dir=args.attempt_dir.resolve(),
            )
    except (ForecastCampaignError, CampaignError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
