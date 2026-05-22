"""Typed execution plans from a model profile and checked panel facts.

This is intentionally an adapter-owned bridge.  YAML provides declared values, but this module
decides which values are executable and how a custom architecture is resolved.  It never forwards
an arbitrary YAML mapping into an adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from spec.forecast_spec import ForecastDataSpec, ForecastModelSpec
from spec.resolved import ResolvedBuild

from autoforecast._exact_json import ExactJSONError, loads_exact_json
from autoforecast.model_catalog import Candidate
from autoforecast.model_names import REVISION_VARIANTS
from autoforecast.panel import Panel
from autoforecast.revision_config import ROLLING_CONFIG_FIELDS, validate_revision_recipe
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT
from autoforecast.tuning import TuningError, loads_tuning_spec, materialize_runtime_parameters

if TYPE_CHECKING:
    from autoforecast.model_api import DataFacts

__all__ = [
    "ExecutionPlan",
    "ExecutionPlanError",
    "REVISION_MODEL_PARAMETERS",
    "REVISION_TRAINING_PARAMETERS",
    "resolve_execution_plan",
    "revision_required_history",
    "resolve_revision_rolling_config",
    "validate_revision_parameter_groups",
]

_REVISION_V3_MODEL_PARAMETERS = frozenset(
    {
        "encoder_days",
        "d_model",
        "n_heads",
        "trunk_stages",
        "covariate_dropout",
    }
)
_REVISION_V2_MODEL_PARAMETERS = _REVISION_V3_MODEL_PARAMETERS | {
    "decoder_layers",
    "event_kernel_layers",
    "history_attention_layers",
    "commercial_attention_layers",
    "commercial_attention_heads",
    "demand_dropout",
    "patch_size",
}
REVISION_MODEL_PARAMETERS = MappingProxyType(
    {
        "sft": _REVISION_V2_MODEL_PARAMETERS,
        "hmfnet": _REVISION_V3_MODEL_PARAMETERS,
        "pact": _REVISION_V3_MODEL_PARAMETERS,
    }
)
REVISION_TRAINING_PARAMETERS = frozenset(
    {"learning_rate", "train_stride", "train_windows", "series_loss_weighting"}
)


def revision_required_history(rolling: Mapping[str, object], horizon: int) -> int:
    """History the readiness gate must see before the rolling trainer can start."""
    if rolling.get("observation_contract") == GLOBAL_LEFT_PADDING_CONTRACT:
        return (
            horizon
            + (cast(int, rolling["train_windows"]) - 1) * cast(int, rolling["train_stride"])
            + 1
        )
    return cast(int, rolling["encoder_days"]) + horizon


class ExecutionPlanError(ValueError):
    """A profile default cannot become a safe executable adapter configuration."""


class _Items(tuple[tuple[str, object], ...]):
    """Distinguish a frozen top-level mapping from a frozen JSON sequence."""


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved settings and serializable adapter inputs for exactly one run."""

    readiness_settings: tuple[tuple[str, object], ...]
    adapter_kwargs: tuple[tuple[str, object], ...]
    effective_parameters: tuple[tuple[str, object], ...]

    def as_record(self) -> dict[str, object]:
        return {
            "schema": "autoforecast.execution-plan.v1",
            "readiness_settings": _thaw(self.readiness_settings),
            "adapter_kwargs": _thaw(self.adapter_kwargs),
            "effective_parameters": _thaw(self.effective_parameters),
        }


def _freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze(item) for key, item in cast(dict[str, object], value).items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, _Items):
        return {key: _thaw(item) for key, item in value}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _frozen(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return _Items((name, _freeze(value)) for name, value in sorted(values.items()))


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ExecutionPlanError(f"custom model {label} must be a positive integer")
    return value


def _fraction(value: object, label: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ExecutionPlanError(f"custom model {label} must be a finite number")
    return float(value)


def _factor_dynamics(value: object) -> Literal["ar1", "iid"]:
    """Accept only the two implemented, comparable shared-factor dynamics."""
    if value not in ("ar1", "iid"):
        raise ExecutionPlanError("custom model factor_dynamics must be 'ar1' or 'iid'")
    return cast(Literal["ar1", "iid"], value)


def validate_revision_parameter_groups(
    candidate: Candidate,
    model_parameters: object,
    training_parameters: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Reject revision parameters that the SFT/HMFNet execution adapter cannot consume."""
    expected_model = REVISION_MODEL_PARAMETERS.get(candidate.adapter)
    if expected_model is None:
        raise ExecutionPlanError(f"candidate {candidate.id!r} is not a revision model")
    if (
        type(model_parameters) is not dict
        or set(model_parameters) != expected_model
        or type(training_parameters) is not dict
        or set(training_parameters) != REVISION_TRAINING_PARAMETERS
    ):
        model_names = sorted(model_parameters) if type(model_parameters) is dict else []
        training_names = sorted(training_parameters) if type(training_parameters) is dict else []
        raise ExecutionPlanError(
            "revision parameters must contain exactly the typed executable settings; "
            f"model={model_names}, training={training_names}"
        )
    return cast(dict[str, object], model_parameters), cast(dict[str, object], training_parameters)


def _declared_parameters(
    candidate: Candidate,
    parameter_overrides: dict[str, object] | None,
    *,
    expected_model: set[str],
    expected_training: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve one profile's complete, allowlisted HPO surface.

    This deliberately does not forward a YAML dictionary to an adapter. A profile either names
    every executable setting, with a checked domain, or it is not a tuneable profile.
    """
    try:
        tuning = loads_tuning_spec(candidate.source_bytes, source_path=candidate.source_path)
        resolved = (
            tuning.defaults()
            if parameter_overrides is None
            else materialize_runtime_parameters(tuning, parameter_overrides)
        )
    except TuningError as error:
        raise ExecutionPlanError(
            f"model profile has no valid declared tuning surface: {error}"
        ) from error
    model = resolved["model"]
    training = resolved["training"]
    if set(model) != expected_model or set(training) != expected_training:
        raise ExecutionPlanError(
            "model profile must declare exactly its typed executable settings; "
            f"model={sorted(model)}, training={sorted(training)}"
        )
    return model, training


def _custom_plan(
    candidate: Candidate,
    panel: Panel,
    facts: DataFacts,
    *,
    horizon: int,
    epochs: int,
    seed: int,
    calendar_events: tuple[tuple[str, str, str | None], ...],
    event_max_support: int,
    parameter_overrides: dict[str, object] | None,
) -> ExecutionPlan:
    required_model = {
        "d_model",
        "heads",
        "patch_size",
        "factor_rank",
        "factor_dynamics",
        "demand_branch_dropout",
        "covariate_dropout",
        "encoder_days",
        "trunk_stages",
        "decoder_layers",
        "history_attention_layers",
        "commercial_attention_layers",
        "event_kernel_layers",
    }
    model, training = _declared_parameters(
        candidate,
        parameter_overrides,
        expected_model=required_model,
        expected_training={"learning_rate"},
    )
    encoder_days = _integer(model["encoder_days"], "encoder_days")
    if type(event_max_support) is not int or event_max_support < 1:
        raise ExecutionPlanError("event_max_support must be a positive integer")
    # The request recipe is the source-controlled calendar data plane: its parser has already
    # proved every occurrence and announcement date.  Name that source in the architecture
    # contract so the resolved build commits to the same branch the worker will construct.
    calendar_source = "run.calendar_events" if calendar_events else None
    target_estimand = panel.target_estimand
    if target_estimand not in ("sales_given_availability", "unconstrained_demand"):
        raise ExecutionPlanError(
            f"custom model has unsupported target estimand: {target_estimand!r}"
        )
    data = ForecastDataSpec(
        time_idx=panel.time_col,
        target=panel.target_col,
        group_ids=(panel.series_col,),
        target_estimand=cast(
            Literal["sales_given_availability", "unconstrained_demand"],
            target_estimand,
        ),
        freq=panel.frequency,
        max_encoder_length=encoder_days,
        max_prediction_length=horizon,
        price_col=panel.price_col,
        factor_group_col=panel.factor_group,
        static_categoricals=panel.static,
        hierarchy=(() if panel.factor_group is None else (panel.factor_group,)),
        events=calendar_source,
        event_known_at_col=("run.calendar_events.known_at" if calendar_events else None),
        event_episode_col=("run.calendar_events.date" if calendar_events else None),
    )
    specification = ForecastModelSpec(
        d_model=_integer(model["d_model"], "d_model"),
        heads=_integer(model["heads"], "heads"),
        patch_size=_integer(model["patch_size"], "patch_size"),
        factor_rank=_integer(model["factor_rank"], "factor_rank"),
        factor_dynamics=_factor_dynamics(model["factor_dynamics"]),
        demand_branch_dropout=_fraction(model["demand_branch_dropout"], "demand_branch_dropout"),
        covariate_dropout=_fraction(model["covariate_dropout"], "covariate_dropout"),
        learning_rate=_fraction(training["learning_rate"], "training.learning_rate"),
        event_max_support=(-event_max_support, event_max_support),
    )
    try:
        resolved = ResolvedBuild.from_specs(
            data,
            specification,
            zero_share=None if facts.profile is None else facts.profile.zero_rate_eligible,
        )
    except (TypeError, ValueError) as error:
        raise ExecutionPlanError(f"custom architecture cannot be resolved: {error}") from error
    resolved_record = {**resolved.canonical(), "notes": list(resolved.notes)}
    kwargs = {
        "encoder_days": encoder_days,
        "trunk_stages": _integer(model["trunk_stages"], "trunk_stages"),
        "decoder_layers": _integer(model["decoder_layers"], "decoder_layers"),
        "history_attention_layers": _integer(
            model["history_attention_layers"], "history_attention_layers"
        ),
        "commercial_attention_layers": _integer(
            model["commercial_attention_layers"], "commercial_attention_layers"
        ),
        "event_kernel_layers": _integer(model["event_kernel_layers"], "event_kernel_layers"),
        "epochs": epochs,
        "seed": seed,
        "lr": training["learning_rate"],
        "resolved_build": resolved_record,
    }
    effective = {
        **{f"model.{name}": value for name, value in model.items()},
        **{f"training.{name}": value for name, value in training.items()},
        "data.horizon": horizon,
        "training.epochs": epochs,
        "training.seed": seed,
        "resolved.build_hash": resolved.build_hash,
    }
    return ExecutionPlan(
        readiness_settings=_frozen({"encoder_days": encoder_days}),
        adapter_kwargs=_frozen(kwargs),
        effective_parameters=_frozen(effective),
    )


def resolve_revision_rolling_config(
    candidate: Candidate,
    *,
    model_parameters: object,
    training_parameters: object,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    """Return the exact validated pinned revision recipe plus its permitted run overrides."""
    model, training = validate_revision_parameter_groups(
        candidate, model_parameters, training_parameters
    )
    config_path = candidate.implementation_config_path
    config_sha256 = candidate.implementation_config_sha256
    config_bytes = candidate.implementation_config_bytes
    if config_path is None or config_sha256 is None or config_bytes is None:
        raise ExecutionPlanError("revision model profile has no pinned implementation config")
    try:
        raw = loads_exact_json(config_bytes)
    except ExactJSONError as error:
        raise ExecutionPlanError(f"revision implementation config is invalid: {error}") from error
    if type(raw) is not dict:
        raise ExecutionPlanError("revision implementation config must be one JSON object")
    expected_variant = REVISION_VARIANTS[candidate.adapter]
    legacy_fields = ROLLING_CONFIG_FIELDS - {"v5", "series_loss_weighting"}
    legacy_pact_fields = ROLLING_CONFIG_FIELDS - {"series_loss_weighting"}
    if set(raw) != ROLLING_CONFIG_FIELDS and not (
        expected_variant != "v5" and set(raw) == legacy_fields
        or expected_variant == "v5" and set(raw) == legacy_pact_fields
    ):
        raise ExecutionPlanError(
            "revision implementation config must fully serialize RollingConfig; "
            f"missing={sorted(ROLLING_CONFIG_FIELDS - set(raw))}, "
            f"extra={sorted(set(raw) - ROLLING_CONFIG_FIELDS)}"
        )
    if "v5" not in raw or "series_loss_weighting" not in raw:
        raw = {
            **raw,
            "v5": raw.get("v5"),
            "series_loss_weighting": raw.get("series_loss_weighting", "uniform"),
        }
    if raw.get("variant") != expected_variant:
        raise ExecutionPlanError("revision implementation config and adapter variant disagree")
    integer_fields = set(model) - {"demand_dropout", "covariate_dropout"}
    rolling = {
        **raw,
        **{name: _integer(model[name], name) for name in sorted(integer_fields)},
        **{
            name: _fraction(model[name], name)
            for name in sorted(set(model) - integer_fields)
        },
        "learning_rate": _fraction(
            training["learning_rate"], "training.learning_rate"
        ),
        "train_stride": _integer(training["train_stride"], "training.train_stride"),
        "train_windows": _integer(training["train_windows"], "training.train_windows"),
        "series_loss_weighting": training["series_loss_weighting"],
        "epochs": epochs,
        "seed": seed,
    }
    if expected_variant == "v3" and type(rolling["v3"]) is dict:
        rolling["v3"] = {
            **cast(dict[str, object], rolling["v3"]),
            "local_blocks": rolling["trunk_stages"],
        }
    if expected_variant == "v5" and type(rolling["v5"]) is dict:
        rolling["v5"] = {
            **cast(dict[str, object], rolling["v5"]),
            "n_layers": rolling["trunk_stages"],
        }
    try:
        validate_revision_recipe(rolling)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ExecutionPlanError(f"revision implementation config is invalid: {error}") from error
    return cast(dict[str, object], rolling)


def _revision_plan(
    candidate: Candidate,
    *,
    horizon: int,
    epochs: int,
    seed: int,
    parameter_overrides: dict[str, object] | None,
) -> ExecutionPlan:
    """Resolve a registered revision from its pinned recipe and typed variant knobs."""
    model, training = _declared_parameters(
        candidate,
        parameter_overrides,
        expected_model=set(REVISION_MODEL_PARAMETERS[candidate.adapter]),
        expected_training=set(REVISION_TRAINING_PARAMETERS),
    )
    rolling = resolve_revision_rolling_config(
        candidate,
        model_parameters=model,
        training_parameters=training,
        epochs=epochs,
        seed=seed,
    )
    config_sha256 = candidate.implementation_config_sha256
    assert config_sha256 is not None
    expected_variant = REVISION_VARIANTS[candidate.adapter]
    effective = {
        **{f"model.{name}": value for name, value in model.items()},
        **{f"training.{name}": value for name, value in training.items()},
        "data.horizon": horizon,
        "training.epochs": epochs,
        "training.seed": seed,
        "implementation.config_sha256": config_sha256,
        "implementation.variant": expected_variant,
    }
    return ExecutionPlan(
        readiness_settings=_frozen(
            {"required_history": revision_required_history(rolling, horizon)}
        ),
        adapter_kwargs=_frozen({"rolling_config": rolling}),
        effective_parameters=_frozen(effective),
    )


def _shallow_plan(
    candidate: Candidate,
    *,
    horizon: int,
    epochs: int,
    seed: int,
    parameter_overrides: dict[str, object] | None,
) -> ExecutionPlan:
    """Resolve the LightGBM knobs whose values affect the fitted shallow model."""
    model, training = _declared_parameters(
        candidate,
        parameter_overrides,
        expected_model={"num_leaves", "min_data_in_leaf", "num_boost_round"},
        expected_training={"learning_rate"},
    )
    num_leaves = _integer(model["num_leaves"], "num_leaves")
    min_data_in_leaf = _integer(model["min_data_in_leaf"], "min_data_in_leaf")
    num_boost_round = _integer(model["num_boost_round"], "num_boost_round")
    learning_rate = _fraction(training["learning_rate"], "training.learning_rate")
    kwargs = {
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "num_boost_round": num_boost_round,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    effective = {
        **{f"model.{name}": value for name, value in model.items()},
        **{f"training.{name}": value for name, value in training.items()},
        "data.horizon": horizon,
        "training.epochs": epochs,
        "training.seed": seed,
    }
    return ExecutionPlan(_frozen({}), _frozen(kwargs), _frozen(effective))


def _tft_plan(
    candidate: Candidate,
    *,
    horizon: int,
    epochs: int,
    seed: int,
    parameter_overrides: dict[str, object] | None,
) -> ExecutionPlan:
    """Resolve TFT architecture/optimizer knobs, leaving budget and hardware controls fixed."""
    model, training = _declared_parameters(
        candidate,
        parameter_overrides,
        expected_model={"hidden_size", "attention_head_size", "dropout"},
        expected_training={"learning_rate"},
    )
    hidden_size = _integer(model["hidden_size"], "hidden_size")
    attention_head_size = _integer(model["attention_head_size"], "attention_head_size")
    dropout = _fraction(model["dropout"], "dropout")
    learning_rate = _fraction(training["learning_rate"], "training.learning_rate")
    if not 0 <= dropout < 1:
        raise ExecutionPlanError("TFT model.dropout must be in [0, 1)")
    kwargs = {
        "max_epochs": epochs,
        "hidden_size": hidden_size,
        "attention_head_size": attention_head_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    effective = {
        **{f"model.{name}": value for name, value in model.items()},
        **{f"training.{name}": value for name, value in training.items()},
        "data.horizon": horizon,
        "training.epochs": epochs,
        "training.seed": seed,
    }
    return ExecutionPlan(_frozen({}), _frozen(kwargs), _frozen(effective))


def _tft_q_plan(
    candidate: Candidate,
    *,
    horizon: int,
    epochs: int,
    seed: int,
    parameter_overrides: dict[str, object] | None,
) -> ExecutionPlan:
    """Resolve the frozen daily M5 translation of the reference TFT profile."""
    expected_model = {
        "hidden_size", "hidden_continuous_size", "attention_head_size", "dropout",
        "max_encoder_length", "min_encoder_length", "lag_short_days", "lag_long_days",
        "output_size", "loss_kind",
    }
    model, training = _declared_parameters(
        candidate,
        parameter_overrides,
        expected_model=expected_model,
        expected_training={"learning_rate"},
    )
    if horizon != 28:
        raise ExecutionPlanError("tft_q is frozen to the daily benchmark 28-day horizon")
    kwargs = {
        "max_epochs": epochs,
        "hidden_size": _integer(model["hidden_size"], "hidden_size"),
        "hidden_continuous_size": _integer(
            model["hidden_continuous_size"], "hidden_continuous_size"
        ),
        "attention_head_size": _integer(model["attention_head_size"], "attention_head_size"),
        "dropout": _fraction(model["dropout"], "dropout"),
        "learning_rate": _fraction(training["learning_rate"], "training.learning_rate"),
        "max_encoder_length": _integer(model["max_encoder_length"], "max_encoder_length"),
        "min_encoder_length": _integer(model["min_encoder_length"], "min_encoder_length"),
        "lag_short_days": _integer(model["lag_short_days"], "lag_short_days"),
        "lag_long_days": _integer(model["lag_long_days"], "lag_long_days"),
        "output_size": _integer(model["output_size"], "output_size"),
        "loss_kind": model["loss_kind"],
        "seed": seed,
    }
    if kwargs["attention_head_size"] > kwargs["hidden_size"]:
        raise ExecutionPlanError("tft_q attention heads must not exceed hidden size")
    if kwargs["min_encoder_length"] > kwargs["max_encoder_length"]:
        raise ExecutionPlanError("tft_q minimum encoder exceeds maximum encoder")
    if kwargs["lag_short_days"] >= kwargs["lag_long_days"]:
        raise ExecutionPlanError("tft_q lag days must be strictly increasing")
    loss_sizes = {"quantile": 4, "v3_hurdle_nb": 3}
    loss_kind = kwargs["loss_kind"]
    if type(loss_kind) is not str or loss_kind not in loss_sizes:
        raise ExecutionPlanError("tft_q requires a declared loss kind")
    expected_loss = "quantile" if candidate.adapter == "tft_q" else "v3_hurdle_nb"
    if loss_kind != expected_loss:
        raise ExecutionPlanError(f"{candidate.adapter} requires loss kind {expected_loss}")
    if kwargs["output_size"] != loss_sizes[loss_kind]:
        raise ExecutionPlanError("tft_q output_size disagrees with loss kind")
    effective = {
        **{f"model.{name}": value for name, value in model.items()},
        **{f"training.{name}": value for name, value in training.items()},
        "data.horizon": horizon,
        "training.epochs": epochs,
        "training.seed": seed,
        "profile.source": "configs/models/neural/tft_q/model.yaml",
        "profile.translation": "weekly_to_daily_v1",
    }
    return ExecutionPlan(
        _frozen({"min_encoder_length": kwargs["min_encoder_length"]}),
        _frozen(kwargs),
        _frozen(effective),
    )


def resolve_execution_plan(
    candidate: Candidate,
    panel: Panel,
    facts: DataFacts,
    *,
    horizon: int,
    epochs: int,
    seed: int = 0,
    calendar_events: tuple[tuple[str, str, str | None], ...] = (),
    event_max_support: int = 30,
    parameter_overrides: dict[str, object] | None = None,
) -> ExecutionPlan:
    """Resolve the only typed execution inputs supported by each adapter."""
    if epochs < 1 or horizon < 1:
        raise ExecutionPlanError("epochs and horizon must be positive")
    if candidate.adapter == "mft":
        return _custom_plan(
            candidate,
            panel,
            facts,
            horizon=horizon,
            epochs=epochs,
            seed=seed,
            calendar_events=calendar_events,
            event_max_support=event_max_support,
            parameter_overrides=parameter_overrides,
        )
    if candidate.adapter in {
        "sft",
        "hmfnet",
        "pact",
    }:
        return _revision_plan(
            candidate,
            horizon=horizon,
            epochs=epochs,
            seed=seed,
            parameter_overrides=parameter_overrides,
        )
    if candidate.adapter == "shallow_global":
        return _shallow_plan(
            candidate,
            horizon=horizon,
            epochs=epochs,
            seed=seed,
            parameter_overrides=parameter_overrides,
        )
    if candidate.adapter == "tft":
        return _tft_plan(
            candidate,
            horizon=horizon,
            epochs=epochs,
            seed=seed,
            parameter_overrides=parameter_overrides,
        )
    if candidate.adapter in {"tft_q", "tft_hnb"}:
        return _tft_q_plan(
            candidate,
            horizon=horizon,
            epochs=epochs,
            seed=seed,
            parameter_overrides=parameter_overrides,
        )
    if parameter_overrides is not None:
        raise ExecutionPlanError(
            f"model {candidate.id!r} has no supported typed HPO execution bridge"
        )
    settings: dict[str, object] = {}
    kwargs: dict[str, object] = {}
    if candidate.adapter == "timesfm_standalone":
        settings["context_days"] = 1024
    return ExecutionPlan(
        _frozen(settings),
        _frozen(kwargs),
        _frozen(
            {
                "training.epochs": epochs,
                **({"training.seed": seed} if "seed" in kwargs else {}),
            }
        ),
    )
