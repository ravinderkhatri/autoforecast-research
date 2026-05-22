"""Dependency-free schema checks for serialized SFT/HMFNet/PACT rolling-training recipes."""

from __future__ import annotations

import math

from autoforecast._numeric import is_finite_real

ROLLING_CONFIG_FIELDS = frozenset(
    {
        "variant", "encoder_days", "d_model", "n_heads", "trunk_stages",
        "decoder_layers", "history_attention_layers", "commercial_attention_layers",
        "commercial_attention_heads", "event_kernel_layers", "demand_dropout",
        "covariate_dropout", "factor_rank", "factor_draws", "train_windows",
        "train_stride", "max_updates", "batch_size", "evaluate_every",
        "validation_every_epochs", "validation_backend", "checkpoint_every_updates",
        "checkpoint_every_seconds", "checkpoint_keep_last", "progress", "patience",
        "early_stopping_absolute_min_delta", "early_stopping_relative_min_delta",
        "learning_rate", "weight_decay", "gradient_clip", "mean_loss_weight",
        "hierarchy_loss_weight", "hierarchy_every", "hierarchy_epsilon",
        "event_penalty_weight", "price_penalty_weight", "event_max_support",
        "excess_floor", "seed", "device", "torch_threads", "max_array_bytes",
        "aggregate_history", "epochs", "head", "patch_size", "statistic_windows",
        "seasonal_window", "reference_window", "selection_metric", "mean_weighting",
        "series_loss_weighting",
        "hierarchy_weighting", "identity_columns", "factor_columns",
        "aggregate_columns", "baseline_columns", "event_pool_columns",
        "calendar_group_column", "observation_contract", "cycles", "v3", "v5",
    }
)

HMFNET_CONFIG_FIELDS = frozenset(
    {
        "recent", "middle_end", "middle_width", "old_width", "old_slots", "span",
        "local_blocks", "phase_cycles", "context_levels", "expert_levels", "feedback",
        "expert_batch_per_group", "expert_loss_weight", "expert_seed", "anchor_floor",
        "numeric_scale_floor", "categorical_width", "max_aggregate_bytes",
        "feature_policy", "aggregate_covariates",
    }
)

PACT_CONFIG_FIELDS = frozenset(
    {
        "recent", "middle_end", "middle_width", "old_width", "old_slots", "span",
        "n_layers", "ff_multiplier", "rope_theta", "position_mode", "context_levels",
        "max_streams", "head_law", "categorical_width", "numeric_scale_floor",
        "max_aggregate_bytes", "mask_policy", "token_policy",
    }
)

_POSITIVE_INTS = (
    "encoder_days", "d_model", "n_heads", "trunk_stages", "decoder_layers",
    "history_attention_layers", "commercial_attention_layers", "commercial_attention_heads",
    "event_kernel_layers", "factor_draws", "train_windows", "train_stride", "max_updates",
    "batch_size", "evaluate_every", "validation_every_epochs", "checkpoint_every_updates",
    "checkpoint_keep_last", "patience", "hierarchy_every", "event_max_support",
    "torch_threads", "max_array_bytes", "patch_size", "seasonal_window", "reference_window",
)
_NONNEGATIVE_NUMBERS = (
    "demand_dropout", "covariate_dropout", "learning_rate", "weight_decay", "gradient_clip",
    "mean_loss_weight", "hierarchy_loss_weight", "hierarchy_epsilon", "event_penalty_weight",
    "price_penalty_weight", "checkpoint_every_seconds", "early_stopping_absolute_min_delta",
    "early_stopping_relative_min_delta", "excess_floor",
)
_COLUMN_LISTS = (
    "identity_columns", "factor_columns", "aggregate_columns", "baseline_columns",
    "event_pool_columns",
)


def _finite_number(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_v3(value: object) -> None:
    if type(value) is not dict or set(value) != HMFNET_CONFIG_FIELDS:
        raise ValueError("HMFNet recipe must fully serialize its exact feature/objective fields")
    counts = (
        "recent", "middle_end", "middle_width", "old_width", "old_slots", "local_blocks",
        "expert_batch_per_group", "categorical_width", "max_aggregate_bytes",
    )
    if any(type(value[name]) is not int or value[name] < 1 for name in counts):
        raise ValueError("HMFNet dimensions and budgets must be positive integers")
    if (
        value["middle_end"] <= value["recent"]
        or (value["middle_end"] - value["recent"]) % value["middle_width"]
        or value["old_slots"] < 2
        or value["span"] is not None
        and (type(value["span"]) is not int or value["span"] < 1)
        or type(value["expert_seed"]) is not int
        or value["expert_seed"] < 0
        or type(value["feedback"]) is not bool
    ):
        raise ValueError("invalid HMFNet bands, span, seed or feedback flag")
    for name in ("context_levels", "expert_levels"):
        groups = value[name]
        if type(groups) not in {list, tuple} or any(
            type(group) not in {list, tuple} for group in groups
        ):
            raise ValueError("HMFNet grouping sets must be sequences of column sequences")
        normalized = [tuple(group) for group in groups]
        if any(
            any(type(item) is not str or not item.strip() for item in group)
            or len(set(group)) != len(group)
            for group in normalized
        ) or len({frozenset(group) for group in normalized}) != len(normalized):
            raise ValueError("HMFNet grouping sets are invalid or duplicated")
    cycles = value["phase_cycles"]
    if type(cycles) not in {list, tuple} or any(
        type(item) not in {list, tuple}
        or len(item) != 2
        or not _finite_number(item[0])
        or item[0] <= 0
        or type(item[1]) is not int
        or item[1] < 1
        for item in cycles
    ) or len({item[0] for item in cycles}) != len(cycles):
        raise ValueError("HMFNet phase cycles are invalid or duplicated")
    if any(
        not _finite_number(value[name]) or value[name] <= 0
        for name in ("anchor_floor", "numeric_scale_floor")
    ) or (
        not _finite_number(value["expert_loss_weight"])
        or value["expert_loss_weight"] < 0
        or value["feature_policy"] != "masked_moments_phase_v1"
        or value["aggregate_covariates"] != "none"
    ):
        raise ValueError("unsupported HMFNet numerical, objective or feature policy")


def _validate_v5(value: object) -> None:
    if type(value) is not dict or set(value) != PACT_CONFIG_FIELDS:
        raise ValueError("PACT recipe must fully serialize its exact architecture fields")
    counts = (
        "recent", "middle_end", "middle_width", "old_width", "old_slots", "n_layers",
        "ff_multiplier", "max_streams", "categorical_width", "max_aggregate_bytes",
    )
    if any(type(value[name]) is not int or value[name] < 1 for name in counts):
        raise ValueError("PACT dimensions and budgets must be positive integers")
    levels = value["context_levels"]
    if type(levels) not in {list, tuple} or any(
        type(level) not in {list, tuple} for level in levels
    ):
        raise ValueError("PACT context levels must be sequences of column sequences")
    normalized = [tuple(level) for level in levels]
    if (
        any(not level or len(set(level)) != len(level)
            or any(type(name) is not str or not name.strip() for name in level)
            for level in normalized)
        or len({frozenset(level) for level in normalized}) != len(normalized)
        or len(normalized) + 1 > value["max_streams"]
        or normalized and value["n_layers"] < 2
    ):
        raise ValueError("PACT context streams are invalid, duplicated or unreachable")
    if (
        value["middle_end"] <= value["recent"]
        or (value["middle_end"] - value["recent"]) % value["middle_width"]
        or value["old_slots"] < 2
        or value["span"] is not None
        and (type(value["span"]) is not int or value["span"] < 1)
        or not _finite_number(value["rope_theta"]) or value["rope_theta"] <= 0
        or not _finite_number(value["numeric_scale_floor"])
        or value["numeric_scale_floor"] <= 0
        or value["position_mode"] not in {"elapsed_day", "token_index"}
        or value["head_law"] not in {"negbin", "zero_truncated_hurdle_negbin"}
        or value["mask_policy"] != "prefix_causal_future_full_v1"
        or value["token_policy"] != "v5_multiresolution_calendar_segments_v1"
    ):
        raise ValueError("unsupported PACT geometry, law, position or mask policy")


def validate_revision_recipe(recipe: object) -> None:
    """Validate the complete JSON-facing RollingConfig contract before worker launch."""
    actual = set(recipe) if type(recipe) is dict else set()
    if type(recipe) is not dict or actual != ROLLING_CONFIG_FIELDS:
        raise ValueError(
            "revision recipe must fully serialize RollingConfig; "
            f"missing={sorted(ROLLING_CONFIG_FIELDS - actual)}, "
            f"extra={sorted(actual - ROLLING_CONFIG_FIELDS)}"
        )
    if any(type(recipe[name]) is not int or recipe[name] < 1 for name in _POSITIVE_INTS):
        raise ValueError("revision counts and budgets must be positive integers")
    if any(type(recipe[name]) is not int or recipe[name] < 0 for name in ("seed", "factor_rank")):
        raise ValueError("seed and factor_rank must be nonnegative integers")
    if any(
        not _finite_number(recipe[name]) or recipe[name] < 0 for name in _NONNEGATIVE_NUMBERS
    ):
        raise ValueError("revision numerical controls must be finite nonnegative numbers")
    for name in ("statistic_windows", *_COLUMN_LISTS):
        values = recipe[name]
        if type(values) not in {list, tuple}:
            raise ValueError(f"{name} must be a sequence")
    if any(type(value) is not int or value < 1 for value in recipe["statistic_windows"]):
        raise ValueError("statistic_windows must contain positive integers")
    for name in _COLUMN_LISTS:
        values = recipe[name]
        if len(set(values)) != len(values) or any(
            type(value) is not str or not value.strip() for value in values
        ):
            raise ValueError(f"{name} needs unique nonempty column names")
    cycles = recipe["cycles"]
    if cycles is not None and (
        type(cycles) not in {list, tuple}
        or any(
            type(item) not in {list, tuple}
            or len(item) != 3
            or type(item[0]) is not str
            or not item[0].strip()
            or not _finite_number(item[1])
            or item[1] <= 0
            or type(item[2]) is not int
            or item[2] < 1
            for item in cycles
        )
        or len({item[0] for item in cycles}) != len(cycles)
    ):
        raise ValueError("cycles require unique names, positive periods and harmonics")
    if recipe["calendar_group_column"] is not None and (
        type(recipe["calendar_group_column"]) is not str
        or not recipe["calendar_group_column"].strip()
    ):
        raise ValueError("calendar_group_column must be a non-blank string or null")
    if recipe["epochs"] is not None and (
        type(recipe["epochs"]) is not int or recipe["epochs"] < 1
    ):
        raise ValueError("epochs must be a positive integer or null")
    if (
        recipe["variant"] not in {"v1_corrected", "v2", "v3", "v5"}
        or recipe["device"] not in {"cpu", "cuda"}
        or type(recipe["aggregate_history"]) is not bool
        or recipe["head"] not in {"auto", "hurdle_negbin", "negbin"}
        or recipe["selection_metric"] not in {"auto", "wrmsse", "rmse"}
        or recipe["mean_weighting"] not in {"auto", "revenue", "uniform"}
        or recipe["series_loss_weighting"] not in {"uniform", "revenue", "output_gradient"}
        or recipe["hierarchy_weighting"] not in {"auto", "revenue", "uniform"}
        or recipe["validation_backend"] not in {"torch", "numpy"}
        or recipe["progress"] not in {"auto", "plain", "off"}
        or recipe["observation_contract"]
        not in {
            "masked_observed_counts_v1",
            "complete_recorded_sales_prelaunch_zeros_v1",
            "masked_observed_counts_global_left_padding_v1",
        }
        or recipe["encoder_days"] % recipe["patch_size"]
        or recipe["d_model"] % recipe["n_heads"]
        or recipe["variant"] == "v5" and (recipe["d_model"] // recipe["n_heads"]) % 2
        or recipe["demand_dropout"] >= 1
        or recipe["covariate_dropout"] >= 1
        or min(recipe["learning_rate"], recipe["gradient_clip"], recipe["hierarchy_epsilon"])
        <= 0
        or not 0 < recipe["excess_floor"] <= 1e-3
        or recipe["checkpoint_keep_last"] < 2
        or recipe["checkpoint_every_seconds"] <= 0
    ):
        raise ValueError("unsupported revision topology, device or numerical control")
    v3 = recipe["v3"]
    if recipe["variant"] == "v3":
        _validate_v3(v3)
        assert isinstance(v3, dict)
        if (
            recipe["demand_dropout"] != 0
            or recipe["decoder_layers"] != 1
            or recipe["history_attention_layers"] != 1
            or recipe["commercial_attention_layers"] != 1
            or recipe["trunk_stages"] != v3["local_blocks"]
        ):
            raise ValueError("HMFNet requires its compact local/direct topology")
    elif v3 is not None:
        raise ValueError("only HMFNet may contain a HMFNet feature/objective recipe")
    v5 = recipe.get("v5")
    if recipe["variant"] == "v5":
        _validate_v5(v5)
        assert isinstance(v5, dict)
        expected_head = "negbin" if v5["head_law"] == "negbin" else "hurdle_negbin"
        if (
            recipe["demand_dropout"] != 0
            or recipe["covariate_dropout"] != 0
            or recipe["decoder_layers"] != 1
            or recipe["history_attention_layers"] != 1
            or recipe["commercial_attention_layers"] != 1
            or recipe["trunk_stages"] != v5["n_layers"]
            or recipe["aggregate_history"] != bool(v5["context_levels"])
            or recipe["head"] != expected_head
            or any(recipe[name] != 0 for name in (
                "mean_loss_weight", "hierarchy_loss_weight", "event_penalty_weight"))
            or recipe["train_windows"] < 7
            or math.gcd(recipe["train_stride"], 7) != 1
        ):
            raise ValueError("PACT rolling controls disagree with its registered architecture")
    elif v5 is not None:
        raise ValueError("only PACT may contain a PACT architecture recipe")
    if (
        recipe["variant"] == "v1_corrected"
        and recipe["observation_contract"]
        == "masked_observed_counts_global_left_padding_v1"
    ):
        raise ValueError("corrected MFT does not implement global left padding")
    if recipe["variant"] in {"v2", "v3", "v5"} and (
        recipe["factor_rank"] or recipe["factor_draws"] != 1 or recipe["price_penalty_weight"]
    ):
        raise ValueError("SFT/HMFNet/PACT are factorless and have no legacy price penalty")


def verify_validation_ledger(
    slots: object,
    evaluations: object,
    *,
    metric: str,
    validation_interval_updates: int,
    budget_updates: int,
    last_update: int,
    trainable_series: int,
) -> None:
    """Reject an impossible fixed-validation prefix, including delayed coverage slots."""
    if (
        type(slots) is not list or type(evaluations) is not list
        or metric not in {"rmse", "mae", "wape", "wrmsse", "uniform_hierarchy_rmsse"}
        or type(validation_interval_updates) is not int or validation_interval_updates < 1
        or type(budget_updates) is not int or budget_updates < 1
        or type(last_update) is not int or not 0 <= last_update <= budget_updates
        or type(trainable_series) is not int or trainable_series < 1
    ):
        raise ValueError("invalid validation ledger controls")
    due = list(range(validation_interval_updates, last_update + 1,
                     validation_interval_updates))
    if last_update == budget_updates and (not due or due[-1] != last_update):
        due.append(last_update)
    if len(slots) != len(due):
        raise ValueError("validation ledger misses a due update")
    evaluated_updates: list[int] = []
    covered_before = 0
    for expected_update, slot in zip(due, slots, strict=True):
        if (
            type(slot) is not dict
            or set(slot) != {"update", "covered_series", "trainable_series", "evaluated"}
            or type(slot["update"]) is not int or slot["update"] != expected_update
            or type(slot["covered_series"]) is not int
            or not covered_before <= slot["covered_series"] <= trainable_series
            or type(slot["trainable_series"]) is not int
            or slot["trainable_series"] != trainable_series
            or type(slot["evaluated"]) is not bool
            or slot["evaluated"] != (slot["covered_series"] == trainable_series)
        ):
            raise ValueError("invalid validation ledger slot")
        covered_before = slot["covered_series"]
        if slot["evaluated"]:
            evaluated_updates.append(expected_update)
    observed_updates: list[int] = []
    for row in evaluations:
        score = row.get(f"mean_{metric}") if type(row) is dict else None
        if (
            type(row) is not dict or set(row) != {"update", f"mean_{metric}"}
            or type(row["update"]) is not int
            or type(score) is not float or not is_finite_real(score) or score < 0
        ):
            raise ValueError("invalid validation ledger evaluation")
        observed_updates.append(row["update"])
    if observed_updates != evaluated_updates:
        raise ValueError("validation ledger evaluations differ from due slots")


__all__ = [
    "ROLLING_CONFIG_FIELDS", "HMFNET_CONFIG_FIELDS", "PACT_CONFIG_FIELDS",
    "validate_revision_recipe", "verify_validation_ledger",
]
