"""Functional execution-plan checks: profile defaults reach only their typed adapter bridge."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pathlib

import pytest
import torch
from spec.resolved import ResolvedBuild

from autoforecast.adapters import validate_execution_controls
from autoforecast.adapters.mft import CustomModel
from autoforecast.model_api import DataFacts
from autoforecast.model_catalog import discover_candidates
from autoforecast.model_execution import (
    ExecutionPlanError,
    resolve_execution_plan,
    resolve_revision_rolling_config,
    revision_required_history,
    validate_revision_parameter_groups,
)
from autoforecast.model_readiness import create_resolved_run, model_registration
from autoforecast.panel import Panel
from autoforecast.parameter_sources import _execution_plan_matches
from autoforecast.revision_config import validate_revision_recipe
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT
from autoforecast.rolling_training import RollingConfig
from autoforecast.tuning import loads_tuning_spec

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _facts() -> DataFacts:
    return DataFacts(
        "2016-03-25",
        "a" * 64,
        ("2011-01-29", "2016-03-25"),
        100,
        2,
        50,
        0,
        0,
        {},
        {},
        None,
        {},
    )


def _revision_parameters(candidate, **changes: object) -> dict[str, object]:
    grouped = loads_tuning_spec(
        candidate.source_bytes, source_path=candidate.source_path
    ).defaults()
    values = {
        f"{namespace}.{name}": value
        for namespace, parameters in grouped.items()
        for name, value in parameters.items()
    }
    values.update(changes)
    return values


def test_custom_profile_defaults_become_one_resolved_execution_plan() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.adapter == "mft"
    )
    panel = Panel(
        path="functional.parquet",
        series_col="id",
        time_col="date",
        target_col="qty",
        exposure_col="observation_state",
        price_col="sell_price",
        static=("dept_id",),
        factor_group="state_id",
        product_col="item_id",
        store_col="store_id",
        category_col="dept_id",
    )
    plan = resolve_execution_plan(candidate, panel, _facts(), horizon=14, epochs=3, seed=7)
    kwargs = plan.as_record()["adapter_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["encoder_days"] == 28
    assert kwargs["trunk_stages"] == 4
    assert kwargs["decoder_layers"] == 4
    assert kwargs["history_attention_layers"] == 4
    assert kwargs["commercial_attention_layers"] == 4
    assert kwargs["event_kernel_layers"] == 4
    assert kwargs["epochs"] == 3
    assert kwargs["lr"] == 0.001
    resolved = ResolvedBuild.from_record(kwargs["resolved_build"])
    assert resolved.d_model == 64 and resolved.factor_rank == 8
    assert resolved.factor_dynamics == "ar1"
    assert dict(plan.effective_parameters)["data.horizon"] == 14
    assert dict(plan.effective_parameters)["resolved.build_hash"] == resolved.build_hash

    expected = {
        name: value
        for name, value in dict(plan.effective_parameters).items()
        if name not in {"data.horizon", "training.epochs", "training.seed", "resolved.build_hash"}
    }
    assert _execution_plan_matches(plan.as_record(), candidate, expected)


def test_source_controlled_calendar_recipe_is_part_of_the_resolved_build() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.adapter == "mft"
    )
    plan = resolve_execution_plan(
        candidate,
        Panel("functional.parquet", "id", "date", "qty", factor_group="state_id"),
        _facts(),
        horizon=28,
        epochs=2,
        calendar_events=(("Memorial", "2016-05-30", None),),
        event_max_support=30,
    )
    kwargs = plan.as_record()["adapter_kwargs"]
    assert isinstance(kwargs, dict)
    resolved = ResolvedBuild.from_record(kwargs["resolved_build"])
    assert resolved.event_kernel == "fixed_window"
    assert resolved.event_max_support == (-30, 30)


def test_sft_and_hmfnet_catalog_models_resolve_their_actual_revision_recipes() -> None:
    candidates = {
        item.id: item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id in {"sft", "hmfnet"}
    }
    panel = Panel("functional.parquet", "id", "date", "qty")
    for model_id, variant in (
        ("sft", "v2"),
        ("hmfnet", "v3"),
    ):
        candidate = candidates[model_id]
        changes: dict[str, object] = {
            "model.encoder_days": 28,
            "model.d_model": 32,
            "model.n_heads": 2,
            "model.trunk_stages": 3,
            "model.covariate_dropout": 0.4,
            "training.train_stride": 7,
            "training.train_windows": 3,
        }
        if variant == "v2":
            changes.update(
                {
                    "model.decoder_layers": 2,
                    "model.history_attention_layers": 2,
                    "model.commercial_attention_layers": 2,
                    "model.commercial_attention_heads": 1,
                    "model.demand_dropout": 0.2,
                    "model.event_kernel_layers": 3,
                    "model.patch_size": 14,
                }
            )
        parameters = _revision_parameters(candidate, **changes)
        plan = resolve_execution_plan(
            candidate,
            panel,
            _facts(),
            horizon=7,
            epochs=2,
            seed=13,
            parameter_overrides=parameters,
        )
        kwargs = plan.as_record()["adapter_kwargs"]
        assert isinstance(kwargs, dict)
        rolling = kwargs["rolling_config"]
        assert isinstance(rolling, dict)
        assert rolling["variant"] == variant
        assert rolling["epochs"] == 2 and rolling["seed"] == 13
        assert rolling["encoder_days"] == 28 and rolling["train_windows"] == 3
        assert all(
            rolling[name.removeprefix("model.")] == value
            for name, value in parameters.items()
            if name.startswith("model.")
        )
        if variant == "v3":
            v3 = rolling["v3"]
            assert isinstance(v3, dict)
            assert v3["local_blocks"] == rolling["trunk_stages"] == 3
        assert plan.as_record()["readiness_settings"] == {"required_history": 22}
        run = create_resolved_run(
            model_id=model_id,
            adapter=candidate.adapter,
            origin="2016-03-25",
            horizon=7,
            settings=dict(plan.readiness_settings),
        )
        assert model_registration(candidate.adapter).required_history(panel, run) == 22
        expected = {
            name: value
            for name, value in dict(plan.effective_parameters).items()
            if not name.startswith(("data.", "implementation."))
            and name not in {"training.epochs", "training.seed"}
        }
        assert _execution_plan_matches(plan.as_record(), candidate, expected)


def test_full_history_padding_readiness_uses_rolling_span_not_encoder_length() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "sft"
    )
    plan = resolve_execution_plan(
        candidate,
        Panel("functional.parquet", "id", "date", "qty"),
        _facts(),
        horizon=28,
        epochs=2,
        parameter_overrides=_revision_parameters(
            candidate, **{"model.encoder_days": 1911}
        ),
    )
    record = plan.as_record()
    assert record["readiness_settings"] == {"required_history": 1457}
    adapter_kwargs = record["adapter_kwargs"]
    assert isinstance(adapter_kwargs, dict)
    rolling = adapter_kwargs["rolling_config"]
    assert isinstance(rolling, dict)
    assert revision_required_history(rolling, 28) == 1457
    assert revision_required_history(
        {"observation_contract": "masked_observed_counts_v1", "encoder_days": 35}, 7
    ) == 42


def test_validate_revision_parameter_groups_accepts_only_the_execution_surface() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "sft"
    )
    grouped = loads_tuning_spec(
        candidate.source_bytes, source_path=candidate.source_path
    ).defaults()
    model, training = grouped["model"], grouped["training"]
    assert validate_revision_parameter_groups(candidate, model, training) == (model, training)
    with pytest.raises(ExecutionPlanError, match="exactly the typed executable settings"):
        validate_revision_parameter_groups(candidate, {**model, "unknown": 1}, training)
    v3_candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "hmfnet"
    )
    v3_grouped = loads_tuning_spec(
        v3_candidate.source_bytes, source_path=v3_candidate.source_path
    ).defaults()
    for inactive_name in (
        "commercial_attention_heads",
        "commercial_attention_layers",
        "decoder_layers",
        "demand_dropout",
        "event_kernel_layers",
        "history_attention_layers",
        "patch_size",
    ):
        with pytest.raises(ExecutionPlanError, match="exactly the typed executable settings"):
            validate_revision_parameter_groups(
                v3_candidate,
                {**v3_grouped["model"], inactive_name: 1},
                v3_grouped["training"],
            )


def test_resolve_revision_rolling_config_returns_a_validate_revision_recipe_payload() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "hmfnet"
    )
    rolling = resolve_revision_rolling_config(
        candidate,
        model_parameters={
            **loads_tuning_spec(
                candidate.source_bytes, source_path=candidate.source_path
            ).defaults()["model"],
            "encoder_days": 28,
            "trunk_stages": 3,
            "n_heads": 8,
        },
        training_parameters={
            "learning_rate": 0.0003,
            "train_stride": 7,
            "train_windows": 3,
            "series_loss_weighting": "output_gradient",
        },
        epochs=2,
        seed=13,
    )
    validate_revision_recipe(rolling)
    assert rolling["variant"] == "v3" and rolling["epochs"] == 2 and rolling["seed"] == 13
    assert rolling["n_heads"] == 8
    assert rolling["series_loss_weighting"] == "output_gradient"
    v3 = rolling["v3"]
    assert isinstance(v3, dict)
    assert rolling["trunk_stages"] == 3 and v3["local_blocks"] == 3


def test_legacy_pact_recipe_defaults_missing_series_loss_weighting_to_uniform() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "pact"
    )
    legacy = json.loads(candidate.implementation_config_bytes)
    legacy.pop("series_loss_weighting")
    validate_revision_recipe(legacy)
    payload = (json.dumps(legacy, allow_nan=False) + "\n").encode()
    changed = dataclasses.replace(
        candidate,
        implementation_config_bytes=payload,
        implementation_config_sha256=hashlib.sha256(payload).hexdigest(),
    )

    plan = resolve_execution_plan(
        changed,
        Panel("functional.parquet", "id", "date", "qty"),
        _facts(),
        horizon=7,
        epochs=2,
    )

    rolling = plan.as_record()["adapter_kwargs"]["rolling_config"]
    assert rolling["variant"] == "v5"
    assert rolling["v5"] == legacy["v5"]
    assert rolling["series_loss_weighting"] == "uniform"


def test_global_left_padding_is_limited_to_mask_aware_sft_and_hmfnet() -> None:
    recipe = json.loads(
        (
            ROOT
            / "configs/experiments/multibranch_revisions/sft_campaign_default.json"
        ).read_text()
    )
    recipe.update(
        variant="v1_corrected",
        aggregate_history=False,
        mean_loss_weight=0.0,
        hierarchy_loss_weight=0.0,
        event_penalty_weight=0.0,
        factor_rank=2,
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    with pytest.raises(ValueError, match="does not implement global left padding"):
        validate_revision_recipe(recipe)
    with pytest.raises(ValueError, match="unpadded backbone"):
        RollingConfig(**recipe)


def test_validate_execution_controls_rejects_revision_gradient_accumulation() -> None:
    validate_execution_controls(
        "sft",
        device="cpu",
        batch_size=32,
        gradient_microbatch_size=32,
        has_generic_events=False,
    )
    with pytest.raises(ValueError, match="no gradient accumulation"):
        validate_execution_controls(
            "sft",
            device="cpu",
            batch_size=32,
            gradient_microbatch_size=16,
            has_generic_events=False,
        )


@pytest.mark.parametrize("adapter", ["tft", "tft_q"])
def test_validate_execution_controls_rejects_cpu_tft_fp16_before_runtime(adapter: str) -> None:
    with pytest.raises(ValueError, match="needs CUDA or MPS"):
        validate_execution_controls(
            adapter, device="cpu", batch_size=2,
            gradient_microbatch_size=None, has_generic_events=False,
            precision="16-mixed",
        )
    validate_execution_controls(
        adapter, device="cpu", batch_size=2,
        gradient_microbatch_size=None, has_generic_events=False,
        precision="32-true",
    )


def test_revision_execution_plan_authenticates_every_recipe_field_and_is_immutable() -> None:
    candidates = {
        item.id: item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id in {"sft", "hmfnet"}
    }
    for candidate in candidates.values():
        plan = resolve_execution_plan(
            candidate, Panel("functional.parquet", "id", "date", "qty"), _facts(),
            horizon=7, epochs=2, seed=13,
        )
        record = plan.as_record()
        effective = record["effective_parameters"]
        assert isinstance(effective, dict)
        expected = {
            name: value
            for name, value in effective.items()
            if not name.startswith(("data.", "implementation."))
            and name not in {"training.epochs", "training.seed"}
        }
        rolling = record["adapter_kwargs"]["rolling_config"]
        assert isinstance(rolling, dict)
        for field in tuple(rolling):
            mutated = copy.deepcopy(record)
            del mutated["adapter_kwargs"]["rolling_config"][field]
            assert not _execution_plan_matches(mutated, candidate, expected), field
        extra = copy.deepcopy(record)
        extra["adapter_kwargs"]["rolling_config"]["undeclared"] = 1
        assert not _execution_plan_matches(extra, candidate, expected)
        if candidate.adapter.endswith("_v3"):
            nested = rolling["v3"]
            assert isinstance(nested, dict)
            for field in tuple(nested):
                mutated = copy.deepcopy(record)
                del mutated["adapter_kwargs"]["rolling_config"]["v3"][field]
                assert not _execution_plan_matches(mutated, candidate, expected), field

        mutable_copy = plan.as_record()
        mutable_copy["adapter_kwargs"]["rolling_config"]["d_model"] = 128
        assert plan.as_record() == record
        frozen = dict(plan.adapter_kwargs)["rolling_config"]
        with pytest.raises(TypeError):
            frozen["d_model"] = 128


def test_revision_plan_uses_discovery_bytes_not_later_path_contents(tmp_path: pathlib.Path) -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "sft"
    )
    changed_profile = tmp_path / "model.yaml"
    changed_profile.write_text("not: the retained profile\n", encoding="utf-8")
    changed_recipe = tmp_path / "recipe.json"
    changed_recipe.write_text('{"variant":"v2","d_model":128}\n', encoding="utf-8")
    moved = dataclasses.replace(
        candidate,
        source_path=changed_profile,
        implementation_config_path=changed_recipe,
    )
    plan = resolve_execution_plan(
        moved, Panel("functional.parquet", "id", "date", "qty"), _facts(),
        horizon=7, epochs=2,
    )
    rolling = plan.as_record()["adapter_kwargs"]["rolling_config"]
    assert rolling["d_model"] == 64


@pytest.mark.parametrize(
    "model_id,mutation",
    [
        ("sft", lambda value: value.update(unknown=1)),
        ("sft", lambda value: value.pop("max_updates")),
        ("sft", lambda value: value.update(checkpoint_keep_last=1)),
        ("sft", lambda value: value.update(gradient_clip=0)),
        ("hmfnet", lambda value: value["v3"].update(recent=0)),
        ("hmfnet", lambda value: value.update(v3=None)),
    ],
)
def test_revision_recipe_is_rejected_in_parent_before_worker(model_id, mutation) -> None:
    candidate = next(
        item for item in discover_candidates(ROOT / "configs/models") if item.id == model_id
    )
    value = json.loads(candidate.implementation_config_bytes)
    mutation(value)
    payload = (json.dumps(value, allow_nan=False) + "\n").encode()
    changed = dataclasses.replace(
        candidate,
        implementation_config_bytes=payload,
        implementation_config_sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ExecutionPlanError, match="revision implementation config"):
        resolve_execution_plan(
            changed, Panel("functional.parquet", "id", "date", "qty"), _facts(),
            horizon=7, epochs=2,
        )


def test_custom_depth_settings_build_the_requested_model_depth() -> None:
    """YAML depth values build real layers, including four compact event-kernel refinements."""
    shallow = CustomModel(
        16, 28, n_attributes=2, n_series=8, n_known=4, horizon=7,
        factor_rank=0, n_heads=4, trunk_stages=2, decoder_layers=2,
        history_attention_layers=1, commercial_attention_layers=1,
        n_commercial=3, n_events=1, event_kernel_layers=1,
    )
    deep = CustomModel(
        16, 28, n_attributes=2, n_series=8, n_known=4, horizon=7,
        factor_rank=0, n_heads=4, trunk_stages=4, decoder_layers=4,
        history_attention_layers=4, commercial_attention_layers=4,
        n_commercial=3, n_events=1, event_kernel_layers=4,
    )
    assert deep.trunk.n_stages == 4
    assert len(deep.decoder.layers) == 4
    assert deep.demand.attention_layers == 4
    _, history_sequence = deep.demand(torch.zeros(2, 28))
    assert history_sequence.shape == (2, 4, 16)
    assert deep.commercial is not None
    assert deep.commercial.attention_layers == 4
    assert deep.commercial(torch.zeros(2, 3)).shape == (2, 16)
    assert deep.calendar is not None
    assert deep.calendar.kernel_layers_count == 4
    assert len(deep.calendar.kernel_layers) == 4
    event_output = deep.calendar(
        torch.zeros(2, 7, 1),
        present=torch.ones(2, 7, 1),
        series_index=torch.tensor([0, 1]),
    )
    assert event_output.shape == (2, 7, 16)
    assert sum(parameter.numel() for parameter in deep.parameters()) > sum(
        parameter.numel() for parameter in shallow.parameters()
    )


def test_hpo_depth_trial_reaches_the_real_custom_model() -> None:
    """A mixed HPO trial must not be reduced to the profile defaults before model assembly."""
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.adapter == "mft"
    )
    plan = resolve_execution_plan(
        candidate,
        Panel("functional.parquet", "id", "date", "qty", factor_group="state_id"),
        _facts(),
        horizon=7,
        epochs=2,
        seed=13,
        parameter_overrides={
            "model.encoder_days": 28,
            "model.trunk_stages": 2,
            "model.decoder_layers": 1,
            "model.history_attention_layers": 2,
            "model.commercial_attention_layers": 1,
            "model.event_kernel_layers": 2,
            "model.d_model": 64,
            "model.heads": 4,
            "model.patch_size": 7,
            "model.factor_rank": 8,
            "model.factor_dynamics": "iid",
            "model.demand_branch_dropout": 0.15,
            "model.covariate_dropout": 0.2,
            "training.learning_rate": 0.001,
        },
    )
    kwargs = dict(plan.adapter_kwargs)
    model = CustomModel(
        64, kwargs["encoder_days"], n_attributes=2, n_series=8, n_known=4, horizon=7,
        factor_rank=8, factor_dynamics="iid", n_heads=4, n_commercial=3, n_events=1,
        trunk_stages=kwargs["trunk_stages"], decoder_layers=kwargs["decoder_layers"],
        history_attention_layers=kwargs["history_attention_layers"],
        commercial_attention_layers=kwargs["commercial_attention_layers"],
        event_kernel_layers=kwargs["event_kernel_layers"],
    )
    assert model.trunk.n_stages == 2
    assert len(model.decoder.layers) == 1
    assert model.demand.attention_layers == 2
    assert model.commercial is not None and model.commercial.attention_layers == 1
    assert model.calendar is not None and model.calendar.kernel_layers_count == 2
    assert model.factor is not None and model.factor.factor_dynamics == "iid"


def test_baseline_plan_cannot_receive_custom_yaml_parameters() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.adapter == "seasonal_naive"
    )
    plan = resolve_execution_plan(
        candidate,
        Panel("functional.parquet", "id", "date", "qty"),
        _facts(),
        horizon=28,
        epochs=2,
    )
    assert dict(plan.adapter_kwargs) == {}
    assert dict(plan.readiness_settings) == {}


def test_shallow_and_tft_profiles_accept_only_declared_hpo_parameters() -> None:
    candidates = {item.adapter: item for item in discover_candidates(ROOT / "configs/models")}
    panel = Panel("functional.parquet", "id", "date", "qty")
    shallow = resolve_execution_plan(
        candidates["shallow_global"],
        panel,
        _facts(),
        horizon=7,
        epochs=2,
        seed=11,
        parameter_overrides={
            "model.num_leaves": 31,
            "model.min_data_in_leaf": 50,
            "model.num_boost_round": 100,
            "training.learning_rate": 0.01,
        },
    )
    assert dict(shallow.adapter_kwargs) == {
        "learning_rate": 0.01,
        "min_data_in_leaf": 50,
        "num_boost_round": 100,
        "num_leaves": 31,
        "seed": 11,
    }
    assert shallow.as_record()["readiness_settings"] == {}
    tft = resolve_execution_plan(
        candidates["tft"],
        panel,
        _facts(),
        horizon=7,
        epochs=2,
        seed=11,
        parameter_overrides={
            "model.hidden_size": 8,
            "model.attention_head_size": 1,
            "model.dropout": 0.0,
            "training.learning_rate": 0.0001,
        },
    )
    assert dict(tft.adapter_kwargs) == {
        "attention_head_size": 1,
        "dropout": 0.0,
        "hidden_size": 8,
        "learning_rate": 0.0001,
        "max_epochs": 2,
        "seed": 11,
    }
    assert tft.as_record()["readiness_settings"] == {}


def test_tft_tft_profile_freezes_the_reference_to_m5_translation() -> None:
    candidate = next(
        item for item in discover_candidates(ROOT / "configs/models") if item.adapter == "tft_q"
    )
    plan = resolve_execution_plan(
        candidate,
        Panel("functional.parquet", "id", "date", "qty", price_col="sell_price",
              known_future=("sell_price",)),
        _facts(),
        horizon=28,
        epochs=2,
        seed=11,
    )
    assert dict(plan.adapter_kwargs) == {
        "attention_head_size": 2,
        "dropout": 0.25,
        "hidden_continuous_size": 8,
        "hidden_size": 50,
        "lag_long_days": 42,
        "lag_short_days": 7,
        "learning_rate": 0.015,
        "max_encoder_length": 364,
        "max_epochs": 2,
        "min_encoder_length": 7,
        "output_size": 4,
        "loss_kind": "quantile",
        "seed": 11,
    }
    assert plan.as_record()["readiness_settings"] == {"min_encoder_length": 7}

    count_candidate = next(
        item for item in discover_candidates(ROOT / "configs/models")
        if item.adapter == "tft_hnb"
    )
    alternate = resolve_execution_plan(
        count_candidate,
        Panel("functional.parquet", "id", "date", "qty", price_col="sell_price",
              known_future=("sell_price",)),
        _facts(),
        horizon=28,
        epochs=2,
        seed=11,
        parameter_overrides=_revision_parameters(
            count_candidate, **{"model.loss_kind": "v3_hurdle_nb", "model.output_size": 3}
        ),
    )
    assert dict(alternate.adapter_kwargs) == {
        **dict(plan.adapter_kwargs), "loss_kind": "v3_hurdle_nb", "output_size": 3
    }
    with pytest.raises(ExecutionPlanError, match="outside its valid domain"):
        resolve_execution_plan(
            candidate,
            Panel("functional.parquet", "id", "date", "qty"),
            _facts(), horizon=28, epochs=2,
            parameter_overrides=_revision_parameters(
                candidate, **{"model.loss_kind": "v3_hurdle_nb"}
            ),
        )
