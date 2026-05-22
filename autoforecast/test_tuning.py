"""Functional contract for the one declared custom-model tuning surface."""

from __future__ import annotations

import pathlib
import sys
import types
from typing import cast

import pytest

from autoforecast.model_catalog import discover_candidates
from autoforecast.tuning import (
    SearchDomain,
    TuningError,
    TuningParameter,
    TuningSpec,
    load_tuning_spec,
    loads_tuning_spec,
    materialize_manual_parameters,
    materialize_runtime_parameters,
    materialize_trial_parameters,
    ray_param_space,
    suggest_optuna_parameters,
)
from autoforecast.tuning_support import tuning_support

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_loads_tuning_spec_parses_authenticated_candidate_bytes() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "sft"
    )
    spec = loads_tuning_spec(candidate.source_bytes, source_path=candidate.source_path)
    assert spec.model_id == candidate.id and spec.source_sha256 == candidate.source_sha256


def test_custom_model_tuning_yaml_loads_and_exposes_the_declared_defaults() -> None:
    candidates = discover_candidates(ROOT / "configs/models")
    candidate = next(item for item in candidates if item.id == "mft")
    spec = load_tuning_spec(candidate.source_path)
    assert spec.model_id == candidate.id
    assert spec.defaults() == {
        "model": {
            "covariate_dropout": 0.2,
            "commercial_attention_layers": 4,
            "d_model": 64,
            "decoder_layers": 4,
            "demand_branch_dropout": 0.15,
            "encoder_days": 28,
            "event_kernel_layers": 4,
            "factor_dynamics": "ar1",
            "factor_rank": 8,
            "heads": 4,
            "history_attention_layers": 4,
            "patch_size": 7,
            "trunk_stages": 4,
        },
        "training": {"learning_rate": 0.001},
    }
    assert {parameter.name for parameter in spec.parameters} == {
        "model.d_model",
        "model.heads",
        "model.patch_size",
        "model.factor_rank",
        "model.factor_dynamics",
        "model.demand_branch_dropout",
        "model.encoder_days",
        "model.covariate_dropout",
        "model.trunk_stages",
        "model.decoder_layers",
        "model.history_attention_layers",
        "model.commercial_attention_layers",
        "model.event_kernel_layers",
        "training.learning_rate",
    }


def _custom_spec():
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "mft"
    )
    return load_tuning_spec(candidate.source_path)


def test_optuna_samples_the_declared_yaml_contract() -> None:
    optuna = pytest.importorskip("optuna")
    spec = _custom_spec()
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        values = suggest_optuna_parameters(spec, trial)
        grouped = materialize_trial_parameters(spec, values)
        assert set(grouped) == {"model", "training"}
        learning_rate = grouped["training"]["learning_rate"]
        assert type(learning_rate) is float
        return cast(float, learning_rate)

    study.optimize(objective, n_trials=1)
    assert set(study.best_trial.params) == {parameter.name for parameter in spec.parameters}


def test_suggest_categorical_suggest_int_and_suggest_float_match_optuna_trial_contract() -> None:
    """The Protocol methods are stubs; exercise their real pinned-backend counterparts."""
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial(
        {
            "model.family": "ar1",
            "model.layers": 3,
            "training.learning_rate": 0.001,
        }
    )

    assert trial.suggest_categorical("model.family", ("iid", "ar1")) == "ar1"
    assert trial.suggest_int("model.layers", 1, 5, step=2) == 3
    assert trial.suggest_float("training.learning_rate", 0.0001, 0.01, log=True) == 0.001


def test_ray_param_space_has_one_native_domain_per_yaml_parameter() -> None:
    pytest.importorskip("ray.tune")
    spec = _custom_spec()
    space = ray_param_space(spec)
    assert set(space) == {parameter.name for parameter in spec.parameters}
    sampled: dict[str, object] = {}
    for name, domain in space.items():
        sampler = getattr(domain, "sample", None)
        assert callable(sampler)
        sampled[name] = sampler()
    assert materialize_trial_parameters(spec, sampled)


def test_ray_stepped_integer_domain_preserves_nonzero_offset(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("ray.tune")
    domain = SearchDomain(kind="int", low=31, high=127, step=32)
    spec = TuningSpec(
        model_id="offset_model",
        adapter="tft",
        parameters=(TuningParameter("model.hidden_size", 31, domain, domain),),
        constraints=(),
        source_path=tmp_path / "model.yaml",
        source_sha256="a" * 64,
    )
    ray_domain = ray_param_space(spec)["model.hidden_size"]
    assert tuple(ray_domain.categories) == (31, 63, 95, 127)
    for _ in range(50):
        sampled = {"model.hidden_size": ray_domain.sample()}
        assert materialize_trial_parameters(spec, sampled)


def test_trial_materialisation_refuses_missing_or_out_of_range_values() -> None:
    spec = _custom_spec()
    values = {parameter.name: parameter.default for parameter in spec.parameters}
    values.pop("model.d_model")
    with pytest.raises(TuningError, match="exactly"):
        materialize_trial_parameters(spec, values)
    values = {parameter.name: parameter.default for parameter in spec.parameters}
    values["model.d_model"] = 48
    with pytest.raises(TuningError, match="outside"):
        materialize_trial_parameters(spec, values)


def test_manual_runtime_domain_is_independent_of_narrow_hpo_search() -> None:
    spec = _custom_spec()
    manual = materialize_manual_parameters(spec, {"model.d_model": 40})
    assert manual["model"]["d_model"] == 40
    complete = {
        f"{namespace}.{name}": value
        for namespace, values in manual.items()
        for name, value in values.items()
    }
    assert materialize_runtime_parameters(spec, complete) == manual
    with pytest.raises(TuningError, match="outside its declared range"):
        materialize_trial_parameters(spec, complete)
    with pytest.raises(TuningError, match="outside its valid domain"):
        materialize_manual_parameters(spec, {"model.d_model": 41})


def test_manual_positive_integer_domain_has_no_artificial_hpo_ceiling(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
model: {id: positive_integer, adapter: tft}
hyperparameters:
  model.hidden_size:
    default: 4
    valid: {kind: int, low: 1, step: 1}
    search: {kind: int, low: 2, high: 8, step: 2}
""",
        encoding="utf-8",
    )
    spec = load_tuning_spec(path)

    assert materialize_manual_parameters(spec, {"model.hidden_size": 10})["model"] == {
        "hidden_size": 10
    }
    with pytest.raises(TuningError, match="outside its valid domain"):
        materialize_manual_parameters(spec, {"model.hidden_size": 0})
    with pytest.raises(TuningError, match="outside its declared range"):
        materialize_trial_parameters(spec, {"model.hidden_size": 10})


@pytest.mark.parametrize(
    ("search", "message"),
    [
        ("{kind: int, low: 2, step: 2}", r"exactly \['high', 'kind', 'low', 'step'\]"),
        ("{kind: int, low: 2, high: null, step: 2}", "integer bounds"),
    ],
)
def test_schema_two_integer_search_requires_finite_high(
    tmp_path: pathlib.Path, search: str, message: str
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        f"""schema_version: 2
model: {{id: finite_search, adapter: tft}}
hyperparameters:
  model.hidden_size:
    default: 4
    valid: {{kind: int, low: 1, step: 1}}
    search: {search}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match=message):
        load_tuning_spec(path)


def test_open_runtime_domain_keeps_finite_optuna_sampling_bounds(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
model: {id: finite_search, adapter: tft}
hyperparameters:
  model.hidden_size:
    default: 4
    valid: {kind: int, low: 1, step: 1}
    search: {kind: int, low: 2, high: 8, step: 2}
""",
        encoding="utf-8",
    )
    calls: list[tuple[str, int, int, int]] = []

    class Trial:
        def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int:
            calls.append((name, low, high, step))
            return 4

        def suggest_categorical(self, name: str, choices: tuple[object, ...]) -> object:
            raise AssertionError((name, choices))

        def suggest_float(
            self, name: str, low: float, high: float, *, log: bool = False
        ) -> float:
            raise AssertionError((name, low, high, log))

    assert suggest_optuna_parameters(load_tuning_spec(path), Trial()) == {
        "model.hidden_size": 4
    }
    assert calls == [("model.hidden_size", 2, 8, 2)]


def test_open_runtime_domain_keeps_finite_ray_sampling_bounds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = SearchDomain(kind="int", low=1, step=1)
    search = SearchDomain(kind="int", low=2, high=8, step=2)
    spec = TuningSpec(
        model_id="finite_search",
        adapter="tft",
        parameters=(TuningParameter("model.hidden_size", 4, valid, search),),
        constraints=(),
        source_path=tmp_path / "model.yaml",
        source_sha256="a" * 64,
    )

    class Tune:
        @staticmethod
        def choice(values: list[object]) -> tuple[object, ...]:
            return tuple(values)

    ray = types.ModuleType("ray")
    ray.tune = Tune
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert ray_param_space(spec) == {"model.hidden_size": (2, 4, 6, 8)}


@pytest.mark.parametrize(
    ("model_id", "parameter_name"),
    [
        ("sft", "model.n_heads"),
        ("sft", "model.trunk_stages"),
        ("sft", "model.decoder_layers"),
        ("sft", "model.history_attention_layers"),
        ("sft", "model.commercial_attention_layers"),
        ("sft", "model.event_kernel_layers"),
        ("sft", "model.commercial_attention_heads"),
        ("hmfnet", "model.n_heads"),
        ("hmfnet", "model.trunk_stages"),
    ],
)
def test_revision_catalog_manual_integer_domains_are_open(
    model_id: str, parameter_name: str
) -> None:
    candidate = next(
        item for item in discover_candidates(ROOT / "configs/models") if item.id == model_id
    )
    parameter = next(
        item
        for item in load_tuning_spec(candidate.source_path).parameters
        if item.name == parameter_name
    )
    assert parameter.valid == SearchDomain(kind="int", low=1, step=1)
    assert parameter.search.values is not None or parameter.search.high is not None


@pytest.mark.parametrize(
    ("model_id", "values"),
    [
        ("sft", {"model.d_model": 80, "model.n_heads": 6}),
        (
            "sft",
            {"model.d_model": 80, "model.commercial_attention_heads": 6},
        ),
        ("hmfnet", {"model.d_model": 80, "model.n_heads": 6}),
    ],
)
def test_revision_catalog_head_divisibility_is_authoritative(
    model_id: str, values: dict[str, object]
) -> None:
    candidate = next(
        item for item in discover_candidates(ROOT / "configs/models") if item.id == model_id
    )
    with pytest.raises(TuningError, match="must be divisible"):
        materialize_manual_parameters(load_tuning_spec(candidate.source_path), values)


@pytest.mark.parametrize(
    "values",
    [
        {"model.d_model": 40, "model.heads": 16},
        {"model.encoder_days": 29, "model.patch_size": 7},
    ],
)
def test_profile_owned_cross_parameter_constraints_reject_invalid_manual_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(TuningError, match="must be divisible"):
        materialize_manual_parameters(_custom_spec(), values)


def test_tft_runtime_rejects_more_attention_heads_than_hidden_units() -> None:
    spec = load_tuning_spec(ROOT / "configs/models/neural/tft/model.yaml")

    with pytest.raises(TuningError, match="less than or equal"):
        materialize_manual_parameters(
            spec,
            {"model.hidden_size": 8, "model.attention_head_size": 16},
        )


def test_divisible_constraint_refuses_a_zero_divisor_as_a_configuration_error(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
model: {id: zero_divisor, adapter: tft, family: neural, runtime: tft, status: candidate, tags: [development]}
hyperparameters:
  model.hidden_size:
    default: 4
    valid: {kind: int, low: 1, high: 8, step: 1}
    search: {kind: categorical, values: [4]}
  model.attention_head_size:
    default: 0
    valid: {kind: int, low: 0, high: 2, step: 1}
    search: {kind: categorical, values: [0, 1, 2]}
constraints:
  - {kind: divisible, dividend: model.hidden_size, divisor: model.attention_head_size}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="non-zero"):
        load_tuning_spec(path)


def test_divisible_constraint_refuses_an_invalid_independent_search_product(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
model: {id: invalid_product, adapter: tft, family: neural, runtime: tft, status: candidate, tags: [development]}
hyperparameters:
  model.hidden_size:
    default: 4
    valid: {kind: categorical, values: [4, 6]}
    search: {kind: categorical, values: [4, 6]}
  model.attention_head_size:
    default: 2
    valid: {kind: categorical, values: [2, 3]}
    search: {kind: categorical, values: [2, 3]}
constraints:
  - {kind: divisible, dividend: model.hidden_size, divisor: model.attention_head_size}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="independent HPO search domains"):
        load_tuning_spec(path)


def test_tuning_yaml_rejects_duplicate_keys_and_non_finite_categories(
    tmp_path: pathlib.Path,
) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """schema_version: 2
model:
  id: invalid_model
  adapter: tft
  family: neural
  runtime: tft
  status: candidate
  tags: [development]
hyperparameters:
  model.hidden_size:
    default: 16
    default: 32
    valid: {kind: categorical, values: [16, 32]}
    search: {kind: categorical, values: [16, 32]}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="duplicate YAML mapping key"):
        load_tuning_spec(duplicate)

    non_finite = tmp_path / "non-finite.yaml"
    non_finite.write_text(
        duplicate.read_text(encoding="utf-8")
        .replace("    default: 16\n    default: 32\n", "    default: 16\n")
        .replace("values: [16, 32]", "values: [16, .inf]"),
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="must be finite"):
        load_tuning_spec(non_finite)


def test_tuning_yaml_rejects_numbers_outside_binary64(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "overflow.yaml"
    huge = "1" + "0" * 400
    path.write_text(
        f"""schema_version: 2
model: {{id: overflow_model, adapter: tft}}
hyperparameters:
  model.hidden_size:
    default: 16
    valid: {{kind: float, low: 1, high: {huge}, log: false}}
    search: {{kind: categorical, values: [16]}}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="finite built-in number"):
        load_tuning_spec(path)


def test_schema_one_refuses_schema_two_constraints(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "v1-constraints.yaml"
    path.write_text(
        """schema_version: 1
model: {id: v1_constraints, adapter: tft}
hyperparameters:
  model.hidden_size:
    default: 4
    search: {kind: categorical, values: [4]}
  model.attention_head_size:
    default: 2
    search: {kind: categorical, values: [2]}
constraints:
  - {kind: divisible, dividend: model.hidden_size, divisor: model.attention_head_size}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="constraints require schema_version 2"):
        load_tuning_spec(path)


def test_schema_two_refuses_search_values_outside_runtime_validity(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 2
model:
  id: invalid_model
  adapter: tft
  family: neural
  status: candidate
  tags: [development]
  runtime: tft
hyperparameters:
  model.hidden_size:
    default: 16
    valid: {kind: int, low: 8, high: 64, step: 8}
    search: {kind: categorical, values: [16, 128]}
""",
        encoding="utf-8",
    )
    with pytest.raises(TuningError, match="subset of valid"):
        load_tuning_spec(path)


def test_every_registered_adapter_declares_whether_outer_loop_hpo_is_honest() -> None:
    candidates = discover_candidates(ROOT / "configs/models")
    supported = {
        "mft",
        "sft",
        "hmfnet",
        "pact",
        "shallow_global",
        "tft",
        "tft_q",
        "tft_hnb",

    }
    for candidate in candidates:
        eligibility = tuning_support(candidate.adapter)
        assert eligibility.supported is (candidate.adapter in supported)
        if eligibility.supported:
            assert load_tuning_spec(candidate.source_path).adapter == candidate.adapter
        else:
            with pytest.raises(TuningError, match="intentionally not tuneable"):
                load_tuning_spec(candidate.source_path)
