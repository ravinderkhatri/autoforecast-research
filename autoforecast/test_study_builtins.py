"""Built-in models trained by name on a saved study, on a synthetic daily study."""

import datetime as dt
import hashlib
import importlib
import json
import pathlib
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import run_study
from autoforecast.study_builtins import BUILTIN_STUDY_MODELS, builtin_profile, builtin_study_model
from autoforecast.study_registration import StudyRegistry, register_study
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_tuning import spark as _spark

START = dt.date(2024, 1, 1)
DAYS = 121  # 1 January to 30 April 2024


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def _sales(series: int, day: int) -> float:
    """A weekly pattern with some zero days, different per series."""
    date = START + dt.timedelta(days=day)
    if (day + series) % 9 == 0:
        return 0.0
    return float(10 + 3 * series + 6 * (date.weekday() >= 5) + (day * 7 + series) % 4)


def daily_study(root: pathlib.Path) -> pathlib.Path:
    """Three daily series over four months: tuning, validation and test windows of 7 days."""
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([
        {"item": f"S{series}", "day": START + dt.timedelta(days=day), "value": _sales(series, day)}
        for series in range(3) for day in range(DAYS)
    ]), root / "source.parquet")
    value = {
        "schema_version": 1,
        "dataset": {"schema_version": 1, "dataset": {"id": "synthetic-daily", "panel": {
            "series_col": "item", "time_col": "day", "target_col": "value",
            "frequency": "D", "season_length": 7, "target_estimand": "observed_nonnegative",
        }}},
        "recipe": {
            "schema_version": 2, "source_keys": ["item"], "training_start": "2024-01-01",
            "validation_origins": ["2024-04-09"], "validation_horizon": 7,
            "final_train_cutoff": "2024-04-23", "final_test_horizon": 7,
            "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
            "feature_availability": {}, "metric": "mae", "history_missing": "zero",
            "actual_missing": "mask",
            "tuning_windows": [{"origin": "2024-03-26", "horizon": 7}],
        },
        "source": {"kind": "parquet", "path": "source.parquet"},
        "storage": {"root": "registry"},
    }
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.fixture(scope="module")
def study(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("builtin-study")
    path = daily_study(root)
    registry = StudyRegistry.from_config(path, spark=spark)
    return registry, register_study(path, registry=registry)


def _run(study, model, **options):
    registry, study_id = study
    return run_study(study_info={"study_id": study_id, "create_study": False,
                                 "publish_results": False},
                     model=model, registry=registry, **options)


@pytest.mark.parametrize("name", sorted(BUILTIN_STUDY_MODELS))
def test_each_built_in_model_trains_by_name_on_a_saved_study(study, name):
    """The name alone runs validation and test with default settings and keeps the model."""
    result = _run(study, name)
    assert result.status == "completed"
    assert result.validation is not None and result.testing is not None
    for evidence in (result.validation, result.testing):
        assert evidence["score"]["population"]["forecasted_series"] == 3
        assert evidence["score"]["mae"] >= 0
    # Default settings come from the study, with no search.
    expected = BUILTIN_STUDY_MODELS[name].defaults(types.SimpleNamespace(season_length=7))
    assert result.parameters == {f"model.{key}": value for key, value in expected.items()}
    assert result.tuning_path is None
    # The implementation identity names the model and its adapter source.
    implementation = result.provenance["implementation"]
    module = importlib.import_module(f"autoforecast.adapters.{BUILTIN_STUDY_MODELS[name].module}")
    assert implementation["name"] == name
    assert implementation["source_sha256"] == hashlib.sha256(
        pathlib.Path(module.__file__).read_bytes()).hexdigest()
    assert set(implementation["dependencies"]) == set(BUILTIN_STUDY_MODELS[name].packages)
    # The saved model: the fitted history for the history-based models, LightGBM files otherwise.
    model = pathlib.Path(result.model_links["test"]["path"])
    saved = sorted(path.name for path in model.iterdir())
    assert saved == (["occurrence.txt", "positive_size.txt"] if name == "shallow_global"
                     else ["history", "panel.json"])
    if name != "shallow_global":
        # The test fit's origin, season and own settings are recorded with its history.
        record = json.loads((model / "panel.json").read_text())
        assert (record["path"], record["origin"], record["season_length"]) == (
            "history", "2024-04-23", 7)
        assert record.get("window") == (7 if name == "window_average" else None)
        assert record.get("model") == (
            name if name in {"croston", "auto_ets", "auto_theta"} else None)
    # The run reloads with the same scores.
    assert load_forecast_workflow(result.path).testing_metrics == result.testing_metrics


def test_window_average_search_tries_each_window_once(study, tmp_path):
    """Four windows with a budget of eight: four trials, each window once."""
    profile = builtin_profile("window_average", types.SimpleNamespace(season_length=7))
    domain = {"kind": "categorical", "values": [7, 14, 21, 28]}
    profile["hyperparameters"]["model.window"].update(valid=domain, search=domain)
    profile["hpo"] = {"backend": "optuna", "max_trials": 8, "seed": 42}
    path = tmp_path / "window_average.json"
    path.write_text(json.dumps(profile))
    result = _run(study, "window_average", model_config=path)
    selection = json.loads((result.tuning_path / "selection.json").read_text())
    tried = sorted(
        json.loads((result.tuning_path / candidate["attempt_path"] / "metadata/attempt.json")
                   .read_text())["hyperparameters"]["model.window"]
        for candidate in selection["candidates"])
    assert tried == [7, 14, 21, 28]
    assert result.parameters["model.window"] in tried


def test_a_search_larger_than_its_budget_keeps_drawing_at_random(study, tmp_path):
    """Four windows with a budget of two: the seeded random search runs two trials."""
    profile = builtin_profile("window_average", types.SimpleNamespace(season_length=7))
    domain = {"kind": "categorical", "values": [7, 14, 21, 28]}
    profile["hyperparameters"]["model.window"].update(valid=domain, search=domain)
    profile["hpo"] = {"backend": "optuna", "max_trials": 2, "seed": 42}
    path = tmp_path / "window_average.json"
    path.write_text(json.dumps(profile))
    result = _run(study, "window_average", model_config=path)
    selection = json.loads((result.tuning_path / "selection.json").read_text())
    assert len(selection["candidates"]) == 2


def test_an_unknown_model_name_lists_the_built_in_models(study):
    with pytest.raises(ValueError, match="built-in model name: auto_ets, auto_theta, croston"):
        _run(study, "prophet")


def test_a_built_in_model_refuses_another_implementation(study):
    with pytest.raises(ValueError, match="declares its own implementation"):
        _run(study, "auto_ets", implementation={"name": "x", "version": "1",
                                                "source_sha256": "0" * 64, "dependencies": {}})


@pytest.mark.parametrize("name", sorted(BUILTIN_STUDY_MODELS))
def test_each_fit_refuses_settings_it_does_not_take(name):
    """Only the model's own settings reach it; training settings are refused."""
    fit, _ = builtin_study_model(name)
    settings = BUILTIN_STUDY_MODELS[name].defaults(types.SimpleNamespace(season_length=7))
    for parameters, training in (({**settings, "extra": 1}, {}), (settings, {"epochs": 1}),
                                 ({}, {})):
        request = types.SimpleNamespace(model_parameters=parameters, training_parameters=training)
        with pytest.raises(ValueError, match="study settings require"):
            fit(request)


def test_default_profiles_fix_each_setting_at_its_default():
    profile = builtin_profile("shallow_global", types.SimpleNamespace(season_length=7))
    assert profile["model"] == {"id": "shallow_global", "adapter": "external"}
    assert "hpo" not in profile
    learning_rate = profile["hyperparameters"]["model.learning_rate"]
    assert learning_rate == {"default": 0.05, "valid": {"kind": "categorical", "values": [0.05]},
                             "search": {"kind": "categorical", "values": [0.05]}}
    season = builtin_profile("seasonal_naive", types.SimpleNamespace(season_length=7))
    assert season["hyperparameters"] == {"model.season_length": {
        "default": 7, "valid": {"kind": "int", "low": 7, "high": 7, "step": 1},
        "search": {"kind": "int", "low": 7, "high": 7, "step": 1}}}
