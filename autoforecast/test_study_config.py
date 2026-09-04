"""Offline configuration checks use the same schema and semantic rules as registration."""

import json

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoforecast import study_config_schema, validate_study_config
from autoforecast.test_study_registration import configuration


def test_offline_config_checks_sql_without_runtime_or_sources(tmp_path, monkeypatch):
    from autoforecast import study_registration
    from autoforecast.integrations import delta

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["source"] = {"kind": "delta", "sql": "source.sql", "tables": {"input": "catalog.demo.data"}}
    (tmp_path / "source.sql").write_text("SELECT item, day, value FROM input")
    path.write_text(yaml.safe_dump(value))
    monkeypatch.setattr(study_registration.StudyRegistry, "_from_storage",
                        lambda *a, **k: pytest.fail("offline validation constructed a registry"))
    monkeypatch.setattr(delta, "read_sources", lambda *a, **k: pytest.fail("offline source read"))
    monkeypatch.setattr(delta, "latest_delta_version", lambda *a: pytest.fail("offline lookup"))
    result = validate_study_config(path)
    assert "credentials" in result.not_checked and "data quality" in result.not_checked
    assert result.configuration["recipe"]["frequency"] == "D"
    copy = result.configuration
    copy["recipe"]["metric"] = "unknown"
    assert result.configuration["recipe"]["metric"] == "mae"
    assert not (tmp_path / "registry").exists()
    (tmp_path / "source.sql").write_text("DELETE FROM input")
    with pytest.raises(ValueError, match="read-only SELECT"):
        validate_study_config(path)
    assert not (tmp_path / "registry").exists()


def test_schema_and_runtime_reject_unknown_and_conditional_fields(tmp_path):
    path = configuration(tmp_path)
    baseline = yaml.safe_load(path.read_text())
    schema = study_config_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(baseline)
    for alter in (
        lambda value: value.update(metirc="mae"),
        lambda value: value["recipe"].update(metirc="mae"),
        lambda value: value["storage"].pop("root"),
        lambda value: value["recipe"].update(metric="wrmsse"),
        lambda value: value["recipe"].update(validation_origins=[]),
        lambda value: value["recipe"].update(tuning_schedule={"horizon": 1, "stride": 1}),
    ):
        value = json.loads(json.dumps(baseline))
        alter(value)
        assert not validator.is_valid(value)
        path.write_text(yaml.safe_dump(value))
        with pytest.raises(ValueError):
            validate_study_config(path)
    baseline["recipe"].update(schema_version=3, validation_origins=[], validation_horizon=None)
    path.write_text(json.dumps(baseline))
    assert validator.is_valid(baseline)
    result = validate_study_config(path)
    assert result.configuration["recipe"]["validation_origins"] == []
    assert not (tmp_path / "registry").exists()


def test_offline_error_names_required_field_and_keeps_values_private(tmp_path):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["storage"].pop("root")
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="storage.root: required"):
        validate_study_config(path)
    value["storage"]["root"] = "registry"
    value["runtime"] = {"tracking": "private-token-example"}
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="runtime.tracking") as caught:
        validate_study_config(path)
    assert "private-token-example" not in str(caught.value)


@pytest.mark.parametrize("hook", [object(), type("OnlyValidate", (), {"validate": lambda s, v: None})()])
def test_preparation_methods_checked_before_registry_or_data(tmp_path, hook, monkeypatch):
    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry

    path = configuration(tmp_path)
    monkeypatch.setattr(StudyRegistry, "_from_storage",
                        lambda *a, **k: pytest.fail("invalid hook acquired runtime"))
    with pytest.raises(TypeError, match="callable validate.*prepare"):
        register_study(path, prepare=hook)
    assert not (tmp_path / "registry").exists()


@pytest.mark.parametrize("frequency,training,testing", [
    ("H", ("2024-01-01T01:00:00", "2024-01-01T02:00:00"),
     ("2024-01-01T03:00:00", "2024-01-01T04:00:00")),
    ("W", ("2024-01-03", "2024-01-10"), ("2024-01-17", "2024-01-24")),
    ("M", ("2024-01-31", "2024-02-29"), ("2024-03-31", "2024-04-30")),
    ("2D", ("2024-01-01", "2024-01-03"), ("2024-01-05", "2024-01-07")),
])
def test_direct_ranges_use_cadence_without_source_access(tmp_path, frequency, training, testing):
    import inspect
    from types import SimpleNamespace

    from autoforecast import DeltaTable
    from autoforecast.study_registration import _direct_configuration, register_study

    signature = inspect.signature(register_study)
    inputs = {name: parameter.default for name, parameter in signature.parameters.items()
              if name not in {"config", "registry", "spark", "prepare", "log_level"}}
    inputs.update(training_data=DeltaTable("training"), testing_data=DeltaTable("testing"),
                  series_id=["item"], date_col="at", target_col="value", target_unit="units",
                  frequency=frequency, metric="mae", training_date_range=training,
                  testing_date_range=testing)
    value, _ = _direct_configuration(SimpleNamespace(root=tmp_path, runtime={}), inputs)
    assert value["recipe"]["final_test_horizon"] == 2
    assert value["recipe"]["validation_origins"] == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("storage,field", [
    ({"root": "s3://bucket/studies"}, "storage.root"),
    ({"root": "   "}, "storage.root"),
    ({"root": "registry", "scratch_root": "https://private.invalid/value"}, "storage.scratch_root"),
    ({"root": "registry", "namespace": "bad..name"}, "storage.namespace"),
    ({"root": "registry", "namespace": "demo", "tables": {"unknown": "demo.data"}}, "storage.tables"),
    ({"root": "registry", "namespace": "demo", "tables": {"studies": "demo.study_results"}},
     "storage.tables"),
])
def test_offline_storage_errors_before_runtime(tmp_path, storage, field, monkeypatch):
    from autoforecast.study_registration import StudyRegistry

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["storage"] = storage
    path.write_text(yaml.safe_dump(value))
    monkeypatch.setattr(StudyRegistry, "__init__",
                        lambda *a, **k: pytest.fail("offline validation constructed storage"))
    with pytest.raises(ValueError, match=field) as caught:
        validate_study_config(path)
    assert "private.invalid" not in str(caught.value)
    assert not (tmp_path / "registry").exists()


def test_schema_nested_settings_match_offline_checks(tmp_path):
    from autoforecast.integrations.delta import SCHEMAS

    path = configuration(tmp_path)
    baseline = yaml.safe_load(path.read_text())
    validator = Draft202012Validator(study_config_schema())
    invalid = [
        ("storage", {"root": "s3://example/path"}),
        ("storage", {"root": "   "}),
        ("storage", {"root": "registry", "namespace": "demo", "tables": {"typo": "demo.x"}}),
        ("preparation", {"": 1}),
        ("preparation", {"  ": 1}),
        ("runtime", {"implementation": {"surprise": "private-value"}}),
        ("runtime", {"implementation": {"source_roots": {"wrong name": "src"}}}),
        ("runtime", {"implementation": {"source_roots": {"model": "src"}, "include": ["../x"]}}),
    ]
    for field, settings in invalid:
        value = {**baseline, field: settings}
        assert not validator.is_valid(value), (field, settings)
        path.write_text(yaml.safe_dump(value))
        with pytest.raises(ValueError) as caught:
            validate_study_config(path)
        assert "private-value" not in str(caught.value)
    for role in SCHEMAS:
        value = {**baseline, "storage": {"root": "registry", "namespace": "demo",
                                        "tables": {role: "demo.override_" + role}}}
        assert validator.is_valid(value)
        path.write_text(yaml.safe_dump(value))
        checked = validate_study_config(path)
        assert "source SQL syntax" not in checked.checked
    value["storage"]["tables"] = {"studies": "demo.study_results"}
    assert validator.is_valid(value)  # Effective-name uniqueness is a semantic check.
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="distinct names"):
        validate_study_config(path)


@pytest.mark.parametrize("field,settings,expected", [
    ("source", {"kind": "parquet", "path": "source.parquet", "misspelt": "private-value"}, "misspelt"),
    ("source", {"kind": "delta", "sql": "source.sql", "tables": {"data": "demo.data"},
                "misspelt": "private-value"}, "misspelt"),
    ("source", {"kind": "delta", "sql": "source.sql"}, "tables"),
    ("source", {"kind": "parquet"}, "path"),
    ("references", {"path": "refs.parquet", "typo": "private-value"}, "typo"),
    ("references", {"prepared": True, "typo": "private-value"}, "typo"),
    ("references", {"weight_col": "revenue", "missing_history": "na", "duplicate_policy": "error",
                    "typo": "private-value"}, "typo"),
    ("references", {"weight_col": "revenue", "duplicate_policy": "error"}, "missing_history"),
    ("references", {"path": "refs", "prepared": True}, "exactly one"),
])
def test_nested_schema_errors_name_the_field_without_showing_values(tmp_path, field, settings, expected):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value[field] = settings
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match=expected) as caught:
        validate_study_config(path)
    assert field in str(caught.value) and "private-value" not in str(caught.value)
    assert not (tmp_path / "registry").exists()
