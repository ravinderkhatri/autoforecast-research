"""Malformed user inputs fail at the command boundary without starting a runtime."""

import json
import subprocess
import sys

import pytest
import yaml

from autoforecast.cli import create_example


def cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "autoforecast.cli", *map(str, arguments)],
        capture_output=True, text=True,
    )


def assert_input_error(result, message):
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert "SparkContext" not in result.stderr


@pytest.mark.parametrize("field,value,message", [
    ("source_keys", [["nested"]], "source_keys"),
    ("source_keys", ["shop", "shop"], "source_keys"),
    ("history_policy", [], "history_policy"),
    ("new_series_policy", {}, "new_series_policy"),
    ("training_start", 123, "ISO YYYY-MM-DD"),
    ("training_start", "20240101", "ISO YYYY-MM-DD"),
    ("validation_origins", "2024-01-21", "validation_origins"),
    ("validation_origins", [None], "ISO YYYY-MM-DD"),
    ("validation_horizon", True, "boundaries"),
    ("final_test_horizon", 10**20, "boundaries"),
    ("feature_availability", {"price": []}, "feature_availability"),
])
def test_freeze_rejects_malformed_recipe_before_starting_spark(tmp_path, field, value, message):
    project = tmp_path / "project"
    create_example(project)
    recipe = project / "recipe.json"
    value_dict = json.loads(recipe.read_bytes())
    value_dict[field] = value
    recipe.write_text(json.dumps(value_dict))
    assert_input_error(cli("freeze", "--project-root", project, "--source", "data"), message)
    assert not (project / ".autoforecast").exists()


@pytest.mark.parametrize("value", [
    {}, None, {"forecast_campaign": None},
    {"schema_version": 5, "forecast_campaign": {"dataset": []}},
])
def test_run_rejects_malformed_request_without_traceback(tmp_path, value):
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(value))
    assert_input_error(cli(
        "run", "--project-root", tmp_path, "--benchmark", "absent.json",
    ), "autoforecast:")
    assert not (tmp_path / ".autoforecast").exists()


@pytest.mark.parametrize("text,line", [
    ('id,date,value\n1,2024-01-01,"unclosed', 2),
    ('"unclosed,id,date,value\n', 1),
])
def test_csv_reports_malformed_quoting(tmp_path, text, line):
    source = tmp_path / "input.csv"
    source.write_text(text)
    output = tmp_path / "output.parquet"
    assert_input_error(cli(
        "import-csv", source, output, "--id-columns", "id", "--date-column", "date",
        "--target-column", "value",
    ), f"CSV row {line}: malformed CSV")
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("dataset", []), ("dataset", None), ("contract", None), ("contract", [])])
def test_run_rejects_malformed_dataset_mapping(tmp_path, field, value):
    project = tmp_path / "project"
    create_example(project)
    request = project / "campaign.yaml"
    payload = yaml.safe_load(request.read_text())
    if field == "dataset":
        payload["forecast_campaign"][field] = value
    else:
        payload["forecast_campaign"]["dataset"][field] = value
    request.write_text(yaml.safe_dump(payload))
    assert_input_error(cli(
        "run", "--project-root", project, "--benchmark", "absent.json",
    ), "dataset")
    assert not (project / ".autoforecast").exists()


@pytest.mark.parametrize("column", ["meter.id", "meter id", "meter`id", "mètre", "__af_id"])
def test_csv_refuses_unsupported_mapped_headers_before_output(tmp_path, column):
    source = tmp_path / "input.csv"
    source.write_text(f"{column},date,value\n001,2024-01-01,1\n")
    output = tmp_path / "output.parquet"
    assert_input_error(cli(
        "import-csv", source, output, "--id-columns", column,
        "--date-column", "date", "--target-column", "value",
    ), "rename")
    assert not output.exists()


@pytest.mark.parametrize("source", ["dataset", "recipe"])
def test_freeze_refuses_unsupported_identifiers_before_spark(tmp_path, source):
    project = tmp_path / "project"
    create_example(project)
    if source == "dataset":
        path = project / "dataset.yaml"
        record = yaml.safe_load(path.read_text())
        record["dataset"]["panel"]["target_col"] = "energy.kwh"
        path.write_text(yaml.safe_dump(record))
    else:
        path = project / "recipe.json"
        record = json.loads(path.read_bytes())
        record["source_keys"] = ["meter.id"]
        path.write_text(json.dumps(record))
    assert_input_error(cli("freeze", "--project-root", project, "--source", "data"), "rename")
    assert not (project / ".autoforecast").exists()
