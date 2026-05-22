"""Small shared-study HPO/native acceptance using caller-supplied model configuration files."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import pathlib
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from autoforecast.benchmark import verify_benchmark
from autoforecast.results import read_results

from scripts.verify_timeseries_release import run_cli


def verify_hpo_reuse_identity(project: pathlib.Path, selection: pathlib.Path) -> None:
    """A valid retained trial must not be reusable under a changed study or training protocol."""
    from autoforecast.hpo import load_hpo_study
    from autoforecast.model_catalog import discover_candidates
    from autoforecast.parameter_sources import (
        ParameterSourceError,
        parameter_resolution_from_selection,
    )
    from autoforecast.project import experiment_project

    with experiment_project(project):
        candidate = next(item for item in discover_candidates(project / "models")
                         if item.id == "sft")
        study = load_hpo_study(project / "hpo.yaml", project_root=project)
        resolution = parameter_resolution_from_selection(
            candidate, study, selection, kind="reuse_selection", project_root=project,
        )
        source = cast(dict, resolution["source"])
        protocol = source["protocol"]
        assert protocol["library"]["schema"] == "autoforecast.installed-library.v1"
        assert source["selected_source_commit"] is None
        for field in ("study_id", "boundary", "evaluation_contract_sha256", "precision", "library"):
            changed = copy.deepcopy(protocol)
            if field in {"study_id", "evaluation_contract_sha256"}:
                changed["study_evaluation"][field] = "0" * 64
            elif field == "boundary":
                changed["study_evaluation"][field] = "validation/2024-04-21/7/D"
            elif field == "precision":
                changed[field] = "16-mixed"
            else:
                changed[field]["artifact_sha256"] = "0" * 64
            try:
                parameter_resolution_from_selection(
                    candidate, study, selection, kind="reuse_selection", project_root=project,
                    expected_protocol=changed,
                )
            except ParameterSourceError as error:
                assert "protocol differs" in str(error), str(error)
            else:
                raise AssertionError(f"HPO reuse accepted changed {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=pathlib.Path)
    parser.add_argument("--model-profile", type=pathlib.Path, required=True)
    parser.add_argument("--implementation-config", type=pathlib.Path, required=True)
    parser.add_argument("--metric", choices=("rmse", "mae", "wape"), default="mae")
    args = parser.parse_args()
    project = args.project.resolve()
    print(
        "Execution-test population: two synthetic series, 126 days each, no exclusions. "
        "Each seven-period boundary requires 14 forecasts. The tuning actuals are complete; "
        "two outer "
        "validation actuals are missing and masked. These fits make no accuracy claim.", flush=True,
    )
    run_cli("init-example", project, cwd=project.parent)
    start = dt.date(2024, 1, 1)
    rows = [{
        "series": series, "date": start + dt.timedelta(days=offset),
        "sales": None if (series, offset) in {("a", 112), ("b", 113)}
        else (offset + index) % 5 + 1, "state": "exact",
    } for index, series in enumerate(("a", "b")) for offset in range(126)]
    labels = project / "validation-actuals.parquet"
    pq.write_table(pa.Table.from_pylist(rows), labels)
    # Training requires complete caller-supplied observations; scoring retains missing labels.
    history_rows = [{**row, "sales": ((row["date"] - start).days
                                     + (0 if row["series"] == "a" else 1)) % 5 + 1}
                    for row in rows]
    pq.write_table(pa.Table.from_pylist(history_rows), project / "data/observations.parquet")
    (project / "dataset.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "dataset": {"id": "native-study-example", "panel": {
            "series_col": "series", "time_col": "date", "target_col": "sales",
            "exposure_col": "state", "frequency": "D", "season_length": 7,
            "target_estimand": "sales_given_availability",
        }},
    }))
    def day(offset):
        return (start + dt.timedelta(days=offset)).isoformat()
    recipe = {
        "schema_version": 2, "source_keys": ["series"], "training_start": day(0),
        "tuning_windows": [{"origin": day(90), "horizon": 7}],
        "validation_origins": [day(111)], "validation_horizon": 7,
        "final_train_cutoff": day(118), "final_test_horizon": 7,
        "history_policy": "fixed", "new_series_policy": "error", "target_unit": "counts",
        "feature_availability": {}, "metric": args.metric, "actual_missing": "mask",
    }
    (project / "recipe.json").write_text(json.dumps(recipe))
    profile = yaml.safe_load(args.model_profile.read_text())
    assert profile["model"]["id"] == profile["model"]["adapter"] == "sft"
    implementation = args.implementation_config.read_bytes()
    assert hashlib.sha256(implementation).hexdigest() == profile["implementation"]["sha256"]
    # The frozen study overrides this valid recipe field without changing its provenance.
    implementation_config = json.loads(implementation)
    implementation_config["selection_metric"] = "wrmsse"
    implementation = (json.dumps(implementation_config, indent=2) + "\n").encode()
    profile["implementation"]["sha256"] = hashlib.sha256(implementation).hexdigest()
    (project / "implementation.json").write_bytes(implementation)
    profile["implementation"]["config"] = "implementation.json"
    compact = {
        "model.encoder_days": 28, "model.d_model": 8, "model.n_heads": 2,
        "model.trunk_stages": 1, "model.decoder_layers": 1, "model.history_attention_layers": 1,
        "model.commercial_attention_layers": 1, "model.event_kernel_layers": 1,
        "training.train_windows": 1, "training.train_stride": 7,
    }
    for name, parameter in profile["hyperparameters"].items():
        parameter["default"] = compact.get(name, parameter["default"])
        parameter["search"] = {"kind": "categorical", "values": [parameter["default"]]}
    (project / "models/sft").mkdir()
    (project / "models/sft/model.yaml").write_text(yaml.safe_dump(profile))
    request_path = project / "campaign.yaml"
    request = yaml.safe_load(request_path.read_text())
    campaign = request["forecast_campaign"]
    campaign["name"] = "native-study-hpo"
    campaign["models"]["candidates"] = [{
        "model_id": "sft", "parameters": {"mode": "tune", "study_config": "hpo.yaml"},
    }]
    campaign["temporal"].update({key: recipe[key] for key in (
        "training_start", "validation_origins", "validation_horizon",
        "final_train_cutoff", "final_test_horizon",
    )})
    campaign["selection"].update(metric=args.metric, missing_actual="no_data")
    campaign["compute"].update(device="cpu", batch_size=2, precision="32-true")
    for budget in campaign["budgets"].values():
        budget.update(maximum_epochs=2, patience=2)
    request_path.write_text(yaml.safe_dump(request))
    frozen_path = run_cli("freeze", "--project-root", project, "--source", "data",
                          "--validation-score", labels, cwd=project)
    _, frozen = verify_benchmark(frozen_path)
    study = {
        "id": "native-inner-study", "model_config": "models/sft/model.yaml", "backend": "optuna",
        "direction": "minimize", "objective": "validation/objective", "max_trials": 1,
        "top_k": 1, "seed": 7, "comparability_fingerprint": frozen["study_id"],
        "evaluator": "builtin", "data": {
            "id": "native-inner-data", "fingerprint": frozen["artifacts"]["selection"],
            "observed_date_range": [day(0), day(111)], "training_date_range": [day(0), day(90)],
            "validation_date_range": [day(0), day(90)],
            "validation_target_date_range": [day(91), day(97)],
            "test_date_range": [day(98), day(104)], "test_target_date_range": [day(98), day(104)],
            "training_series": 2, "validation_series": 2, "test": "reserved_not_evaluated",
        },
    }
    (project / "hpo.yaml").write_text(yaml.safe_dump({"schema_version": 1, "study": study}))
    run = run_cli("run", "--project-root", project, "--study-id", frozen["study_id"], cwd=project)
    result = read_results(run)
    assert result.metadata["study_id"] == frozen["study_id"]
    assert len(result.runs) == 2 and result.rows.num_rows == 28
    assert result.metadata["final_score"]["status"] == "scored"
    selections = list(run.glob("**/hpo/studies/native-inner-study/selection.json"))
    assert len(selections) == 1
    verify_hpo_reuse_identity(project, selections[0])
    governed_paths = list(selections[0].parent.glob(
        "trials/*/attempts/*/evaluation/governed-run.json",
    ))
    assert len(governed_paths) == 1
    governed = json.loads(governed_paths[0].read_bytes())
    assert governed["study_evaluation"]["boundary"] == f"tuning/{day(90)}/7/D"
    assert governed["study_evaluation"]["study_id"] == frozen["study_id"]
    comparison = next(governed_paths[0].parent.parent.glob(
        "source-run/*/evaluation/candidate-comparison.json",
    ))
    inner = json.loads(comparison.read_bytes())
    assert inner["candidates"][0]["score"]["n_scored"] == 14
    resumed = run_cli(
        "run", "--project-root", project, "--study-id", frozen["study_id"], cwd=project,
    )
    assert resumed == run
    campaign["name"] = "native-study-reused"
    campaign["models"]["candidates"][0]["parameters"] = {
        "mode": "reuse_tuned", "study_config": "hpo.yaml", "selection": str(selections[0]),
    }
    request_path.write_text(yaml.safe_dump(request))
    reused = run_cli(
        "run", "--project-root", project, "--study-id", frozen["study_id"], cwd=project,
    )
    reused_result = read_results(reused)
    assert reused_result.metadata["final_score"]["score"] == result.metadata["final_score"]["score"]
    print(f"PASS: frozen inner HPO, native selection/refit, results, resume and reuse: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
