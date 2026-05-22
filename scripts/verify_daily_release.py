"""Acceptance gate, executed from a wheel in a project outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

from autoforecast.benchmark import verify_benchmark
from autoforecast.data_admission import load_dataset_contract
from autoforecast.results import comparison_plot, export_results, read_results


def run_cli(*arguments: object, cwd: pathlib.Path) -> pathlib.Path:
    """Exercise the installed console entry point, including its argument and error handling."""
    executable = pathlib.Path(sys.executable).with_name("autoforecast")
    result = subprocess.run(
        [str(executable), *(str(value) for value in arguments)],
        cwd=cwd, capture_output=True, text=True,
    )
    if result.returncode:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
    return pathlib.Path(result.stdout.strip().splitlines()[-1]).resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=pathlib.Path)
    parser.add_argument("--csv", action="store_true", help="exercise the CSV input route")
    args = parser.parse_args()
    project = args.project.resolve()
    assert run_cli("init-example", project, cwd=project.parent) == project
    source = project / "data"
    if args.csv:
        import pyarrow as pa
        import pyarrow.csv as csv
        import pyarrow.parquet as pq
        import yaml

        # Exercise a second domain, fractional targets, renamed columns, and leading-zero IDs.
        rows = [
            {"site": row["shop"], "meter": f'{row["sku"]:03d}',
             "date": row["day"], "kwh": row["units"] / 4}
            for row in pq.read_table(source / "observations.parquet").to_pylist()
        ]
        # Arrow re-exports this public API from its compiled extension.
        csv.write_csv(  # pyright: ignore[reportPrivateImportUsage]
            pa.Table.from_pylist(rows), project / "observations.csv"
        )
        source = project / "csv-data"
        run_cli(
            "import-csv", project / "observations.csv", source / "observations.parquet",
            "--id-columns", "site", "meter", "--date-column", "date", "--target-column", "kwh",
            cwd=project,
        )
        contract = yaml.safe_load((project / "dataset.yaml").read_text())
        contract["dataset"]["id"] = "energy-example"
        contract["dataset"]["panel"].update(
            series_col="site", time_col="date", target_col="kwh",
            target_estimand="observed_nonnegative",
        )
        (project / "dataset.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
        recipe = json.loads((project / "recipe.json").read_text())
        recipe.update(source_keys=["site", "meter"], target_unit="kwh")
        (project / "recipe.json").write_text(json.dumps(recipe, indent=2))
    benchmark = run_cli(
        "freeze", "--project-root", project, "--source", source, cwd=project,
    )
    frozen = verify_benchmark(benchmark)[1]
    assert frozen["source_series_count"] == frozen["cohort_series_count"] == 2
    assert frozen["excluded_new_series"] == 0
    estimand = "observed_nonnegative" if args.csv else "sales_given_availability"
    assert load_dataset_contract(benchmark.parent / "dataset.yaml").panel(
        benchmark.parent / "selection"
    ).target_estimand == estimand
    run_arguments = ("run", "--project-root", project, "--benchmark", benchmark)
    campaign = run_cli(*run_arguments, cwd=project)
    results = read_results(campaign)
    assert len(results.runs) == 5 and results.rows.num_rows == 70
    assert results.metadata["target_estimand"] == estimand
    assert results.metadata["final_score"]["score"] == 0
    assert results.metadata["final_score"]["model_id"] == "seasonal_naive"
    export_results(results, project / "export")
    export_results(results, project / "export")
    from dataclasses import replace

    try:
        export_results(
            replace(results, metadata={**results.metadata, "conflict": True}), project / "export"
        )
    except OSError:
        pass
    else:
        raise AssertionError("conflicting export retry was accepted")
    comparison_plot(
        results, series_id=results.rows["series_id"][0].as_py(), output=project / "comparison.png"
    )
    for run in results.runs:
        forecast = json.loads(pathlib.Path(run["forecast_record_path"]).read_bytes())
        assert forecast["target_estimand"] == estimand
        run_path = pathlib.Path(run["run_path"])
        provenance = list(run_path.glob("attempts/*/*/provenance/manifest.json"))
        assert len(provenance) == 1
        record = json.loads(provenance[0].read_bytes())
        assert record["source"]["schema"] == "autoforecast.installed-library.v1"
        assert record["runtime"]["installed_packages"]
        assert record["data"]["fingerprint"]
        assert record["configuration"]
        workers = list(run_path.glob("attempts/*/*/evaluation/worker-runtime.json"))
        assert len(workers) == 1
        assert json.loads(workers[0].read_bytes())["spark_python_worker"]
    events = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (campaign / "events").glob("*.json")
    }
    resumed = run_cli(*run_arguments, cwd=project)
    assert resumed == campaign
    assert all(
        hashlib.sha256((campaign / "events" / name).read_bytes()).hexdigest() == value
        for name, value in events.items()
    )
    # A changed forecast cannot be presented as the same result. Restore bytes after proof.
    prediction = next(
        pathlib.Path(results.runs[0]["forecast_record_path"]).parent.glob("forecast/*.parquet")
    )
    original = prediction.read_bytes()
    try:
        prediction.write_bytes(original + b"tampered")
        try:
            read_results(campaign)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("changed forecast was accepted")
    finally:
        prediction.write_bytes(original)
    frozen_request = next((project / ".autoforecast/requests").glob("*/campaign.yaml"))
    original_request = frozen_request.read_bytes()
    try:
        frozen_request.write_bytes(original_request + b"\n# changed frozen configuration\n")
        try:
            run_cli(*run_arguments, cwd=project)
        except subprocess.CalledProcessError as error:
            assert error.returncode == 2
        else:
            raise AssertionError("changed frozen configuration was accepted")
    finally:
        frozen_request.write_bytes(original_request)
    print(
        json.dumps(
            {
                "status": "passed",
                "campaign": str(campaign),
                "prediction_rows": 70,
                "validation_runs": 4,
                "test_runs": 1,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
