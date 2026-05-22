"""Thin installed-package interface to preparation and the existing campaign coordinator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

from autoforecast.frequency import advance, parse_frequency


def create_example(
    project: pathlib.Path, *, frequency: str = "D", recipe_version: int = 2,
) -> None:
    """Write a deterministic, non-client example without any repository resources."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    if recipe_version not in (1, 2):
        raise ValueError("example recipe_version must be 1 or 2")
    cadence = parse_frequency(frequency)
    starts = {
        "H": dt.datetime(2024, 1, 1),
        "D": dt.date(2024, 1, 1),
        "W": dt.date(2024, 1, 1),
        "M": dt.date(2024, 1, 31),
        "MS": dt.date(2024, 1, 1),
        "Y": dt.date(2024, 12, 31),
        "YS": dt.date(2024, 1, 1),
    }
    if cadence.code not in starts:
        raise ValueError("example frequency must be H, D, W, M, MS, Y, or YS")
    start = starts[cadence.code]
    points = [advance(start, cadence, index) for index in range(42)]
    example_id = "daily-example" if cadence.code == "D" else f"{cadence.code.lower()}-example"
    project.mkdir(parents=True, exist_ok=False)
    rows = [
        {
            "shop": shop,
            "sku": sku,
            "day": points[i],
            "units": float(2 + i % 7 + sku),
        }
        for shop, sku in (("a|b", 1), ("a", 2))
        for i in range(42)
    ]
    (project / "data").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), project / "data/observations.parquet")

    def write(name, value):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    write(
        "dataset.yaml",
        {
            "schema_version": 1,
            "dataset": {
                "id": example_id,
                "panel": {
                    "series_col": "shop",
                    "time_col": "day",
                    "target_col": "units",
                    "frequency": cadence.code,
                    "target_estimand": "sales_given_availability",
                    "season_length": 7,
                },
            },
        },
    )
    recipe = {
        "schema_version": recipe_version,
        "source_keys": ["shop", "sku"],
        "training_start": points[0].isoformat(),
        "validation_origins": [points[20].isoformat(), points[27].isoformat()],
        "validation_horizon": 7,
        "final_train_cutoff": points[34].isoformat(),
        "final_test_horizon": 7,
        "history_policy": "fixed",
        "new_series_policy": "error",
        "target_unit": "units",
        "feature_availability": {},
    }
    if recipe_version == 2:
        recipe["metric"] = "rmse"
    if cadence.code != "D":
        recipe["frequency"] = cadence.code
    (project / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    for name in ("seasonal_naive", "last_value"):
        write(
            f"models/{name}/model.yaml",
            {
                "schema_version": 1,
                "model": {
                    "id": name,
                    "adapter": name,
                    "family": "baseline",
                    "runtime": "builtin",
                    "status": "candidate",
                    "tags": ["example", "baseline"],
                },
            },
        )
    budget = {
        "maximum_epochs": 1,
        "patience": 1,
        "timeout_seconds": 600,
        "max_attempts": 1,
        "early_stopping_absolute_min_delta": 0.000001,
        "early_stopping_relative_min_delta": 0.0,
    }
    write(
        "campaign.yaml",
        {
            "schema_version": 5,
            "forecast_campaign": {
                "name": example_id,
                "purpose": "forecast_campaign",
                "dataset": {
                    "contract": "dataset.yaml",
                    **{
                        role: {"source_uri": f"local://{example_id}-{role}"}
                        for role in ("selection", "final_training", "final_score")
                    },
                },
                "models": {
                    "catalog": "models",
                    "candidates": [
                        {"model_id": name, "parameters": {"mode": "defaults"}}
                        for name in ("seasonal_naive", "last_value")
                    ],
                },
                "temporal": {
                    key: recipe[key]
                    for key in (
                        "training_start",
                        "validation_origins",
                        "validation_horizon",
                        "final_train_cutoff",
                        "final_test_horizon",
                    )
                }
                | {"initialization_origin": None},
                "seeds": [0],
                "initialization_arms": ["fresh"],
                "selection": {
                    "metric": "rmse",
                    "missing_actual": "sparse_zero" if recipe_version == 2 else "no_data",
                    "direction": "minimize",
                    "tie_break": [
                        "mean_score",
                        "score_stddev",
                        "model_id",
                        "variant_id",
                        "initialization",
                    ],
                },
                "final_evidence_role": "synthetic_pipeline_validation",
                "budgets": {
                    role: dict(budget) for role in ("initialization", "validation", "final_fit")
                },
                "compute": {
                    "device": "cpu",
                    "batch_size": 2,
                    "gradient_microbatch_size": None,
                    "memory_gib": 6.0,
                    "min_free_disk_gib": 0.1,
                },
                "tracking": {"mlflow": False},
            },
        },
    )


def prepare(
    project: pathlib.Path,
    recipe_path: pathlib.Path,
    contract_path: pathlib.Path,
    sources: dict[str, pathlib.Path],
    output_root: pathlib.Path,
    *, study_id: str | None = None, scoring_references: pathlib.Path | None = None,
) -> pathlib.Path:
    """Own one preparation runtime; return the verified benchmark manifest."""
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    from autoforecast import resources
    from autoforecast._exact_json import load_exact_json
    from autoforecast.benchmark import freeze_benchmark, validate_recipe
    from autoforecast.data_admission import load_dataset_contract

    recipe = load_exact_json(recipe_path)
    if type(recipe) is not dict:
        raise ValueError("preparation recipe must be an object")
    validate_recipe(recipe)
    load_dataset_contract(contract_path)
    profile = resources.spark_profile(python_udfs=False)
    spark = resources.acquire_local_runtime(
        SparkSession, SparkContext, profile=profile, app_name="autoforecast_prepare"
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        return freeze_benchmark(
            contract_path=contract_path,
            sources=sources,
            recipe=recipe,
            output_root=output_root,
            spark=spark, study_id=study_id, scoring_references=scoring_references,
        )
    finally:
        from scripts.run_autoforecast import _shutdown_owned_preflight_runtime

        _shutdown_owned_preflight_runtime(spark)


def run_campaign(
    *,
    project: pathlib.Path,
    request_path: pathlib.Path,
    benchmark_path: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path:
    """Compile and resume using the existing durable supervisor and selection/refit stages."""
    import hashlib

    import yaml
    from mlops._safe_files import publish_bytes_set

    from autoforecast._exact_yaml import loads_exact_yaml
    from autoforecast.benchmark import artifact_path, read_benchmark_manifest
    from autoforecast.campaign import CampaignStore, CampaignSupervisor
    from autoforecast.forecast_campaign import (
        compile_forecast_campaign,
        load_forecast_campaign_request,
    )
    from autoforecast.project import experiment_project

    parsed = load_forecast_campaign_request(request_path)
    benchmark_path, benchmark = read_benchmark_manifest(benchmark_path)
    original = request_path.read_bytes()
    request = loads_exact_yaml(original)
    if type(request) is not dict:
        raise ValueError("campaign request must be an object")
    contract = project / parsed.contract
    if hashlib.sha256(contract.read_bytes()).hexdigest() != benchmark["contract_sha256"]:
        raise ValueError("request dataset contract differs from preparation")
    if "study_id" in benchmark:
        request["forecast_campaign"]["selection"]["missing_actual"] = (
            "sparse_zero" if benchmark["recipe"]["actual_missing"] == "zero" else "no_data"
        )
    request["forecast_campaign"]["dataset"]["contract"] = (
        (benchmark_path.parent / "dataset.yaml").relative_to(project).as_posix()
    )
    payload = yaml.safe_dump(request, sort_keys=False).encode()
    frozen_request_dir = (
        project / ".autoforecast/requests" / hashlib.sha256(original + payload).hexdigest()
    )
    frozen_request_dir.mkdir(parents=True, exist_ok=True)
    publish_bytes_set(frozen_request_dir, {"original.yaml": original, "campaign.yaml": payload})
    request_path = frozen_request_dir / "campaign.yaml"
    with experiment_project(project):
        compiled = compile_forecast_campaign(
            request_path,
            benchmark_path=benchmark_path,
            selection_data_path=artifact_path(benchmark_path, benchmark, "selection"),
            final_training_data_path=artifact_path(benchmark_path, benchmark, "final_training"),
            final_score_data_path=artifact_path(benchmark_path, benchmark, "final_score"),
            compiled_root=project / ".autoforecast/compiled",
        )
        store = CampaignStore.open_or_create(
            compiled.plan_path, output_root=output_root, working_directory=project
        )
        CampaignSupervisor(store, poll_seconds=0.1).run()
        if store.state()["status"] != "completed":
            raise RuntimeError(
                f"campaign incomplete: {store.root}; inspect state.json and stage logs"
            )
        return store.root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoForecast: local observed-target time-series benchmarks"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("init-example", help="create a tiny two-baseline project")
    example.add_argument("project", type=pathlib.Path)
    example.add_argument(
        "--frequency",
        choices=("H", "D", "W", "M", "ME", "MS", "Y", "YE", "YS"),
        default="D",
    )
    convert = commands.add_parser("import-csv", help="map CSV observations to typed Parquet")
    convert.add_argument("source", type=pathlib.Path)
    convert.add_argument("output", type=pathlib.Path)
    convert.add_argument("--id-columns", nargs="+", required=True)
    convert.add_argument("--date-column", required=True)
    convert.add_argument("--target-column", required=True)
    convert.add_argument("--frequency", default="D")
    freeze = commands.add_parser("freeze", help="validate data and freeze a benchmark")
    freeze.add_argument("--project-root", type=pathlib.Path, required=True)
    freeze.add_argument("--recipe", type=pathlib.Path, default=pathlib.Path("recipe.json"))
    freeze.add_argument("--contract", type=pathlib.Path, default=pathlib.Path("dataset.yaml"))
    freeze.add_argument("--source", type=pathlib.Path, help="one prepared snapshot for all roles")
    for role in ("selection", "final-training", "validation-score", "final-score"):
        freeze.add_argument(f"--{role}", type=pathlib.Path)
    freeze.add_argument("--study-id", help="verify registration against an existing study")
    freeze.add_argument("--scoring-references", type=pathlib.Path)
    run = commands.add_parser("run", help="resume validation, selection, refit, and test scoring")
    run.add_argument("--project-root", type=pathlib.Path, required=True)
    run.add_argument("--request", type=pathlib.Path, default=pathlib.Path("campaign.yaml"))
    lookup = run.add_mutually_exclusive_group(required=True)
    lookup.add_argument("--benchmark", type=pathlib.Path)
    lookup.add_argument("--study-id")
    run.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path(".autoforecast/runs"))
    rescore = commands.add_parser("rescore", help="score retained forecasts against a new study")
    rescore.add_argument("--project-root", type=pathlib.Path, required=True)
    rescore.add_argument("--campaign", type=pathlib.Path, required=True)
    rescore.add_argument("--study-id", required=True)
    rescore.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "import-csv":
            from autoforecast.tabular import import_csv

            print(import_csv(
                args.source, args.output, source_keys=args.id_columns,
                time_col=args.date_column, target_col=args.target_column,
                frequency=args.frequency,
            ))
        elif args.command == "init-example":
            create_example(args.project.resolve(), frequency=args.frequency)
            print(args.project.resolve())
        else:
            project = args.project_root.resolve(strict=True)
            if args.command == "rescore":
                from pyspark import SparkContext
                from pyspark.sql import SparkSession
                from scripts.run_autoforecast import _shutdown_owned_preflight_runtime

                from autoforecast import resources
                from autoforecast.benchmark import resolve_study
                from autoforecast.rescore import rescore_campaign

                study_path, _ = resolve_study(project / ".autoforecast/benchmarks", args.study_id)
                spark = resources.acquire_local_runtime(
                    SparkSession, SparkContext, profile=resources.spark_profile(python_udfs=False),
                    app_name="autoforecast_rescore",
                )
                try:
                    spark.sparkContext.setLogLevel("ERROR")
                    print(rescore_campaign(project / args.campaign, study_path,
                                           project / args.output, spark))
                finally:
                    _shutdown_owned_preflight_runtime(spark)
            elif args.command == "freeze":
                from autoforecast._exact_json import load_exact_json
                recipe = load_exact_json(project / args.recipe)
                if type(recipe) is not dict:
                    raise ValueError("preparation recipe must be an object")
                roles = ["selection", "final_training", "final_score"]
                if recipe.get("schema_version") == 2:
                    roles.append("validation_score")
                sources = {
                    role: getattr(args, role) or args.source
                    for role in roles
                }
                if any(path is None for path in sources.values()):
                    parser.error("provide --source or every recipe role path")
                benchmark = prepare(
                    project,
                    project / args.recipe,
                    project / args.contract,
                    {role: project / path for role, path in sources.items()},
                    project / ".autoforecast/benchmarks",
                    study_id=args.study_id,
                    scoring_references=(project / args.scoring_references
                                        if args.scoring_references else None),
                )
                print(benchmark)
            else:
                from autoforecast.benchmark import resolve_study
                benchmark_path = (
                    resolve_study(project / ".autoforecast/benchmarks", args.study_id)[0]
                    if args.study_id else project / args.benchmark
                )
                print(
                    run_campaign(
                        project=project,
                        request_path=project / args.request,
                        benchmark_path=benchmark_path,
                        output_root=project / args.output_root,
                    )
                )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"autoforecast: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
