"""Run a frozen Optuna or Ray study through its declared model-specific evaluator bridge."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import importlib
import pathlib
import sys
from typing import cast

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoforecast.hpo import (
    HPOError,
    TrialEvaluator,
    load_hpo_study,
    run_hpo_study,
    sync_hpo_selection_to_mlflow,
)


def _evaluator(reference: str) -> TrialEvaluator:
    if reference == "builtin":
        raise HPOError(
            "builtin forecast HPO must run as a forecast-campaign tune stage so the training "
            "and validation panels are deterministically projected from one authenticated source"
        )
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise HPOError("evaluator reference must be module:function")
    try:
        result = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise HPOError(f"could not load evaluator {reference!r}: {exc}") from exc
    if not callable(result):
        raise HPOError(f"evaluator {reference!r} is not callable")
    return cast(TrialEvaluator, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("study_config", type=pathlib.Path)
    parser.add_argument("--output-root", type=pathlib.Path, default=ROOT / ".autoforecast/hpo")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment", default="AutoForecast HPO")
    parser.add_argument(
        "--sync-selection-only",
        action="store_true",
        help="Retry only the idempotent MLflow selection-tag projection for a completed study.",
    )
    args = parser.parse_args()
    study = load_hpo_study(args.study_config, project_root=ROOT)
    if args.sync_selection_only:
        if not args.mlflow_tracking_uri:
            raise HPOError("--sync-selection-only requires --mlflow-tracking-uri")
        record = sync_hpo_selection_to_mlflow(
            args.output_root / "studies" / study.id,
            tracking_uri=args.mlflow_tracking_uri,
        )
        print(f"MLflow selection sync: {record}")
        return 0
    report = run_hpo_study(
        study,
        args.output_root,
        _evaluator(study.evaluator),
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
    print(f"Study report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
