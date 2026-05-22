"""Run corrected MFT, SFT, HMFNet or PACT on public M5 history, without final-label access.

This explicit development entry is intentionally not a final-lock/candidate-promotion path.
Tests: ``autoforecast/test_multibranch_revisions.py`` (CLI and real-data fit/replay).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pyarrow as pa
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoforecast._exact_json import load_exact_json  # noqa: E402
from autoforecast.adapters.mft import fit_predict_revision  # noqa: E402
from autoforecast.benchmarks.m5 import (  # noqa: E402
    M5_LEVELS,
    build_revision_calendar,
    panel_generation_lock,
)
from autoforecast.hierarchy_metric import HierarchyMetric, read_forecast_grid  # noqa: E402
from autoforecast.panel import Panel  # noqa: E402
from autoforecast.rolling_data import PanelArrays  # noqa: E402
from autoforecast.rolling_training import RollingConfig  # noqa: E402

from provenance.lineage import directory_digest  # noqa: E402

CODE_SOURCES = (
    *(ROOT / name for name in ("autoforecast", "forecast", "spec", "provenance")),
    pathlib.Path(__file__).resolve(),
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
)


def _scoring_binding() -> dict:
    """Identify scoring source/runtime before and after execution; tested by the CLI score test."""
    return dict(
        sources={
            str(path): directory_digest(path, suffixes=(".py",) if path.is_dir() else ())
            for path in CODE_SOURCES
        },
        runtime=dict(
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            pyarrow_version=pa.__version__,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Validate a recipe and launch only the requested development run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument(
        "--panel", type=pathlib.Path, default=ROOT / "data/m5_public_training_d1913_v2"
    )
    parser.add_argument(
        "--calendar", type=pathlib.Path, default=ROOT / "data/m5_revision_calendar_v2.parquet"
    )
    parser.add_argument("--m5-dir", type=pathlib.Path, default=ROOT / "data/m5")
    parser.add_argument("--origin", default="2016-04-24")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--replay-checkpoint", type=pathlib.Path)
    parser.add_argument(
        "--resume-training-checkpoint",
        type=pathlib.Path,
        help="Continue this --out from its latest training_resume.json",
    )
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--validation-every-epochs", type=int)
    parser.add_argument("--validation-backend", choices=("torch", "numpy"))
    parser.add_argument("--checkpoint-every-updates", type=int)
    parser.add_argument("--checkpoint-every-seconds", type=float)
    parser.add_argument("--checkpoint-keep-last", type=int)
    parser.add_argument("--progress", choices=("auto", "plain", "off"))
    parser.add_argument(
        "--epochs",
        type=int,
        help="Fixed passes per selection/refit phase; no update-count selection",
    )
    parser.add_argument(
        "--actuals", type=pathlib.Path, default=ROOT / "data/m5_public_observed_d1914_d1941_v2"
    )
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    if (
        sum((bool(args.replay_checkpoint), bool(args.resume_training_checkpoint), args.score_only))
        > 1
    ):
        raise ValueError(
            "scoring cannot also replay or resume; all three modes are mutually exclusive"
        )
    if args.resume_training_checkpoint:
        if args.resume_training_checkpoint.resolve() != args.out.resolve() / "training_resume.json":
            raise ValueError("resume requires this output's latest training_resume.json")
        if not args.resume_training_checkpoint.is_file() or not args.calendar.is_file():
            raise ValueError("resume requires an existing checkpoint pointer and bound calendar")
    try:
        origin = dt.date.fromisoformat(args.origin)
    except ValueError as exc:
        raise ValueError("origin must be canonical ISO YYYY-MM-DD") from exc
    if origin.isoformat() != args.origin:
        raise ValueError("origin must be canonical ISO YYYY-MM-DD")
    if origin > dt.date(2016, 4, 24):
        raise ValueError("this command cannot train or select using later/archived-final labels")
    forbidden = "organizer_private"
    if any(
        forbidden in str(path.resolve())
        for path in (args.panel, args.calendar, args.m5_dir, args.actuals)
    ):
        raise ValueError("archived-private input paths are outside this development command")
    record = load_exact_json(args.profile)
    if not isinstance(record, dict):
        raise ValueError("profile must be a JSON object")
    config = RollingConfig(**record)
    if args.epochs is not None:
        config = dataclasses.replace(config, epochs=args.epochs)
    overrides = {
        name: getattr(args, name)
        for name in (
            "validation_every_epochs",
            "validation_backend",
            "checkpoint_every_updates",
            "checkpoint_every_seconds",
            "checkpoint_keep_last",
            "progress",
        )
        if getattr(args, name) is not None
    }
    config = dataclasses.replace(config, **overrides)
    config_record = json.loads(json.dumps(dataclasses.asdict(config)))
    if args.out.exists() and not args.score_only and not args.resume_training_checkpoint:
        raise ValueError("output exists; use a fresh run directory")
    if not args.score_only:
        # Resolve every declared input/output before a calendar builder can create anything.
        # An extra Parquet file inside a sales dataset corrupts its schema even without overwrite.
        calendar_path, output_path = args.calendar.resolve(), args.out.resolve()
        protected = [
            args.panel.resolve(),
            args.actuals.resolve(),
            args.m5_dir.resolve(),
            args.profile.resolve(),
            *CODE_SOURCES,
            *(ROOT / name for name in ("scripts", "configs", "llm", "docs", ".git", ".venv")),
        ]
        if args.replay_checkpoint:
            protected.append(args.replay_checkpoint.resolve().parent)
        if (
            calendar_path.is_relative_to(output_path)
            or output_path.is_relative_to(calendar_path)
            or any(
                candidate.is_relative_to(source) or source.is_relative_to(candidate)
                for candidate in (calendar_path, output_path)
                for source in protected
            )
        ):
            raise ValueError("calendar/run outputs overlap an input, source or each other")
    panel = Panel(
        path=str(args.panel),
        series_col="id",
        time_col="date",
        target_col="qty",
        exposure_col="observation_state",
        known_future=("sell_price",),
        price_col="sell_price",
        static=("dept_id", "cat_id", "state_id", "store_id"),
        factor_group="state_id",
        product_col="item_id",
        store_col="store_id",
        category_col="dept_id",
        frequency="D",
        season_length=7,
    )
    if args.score_only:
        if config.progress != "off":
            print(
                "scoring completed full-grid forecast with NumPy reference",
                file=sys.stderr,
                flush=True,
            )
        if args.replay_checkpoint:
            raise ValueError("scoring cannot also replay a checkpoint")
        if args.actuals.resolve() == args.panel.resolve():
            raise ValueError("scoring requires a distinct actuals panel")
        completion_path = args.out / "COMPLETE.json"
        completion_hash = directory_digest(completion_path)
        complete = load_exact_json(completion_path)
        if not isinstance(complete, dict) or not isinstance(complete.get("binding"), dict):
            raise ValueError("scoring requires a complete revision record")
        forecast_path = args.out / "forecast.parquet"
        checkpoint_path = args.out / "model.pt"
        binding = complete["binding"]
        scoring = _scoring_binding()
        if any(
            binding["inputs"].get(path) != digest for path, digest in scoring["sources"].items()
        ) or any(binding.get(key) != value for key, value in scoring["runtime"].items()):
            raise ValueError("scoring source/runtime differs from the sealed run")
        if (
            binding["origin"] != args.origin
            or binding["config"] != config_record
            or binding["schema"]
            != (
                "multibranch_revision_v5"
                if config.variant == "v5"
                else "multibranch_revision_v3"
                if config.variant == "v3"
                else "multibranch_revision_v2"
            )
            or binding["horizon"] != 28
            or binding["levels"] != [list(v) for v in M5_LEVELS]
            or complete["forecast_sha256"] != directory_digest(forecast_path)
            or binding["inputs"][str(args.panel.resolve())] != directory_digest(args.panel)
        ):
            raise ValueError("score input/config/forecast binding differs from the sealed run")
        if not checkpoint_path.is_file() or complete.get("checkpoint_sha256") != directory_digest(
            checkpoint_path
        ):
            raise ValueError("score checkpoint binding differs from the sealed run")
        score_calendar = args.calendar.resolve()
        if (
            not score_calendar.is_file()
            or set(binding["inputs"])
            != {str(args.panel.resolve()), str(score_calendar), *scoring["sources"]}
            or binding["inputs"].get(str(score_calendar)) != directory_digest(score_calendar)
        ):
            raise ValueError("score calendar binding differs from the sealed run")
        with panel_generation_lock(args.panel, exclusive=False):
            data = PanelArrays.read(
                panel,
                args.origin,
                28,
                max_bytes=config.max_array_bytes,
                route_lower_bound_to_exact=config.variant == "v5",
            )
            if list(data.series_ids) != binding["series_ids"]:
                raise ValueError("scoring population differs from the sealed run")
            metric = HierarchyMetric.build(
                data.metadata, data.target, data.price[:, : data.target.shape[1]], M5_LEVELS
            )
            prediction = read_forecast_grid(
                str(forecast_path),
                data.series_ids,
                args.origin,
                28,
                series_col="series_id",
                time_col="date",
                value_col="yhat",
            )
            actual_hash = directory_digest(args.actuals)
            actual = read_forecast_grid(
                str(args.actuals),
                data.series_ids,
                args.origin,
                28,
                series_col="id",
                time_col="date",
                value_col="qty",
                filter_actuals=True,
            )
            if actual_hash != directory_digest(args.actuals):
                raise ValueError("actuals changed while scoring")
            result = metric.score(prediction, actual)
            if not checkpoint_path.is_file() or complete["checkpoint_sha256"] != directory_digest(
                checkpoint_path
            ):
                raise ValueError("checkpoint changed while scoring")
            if not score_calendar.is_file() or binding["inputs"][
                str(score_calendar)
            ] != directory_digest(score_calendar):
                raise ValueError("calendar changed while scoring")
            if complete["forecast_sha256"] != directory_digest(forecast_path) or binding["inputs"][
                str(args.panel.resolve())
            ] != directory_digest(args.panel):
                raise ValueError("training inputs or forecast changed while scoring")
            if scoring != _scoring_binding() or completion_hash != directory_digest(
                completion_path
            ):
                raise ValueError(
                    "scoring source/runtime or completion record changed while scoring"
                )
        result.update(
            role="retrospective_development_only",
            actuals_sha256=actual_hash,
            forecast_sha256=complete["forecast_sha256"],
            origin=args.origin,
            horizon=28,
            checkpoint_sha256=complete["checkpoint_sha256"],
            calendar_sha256=binding["inputs"][str(score_calendar)],
            run_directory=str(args.out.resolve()),
            completion_sha256=completion_hash,
            training_panel_sha256=binding["inputs"][str(args.panel.resolve())],
            scoring_binding=scoring,
        )
        print(json.dumps(result, allow_nan=False))
        return 0
    if not args.calendar.exists():
        build_revision_calendar(args.calendar, args.m5_dir)
    with panel_generation_lock(args.panel, exclusive=False):
        forecast = fit_predict_revision(
            panel,
            args.origin,
            28,
            calendar_path=str(args.calendar),
            config=config,
            levels=M5_LEVELS,
            out_dir=str(args.out),
            replay_checkpoint=str(args.replay_checkpoint) if args.replay_checkpoint else None,
            resume_training_checkpoint=str(args.resume_training_checkpoint)
            if args.resume_training_checkpoint
            else None,
        )
    print(
        json.dumps(
            {
                "forecast": forecast.path,
                "n_series": forecast.n_series,
                "horizon": forecast.horizon,
                "role": "retrospective_development_only",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
