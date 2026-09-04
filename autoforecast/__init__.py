"""Autoforecast package initialization."""

from __future__ import annotations


def register_study(config, *, registry=None, spark=None, prepare=None, log_level=None):
    """Register configured data once without fitting; see study_registration."""
    from autoforecast.study_registration import register_study as register

    return register(config, registry=registry, spark=spark, prepare=prepare, log_level=log_level)


def run_study(study_id, model, **options):
    """Run one model against an existing registered study; see study_workflow."""
    from autoforecast.study_workflow import run_study as run

    return run(study_id, model, **options)


def configure_logging(*, stream=None, progress="auto", interval=10.0):
    """Enable readable library progress; see tutorials/README.md for examples."""
    from autoforecast._execution import configure_logging as configure

    return configure(stream=stream, progress=progress, interval=interval)


__all__ = ["register_study", "run_study", "configure_logging"]
