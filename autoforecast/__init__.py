"""Autoforecast package initialization."""

from __future__ import annotations


def register_study(config, *, registry=None, spark=None):
    """Register configured data once without fitting; see study_registration."""
    from autoforecast.study_registration import register_study as register

    return register(config, registry=registry, spark=spark)


def run_study(study_id, model, **options):
    """Run one model against an existing registered study; see study_workflow."""
    from autoforecast.study_workflow import run_study as run

    return run(study_id, model, **options)


__all__ = ["register_study", "run_study"]
