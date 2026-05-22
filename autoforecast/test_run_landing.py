"""Functional check for the single-page static run-observability entry point."""

from __future__ import annotations

import pathlib

from scripts.run_autoforecast import _write_observability_landing

from autoforecast.model_api import CheckReport, CheckResult, DataFacts
from autoforecast.train_workflow import ModelOutcome, RunSummary


def test_landing_links_model_checks_configuration_and_recorded_tensorboard(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    attempt = tmp_path / "attempts" / "mft" / "one"
    for relative in ("checks/data.json", "checks/model.json", "metadata/execution-plan.json"):
        target = attempt / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    (attempt / "completion.json").write_text("{}\n", encoding="utf-8")
    events = attempt / "tensorboard" / "events.out.tfevents.functional"
    events.parent.mkdir(parents=True)
    events.write_bytes(b"event")
    (tmp_path / "report.html").write_text("<h1>Run</h1>", encoding="utf-8")
    facts = DataFacts("2026-09-10", "a" * 64, ("2026-01-01", "2026-09-10"), 1, 1, 1, 0, 0, {}, {}, None, {})
    data = CheckReport("data", facts, (CheckResult("ok", "OK", "passed", "ok"),))
    summary = RunSummary(
        "functional",
        tmp_path,
        (ModelOutcome("mft", "completed", "done"),),
        data,
        None,
    )
    monkeypatch.setenv("AUTOFORECAST_TENSORBOARD_URL", "http://127.0.0.1:6007")
    _write_observability_landing(summary)
    page = (tmp_path / "observability.html").read_text(encoding="utf-8")
    assert "Data checks" in page and "Model checks" in page and "Open TensorBoard" in page
    assert "observability.html" in (tmp_path / "report.html").read_text(encoding="utf-8")
