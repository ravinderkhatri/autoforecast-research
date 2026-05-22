"""Functional test for explicit Run Center report integration."""

from __future__ import annotations

import pathlib
import tempfile

from observability.run_center import build_run_center


def test_run_center_links_explicit_evidence_without_scanning_or_localhost() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        reports = []
        for name in ("data", "study", "benchmark", "catalogue"):
            report = root / name / "report.html"
            report.parent.mkdir()
            report.write_text(f"<h1>{name}</h1>", encoding="utf-8")
            reports.append(report)
        page = build_run_center(
            root / "center" / "index.html",
            data_reports=[reports[0]],
            study_reports=[reports[1]],
            benchmark_reports=[reports[2]],
            model_catalogues=[reports[3]],
            tensorboard_url="https://traces.example.test/project",
            mlflow_url="https://mlflow.example.test/project",
        ).read_text(encoding="utf-8")
        assert "Verified forecasting evidence" in page
        assert "data" in page and "study" in page and "benchmark" in page
        assert "https://traces.example.test/project" in page
        assert "https://mlflow.example.test/project" in page
        assert "127.0.0.1" not in page
