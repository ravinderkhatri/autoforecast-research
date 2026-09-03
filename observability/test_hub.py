"""Functional checks for the static observability-hub publication boundary."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

from observability.hub import build_hub
from observability.tensorboard_card import write_tensorboard_navigation


def _write_source_config(root: pathlib.Path) -> pathlib.Path:
    config = root / "sources.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "data-preflight",
                        "evidence_class": "data preflight",
                        "report_glob": ".autoforecast/data-preflight/*/report.html",
                        "tensorboard": False,
                    },
                {
                    "id": "hpo",
                    "evidence_class": "HPO validation",
                        "report_glob": ".autoforecast/hpo/**/studies/*/report.html",
                    "tensorboard": True,
                },
                {
                    "id": "model-execution-smoke",
                    "evidence_class": "model execution smoke",
                        "report_glob": ".autoforecast/smoke-runs/*/report.html",
                    "tensorboard": False,
                },
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_build_hub_links_only_allowlisted_reports_and_preserves_data_admission_status() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        preflight = root / ".autoforecast/data-preflight/client-v1"
        preflight.mkdir(parents=True)
        (preflight / "report.html").write_text("preflight", encoding="utf-8")
        (preflight / "data_validation_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 2, "kind": "data_checks", "accepted": True,
                    "facts": {"data_fingerprint": "fingerprint-1", "as_of": "2026-08-31"},
                    "checks": [{"status": "warning", "summary": "one expected cadence warning"}],
                }
            ),
            encoding="utf-8",
        )
        study = root / ".autoforecast/hpo/dev/studies/study-v1"
        study.mkdir(parents=True)
        (study / "report.html").write_text("study", encoding="utf-8")
        archived = root / "gates/archived_misleading_tensorboard/report.html"
        archived.parent.mkdir(parents=True)
        archived.write_text("must not be indexed", encoding="utf-8")
        output = build_hub(
            _write_source_config(root),
            root / "published/index.html",
            repository_root=root,
            public_base_url="/",
            tensorboard_url="https://tensorboard.example.test/project",
        )
        page = output.read_text(encoding="utf-8")
        assert "data checks passed" in page
        assert "Latest completed check: data checks passed (client-v1)." in page
        assert "History: 1 passed, 0 failed, 0 not checked." in page
        assert "0 failed · 1 warning(s)" in page
        assert "one expected cadence warning" in page
        assert "fingerprint-1" in page
        assert "/.autoforecast/data-preflight/client-v1/report.html" in page
        assert "/.autoforecast/hpo/dev/studies/study-v1/report.html" in page
        assert "https://tensorboard.example.test/project" in page
        assert "archived_misleading_tensorboard" not in page
        assert "Data checks, model runs, and training traces" in page
        assert ">Open data-check report</a>" in page
        assert "/.autoforecast/data-preflight/client-v1/report.html" in page
        assert "target='_blank'" in page


def test_hub_does_not_interpret_obsolete_report_fields() -> None:
    """Obsolete reports cannot assert current data-quality or incident status."""
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        preflight = root / ".autoforecast/data-preflight/legacy-read-error"
        preflight.mkdir(parents=True)
        (preflight / "report.html").write_text("legacy", encoding="utf-8")
        (preflight / "data_validation_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "as_of": "2026-09-10",
                    "profile": None,
                    "blockers": ["Spark inspection failed: Py4JJavaError"],
                }
            ),
            encoding="utf-8",
        )
        output = build_hub(
            _write_source_config(root), root / "published/index.html", repository_root=root
        )
        page = output.read_text(encoding="utf-8")
        assert "unsupported report schema" in page
        assert "Regenerate using the current report format." in page
        assert "Spark inspection failed" not in page


def test_hub_preserves_current_failed_checks_in_summary_and_history(tmp_path) -> None:
    preflight = tmp_path / ".autoforecast/data-preflight/rejected"
    preflight.mkdir(parents=True)
    (preflight / "report.html").write_text("data checks", encoding="utf-8")
    (preflight / "data_validation_report.json").write_text(json.dumps({
        "schema_version": 2,
        "kind": "data_checks",
        "accepted": False,
        "facts": {"data_fingerprint": "failed-snapshot", "as_of": "2026-09-03"},
        "checks": [{"status": "failed", "summary": "required values absent"}],
    }), encoding="utf-8")

    page = build_hub(
        _write_source_config(tmp_path), tmp_path / "published/index.html", repository_root=tmp_path,
    ).read_text(encoding="utf-8")

    assert "Latest completed check: data checks failed (rejected)." in page
    assert "History: 0 passed, 1 failed, 0 not checked." in page
    assert "1 failed · 0 warning(s) — required values absent" in page
    assert "failed-snapshot" in page
    import shutil

    shutil.copytree(preflight, preflight.with_name("rejected-copy"))
    repeated = build_hub(
        _write_source_config(tmp_path), tmp_path / "published/index.html", repository_root=tmp_path,
    ).read_text(encoding="utf-8")
    assert "History: 0 passed, 1 failed, 0 not checked." in repeated
    assert "Latest completed check: data checks failed" in repeated


def test_hub_counts_one_snapshot_once_when_a_report_is_rendered_again() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        for version in ("v1", "v2"):
            preflight = root / f".autoforecast/data-preflight/snapshot-{version}"
            preflight.mkdir(parents=True)
            (preflight / "report.html").write_text(version, encoding="utf-8")
            (preflight / "data_validation_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2, "kind": "data_checks", "accepted": True,
                        "facts": {"data_fingerprint": "same-snapshot", "as_of": "2026-09-10"},
                        "checks": [],
                    }
                ),
                encoding="utf-8",
            )
        output = build_hub(
            _write_source_config(root), root / "published/index.html", repository_root=root
        )
        page = output.read_text(encoding="utf-8")
        assert "History: 1 passed, 0 failed, 0 not checked." in page


def test_main_generates_a_complete_static_page_without_a_tensorboard_endpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        report = root / ".autoforecast/hpo/dev/studies/study-v1/report.html"
        report.parent.mkdir(parents=True)
        report.write_text("preflight", encoding="utf-8")
        config = _write_source_config(root)
        output = root / "published/index.html"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "observability.hub",
                "--sources",
                str(config),
                "--output",
                str(output),
                "--repository-root",
                str(root),
            ],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        page = output.read_text(encoding="utf-8")
        assert "Data checks, model runs, and training traces" in page
        assert "No configured reports found." not in page
        assert "training traces" in page
        assert "not configured" in page
        assert "Data quality is unknown; it is not an implied pass." in page


def test_hub_shows_a_v2_read_failure_as_run_status_not_a_failed_data_check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        preflight = root / ".autoforecast/data-preflight/broken-source"
        preflight.mkdir(parents=True)
        (preflight / "report.html").write_text("incident", encoding="utf-8")
        (preflight / "data_validation_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "run_incident",
                    "status": "not_checked",
                    "incident": {
                        "id": "data_read_failed",
                        "name": "Could not read the data",
                        "summary": "Could not read the requested data. Data checks were not run.",
                    },
                }
            ),
            encoding="utf-8",
        )
        output = build_hub(
            _write_source_config(root), root / "published/index.html", repository_root=root
        )
        page = output.read_text(encoding="utf-8")
        assert "data not checked" in page
        assert "Could not read the requested data. Data checks were not run." in page
        assert "No completed data check yet. 1 published attempt(s) did not run." in page


def test_write_tensorboard_navigation_creates_a_real_text_event_with_the_hub_link() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        log_dir = pathlib.Path(temporary) / "tensorboard" / "navigation"
        hub_url = "http://localhost:8000/.autoforecast/observability/index.html"
        write_tensorboard_navigation(log_dir, hub_url)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(log_dir))
        events.Reload()
        tag = "Navigation/Observability Hub/text_summary"
        assert tag in events.Tags()["tensors"]
        text = events.Tensors(tag)[0].tensor_proto.string_val[0].decode("utf-8")
        assert hub_url in text


def test_unversioned_sidecar_cannot_supply_status_or_snapshot_identity(tmp_path):
    from observability.hub import _metadata

    report = tmp_path / "report.html"
    report.write_text("report")
    (tmp_path / "data_validation_report.json").write_text(json.dumps({
        "accepted": True, "data_fingerprint": "old-identity", "as_of": "2026-09-10",
        "blockers": [], "warnings": [],
    }))
    status, fingerprint, as_of, _ = _metadata(report)
    assert status == "unsupported report schema"
    assert fingerprint == as_of == "—"
