"""Build provenance must survive real packaging and refuse ambiguous or changed installations."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import pathlib
import runpy
import subprocess
from unittest.mock import patch

import pytest

from autoforecast.project import installed_library_identity

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_functions():
    with patch("setuptools.setup"):
        return runpy.run_path(str(ROOT / "setup.py"))


def test_build_identity_does_not_record_failed_rev_parse_stdout(tmp_path):
    function = build_functions()["source_commit"]
    owner = subprocess.CompletedProcess([], 0, str(tmp_path), "")
    failed = subprocess.CompletedProcess([], 128, "HEAD\n", "unborn repository")
    with patch("subprocess.run", side_effect=[owner, failed]):
        assert function(tmp_path) is None


def test_build_identity_does_not_inherit_parent_repository(tmp_path):
    function = build_functions()["source_commit"]
    owner = subprocess.CompletedProcess([], 0, str(tmp_path.parent), "")
    with patch("subprocess.run", return_value=owner) as run:
        assert function(tmp_path) is None
        assert run.call_count == 1


@pytest.mark.parametrize("commit", ["HEAD", "f" * 39, "F" * 40, "f" * 65, "", "f" * 40 + "\nextra"])
def test_build_identity_rejects_successful_but_invalid_git_output(tmp_path, commit):
    function = build_functions()["source_commit"]
    owner = subprocess.CompletedProcess([], 0, str(tmp_path), "")
    result = subprocess.CompletedProcess([], 0, commit, "")
    with patch("subprocess.run", side_effect=[owner, result]):
        with pytest.raises(ValueError, match="invalid source commit"):
            function(tmp_path)


@pytest.mark.parametrize("commit", ["a" * 40, "b" * 64])
def test_build_identity_accepts_valid_owned_commits(tmp_path, commit):
    function = build_functions()["source_commit"]
    owner = subprocess.CompletedProcess([], 0, str(tmp_path), "")
    result = subprocess.CompletedProcess([], 0, commit, "")
    with patch("subprocess.run", side_effect=[owner, result]):
        assert function(tmp_path) == commit


def test_build_identity_without_git_is_an_archive(tmp_path):
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert build_functions()["source_commit"](tmp_path) is None


@pytest.fixture
def installed_fixture(tmp_path, monkeypatch):
    package = tmp_path / "autoforecast"
    package.mkdir()
    (package / "project.py").write_text("VALUE = 1\n")
    info = tmp_path / "autoforecast-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: autoforecast\nVersion: 0.1.0\n")
    code = {
        "autoforecast/project.py": hashlib.sha256((package / "project.py").read_bytes()).hexdigest()
    }
    build = {
        "schema": "autoforecast.build.v1",
        "source_commit": "a" * 40,
        "code_sha256": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
    }

    def write_record(record=build, *, missing_hash=None, raw=None):
        (package / "_build.json").write_text(json.dumps(record) if raw is None else raw)
        rows = []
        for path in sorted(tmp_path.rglob("*")):
            if not path.is_file() or path.name == "RECORD":
                continue
            name = path.relative_to(tmp_path).as_posix()
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            rows.append(
                f"{name},{'' if name == missing_hash else 'sha256=' + digest},{path.stat().st_size}"
            )
        (info / "RECORD").write_text("\n".join(rows) + f"\n{info.name}/RECORD,,\n")

    write_record()
    distribution = importlib.metadata.PathDistribution(info)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    return package, build, write_record


def test_installed_identity_accepts_verified_build_and_archive(installed_fixture):
    _, build, write_record = installed_fixture
    first = installed_library_identity()
    assert first.build == build
    assert first.version == "0.1.0"
    build["source_commit"] = None
    write_record(build)
    assert installed_library_identity().build["source_commit"] is None
    assert installed_library_identity().artifact_sha256 != first.artifact_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        [],
        "record",
        1,
        {"schema": "unknown"},
        {"source_commit": "HEAD"},
        {"source_commit": False},
        {"source_commit": "A" * 40},
        {"source_commit": "a" * 39},
        {"code_sha256": None},
        {"code_sha256": 1},
        {"code_sha256": "z" * 64},
        {"extra": "not in the schema"},
    ],
)
def test_installed_library_identity_rejects_malformed_build_record(installed_fixture, mutation):
    _, build, write_record = installed_fixture
    write_record(build | mutation if isinstance(mutation, dict) else mutation)
    with pytest.raises(ValueError, match="invalid build record"):
        installed_library_identity()


@pytest.mark.parametrize("missing", ["schema", "source_commit", "code_sha256"])
def test_installed_identity_requires_every_build_field(installed_fixture, missing):
    _, build, write_record = installed_fixture
    write_record({key: value for key, value in build.items() if key != missing})
    with pytest.raises(ValueError, match="invalid build record"):
        installed_library_identity()


def test_installed_identity_rejects_duplicate_build_fields(installed_fixture):
    _, build, write_record = installed_fixture
    raw = json.dumps(build)[:-1] + ', "schema": "autoforecast.build.v1"}'
    write_record(raw=raw)
    with pytest.raises(ValueError, match="duplicate JSON"):
        installed_library_identity()


def test_installed_identity_recomputes_code_even_when_record_is_consistent(installed_fixture):
    package, build, write_record = installed_fixture
    (package / "project.py").write_text("VALUE = 2\n")
    write_record(build)
    with pytest.raises(ValueError, match="code differs from its build record"):
        installed_library_identity()


def test_installed_identity_still_authenticates_record(installed_fixture):
    package, _, _ = installed_fixture
    (package / "project.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="installed library file changed"):
        installed_library_identity()


@pytest.mark.parametrize("name", ["project.py", "_build.json"])
def test_installed_identity_requires_record_hash_for_code_and_build(installed_fixture, name):
    _, _, write_record = installed_fixture
    write_record(missing_hash=f"autoforecast/{name}")
    with pytest.raises(ValueError, match="has no RECORD hash"):
        installed_library_identity()


def test_installed_identity_rejects_unrecorded_module(installed_fixture):
    package, _, _ = installed_fixture
    (package / "unexpected.py").write_text("VALUE = 4\n")
    with pytest.raises(ValueError, match="Python files differ from RECORD"):
        installed_library_identity()


def test_installed_identity_rejects_linked_code(installed_fixture):
    package, _, _ = installed_fixture
    (package / "unexpected.py").symlink_to(package / "project.py")
    with pytest.raises(ValueError, match="symbolic link"):
        installed_library_identity()
