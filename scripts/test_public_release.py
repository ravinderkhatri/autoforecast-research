"""Publication checks reject force-added private artifacts and stale wheel modules."""

from __future__ import annotations

import importlib.metadata
import pathlib
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import pytest
from autoforecast.project import installed_library_identity

from scripts.verify_public_release import README_IMAGES, ROOT_FILES, tracked_inventory, verify_wheel

ROOT = pathlib.Path(__file__).resolve().parents[1]


def command(root, *args):
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def commit(root, message="Public source"):
    command(root, "git", "add", ".")
    command(
        root,
        "git",
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        message,
    )


@pytest.fixture(scope="module")
def clean_wheel(tmp_path_factory):
    root = tmp_path_factory.mktemp("public-wheel")
    (root / "autoforecast").mkdir()
    for name in ("__init__.py", "project.py", "_exact_json.py"):
        shutil.copyfile(ROOT / "autoforecast" / name, root / "autoforecast" / name)
    (root / "forecast/data").mkdir(parents=True)
    for path in (ROOT / "forecast/data").glob("*.py"):
        shutil.copyfile(path, root / "forecast/data" / path.name)
    shutil.copyfile(ROOT / "setup.py", root / "setup.py")
    (root / "autoforecast/test_example.py").write_text("def test_example():\n    assert True\n")
    (root / ".gitignore").write_text("build/\ndist/\n*.egg-info/\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "autoforecast"\nversion = "0.1.0"\n'
        '[project.scripts]\nautoforecast = "autoforecast.project:installed_library_identity"\n'
        '[tool.setuptools.packages.find]\nwhere = ["."]\n'
        'include = ["autoforecast*", "forecast*"]\n'
    )
    command(root, "git", "init", "-b", "main")
    commit(root)
    command(root, sys.executable, "setup.py", "bdist_wheel", "--dist-dir", "dist")
    return root, next((root / "dist").glob("*.whl"))


def test_release_wheel_matches_tracked_runtime_and_excludes_tests(
    clean_wheel, tmp_path, monkeypatch
):
    root, wheel = clean_wheel
    names, config = tracked_inventory(root)
    count = verify_wheel(root, wheel, names, config)
    with zipfile.ZipFile(wheel) as archive:
        assert count == len(archive.namelist())
        assert "autoforecast/project.py" in archive.namelist()
        assert "autoforecast/test_example.py" not in archive.namelist()
        archive.extractall(tmp_path)
    distribution = importlib.metadata.PathDistribution(tmp_path / "autoforecast-0.1.0.dist-info")
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    identity = installed_library_identity()
    assert identity.build["source_commit"] == command(root, "git", "rev-parse", "HEAD").strip()


def test_public_inventory_and_wheel_include_maintained_forecast_data(clean_wheel):
    root, wheel = clean_wheel
    names, config = tracked_inventory(root)
    runtime = {
        "forecast/data/exposure.py",
        "forecast/data/observation.py",
        "forecast/data/reference.py",
        "forecast/data/rules.py",
    }
    assert runtime <= names
    assert "forecast/data/test_reference.py" in names
    verify_wheel(root, wheel, names, config)
    with zipfile.ZipFile(wheel) as archive:
        packaged = {name for name in archive.namelist() if name.startswith("forecast/data/")}
        assert packaged == runtime
        for name in runtime:
            assert archive.read(name) == (ROOT / name).read_bytes()


def test_public_inventory_accepts_the_approved_readme_diagram(clean_wheel, tmp_path):
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    name = "assets/autoforecast-overview.svg"
    path = root / name
    path.parent.mkdir()
    shutil.copyfile(ROOT / name, path)
    command(root, "git", "add", name)

    names, _ = tracked_inventory(root)

    assert name in names


def test_public_inventory_accepts_root_readme(clean_wheel, tmp_path):
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    (root / "README.md").write_text("# Public package\n")
    command(root, "git", "add", "README.md")
    names, _ = tracked_inventory(root)
    assert "README.md" in names


def test_readme_diagrams_are_present_valid_and_approved():
    images = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", (ROOT / "README.md").read_text()))
    assert images == README_IMAGES
    for name in images:
        assert pathlib.PurePosixPath(name).suffix == ".svg"
        assert ET.parse(ROOT / name).getroot().tag == "{http://www.w3.org/2000/svg}svg"


@pytest.mark.parametrize("name", sorted(README_IMAGES))
def test_public_inventory_accepts_only_approved_readme_diagrams(clean_wheel, tmp_path, name):
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    command(root, "git", "add", name)
    names, _ = tracked_inventory(root)
    assert name in names
    sibling = path.with_name("unapproved-diagram.svg")
    sibling.write_text(path.read_text())
    command(root, "git", "add", str(sibling.relative_to(root)))
    with pytest.raises(ValueError, match="public inventory"):
        tracked_inventory(root)


@pytest.mark.parametrize("name", sorted(
    {name for name in ROOT_FILES | README_IMAGES
     if name.lower().endswith(".md") and pathlib.Path(name).name.lower() != "readme.md"}
    | {"plan.md", "notes.MD", "docs/plan.md"}
))
def test_public_inventory_rejects_every_non_readme_markdown(clean_wheel, tmp_path, name):
    # Exercise any future allowlist exception as well as ordinary private notes.
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Local notes\n")
    command(root, "git", "add", "-f", name)
    with pytest.raises(ValueError, match="public inventory"):
        tracked_inventory(root)


@pytest.mark.parametrize(
    "name, content",
    [
        ("INTERNAL.md", "private notes"),
        ("autoforecast/notes.md", "private notes"),
        ("configs/diagram.svg", "<svg/>"),
        ("assets/internal.svg", "<svg/>"),
        ("configs/conversation.json", "{}"),
        ("configs/messages.json", '[{"ro' + 'le": "assistant", "content": "private"}]'),
        ("configs/session.jsonl", "{}\n"),
        ("autoforecast/transcript.py", "pass\n"),
        ("migration/old.py", "pass\n"),
        ("autoforecast/signed.py", '# module\n"""\nGenerated-' + 'by: automation\n"""\n'),
    ],
)
def test_public_release_inventory_rejects_private_artifacts(clean_wheel, tmp_path, name, content):
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    command(root, "git", "add", "-f", name)
    with pytest.raises(ValueError, match="public (inventory|file)"):
        tracked_inventory(root)


def test_public_release_inventory_rejects_commit_attribution(clean_wheel, tmp_path):
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    commit(root, "Public source\n\nCo-" + "authored-by: automation <test@example.invalid>")
    commit(root, "Clean later commit")
    with pytest.raises(ValueError, match="commit history"):
        tracked_inventory(root)


def test_public_release_inventory_refuses_shallow_history(clean_wheel, tmp_path):
    original, _ = clean_wheel
    root = tmp_path / "shallow"
    command(tmp_path, "git", "clone", "--depth", "1", original.as_uri(), str(root))
    with pytest.raises(ValueError, match="full Git history"):
        tracked_inventory(root)


@pytest.mark.parametrize(
    "name",
    [
        "forecast/data/absent.py",
        "autoforecast/test_example.py",
        "autoforecast/new.py",
        "notes.md",
        "conversations/session.jsonl",
    ],
)
def test_release_wheel_rejects_absent_retired_or_internal_members(clean_wheel, tmp_path, name):
    root, wheel = clean_wheel
    changed = tmp_path / "injected.whl"
    shutil.copyfile(wheel, changed)
    with zipfile.ZipFile(changed, "a") as archive:
        archive.writestr(name, "unpublished bytes")
    names, config = tracked_inventory(root)
    with pytest.raises(ValueError, match="wheel inventory differs"):
        verify_wheel(root, changed, names, config)


@pytest.mark.parametrize(
    "member, message",
    [
        ("autoforecast/project.py", "source differs"),
        ("autoforecast/_build.json", "build identity differs"),
        ("autoforecast-0.1.0.dist-info/METADATA", "RECORD differs"),
    ],
)
def test_release_wheel_rejects_changed_bytes_and_identity(clean_wheel, tmp_path, member, message):
    root, wheel = clean_wheel
    changed = tmp_path / "changed.whl"
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(changed, "w") as modified:
        for name in original.namelist():
            payload = original.read(name)
            if name == member:
                payload = b"{}" if name.endswith("_build.json") else payload + b"\nchanged\n"
            modified.writestr(name, payload)
    names, config = tracked_inventory(root)
    with pytest.raises(ValueError, match=message):
        verify_wheel(root, changed, names, config)


def test_build_refuses_stale_python_modules(clean_wheel, tmp_path):
    original, _ = clean_wheel
    root = tmp_path / "repo"
    shutil.copytree(original, root)
    (root / "build/lib/autoforecast/retired.py").write_text("pass\n")
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", "dist"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "stale Python modules" in result.stderr


@pytest.fixture(scope="module")
def legal_wheel(clean_wheel, tmp_path_factory):
    original, _ = clean_wheel
    root = tmp_path_factory.mktemp("synthetic-legal") / "repo"
    shutil.copytree(original, root)
    for name in ("LICENSE", "COPYING", "NOTICE"):
        (root / name).write_text(f"Synthetic test fixture {name}; no license grant.\n")
    config = root / "pyproject.toml"
    config.write_text(
        config.read_text().replace(
            "[project]\n",
            '[project]\nlicense = "LicenseRef-Test-Only"\n'
            'license-files = ["LICENSE", "COPYING", "NOTICE"]\n',
        )
    )
    commit(root)
    command(root, sys.executable, "setup.py", "bdist_wheel", "--dist-dir", "dist")
    return root, next((root / "dist").glob("*.whl"))


def test_release_wheel_preserves_declared_synthetic_legal_files(legal_wheel):
    root, wheel = legal_wheel
    names, config = tracked_inventory(root)
    verify_wheel(root, wheel, names, config)
    with zipfile.ZipFile(wheel) as archive:
        for name in ("LICENSE", "COPYING", "NOTICE"):
            assert (
                archive.read(f"autoforecast-0.1.0.dist-info/licenses/{name}")
                == (root / name).read_bytes()
            )


@pytest.mark.parametrize("mutation", ["missing", "changed", "metadata", "expression"])
def test_release_wheel_rejects_legal_file_or_metadata_changes(legal_wheel, tmp_path, mutation):
    root, wheel = legal_wheel
    changed = tmp_path / "changed.whl"
    member = "autoforecast-0.1.0.dist-info/licenses/LICENSE"
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(changed, "w") as modified:
        for name in original.namelist():
            payload = original.read(name)
            if name == member and mutation == "missing":
                continue
            if name == member and mutation == "changed":
                payload += b"changed\n"
            if name.endswith("/METADATA") and mutation == "metadata":
                payload = payload.replace(b"License-File: LICENSE\n", b"")
            if name.endswith("/METADATA") and mutation == "expression":
                payload = payload.replace(b"LicenseRef-Test-Only", b"LicenseRef-Other-Test")
            modified.writestr(name, payload)
    names, config = tracked_inventory(root)
    with pytest.raises(ValueError, match="wheel (inventory|license)"):
        verify_wheel(root, changed, names, config)


def test_release_wheel_requires_each_declared_legal_file(legal_wheel, tmp_path):
    original, wheel = legal_wheel
    root = tmp_path / "missing-legal"
    shutil.copytree(original, root)
    (root / "LICENSE").unlink()
    commit(root)
    names, config = tracked_inventory(root)
    with pytest.raises(ValueError, match="declared license pattern has no tracked file"):
        verify_wheel(root, wheel, names, config)


def test_standard_dependencies_have_no_compatibility_only_extras():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert set(extras) == {"all-models", "data", "custom-torch", "shallow", "tft",
                           "autogluon-zoo", "timesfm", "timesfm3"}
    core = {value.split("==")[0] for value in project["dependencies"]}
    assert {"pyspark", "pyarrow", "statsforecast", "optuna", "ray[tune]", "mlflow-skinny"} <= core
