"""Source packages must remain visible to Git when runtime outputs are ignored."""

import pathlib
import subprocess
import tomllib

import pytest


def test_source_packages_are_not_hidden_by_output_ignore_rules():
    root = pathlib.Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    modules = sorted(
        path.relative_to(root).as_posix()
        for name in config["tool"]["pytest"]["ini_options"]["testpaths"]
        for path in (root / name).rglob("*.py")
    )
    assert modules, "the configured source packages must contain Python modules"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(modules) + "\n", cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("name, ignored", [
    ("plan.md", True),
    ("notes.mD", True),
    ("docs/plan.Md", True),
    ("scripts/RUN.MD", True),
    ("README.md", False),
    ("readme.md", False),
    ("docs/ReadMe.MD", False),
    (".github/README.Md", False),
])
def test_markdown_ignore_policy(tmp_path, name, ignored):
    root = pathlib.Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "-c", "core.ignoreCase=false",
         "check-ignore", "--no-index", "-q", name],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == (0 if ignored else 1), result.stdout + result.stderr
