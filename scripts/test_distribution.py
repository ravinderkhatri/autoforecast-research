"""Source packages must remain visible to Git when runtime outputs are ignored."""

import pathlib
import subprocess
import tomllib


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
