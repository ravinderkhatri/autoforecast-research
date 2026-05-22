"""The module runner must preserve pytest failures and reject empty suites."""

import pathlib
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("filename", ["test_example.py", "example_test.py"])
@pytest.mark.parametrize("body,expected", [
    ("def test_ok():\n    import os, sys\n"
     "    assert os.environ['PYSPARK_PYTHON'] == sys.executable\n", 0),
    ("def test_skip():\n    import pytest\n    pytest.skip('external fixture absent')\n", 0),
    ("def test_fail():\n    assert False\n", 1),
    ("# accidentally empty module\n", 1),
    (None, 1),
])
def test_runner_preserves_collection_and_execution_outcomes(tmp_path, filename, body, expected):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run_tests.py"
    shutil.copyfile(pathlib.Path(__file__).with_name("run_tests.py"), runner)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["checks"]\n'
    )
    checks = tmp_path / "checks"
    checks.mkdir()
    if body is not None:
        (checks / filename).write_text(body)
    result = subprocess.run(
        [sys.executable, str(runner)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if body is None:
        assert "No test modules found" in result.stderr
    else:
        assert f"Modules: 1; failed: {expected}" in result.stdout
