"""The module runner must preserve pytest failures and reject empty suites."""

import itertools
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def runner_project(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run_tests.py"
    shutil.copyfile(pathlib.Path(__file__).with_name("run_tests.py"), runner)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["checks"]\n'
    )
    checks = tmp_path / "checks"
    checks.mkdir()
    return runner, checks


def workflow_jobs():
    root = pathlib.Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / ".github/workflows/ci.yaml").read_text())["jobs"]


@pytest.mark.parametrize("filename", ["test_example.py", "example_test.py"])
@pytest.mark.parametrize("body,expected", [
    ("def test_ok():\n    import os, sys\n"
     "    assert os.environ['PYSPARK_PYTHON'] == sys.executable\n", 0),
    ("def test_skip():\n    import pytest\n    pytest.skip('external fixture absent')\n", 0),
    ("def test_fail():\n    assert False\n", 1),
    ("# accidentally empty module\n", 1),
    (None, 1),
])
def test_runner_preserves_collection_and_execution_outcomes(
    tmp_path, runner_project, filename, body, expected,
):
    runner, checks = runner_project
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


def test_ci_shards_execute_every_discovered_module_once(tmp_path, runner_project):
    _, checks = runner_project
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["checks", "other", "checks"]\n'
        'python_files = ["case_*.py", "*_case.py"]\n'
    )
    names = {
        "checks/case_a.py", "checks/case_overlap_case.py", "checks/deep/case_b.py",
        "checks/deep/more/case_c.py", "checks/suffix_case.py", "other/case_a.py",
        "other/case_b.py", "other/deep/case_c.py", "other/deep/suffix_case.py",
    }
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def test_receipt():\n"
            "    import builtins, os, sys\n"
            "    from pathlib import Path\n"
            "    assert os.environ['PYSPARK_PYTHON'] == sys.executable\n"
            "    assert os.environ['RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES'] == '134217728'\n"
            "    assert os.environ['FORECAST_SPARK_CORES'] == '2'\n"
            "    assert os.environ['FORECAST_SPARK_MEMORY'] == '1g'\n"
            "    assert os.environ['SPARK_LOCAL_IP'] == '127.0.0.1'\n"
            "    assert not hasattr(builtins, 'previous_module')\n"
            "    builtins.previous_module = True\n"
            "    with Path('executed.txt').open('a') as stream:\n"
            f"        stream.write({name!r} + '|' + str(os.getpid()) + '\\n')\n"
        )
    (checks / "test_excluded.py").write_text("def test_not_selected(): assert False\n")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside/case_excluded.py").write_text("def test_not_selected(): assert False\n")
    heavy, unknown = min(names), max(names)
    durations = {name: (100 if name == heavy else 1) for name in names if name != unknown}
    durations["outside/case_excluded.py"] = 1000
    timing_path = tmp_path / "configs/ci/test_durations.json"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_text(json.dumps(durations))
    job = workflow_jobs()["regressions"]
    step = next(s for s in job["steps"] if s.get("name") == "Unit and workflow regressions")
    assert step["if"] == "matrix.shard != 'native'"
    assert step["env"]["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == "134217728"
    shards = job["strategy"]["matrix"]["shard"]
    assert shards.count("native") == 1
    for shard in shards:
        if shard == "native":
            continue
        command = shlex.split(step["run"].replace("${{ matrix.shard }}", str(shard)))
        assert command[:2] == ["python", "scripts/run_tests.py"]
        command[0] = sys.executable
        receipt = tmp_path / "executed.txt"
        before = len(receipt.read_text().splitlines()) if receipt.exists() else 0
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True,
            env={**os.environ, **step["env"], "PYSPARK_PYTHON": "/wrong/interpreter"},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        emitted = receipt.read_text().splitlines()[before:]
        selected = [line.split("|")[0] for line in emitted]
        assert selected == sorted(selected)
        if shard == 0:
            assert selected == [heavy]
        elif shard == 1:
            assert selected == [unknown]
        else:
            assert 1 <= len(selected) <= 2
    executed = [line.split("|") for line in receipt.read_text().splitlines()]
    assert {name for name, _ in executed} == names
    assert len(executed) == len(names)
    assert len({pid for _, pid in executed}) == len(names)


@pytest.mark.parametrize("contents", ["{", "[]", '{"checks/test_one.py":true}',
                                      '{"checks/test_one.py":0}',
                                      '{"checks/test_one.py":-1}',
                                      '{"checks/test_one.py":NaN}', None])
def test_invalid_duration_profile_fails_before_execution(tmp_path, runner_project, contents):
    runner, checks = runner_project
    (checks / "test_one.py").write_text(
        "def test_receipt():\n    from pathlib import Path\n    Path('executed.txt').touch()\n"
    )
    profile = tmp_path / "durations.json"
    if contents is not None:
        profile.write_text(contents)
    result = subprocess.run(
        [sys.executable, str(runner), "--durations", str(profile)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (tmp_path / "executed.txt").exists()


def test_balancing_bounds_the_longest_shard_and_ignores_stale_entries():
    from scripts.run_tests import balanced_shards

    durations = {"a.py": 90, "b.py": 80, "c.py": 70, "d.py": 20, "e.py": 10,
                 "not_discovered.py": 1000}
    paths = ["a.py", "b.py", "c.py", "d.py", "e.py", "new.py"]
    shards = balanced_shards(paths, durations, 3)
    assert sorted(itertools.chain.from_iterable(shards)) == paths
    assert shards == balanced_shards(list(reversed(paths)), durations, 3)
    assert max(sum(durations.get(path, 60) for path in shard) for shard in shards) == 130


@pytest.mark.parametrize("arguments", [
    ["--shard-count", "0"],
    ["--shard-count", "-1"],
    ["--shard-index", "-1"],
    ["--shard-index", "2", "--shard-count", "2"],
    ["--shard-index", "not-an-integer"],
    ["--shard-index", "1", "--shard-count", "2"],
])
def test_invalid_or_empty_shard_fails_without_execution(tmp_path, runner_project, arguments):
    runner, checks = runner_project
    (checks / "test_one.py").write_text(
        "def test_receipt():\n"
        "    from pathlib import Path\n"
        "    Path('executed.txt').touch()\n"
    )
    result = subprocess.run(
        [sys.executable, str(runner), *arguments],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (tmp_path / "executed.txt").exists()


@pytest.mark.parametrize("failure", [
    "def test_failure(): assert False\n",
    "raise RuntimeError('collection failed')\n",
    *[f"def test_failure():\n    import pytest\n    pytest.exit('stop', returncode={code})\n"
      for code in (3, 4, 7)],
    "def test_failure():\n    import os, signal\n    os.kill(os.getpid(), signal.SIGTERM)\n",
])
def test_runner_reports_first_failure_without_running_later_work(
    tmp_path, runner_project, failure,
):
    runner, checks = runner_project
    late_test = (
        "def test_later():\n"
        "    from pathlib import Path\n"
        "    Path('later.txt').touch()\n"
    )
    (checks / "test_a.py").write_text(failure + "\n" + late_test)
    (checks / "test_z.py").write_text(late_test)
    result = subprocess.run(
        [sys.executable, str(runner)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED checks/test_a.py" in result.stdout
    assert "Modules: 1; failed: 1" in result.stdout
    assert not (tmp_path / "later.txt").exists()


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("index,count", [(0, 1), (1, 2)])
def test_runner_propagates_later_failure_in_default_and_ci_shards(
    tmp_path, runner_project, index, count, weighted,
):
    runner, checks = runner_project
    for number in range(3 * count):
        body = "def test_ok(): pass\n"
        if number == index + count:
            body = "def test_failure(): assert False\n"
        elif number == index + 2 * count:
            body = (
                "def test_later():\n    from pathlib import Path\n    Path('later.txt').touch()\n"
            )
        (checks / f"test_{number}.py").write_text(body)
    arguments = ["--shard-index", str(index), "--shard-count", str(count)]
    if weighted:
        profile = tmp_path / "durations.json"
        profile.write_text("{}")
        arguments.extend(["--durations", str(profile)])
    result = subprocess.run(
        [sys.executable, str(runner), *arguments],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAILED checks/test_{index + count}.py" in result.stdout
    assert "Modules: 2; failed: 1" in result.stdout
    assert not (tmp_path / "later.txt").exists()


def test_ci_keeps_required_checks_and_acceptance_variants():
    jobs = workflow_jobs()
    matrix = jobs["installed-wheel"]["strategy"]["matrix"]
    cases = set(itertools.product(matrix["python"], matrix["frequency"], matrix["input"]))
    cases.update((row["python"], row["frequency"], row["input"]) for row in matrix["include"])
    assert cases == {
        (python, frequency, source)
        for python in ("3.12", "3.13")
        for frequency in (("H", "D", "W", "M", "MS", "Y", "YS") if python == "3.12" else ("D",))
        for source in ("csv", "parquet")
    }
    assert set(jobs["installed-models"]["strategy"]["matrix"]["extra"]) == {"timesfm", "all-models"}
    preflight = jobs["preflight"]["steps"]
    checkout = next(
        step for step in preflight if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == 0
    for step in preflight:
        assert step.get("if") is None and not step.get("continue-on-error", False)
    required = {
        ("preflight", "Install checked CI tools"): (
            "uv lock --check", "uv export --frozen", '--constraint "$RUNNER_TEMP/constraints.txt"',
            '--build-constraints "$RUNNER_TEMP/constraints.txt"', "--group ci PyYAML",
        ),
        ("preflight", "Lint and publication checks"): (
            "ruff check .", "python -m scripts.verify_public_release",
            "python -m pytest -q -rs -x scripts/test_run_tests.py "
            "scripts/test_distribution.py scripts/test_public_release.py",
        ),
        ("regressions", "Type check"): ("pyright --project pyrightconfig.governed.json",),
        ("regressions", "Installed native model and HPO acceptance"): (
            "for metric in mae wape; do", "python -m scripts.verify_native_study_release",
            '--metric "$metric"', "--model-profile", "--implementation-config",
        ),
        ("regressions", "Export checked lock and install CPU test runtimes"): (
            "uv lock --check", "uv export --frozen", "python -m build --wheel",
            "python -m scripts.verify_public_release --wheel", "uv pip install --system",
            '--constraint "$RUNNER_TEMP/constraints.txt"',
            '--build-constraints "$RUNNER_TEMP/constraints.txt"',
            "uv pip check --system",
        ),
    }
    for (job, name), commands in required.items():
        step = next(step for step in jobs[job]["steps"] if step.get("name") == name)
        assert all(command in step["run"] for command in commands), name


def test_ci_orders_stages_and_requires_every_result():
    jobs = workflow_jobs()
    assert jobs["regressions"]["needs"] == "preflight"
    assert jobs["regressions"]["strategy"]["fail-fast"] is False
    for name in ("installed-wheel", "installed-models"):
        assert jobs[name]["needs"] == "preflight"
        assert jobs[name]["strategy"]["fail-fast"] is False
    gate = jobs["checks"]
    assert gate["if"] == "always()"
    assert set(gate["needs"]) == set(jobs) - {"checks"}
    assert set(gate["env"].values()) == {
        "${{ needs." + name + ".result }}" for name in gate["needs"]
    }
    for job in jobs.values():
        assert not job.get("continue-on-error", False)
    expected_conditions = {
        "Type check": "matrix.shard == 'native'",
        "Installed native model and HPO acceptance": "matrix.shard == 'native'",
        "Unit and workflow regressions": "matrix.shard != 'native'",
    }
    for name, condition in expected_conditions.items():
        step = next(step for step in jobs["regressions"]["steps"] if step.get("name") == name)
        assert step.get("if") == condition
        assert not step.get("continue-on-error", False)
    step = gate["steps"][0]
    assert step.get("if") is None
    assert not step.get("continue-on-error", False)
    assert step["shell"] == "bash"
    assert step.get("env", {}) == {}
    for values in itertools.product(("success", "failure", "cancelled", "skipped"), repeat=4):
        result = subprocess.run(
            ["bash", "-c", step["run"]],
            env={**os.environ, **dict(zip(gate["env"], values, strict=True))},
        )
        assert (result.returncode == 0) == all(value == "success" for value in values), values


def test_ci_cancels_superseded_pull_requests_without_cancelling_main():
    root = pathlib.Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yaml").read_text())
    assert workflow["concurrency"] == {
        "group": "${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
