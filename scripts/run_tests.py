"""Run each test module in a separate process so Spark ownership cannot leak between modules."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tomllib


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = config["tool"]["pytest"]["ini_options"].get(
        "python_files", ["test_*.py", "*_test.py"]
    )
    paths = sorted({
        path
        for name in config["tool"]["pytest"]["ini_options"]["testpaths"]
        for pattern in patterns
        for path in (root / name).rglob(pattern)
    })
    if not paths:
        print("No test modules found in configured testpaths", file=sys.stderr)
        return 1
    environment = {**os.environ, "PYSPARK_PYTHON": sys.executable}
    failures = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        print(f"\nTesting {relative}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rs", relative],
            cwd=root, env=environment, check=False,
        )
        # Explicitly skipped test cases return zero; empty modules must fail collection.
        if result.returncode != 0:
            failures.append(relative)
    print(f"\nModules: {len(paths)}; failed: {len(failures)}", flush=True)
    for path in failures:
        print(f"FAILED {path}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
