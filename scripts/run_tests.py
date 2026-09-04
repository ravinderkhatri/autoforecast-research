"""Run each test module in a separate process so Spark ownership cannot leak between modules."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tomllib


def balanced_shards(paths: list[str], durations: dict[str, float], count: int) -> list[list[str]]:
    """Assign every discovered module once, placing long modules on the lightest shard."""
    shards: list[list[str]] = [[] for _ in range(count)]
    totals = [0.0] * count
    for path in sorted(paths, key=lambda name: (-durations.get(name, 60.0), name)):
        index = min(range(count), key=lambda index: (totals[index], index))
        shards[index].append(path)
        totals[index] += durations.get(path, 60.0)
    return [sorted(shard) for shard in shards]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--durations", type=pathlib.Path,
                        help="JSON module durations in seconds; unknown modules use 60 seconds")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count > 0 and 0 <= shard-index < shard-count")
    root = pathlib.Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = config["tool"]["pytest"]["ini_options"].get(
        "python_files", ["test_*.py", "*_test.py"]
    )
    paths = sorted({
        path.relative_to(root).as_posix()
        for name in config["tool"]["pytest"]["ini_options"]["testpaths"]
        for pattern in patterns
        for path in (root / name).rglob(pattern)
    })
    if not paths:
        print("No test modules found in configured testpaths", file=sys.stderr)
        return 1
    total = len(paths)
    if args.durations is not None:
        try:
            durations = json.loads(args.durations.read_text())
            if not isinstance(durations, dict) or any(
                not isinstance(name, str) or not name
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                for name, seconds in durations.items()
            ):
                raise ValueError("durations require module names and finite positive seconds")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        paths = balanced_shards(paths, durations, args.shard_count)[args.shard_index]
    else:
        paths = paths[args.shard_index::args.shard_count]
    if not paths:
        print("No test modules assigned to this shard", file=sys.stderr)
        return 1
    print(
        f"Shard: {args.shard_index + 1}/{args.shard_count}; modules: {len(paths)}/{total}",
        flush=True,
    )
    environment = {**os.environ, "PYSPARK_PYTHON": sys.executable}
    for completed, relative in enumerate(paths, start=1):
        print(f"\nTesting {relative}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rs", "-x", relative],
            cwd=root, env=environment, check=False,
        )
        # Explicitly skipped test cases return zero; empty modules must fail collection.
        if result.returncode != 0:
            print(f"\nModules: {completed}; failed: 1", flush=True)
            print(f"FAILED {relative}", flush=True)
            return 1
    print(f"\nModules: {len(paths)}; failed: 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
