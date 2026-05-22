"""Retain build provenance inside non-editable wheels."""

import hashlib
import json
import pathlib
import re
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


def source_commit(root):
    """Archives and unborn repositories have no source commit; parent repos do not own them."""
    root = pathlib.Path(root).resolve()
    try:
        owner = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if owner.returncode or pathlib.Path(owner.stdout.strip()).resolve() != root:
            return None
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode:
        return None
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ValueError("Git returned an invalid source commit")
    return commit


class BuildWithIdentity(build_py):
    def find_package_modules(self, package, package_dir):
        return [
            entry
            for entry in super().find_package_modules(package, package_dir)
            if not (entry[1].startswith("test_") or entry[1].endswith("_test"))
        ]

    def run(self):
        super().run()
        root = pathlib.Path(__file__).resolve().parent
        outputs = {pathlib.Path(path) for path in self.get_outputs() if path.endswith(".py")}
        built = set(pathlib.Path(self.build_lib).rglob("*.py"))
        if built != outputs:
            raise ValueError("build directory contains stale Python modules; use a clean build")
        files = {}
        for package in (
            "autoforecast",
            "forecast",
            "mlops",
            "observability",
            "provenance",
            "scripts",
            "spec",
        ):
            for path in sorted((pathlib.Path(self.build_lib) / package).rglob("*.py")):
                files[path.relative_to(self.build_lib).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        record = {
            "schema": "autoforecast.build.v1",
            "source_commit": source_commit(root),
            "code_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        }
        (pathlib.Path(self.build_lib) / "autoforecast/_build.json").write_text(
            json.dumps(record, sort_keys=True) + "\n"
        )


setup(cmdclass={"build_py": BuildWithIdentity})
