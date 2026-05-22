"""Explicit experiment ownership; source-checkout entrypoints keep their strict defaults."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import importlib.metadata
import json
import pathlib
import re
import sys
from dataclasses import dataclass

from autoforecast._exact_json import loads_exact_json

_PROJECT: contextvars.ContextVar[pathlib.Path | None] = contextvars.ContextVar(
    "autoforecast_project", default=None
)
LIBRARY_ROOT = pathlib.Path(__file__).resolve().parent.parent


def project_root(default: pathlib.Path) -> pathlib.Path:
    """Return the caller's experiment project or the source checkout boundary."""
    return _PROJECT.get() or default


def installed_mode() -> bool:
    return _PROJECT.get() is not None


def project_arguments() -> list[str]:
    return ["--project-root", str(_PROJECT.get())] if installed_mode() else []


def project_entrypoint(function):
    """Give thin CLIs an explicit experiment-project option."""

    @functools.wraps(function)
    def wrapped():
        if "--project-root" not in sys.argv:
            return function()
        arguments = list(sys.argv)
        index = arguments.index("--project-root")
        if index + 1 >= len(arguments):
            raise ValueError("--project-root requires a directory")
        root = arguments[index + 1]
        try:
            sys.argv = arguments[:index] + arguments[index + 2 :]
            with experiment_project(root):
                return function()
        finally:
            sys.argv = arguments

    return wrapped


@contextlib.contextmanager
def experiment_project(path: str | pathlib.Path):
    """Scope public API calls without process-global root mutation."""
    root = pathlib.Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("experiment project must be a directory")
    token = _PROJECT.set(root)
    try:
        yield root
    finally:
        _PROJECT.reset(token)


@dataclass(frozen=True)
class LibraryIdentity:
    """An installed distribution is identified by its bytes, never by a fictional Git commit."""

    version: str
    artifact_sha256: str
    build: dict

    def as_record(self) -> dict:
        return {
            "schema": "autoforecast.installed-library.v1",
            "name": "autoforecast",
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "build": self.build,
        }


def installed_library_identity() -> LibraryIdentity:
    """Authenticate installed package files against the wheel's RECORD hashes."""
    import base64

    distribution = importlib.metadata.distribution("autoforecast")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable"):
        raise ValueError("external projects require a non-editable wheel installation")
    entries = {}
    for entry in distribution.files or ():
        name = str(entry)
        if name.endswith(("/direct_url.json", "/INSTALLER", "/REQUESTED")):
            continue
        if name.endswith(".pyc") or name.startswith("../"):
            continue
        if entry.hash is None:
            if name.endswith(".py") or name == "autoforecast/_build.json":
                raise ValueError(f"installed library file has no RECORD hash: {name}")
            continue
        path = pathlib.Path(str(distribution.locate_file(entry)))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).digest()
        if (
            entry.hash.mode != "sha256"
            or base64.urlsafe_b64encode(digest).decode().rstrip("=") != entry.hash.value
        ):
            raise ValueError(f"installed library file changed: {name}")
        entries[name] = digest.hex()
    if not {"autoforecast/project.py", "autoforecast/_build.json"} <= entries.keys():
        raise ValueError("installed wheel has no verifiable library manifest")
    build_path = pathlib.Path(str(distribution.locate_file("autoforecast/_build.json")))
    build = loads_exact_json(build_path.read_bytes())
    if (
        not isinstance(build, dict)
        or set(build) != {"schema", "source_commit", "code_sha256"}
        or build["schema"] != "autoforecast.build.v1"
        or not isinstance(build["code_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", build["code_sha256"]) is None
        or (
            build["source_commit"] is not None
            and (
                not isinstance(build["source_commit"], str)
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", build["source_commit"]) is None
            )
        )
    ):
        raise ValueError("installed wheel has an invalid build record")
    code = {name: digest for name, digest in entries.items() if name.endswith(".py")}
    actual = set()
    for package in {name.split("/", 1)[0] for name in code}:
        root = pathlib.Path(str(distribution.locate_file(package)))
        if root.is_symlink():
            raise ValueError("installed library package is a symbolic link")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("installed library contains a symbolic link")
            if path.is_file() and path.suffix == ".py":
                actual.add(f"{package}/{path.relative_to(root).as_posix()}")
    if actual != code.keys():
        raise ValueError("installed library Python files differ from RECORD")
    code_digest = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    if code_digest != build["code_sha256"]:
        raise ValueError("installed library code differs from its build record")
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return LibraryIdentity(distribution.version, digest, build)


def installed_dependencies() -> tuple[tuple[str, str], ...]:
    """Record the actual environment without capturing credentials or private index URLs."""
    return tuple(
        sorted(
            (d.metadata["Name"], d.version)
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        )
    )
