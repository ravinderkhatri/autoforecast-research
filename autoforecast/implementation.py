"""Capture explicitly scoped saved model source using existing artifact storage."""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import hashlib
import inspect
import json
import logging
import os
import pathlib
import platform
import re
from collections.abc import Mapping
from contextlib import contextmanager

from mlops._safe_files import publish_bytes_set, read_regular_bytes

from autoforecast.project import installed_dependencies, library_revision

_DEFAULT_INCLUDE = ("*.py", "*.yaml", "*.yml", "*.json", "*.toml", "*.ini", "*.cfg", "*.sql")
_EXCLUDED = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", "node_modules", "build", "dist", "data",
    "outputs", "runs", "artifacts", ".autoforecast", ".ipynb_checkpoints",
})
_SECRET = (".env*", "*credentials*", "*secret*", "*.pem", "*.key", "*.p12", "*.pfx")
_PREFIX = "metadata/implementation"


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _event(stage):
    logger = logging.getLogger(__name__)
    logger.info("implementation %s START", stage)
    try:
        yield
    except BaseException:
        logger.error("implementation %s FAIL", stage)
        raise
    else:
        logger.info("implementation %s PASS", stage)


def source_scope(value, base=None):
    """Validate scope settings; resolve roots once against the original config directory."""
    with _event("scope"):
        if not isinstance(value, Mapping) or set(value) - {"source_roots", "include", "exclude"}:
            raise ValueError("implementation scope requires source_roots")
        roots = value.get("source_roots")
        if not isinstance(roots, Mapping) or not roots:
            raise ValueError("implementation source_roots must be a nonempty mapping")
        result: dict = {"source_roots": {}}
        for name, path in roots.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
                    or not isinstance(path, str) or not path.strip() or "://" in path):
                raise ValueError("source_roots requires logical names and filesystem paths")
            root = pathlib.Path(path).expanduser()
            if base is not None:
                root = pathlib.Path(os.path.abspath(pathlib.Path(base) / root))
            result["source_roots"][name] = str(root)
        for key, default in (("include", _DEFAULT_INCLUDE), ("exclude", ())):
            patterns = value.get(key, list(default))
            if (not isinstance(patterns, list) or (key == "include" and not patterns)
                    or any(not isinstance(p, str) or not p or p.startswith("/")
                           or ".." in p.split("/") for p in patterns)):
                raise ValueError(f"implementation {key} must contain relative glob patterns")
            result[key] = sorted(set(patterns))
        return result


def _collect(scope):
    roots = {name: pathlib.Path(path) for name, path in scope["source_roots"].items()}
    resolved = [path.resolve() for path in roots.values()]
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(resolved) for b in resolved[i + 1:]):
        raise ValueError("implementation source roots must not overlap")
    contents = {}
    for name, root in sorted(roots.items()):
        if not root.is_dir() or root.is_symlink():
            raise ValueError("implementation source root must be an existing regular directory")
        count = 0
        def on_error(error):
            raise error
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
            relative = pathlib.Path(directory).relative_to(root)
            def excluded(filename, relative=relative):
                path = (relative / filename).as_posix()
                return (filename.lower() in _EXCLUDED
                        or any(fnmatch.fnmatch(filename.lower(), p) for p in _SECRET)
                        or any(fnmatch.fnmatch(path, p) for p in scope["exclude"]))
            dirs[:] = sorted(d for d in dirs if not excluded(d))
            for dirname in dirs:
                if (pathlib.Path(directory) / dirname).is_symlink():
                    raise ValueError("implementation source directories must not be symlinks")
            for filename in sorted(files):
                path = (relative / filename).as_posix()
                if excluded(filename) or not any(
                    fnmatch.fnmatch(path, p) for p in scope["include"]
                ):
                    continue
                contents[f"{name}/{path}"] = read_regular_bytes(root, path)
                count += 1
        if not count:
            raise ValueError(f"implementation source root {name} contains no selected files")
    manifest = {
        "schema": "autoforecast.saved-source.v1",
        "scope": {"roots": sorted(roots), "include": scope["include"], "exclude": scope["exclude"],
                  "always_excluded_directories": sorted(_EXCLUDED),
                  "always_excluded_names": list(_SECRET)},
        "files": {path: {"sha256": _sha(payload), "size": len(payload)}
                  for path, payload in sorted(contents.items())},
    }
    return _bytes(manifest), tuple(sorted(contents.items()))


@dataclasses.dataclass(frozen=True)
class CapturedImplementation(Mapping):
    """Immutable baseline; its four-field view remains the existing model identity."""

    identity_bytes: bytes
    scope_bytes: bytes
    manifest_bytes: bytes
    source_bytes: tuple[tuple[str, bytes], ...]
    evidence_bytes: bytes

    def __getitem__(self, key):
        return json.loads(self.identity_bytes)[key]

    def __iter__(self):
        return iter(json.loads(self.identity_bytes))

    def __len__(self):
        return 4

    @property
    def binding(self):
        return {"mode": "saved_source", "manifest_sha256": _sha(self.manifest_bytes),
                "evidence_sha256": _sha(self.evidence_bytes)}

    def verify(self):
        """Check all selected files, including additions and deletions, against capture."""
        with _event("live_verification"):
            manifest, _ = _collect(json.loads(self.scope_bytes))
            expected = json.loads(self.manifest_bytes)
            expected.pop("callback")
            if manifest != _bytes(expected):
                raise ValueError("implementation source changed since capture")

    def retain(self, root):
        """Retain the manifest, selected bytes and separately labeled environment evidence."""
        self.verify()
        root = pathlib.Path(root)
        payloads = {f"{_PREFIX}/source/{path}": payload for path, payload in self.source_bytes}
        payloads[f"{_PREFIX}/manifest.json"] = self.manifest_bytes
        payloads[f"{_PREFIX}/evidence.json"] = self.evidence_bytes
        publish_bytes_set(root, payloads)
        self.verify()
        return self.binding


def _callback_record(callback, scope, manifest):
    if not (inspect.isfunction(callback) or inspect.ismethod(callback)
            or inspect.isclass(callback)):
        callback = inspect.unwrap(callback)
    while isinstance(callback, functools.partial):
        callback = callback.func
        if not (inspect.isfunction(callback) or inspect.ismethod(callback)
                or inspect.isclass(callback)):
            callback = inspect.unwrap(callback)
    if inspect.isclass(callback):
        raise ValueError(
            "automatic source capture requires a function wrapper for class callbacks; "
            "alternatively supply implementation= explicitly"
        )
    if isinstance(callback, staticmethod):
        callback = callback.__func__
    if inspect.ismethod(callback):
        callback = callback.__func__
    elif not inspect.isroutine(callback) and not inspect.isclass(callback):
        callback = type(callback).__call__
    module = getattr(callback, "__module__", None)
    name = getattr(callback, "__qualname__", type(callback).__qualname__)
    try:
        filename = inspect.getsourcefile(callback)
    except TypeError:
        filename = None
    record = {"module": module, "name": name, "evidence": "saved_source",
              "execution_attested": False}
    if inspect.isfunction(callback):
        record["code_name"] = callback.__code__.co_qualname
        record["first_line"] = callback.__code__.co_firstlineno
    if filename and not filename.startswith("<") and module != "__main__":
        actual = pathlib.Path(filename).resolve()
        for label, root in scope["source_roots"].items():
            if actual.is_relative_to(pathlib.Path(root).resolve()):
                logical = label + "/" + actual.relative_to(pathlib.Path(root).resolve()).as_posix()
                if logical in json.loads(manifest)["files"]:
                    record["source"] = logical
                    return record
        raise ValueError("importable callback source must belong to the selected source scope")
    logging.getLogger(__name__).warning(
        "Implementation evidence identifies saved source; interactive execution is not attested"
    )
    return record


def resolve_implementation(value, scope, callback, model_id):
    """Use an explicit identity unchanged, or capture the configured saved source."""
    if isinstance(value, CapturedImplementation):
        value.verify()
        actual = _callback_record(callback, json.loads(value.scope_bytes), value.manifest_bytes)
        if actual != json.loads(value.manifest_bytes)["callback"]:
            raise ValueError("callback differs from captured implementation")
        return value
    if value is not None:
        return value
    if scope is None:
        raise ValueError("supply implementation= or configure runtime.implementation.source_roots")
    with _event("capture"):
        scope = source_scope(scope, pathlib.Path.cwd())
        manifest, contents = _collect(scope)
        callback_record = _callback_record(callback, scope, manifest)
        manifest = _bytes({**json.loads(manifest), "callback": callback_record})
        evidence = {"callback": callback_record,
                    "root_locations": scope["source_roots"],
                    "environment": {"python": platform.python_version(),
                                    "installed_packages": installed_dependencies()},
                    "library": library_revision()}
        result = CapturedImplementation(
            _bytes({"name": model_id, "version": "saved-source-v1",
                    "source_sha256": _sha(manifest), "dependencies": {}}),
            _bytes(scope), manifest, contents, _bytes(evidence),
        )
        result.verify()
        logging.getLogger(__name__).info(
            "implementation captured files=%s sha256=%s mode=saved_source",
            len(contents), result["source_sha256"],
        )
        return result


def verify_retained(root, identity, binding):
    """Authenticate saved source without requiring the original machine's folders."""
    if binding is None or binding == {"mode": "declared"}:
        return
    with _event("reload"):
        payload = read_regular_bytes(pathlib.Path(root), f"{_PREFIX}/manifest.json")
        evidence = read_regular_bytes(pathlib.Path(root), f"{_PREFIX}/evidence.json")
        if (binding != {"mode": "saved_source", "manifest_sha256": _sha(payload),
                        "evidence_sha256": _sha(evidence)}
                or _sha(payload) != identity["source_sha256"]):
            raise ValueError("retained implementation manifest or evidence differs from binding")
        manifest = json.loads(payload)
        if manifest["schema"] != "autoforecast.saved-source.v1" or not manifest["files"]:
            raise ValueError("unsupported or empty implementation manifest")
        for path, expected in manifest["files"].items():
            source = read_regular_bytes(pathlib.Path(root), f"{_PREFIX}/source/{path}")
            if expected != {"sha256": _sha(source), "size": len(source)}:
                raise ValueError("retained implementation source differs from manifest")
        if json.loads(evidence)["callback"] != manifest["callback"]:
            raise ValueError("retained callback evidence differs from manifest")


def worker_implementation(record, callback=None):
    """Reconstruct a frozen baseline from authenticated references, then check local roots."""
    with _event("worker"):
        root = pathlib.Path(record["path"])
        verify_retained(root, record["identity"], record["binding"])
        manifest = read_regular_bytes(root, f"{_PREFIX}/manifest.json")
        scope = dict(record["scope"])
        if "AUTOFORECAST_SOURCE_ROOTS" in os.environ:
            replacement = json.loads(os.environ["AUTOFORECAST_SOURCE_ROOTS"])
            if set(replacement) != set(scope["source_roots"]):
                raise ValueError("worker source root labels must match the captured scope")
            scope["source_roots"] = replacement
        scope = source_scope(scope, pathlib.Path.cwd())
        contents = tuple((p, read_regular_bytes(root, f"{_PREFIX}/source/{p}"))
                         for p in json.loads(manifest)["files"])
        evidence = json.loads(read_regular_bytes(root, f"{_PREFIX}/evidence.json"))
        evidence["worker_root_locations"] = scope["source_roots"]
        evidence["worker_environment"] = {"python": platform.python_version(),
                                          "installed_packages": installed_dependencies()}
        evidence["worker_library"] = library_revision()
        result = CapturedImplementation(_bytes(record["identity"]), _bytes(scope), manifest,
                                        contents, _bytes(evidence))
        result.verify()
        if callback is not None:
            resolve_implementation(result, None, callback, record["identity"]["name"])
        return result
