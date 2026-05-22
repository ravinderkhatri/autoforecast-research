"""Content-addressed, read-only executable-source snapshots for isolated experiment workers.

The governed experiment never asks a long-lived worker to import from a mutable checkout.  The
parent records every Python file in the import trees plus the environment lock/config files that
training provenance hashes, copies exactly those bytes to a private content-addressed directory,
and workers authenticate that directory before and after their callback. Data, model weights,
gates, caches, and generated outputs are deliberately not part of this source snapshot; those
inputs have their own artifact contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from typing import cast

EXECUTABLE_SOURCE_MANIFEST_SCHEMA = "autoforecast.executable-source-manifest.v2"
EXECUTABLE_SOURCE_SNAPSHOT_SCHEMA = "autoforecast.executable-source-snapshot.v2"
SOURCE_MANIFEST_FILENAME = "_autoforecast_source_manifest.json"
SOURCE_ROOTS = ("autoforecast", "forecast", "provenance", "scripts", "spec")
SUPPORTING_EXECUTABLE_SOURCES = ("pyproject.toml", "uv.lock")

# These are called out separately so an apparently complete manifest cannot silently omit the
# experiment boundary, provider implementations, isolated bridge, structural producer, or final
# scorers while still containing some unrelated Python files.
CRITICAL_EXECUTABLE_SOURCES = (
    "autoforecast/adapters/foundation.py",
    "autoforecast/adapters/isolated.py",
    "autoforecast/adapters/mft.py",
    "autoforecast/adapters/timesfm_transformers.py",
    "autoforecast/adapters/worker.py",
    "autoforecast/calibration.py",
    "autoforecast/executable_source.py",
    "autoforecast/score.py",
    "autoforecast/score_spark.py",
    "autoforecast/wrmsse.py",
    "forecast/features/commercial.py",
    "forecast/models/mft/trunk.py",
)


class ExecutableSourceError(ValueError):
    """The executable source tree is not the exact authenticated snapshot."""


def build_installed_executable_source_manifest(root: pathlib.Path) -> dict[str, object]:
    """Retain installed code with wheel provenance and no source-checkout requirement."""
    from autoforecast.project import installed_library_identity

    identity = installed_library_identity().as_record()
    files = {}
    for name in SOURCE_ROOTS:
        for path in sorted((root / name).rglob("*.py")):
            if path.is_symlink():
                raise ExecutableSourceError("installed source contains a symlink")
            files[path.relative_to(root).as_posix()] = _file_sha256(path)
    if set(CRITICAL_EXECUTABLE_SOURCES) - set(files):
        raise ExecutableSourceError("installed wheel lacks required executable sources")
    return {"schema": "autoforecast.installed-executable-source.v1",
            "library": identity, "library_root": str(root),
            "source_roots": list(SOURCE_ROOTS),
            "critical_sources": list(CRITICAL_EXECUTABLE_SOURCES), "files": files}


def _canonical_directory(
    path: pathlib.Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> pathlib.Path:
    """Return one lexical/canonical directory path with no symbolic ancestor."""
    if not path.is_absolute():
        raise ExecutableSourceError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=not allow_missing)
    except OSError as exc:
        raise ExecutableSourceError(f"{label} is not resolvable") from exc
    if resolved != path:
        raise ExecutableSourceError(f"{label} must be canonical and symlink-free")
    if allow_missing and not path.exists():
        return path
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ExecutableSourceError(f"{label} is not readable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ExecutableSourceError(f"{label} must be a real directory")
    return path


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: pathlib.Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExecutableSourceError(
            f"cannot authenticate executable source Git identity: {completed.stderr.strip()}"
        )
    return completed.stdout


def _git_bytes(
    root: pathlib.Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutableSourceError(
            f"cannot authenticate executable source Git identity: {message}"
        )
    return completed.stdout


def _relative_source_path(raw_path: bytes) -> str | None:
    try:
        relative = raw_path.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExecutableSourceError("executable source Git path is not valid UTF-8") from exc
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != relative
    ):
        raise ExecutableSourceError(
            f"executable source Git path is outside the authorized roots: {relative!r}"
        )
    if relative in SUPPORTING_EXECUTABLE_SOURCES:
        return relative
    if pure.parts[0] not in SOURCE_ROOTS:
        raise ExecutableSourceError(
            f"executable source Git path is outside the authorized roots: {relative!r}"
        )
    return relative if pure.suffix == ".py" else None


def _committed_blob_sha256s(
    root: pathlib.Path,
    object_ids: set[str],
) -> dict[str, str]:
    ordered = sorted(object_ids)
    output = _git_bytes(
        root,
        "cat-file",
        "--batch",
        input_bytes="".join(f"{object_id}\n" for object_id in ordered).encode("ascii"),
    )
    offset = 0
    digests: dict[str, str] = {}
    for expected_object_id in ordered:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise ExecutableSourceError("committed executable source blob response is truncated")
        header = output[offset:header_end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ExecutableSourceError(
                f"committed executable source object is not a blob: {expected_object_id}"
            )
        try:
            returned_object_id = header[0].decode("ascii")
            size = int(header[2])
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExecutableSourceError(
                f"committed executable source blob header is malformed: {expected_object_id}"
            ) from exc
        content_start = header_end + 1
        content_end = content_start + size
        if (
            returned_object_id != expected_object_id
            or content_end >= len(output)
            or output[content_end : content_end + 1] != b"\n"
        ):
            raise ExecutableSourceError(
                f"committed executable source blob response differs: {expected_object_id}"
            )
        digests[expected_object_id] = hashlib.sha256(output[content_start:content_end]).hexdigest()
        offset = content_end + 1
    if offset != len(output):
        raise ExecutableSourceError("committed executable source blob response has extra bytes")
    return digests


def _head_python_blobs(
    root: pathlib.Path,
    commit: str,
) -> dict[str, tuple[str, str, str]]:
    output = _git_bytes(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
        "--",
        *SOURCE_ROOTS,
        *SUPPORTING_EXECUTABLE_SOURCES,
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_type, raw_object_id = metadata.split()
            mode = raw_mode.decode("ascii")
            object_type = raw_type.decode("ascii")
            object_id = raw_object_id.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ExecutableSourceError(
                "committed executable source tree entry is malformed"
            ) from exc
        relative = _relative_source_path(raw_path)
        if relative is None:
            continue
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise ExecutableSourceError(
                f"committed Python source is not a regular blob: {relative}"
            )
        if relative in entries:
            raise ExecutableSourceError(
                f"committed executable Python source is duplicated: {relative}"
            )
        entries[relative] = (mode, object_id)
    missing = sorted(
        (set(CRITICAL_EXECUTABLE_SOURCES) | set(SUPPORTING_EXECUTABLE_SOURCES)) - set(entries)
    )
    if missing:
        raise ExecutableSourceError(f"critical executable sources are absent from HEAD: {missing}")
    blob_digests = _committed_blob_sha256s(root, {object_id for _, object_id in entries.values()})
    return {
        relative: (mode, object_id, blob_digests[object_id])
        for relative, (mode, object_id) in sorted(entries.items())
    }


def _index_python_blobs(root: pathlib.Path) -> dict[str, tuple[str, str]]:
    output = _git_bytes(
        root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        *SOURCE_ROOTS,
        *SUPPORTING_EXECUTABLE_SOURCES,
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_object_id, raw_stage = metadata.split()
            mode = raw_mode.decode("ascii")
            object_id = raw_object_id.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ExecutableSourceError("executable source index entry is malformed") from exc
        relative = _relative_source_path(raw_path)
        if relative is None:
            continue
        if stage != "0" or relative in entries:
            raise ExecutableSourceError(
                f"executable Python source index has unresolved entries: {relative}"
            )
        entries[relative] = (mode, object_id)
    return entries


def _assert_regular_live_source(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = root
    parts = pathlib.PurePosixPath(relative).parts
    for part in parts[:-1]:
        path = path / part
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise ExecutableSourceError(
                f"executable Python source path is absent: {relative}"
            ) from exc
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ExecutableSourceError(
                f"executable Python source parent is not a real directory: {relative}"
            )
    path = path / parts[-1]
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ExecutableSourceError(f"executable Python source path is absent: {relative}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ExecutableSourceError(
            f"executable Python source path is not a regular file: {relative}"
        )
    if metadata.st_nlink != 1:
        raise ExecutableSourceError(
            f"executable Python source path must not be hard-linked: {relative}"
        )
    return path


def _filesystem_python_sources(root: pathlib.Path) -> set[str]:
    discovered: set[str] = set(SUPPORTING_EXECUTABLE_SOURCES)

    def refuse_walk_error(error: OSError) -> None:
        raise ExecutableSourceError(
            f"executable source tree is unreadable: {error.filename}"
        ) from error

    for relative_root in SOURCE_ROOTS:
        source_root = root / relative_root
        try:
            root_mode = source_root.lstat().st_mode
        except FileNotFoundError as exc:
            raise ExecutableSourceError(f"executable source root is absent: {source_root}") from exc
        if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
            raise ExecutableSourceError(
                f"executable source root is not a real directory: {source_root}"
            )
        for directory, child_directories, filenames in os.walk(
            source_root,
            followlinks=False,
            onerror=refuse_walk_error,
        ):
            directory_path = pathlib.Path(directory)
            candidates = [*child_directories, *filenames]
            for name in candidates:
                if pathlib.PurePath(name).suffix != ".py":
                    continue
                path = directory_path / name
                relative = str(path.relative_to(root))
                mode = path.lstat().st_mode
                if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                    raise ExecutableSourceError(
                        f"executable Python source path is not a regular file: {relative}"
                    )
                discovered.add(relative)
    return discovered


def build_live_executable_source_manifest(root: str | pathlib.Path) -> dict[str, object]:
    """Hash tracked import trees, environment support files, and exact Git identity."""
    repository = pathlib.Path(root)
    if not repository.is_absolute() or not repository.is_dir() or repository.is_symlink():
        raise ExecutableSourceError("executable source repository must be an absolute directory")
    repository = repository.resolve()
    git_root = pathlib.Path(_git(repository, "rev-parse", "--show-toplevel").strip()).resolve()
    if git_root != repository:
        raise ExecutableSourceError(
            f"executable source root {repository} is not the Git worktree root {git_root}"
        )
    commit = _git(repository, "rev-parse", "HEAD").strip()
    tree = _git(repository, "rev-parse", f"{commit}^{{tree}}").strip()
    committed = _head_python_blobs(repository, commit)
    committed_index = {
        relative: (mode, object_id) for relative, (mode, object_id, _) in committed.items()
    }
    current_index = _index_python_blobs(repository)
    if current_index != committed_index:
        changed = sorted(
            relative
            for relative in set(current_index) | set(committed_index)
            if current_index.get(relative) != committed_index.get(relative)
        )
        raise ExecutableSourceError(f"executable Python source index differs from HEAD: {changed}")
    for relative in committed:
        _assert_regular_live_source(repository, relative)
    discovered = _filesystem_python_sources(repository)
    if discovered != set(committed):
        changed = sorted(discovered ^ set(committed))
        raise ExecutableSourceError(
            f"executable Python source path set differs from HEAD: {changed}"
        )
    files: dict[str, str] = {}
    for relative, (_, _, committed_digest) in committed.items():
        actual_digest = _file_sha256(repository / relative)
        if actual_digest != committed_digest:
            raise ExecutableSourceError(
                f"executable Python source differs from its committed blob: {relative}"
            )
        files[relative] = committed_digest
    if _git(repository, "rev-parse", "HEAD").strip() != commit:
        raise ExecutableSourceError(
            "executable source HEAD changed while its identity was authenticated"
        )
    return {
        "schema": EXECUTABLE_SOURCE_MANIFEST_SCHEMA,
        "git": {
            "repository_root": str(repository),
            "commit": commit,
            "tree": tree,
        },
        "source_roots": list(SOURCE_ROOTS),
        "critical_sources": list(CRITICAL_EXECUTABLE_SOURCES),
        "files": files,
    }


def _strict_authorization(authorization: object) -> tuple[pathlib.Path, dict[str, object]]:
    if type(authorization) is not dict or set(authorization) != {
        "schema",
        "root",
        "manifest",
        "manifest_sha256",
    }:
        raise ExecutableSourceError(
            "executable source snapshot must contain schema, root, manifest and manifest_sha256"
        )
    if authorization.get("schema") != EXECUTABLE_SOURCE_SNAPSHOT_SCHEMA:
        raise ExecutableSourceError("executable source snapshot schema is not current")
    manifest = authorization.get("manifest")
    digest = authorization.get("manifest_sha256")
    if type(manifest) is not dict or type(digest) is not str:
        raise ExecutableSourceError("executable source snapshot manifest identity is malformed")
    if _canonical_sha256(manifest) != digest:
        raise ExecutableSourceError("executable source snapshot manifest hash does not verify")
    root_value = authorization.get("root")
    if type(root_value) is not str:
        raise ExecutableSourceError("executable source snapshot root must be a string")
    root = _canonical_directory(
        pathlib.Path(root_value),
        "executable source snapshot root",
    )
    if root.name != digest:
        raise ExecutableSourceError("executable source snapshot path is not content-addressed")
    return root, manifest


def _manifest_source_path(relative: str) -> pathlib.PurePosixPath:
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != relative
        or relative not in SUPPORTING_EXECUTABLE_SOURCES
        and (pure.parts[0] not in SOURCE_ROOTS or pure.suffix != ".py")
    ):
        raise ExecutableSourceError(f"executable source manifest path is invalid: {relative!r}")
    return pure


def _expected_snapshot_tree(files: Mapping[str, str]) -> tuple[set[str], set[str]]:
    expected_files = {SOURCE_MANIFEST_FILENAME}
    expected_directories: set[str] = set(SOURCE_ROOTS)
    for relative in files:
        pure = _manifest_source_path(relative)
        expected_files.add(relative)
        for parent in pure.parents:
            if str(parent) != ".":
                expected_directories.add(str(parent))
    return expected_directories, expected_files


def _snapshot_tree(root: pathlib.Path) -> tuple[set[str], set[str]]:
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise ExecutableSourceError("executable source snapshot root is absent") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise ExecutableSourceError("executable source snapshot root is not a real directory")
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if root_mode & writable:
        raise ExecutableSourceError("executable source snapshot root is writable")

    directories: set[str] = set()
    files: set[str] = set()
    modes: dict[str, int] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ExecutableSourceError(
                f"executable source snapshot directory is unreadable: {directory}"
            ) from exc
        for entry in entries:
            path = pathlib.Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ExecutableSourceError(
                    f"executable source snapshot entry is unreadable: {relative}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise ExecutableSourceError(
                    f"executable source snapshot contains a symbolic entry: {relative}"
                )
            if stat.S_ISDIR(mode):
                directories.add(relative)
                modes[relative] = mode
                pending.append(path)
            elif stat.S_ISREG(mode):
                if entry.stat(follow_symlinks=False).st_nlink != 1:
                    raise ExecutableSourceError(
                        f"executable source snapshot contains a hard-linked file: {relative}"
                    )
                files.add(relative)
                modes[relative] = mode
            else:
                raise ExecutableSourceError(
                    f"executable source snapshot contains a special entry: {relative}"
                )
    for relative, mode in modes.items():
        if mode & writable:
            kind = "directory" if relative in directories else "file"
            raise ExecutableSourceError(
                f"executable source snapshot {kind} is writable: {relative}"
            )
    return directories, files


def _authorization_bytes(authorization: object) -> bytes:
    return (json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def validate_executable_source_snapshot(authorization: object) -> dict[str, object]:
    """Authenticate every executable-source byte and the read-only snapshot boundary."""
    root, manifest = _strict_authorization(authorization)
    installed = manifest.get("schema") == "autoforecast.installed-executable-source.v1"
    identity_keys = {"library", "library_root"} if installed else {"git"}
    if set(manifest) != {"schema", "source_roots", "critical_sources", "files"} | identity_keys:
        raise ExecutableSourceError("executable source manifest field set is not exact")
    if (not installed and manifest.get("schema") != EXECUTABLE_SOURCE_MANIFEST_SCHEMA
        or manifest.get("source_roots") != list(SOURCE_ROOTS)
        or manifest.get("critical_sources") != list(CRITICAL_EXECUTABLE_SOURCES)):
        raise ExecutableSourceError("executable source manifest contract differs")
    files = manifest.get("files")
    if type(files) is not dict or any(
        type(k) is not str or type(v) is not str for k, v in files.items()
    ):
        raise ExecutableSourceError("executable source file identity is malformed")
    if installed:
        if (type(manifest.get("library")) is not dict
                or type(manifest.get("library_root")) is not str):
            raise ExecutableSourceError("installed library identity is malformed")
    else:
        git = manifest.get("git")
        if (type(git) is not dict or set(git) != {"repository_root", "commit", "tree"}
            or any(type(git.get(field)) is not str or not git[field] for field in git)):
            raise ExecutableSourceError("executable source manifest Git identity is malformed")
    required_sources = set(CRITICAL_EXECUTABLE_SOURCES)
    if not installed:
        required_sources.update(SUPPORTING_EXECUTABLE_SOURCES)
    if required_sources - set(files):
        raise ExecutableSourceError(
            "critical or supporting executable sources are absent from the manifest"
        )
    expected_directories, expected_files = _expected_snapshot_tree(files)
    discovered_directories, discovered_files = _snapshot_tree(root)
    if discovered_directories != expected_directories or discovered_files != expected_files:
        changed = sorted(
            (discovered_directories ^ expected_directories) | (discovered_files ^ expected_files)
        )
        raise ExecutableSourceError(f"executable source snapshot entry tree differs: {changed}")
    for relative, expected in files.items():
        path = root / relative
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or _file_sha256(path) != expected
        ):
            raise ExecutableSourceError(
                f"executable source snapshot file identity differs: {relative}"
            )
    stored_manifest = root / SOURCE_MANIFEST_FILENAME
    try:
        stored_bytes = stored_manifest.read_bytes()
        stored = json.loads(stored_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutableSourceError("stored executable source manifest is unreadable") from exc
    if stored != authorization or stored_bytes != _authorization_bytes(authorization):
        raise ExecutableSourceError("stored executable source authorization differs")
    return dict(cast(dict[str, object], authorization))


def validate_live_executable_source_identity(authorization: object) -> dict[str, object]:
    """Recheck that the source checkout still has the exact pre-snapshot Git/files identity."""
    validated = validate_executable_source_snapshot(authorization)
    manifest = validated["manifest"]
    assert isinstance(manifest, Mapping)
    if manifest.get("schema") == "autoforecast.installed-executable-source.v1":
        rebuilt = build_installed_executable_source_manifest(pathlib.Path(manifest["library_root"]))
    else:
        git = manifest["git"]
        assert isinstance(git, Mapping)
        rebuilt = build_live_executable_source_manifest(git["repository_root"])
    if rebuilt != manifest:
        raise ExecutableSourceError(
            "live executable source hashes or Git identity changed after snapshot publication"
        )
    return validated


def _validate_preserved_snapshot_directory(path: pathlib.Path) -> None:
    """Accept only a complete self-authenticating prior content-addressed snapshot."""
    try:
        authorization = json.loads((path / SOURCE_MANIFEST_FILENAME).read_bytes())
        validated = validate_executable_source_snapshot(authorization)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ExecutableSourceError) as exc:
        raise ExecutableSourceError(
            f"executable source snapshot parent contains invalid preserved evidence: {path}"
        ) from exc
    if validated.get("root") != str(path):
        raise ExecutableSourceError(
            f"executable source snapshot parent contains redirected preserved evidence: {path}"
        )


def materialize_executable_source_snapshot(
    repository_root: str | pathlib.Path,
    snapshot_parent: str | pathlib.Path,
    *, installed: bool = False,
) -> dict[str, object]:
    """Publish one private content-addressed snapshot after before/copy/after authentication."""
    repository = _canonical_directory(
        pathlib.Path(repository_root),
        "executable source repository",
    )
    parent = pathlib.Path(snapshot_parent)
    _canonical_directory(
        parent,
        "executable source snapshot parent",
        allow_missing=True,
    )
    build_manifest = (build_installed_executable_source_manifest if installed
                      else build_live_executable_source_manifest)
    before = build_manifest(repository)
    manifest_sha256 = _canonical_sha256(before)
    parent.mkdir(parents=True, exist_ok=True)
    _canonical_directory(parent, "executable source snapshot parent")
    target = parent / manifest_sha256
    authorization = {
        "schema": EXECUTABLE_SOURCE_SNAPSHOT_SCHEMA,
        "root": str(target),
        "manifest": before,
        "manifest_sha256": manifest_sha256,
    }
    if target.exists() or target.is_symlink():
        validate_executable_source_snapshot(authorization)
        validate_live_executable_source_identity(authorization)
        return authorization
    for existing in parent.iterdir():
        metadata = existing.lstat()
        if existing.name.startswith(".source."):
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ExecutableSourceError(
                    "executable source snapshot parent contains invalid partial evidence: "
                    f"{existing}"
                )
            continue
        if (
            len(existing.name) == 64
            and all(character in "0123456789abcdef" for character in existing.name)
            and stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
        ):
            _validate_preserved_snapshot_directory(existing)
            continue
        raise ExecutableSourceError(
            f"executable source snapshot parent contains unknown evidence: {existing}"
        )
    temporary = pathlib.Path(tempfile.mkdtemp(dir=parent, prefix=".source."))
    try:
        files = before["files"]
        assert isinstance(files, dict)
        for source_root in SOURCE_ROOTS:
            (temporary / source_root).mkdir(parents=True, exist_ok=True)
        for relative in files:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repository / relative, destination)
        after_copy = build_manifest(repository)
        if after_copy != before:
            raise ExecutableSourceError(
                "live executable source hashes or Git identity changed while snapshotting"
            )
        (temporary / SOURCE_MANIFEST_FILENAME).write_text(
            json.dumps(authorization, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        for path in sorted(temporary.rglob("*"), key=lambda value: len(value.parts), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        temporary.chmod(0o555)
        os.replace(temporary, target)
        validate_executable_source_snapshot(authorization)
        validate_live_executable_source_identity(authorization)
        return authorization
    except BaseException:
        if temporary.exists():
            for path in temporary.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
            temporary.chmod(0o755)
            shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "CRITICAL_EXECUTABLE_SOURCES",
    "EXECUTABLE_SOURCE_MANIFEST_SCHEMA",
    "EXECUTABLE_SOURCE_SNAPSHOT_SCHEMA",
    "ExecutableSourceError",
    "SOURCE_MANIFEST_FILENAME",
    "SUPPORTING_EXECUTABLE_SOURCES",
    "build_live_executable_source_manifest",
    "materialize_executable_source_snapshot",
    "validate_executable_source_snapshot",
    "validate_live_executable_source_identity",
]
