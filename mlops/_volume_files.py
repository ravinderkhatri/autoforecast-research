"""Native file operations for Unity Catalog volumes.

Volumes look like local directories but do not support append or hard links.
The Files API supplies create-if-absent writes. Credentials come from the
Databricks SDK's normal authentication; no workspace or client is hard-coded.

Callers own their output directories and keep source snapshots unchanged while
reading. These path-based operations cannot pin a remote directory against
external replacement. Locks coordinate only writers that follow this contract.
"""

from __future__ import annotations

import contextlib
import contextvars
import errno
import hashlib
import io
import json
import os
import pathlib
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TypeVar, cast

from autoforecast._execution import operation

_T = TypeVar("_T")


def is_volume(path: pathlib.Path) -> bool:
    """Return whether this is an absolute Unity Catalog volume path."""
    return pathlib.Path(os.path.abspath(path)).parts[1:2] == ("Volumes",)


@lru_cache(maxsize=1)
def _client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _path(path: pathlib.Path) -> str:
    path = pathlib.Path(os.path.abspath(path))
    if not is_volume(path) or len(path.parts) < 6:
        raise ValueError("volume files require /Volumes/catalog/schema/volume/file")
    return str(path)


def _put(path: pathlib.Path, contents, *, overwrite: bool) -> None:
    from databricks.sdk.service.files import FilesAPI

    reads = _FROZEN.get()
    if reads is not None:
        # A written file is never served from an older copy.
        reads.drop_file(os.path.abspath(path))

    # Use the stable upload API so the SDK also supplies workspace routing.
    # Pass overwrite explicitly; the API defaults to replacing existing files.
    FilesAPI(_client().api_client).upload(
        _path(path), io.BytesIO(contents) if isinstance(contents, bytes) else contents,
        overwrite=overwrite,
    )


class _FrozenReads:
    """Local copies of frozen volume files, kept for one run scope."""

    def __init__(self, directory: pathlib.Path):
        self.directory = directory
        self.roots: set[str] = set()
        # Each copy records the fresh check it was downloaded in (0: none).
        self.files: dict[str, tuple[int, pathlib.Path]] = {}
        self.trees: dict[tuple[str, bool], tuple[int, pathlib.Path]] = {}
        self.check = 0
        self.depth = 0
        # Owned copies of frozen inputs made in this run, by (source, SHA, scratch folder).
        self.copies: dict[tuple[str, str, str], pathlib.Path] = {}
        # Digests of the local copies above, by (copy, kind): each copy is hashed once.
        self.digests: dict[tuple[str, object], object] = {}
        # Verified read-backs of files this run created; served once a mark covers them.
        self.published: dict[str, tuple[int, pathlib.Path]] = {}

    def covers(self, path: str) -> bool:
        return any(path == root or path.startswith(root + os.sep) for root in self.roots)

    def reusable(self, entry) -> bool:
        return entry is not None and (not self.depth or entry[0] == self.check)

    def within(self, path: str, parquet_only: bool) -> pathlib.Path | None:
        """A reusable copy of this tree, or of a tree that contains it, if one exists."""
        for (root, copied_parquet_only), tree in list(self.trees.items()):
            # A Parquet-only copy has empty markers, so it serves only Parquet-only reads.
            if (not self.reusable(tree) or copied_parquet_only and not parquet_only
                    or not (path == root or path.startswith(root + os.sep))):
                continue
            local = tree[1] / os.path.relpath(path, root) if path != root else tree[1]
            if local.exists():
                return local
        return None

    def existing(self, path: str) -> pathlib.Path | None:
        """A reusable local copy of one frozen file, alone or inside a copied tree."""
        from provenance.lineage import hidden_from_readers

        for entry in (self.files.get(path), self.published.get(path)):
            if entry is not None and self.reusable(entry):
                return entry[1]
        for (root, parquet_only), tree in list(self.trees.items()):
            if not self.reusable(tree) or not (path == root or path.startswith(root + os.sep)):
                continue
            relative = os.path.relpath(path, root)
            # Markers in a Parquet-only copy kept their names but not their bytes.
            if parquet_only and relative != "." and hidden_from_readers(
                    pathlib.PurePosixPath(relative)):
                continue
            local = tree[1] if relative == "." else tree[1] / relative
            if local.is_file():
                return local
        return None

    def local(self, path: str) -> pathlib.Path:
        """A local copy of one frozen file: reused, or downloaded once."""
        reused = self.existing(path)
        if reused is not None:
            return reused
        local = self.directory / uuid.uuid4().hex / pathlib.Path(path).name
        local.parent.mkdir()
        with local.open("wb") as output:
            for block in _download_blocks(pathlib.Path(path)):
                output.write(block)
        _read_only(local)
        self.files[path] = (self.check if self.depth else 0, local)
        return local

    def drop_file(self, path: str) -> None:
        """Stop serving copies of one remote file, alone or inside a copied tree."""
        self.files.pop(path, None)
        self.published.pop(path, None)
        for key in [key for key in self.trees if path.startswith(key[0] + os.sep)]:
            del self.trees[key]

    def drop(self, path: str) -> None:
        """Stop serving copies of a remote file or tree, and of any tree that contains it."""
        def related(root: str) -> bool:
            return (path == root or root.startswith(path + os.sep)
                    or path.startswith(root + os.sep))

        for store in (self.files, self.published):
            for name in [name for name in store if related(name)]:
                del store[name]
        for key in [key for key in self.trees if related(key[0])]:
            del self.trees[key]


def _read_only(local: pathlib.Path) -> None:
    """Make a run-scope copy's files read-only: the run hashes each copy once."""
    for item in [local] if local.is_file() else local.rglob("*"):
        if item.is_file():
            item.chmod(0o444)


# Inside ``frozen_reads()``, each file below a tree registered with ``mark_frozen`` is
# downloaded and hashed once, and later reads in the scope reuse those bytes. The check
# that ends a stage runs inside ``fresh_reads()``. With ``file_checks("every_stage")``
# each frozen file is downloaded and hashed again there, once, and the new bytes replace
# the old. With "once", the default, the run keeps the copies it already checked.
# Outside a scope every read downloads.
_FROZEN: ContextVar[_FrozenReads | None] = ContextVar("volume_frozen_reads", default=None)
FILE_CHECKS = ("once", "every_stage")
_CHECKS: ContextVar[str] = ContextVar("volume_file_checks", default="once")


@contextmanager
def file_checks(mode: str | None):
    """Choose when a run checks a file again: "once" (default) or "every_stage".

    "once": each file is downloaded and checked once per run, and the run trusts that
    nothing changes its files while it runs. "every_stage": each stage also ends with a
    fresh download and check, so a change made during the run is caught. None keeps the
    current choice. A nested choice applies only inside its block.
    """
    if mode is None:
        yield
        return
    if mode not in FILE_CHECKS:
        raise ValueError('file_checks must be "once" or "every_stage"')
    token = _CHECKS.set(mode)
    try:
        yield
    finally:
        _CHECKS.reset(token)


def every_stage() -> bool:
    """True when each stage ends with a fresh check (``file_checks("every_stage")``)."""
    return _CHECKS.get() == "every_stage"


@contextmanager
def frozen_reads():
    """Open a run scope for frozen reads; a nested scope reuses the outer one."""
    if _FROZEN.get() is not None:
        yield
        return
    with tempfile.TemporaryDirectory(prefix="autoforecast-frozen-reads-") as directory:
        token = _FROZEN.set(_FrozenReads(pathlib.Path(directory)))
        try:
            yield
        finally:
            _FROZEN.reset(token)


def mark_frozen(path: pathlib.Path) -> None:
    """Treat a volume tree as unchanged until the current run scope ends or a fresh read."""
    reads = _FROZEN.get()
    if reads is not None and is_volume(pathlib.Path(path)):
        reads.roots.add(os.path.abspath(path))


def forget_frozen() -> None:
    """User code has run: drop every frozen copy and mark, so later reads download again."""
    reads = _FROZEN.get()
    if reads is not None:
        for store in (reads.roots, reads.files, reads.trees, reads.copies, reads.digests,
                      reads.published):
            store.clear()


def owned_copy(key: tuple[str, str, str]) -> pathlib.Path | None:
    """A copy of a frozen input this run already made and checked, if any."""
    reads = _FROZEN.get()
    return None if reads is None else reads.copies.get(key)


def remember_owned_copy(key: tuple[str, str, str], copy: pathlib.Path) -> bool:
    """Keep a checked copy for the rest of the run; False outside a run scope."""
    reads = _FROZEN.get()
    if reads is None:
        return False
    reads.copies[key] = pathlib.Path(copy)
    mark_frozen(copy)
    return True


def owned_copies() -> list[tuple[pathlib.Path, str]]:
    """Every owned copy this run kept, with the SHA its bytes must still have."""
    reads = _FROZEN.get()
    return [] if reads is None else [(copy, sha) for (_, sha, _), copy in reads.copies.items()]


@contextmanager
def fresh_reads():
    """A stage's final check: with "every_stage" checks, download each frozen file again.

    Each file is downloaded and hashed once inside this block; a nested block joins the
    outer check. With "once" checks (the default) the run keeps the copies it checked.
    """
    reads = _FROZEN.get()
    if reads is None or not every_stage():
        yield
        return
    if not reads.depth:
        reads.check += 1
    reads.depth += 1
    try:
        yield
    finally:
        reads.depth -= 1


def _frozen(path) -> _FrozenReads | None:
    reads = _FROZEN.get()
    return reads if reads is not None and reads.covers(os.path.abspath(path)) else None


def frozen_copy(path: pathlib.Path) -> pathlib.Path | None:
    """The run's local copy of a frozen volume file (downloaded once), or None."""
    reads = _frozen(path)
    return None if reads is None else reads.local(os.path.abspath(path))


def checked_once(local: pathlib.Path | str, kind: object, compute: Callable[[], _T]) -> _T:
    """Hash a run-scope copy once: the copy is read-only, so its digest is reused.

    Anything else, including every path outside a run, is hashed on every call.
    """
    reads = _FROZEN.get()
    copy = os.path.abspath(local)
    if reads is None or not pathlib.Path(copy).is_relative_to(os.path.abspath(reads.directory)):
        return compute()
    key = (copy, kind)
    if key not in reads.digests:
        reads.digests[key] = compute()
    return cast(_T, reads.digests[key])


def read_current_bytes(path: pathlib.Path) -> bytes:
    """Download a file's current bytes, never a run copy: for checks of who owns it."""
    return b"".join(_download_blocks(pathlib.Path(path)))


def read_bytes(path: pathlib.Path) -> bytes:
    """Read through the API so FUSE's cached contents cannot hide a recent write.

    A frozen file is downloaded once per run scope; see ``frozen_reads``.
    """
    return b"".join(iter_blocks(path))


def iter_blocks(path: pathlib.Path):
    """Stream a complete remote file without loading a model into memory.

    A frozen file is downloaded once per run scope; see ``frozen_reads``.
    """
    reads = _frozen(path)
    if reads is None:
        yield from _download_blocks(path)
        return
    with reads.local(os.path.abspath(path)).open("rb") as source:
        while block := source.read(1024 * 1024):
            yield block


def _download_blocks(path: pathlib.Path):
    """Download one remote file's current bytes."""
    from databricks.sdk.errors import NotFound

    try:
        response = _client().files.download(_path(path))
        if response.contents is None:
            raise OSError("volume download returned no content")
        with response.contents as stream:
            while block := stream.read(1024 * 1024):
                yield block
    except NotFound as error:
        raise FileNotFoundError(errno.ENOENT, "volume file not found", str(path)) from error
    except Exception as error:
        raise OSError(f"could not read volume file: {path}") from error


def exists(path: pathlib.Path) -> bool:
    """Check remote file existence, without relying on a cached directory listing."""
    from databricks.sdk.errors import NotFound

    try:
        _client().files.get_metadata(_path(path))
    except NotFound:
        return False
    return True


def directory_exists(path: pathlib.Path) -> bool:
    """Check a directory without relying on the mounted listing."""
    from databricks.sdk.errors import NotFound

    try:
        _client().files.get_directory_metadata(os.path.abspath(path))
    except NotFound:
        return False
    return True


def publish_blocks(path: pathlib.Path, blocks: Iterable[bytes]) -> None:
    """Create one complete file, refusing any existing destination.

    Stage the bytes locally so memory use does not grow with model size. Files
    API 2.0 accepts at most 5 GiB per file; reject larger files before uploading.
    Read back the complete result before returning success.
    """
    expected = hashlib.sha256()
    size = 0
    try:
        with tempfile.TemporaryFile() as payload:
            for block in blocks:
                size += len(block)
                if size > 5 * 1024 ** 3:
                    raise ValueError("volume artifact exceeds the Files API 2.0 limit of 5 GiB")
                payload.write(block)
                expected.update(block)
            payload.seek(0)
            _put(path, payload, overwrite=False)
        response = _client().files.download(_path(path))
        if response.contents is None:
            raise OSError("volume download returned no content")
        observed = hashlib.sha256()
        count = 0
        # Inside a run, keep the verified read-back: once a mark covers this create-only
        # file, it is the run's one download of it.
        reads = _FROZEN.get()
        kept = (None if reads is None
                else reads.directory / uuid.uuid4().hex / pathlib.Path(path).name)
        if kept is not None:
            kept.parent.mkdir()
        with response.contents as stream, (
                kept.open("wb") if kept is not None else contextlib.nullcontext()) as copy:
            while block := stream.read(1024 * 1024):
                observed.update(block)
                count += len(block)
                if copy is not None:
                    copy.write(block)
        if count != size or observed.digest() != expected.digest():
            raise OSError("published volume artifact differs from the supplied bytes")
        if reads is not None and kept is not None:
            _read_only(kept)
            reads.published[os.path.abspath(path)] = (reads.check if reads.depth else 0, kept)
    except Exception as error:
        raise OSError(f"could not publish immutable volume artifact: {path}") from error


@contextmanager
def exclusive_lock(path: pathlib.Path):
    """Claim a shared lock once; never remove or steal another writer's lock.

    This lock does not prevent external deletion or replacement of a directory.
    An interrupted writer can leave this small ownership file behind. Recovery
    must confirm that the writer stopped before deleting it. Failing closed is
    safer than treating a driver-local FUSE flock as a distributed lock.
    """
    owner = uuid.uuid4().hex.encode()
    try:
        _put(path, owner, overwrite=False)
    except Exception as error:
        raise OSError(
            f"could not claim volume lock {path}; a writer may be active or interrupted. "
            "Confirm the owner has stopped before removing an abandoned lock."
        ) from error
    failure = None
    try:
        if read_bytes(path) != owner:
            raise OSError("volume lock owner changed before use")
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            if read_bytes(path) != owner:
                raise OSError("volume lock owner changed; refusing to delete it")
            _client().files.delete(_path(path))
        except Exception as error:
            if failure is None:
                raise
            failure.add_note(f"Volume lock cleanup also failed: {error}")


def append_bytes(path: pathlib.Path, payload: bytes) -> None:
    """Keep the existing JSONL bytes and add an event under a shared file lock.

    The API replaces the complete file because volumes cannot append. All
    cooperating writers hold this lock; previous events remain the same bytes.
    """
    with exclusive_lock(path.with_name(path.name + ".lock")):
        if directory_exists(path.with_name(path.name + ".events")):
            raise OSError("segmented audit writes require their owning execution")
        previous = read_bytes(path) if exists(path) else b""
        try:
            descriptor = json.loads(previous)
        except ValueError:
            descriptor = None
        if (isinstance(descriptor, dict)
                and descriptor.get("schema") == "autoforecast.audit-segments.v1"):
            raise OSError("segmented audit writes require their owning execution")
        expected = previous + payload
        _put(path, expected, overwrite=True)
        if read_bytes(path) != expected:
            raise OSError("volume audit log differs from the recorded events")


def _publish_audit_bytes(target, value):
    try:
        publish_blocks(target, (value,))
    except OSError:
        # Upload may have committed before the response was lost. Existing
        # bytes count as success only when they are exactly this operation.
        if read_bytes(target) != value:
            raise OSError("audit segment conflicts with an existing writer") from None


def publish_audit_segment(path, payload, *, owner, sequence):
    """Save an immutable event; an uncertain acknowledgement retries identical bytes.

    The caller serializes its own sequence. The immutable descriptor refuses a
    second writer, including another process, instead of coordinating by FUSE.
    """
    if sequence == 0:
        descriptor = json.dumps({"schema": "autoforecast.audit-segments.v1", "owner": owner},
                                sort_keys=True).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Both formats claim the same first-writer lock. A legacy read/replace
        # cannot pass its absence check and then overwrite this descriptor.
        with exclusive_lock(path.with_name(path.name + ".lock")):
            _publish_audit_bytes(path, descriptor)
    directory = path.with_name(path.name + ".events")
    directory.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(payload).hexdigest()
    _publish_audit_bytes(directory / f"{sequence:012d}-{sha}.json", payload)
    return sha


def finish_audit(path, *, owner, count, last, status):
    """Close the owning execution's audit after all event acknowledgements succeed."""
    value = json.dumps({"owner": owner, "count": count, "last": last, "status": status},
                       sort_keys=True).encode()
    _publish_audit_bytes(path.with_name(path.name + ".events") / "closed.json", value)


def read_audit_bytes(path):
    """Read JSONL or an immutable segmented audit, checking sequence and hash links.

    Events alone do not establish successful completion. A completed run must
    also pass its normal completion/manifest checks.
    """
    from mlops._safe_files import read_path_bytes, snapshot_tree_types

    payload = read_path_bytes(path)
    directory = path.with_name(path.name + ".events")
    has_segments = directory_exists(directory) if is_volume(directory) else directory.exists()
    try:
        descriptor = json.loads(payload)
    except ValueError as error:
        if has_segments:
            raise ValueError("invalid audit descriptor") from error
        return payload
    if (not isinstance(descriptor, dict)
            or descriptor.get("schema") != "autoforecast.audit-segments.v1"):
        if has_segments:
            raise ValueError("invalid audit descriptor")
        return payload
    if set(descriptor) != {"schema", "owner"} or not isinstance(descriptor["owner"], str):
        raise ValueError("invalid audit descriptor")
    directory = path.with_name(path.name + ".events")
    if is_volume(directory):
        _, entries = _inventory(directory)
    else:
        entries = {
            str(member.relative_to(directory)): (kind == "directory", None, None)
            for member, kind in snapshot_tree_types(directory).items()
        }
    previous = None
    result = []
    closed = entries.pop("closed.json", None)
    for sequence, (relative, metadata) in enumerate(sorted(entries.items())):
        is_directory, _size, _modified = metadata
        if is_directory or pathlib.Path(relative).name != relative:
            raise ValueError("invalid audit segment path")
        event_bytes = read_path_bytes(directory / relative)
        sha = hashlib.sha256(event_bytes).hexdigest()
        if relative != f"{sequence:012d}-{sha}.json":
            raise ValueError("audit segment is missing, duplicated or changed")
        event = json.loads(event_bytes)
        if not isinstance(event, dict) or event.get("audit_segment") != {
            "owner": descriptor["owner"], "sequence": sequence, "previous": previous,
        }:
            raise ValueError("audit event belongs to a different sequence or writer")
        result.append(event_bytes)
        previous = sha
    if closed is not None:
        terminal = json.loads(read_path_bytes(directory / "closed.json"))
        if (closed[0] or not isinstance(terminal, dict)
                or set(terminal) != {"owner", "count", "last", "status"}
                or terminal["owner"] != descriptor["owner"]
                or type(terminal["count"]) is not int or terminal["count"] != len(result)
                or terminal["last"] != previous
                or not isinstance(terminal["status"], str)
                or terminal["status"] not in {
                    "PASS", "RESOLVED", "FAIL", "CANCELLED", "INCOMPLETE",
                }):
            raise ValueError("audit closure differs from the recorded sequence")
    if read_path_bytes(path) != payload:
        raise ValueError("audit descriptor changed while reading")
    return b"".join(result)


def _inventory(path: pathlib.Path):
    """List the real remote tree, including markers needed by the existing validators."""
    from databricks.sdk.errors import NotFound

    path = pathlib.Path(os.path.abspath(path))
    if not is_volume(path) or len(path.parts) < 5:
        raise ValueError("volume directories require /Volumes/catalog/schema/volume")
    files = _client().files
    if len(path.parts) > 5:
        try:
            metadata = files.get_metadata(_path(path))
        except NotFound:
            pass
        else:
            return False, {"": (False, metadata.content_length, metadata.last_modified)}
    files.get_directory_metadata(str(path))
    entries = {}
    pending = [path]
    while pending:
        parent = pending.pop()
        for entry in files.list_directory_contents(str(parent)):
            if entry.path is None or type(entry.is_directory) is not bool:
                raise OSError("volume listing has an incomplete entry")
            member = pathlib.Path(entry.path)
            relative = member.relative_to(path)
            if not relative.parts or ".." in relative.parts or str(relative) in entries:
                raise OSError("volume listing contains an invalid or repeated path")
            entries[str(relative)] = (entry.is_directory, entry.file_size, entry.last_modified)
            if entry.is_directory:
                pending.append(member)
    return True, entries


def child_directories(path: pathlib.Path) -> list[pathlib.Path]:
    """List the directories directly below a volume directory, or none if it is absent."""
    from databricks.sdk.errors import NotFound

    try:
        entries = list(_client().files.list_directory_contents(os.path.abspath(path)))
    except NotFound:
        return []
    directories = []
    for entry in entries:
        if entry.path is None or type(entry.is_directory) is not bool:
            raise OSError("volume listing has an incomplete entry")
        if entry.is_directory:
            directories.append(pathlib.Path(entry.path))
    return sorted(directories)


def remove_tree(path: pathlib.Path) -> None:
    """Remove an operation-owned directory, including parts absent from the mount.

    Call only after this operation created the directory. Never use this to
    recover an existing study, another writer's output, or an entire volume.
    """
    path = pathlib.Path(_path(path))
    reads = _FROZEN.get()
    if reads is not None:
        # A deleted file is never served from this run's copies.
        reads.drop(str(path))
    directory, entries = _inventory(path)
    if not directory:
        raise ValueError("snapshot cleanup requires a directory")
    files = _client().files
    for relative, (is_directory, _size, _modified) in sorted(
        entries.items(), key=lambda item: len(pathlib.Path(item[0]).parts), reverse=True,
    ):
        target = str(path / relative)
        if is_directory:
            files.delete_directory(target)
        else:
            files.delete(target)
    files.delete_directory(str(path))


# Each Files API request costs about 0.2 s of overhead regardless of size, so many small
# files are bound by round trips, not bandwidth. On serverless, 3,545 files took 93 s
# with 8 workers and 30 s with 32.
_DOWNLOAD_WORKERS = 32


def _download(remote: str, destination: pathlib.Path, copied: list[int], slot: int) -> None:
    """Copy one file, counting its bytes in its own ``copied`` slot as they arrive."""
    response = _client().files.download(remote)
    if response.contents is None:
        raise OSError("volume download returned no content")
    with response.contents as source, destination.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
            copied[slot] += len(chunk)


def _local_blocks(path: pathlib.Path):
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            yield block


def publish_files(pairs: Iterable[tuple[pathlib.Path, pathlib.Path]]) -> None:
    """Publish local files as new volume files concurrently.

    Each pair is (local source, volume destination). Every file is created only if absent
    and read back in full before it counts as published.
    """
    pairs = list(pairs)
    _client()  # Construct the shared client once, before worker threads use it.
    with operation("Copy files to the volume", engine="driver", files_total=len(pairs)) as update:
        with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
            # Workers share the caller's run scope, so each file's read-back is kept.
            running = {pool.submit(contextvars.copy_context().run, publish_blocks, remote,
                                   _local_blocks(local))
                       for local, remote in pairs}
            finished = 0
            try:
                while running:
                    done, running = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        finished += 1
                    update(completed=finished, total=len(pairs), unit="files")
            except BaseException:
                for future in running:
                    future.cancel()
                raise


@contextmanager
def local_snapshot(path: pathlib.Path, *, parquet_only: bool = False):
    """Give existing hash algorithms fresh remote bytes in a temporary local tree.

    FUSE may cache old bytes after a remote edit. Stream through the native API
    instead, preserve every relative filename, and reject an inventory change
    during the copy. The caller's existing SHA checks still decide validity.

    With ``parquet_only``, files that both directory readers ignore by name (dot
    files and Spark markers) keep their names but not their bytes. Use it only
    for callers that read nothing but Parquet parts; the inventory check still
    covers every name, size and modification time.

    Inside ``frozen_reads()``, a tree below a ``mark_frozen`` root is downloaded once and
    later calls yield that same copy; callers must only read it.
    """
    path = pathlib.Path(os.path.abspath(path))
    reads = _frozen(path)
    if reads is not None:
        # A frozen tree is downloaded once per run scope; callers only read the copy.
        key = (str(path), parquet_only)
        local = reads.within(str(path), parquet_only)
        if local is None:
            local = reads.directory / uuid.uuid4().hex / path.name
            local.parent.mkdir()
            _copy_tree(path, local, parquet_only=parquet_only, reads=reads)
            _read_only(local)
            reads.trees[key] = (reads.check if reads.depth else 0, local)
        yield local
        return
    with tempfile.TemporaryDirectory(prefix="autoforecast-volume-read-") as temporary:
        target = pathlib.Path(temporary) / path.name
        _copy_tree(path, target, parquet_only=parquet_only)
        yield target


def _copy_tree(path: pathlib.Path, target: pathlib.Path, *, parquet_only: bool,
               reads: _FrozenReads | None = None) -> None:
    """Download a remote file or tree to ``target``, refusing an inventory change.

    Within a run scope, files already downloaded in this stage are copied locally instead.
    """
    from provenance.lineage import hidden_from_readers

    directory, inventory = _inventory(path)
    if directory:
        target.mkdir()
    downloads, skipped = [], 0
    for relative, (is_directory, expected_size, _modified) in sorted(inventory.items()):
        destination = target / relative if directory else target
        if is_directory:
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if parquet_only and directory and hidden_from_readers(pathlib.PurePosixPath(relative)):
            destination.touch()
            skipped += 1
            continue
        remote = str(path / relative) if directory else str(path)
        reused = reads.existing(remote) if reads is not None else None
        if reused is not None:
            shutil.copyfile(reused, destination)
            if destination.stat().st_size != expected_size:
                raise OSError("volume file changed size since it was downloaded")
            continue
        downloads.append((remote, destination, expected_size))
    sizes = [size for _remote, _destination, size in downloads]
    total = sum(sizes) if None not in sizes else None
    copied, finished, reported = [0] * len(downloads), 0, None
    _client()  # Construct the shared client once, before worker threads use it.
    with operation("Download snapshot for integrity checks", dataset=str(path),
                   engine="driver", files_total=len(downloads),
                   **({"marker_files_not_downloaded": skipped} if skipped else {})) as update:
        with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
            pending = {pool.submit(_download, remote, destination, copied, slot):
                       (destination, size)
                       for slot, (remote, destination, size) in enumerate(downloads)}
            running = set(pending)
            try:
                # Workers only count bytes; progress is reported from this thread,
                # at least once a second while a large file is still arriving.
                while running:
                    done, running = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in done:
                        destination, expected_size = pending[future]
                        future.result()
                        finished += 1
                        if destination.stat().st_size != expected_size:
                            raise OSError("volume file changed size while it was copied")
                    if (sum(copied), finished) != reported:
                        reported = (sum(copied), finished)
                        update(completed=reported[0], total=total, unit="bytes",
                               files=finished)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        if _inventory(path) != (directory, inventory):
            raise OSError("volume directory changed while it was copied")
