"""Descriptor-relative publication of immutable local artifact files."""

from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import sys
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager

from mlops import _volume_files


class SafeFileError(OSError):
    """A path could not be published without following links or replacing bytes."""


_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _require_stable_read(before: os.stat_result, after: os.stat_result, label: object) -> None:
    if before.st_nlink != 1 or after.st_nlink != 1:
        raise SafeFileError(f"artifact source must have one link: {label}")
    if any(getattr(before, field) != getattr(after, field) for field in _STABLE_STAT_FIELDS):
        raise SafeFileError(f"artifact source changed while it was read: {label}")


def snapshot_tree_types(root: pathlib.Path) -> dict[pathlib.Path, str]:
    """Snapshot every descendant's lexical path and non-followed filesystem type."""
    if _volume_files.is_volume(root):
        try:
            _, entries = _volume_files._inventory(root)
            return {root / name: "directory" if value[0] else "file"
                    for name, value in entries.items()}
        except Exception as error:
            raise SafeFileError(f"could not inspect artifact tree: {root}") from error
    observed: dict[pathlib.Path, str] = {}
    try:
        descendants = tuple(root.rglob("*"))
        for candidate in descendants:
            mode = candidate.lstat().st_mode
            if stat.S_ISREG(mode):
                kind = "file"
            elif stat.S_ISDIR(mode):
                kind = "directory"
            elif stat.S_ISLNK(mode):
                kind = "symlink"
            else:
                kind = "special"
            observed[candidate] = kind
    except OSError as exc:
        raise SafeFileError(f"could not inspect artifact tree: {root}") from exc
    return observed


def _platform_absolute(path: pathlib.Path) -> pathlib.Path:
    """Return a lexical absolute path, normalizing only macOS's system aliases."""
    absolute = pathlib.Path(os.path.abspath(path.expanduser()))
    if sys.platform == "darwin":
        for alias, target in (
            (pathlib.Path("/tmp"), pathlib.Path("/private/tmp")),
            (pathlib.Path("/var"), pathlib.Path("/private/var")),
        ):
            try:
                suffix = absolute.relative_to(alias)
            except ValueError:
                continue
            if alias.is_symlink() and alias.resolve() == target:
                return target / suffix
    return absolute


def _open_directory_fd(path: pathlib.Path) -> int:
    """Open an absolute directory one no-follow component at a time."""
    absolute = _platform_absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except OSError:
        os.close(current)
        raise


def _create_directory_fd(path: pathlib.Path) -> int:
    """Create and open one new directory without following any parent component."""
    absolute = _platform_absolute(path)
    if len(absolute.parts) < 2:
        raise SafeFileError(f"artifact snapshot must name a new directory: {path}")
    parent_fd = _open_directory_fd(absolute.parent)
    try:
        os.mkdir(absolute.name, mode=0o755, dir_fd=parent_fd)
        return os.open(
            absolute.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)


def _create_fresh_directory(
    path: pathlib.Path, child_directories: Iterable[str] = ()
) -> pathlib.Path:
    """Claim a new directory hierarchy without following any existing symlink component."""
    absolute = _platform_absolute(path)
    if len(absolute.parts) < 2:
        raise SafeFileError(f"fresh directory must have a parent: {path}")
    children = tuple(child_directories)
    if any(len(_relative_parts(child)) != 1 for child in children):
        raise SafeFileError("fresh child directories must be single relative names")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:-1]:
            try:
                os.mkdir(part, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            child_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child_fd
        try:
            os.mkdir(absolute.name, mode=0o755, dir_fd=current)
        except FileExistsError as exc:
            raise SafeFileError(f"fresh directory already exists: {path}") from exc
        leaf = os.open(absolute.name, flags, dir_fd=current)
        try:
            for child in children:
                os.mkdir(child, mode=0o755, dir_fd=leaf)
            os.fsync(leaf)
        finally:
            os.close(leaf)
        os.fsync(current)
    except SafeFileError:
        raise
    except OSError as exc:
        raise SafeFileError(f"could not create link-free directory hierarchy: {path}") from exc
    finally:
        os.close(current)
    return path


def _open_regular_fd(path: pathlib.Path) -> int:
    """Open a regular file without following any component of its path."""
    absolute = _platform_absolute(path)
    if len(absolute.parts) < 2:
        raise SafeFileError(f"artifact source must name a file: {path}")
    parent = _open_directory_fd(absolute.parent)
    try:
        descriptor = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SafeFileError(f"artifact source must be a regular non-link file: {path}")
    return descriptor


def _regular_file_identity(path: pathlib.Path) -> tuple[str, int]:
    """Hash one stable regular file without following any path component."""
    descriptor: int | None = None
    try:
        if _volume_files.is_volume(path):
            descriptor = _open_directory_fd(path.parent)
            digest = hashlib.sha256()
            size = 0
            for block in _volume_files.iter_blocks(path):
                digest.update(block)
                size += len(block)
            return digest.hexdigest(), size
        descriptor = _open_regular_fd(path)
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        _require_stable_read(before, after, path)
        return digest.hexdigest(), after.st_size
    except SafeFileError:
        raise
    except OSError as exc:
        raise SafeFileError(f"could not read immutable artifact: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    relative = pathlib.PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or relative_path in {"", "."}
        or ".." in relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise SafeFileError("artifact path must be relative and must not contain '..'")
    return relative.parts


@contextmanager
def _parent_fd_at(root_fd: int, relative_path: str) -> Iterator[tuple[int, str]]:
    parts = _relative_parts(relative_path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = [os.dup(root_fd)]
    try:
        try:
            current = descriptors[0]
            for part in parts[:-1]:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=current)
                descriptors.append(child)
                current = child
        except OSError as exc:
            raise SafeFileError(
                f"artifact path is not a link-free directory hierarchy: {relative_path}"
            ) from exc
        yield current, parts[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _open_regular_at(root_fd: int, relative_path: str) -> int:
    """Open a relative regular file beneath one already-pinned root descriptor."""
    parts = _relative_parts(relative_path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
    except OSError as exc:
        raise SafeFileError(
            f"artifact source must be a regular non-link file: {relative_path}"
        ) from exc
    finally:
        os.close(current)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SafeFileError(f"artifact source must be a regular non-link file: {relative_path}")
    return descriptor


def _publish_blocks_at(root_fd: int, relative_path: str, blocks: Iterable[bytes],
                       *, volume_root: pathlib.Path | None = None) -> None:
    """Publish below one pinned root so a path replacement cannot redirect later files."""
    with _parent_fd_at(root_fd, relative_path) as (parent_fd, name):
        if volume_root is not None:
            # A volume cannot create hard links. The native API refuses replacement.
            try:
                _volume_files.publish_blocks(volume_root / relative_path, blocks)
            except OSError as error:
                raise SafeFileError(
                    f"could not publish immutable artifact: {relative_path}"
                ) from error
            return
        temporary = f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        destination_fd: int | None = None
        try:
            destination_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=parent_fd,
            )
            for block in blocks:
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
            os.close(destination_fd)
            destination_fd = None
            # A hard link publishes the complete temporary inode without replacing an
            # existing path. Both names are descriptor-relative to the verified directory.
            os.link(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            if destination_fd is not None:
                os.close(destination_fd)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise SafeFileError(f"could not publish immutable artifact: {relative_path}") from exc


def _publish_blocks(
    root: pathlib.Path, relative_path: str, blocks: Iterable[bytes]
) -> pathlib.Path:
    try:
        root_fd = _open_directory_fd(root)
    except OSError as exc:
        raise SafeFileError(
            f"artifact path is not a link-free directory hierarchy: {relative_path}"
        ) from exc
    try:
        _publish_blocks_at(root_fd, relative_path, blocks,
                           **({"volume_root": root} if _volume_files.is_volume(root) else {}))
    finally:
        os.close(root_fd)
    return root.joinpath(*_relative_parts(relative_path))


def publish_bytes(root: pathlib.Path, relative_path: str, payload: bytes) -> pathlib.Path:
    """Publish bytes once below *root* without following destination links."""
    return _publish_blocks(root, relative_path, (payload,))


def read_path_bytes(path: pathlib.Path) -> bytes:
    """Read current volume bytes; preserve normal Path.read_bytes behavior elsewhere."""
    return _volume_files.read_bytes(path) if _volume_files.is_volume(path) else path.read_bytes()


def path_exists(path: pathlib.Path) -> bool:
    """Check files and directories using the storage system that owns the path."""
    if _volume_files.is_volume(path):
        return _volume_files.directory_exists(path) or _volume_files.exists(path)
    return path.exists()


def is_regular_file(path: pathlib.Path) -> bool:
    """Check the native volume metadata, or the ordinary filesystem for local files."""
    return _volume_files.exists(path) if _volume_files.is_volume(path) else path.is_file()


def append_audit_bytes(path: pathlib.Path, payload: bytes) -> None:
    """Add an audit event, using a shared lock when the filesystem cannot append."""
    if _volume_files.is_volume(path):
        _volume_files.append_bytes(path, payload)
    else:
        with path.open("ab") as stream:
            stream.write(payload)


def _read_regular_bytes_at(root_fd: int, relative_path: str,
                           *, volume_root: pathlib.Path | None = None) -> bytes:
    if volume_root is not None:
        return _volume_files.read_bytes(volume_root.joinpath(*_relative_parts(relative_path)))
    descriptor = _open_regular_at(root_fd, relative_path)
    try:
        before = os.fstat(descriptor)
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        _require_stable_read(before, os.fstat(descriptor), relative_path)
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def read_regular_bytes(root: pathlib.Path, relative_path: str) -> bytes:
    """Read a regular artifact below a pinned, link-free root."""
    try:
        root_fd = _open_directory_fd(root)
        try:
            return _read_regular_bytes_at(
                root_fd, relative_path,
                **({"volume_root": root} if _volume_files.is_volume(root) else {}),
            )
        finally:
            os.close(root_fd)
    except SafeFileError:
        raise
    except OSError as exc:
        raise SafeFileError(f"could not read immutable artifact: {relative_path}") from exc


def publish_bytes_set(root: pathlib.Path, payloads: Mapping[str, bytes]) -> pathlib.Path:
    """Idempotently publish a byte set through one pinned directory descriptor."""
    if not payloads:
        raise SafeFileError("immutable artifact set must not be empty")
    for relative_path in payloads:
        _relative_parts(relative_path)
    root_fd: int | None = None
    try:
        root_fd = _open_directory_fd(root)
        root_identity = os.fstat(root_fd)
        for relative_path, payload in sorted(payloads.items()):
            try:
                _publish_blocks_at(
                    root_fd, relative_path, (payload,),
                    **({"volume_root": root} if _volume_files.is_volume(root) else {}),
                )
            except SafeFileError as error:
                try:
                    observed = _read_regular_bytes_at(
                        root_fd, relative_path,
                        **({"volume_root": root} if _volume_files.is_volume(root) else {}),
                    )
                except SafeFileError:
                    raise error from None
                if observed != payload:
                    raise SafeFileError(f"immutable artifact differs: {relative_path}") from error
        observed_fd = _open_directory_fd(root)
        try:
            observed_identity = os.fstat(observed_fd)
        finally:
            os.close(observed_fd)
        if (
            observed_identity.st_dev != root_identity.st_dev
            or observed_identity.st_ino != root_identity.st_ino
        ):
            raise SafeFileError("immutable artifact root changed during publication")
    except SafeFileError:
        raise
    except OSError as exc:
        raise SafeFileError(f"could not publish immutable artifact set: {root}") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return root


def copy_regular_file(root: pathlib.Path, relative_path: str, source: pathlib.Path) -> pathlib.Path:
    """Copy a regular source once below *root*, refusing source and destination links."""
    if _volume_files.is_volume(source):
        descriptor = _open_directory_fd(source.parent)
        try:
            return _publish_blocks(root, relative_path, _volume_files.iter_blocks(source))
        finally:
            os.close(descriptor)
    try:
        source_fd = _open_regular_fd(source)
    except OSError as exc:
        raise SafeFileError(f"artifact source must be a regular non-link file: {source}") from exc
    try:

        def blocks() -> Iterator[bytes]:
            while True:
                block = os.read(source_fd, 1024 * 1024)
                if not block:
                    return
                yield block

        return _publish_blocks(root, relative_path, blocks())
    finally:
        os.close(source_fd)


def snapshot_regular_files(
    source_root: pathlib.Path,
    destination_root: pathlib.Path,
    relative_paths: Iterable[str],
) -> pathlib.Path:
    """Copy an explicit immutable file set into a new link-free snapshot directory."""
    paths = tuple(relative_paths)
    if len(set(paths)) != len(paths):
        raise SafeFileError("snapshot file paths must be unique")
    for relative_path in paths:
        _relative_parts(relative_path)
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = _open_directory_fd(source_root)
        destination_fd = _create_directory_fd(destination_root)
        destination_identity = os.fstat(destination_fd)
        for relative_path in sorted(paths):
            if _volume_files.is_volume(source_root):
                _publish_blocks_at(
                    destination_fd, relative_path,
                    _volume_files.iter_blocks(source_root / relative_path),
                    **({"volume_root": destination_root}
                       if _volume_files.is_volume(destination_root) else {}),
                )
                continue
            file_fd = _open_regular_at(source_fd, relative_path)
            try:

                def blocks(descriptor: int = file_fd) -> Iterator[bytes]:
                    while True:
                        block = os.read(descriptor, 1024 * 1024)
                        if not block:
                            return
                        yield block

                _publish_blocks_at(
                    destination_fd, relative_path, blocks(),
                    **({"volume_root": destination_root}
                       if _volume_files.is_volume(destination_root) else {}),
                )
            finally:
                os.close(file_fd)
        # The pinned descriptor protects writes, but callers receive the lexical path. Refuse a
        # rename/replacement rather than returning a path to a different directory.
        observed_fd = _open_directory_fd(destination_root)
        try:
            observed_identity = os.fstat(observed_fd)
        finally:
            os.close(observed_fd)
        if (
            observed_identity.st_dev != destination_identity.st_dev
            or observed_identity.st_ino != destination_identity.st_ino
        ):
            raise SafeFileError("artifact snapshot destination changed during publication")
    except SafeFileError:
        raise
    except OSError as exc:
        raise SafeFileError(
            f"could not create or publish artifact snapshot: {destination_root}"
        ) from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
    return destination_root
