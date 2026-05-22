"""One per-user, host-wide lease for trainable forecast runs."""

from __future__ import annotations

import contextlib
import fcntl
import functools
import os
import pathlib
import pwd
import stat
from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar

HOST_TRAINING_LEASE_FD = "AUTOFORECAST_HOST_TRAINING_LEASE_FD"
_P = ParamSpec("_P")
_R = TypeVar("_R")


class TrainingLeaseError(RuntimeError):
    """Another run owns the host, or its lease cannot be verified safely."""


def host_training_lease_path() -> pathlib.Path:
    """Use the OS account home, independent of a worktree or inherited HOME."""
    home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    root = home / ".autoforecast"
    if root.is_symlink():
        raise TrainingLeaseError("host training lease directory may not be a symlink")
    root.mkdir(mode=0o700, exist_ok=True)
    metadata = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise TrainingLeaseError("host training lease directory is unsafe")
    return root / "training.lock"


def open_host_training_lease():
    """Open a stable regular lock inode without following a symlink."""
    path = host_training_lease_path()
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TrainingLeaseError("host training lease file is unsafe")
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        if "descriptor" in locals():
            os.close(descriptor)
        raise


def verify_inherited_host_training_lease(descriptor: int) -> None:
    """Require the inherited descriptor itself to own the host lease."""
    path = host_training_lease_path()
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            descriptor < 0
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise TrainingLeaseError("inherited host training lease is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError) as error:
        raise TrainingLeaseError("inherited host training lease is unavailable") from error


def inherited_training_lease_pass_fds() -> tuple[int, ...]:
    """Keep an authenticated campaign lease alive through a nested subprocess."""
    inherited = os.environ.get(HOST_TRAINING_LEASE_FD)
    if inherited is None:
        return ()
    try:
        descriptor = int(inherited)
    except ValueError as error:
        raise TrainingLeaseError("inherited host training lease FD is invalid") from error
    verify_inherited_host_training_lease(descriptor)
    return (descriptor,)


@contextlib.contextmanager
def training_lease() -> Iterator[None]:
    """Hold the host lease across a direct run or reuse an authenticated inherited FD."""
    inherited = os.environ.get(HOST_TRAINING_LEASE_FD)
    if inherited is not None:
        try:
            descriptor = int(inherited)
        except ValueError as error:
            raise TrainingLeaseError("inherited host training lease FD is invalid") from error
        verify_inherited_host_training_lease(descriptor)
        yield
        return
    with open_host_training_lease() as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrainingLeaseError("another training run owns the host lease") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def with_training_lease(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Keep the public direct-run function and its CLI on the same lease boundary."""

    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with training_lease():
            return function(*args, **kwargs)

    return guarded
