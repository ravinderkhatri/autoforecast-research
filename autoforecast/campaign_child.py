"""Exec one campaign command while retaining its child and device leases."""

from __future__ import annotations

import os
import pathlib
import socket
import stat
import sys
import time
from typing import cast

from autoforecast.campaign import (
    CHILD_IDENTITY_SCHEMA,
    CampaignError,
    _current_process_create_time_ns,
    _digest_path,
    _strict_json,
    _validate_worker_request,
    _write_new_json,
)
from autoforecast.training_lease import (
    HOST_TRAINING_LEASE_FD,
    host_training_lease_path,
    verify_inherited_host_training_lease,
)


def _inherited_regular_file(fd_name: str, expected: pathlib.Path) -> int:
    raw = os.environ.pop(fd_name, None)
    try:
        descriptor = int(raw or "")
        opened = os.fstat(descriptor)
        named = os.stat(expected, follow_symlinks=False)
    except (OSError, TypeError, ValueError) as error:
        raise CampaignError(f"campaign child did not inherit a valid {fd_name}") from error
    if (
        descriptor < 0
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise CampaignError(f"campaign child inherited an unsafe {fd_name}")
    return descriptor


def main(request_argument: str) -> int:
    request_path = pathlib.Path(os.path.abspath(pathlib.Path(request_argument).expanduser()))
    request = _strict_json(request_path, "campaign child request")
    _validate_worker_request(request, request_path)
    attempt_dir = pathlib.Path(cast(str, request["attempt_dir"]))
    _inherited_regular_file("AUTOFORECAST_CAMPAIGN_CHILD_LOCK_FD", attempt_dir / ".child.lock")
    _inherited_regular_file(
        "AUTOFORECAST_CAMPAIGN_DEVICE_LOCK_FD",
        pathlib.Path(cast(str, request["device_lease_path"])),
    )
    host_descriptor = _inherited_regular_file(
        HOST_TRAINING_LEASE_FD, host_training_lease_path()
    )
    verify_inherited_host_training_lease(host_descriptor)
    os.environ[HOST_TRAINING_LEASE_FD] = str(host_descriptor)
    identity = {
        "schema": CHILD_IDENTITY_SCHEMA,
        "worker_request_sha256": _digest_path(request_path),
        "child_pid": os.getpid(),
        "process_create_time_ns": _current_process_create_time_ns(),
        "host": socket.gethostname(),
        "started_at_ns": time.time_ns(),
    }
    _write_new_json(attempt_dir / "child-identity.json", identity)
    command = cast(list[str], request["command"])
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError as error:
        print(f"campaign command could not start: {type(error).__name__}: {error}", file=sys.stderr)
        return 127 if error.errno == 2 else 126


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m autoforecast.campaign_child WORKER_REQUEST")
    raise SystemExit(main(sys.argv[1]))
