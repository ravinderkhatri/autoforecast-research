"""Where things live, so a run does not have to know whether that is a disk or a bucket.

**THIS IS THE MIGRATION, and the tracking backend is not.** Every runner in this repository
hardcodes a local path -- `ROOT / "data" / "m5_parquet"`, an output root, a scratch directory --
and each one hardcodes it separately. On a cloud runner the panel is on GCS, outputs are on GCS,
and compute is a container with none of those directories. A project can have flawless lineage
recorded in Vertex ML Metadata and still be unable to run a single job there, because nothing in
it can express a path that is not on this laptop.

WHAT THIS IS NOT. It is not a filesystem abstraction layer, and it does not try to make `gs://`
behave like a POSIX directory. Spark, pandas and pyarrow all already read `gs://` given the right
connector; what they need is the STRING, and what this repository lacks is any way to supply one.
So `Location` is mostly a typed string with the three operations the runners actually perform --
join a child, ask whether it exists, make a local directory -- and it is explicit about which of
those a remote location cannot do.

**A remote location cannot be `mkdir`-ed and does not need to be.** Object stores have no
directories; a key with slashes in it is the whole of it. Calling `ensure_dir` on a `gs://`
location is therefore a no-op rather than an error, because the alternative is every caller
learning which kind it holds -- which is the coupling this module exists to remove.
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass

__all__ = ["Location", "StorageError", "Workspace", "workspace_from_env"]

_REMOTE = re.compile(r"^(gs|s3|abfss?|wasbs?)://", re.IGNORECASE)


class StorageError(ValueError):
    """A location was used in a way its backing store cannot support."""


@dataclass(frozen=True)
class Location:
    """A directory or object prefix, local or remote.

    Attributes:
        uri: The location as a string. `/data/panel` or `gs://bucket/panel`. Local paths are
            normalised to absolute on construction, because a relative path means something
            different depending on which directory a job happened to start in -- and a cloud
            runner starts somewhere else.
    """

    uri: str

    def __post_init__(self) -> None:
        """Raises: StorageError: the uri is empty."""
        if not str(self.uri).strip():
            raise StorageError("a location needs a uri; an empty one silently becomes the "
                               "working directory, which differs between a laptop and a container")
        if not self.is_remote:
            object.__setattr__(self, "uri", str(pathlib.Path(self.uri).expanduser().resolve()))

    @property
    def is_remote(self) -> bool:
        """Whether this lives in an object store rather than on a filesystem."""
        return bool(_REMOTE.match(str(self.uri)))

    @property
    def scheme(self) -> str:
        """`gs`, `s3`, ... or `file` for a local path."""
        match = _REMOTE.match(str(self.uri))
        return match.group(1).lower() if match else "file"

    def child(self, *parts: str) -> Location:
        """A location beneath this one.

        Raises:
            StorageError: a part is absolute or contains `..`. Both escape the workspace, and on
                an object store `..` is not even resolvable -- it is a literal key segment, so the
                path would silently point at something that does not exist rather than at a parent.
        """
        for part in parts:
            text = str(part)
            if not text:
                raise StorageError("empty path segment")
            if text.startswith("/") or _REMOTE.match(text):
                raise StorageError(
                    f"{text!r} is absolute; `child` composes RELATIVE segments so a workspace "
                    "cannot be escaped by accident")
            if ".." in pathlib.PurePosixPath(text).parts:
                raise StorageError(
                    f"{text!r} contains '..'. On an object store that is a literal key segment, "
                    "not a parent directory, so the result would point at nothing rather than up")
        if self.is_remote:
            return Location(str(self.uri).rstrip("/") + "/" + "/".join(str(p) for p in parts))
        return Location(str(pathlib.Path(self.uri).joinpath(*[str(p) for p in parts])))

    def exists(self) -> bool:
        """Whether anything is here.

        Raises:
            StorageError: the location is remote. Answering would need a cloud client, and this
                module deliberately has no cloud dependency -- a caller that needs it should ask
                the engine already reading the data (Spark, pyarrow) rather than have this grow
                one.
        """
        if self.is_remote:
            raise StorageError(
                f"cannot test existence of {self.uri} without a {self.scheme} client. Ask the "
                "engine that reads it; this module stays dependency-free on purpose")
        return pathlib.Path(self.uri).exists()

    def ensure_dir(self) -> Location:
        """Create the directory if local. A NO-OP when remote, and deliberately not an error.

        Object stores have no directories -- a key containing slashes is the whole of it -- so
        there is nothing to create. Raising here would make every caller check which kind of
        location it is holding, which is exactly the coupling this module removes.
        """
        if not self.is_remote:
            pathlib.Path(self.uri).mkdir(parents=True, exist_ok=True)
        return self

    def as_path(self) -> pathlib.Path:
        """The local `Path`.

        Raises:
            StorageError: the location is remote. A caller reaching for `Path` is about to use an
                API that only understands filesystems, and the failure would otherwise appear
                somewhere further away as a confusing `FileNotFoundError` on a literal `gs:/...`.
            """
        if self.is_remote:
            raise StorageError(
                f"{self.uri} is not a filesystem path. Pass `.uri` to a reader that understands "
                f"{self.scheme}://, rather than converting it to a Path")
        return pathlib.Path(self.uri)

    def __str__(self) -> str:
        return str(self.uri)


@dataclass(frozen=True)
class Workspace:
    """The three roots a run needs, so nothing has to name a directory twice.

    Attributes:
        panel: Where the input data lives. Read-only as far as any runner is concerned.
        outputs: Where results, gate records and lineage go. The only durable output root.
        scratch: Working space that may be deleted between runs. **Local by design** -- Spark
            spills, feature windows and checkpoints are re-derivable, and putting them in an
            object store buys nothing while costing every write a network round trip.
    """

    panel: Location
    outputs: Location
    scratch: Location

    def __post_init__(self) -> None:
        """Raises: StorageError: scratch is remote."""
        if self.scratch.is_remote:
            raise StorageError(
                f"scratch is {self.scratch.uri}, which is remote. Scratch holds re-derivable "
                "intermediates -- spills, feature windows -- and every write would become a "
                "network round trip for data nobody keeps")

    def run_dir(self, *parts: str) -> Location:
        """An output location for one run, created if it is local."""
        return self.outputs.child(*parts).ensure_dir()

    def describe(self) -> dict[str, str]:
        """The roots, for the run record. A result that does not say where it read from cannot
        be repeated on another machine."""
        return {"panel": str(self.panel), "outputs": str(self.outputs),
                "scratch": str(self.scratch), "panel_scheme": self.panel.scheme,
                "outputs_scheme": self.outputs.scheme}


def workspace_from_env(root: str | pathlib.Path | None = None) -> Workspace:
    """A `Workspace` from `FORECAST_PANEL`, `FORECAST_OUTPUTS` and `FORECAST_SCRATCH`.

    Args:
        root: Fallback repository root for the defaults, when the variables are unset.

    Returns:
        A `Workspace`. The defaults reproduce today's local layout exactly, so setting nothing
        changes nothing -- which is the point. A cloud runner sets two environment variables and
        the same script runs against buckets.
    """
    base = pathlib.Path(root or pathlib.Path(__file__).resolve().parent.parent)
    panel = os.environ.get("FORECAST_PANEL") or str(base / "data" / "m5_parquet")
    outputs = os.environ.get("FORECAST_OUTPUTS") or str(base / "gates")
    scratch = os.environ.get("FORECAST_SCRATCH") or str(base / ".scratch")
    return Workspace(panel=Location(panel), outputs=Location(outputs),
                     scratch=Location(scratch))
