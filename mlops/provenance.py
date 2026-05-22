"""Task-agnostic source, runtime and checkpoint provenance.

The durable source of code is Git plus a separately governed mirror.  A run records a clean
commit and does not duplicate the repository.  Small resolved configuration and lockfile inputs
are copied into the immutable run bundle because they are run-specific and cheap to retain.
"""

from __future__ import annotations

import hashlib
import pathlib
import platform
import re
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

__all__ = [
    "CheckpointCapability",
    "DataSnapshot",
    "ProvenanceError",
    "RuntimeIdentity",
    "SourceIdentity",
    "capture_runtime_identity",
    "capture_source_identity",
    "local_data_reference",
    "sanitize_repository_origin",
    "sanitize_web_url",
    "validate_data_reference",
    "verify_retained_ref",
    "write_run_provenance",
]

_SHA256 = frozenset("0123456789abcdef")
_SCP_ORIGIN = re.compile(
    r"(?:(?P<user>[A-Za-z0-9._~+-]+)@)?"
    r"(?P<host>[A-Za-z0-9._~-]+):(?P<path>.+)"
)
_SCP_PATH = re.compile(r"(?:[A-Za-z0-9._~!$&'()*+,;=/-]|%[0-9A-Fa-f]{2})+")
_URI_PATH = re.compile(r"(?:[A-Za-z0-9._~!$&'()*+,;=:@/-]|%[0-9A-Fa-f]{2})*")
_DEFAULT_RELEVANT_PATHS = (
    "autoforecast",
    "forecast",
    "mlops",
    "provenance",
    "scripts",
    "spec",
    "configs",
    "experiments",
    "pyproject.toml",
    "uv.lock",
)


class ProvenanceError(ValueError):
    """A run cannot be identified or retained safely."""


def validate_data_reference(reference: object) -> str:
    """Return a stable non-secret dataset reference or refuse credential-bearing input."""
    if (
        type(reference) is not str
        or not reference.strip()
        or any(ord(char) < 32 for char in reference)
    ):
        raise ProvenanceError("data snapshot reference must be a non-blank printable string")
    try:
        parsed = urllib.parse.urlsplit(reference)
        _ = parsed.port
    except ValueError as error:
        raise ProvenanceError("data snapshot reference has an invalid authority") from error
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in reference
    ):
        raise ProvenanceError(
            "data snapshot reference must not contain credentials or signed query parameters, "
            "fragments, or backslashes"
        )
    if re.fullmatch(r"[^@/:?#]+@[^@/:?#]+:[^?#]+", reference) is not None:
        raise ProvenanceError(
            "data snapshot reference must not contain credentials or signed query parameters"
        )
    if parsed.scheme and (
        not reference.startswith(f"{parsed.scheme}://") or not parsed.netloc
    ):
        raise ProvenanceError(
            "data snapshot URI schemes must use hierarchical 'scheme://authority/path' form"
        )
    return reference


def local_data_reference(path: str | pathlib.Path, *, authority: str) -> str:
    """Return a validated, credential-free local URI for an absolute filesystem path."""
    if re.fullmatch(r"[a-z][a-z0-9_-]*", authority) is None:
        raise ProvenanceError("local data reference authority must be a lowercase identifier")
    encoded = urllib.parse.quote(pathlib.Path(path).resolve().as_posix(), safe="/")
    return validate_data_reference(
        urllib.parse.urlunsplit(("local", authority, encoded, "", ""))
    )


def sanitize_repository_origin(origin: str | None) -> str | None:
    """Retain repository location without URL credentials or signed components."""
    if origin is None:
        return None
    if not origin.strip() or any(ord(char) < 32 or ord(char) == 127 for char in origin):
        return None
    if "\\" in origin:
        return None
    if "://" not in origin:
        scp = _SCP_ORIGIN.fullmatch(origin)
        if scp is not None:
            host, path = scp.group("host"), scp.group("path")
            if _SCP_PATH.fullmatch(path) is None:
                return None
            return f"{host}:{path}"
    try:
        parsed = urllib.parse.urlsplit(origin)
    except ValueError:
        return None
    if not parsed.scheme:
        if "://" in origin:
            return None
        if any(marker in origin for marker in ("@", "?", "#")):
            return None
        return origin
    if (
        not parsed.netloc
        or any(char.isspace() for char in origin)
        or re.search(r"%(?![0-9A-Fa-f]{2})", origin) is not None
        or _URI_PATH.fullmatch(parsed.path) is None
    ):
        return None
    try:
        host = parsed.hostname or ""
        port = parsed.port
        ascii_host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return None
    if not host or re.fullmatch(r"[A-Za-z0-9._~:-]+", ascii_host) is None:
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def sanitize_web_url(value: object) -> str | None:
    """Return a credential-free HTTP(S) base URL suitable for retained dashboard links."""
    if type(value) is not str or any(ord(char) < 32 for char in value) or "\\" in value:
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        port = parsed.port
        host = parsed.hostname or ""
    except ValueError:
        return None
    if not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class SourceIdentity:
    """One clean, committed source tree; Git stores its bytes only once."""

    repository_root: str
    commit: str
    tree: str
    origin_url: str | None
    relevant_paths: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        return {"schema": "mlops.source-identity.v1", **asdict(self)}


@dataclass(frozen=True)
class RuntimeIdentity:
    """The dependency/runtime identity required beside a source commit."""

    python: str
    implementation: str
    platform: str
    lockfile: str | None
    lockfile_sha256: str | None
    image_digest: str | None
    installed_packages: tuple[tuple[str, str], ...] = ()

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        if not self.installed_packages:
            record.pop("installed_packages")
        version = 2 if self.installed_packages else 1
        schema = f"mlops.runtime-identity.v{version}"
        return {"schema": schema, **record}


@dataclass(frozen=True)
class DataSnapshot:
    """Non-secret immutable identity for training input; never a mutable connection URL."""

    id: str
    fingerprint: str
    reference: str
    version: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.reference.strip():
            raise ProvenanceError("data snapshot id and reference must be non-blank")
        if not _is_sha256(self.fingerprint):
            raise ProvenanceError("data snapshot fingerprint must be a lowercase SHA-256")
        validate_data_reference(self.reference)

    def as_record(self) -> dict[str, object]:
        return {"schema": "mlops.data-snapshot.v1", **asdict(self)}


@dataclass(frozen=True)
class CheckpointCapability:
    """Truthful statement of what a saved checkpoint can recover."""

    kind: str
    checkpoint_path: str
    checkpoint_sha256: str
    parent_run_id: str | None = None
    missing_resume_state: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"inference_only", "resume_capable"}:
            raise ProvenanceError("checkpoint kind must be inference_only or resume_capable")
        if not self.checkpoint_path or not _is_sha256(self.checkpoint_sha256):
            raise ProvenanceError("checkpoint path and SHA-256 are required")
        if self.kind == "resume_capable" and self.missing_resume_state:
            raise ProvenanceError("a resume-capable checkpoint cannot report missing resume state")
        if self.kind == "inference_only" and not self.missing_resume_state:
            raise ProvenanceError(
                "an inference-only checkpoint must state why exact resume is unavailable"
            )

    def as_record(self) -> dict[str, object]:
        return {"schema": "mlops.checkpoint-capability.v1", **asdict(self)}


class _BundleWriter(Protocol):
    def write_json(self, relative_path: str, payload: Mapping[str, object]) -> pathlib.Path: ...

    def write_bytes(self, relative_path: str, payload: bytes) -> pathlib.Path: ...

    def copy_file(self, relative_path: str, source: str | pathlib.Path) -> pathlib.Path: ...


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise ProvenanceError(
            f"cannot identify source through Git: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _normalise_root(root: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(root).resolve()
    if not candidate.is_dir():
        raise ProvenanceError(f"repository root is not a directory: {candidate}")
    return candidate


def _is_relevant(path: str, relevant_paths: Sequence[str]) -> bool:
    return any(path == item or path.startswith(item.rstrip("/") + "/") for item in relevant_paths)


def capture_source_identity(
    root: str | pathlib.Path,
    *,
    relevant_paths: Sequence[str] = _DEFAULT_RELEVANT_PATHS,
    require_clean: bool = True,
) -> SourceIdentity:
    """Capture a committed source identity and reject relevant dirty changes.

    Raw datasets and generated reports are intentionally outside the default paths, so an
    untracked ``data/`` directory cannot make a source tree look different from its commit.
    """
    repository = _normalise_root(root)
    git_root = pathlib.Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if git_root != repository:
        raise ProvenanceError(f"repository root must be the Git root: {git_root}")
    paths = tuple(relevant_paths)
    if not paths or any(not item or pathlib.PurePosixPath(item).is_absolute() for item in paths):
        raise ProvenanceError("relevant paths must be non-empty relative paths")
    if require_clean:
        changed = _git(repository, "diff", "--name-only", "HEAD").splitlines()
        untracked = _git(repository, "ls-files", "--others", "--exclude-standard").splitlines()
        dirty = sorted({path for path in (*changed, *untracked) if _is_relevant(path, paths)})
        if dirty:
            raise ProvenanceError(
                "official runs require committed source/configuration changes; dirty paths="
                f"{dirty}"
            )
    origin: str | None
    result = subprocess.run(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    origin = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    return SourceIdentity(
        repository_root=str(repository),
        commit=_git(repository, "rev-parse", "HEAD"),
        tree=_git(repository, "rev-parse", "HEAD^{tree}"),
        origin_url=sanitize_repository_origin(origin),
        relevant_paths=paths,
    )


def verify_retained_ref(
    source: SourceIdentity, retained_ref: str,
    *,
    root: str | pathlib.Path | None = None,
) -> None:
    """Verify a local protected-run ref points at the captured commit.

    Hosting providers enforce protection and mirror replication; this deliberately does not
    pretend local Python can configure their IAM or retention policies.
    """
    if not retained_ref.startswith("refs/tags/autoforecast/run-"):
        raise ProvenanceError("retained ref must be refs/tags/autoforecast/run-<run-id>")
    repository = _normalise_root(source.repository_root if root is None else root)
    resolved = _git(repository, "rev-parse", f"{retained_ref}^{{commit}}")
    if resolved != source.commit:
        raise ProvenanceError(
            f"retained ref {retained_ref!r} points to {resolved}, not source commit {source.commit}"
        )


def capture_runtime_identity(
    root: str | pathlib.Path,
    *,
    lockfile: str | pathlib.Path | None = "uv.lock",
    image_digest: str | None = None,
) -> RuntimeIdentity:
    """Capture the lockfile/runtime now; image building remains an external CI concern."""
    repository = _normalise_root(root)
    if lockfile is None:
        from autoforecast.project import installed_dependencies
        return RuntimeIdentity(sys.version.split()[0], platform.python_implementation(),
                               platform.platform(), None, None, image_digest,
                               installed_dependencies())
    candidate = repository / lockfile
    if candidate.is_symlink() or not candidate.is_file():
        raise ProvenanceError(f"runtime lockfile must be a regular file: {candidate}")
    if image_digest is not None and not (
        image_digest.startswith("sha256:") and _is_sha256(image_digest.removeprefix("sha256:"))
    ):
        raise ProvenanceError(
            "OCI image digest must be sha256:<64 lowercase hexadecimal characters>"
        )
    return RuntimeIdentity(
        python=sys.version.split()[0],
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        lockfile=pathlib.PurePosixPath(lockfile).as_posix(),
        lockfile_sha256=_sha256(candidate),
        image_digest=image_digest,
    )


class SourceRecord(Protocol):
    """Source-checkout and installed-library records share serialization, not Git semantics."""

    def as_record(self) -> dict[str, object]: ...


def write_run_provenance(
    bundle: _BundleWriter,
    *,
    source: SourceRecord,
    runtime: RuntimeIdentity,
    data: DataSnapshot,
    configuration_files: Mapping[str, str | pathlib.Path],
    configuration_payloads: Mapping[str, tuple[str, bytes]] | None = None,
    lockfile_source: str | pathlib.Path | None,
    checkpoint: CheckpointCapability | None = None,
) -> None:
    """Write small, immutable provenance records before the bundle completes.

    The caller owns bundle finalization.  Configuration file names are explicit logical labels,
    so a consumer never has to infer whether a mutable current YAML was used.
    """
    payloads = configuration_payloads or {}
    if not configuration_files and not payloads:
        raise ProvenanceError("at least one resolved configuration file is required")
    if set(configuration_files) & set(payloads):
        raise ProvenanceError("configuration file and retained-payload labels must be disjoint")
    copied: dict[str, dict[str, str]] = {}
    for label, raw_source in sorted(configuration_files.items()):
        if not label or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in label):
            raise ProvenanceError("configuration labels must be lowercase letters, digits, _ or -")
        source_path = pathlib.Path(raw_source)
        if source_path.is_symlink() or not source_path.is_file():
            raise ProvenanceError(f"configuration source must be a regular file: {source_path}")
        suffix = source_path.suffix or ".txt"
        target = f"provenance/config/{label}{suffix}"
        before = _sha256(source_path)
        copied_path = bundle.copy_file(target, source_path)
        after = _sha256(source_path)
        if before != after or _sha256(copied_path) != before:
            raise ProvenanceError(
                f"configuration source changed while it was copied: {source_path}"
            )
        copied[label] = {"path": target, "sha256": before}
    for label, (suffix, payload) in sorted(payloads.items()):
        if not label or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in label):
            raise ProvenanceError("configuration labels must be lowercase letters, digits, _ or -")
        if (
            not suffix.startswith(".")
            or "/" in suffix
            or "\\" in suffix
            or type(payload) is not bytes
        ):
            raise ProvenanceError("retained configuration payload metadata is invalid")
        target = f"provenance/config/{label}{suffix}"
        copied_path = bundle.write_bytes(target, payload)
        digest = hashlib.sha256(payload).hexdigest()
        if _sha256(copied_path) != digest:
            raise ProvenanceError("retained configuration payload changed while it was published")
        copied[label] = {"path": target, "sha256": digest}
    if lockfile_source is None:
        if runtime.lockfile_sha256 is not None or not runtime.installed_packages:
            raise ProvenanceError("installed runtime requires its dependency inventory")
    else:
        lock = pathlib.Path(lockfile_source)
        if lock.is_symlink() or not lock.is_file():
            raise ProvenanceError(f"lockfile source must be a regular file: {lock}")
        lock_before = _sha256(lock)
        if runtime.lockfile_sha256 != lock_before:
            raise ProvenanceError("runtime lockfile changed after its identity was captured")
        copied_lock = bundle.copy_file("provenance/runtime/uv.lock", lock)
        if _sha256(lock) != lock_before or _sha256(copied_lock) != lock_before:
            raise ProvenanceError("runtime lockfile changed while it was copied")
    bundle.write_json("provenance/source.json", source.as_record())
    bundle.write_json("provenance/runtime.json", runtime.as_record())
    bundle.write_json("provenance/data.json", data.as_record())
    bundle.write_json("provenance/manifest.json", {
        "schema": "mlops.run-provenance.v1",
        "source": source.as_record(),
        "runtime": runtime.as_record(),
        "data": data.as_record(),
        "configuration": copied,
    })
    if checkpoint is not None:
        bundle.write_json("provenance/checkpoint.json", checkpoint.as_record())
