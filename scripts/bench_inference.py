#!/usr/bin/env python3
"""Measure inference cost by stage with reproducible input and model identities.

Report deterministic encoder, factor integration and sampled-path work separately.
Compare timing ratios only for computationally compatible sealed records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from autoforecast._exact_json import ExactJSONError, load_exact_json, loads_exact_json
from autoforecast._numeric import is_finite_real
from autoforecast.adapters import mft as ours
from forecast.models.mft.sample import (
    MAX_SUPPORT,
    SampleError,
    _support_sizes_negbin,
    support_size,
    support_sizes,
)

POINT_CELLS = 30_490
"""The real M5 point batch. Ratios measured at 2,000 cells do not transfer: the encoder/head
split moves with batch size, and the point forecast runs over the WHOLE catalogue."""

PATH_CELLS = 2_000
"""`paths_sample` -- paths are drawn only for the persisted sample, not the catalogue."""

DRAW_CHUNK = 25
"""Factor realisations per `apply_factor` call when timing the per-draw half.

E and h are now MEASURED DIRECTLY rather than fitted. Fitting `T = E + draws*h` from whole
forward passes was the only option while the two halves were one method; it stopped working once
they were separable, in both directions -- a (1,5,25) grid gave a negative slope because the
per-draw half was below timing noise, and a (1,50,200) grid gave a negative intercept because at
200 draws x 30,490 cells x 28 steps the call turns memory-bound and the cost stops being linear.
Timing `encode_base` and `apply_factor` separately needs no model of the cost at all.
"""

SAMPLER_FACTOR_FIXTURES = 50
"""Distinct deterministic factor realisations used to measure the value-dependent sampler.

One unseeded factor draw made the support, tier mix and projected path cost change with whatever
had consumed Torch's global RNG earlier. Fifty sequential paths from the production domain seeds
give a useful mean/P95 and support range; the full 1,000/1,000 route is also timed once. Every
fixture's factor path, count uniforms, parameters and output is content-hashed for replay.
"""

SAMPLER_FACTOR_SEED = 0

BENCHMARK_CONTRACT = 3
"""Version of the timed fixture and comparison contract.

Version 1 used random identity rows, zero static attributes, a smaller-than-production factor
topology, and timed an explicitly supplied factor path. Version 2 fixed the production topology,
stochastic seals and end-to-end path scope, but named a projection reconstructed from today's
components `legacy` and printed ratios against it as speedups. Version 3 labels that diagnostic as
synthetic and reserves before/after speedup language for compatible sealed records. Older records
remain historical evidence for exactly what they measured, but are not computationally comparable
with this fixture.
"""

MEASUREMENT_PROTOCOL = {
    "clock": "time.perf_counter wall seconds",
    "encode_base": {"warmups": 1, "reps": 3, "scope": "encode_base call"},
    "factor_apply_chunk": {
        "warmups": 1, "reps": 3,
        "scope": "factor draw plus apply_factor for DRAW_CHUNK paths"},
    "point_mc": {
        "warmups": 1, "reps": 2,
        "scope": "mean_over_factor with 1000 factor draws, including encode"},
    "point_quadrature": {
        "warmups": 1, "reps": 2,
        "scope": "mean_over_factor_quadrature, including encode and any GH/RQMC escalation"},
    "path_micro": {
        "warmups": 1, "reps": SAMPLER_FACTOR_FIXTURES,
        "scope": "sequential factor draw/apply and one count path; fingerprints outside timer"},
    "full_paths": {
        "warmups": 0, "reps": 1,
        "scope": "draw_paths_over_factor end to end, 1000 paths/1000 factors, output included"},
    "peak_rss": "whole process through all timing and fingerprint stages",
}


def _machine_hardware() -> dict:
    """Stable hardware identity needed before a wall-clock ratio is meaningful."""
    if sys.platform == "darwin":
        def sysctl(name: str) -> str:
            try:
                proc = subprocess.run(
                    ["sysctl", "-n", name], capture_output=True, text=True,
                    timeout=10, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"cannot establish hardware identity via {name}: {exc}") from exc
            value = proc.stdout.strip()
            if proc.returncode or not value:
                raise RuntimeError(
                    f"cannot establish hardware identity via {name}: "
                    f"exit={proc.returncode}, stderr={proc.stderr.strip()!r}")
            return value

        return {
            "cpu_model": sysctl("machdep.cpu.brand_string"),
            "physical_cores": int(sysctl("hw.physicalcpu")),
            "logical_cores": int(sysctl("hw.logicalcpu")),
            "memory_bytes": int(sysctl("hw.memsize")),
        }
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
            meminfo = pathlib.Path("/proc/meminfo").read_text()
        except OSError as exc:
            raise RuntimeError(f"cannot establish Linux hardware identity: {exc}") from exc
        blocks = []
        for raw_block in cpuinfo.split("\n\n"):
            if not raw_block.strip():
                continue
            block = {}
            for line in raw_block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    block[key.strip()] = value.strip()
            blocks.append(block)
        models = [block.get("model name", block.get("Hardware", "")).strip()
                  for block in blocks]
        cpu_model = next((value for value in models if value), "")
        physical_pairs = {
            (block.get("physical id", "0").strip(), block.get("core id", "").strip())
            for block in blocks if block.get("core id", "").strip()
        }
        memory_line = next((line for line in meminfo.splitlines()
                            if line.startswith("MemTotal:")), "")
        if not cpu_model or not memory_line:
            raise RuntimeError("Linux /proc does not contain CPU model and MemTotal identity")
        memory_kib = int(memory_line.split()[1])
        logical = os.cpu_count()
        if logical is None or logical < 1:
            raise RuntimeError(f"os.cpu_count returned invalid logical core count {logical!r}")
        return {
            "cpu_model": cpu_model,
            "physical_cores": len(physical_pairs) if physical_pairs else None,
            "logical_cores": logical,
            "memory_bytes": memory_kib * 1024,
        }
    raise RuntimeError(f"unsupported benchmark platform for hardware identity: {sys.platform}")


def _same_existing_path(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Whether two existing spellings name one inode, including case/Unicode aliases."""
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(
            f"cannot compare benchmark paths {left} and {right}: {exc}") from exc


def _require_output_separate_from_checkpoint(
    output_path: pathlib.Path, checkpoint_path: pathlib.Path,
) -> None:
    """Refuse lexical, symlink, hard-link, case, and Unicode aliases of the checkpoint."""
    if (output_path.resolve() == checkpoint_path.resolve()
            or _same_existing_path(output_path, checkpoint_path)):
        raise RuntimeError("benchmark JSON output must not overwrite the model checkpoint")


def _require_output_separate_from_baseline(
    output_path: pathlib.Path, baseline_path: pathlib.Path,
) -> None:
    """Refuse every filesystem spelling that would overwrite comparison evidence."""
    if (output_path.resolve() == baseline_path.resolve()
            or _same_existing_path(output_path, baseline_path)):
        raise RuntimeError("benchmark JSON output must not overwrite its comparison baseline")


def _absolute_git_directory(git_command, option: str) -> pathlib.Path:
    """Resolve one Git-owned directory, failing closed on ambiguous or stale output."""
    raw = git_command("rev-parse", "--path-format=absolute", option)
    value = os.fsdecode(raw)
    if value.endswith("\n"):
        value = value[:-1]
    if value.endswith("\r"):
        value = value[:-1]
    if not value or "\n" in value or "\r" in value:
        raise RuntimeError(f"git rev-parse returned an invalid {option} path: {value!r}")
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"git rev-parse returned a non-absolute {option} path: {value!r}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve Git administrative directory {path}: {exc}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"Git administrative path is not a directory: {resolved}")
    return resolved


def _path_is_at_or_below(path: pathlib.Path, directory: pathlib.Path) -> bool:
    """Include filesystem aliases of an existing ancestor, not only lexical descendants."""
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise RuntimeError(f"cannot resolve benchmark output path {path}: {exc}") from exc
    if resolved == directory or directory in resolved.parents:
        return True
    return any(_same_existing_path(ancestor, directory)
               for ancestor in (path, *path.parents))


def _require_output_outside_git_admin(output_path: pathlib.Path, git_command) -> None:
    """Refuse the worktree gitfile and both per-worktree and common Git state trees."""
    git_dir = _absolute_git_directory(git_command, "--git-dir")
    common_dir = _absolute_git_directory(git_command, "--git-common-dir")
    git_entry = ROOT / ".git"
    if not os.path.lexists(git_entry):
        raise RuntimeError(f"cannot establish worktree Git administrative entry: {git_entry}")
    try:
        resolved_output = output_path.resolve()
        resolved_entry = git_entry.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve worktree Git administrative entry: {exc}") from exc
    if (resolved_output == resolved_entry
            or _same_existing_path(output_path, git_entry)):
        raise RuntimeError(
            f"benchmark JSON output must not replace worktree Git administrative entry "
            f"{git_entry}")
    for directory in {git_dir, common_dir}:
        if _path_is_at_or_below(output_path, directory):
            raise RuntimeError(
                f"benchmark JSON output must be outside Git administrative directory "
                f"{directory}")


def identity(output_path: pathlib.Path | None = None) -> dict:
    """Everything needed to know whether two benchmark runs are comparable.

    A timing without this is a number with no provenance -- the exact failure this repository
    refuses everywhere else.
    """
    def _git(*args: str) -> bytes:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"cannot establish benchmark git identity: {exc}") from exc
        if proc.returncode:
            detail = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"cannot establish benchmark git identity: git {' '.join(args)} failed "
                f"with exit {proc.returncode}: {detail}")
        return proc.stdout

    git_sha = _git("rev-parse", "HEAD").decode().strip()
    if len(git_sha) != 40 or any(c not in "0123456789abcdef" for c in git_sha.lower()):
        raise RuntimeError(f"git rev-parse returned a non-SHA identity: {git_sha!r}")

    excluded = None
    if output_path is not None:
        _require_output_outside_git_admin(output_path, _git)
        resolved = output_path.resolve()
        if resolved.is_dir():
            raise RuntimeError(f"benchmark JSON output is a directory: {resolved}")
        try:
            excluded = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            pass

    tracked_index = _git("ls-files", "-s", "-z")
    tracked_paths = set()
    tracked_symlinks = []
    for entry in tracked_index.split(b"\0"):
        if not entry:
            continue
        metadata, separator, path = entry.partition(b"\t")
        if not separator:
            raise RuntimeError("cannot parse the benchmark git index")
        relative = path.decode(errors="surrogateescape")
        tracked_paths.add(relative)
        if metadata.startswith(b"120000 "):
            tracked_symlinks.append(relative)
    if excluded is not None and excluded in tracked_paths:
        raise RuntimeError(
            f"benchmark JSON output must not replace tracked repository file {excluded!r}")
    if output_path is not None:
        alias = next((
            relative for relative in sorted(tracked_paths)
            if _same_existing_path(resolved, ROOT / relative)
        ), None)
        if alias is not None:
            raise RuntimeError(
                "benchmark JSON output must not replace tracked repository file "
                f"{alias!r} through a filesystem alias")

    # `git diff` does not include untracked files. That let a new executable shadow a tracked
    # module while the benchmark still reported a clean identity. Seal both the binary tracked
    # diff and every non-ignored untracked file, with names and contents framed unambiguously.
    # The output record itself is excluded: including its pre-write bytes would create a seal
    # that becomes false as soon as this process writes the record.
    diff_args = ["diff", "--binary", "HEAD", "--", "."]
    if excluded is not None:
        diff_args.append(f":(top,literal,exclude){excluded}")
    tracked_diff = _git(*diff_args)
    raw_untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
    reported_untracked = sorted(
        p.decode(errors="surrogateescape") for p in raw_untracked.split(b"\0") if p)
    excluded_untracked = excluded
    if excluded is not None and excluded not in reported_untracked:
        aliases = [relative for relative in reported_untracked
                   if _same_existing_path(resolved, ROOT / relative)]
        if len(aliases) > 1:
            raise RuntimeError(
                "cannot identify benchmark output among multiple untracked filesystem aliases: "
                + ", ".join(repr(alias) for alias in aliases))
        if aliases:
            excluded_untracked = aliases[0]
    untracked = [relative for relative in reported_untracked
                 if relative != excluded_untracked]
    if tracked_symlinks:
        raise RuntimeError(
            "cannot seal benchmark identity with tracked symlink(s): "
            + ", ".join(tracked_symlinks))
    worktree = hashlib.sha256()
    worktree.update(b"tracked-diff\0")
    worktree.update(len(tracked_diff).to_bytes(8, "big"))
    worktree.update(tracked_diff)
    for relative in untracked:
        path = ROOT / relative
        name = os.fsencode(relative)
        if path.is_symlink():
            # Hashing only readlink(2)'s text does not bind the executable target bytes: the
            # target can change while the identity remains constant. There are no repository
            # symlinks in the benchmark contract, so fail closed instead of guessing which
            # external/ignored target tree must also be sealed.
            raise RuntimeError(
                f"cannot seal benchmark identity with untracked symlink {relative!r}")
        elif path.is_file():
            payload = b"file\0" + path.read_bytes()
        else:
            raise RuntimeError(
                f"cannot seal untracked benchmark input {relative!r}: not a file or symlink")
        worktree.update(b"untracked\0")
        worktree.update(len(name).to_bytes(8, "big"))
        worktree.update(name)
        worktree.update(len(payload).to_bytes(8, "big"))
        worktree.update(payload)
    dirty = bool(tracked_diff or untracked)
    return {
        "git_sha": git_sha,
        "git_dirty_sha256": worktree.hexdigest() if dirty else None,
        "git_tracked_diff_bytes": len(tracked_diff),
        "git_untracked_files": len(untracked),
        "git_identity_excluded_output": excluded_untracked,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "hardware": _machine_hardware(),
    }


def _file_sha256(path: pathlib.Path, *, label: str = "checkpoint") -> str:
    """Hash one artifact from bytes, refusing an unreadable/non-file input."""
    if not path.is_file():
        raise RuntimeError(f"benchmark {label} is not a readable file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"cannot hash benchmark {label} {path}: {exc}") from exc
    return digest.hexdigest()


def _require_checkpoint_unchanged(path: pathlib.Path, expected_sha256: str) -> None:
    current = _file_sha256(path)
    if current != expected_sha256:
        raise RuntimeError(
            f"benchmark checkpoint changed during the run: expected {expected_sha256}, "
            f"now {current}")


def _require_identity_unchanged(
    expected: dict, output_path: pathlib.Path | None,
) -> None:
    current = identity(output_path)
    if current != expected:
        raise RuntimeError(
            "repository execution identity changed during the benchmark; refusing to publish "
            f"mixed-code timing (start={expected!r}, end={current!r})")


def _whole_process_peak_rss_bytes() -> int:
    """Peak resident set for this process so far, normalized to bytes.

    CPython exposes bytes on macOS and KiB on Linux/other supported Unix platforms. This is one
    whole-process high-water mark, not a per-stage allocation measurement.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not is_finite_real(raw) or raw < 0:
        raise RuntimeError(f"resource.getrusage returned invalid ru_maxrss {raw!r}")
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(raw * multiplier)


def _publish_json(path: pathlib.Path, record: dict) -> None:
    """Publish finite JSON with durable old-or-new semantics and verified readback."""
    try:
        payload = json.dumps(record, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"benchmark record is not finite JSON: {exc}") from exc
    parent = path.parent.resolve()
    if not parent.is_dir():
        raise RuntimeError(f"benchmark output directory does not exist: {parent}")
    canonical = parent / path.name
    if canonical.is_symlink():
        raise RuntimeError(f"benchmark output must not be a symlink: {canonical}")
    if os.path.lexists(canonical) and not canonical.is_file():
        raise RuntimeError(f"benchmark output is not a regular file: {canonical}")

    def fsync_directory() -> None:
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    temporary: pathlib.Path | None = None
    backup: pathlib.Path | None = None
    replaced = False
    verified = False
    preserve_backup = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if canonical.exists():
            backup_fd, backup_name = tempfile.mkstemp(
                dir=parent, prefix=f".{path.name}.", suffix=".rollback")
            os.close(backup_fd)
            backup = pathlib.Path(backup_name)
            backup.unlink()
            os.link(canonical, backup)
            fsync_directory()
        os.replace(temporary, canonical)
        temporary = None
        replaced = True
        fsync_directory()
        try:
            readback = load_exact_json(canonical)
        except ExactJSONError as exc:
            raise RuntimeError(f"published benchmark record failed readback: {exc}") from exc
        if readback != json.loads(payload):
            raise RuntimeError("published benchmark record differs from the fsynced payload")
        verified = True
        if backup is not None:
            backup.unlink()
            backup = None
            fsync_directory()
    except BaseException as publication_error:
        # Once replace succeeds, the canonical name looks published. A later directory-fsync or
        # readback failure must restore the previous valid generation, not destroy it. When there
        # was no previous record, remove the unverified new name. The hard-link backup is made and
        # directory-fsynced before replace, so rollback does not depend on copying old bytes after
        # the failure.
        if replaced and not verified:
            try:
                if backup is None:
                    canonical.unlink(missing_ok=True)
                else:
                    os.replace(backup, canonical)
                    backup = None
                fsync_directory()
            except OSError as rollback_error:
                # Keep the backup if it still exists: losing both names would turn a publication
                # error into data loss. Name its location in the raised failure for recovery.
                preserve_backup = True
                failure = RuntimeError(
                    f"benchmark publication failed ({publication_error}); rollback also failed "
                    f"({rollback_error}); prior generation backup is {backup}")
                raise failure from publication_error
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if backup is not None and not preserve_backup:
            backup.unlink(missing_ok=True)


def _load_benchmark_baseline(path: pathlib.Path) -> dict[str, object]:
    """Load one comparison record without permitting ambiguous JSON evidence."""
    record, _digest = _load_benchmark_baseline_snapshot(path)
    return record


def _load_benchmark_baseline_snapshot(
    path: pathlib.Path,
) -> tuple[dict[str, object], str]:
    """Read one exact baseline generation and bind the bytes used for comparison."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"benchmark baseline {path} is unreadable: {exc}") from exc
    try:
        record = loads_exact_json(payload)
    except ExactJSONError as exc:
        raise RuntimeError(
            f"benchmark baseline {path} is unreadable or ambiguous: {exc}") from exc
    if type(record) is not dict:
        raise RuntimeError(f"benchmark baseline {path} is not a JSON object")
    return record, hashlib.sha256(payload).hexdigest()


def _require_baseline_unchanged(path: pathlib.Path, expected_sha256: str) -> None:
    current = _file_sha256(path, label="baseline")
    if current != expected_sha256:
        raise RuntimeError(
            f"benchmark baseline changed during the run: expected {expected_sha256}, "
            f"now {current}")


def _update_tensor_fingerprint(digest, name: str, value: torch.Tensor | None) -> None:
    """Append one unambiguously framed tensor to a content digest without copying its payload."""
    digest.update(name.encode())
    digest.update(b"\0")
    if value is None:
        digest.update(b"none\0")
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"cannot fingerprint non-tensor input {name}: {type(value).__name__}")
    stable = value.detach().cpu().contiguous()
    metadata = f"{stable.dtype}|{tuple(stable.shape)}".encode()
    # A memoryview streams the existing contiguous storage into hashlib. `tobytes()` made a
    # second 448 MB allocation just to seal a [1000, 2000, 28] path result, contaminating the
    # benchmark's own peak-RSS measurement with avoidable publication work.
    # A zero-element tensor (the factorless `[draws, 0, horizon, 0]` path sentinel, or an
    # attribute-free static block) has no bytes to stream and memoryview cannot cast an
    # empty strided view; its dtype and shape metadata above are the whole seal.
    payload = (memoryview(b"") if stable.numel() == 0
               else memoryview(stable.numpy()).cast("B"))
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _tensor_fingerprint(values: dict) -> str:
    """Canonical SHA-256 over named tensors/None values, including dtype and shape."""
    digest = hashlib.sha256()
    for name in sorted(values):
        _update_tensor_fingerprint(digest, name, values[name])
    return digest.hexdigest()


def _checkpoint_rows(record: dict, n: int, seed: int) -> tuple[dict, torch.Tensor, dict] | None:
    """Select aligned production identity/topology tensors saved in the checkpoint, if present."""
    run = record.get("run")
    tensors = None if not isinstance(run, dict) else run.get("tensors")
    required = ("attributes", "group_index", "store_index", "product_index")
    if not isinstance(tensors, dict) or not all(isinstance(tensors.get(k), torch.Tensor)
                                                for k in required):
        return None
    rows = int(tensors["attributes"].shape[0])
    if rows < 1 or any(value.shape[0] != rows for value in tensors.values()
                       if isinstance(value, torch.Tensor)):
        raise RuntimeError("checkpoint conditioning tensors are not aligned by series row")
    series_ids = run.get("series_ids")
    if n == rows:
        chosen = list(range(rows))
        selection = "all production rows in checkpoint order"
    elif isinstance(series_ids, list) and len(series_ids) == rows:
        chosen = ours._sample_series_indices(series_ids, min(n, rows), seed)  # noqa: SLF001
        selection = "production hash sample"
    else:
        chosen = list(range(min(n, rows)))
        selection = "production prefix (checkpoint has no aligned series_ids)"
    if n > rows:
        chosen = [i % rows for i in range(n)]
        selection = "production rows repeated to requested synthetic scale"
    index = torch.tensor(chosen, dtype=torch.long)
    selected = {
        name: (None if tensors.get(name) is None
               else tensors[name].index_select(0, index).clone())
        for name in (*required, "text")
    }
    selected["series_index"] = index.clone()
    return selected, index, {
        "static_source": "checkpoint run.tensors",
        "selection": selection,
        "checkpoint_rows": rows,
        "selected_row_indices_sha256": _tensor_fingerprint({"row": index}),
        "production_fields": [*required, "series_index", "text"],
    }


def benchmark_inputs(cfg: dict, n: int, record: dict, seed: int = 0) -> tuple[dict, dict]:
    """Build a replayable timing fixture with the checkpoint's real identity topology.

    Encoder windows and future covariates are origin-dependent and deliberately absent from the
    checkpoint. They remain seeded fixtures and are named as such in the record. In contrast,
    saved static attributes, series/product/store rows, text and factor-block equality are used
    exactly when available. This avoids timing the OOV embedding and a zero/static topology that
    production never serves.
    """
    if n < 1:
        raise ValueError(f"benchmark cells must be >= 1, got {n}")
    g = torch.Generator().manual_seed(seed)
    enc, horizon = cfg["encoder_days"], cfg["horizon"]
    saved = _checkpoint_rows(record, n, seed)
    if saved is None:
        attributes = torch.rand(n, cfg["n_attributes"], generator=g)
        series_index = torch.randint(0, cfg["n_series"], (n,), generator=g)
        product_index = torch.randint(0, max(cfg["n_products"], 1), (n,), generator=g)
        store_index = torch.randint(0, max(cfg["n_stores"], 1), (n,), generator=g)
        group_index = torch.arange(n, dtype=torch.long)
        text = (None if not cfg.get("text_dim")
                else torch.rand(n, cfg["text_dim"], generator=g))
        source = {
            "static_source": "seeded fallback (checkpoint has no run.tensors)",
            "selection": "synthetic",
            "checkpoint_rows": None,
            "production_fields": [],
        }
    else:
        static, _, source = saved
        attributes = static["attributes"]
        series_index = static["series_index"]
        product_index = static["product_index"]
        store_index = static["store_index"]
        group_index = static["group_index"]
        text = static["text"]

    history = torch.rand(n, enc, generator=g) * 3
    observed = torch.rand(n, enc, generator=g) > 0.08
    known_future = torch.rand(n, horizon, cfg["n_known"], generator=g)
    if cfg.get("price_index") is not None:
        known_future[..., cfg["price_index"]] = 0.5 + 1.5 * torch.rand(
            n, horizon, generator=g)
    if cfg.get("discount_available_index") is not None:
        known_future[..., cfg["discount_available_index"]] = 1.0
    for index in cfg.get("event_distance_index", ()):
        known_future[..., index] = torch.randint(
            -30, 31, (n, horizon), generator=g).to(known_future.dtype)
    for index in cfg.get("event_present_index", ()):
        known_future[..., index] = (
            torch.rand(n, horizon, generator=g) < 0.15).to(known_future.dtype)

    inputs = dict(
        history=history,
        observed=observed,
        censored=torch.zeros(n, enc),
        attributes=attributes,
        series_index=series_index,
        product_index=product_index,
        store_index=store_index,
        group_index=group_index,
        known_future=known_future,
        baseline_logit=-2.0 + 2.0 * torch.rand(n, generator=g),
        baseline_log_size=-0.5 + 1.5 * torch.rand(n, generator=g),
        commercial=(None if not cfg.get("n_commercial") else
                    torch.rand(n, cfg["n_commercial"], generator=g)),
        commercial_mask=(None if not cfg.get("n_commercial") else
                         torch.ones(n, cfg["n_commercial"], dtype=torch.bool)),
        text=text,
    )
    source.update({
        "dynamic_source": "seeded fixture; origin-dependent tensors are not in checkpoint",
        "synthetic_fields": [
            "history", "observed", "censored", "known_future", "baseline_logit",
            "baseline_log_size", "commercial", "commercial_mask",
        ],
        "seed": seed,
        "factor_blocks": int(torch.unique(group_index).numel()),
        "stores_present": int(torch.unique(store_index).numel()),
        "tensor_sha256": _tensor_fingerprint(inputs),
    })
    return inputs, source


def _time(fn, reps: int = 5) -> dict:
    """Median and P95 over repetitions, after a warm-up.

    Minimum alone hides variance; a single sample on a contended machine is noise. Both are
    reported so a later comparison can say which moved.
    """
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {"median_s": statistics.median(samples),
            "p95_s": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
            "min_s": samples[0], "n_reps": reps}


def _summarise(samples: list[float]) -> dict:
    """Same timing record as `_time`, for already measured distinct fixtures."""
    ordered = sorted(samples)
    return {"median_s": statistics.median(ordered),
            "mean_s": statistics.fmean(ordered),
            "stdev_s": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
            "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "min_s": ordered[0], "n_reps": len(ordered)}


def forward_stages(model, cfg: dict, record: dict, cells: int) -> dict:
    """Measure the two halves separately, and report what each fix bought."""
    inputs, fixture = benchmark_inputs(cfg, cells, record)
    base_inputs = {k: v for k, v in inputs.items() if k != "n_draws"}
    captured: dict[str, object] = {}
    with ours.evaluating(model), torch.no_grad():
        encode = _time(lambda: model.encode_base(**base_inputs), reps=3)
        cached = model.encode_base(**base_inputs)
        n_groups = int(torch.unique(cached.group_index).numel())
        factor_seed = 104729
        factor_path = model.draw_factor_paths(
            n_groups, cached.horizon, DRAW_CHUNK,
            generator=torch.Generator().manual_seed(factor_seed))
        factor_path_sha256 = _tensor_fingerprint({"factor_path": factor_path})
        del factor_path

        def apply_chunk_fn():
            # Draw inside the timed work: production does not receive a free factor path.
            generator = torch.Generator().manual_seed(factor_seed)
            captured["apply"] = model.apply_factor(
                cached, n_draws=DRAW_CHUNK, generator=generator)

        apply_chunk = _time(apply_chunk_fn, reps=3)
        apply_output = captured.pop("apply")
        assert isinstance(apply_output, tuple)
        apply_output_sha256 = _tensor_fingerprint({
            "p": apply_output[0], "mu": apply_output[1], "k": apply_output[2]})
        del apply_output
    timings = {"encode_base": encode, f"apply_factor_x{DRAW_CHUNK}": apply_chunk}
    base = encode["median_s"]
    per_draw = apply_chunk["median_s"] / DRAW_CHUNK
    degenerate = per_draw <= 0

    draws = 1000
    # A SYNTHETIC SHAPE PROJECTION, not the removed loop and not a historical speedup baseline.
    # Reconstructing `(draws/chunk) * (E + chunk*h)` from TODAY'S cached components illustrates
    # how repeated deterministic work amplifies cost, but both E and h changed with the split
    # (cache validation made E different; cached head projections made h cheaper). It therefore
    # cannot be labelled a measurement or even a proved bound on the old implementation. Only a
    # compatible sealed record supplied through `--baseline` licenses a before/after speedup.
    synthetic_removed_loop = (draws / DRAW_CHUNK) * (base + DRAW_CHUNK * per_draw)

    # What the current code actually does, timed rather than projected.
    measured = {}
    with ours.evaluating(model), torch.no_grad():
        def mc_fn():
            captured["mc"] = ours.mean_over_factor(
                model, inputs, factor_draws=draws, seed=0)

        def gh_fn():
            captured["gh"] = ours.mean_over_factor_quadrature(model, inputs)

        mc = _time(mc_fn, reps=2)
        gh = _time(gh_fn, reps=2)
    measured["mc_1000_draws"] = mc
    measured["gauss_hermite"] = gh
    return {
        "cells": cells,
        "input_fixture": fixture,
        "per_draw_timings": {str(k): v for k, v in timings.items()},
        "per_draw_factor_seed": factor_seed,
        "per_draw_factor_path_sha256": factor_path_sha256,
        "per_draw_output_sha256": apply_output_sha256,
        "deterministic_base_s": base,
        "per_factor_draw_s": per_draw,
        "base_over_draw_ratio": (base / per_draw) if per_draw > 0 else None,
        "per_draw_below_timing_noise": degenerate,
        "synthetic_removed_loop_projection_s": synthetic_removed_loop,
        "synthetic_removed_loop_projection_note": (
            "(draws/chunk) * (today's encode_base + chunk * today's cached per-draw cost); "
            "illustrative shape projection only, not a measurement or bound for removed code"),
        "measured": measured,
        "measured_output_sha256": {
            "mc_1000_draws": _tensor_fingerprint({"mean": captured["mc"]}),
            "gauss_hermite": _tensor_fingerprint({"mean": captured["gh"]}),
        },
        "quadrature_cost_note": (
            "Value-dependent: rank reduction selects 0-D/1-D/2-D Gauss-Hermite; successive "
            "rules may escalate through 15 nodes per dimension and unresolved cells use fixed "
            "replicated randomized QMC. Shape equality alone does not imply equal cost."),
        "ratio_vs_synthetic_removed_loop_projection": {
            "cached_mc_1000": (
                synthetic_removed_loop / mc["median_s"] if mc["median_s"] else None),
            "quadrature": (
                synthetic_removed_loop / gh["median_s"] if gh["median_s"] else None),
        },
    }


def _uniform_fingerprint(
    p: torch.Tensor | None, mu: torch.Tensor, n_draws: int, seed: int,
) -> str:
    """Hash exactly the count uniforms `predicted_paths` will consume from a fresh generator."""
    generator = torch.Generator().manual_seed(seed)
    return _uniform_fingerprint_from_generator(p, mu, n_draws, generator)


def _uniform_fingerprint_from_generator(
    p: torch.Tensor | None, mu: torch.Tensor, n_draws: int, generator: torch.Generator,
) -> str:
    """Hash the next uniforms without advancing the caller's sequential production stream."""
    replay = torch.Generator(device=generator.device)
    replay.set_state(generator.get_state())
    tail = (*mu.shape, n_draws)
    if p is None:
        values = {
            "size": torch.rand(tail, dtype=torch.float64, generator=replay),
        }
    else:
        values = {
            "occurrence": torch.rand(tail, dtype=p.dtype, generator=replay),
            "size": torch.rand(tail, dtype=torch.float64, generator=replay),
        }
    return _tensor_fingerprint(values)


def _full_stochastic_input_fingerprints(
    model, *, n_groups: int, n_cells: int, horizon: int,
    p_dtype: torch.dtype | None, factor_seed: int, count_seed: int,
    n_paths: int = 1000,
) -> dict:
    """Pre-generate and hash every innovation/uniform consumed by the 1000/1000 path route."""
    factor_digest = hashlib.sha256()
    count_digest = hashlib.sha256()
    count_generator = torch.Generator().manual_seed(count_seed)
    if model.factor is None:
        _update_tensor_fingerprint(factor_digest, "factorless", None)
        tail = (n_cells, horizon, n_paths)
        if p_dtype is not None:
            occurrence = torch.rand(tail, dtype=p_dtype, generator=count_generator)
            _update_tensor_fingerprint(count_digest, "occurrence", occurrence)
        size = torch.rand(tail, dtype=torch.float64, generator=count_generator)
        _update_tensor_fingerprint(count_digest, "size", size)
        factor_shape = None
        count_shape = list(tail)
    else:
        device = next(model.parameters()).device
        factor_generator = torch.Generator(device=device).manual_seed(factor_seed)
        factor_shape = [1, n_groups, horizon, model.factor.rank]
        count_shape = [n_cells, horizon, 1]
        for draw in range(n_paths):
            innovations = torch.randn(
                factor_shape, device=device, dtype=model.factor.theta.dtype,
                generator=factor_generator)
            _update_tensor_fingerprint(
                factor_digest, f"factor_innovations_{draw:04d}", innovations)
            if p_dtype is not None:
                occurrence = torch.rand(count_shape, dtype=p_dtype, generator=count_generator)
                _update_tensor_fingerprint(
                    count_digest, f"occurrence_uniforms_{draw:04d}", occurrence)
            size = torch.rand(count_shape, dtype=torch.float64, generator=count_generator)
            _update_tensor_fingerprint(count_digest, f"size_uniforms_{draw:04d}", size)
    return {
        "n_paths": n_paths,
        "factor_innovations_sha256": factor_digest.hexdigest(),
        "count_uniforms_sha256": count_digest.hexdigest(),
        "factor_innovation_shape_per_draw": factor_shape,
        "count_uniform_shape_per_draw": count_shape,
        "factor_seed": factor_seed,
        "count_seed": count_seed,
    }


def sampler_stage(model, cfg: dict, record: dict) -> dict:
    """The path sampler, and the batch-maximum support problem behind its cost.

    `support_size` returns ONE bound for the whole batch, proved so the omitted zero-truncated
    tail is <= 1e-12. Because it is shared, the batch's LARGEST mu sizes the CDF table every cell
    pays for -- and at one factor per path that table is rebuilt for a single draw. This stage
    reports both the time and the support the parameters actually demanded, plus a sensitivity
    sweep showing where the bound stops being affordable and where it refuses outright.
    """
    inputs, fixture = benchmark_inputs(cfg, PATH_CELLS, record)
    base_inputs = {k: v for k, v in inputs.items() if k != "n_draws"}
    with ours.evaluating(model), torch.no_grad():
        encode = _time(lambda: model.encode_base(**base_inputs), reps=3)
        base = model.encode_base(**base_inputs)
        n_groups = int(torch.unique(base.group_index).numel())
        # Warm both eager routes once without advancing either recorded generator.
        warm_path = model.draw_factor_paths(
            n_groups, base.horizon, 1,
            generator=torch.Generator().manual_seed(SAMPLER_FACTOR_SEED + 10_000))
        warm_p, warm_mu, warm_k = model.apply_factor(
            base, n_draws=1, factor_path=warm_path)
        ours.predicted_paths(
            None if warm_p is None else warm_p[0].cpu(), warm_mu[0].cpu(), warm_k[0].cpu(), 1,
            generator=torch.Generator().manual_seed(SAMPLER_FACTOR_SEED))

        factor_samples, sampling_samples, paired_samples, parameters = [], [], [], []
        seals = []
        factor_domain_seed = ours._domain_seed(  # noqa: SLF001
            SAMPLER_FACTOR_SEED, "factor", ours.RNG_DOMAIN_SCHEME)
        count_domain_seed = ours._domain_seed(  # noqa: SLF001
            SAMPLER_FACTOR_SEED, "count", ours.RNG_DOMAIN_SCHEME)
        factor_generator = torch.Generator().manual_seed(factor_domain_seed)
        count_generator = torch.Generator().manual_seed(count_domain_seed)
        for index in range(SAMPLER_FACTOR_FIXTURES):
            factor_state = factor_generator.get_state().clone()
            count_state = count_generator.get_state().clone()
            factor_state_sha256 = _tensor_fingerprint({"state": factor_state})
            count_state_sha256 = _tensor_fingerprint({"state": count_state})
            paired_start = time.perf_counter()
            factor_start = paired_start
            factor_path = model.draw_factor_paths(
                n_groups, base.horizon, 1, generator=factor_generator)
            p, mu, k = model.apply_factor(base, n_draws=1, factor_path=factor_path)
            factor_end = time.perf_counter()
            factor_samples.append(factor_end - factor_start)
            p0 = None if p is None else p[0].cpu()
            mu0, k0 = mu[0].cpu(), k[0].cpu()
            paths = ours.predicted_paths(
                p0, mu0, k0, 1, generator=count_generator)
            sample_end = time.perf_counter()
            sampling_samples.append(sample_end - factor_end)
            paired_samples.append(sample_end - paired_start)
            parameters.append((p0, mu0, k0))
            count_replay = torch.Generator(device=count_generator.device)
            count_replay.set_state(count_state)
            uniforms_sha256 = _uniform_fingerprint_from_generator(
                p0, mu0, 1, count_replay)
            seals.append({
                "fixture": index,
                "factor_generator_state_sha256": factor_state_sha256,
                "count_generator_state_sha256": count_state_sha256,
                "factor_path_sha256": _tensor_fingerprint({"factor_path": factor_path}),
                "count_uniforms_sha256": uniforms_sha256,
                "parameters_sha256": _tensor_fingerprint({"p": p0, "mu": mu0, "k": k0}),
                "paths_sha256": _tensor_fingerprint({"paths": paths}),
            })
    full_stochastic_inputs = _full_stochastic_input_fingerprints(
        model, n_groups=n_groups, n_cells=PATH_CELLS, horizon=base.horizon,
        p_dtype=None if parameters[0][0] is None else parameters[0][0].dtype,
        factor_seed=factor_domain_seed, count_seed=count_domain_seed)
    # Time the production route itself once. Unlike the micro projection, this includes encoder
    # reuse, all 1,000 sequential factor/count draws, output preallocation and in-place filling.
    full_start = time.perf_counter()
    full_paths = ours.draw_paths_over_factor(
        model, inputs, 1000, factor_draws=1000, seed=SAMPLER_FACTOR_SEED,
        rng_domain_scheme=ours.RNG_DOMAIN_SCHEME)
    full_elapsed = time.perf_counter() - full_start
    full_output_sha256 = _tensor_fingerprint({"paths": full_paths})
    full_shape = list(full_paths.shape)
    del full_paths
    factor_head = _summarise(factor_samples)
    sampling = _summarise(sampling_samples)
    paired = _summarise(paired_samples)

    horizon = cfg["horizon"]
    supports = []
    for fixture_p, fixture_mu, fixture_k in parameters:
        try:
            supports.append(
                support_size(fixture_mu, fixture_k) if fixture_p is not None
                else int(_support_sizes_negbin(fixture_mu, fixture_k).max()))
        except SampleError:
            supports.append(None)
    support = max((value for value in supports if value is not None), default=None)
    mu_min = min(float(fixture_mu.min()) for _, fixture_mu, _ in parameters)
    mu_max = max(float(fixture_mu.max()) for _, fixture_mu, _ in parameters)
    k_min = min(float(fixture_k.min()) for _, _, fixture_k in parameters)
    k_max = max(float(fixture_k.max()) for _, _, fixture_k in parameters)

    sweep = []
    # Every background value is strictly below 2, so `probe.max()` is exactly the declared peak.
    # Dispersion is fixed at k=1.5: support is a function of both mu and k.
    base_mu = torch.rand(PATH_CELLS, generator=torch.Generator().manual_seed(1)) * 1.9 + 0.1
    ks = torch.full((PATH_CELLS,), 1.5)
    hurdle = cfg.get("head_kind") == "hurdle_negbin"
    for peak in (2.0, 20.0, 100.0, 300.0):
        probe = base_mu.clone()
        probe[0] = peak
        actual_peak = float(probe.max())
        if actual_peak != peak:
            raise RuntimeError(
                f"sampler sweep fixture promised max mu {peak} but constructed {actual_peak}")
        try:
            shared = (support_size(probe, ks) if hurdle
                      else int(_support_sizes_negbin(probe, ks).max()))
        except SampleError:
            shared = None
        try:
            per_cell = support_sizes(probe, ks) if hurdle else _support_sizes_negbin(probe, ks)
            median = int(per_cell.median())
            worst = int(per_cell.max())
            refused = False
        except SampleError:
            median = worst = None
            refused = True
        sweep.append({
            "max_mu": actual_peak, "shared_support": shared,
            "shared_table_gb": (None if shared is None
                                # PMF and CDF, both float64. The old report priced one float32
                                # table and understated this by 4x.
                                else PATH_CELLS * horizon
                                * (shared if hurdle else shared + 1) * 2 * 8 / 1e9),
            "per_cell_median_support": median, "per_cell_max_support": worst,
            "refused_per_cell": refused,
            "shared_refused": shared is None})

    return {
        "cells": PATH_CELLS,
        "input_fixture": fixture,
        "encode_base_once": encode,
        "factor_and_head_one_draw": factor_head,
        "sample_one_path": sampling,
        "paired_one_path": paired,
        "sampler_over_factor_head_ratio": (
            sampling["median_s"] / factor_head["median_s"]
            if factor_head["median_s"] else None),
        "support_required": support,
        "support_by_factor_fixture": supports,
        "mu_range": [mu_min, mu_max],
        "k_range": [k_min, k_max],
        "factor_fixture_seed": SAMPLER_FACTOR_SEED,
        "factor_domain_seed": factor_domain_seed,
        "count_domain_seed": count_domain_seed,
        "factor_fixture_draws": SAMPLER_FACTOR_FIXTURES,
        "fixture_fingerprints": seals,
        "full_stochastic_input_fingerprints": full_stochastic_inputs,
        "max_support_ceiling": MAX_SUPPORT,
        "batch_max_mu_sweep": sweep,
        "sampler_head_kind": cfg.get("head_kind"),
        # Production encodes ONCE, then applies one distinct factor/head and one count draw for
        # each emitted systemic path. Multiplying the full forward by 1,000 was the very caching
        # defect this benchmark exists to prevent, reintroduced in its own projection.
        "projected_1000_draws_s": (
            encode["median_s"]
            + 1000 * paired["mean_s"]),
        "projection_method": (
            "encode median once + 1000 * arithmetic mean of 50 sequential paired "
            "factor-draw/head/count iterations. This micro projection omits production output "
            "preallocation/filling; use measured_1000_paths for the end-to-end cost."),
        "measured_1000_paths": {
            "elapsed_s": full_elapsed,
            "n_paths": 1000,
            "factor_draws": 1000,
            "shape": full_shape,
            "output_sha256": full_output_sha256,
            "n_reps": 1,
            "scope": (
                "draw_paths_over_factor end to end: encode, sequential factor/head/count work, "
                "output preallocation and filling"),
        },
    }


def _model_config_record(cfg: dict) -> dict:
    """JSON-shaped constructor config so tuples compare equally after a sealed-record reload."""
    return json.loads(json.dumps({k: v for k, v in cfg.items()
                                  if not isinstance(v, torch.Tensor)}))


def _comparison_mismatches(earlier: dict, current: dict) -> list[str]:
    """Name every property that makes a timing ratio computationally invalid."""
    mismatches = []

    def exact_equal(old, new) -> bool:
        if type(old) is not type(new):
            return False
        if type(new) is dict:
            return (old.keys() == new.keys()
                    and all(exact_equal(old[key], value) for key, value in new.items()))
        if type(new) in (list, tuple):
            return (len(old) == len(new)
                    and all(exact_equal(left, right)
                            for left, right in zip(old, new, strict=True)))
        return old == new

    def compare(label: str, old, new) -> None:
        if not exact_equal(old, new):
            mismatches.append(f"{label}: baseline={old!r}, current={new!r}")

    compare("benchmark_contract", earlier.get("benchmark_contract"),
            current.get("benchmark_contract"))
    compare("checkpoint_sha256", earlier.get("checkpoint_sha256"),
            current.get("checkpoint_sha256"))
    compare("model_config", earlier.get("model_config"), current.get("model_config"))
    compare("benchmark_config", earlier.get("benchmark_config"),
            current.get("benchmark_config"))
    for loop in ("point_loop", "path_loop"):
        compare(f"{loop}.cells", earlier.get(loop, {}).get("cells"),
                current.get(loop, {}).get("cells"))
        compare(f"{loop}.input_fixture.tensor_sha256",
                earlier.get(loop, {}).get("input_fixture", {}).get("tensor_sha256"),
                current.get(loop, {}).get("input_fixture", {}).get("tensor_sha256"))
    compare("point_loop.per_draw_factor_path_sha256",
            earlier.get("point_loop", {}).get("per_draw_factor_path_sha256"),
            current.get("point_loop", {}).get("per_draw_factor_path_sha256"))

    def stochastic_inputs(record: dict) -> list[tuple[object, object]]:
        return [
            (row.get("factor_path_sha256"), row.get("count_uniforms_sha256"))
            for row in record.get("path_loop", {}).get("fixture_fingerprints", [])
        ]

    compare("path_loop.factor_paths_and_count_uniforms",
            stochastic_inputs(earlier), stochastic_inputs(current))
    compare("path_loop.full_stochastic_input_fingerprints",
            earlier.get("path_loop", {}).get("full_stochastic_input_fingerprints"),
            current.get("path_loop", {}).get("full_stochastic_input_fingerprints"))
    compare("point_loop.per_draw_output_sha256",
            earlier.get("point_loop", {}).get("per_draw_output_sha256"),
            current.get("point_loop", {}).get("per_draw_output_sha256"))
    compare("point_loop.measured_output_sha256",
            earlier.get("point_loop", {}).get("measured_output_sha256"),
            current.get("point_loop", {}).get("measured_output_sha256"))

    def stochastic_outputs(record: dict) -> list[tuple[object, object]]:
        return [
            (row.get("parameters_sha256"), row.get("paths_sha256"))
            for row in record.get("path_loop", {}).get("fixture_fingerprints", [])
        ]

    compare("path_loop.fixture_parameters_and_paths",
            stochastic_outputs(earlier), stochastic_outputs(current))
    compare("path_loop.measured_1000_paths.output_sha256",
            earlier.get("path_loop", {}).get("measured_1000_paths", {}).get("output_sha256"),
            current.get("path_loop", {}).get("measured_1000_paths", {}).get("output_sha256"))

    def observed_timing_protocol(record: dict) -> dict:
        point = record.get("point_loop", {})
        path = record.get("path_loop", {})
        return {
            "point_per_draw_reps": {
                name: value.get("n_reps")
                for name, value in point.get("per_draw_timings", {}).items()
            },
            "point_measured_reps": {
                name: value.get("n_reps")
                for name, value in point.get("measured", {}).items()
            },
            "path_reps": {
                name: path.get(name, {}).get("n_reps")
                for name in ("encode_base_once", "factor_and_head_one_draw",
                             "sample_one_path", "paired_one_path")
            },
            "full_path_reps": path.get("measured_1000_paths", {}).get("n_reps"),
            "full_path_scope": path.get("measured_1000_paths", {}).get("scope"),
            "projection_method": path.get("projection_method"),
            "resource_scope": record.get("resources", {}).get("scope"),
        }

    compare("observed_timing_protocol",
            observed_timing_protocol(earlier), observed_timing_protocol(current))
    for field in ("threads", "interop_threads", "platform", "torch", "python"):
        compare(f"identity.{field}", earlier.get("identity", {}).get(field),
                current.get("identity", {}).get(field))
    compare("identity.hardware", earlier.get("identity", {}).get("hardware"),
            current.get("identity", {}).get("hardware"))
    return mismatches


def _same_stage_comparisons(earlier: dict, current: dict) -> list[tuple[str, float, float]]:
    """Comparable timings; never cross MC with GH or projections with measurements."""
    pairs = []
    for key, label in (("mc_1000_draws", "point MC(1,000)"),
                       ("gauss_hermite", "point Gauss-Hermite")):
        old = earlier.get("point_loop", {}).get("measured", {}).get(key, {}).get("median_s")
        new = current.get("point_loop", {}).get("measured", {}).get(key, {}).get("median_s")
        pairs.append((label, old, new))
    pairs.append((
        "paths 1,000/1,000",
        earlier.get("path_loop", {}).get("measured_1000_paths", {}).get("elapsed_s"),
        current.get("path_loop", {}).get("measured_1000_paths", {}).get("elapsed_s"),
    ))
    invalid = [(label, old, new) for label, old, new in pairs
               if (not is_finite_real(old) or old <= 0
                   or not is_finite_real(new) or new <= 0)]
    if invalid:
        raise RuntimeError(f"compatible benchmark has invalid same-stage timing(s): {invalid}")
    return [(label, float(old), float(new)) for label, old, new in pairs]


def _speed_change_text(was: float, now: float) -> str:
    """Name the direction of a same-stage timing ratio; a regression is not a speedup."""
    ratio = was / now
    if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=5e-3):
        return "unchanged"
    if ratio > 1.0:
        return f"{ratio:.2f}x speedup"
    return f"{now / was:.2f}x slowdown"


def _comparison_evidence(
    baseline_path: pathlib.Path,
    baseline_sha256: str,
    comparisons: list[tuple[str, float, float]],
) -> dict[str, object]:
    """Persist the exact baseline identity and every same-stage timing comparison."""
    return {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": baseline_sha256,
        "same_stage": [
            {
                "stage": label,
                "baseline_seconds": was,
                "current_seconds": now,
                "baseline_over_current_ratio": was / now,
                "direction": _speed_change_text(was, now),
            }
            for label, was, now in comparisons
        ],
        "scope": "only computationally compatible, like-for-like measured stages",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--json", default=None, help="write the sealed record here")
    ap.add_argument("--baseline", default=None,
                    help="an earlier sealed record to compare against; the output preserves "
                         "whether its old-loop value was measured or projected")
    ap.add_argument("--point-cells", type=int, default=POINT_CELLS)
    args = ap.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint)
    output_path = None if args.json is None else pathlib.Path(args.json)
    baseline_path = None if args.baseline is None else pathlib.Path(args.baseline)
    if output_path is not None:
        _require_output_separate_from_checkpoint(output_path, checkpoint_path)
        if baseline_path is not None:
            _require_output_separate_from_baseline(output_path, baseline_path)
    earlier = None
    baseline_sha256 = None
    if baseline_path is not None:
        earlier, baseline_sha256 = _load_benchmark_baseline_snapshot(baseline_path)
    start_identity = identity(output_path)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    model, record = ours.load_checkpoint(checkpoint_path)
    _require_checkpoint_unchanged(checkpoint_path, checkpoint_sha256)
    _require_identity_unchanged(start_identity, output_path)
    cfg = record["model_config"]
    sealed = {
        "benchmark_contract": BENCHMARK_CONTRACT,
        "identity": start_identity,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": _model_config_record(cfg),
        "benchmark_config": {
            "benchmark_source_sha256": _file_sha256(pathlib.Path(__file__)),
            "measurement_protocol": MEASUREMENT_PROTOCOL,
            "point_cells": args.point_cells,
            "path_cells": PATH_CELLS,
            "draw_chunk": DRAW_CHUNK,
            "point_mc_draws": 1000,
            "point_mc_seed": 0,
            "path_factor_fixtures": SAMPLER_FACTOR_FIXTURES,
            "full_paths": 1000,
            "full_path_factor_draws": 1000,
            "sampler_factor_seed": SAMPLER_FACTOR_SEED,
            "rng_domain_scheme": ours.RNG_DOMAIN_SCHEME,
        },
        "point_loop": forward_stages(model, cfg, record, args.point_cells),
        "path_loop": sampler_stage(model, cfg, record),
    }
    _require_checkpoint_unchanged(checkpoint_path, checkpoint_sha256)
    _require_identity_unchanged(start_identity, output_path)
    sealed["resources"] = {
        "whole_process_peak_rss_bytes": _whole_process_peak_rss_bytes(),
        "scope": (
            "resource.getrusage high-water mark for the whole benchmark process through all "
            "timed stages; not a per-stage allocation measurement"),
    }

    comparisons = None
    if earlier is not None:
        assert baseline_path is not None and baseline_sha256 is not None
        _require_baseline_unchanged(baseline_path, baseline_sha256)
        mismatches = _comparison_mismatches(earlier, sealed)
        if mismatches:
            formatted = "\n  - ".join(mismatches)
            raise RuntimeError(
                "refusing a benchmark speedup from incompatible records:\n  - " + formatted)
        comparisons = _same_stage_comparisons(earlier, sealed)
        sealed["comparison"] = _comparison_evidence(
            baseline_path, baseline_sha256, comparisons)

    point = sealed["point_loop"]
    synthetic_ratio = point["ratio_vs_synthetic_removed_loop_projection"]
    print(f"\n{'=' * 78}")
    print(f"POINT LOOP  ({point['cells']:,} cells)")
    print(f"  deterministic base E : {point['deterministic_base_s']:.4f}s")
    if point["per_draw_below_timing_noise"]:
        print("  per factor draw   h  : below this machine's timing noise "
              "(the cached base now dominates)")
    else:
        print(f"  per factor draw   h  : {point['per_factor_draw_s']:.5f}s   (E/h = "
              f"{point['base_over_draw_ratio']:.1f})")
    print(f"  synthetic removed-loop projection: "
          f"{point['synthetic_removed_loop_projection_s']:.1f}s")
    print("    (today's component timings in the removed batching shape; not an old measurement "
          "or bound)")
    print(f"  measured MC(1,000)   : {point['measured']['mc_1000_draws']['median_s']:.1f}s "
          f"(ratio {synthetic_ratio['cached_mc_1000']:.1f}x vs that synthetic projection)")
    print(f"  measured quadrature  : {point['measured']['gauss_hermite']['median_s']:.1f}s "
          f"(ratio {synthetic_ratio['quadrature']:.1f}x vs that synthetic projection)"
          "   <- the default")
    path = sealed["path_loop"]
    print(f"\nPATH LOOP  ({path['cells']:,} cells)")
    print(f"  deterministic base   : {path['encode_base_once']['median_s']:.4f}s (once)")
    print(f"  factor + head, 1 draw: {path['factor_and_head_one_draw']['median_s']:.4f}s")
    print(f"  sample one path      : {path['sample_one_path']['median_s']:.4f}s  "
          f"({path['sampler_over_factor_head_ratio']:.1f}x factor + head)")
    print(f"  support required     : {path['support_required']} "
          f"(mu {path['mu_range'][0]:.3f}..{path['mu_range'][1]:.2f})")
    print("  batch-max-mu sweep at fixed k=1.5 -- SHARED vs PER-CELL bounds now used:")
    for row in path["batch_max_mu_sweep"]:
        shared = ("REFUSED" if row["shared_refused"]
                  else f"{row['shared_support']:>6,} ({row['shared_table_gb']:.2f} GB shared)")
        per = ("REFUSED" if row["refused_per_cell"]
               else f"median {row['per_cell_median_support']:>5,}, "
                    f"worst {row['per_cell_max_support']:>6,}")
        print(f"    max mu {row['max_mu']:>6.0f}, k 1.5 -> shared {shared:<28} per-cell {per}")
    print("    (a cell past MAX_SUPPORT still REFUSES under either rule -- capping would put a")
    print("     spike at the ceiling in every quantile. What per-cell bounds changed is that the")
    print("     refusal now NAMES the offending cells, and that a merely busy cell no longer")
    print("     resizes its neighbours' tables: note the median column against the shared one.)"
          )
    print(f"  projected 1,000 draws: {path['projected_1000_draws_s'] / 60:.1f} min")
    measured_paths = path["measured_1000_paths"]
    print(f"  measured 1,000/1,000 : {measured_paths['elapsed_s'] / 60:.1f} min "
          "(production draw_paths_over_factor route, one run)")
    print(f"  whole-process peak RSS: "
          f"{sealed['resources']['whole_process_peak_rss_bytes'] / 1e9:.2f} GB")
    if earlier is not None:
        print(f"\nAGAINST {args.baseline}")
        assert comparisons is not None
        for label, was, now_s in comparisons:
            print(f"  {label}: {was:.1f}s -> {now_s:.1f}s  "
                  f"({_speed_change_text(was, now_s)})")
        print(f"  baseline identity: {earlier['identity'].get('git_sha', '?')[:12]} "
              f"on {earlier['identity'].get('platform', '?')}")
    print("=" * 78)
    print("NOT a gate result. A cost measurement, whose only claim is what it timed.")

    if output_path is not None:
        # Recheck immediately before publication. A pre-run identity is insufficient if a timed
        # stage (or another process) edits source or replaces the checkpoint while this runs.
        _require_checkpoint_unchanged(checkpoint_path, checkpoint_sha256)
        _require_identity_unchanged(start_identity, output_path)
        _require_output_separate_from_checkpoint(output_path, checkpoint_path)
        if baseline_path is not None:
            assert baseline_sha256 is not None
            _require_output_separate_from_baseline(output_path, baseline_path)
            _require_baseline_unchanged(baseline_path, baseline_sha256)
        _publish_json(output_path, sealed)
        print(f"sealed record: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
