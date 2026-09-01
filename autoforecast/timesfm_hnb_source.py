"""Pinned TimesFM 3 inference and content-addressed conditional feature caching."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Protocol

import numpy as np

SOURCE_REVISION = "331c6d33cb1ac2611de3056d0ac7164aab6301eb"
MODEL_ID = "google/timesfm-3.0-pytorch"
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
QUANTILES = tuple(i / 10 for i in range(1, 10))


def json_bytes(value: object) -> bytes:
    """Canonical finite JSON used by requests, manifests and cache envelopes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class SourceRequest:
    """One aligned ordered group; True denotes unavailable, including masked placeholders."""

    target: np.ndarray
    target_mask: np.ndarray
    horizon: int
    schema: tuple[str, ...]
    past_only: np.ndarray | None = None
    past_only_mask: np.ndarray | None = None
    past_future: np.ndarray | None = None
    past_future_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("source horizon must be positive")
        if self.target.ndim != 2 or not all(self.target.shape):
            raise ValueError("source target must be [targets, context]")
        if self.target.shape[1] > 15360:
            raise ValueError("TimesFM-HNB context exceeds the pinned provider policy")
        channels = 0
        for name, length in (
            ("target", self.target.shape[1]),
            ("past_only", self.target.shape[1]),
            ("past_future", self.target.shape[1] + self.horizon),
        ):
            values = getattr(self, name)
            mask = getattr(self, f"{name}_mask")
            if values is None:
                if mask is not None:
                    raise ValueError("an absent source tensor cannot have a mask")
                continue
            if (
                values.ndim != 2
                or values.shape[1] != length
                or not values.shape[0]
                or mask is None
                or mask.dtype != np.bool_
                or mask.shape != values.shape
            ):
                raise ValueError(f"invalid source {name} values/mask shape")
            if not np.isfinite(values[~mask]).all():
                raise ValueError("observed source values must be finite")
            clean = np.where(mask, 0.0, values).astype(np.float32)
            if not np.isfinite(clean).all():
                raise ValueError("source values exceed float32")
            clean.setflags(write=False)
            immutable_mask = mask.copy()
            immutable_mask.setflags(write=False)
            object.__setattr__(self, name, clean)
            object.__setattr__(self, f"{name}_mask", immutable_mask)
            channels += len(values)
        if len(self.schema) != channels or len(set(self.schema)) != channels:
            raise ValueError("ordered source schema must name every unique target/covariate")
        if self.target_mask.all(axis=1).any():
            raise ValueError("each requested target needs some observed history")

    def key(self, identity: dict[str, object]) -> str:
        digest = hashlib.sha256(
            json_bytes(
                {
                    "schema": "autoforecast.timesfm-v7-source-request.v1",
                    "source": identity,
                    "channels": self.schema,
                    "horizon": self.horizon,
                }
            )
        )
        for name in (
            "target",
            "target_mask",
            "past_only",
            "past_only_mask",
            "past_future",
            "past_future_mask",
        ):
            value = getattr(self, name)
            digest.update(json_bytes([name, None if value is None else list(value.shape)]))
            if value is not None:
                digest.update(value.tobytes(order="C"))
        return digest.hexdigest()


class Source(Protocol):
    """The single injectable seam; training and serialization remain real in functional tests."""

    identity: dict[str, object]

    def predict(self, request: SourceRequest) -> np.ndarray: ...


class TimesFM3Source:
    """Load the official pinned full network; never transplant or train source layers."""

    def __init__(self, device: str) -> None:
        import timesfm3
        import torch
        from huggingface_hub import snapshot_download
        from timesfm3.timesfm3_forecaster import TimesFM3Forecaster

        distribution = importlib.metadata.distribution("timesfm")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if (
            distribution.version != "3.0.0"
            or direct.get("vcs_info", {}).get("commit_id") != SOURCE_REVISION
        ):
            raise ValueError(
                "TimesFM-HNB needs the pinned timesfm3 optional runtime in its own environment"
            )
        folder = pathlib.Path(
            snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=["config.json", "model.safetensors"],
            )
        )
        self.provider = TimesFM3Forecaster.from_pretrained(str(folder), device=device)
        self.model = self.provider.model.eval().requires_grad_(False)
        if tuple(self.model.quantiles) != QUANTILES:
            raise ValueError(
                "TimesFM-HNB source quantiles differ from the pinned nine-decile contract"
            )
        source_root = pathlib.Path(timesfm3.__file__).parent
        self.identity = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION,
            "source_files": {
                str(p.relative_to(source_root)): file_sha256(p)
                for p in sorted(source_root.rglob("*.py"))
            },
            "weights_sha256": file_sha256(folder / "model.safetensors"),
            "config_sha256": file_sha256(folder / "config.json"),
            "device": str(self.provider.device),
            "dtype": "float32",
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "postprocessing": "sorted_quantiles_no_outer_znorm_or_symmetric_average",
        }

    def predict(self, request: SourceRequest) -> np.ndarray:
        import torch

        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("TimesFM-HNB source must stay frozen and in evaluation mode")
        kwargs = {"horizon": request.horizon}
        names = {
            "target": "target",
            "target_mask": "target_mask",
            "past_only": "past_only_covariates",
            "past_only_mask": "past_only_mask",
            "past_future": "past_future_covariates",
            "past_future_mask": "past_future_mask",
        }
        for name, argument in names.items():
            value = getattr(request, name)
            if value is not None:
                kwargs[argument] = (
                    torch.from_numpy(value.copy()).unsqueeze(0).to(self.provider.device)
                )
        with torch.inference_mode():
            raw = self.model.decode(**kwargs)[0, : len(request.target)]
        return np.sort(raw.cpu().numpy(), axis=-1)


def cached_quantiles(
    source: Source,
    request: SourceRequest,
    cache_dir: pathlib.Path,
) -> tuple[np.ndarray, dict[str, str]]:
    """Atomically cache full conditional requests and reject corrupt hits."""
    identity_bytes = json_bytes(source.identity)
    key = request.key(source.identity)
    path = cache_dir / f"{key}.npz"
    expected_shape = (len(request.target), request.horizon, len(QUANTILES))
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                values = saved["quantiles"]
                envelope = json.loads(str(saved["envelope"].item()))
            if (
                envelope != {"key": key, "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
                or values.dtype != np.float32
            ):
                raise ValueError("cache envelope differs")
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"invalid TimesFM-HNB source cache {key}") from error
    else:
        values = np.asarray(source.predict(request), dtype=np.float32)
        if identity_bytes != json_bytes(source.identity) or request.key(source.identity) != key:
            raise ValueError("TimesFM-HNB source identity or request changed during inference")
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError("TimesFM-HNB source returned invalid quantile shape or values")
        envelope = {"key": key, "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            try:
                np.savez(stream, quantiles=values, envelope=json_bytes(envelope).decode())
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError("TimesFM-HNB cached quantiles have invalid shape or values")
    return values, {"key": key, "sha256": envelope["sha256"]}
