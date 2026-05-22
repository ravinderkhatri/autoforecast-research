"""Rank adapters honestly: choose on DEVELOPMENT origins, report on ACCEPTANCE origins.

NAMED `ranking`, NOT `select`. Python puts a script's own directory on `sys.path[0]`, so a module
called `select.py` here shadows the STDLIB `select` for anything run from this directory -- and
`selectors`, which `asyncio` and `subprocess` import, does `import select`. It surfaced as a
circular import inside pyspark. Avoid names that collide with the standard library or a dependency.

The whole module exists to stop one thing. AutoGluon's default is "the model with the best
validation score is used"; every AutoML tool does something like it; and if the period that chose
the model is also the period that reports it, the reported number is the maximum of N draws, not
an estimate of anything. `EvaluationManifest` already splits origins and refuses a development
origin whose encoder window reaches the acceptance period. This module is what makes the split
BINDING at ranking time rather than advisory.

Two guarantees, both mechanical:

  · **The winner is chosen before an acceptance score is computed.** `rank()` scores the
    development origins, picks, and only then touches acceptance. A `Ranking` cannot be built
    whose `chosen` is not the development winner -- `__post_init__` refuses it.
  · **Every governed adapter is scored on the frozen cells.** An adapter that declines the hard
    cells would otherwise be rewarded for declining them, because its decline set shrinks the
    intersection for every model. A supplied eligible grid is an exact all-origin contract,
    content-bound by the manifest. Extra or malformed forecast rows fail; an omitted key either
    fails or receives a labelled zero according to the manifest's frozen policy. The finalized
    artifact must then equal the grid before scoring. `common_cells` remains only for explicitly
    exploratory runs that supply no grids.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache, partial, reduce
from typing import Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast import grid, registry
from autoforecast._exact_json import ExactJSONError, load_exact_json
from autoforecast.frequency import Cadence, advance, parse_frequency, parse_origin, period_points
from autoforecast.panel import Forecast, Panel
from autoforecast.protocol import EvaluationManifest
from autoforecast.score import Score
from autoforecast.score_spark import MissingActual, score_forecast
from provenance.lineage import directory_digest, parquet_files

eligible_grid_content_hash = grid.content_hash
"""Backward-compatible harness name for the model-neutral grid content digest."""

RANKABLE = ("mae", "sum_abs_error", "absolute_error_pct_of_volume")
"""Score fields a ranking may order by. All are LOWER-IS-BETTER and all are defined at zero.

`bias` is deliberately absent: a forecast can have zero bias and be wrong in both directions at
every cell, so ranking by it selects for cancellation. It is REPORTED beside every rank instead.
"""


class SelectionError(ValueError):
    """A ranking was asked for something that would make its number mean less than it appears to."""


RunOptions = Callable[[str, str, pathlib.Path], Mapping[str, object]]
"""Return frozen adapter-specific arguments for ``(adapter, origin, output_path)``.

The harness deliberately knows nothing about an adapter's signature.  A real evaluation still
has to pass the configuration it registered, though: calling every adapter with only ``spark`` and
``out_dir`` made the custom model silently run its smoke-test defaults while the manifest claimed
a 364-day encoder and a full training budget.  Keeping the hook generic preserves the registry
boundary without throwing away the experiment configuration at that boundary.
"""

MissingForecastPolicy = Literal["error", "zero"]
"""Frozen handling of canonical-grid keys an adapter did not emit.

``error`` leaves omissions for exact-grid validation to reject. ``zero`` makes the harness emit
an explicit deterministic zero baseline for every omission. The latter is reconciliation at the
evaluation boundary, not an adapter-specific fallback, so every compared model receives exactly
the same treatment. An adapter must still return a valid ``Forecast``: execution exceptions and
malformed artifacts are failures of evidence, not permission to manufacture an all-zero run.
"""

CACHE_IDENTITY_VERSION = 2
"""Version of the semantic identity written beside a resumable forecast.

This is deliberately not inferred from whatever fields happen to be present in a ``Forecast``.
Those fields describe the output; they do not identify the panel bytes, evaluation manifest,
adapter configuration, or implementation that produced it.
"""

_RUNTIME_DISTRIBUTIONS = (
    "autogluon",
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "pyspark",
    "pytorch-forecasting",
    "statsforecast",
    "torch",
)


def _json_identity(value: object) -> object:
    """Convert supported values to unambiguous, deterministic JSON data.

    Plain ``json.dumps(..., default=str)`` is unsafe for an identity: it conflates a Path with a
    string and a tuple with a list, while an arbitrary object's repr may contain a memory address.
    Adapter options are configuration, so unsupported opaque objects are refused instead of being
    assigned a cache key that can change accidentally.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float {value!r} has no stable cache identity")
        return value
    if isinstance(value, pathlib.Path):
        return {"__cache_type__": "path", "value": str(value)}
    if isinstance(value, dt.datetime):
        return {"__cache_type__": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"__cache_type__": "date", "value": value.isoformat()}
    if dataclasses.is_dataclass(value):
        return {"__cache_type__": f"dataclass:{type(value).__module__}.{type(value).__qualname__}",
                "value": _json_identity(dataclasses.asdict(value))}
    if isinstance(value, tuple):
        return {"__cache_type__": "tuple", "items": [_json_identity(v) for v in value]}
    if isinstance(value, list):
        return [_json_identity(v) for v in value]
    if isinstance(value, set | frozenset):
        items = [_json_identity(v) for v in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"__cache_type__": type(value).__name__, "items": items}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("cache-identity mappings require string keys")
        return {key: _json_identity(item) for key, item in value.items()}
    raise TypeError(
        f"{type(value).__module__}.{type(value).__qualname__} is not a supported cache-identity "
        "value; pass inert JSON data, paths, dates, tuples, or sets")


def _canonical_json(value: object) -> str:
    """Stable JSON that preserves the configuration types which can affect adapter behaviour."""
    return json.dumps(_json_identity(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _panel_tree_signature(path: pathlib.Path) -> tuple[tuple[str, int, int, int], ...]:
    """Cheap invalidation key for the content digest cached within one harness process.

    ``directory_digest`` still supplies the identity. Size, mtime and ctime only decide whether
    an already-computed digest may be reused in this process; ctime prevents an in-place rewrite
    with a restored mtime from hiding behind the memoisation.
    """
    files = parquet_files(path)
    root = path.parent if path.is_file() else path
    return tuple((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns,
                  p.stat().st_ctime_ns) for p in files)


@lru_cache(maxsize=8)
def _panel_content_digest(path: str, signature: tuple[tuple[str, int, int, int], ...]) -> str:
    """Hash the Parquet bytes, memoised only while their filesystem signature is unchanged."""
    del signature  # it is the cache invalidator; directory_digest reads the actual bytes
    return directory_digest(path, suffixes=(".parquet",))


def _panel_identity(panel: Panel) -> dict[str, object]:
    """Identity of both a panel's role declaration and the data bytes behind its pointer."""
    if dataclasses.is_dataclass(panel):
        declaration: object = dataclasses.asdict(panel)
    else:  # supports structural tests while production remains typed as Panel
        declaration = {"type": f"{type(panel).__module__}.{type(panel).__qualname__}",
                       "repr": repr(panel)}
    raw_path = getattr(panel, "path", None)
    content_hash: str | None = None
    if raw_path is not None:
        path = pathlib.Path(str(raw_path))
        if path.exists():
            content_hash = _panel_content_digest(str(path), _panel_tree_signature(path))
    return {"declaration": declaration, "parquet_content_hash": content_hash}


def _forecast_content_hash(path: str) -> str | None:
    """Hash a materialised forecast; ``None`` retains support for remote/unmaterialised records."""
    artifact = pathlib.Path(path)
    if not artifact.exists():
        return None
    return directory_digest(artifact, suffixes=(".parquet",))


def _code_identity(adapter: str) -> dict[str, object]:
    """Stable identity of repository model code and numerical runtime versions.

    A git SHA alone is insufficient during development: two dirty-tree runs have the same SHA
    while executing different code. Hashing the Python sources is cheap relative to a forecast
    run and also covers untracked implementation files. Dependency versions matter because the
    registered comparators delegate their fitting algorithms to those packages.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for package in ("autoforecast", "forecast", "spec"):
        package_root = root / package
        for source in sorted(package_root.rglob("*.py")):
            if source.name.startswith("test_") or "__pycache__" in source.parts:
                continue
            digest.update(str(source.relative_to(root)).encode())
            digest.update(source.read_bytes())
    entry: dict[str, object] | None = None
    try:
        registered = registry.get(adapter)
        forecast = registered.forecast
        entry = {"module": getattr(forecast, "__module__", type(forecast).__module__),
                 "qualname": getattr(forecast, "__qualname__", type(forecast).__qualname__),
                 "consumes_exposure": registered.consumes_exposure,
                 "handles_right_censoring": registered.handles_right_censoring,
                 "supported_estimands": registered.supported_estimands,
                 "handicap": registered.handicap}
        if isinstance(forecast, partial):
            entry["partial_args"] = forecast.args
            entry["partial_keywords"] = forecast.keywords or {}
    except registry.RegistryError:  # a structural test may mock run without registering an entry
        entry = None
    versions: dict[str, str] = {}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "absent"
    return {"source_sha256": digest.hexdigest(), "adapter_entry": entry,
            "python": platform.python_version(), "dependencies": versions}


def _cache_identity(
    adapter: str,
    origin: str,
    horizon: int,
    panel_identity: Mapping[str, object],
    options: Mapping[str, object],
    manifest_hash: str | None,
    code_identity: Mapping[str, object],
    missing_forecast_policy: str,
    eligible_grid_sha256: str | None,
) -> dict[str, object]:
    """Everything that must agree before an existing forecast can answer this request."""
    return {
        "version": CACHE_IDENTITY_VERSION,
        "adapter": adapter,
        "origin": origin,
        "horizon": horizon,
        "panel": dict(panel_identity),
        "manifest_hash": manifest_hash,
        "run_options": dict(options),
        "missing_forecast": {
            "policy": missing_forecast_policy,
            "eligible_grid_sha256": eligible_grid_sha256,
        },
        "code": dict(code_identity),
    }


def _reuse_record(
    sidecar: pathlib.Path,
    expected: dict[str, object],
    adapter: str,
    origin: str,
    horizon: int,
) -> Forecast:
    """Load a sidecar only after proving it describes this exact forecast request."""
    try:
        record = load_exact_json(sidecar)
    except ExactJSONError as exc:
        raise SelectionError(f"cannot reuse {adapter}/{origin}: unreadable sidecar {sidecar}: "
                             f"{exc}") from exc
    if type(record) is not dict:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: sidecar {sidecar} is not a JSON object")
    stored = record.get("cache_identity")
    stored_key = record.get("cache_key")
    if not isinstance(stored, dict) or not isinstance(stored_key, str):
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: the sidecar predates cache identity version "
            f"{CACHE_IDENTITY_VERSION}. Re-run this origin once before resuming")
    if stored.get("version") != CACHE_IDENTITY_VERSION:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: sidecar cache identity version "
            f"{stored.get('version')!r} is not {CACHE_IDENTITY_VERSION}. Re-run this origin; "
            "older identities did not bind the missing-forecast policy")
    if _sha256_json(stored) != stored_key:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: its sidecar cache identity was edited or corrupted")
    expected_key = _sha256_json(expected)
    if stored_key != expected_key:
        changed = sorted(key for key in expected if stored.get(key) != expected.get(key))
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: cached request differs in {changed}. Reusing it "
            "would attach the current evaluation contract to another forecast")
    fc = Forecast.from_record(record)
    actual_span = (fc.adapter, fc.origin, fc.horizon)
    expected_span = (adapter, origin, horizon)
    if actual_span != expected_span:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: sidecar forecast identifies {actual_span}, "
            f"expected {expected_span}")
    if "forecast_content_hash" not in record:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: sidecar does not identify the forecast bytes")
    current_content_hash = _forecast_content_hash(fc.path)
    if fc.n_emitted > 0 and current_content_hash is None:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: it claims {fc.n_emitted} emitted cells but its "
            "forecast artifact is not materialised")
    if record["forecast_content_hash"] != current_content_hash:
        raise SelectionError(
            f"cannot reuse {adapter}/{origin}: forecast artifact changed after the sidecar was "
            "written")
    fc = fc.validate()
    _validate_reconciliation_record(record, expected, fc)
    return fc


@dataclass(frozen=True)
class Ranking:
    """Adapters ranked on development origins, then reported on acceptance origins.

    Attributes:
        manifest_hash: Which manifest produced this. A ranking that does not name its manifest
            cannot be checked against the origins it claims to have used.
        metric: The `Score` field ordered by, from `RANKABLE`.
        development: adapter -> pooled `Score` over the development origins. This decided.
        acceptance: adapter -> pooled `Score` over the acceptance origins. This reports. Empty
            when the manifest reserved no acceptance origins, which is legitimate for exploration
            and means nothing here may be quoted as an accepted result.
        chosen: The development winner.
        handicaps: adapter -> what it cannot honour, for every ranked adapter that declared one.
        n_common_cells: Smallest per-origin comparison-cell count. For a governed run with
            canonical grids, every adapter emitted exactly these cells. For an explicitly
            exploratory run without grids, these are the cells every adapter happened to emit.
        acceptance_forecasts: adapter -> origin -> the `Forecast` that produced the acceptance
            score. Carried so a caller can plot or re-score the winner without RUNNING IT AGAIN --
            re-running the shallow model to draw one figure rebuilds eight origins of features.
            Also what lets a gate result cite the exact Parquet its number came from.
    """

    manifest_hash: str
    metric: str
    development: dict[str, Score]
    acceptance: dict[str, Score]
    chosen: str
    handicaps: dict[str, str] = field(default_factory=dict)
    n_common_cells: int = 0
    acceptance_forecasts: dict[str, dict[str, Forecast]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a ranking whose winner is not the development winner.

        The one guard that makes the split binding rather than advisory. Constructing a `Ranking`
        with `chosen` set from the acceptance scores -- by hand, or by a later refactor that
        "simplifies" the two dicts into one -- fails here.

        Raises:
            SelectionError: `metric` is not rankable, `development` is empty, or `chosen` is not
                the best development adapter.
        """
        if not isinstance(self.manifest_hash, str) or not self.manifest_hash.strip():
            raise SelectionError("a ranking needs a non-blank manifest identity")
        if self.metric not in RANKABLE:
            raise SelectionError(
                f"cannot rank by {self.metric!r}; permitted: {RANKABLE}. Ranking by bias selects "
                "for errors that cancel, which is why it is reported and never ordered by")
        if not isinstance(self.development, dict) or not self.development:
            raise SelectionError("a ranking needs at least one development score")
        for label, scores in (("development", self.development),
                              ("acceptance", self.acceptance)):
            if not isinstance(scores, dict):
                raise SelectionError(f"{label} scores must be an adapter-to-Score mapping")
            malformed = {
                adapter: score for adapter, score in scores.items()
                if (not isinstance(adapter, str) or not adapter.strip()
                    or not isinstance(score, Score))
            }
            if malformed:
                raise SelectionError(
                    f"{label} scores need non-blank adapter names and validated Score values")
        if self.acceptance and set(self.acceptance) != set(self.development):
            raise SelectionError(
                "acceptance scores must contain exactly the development adapters; "
                f"missing={sorted(set(self.development) - set(self.acceptance))}, "
                f"unexpected={sorted(set(self.acceptance) - set(self.development))}")
        if (not isinstance(self.chosen, str) or not self.chosen.strip()
                or self.chosen not in self.development):
            raise SelectionError(
                f"chosen adapter {self.chosen!r} is not a named development candidate")
        if (not isinstance(self.n_common_cells, int) or isinstance(self.n_common_cells, bool)
                or self.n_common_cells < 0):
            raise SelectionError("n_common_cells must be a non-negative integer")
        if (not isinstance(self.handicaps, dict)
                or any(name not in self.development or not isinstance(name, str)
                       or not isinstance(reason, str) or not reason.strip()
                       for name, reason in self.handicaps.items())):
            raise SelectionError(
                "handicaps must map ranked adapters to non-blank limitation descriptions")
        winner = _best(self.development, self.metric)
        if self.chosen != winner:
            raise SelectionError(
                f"chosen={self.chosen!r} but the development winner on {self.metric} is "
                f"{winner!r}. The acceptance origins report a result; they do not pick one")

    @property
    def order(self) -> tuple[str, ...]:
        """Adapter names best-first on the development metric."""
        return tuple(sorted(self.development, key=lambda a: _value(self.development[a],
                                                                   self.metric)))

    def as_record(self) -> dict[str, object]:
        """A JSON-safe record for the gate result."""
        def flat(d: dict[str, Score]) -> dict[str, dict[str, float | None]]:
            return {a: {k: getattr(s, k) for k in
                        ("n_scored", "mae", "bias", "bias_pct_of_volume", "sum_abs_error",
                         "sum_error", "sum_actual")}
                    | {"wape": s.absolute_error_pct_of_volume}
                    for a, s in d.items()}
        return {"manifest_hash": self.manifest_hash, "metric": self.metric,
                "chosen": self.chosen, "order": list(self.order),
                "n_common_cells": self.n_common_cells, "handicaps": self.handicaps,
                "development": flat(self.development), "acceptance": flat(self.acceptance),
                # Paths, not the Forecast objects: a record cites where a number came from.
                "acceptance_forecast_paths": {a: {o: f.path for o, f in runs.items()}
                                              for a, runs in self.acceptance_forecasts.items()}}


def _value(s: Score, metric: str) -> float:
    """The rankable value, with `None` mapped to +inf so an undefined score never wins."""
    v = getattr(s, metric)
    return float("inf") if v is None else float(v)


def _best(scores: dict[str, Score], metric: str) -> str:
    """Lowest metric wins; ties break on name so a ranking is reproducible."""
    return min(sorted(scores), key=lambda a: _value(scores[a], metric))


def _require_hourly_key_type(frame: DataFrame, cadence: Cadence, label: str) -> None:
    """Keep a date column from satisfying an hourly contract through a lossy cast."""
    if cadence.key_kind == "timestamp" and (
        frame.schema["date"].dataType.typeName() != "timestamp_ntz"
    ):
        raise SelectionError(
            f"{label} hourly date must have Spark timestamp_ntz type, got "
            f"{frame.schema['date'].dataType.simpleString()}"
        )


def common_cells(
    forecasts: list[Forecast], spark: SparkSession, out_dir: str | pathlib.Path
) -> tuple[str, int]:
    """Write the `series_id · date` cells that EVERY forecast emitted. Returns (path, count).

    Args:
        forecasts: One per adapter, all for the same origin and horizon.
        spark: Active session.
        out_dir: Where the intersection is written.

    Returns:
        The Parquet path and the number of common cells.

    Raises:
        SelectionError: the forecasts disagree on origin or horizon -- an intersection across
            different windows is not a comparison; or the intersection is empty.
    """
    if not forecasts:
        raise SelectionError("no forecasts to intersect")
    for forecast in forecasts:
        if type(forecast.horizon) is not int or forecast.horizon < 1:
            raise SelectionError(
                f"{forecast.adapter}: horizon must be a positive integer, got "
                f"{forecast.horizon!r}"
            )
        try:
            forecast.validate()
            cadence = parse_frequency(forecast.frequency)
            origin = parse_origin(forecast.origin, cadence)
            advance(origin, cadence, forecast.horizon)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SelectionError(
                f"{forecast.adapter}: forecast does not define a valid canonical artifact and "
                f"span: {exc}"
            ) from exc
    spans = {(f.origin, f.horizon, f.frequency) for f in forecasts}
    if len(spans) > 1:
        raise SelectionError(
            f"forecasts span different (origin, horizon, frequency): {sorted(spans)}. "
            "Intersecting cells "
            "across different windows compares forecasts of different things")
    frames = []
    for forecast in forecasts:
        try:
            frames.append(
                grid.read_grid_keys(
                    forecast.path, spark, frequency=forecast.frequency
                )
            )
        except grid.GridError as exc:
            raise SelectionError(
                f"{forecast.adapter}: forecast keys are invalid: {exc}"
            ) from exc
    keep = reduce(lambda a, b: a.intersect(b), frames).distinct()
    out = str(pathlib.Path(out_dir))
    keep.write.mode("overwrite").parquet(out)
    n = spark.read.parquet(out).count()
    if n == 0:
        raise SelectionError(
            f"no cell was emitted by all {len(forecasts)} adapters, so there is nothing they can "
            "be compared on. Coverage: "
            + ", ".join(f"{f.adapter}={f.n_emitted}" for f in forecasts))
    return out, n


def _require_complete_grid_mapping(
    manifest: EvaluationManifest,
    eligible_grids: Mapping[str, str | pathlib.Path],
    spark: SparkSession,
) -> dict[str, str]:
    """Make a supplied grid a complete manifest contract, not a scoring hint.

    A partial mapping is especially dangerous: the origins with a grid look governed while the
    remaining development origins quietly fall back to adapter intersections. The omitted hard
    cells can then change which model wins without changing the manifest. Once a caller supplies
    grids, every frozen origin is therefore required and no undeclared origin is accepted.
    """
    if not isinstance(eligible_grids, Mapping):
        raise SelectionError("eligible_grids must map every manifest origin to a Parquet path")
    if any(not isinstance(origin, str) for origin in eligible_grids):
        raise SelectionError("eligible_grids keys must be manifest origin strings")
    keys = set(eligible_grids)
    expected = set(manifest.origins)
    missing = expected - keys
    unexpected = keys - expected
    if missing or unexpected:
        raise SelectionError(
            "eligible_grids must contain exactly the manifest origins; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if not manifest.eligible_grid_hashes:
        raise SelectionError(
            "a governed rank supplied eligible_grids but the EvaluationManifest has no "
            "eligible_grid_hashes. Describing a grid rule does not bind the cells actually used")
    out: dict[str, str] = {}
    for origin in manifest.origins:
        path = eligible_grids[origin]
        if not isinstance(path, str | pathlib.Path) or not str(path).strip():
            raise SelectionError(
                f"eligible_grids[{origin!r}] must be a non-blank Parquet path")
        out[origin] = str(path)
        try:
            actual_hash = eligible_grid_content_hash(path, spark)
        except Exception as exc:
            raise SelectionError(
                f"cannot hash eligible grid for {origin} at {path!s}: {exc}") from exc
        expected_hash = manifest.eligible_grid_hashes[origin]
        if actual_hash != expected_hash:
            raise SelectionError(
                f"eligible grid for {origin} has content hash {actual_hash}, but the frozen "
                f"manifest requires {expected_hash}. The materialised cell set changed")
    return out


def _exact_grid_restrictions(
    runs: dict[str, dict[str, Forecast]],
    origins: tuple[str, ...],
    eligible_grids: Mapping[str, str],
    spark: SparkSession,
) -> tuple[dict[str, str], int]:
    """Prove every finalized adapter artifact equals each origin's canonical grid.

    Row counts are insufficient: an adapter can replace a hard missing cell with an easy extra
    cell and preserve its count. Both anti-join directions are checked on the actual key sets,
    before a score is calculated. The grid itself is validated here as a unique, non-empty set
    inside the forecast window; ``score_forecast`` independently validates it again at the metric
    boundary.
    """
    if not runs:
        raise SelectionError("no adapter forecasts were supplied for exact-grid validation")
    counts: list[int] = []
    for origin in origins:
        missing_adapters = sorted(adapter for adapter, by_origin in runs.items()
                                  if origin not in by_origin)
        if missing_adapters:
            raise SelectionError(
                f"adapters {missing_adapters} have no forecast for manifest origin {origin}")
        spans = {(by_origin[origin].origin, by_origin[origin].horizon,
                  by_origin[origin].frequency)
                 for by_origin in runs.values()}
        if len(spans) != 1 or next(iter(spans))[0] != origin:
            raise SelectionError(
                f"forecasts for {origin} disagree on (origin, horizon, frequency): "
                f"{sorted(spans)}")
        horizon = next(iter(spans))[1]
        frequency = next(iter(spans))[2]
        path = eligible_grids[origin]
        raw_grid = grid.read_time_parquet(spark, path, "date")
        missing_columns = sorted({"series_id", "date"} - set(raw_grid.columns))
        if missing_columns:
            raise SelectionError(
                f"eligible grid for {origin} lacks required columns {missing_columns}")
        cadence = parse_frequency(frequency)
        _require_hourly_key_type(raw_grid, cadence, f"eligible grid for {origin}")
        grid_key, valid_grid_key = grid.canonical_key("date", cadence)
        canonical_grid = raw_grid.select(
            F.col("series_id").cast("string").alias("series_id"),
            grid_key.alias("date"),
            valid_grid_key.alias("_valid_date"),
        )
        if canonical_grid.filter(
                F.col("series_id").isNull()
                | (F.trim("series_id") == F.lit(""))
                | ~F.col("_valid_date")).limit(1).count():
            raise SelectionError(
                f"eligible grid for {origin} contains a null/blank series_id or invalid time key")
        canonical_grid = canonical_grid.drop("_valid_date")
        if (canonical_grid.groupBy("series_id", "date").count()
                .filter(F.col("count") != 1).limit(1).count()):
            raise SelectionError(
                f"eligible grid for {origin} contains duplicate (series_id, date) keys")
        try:
            allowed_dates = period_points(origin, cadence, horizon)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SelectionError(
                f"forecasts for {origin} do not define a supported period grid") from exc
        start, end = allowed_dates[0], allowed_dates[-1]
        if canonical_grid.filter(
            ~grid.period_membership("date", allowed_dates, cadence)
        ).limit(1).count():
            raise SelectionError(
                f"eligible grid for {origin} contains dates outside the exact period grid "
                f"[{start}, {end}]")
        n_grid = canonical_grid.count()
        if n_grid == 0:
            raise SelectionError(f"eligible grid for {origin} is empty")

        for adapter in sorted(runs):
            forecast = runs[adapter][origin]
            raw_forecast = grid.read_time_parquet(spark, forecast.path, "date")
            missing_forecast_columns = sorted(
                {"series_id", "date"} - set(raw_forecast.columns))
            if missing_forecast_columns:
                raise SelectionError(
                    f"{adapter}/{origin}: forecast lacks key columns "
                    f"{missing_forecast_columns}")
            _require_hourly_key_type(raw_forecast, cadence, f"{adapter}/{origin} forecast")
            forecast_key, valid_forecast_key = grid.canonical_key("date", cadence)
            keys = raw_forecast.select(
                F.col("series_id").cast("string").alias("series_id"),
                forecast_key.alias("date"),
                valid_forecast_key.alias("_valid_date"),
            )
            if keys.filter(~F.col("_valid_date")).limit(1).count():
                raise SelectionError(
                    f"{adapter}/{origin}: forecast contains an invalid time key")
            keys = keys.drop("_valid_date")
            if (keys.groupBy("series_id", "date").count()
                    .filter(F.col("count") != 1).limit(1).count()):
                raise SelectionError(
                    f"{adapter}/{origin}: forecast contains duplicate comparison keys")
            missing = canonical_grid.join(keys, ["series_id", "date"], "left_anti").count()
            extra = keys.join(canonical_grid, ["series_id", "date"], "left_anti").count()
            if missing or extra:
                raise SelectionError(
                    f"{adapter}/{origin}: forecast does not equal the canonical eligible grid: "
                    f"missing={missing}, extra={extra}. A governed ranking cannot let an "
                    "adapter's decline set choose the cells on which every model is selected")
        counts.append(n_grid)
    return {origin: eligible_grids[origin] for origin in origins}, min(counts)


def _canonical_grid(
    path: str | pathlib.Path,
    origin: str,
    horizon: int,
    spark: SparkSession,
    *,
    frequency: str,
) -> DataFrame:
    """Read one eligible grid into canonical join keys after validating its full contract."""
    raw = grid.read_time_parquet(spark, path, "date")
    missing = sorted({"series_id", "date"} - set(raw.columns))
    if missing:
        raise SelectionError(f"eligible grid for {origin} lacks required columns {missing}")
    cadence = parse_frequency(frequency)
    _require_hourly_key_type(raw, cadence, f"eligible grid for {origin}")
    time_key, valid_time_key = grid.canonical_key("date", cadence)
    canonical_grid = raw.select(
        F.col("series_id").alias("_grid_series_id"),
        F.col("series_id").cast("string").alias("_series_key"),
        time_key.alias("date"),
        valid_time_key.alias("_valid_date"),
    )
    if canonical_grid.filter(
            F.col("_series_key").isNull()
            | (F.trim("_series_key") == F.lit(""))
            | ~F.col("_valid_date")).limit(1).count():
        raise SelectionError(
            f"eligible grid for {origin} contains a null/blank series_id or invalid time key")
    canonical_grid = canonical_grid.drop("_valid_date")
    if (canonical_grid.groupBy("_series_key", "date").count()
            .filter(F.col("count") != 1).limit(1).count()):
        raise SelectionError(
            f"eligible grid for {origin} contains duplicate canonical (series_id, date) keys")
    try:
        allowed_dates = period_points(origin, cadence, horizon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SelectionError(
            f"eligible grid for {origin} does not have a supported period cadence") from exc
    start, end = allowed_dates[0], allowed_dates[-1]
    if canonical_grid.filter(
        ~grid.period_membership("date", allowed_dates, cadence)
    ).limit(1).count():
        raise SelectionError(
            f"eligible grid for {origin} contains dates outside the exact period grid "
            f"[{start}, {end}]")
    if canonical_grid.limit(1).count() == 0:
        raise SelectionError(f"eligible grid for {origin} is empty")
    return canonical_grid


def _validated_source_frame(forecast: Forecast, spark: SparkSession) -> DataFrame | None:
    """Validate every raw adapter row before a grid is allowed to add or remove anything.

    In particular this validates values *before* the extra-key anti-join. A malformed value on an
    extra row cannot disappear merely because that row was outside the grid, and a duplicate easy
    key cannot stand in for an omitted hard one while preserving the adapter's row count.
    """
    artifact = pathlib.Path(forecast.path)
    if not artifact.exists():
        if forecast.n_emitted:
            raise SelectionError(
                f"adapter {forecast.adapter!r} claims {forecast.n_emitted} emitted cells but did "
                f"not materialise its forecast artifact at {forecast.path!r}")
        return None

    raw = grid.read_time_parquet(spark, forecast.path, "date")
    missing = sorted({"series_id", "date", "yhat"} - set(raw.columns))
    if missing:
        raise SelectionError(
            f"{forecast.adapter}/{forecast.origin}: forecast lacks required columns {missing}")
    yhat = F.expr("try_cast(yhat as double)")
    cadence = parse_frequency(forecast.frequency)
    _require_hourly_key_type(raw, cadence, f"{forecast.adapter}/{forecast.origin} forecast")
    time_key, valid_time_key = grid.canonical_key("date", cadence)
    source = raw.select(
        F.col("series_id").cast("string").alias("_series_key"),
        time_key.alias("date"),
        valid_time_key.alias("_valid_date"),
        yhat.alias("yhat"),
    )
    if source.filter(
            F.col("_series_key").isNull()
            | (F.trim("_series_key") == F.lit(""))
            | ~F.col("_valid_date")).limit(1).count():
        raise SelectionError(
            f"{forecast.adapter}/{forecast.origin}: forecast contains a null/blank series_id or "
            "invalid time key")
    source = source.drop("_valid_date")
    if source.filter(
            F.col("yhat").isNull()
            | F.isnan("yhat")
            | (F.abs("yhat") == float("inf"))
            | (F.col("yhat") < 0)).limit(1).count():
        raise SelectionError(
            f"{forecast.adapter}/{forecast.origin}: yhat must be finite and non-negative")
    if (source.groupBy("_series_key", "date").count()
            .filter(F.col("count") != 1).limit(1).count()):
        raise SelectionError(
            f"{forecast.adapter}/{forecast.origin}: forecast contains duplicate comparison keys")
    n_rows = source.count()
    if n_rows != forecast.n_emitted:
        raise SelectionError(
            f"{forecast.adapter}/{forecast.origin}: disposition claims {forecast.n_emitted} "
            f"emitted cells but the raw forecast holds {n_rows}")
    return source


def _reconcile_zero_fallback(
    forecast: Forecast,
    eligible_grid: str | pathlib.Path,
    eligible_grid_sha256: str,
    spark: SparkSession,
    final_path: pathlib.Path,
) -> tuple[Forecast, dict[str, object]]:
    """Finalize raw adapter output on the canonical grid with an explicit zero fallback.

    Missing keys are the only condition repaired. Extras, duplicate keys, malformed identities,
    malformed values, and values outside the requested window all fail before the output is
    written. The raw adapter artifact is retained beside the finalized artifact for audit.
    """
    # The adapter-level account is checked first, before reconciliation changes its meaning.
    try:
        forecast.validate()
    except Exception as exc:
        raise SelectionError(
            f"{forecast.adapter}/{forecast.origin}: invalid raw forecast: {exc}") from exc

    grid_frame = _canonical_grid(
        eligible_grid,
        forecast.origin,
        forecast.horizon,
        spark,
        frequency=forecast.frequency,
    ).cache()
    try:
        source = _validated_source_frame(forecast, spark)
        if source is None:
            source = grid_frame.limit(0).select("_series_key", "date").withColumn(
                "yhat", F.lit(None).cast("double"))
        else:
            allowed_dates = period_points(
                forecast.origin, forecast.frequency, forecast.horizon
            )
            start, end = allowed_dates[0], allowed_dates[-1]
            cadence = parse_frequency(forecast.frequency)
            if source.filter(
                ~grid.period_membership("date", allowed_dates, cadence)
            ).limit(1).count():
                raise SelectionError(
                    f"{forecast.adapter}/{forecast.origin}: forecast contains dates outside "
                    f"[{start}, {end}]")

        grid_keys = grid_frame.select("_series_key", "date")
        extra = source.join(grid_keys, ["_series_key", "date"], "left_anti").count()
        if extra:
            raise SelectionError(
                f"{forecast.adapter}/{forecast.origin}: raw forecast contains {extra} key(s) "
                "outside the canonical eligible grid. Zero fallback repairs omissions only; "
                "extras are a malformed forecast and are never hidden")

        present = source.withColumn("_adapter_emitted", F.lit(True))
        joined = grid_frame.join(present, ["_series_key", "date"], "left")
        stats = joined.agg(
            F.count(F.lit(1)).alias("n_grid"),
            F.countDistinct("_series_key").alias("n_series"),
            F.sum(F.when(F.col("_adapter_emitted").isNull(), F.lit(1)).otherwise(F.lit(0)))
            .alias("n_fallback"),
        ).collect()[0]
        n_grid = int(stats["n_grid"])
        n_series = int(stats["n_series"])
        n_fallback = int(stats["n_fallback"] or 0)
        n_adapter = n_grid - n_fallback

        finalized = joined.select(
            F.col("_grid_series_id").alias("series_id"),
            F.col("date"),
            F.when(F.col("_adapter_emitted").isNull(), F.lit(0.0))
            .otherwise(F.col("yhat")).cast("double").alias("yhat"),
            F.when(F.col("_adapter_emitted").isNull(), F.lit("fallback_zero"))
            .otherwise(F.lit("adapter")).alias("forecast_source"),
        )
        finalized.write.mode("overwrite").parquet(str(final_path))

        rectangle = n_series * forecast.horizon
        if n_grid > rectangle:
            # Unique keys within a horizon make this impossible unless the grid contract or
            # Spark canonicalisation has drifted. Keep the check local to the arithmetic it
            # protects.
            raise SelectionError(
                f"eligible grid for {forecast.origin} has {n_grid} cells but only {rectangle} "
                f"fit in {n_series} series x {forecast.horizon} periods")
        disposition = {"emitted_adapter": n_adapter}
        if n_fallback:
            disposition["emitted_fallback_zero"] = n_fallback
        outside = rectangle - n_grid
        if outside:
            disposition["declined_outside_eligible_grid"] = outside
        result = Forecast(
            adapter=forecast.adapter,
            path=str(final_path),
            origin=forecast.origin,
            horizon=forecast.horizon,
            n_series=n_series,
            disposition=disposition,
            target_estimand=forecast.target_estimand,
            frequency=forecast.frequency,
        ).validate()
        source_hash = _forecast_content_hash(forecast.path)
        reconciliation: dict[str, object] = {
            "version": 1,
            "policy": "zero",
            "eligible_grid_path": str(eligible_grid),
            "eligible_grid_sha256": eligible_grid_sha256,
            "adapter_emitted": n_adapter,
            "fallback_emitted": n_fallback,
            "canonical_cells": n_grid,
            "source_forecast": forecast.as_record(),
            "source_forecast_content_hash": source_hash,
            "final_forecast_path": str(final_path),
        }
        return result, reconciliation
    finally:
        grid_frame.unpersist()


def _validate_reconciliation_record(
    record: Mapping[str, object], expected: Mapping[str, object], forecast: Forecast,
) -> None:
    """Require a reusable zero-fallback sidecar to prove how its final artifact was formed."""
    fallback = expected.get("missing_forecast")
    if not isinstance(fallback, Mapping) or fallback.get("policy") != "zero":
        return
    reconciliation = record.get("grid_reconciliation")
    stored_hash = record.get("grid_reconciliation_hash")
    if not isinstance(reconciliation, dict) or not isinstance(stored_hash, str):
        raise SelectionError(
            f"cannot reuse {forecast.adapter}/{forecast.origin}: sidecar lacks its zero-fallback "
            "reconciliation record")
    if _sha256_json(reconciliation) != stored_hash:
        raise SelectionError(
            f"cannot reuse {forecast.adapter}/{forecast.origin}: fallback reconciliation record "
            "was edited or corrupted")
    if (reconciliation.get("policy") != "zero"
            or reconciliation.get("eligible_grid_sha256")
            != fallback.get("eligible_grid_sha256")
            or reconciliation.get("final_forecast_path") != forecast.path):
        raise SelectionError(
            f"cannot reuse {forecast.adapter}/{forecast.origin}: fallback reconciliation does "
            "not match the cached request or finalized forecast")
    n_adapter = forecast.disposition.get("emitted_adapter", 0)
    n_fallback = forecast.disposition.get("emitted_fallback_zero", 0)
    if (reconciliation.get("adapter_emitted") != n_adapter
            or reconciliation.get("fallback_emitted") != n_fallback
            or reconciliation.get("canonical_cells") != forecast.n_emitted):
        raise SelectionError(
            f"cannot reuse {forecast.adapter}/{forecast.origin}: fallback disposition and "
            "reconciliation counts disagree")
    source = reconciliation.get("source_forecast")
    if not isinstance(source, dict):
        raise SelectionError(
            f"cannot reuse {forecast.adapter}/{forecast.origin}: fallback source record is absent")
    source_forecast = Forecast.from_record(source)
    current_source_hash = _forecast_content_hash(source_forecast.path)
    if current_source_hash != reconciliation.get("source_forecast_content_hash"):
        raise SelectionError(
            f"cannot reuse {forecast.adapter}/{forecast.origin}: raw adapter artifact changed "
            "after zero-fallback finalization")


def finalize_forecast_on_grid(
    forecast: Forecast,
    eligible_grid: str | pathlib.Path,
    eligible_grid_sha256: str,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    missing_forecast_policy: MissingForecastPolicy,
) -> tuple[Forecast, dict[str, object] | None]:
    """Apply one frozen missing-key policy and return the artifact used downstream.

    This is public because non-registered harness forecasts such as the Phase-2 headroom oracle
    must receive the same treatment as registered adapters. A policy of ``error`` validates but
    does not rewrite; exact-grid validation remains responsible for rejecting omissions. A policy
    of ``zero`` produces the labelled canonical artifact and its reconciliation provenance.
    """
    if missing_forecast_policy not in ("error", "zero"):
        raise SelectionError(
            "missing_forecast_policy must be 'error' or 'zero', got "
            f"{missing_forecast_policy!r}")
    if (not isinstance(eligible_grid_sha256, str) or len(eligible_grid_sha256) != 64
            or any(c not in "0123456789abcdef" for c in eligible_grid_sha256)):
        raise SelectionError("eligible_grid_sha256 must be a lowercase SHA-256 digest")
    if missing_forecast_policy == "error":
        return forecast.validate(), None
    try:
        actual_hash = eligible_grid_content_hash(eligible_grid, spark)
    except Exception as exc:
        raise SelectionError(
            f"cannot hash eligible grid for {forecast.origin} at {eligible_grid!s}: {exc}") from exc
    if actual_hash != eligible_grid_sha256:
        raise SelectionError(
            f"eligible grid for {forecast.origin} changed after its policy/cache identity was "
            f"frozen: expected {eligible_grid_sha256}, got {actual_hash}")
    return _reconcile_zero_fallback(
        forecast, eligible_grid, eligible_grid_sha256, spark, pathlib.Path(out_dir))


def run_over_origins(
    adapter: str,
    panel: Panel,
    origins: tuple[str, ...],
    horizon: int,
    spark: SparkSession,
    out_root: pathlib.Path,
    *,
    reuse: bool = False,
    run_options: RunOptions | None = None,
    manifest_hash: str | None = None,
    missing_forecast_policy: MissingForecastPolicy = "error",
    eligible_grids: Mapping[str, str | pathlib.Path] | None = None,
    eligible_grid_hashes: Mapping[str, str] | None = None,
) -> dict[str, Forecast]:
    """Run one adapter over several origins. Returns origin -> `Forecast`.

    Separated from scoring because the comparison grid must be checked against every adapter's
    forecast before any score is calculated, and running an adapter twice -- once to compare keys,
    once to score -- doubles the most expensive step in the phase.

    Args:
        adapter: Registry name.
        panel: Data.
        origins: Frozen origins to run.
        horizon: Periods per origin.
        spark: Active session.
        out_root: `<out_root>/<adapter>/<origin>` holds each forecast, with a `.json` sidecar
            recording its disposition.
        reuse: Skip an origin whose sidecar already exists, rebuilding the `Forecast` from it.

            **Explicit opt-in, never a default.** The sidecar is reused only when its versioned
            identity matches the requested adapter, origin, horizon, Panel declaration and
            Parquet content, manifest, adapter options, source code, and numerical dependency
            versions. It exists because a gate run over four origins and four adapters takes hours
            on one machine (measured: AutoETS alone is 28 minutes per origin at M5's 30,490
            series), and losing three completed origins to a crash in the fourth is how a
            measurement stops getting made.
        manifest_hash: Evaluation manifest governing the run. ``rank`` always supplies it. A
            direct exploratory call may omit it, but cannot then reuse a manifest-bound sidecar.
        missing_forecast_policy: Frozen handling of a canonical-grid omission. ``"error"`` leaves
            it for exact-grid validation to reject. ``"zero"`` writes an explicit zero row with
            ``forecast_source="fallback_zero"``. Never inferred from adapter behaviour.
        eligible_grids: Canonical grids used by ``"zero"`` reconciliation, keyed by origin.
        eligible_grid_hashes: Content hashes already checked against the registered manifest by
            ``rank``. They enter the cache identity so equal paths with changed keys cannot reuse.

    Returns:
        origin -> `Forecast`.
    """
    if manifest_hash is not None and (not isinstance(manifest_hash, str)
                                      or not manifest_hash.strip()):
        raise SelectionError("manifest_hash must be a non-blank string or None")
    if missing_forecast_policy not in ("error", "zero"):
        raise SelectionError(
            "missing_forecast_policy must be 'error' or 'zero', got "
            f"{missing_forecast_policy!r}")
    if (eligible_grids is None) != (eligible_grid_hashes is None):
        raise SelectionError(
            "eligible_grids and eligible_grid_hashes must be supplied together so the cache "
            "identity names the exact canonical cells")
    if missing_forecast_policy == "zero" and eligible_grids is None:
        raise SelectionError(
            "missing_forecast_policy='zero' requires a content-bound eligible grid for every "
            "requested origin")
    if eligible_grids is not None:
        missing_paths = set(origins) - set(eligible_grids)
        missing_hashes = set(origins) - set(eligible_grid_hashes or {})
        if missing_paths or missing_hashes:
            raise SelectionError(
                "eligible grid reconciliation mappings omit requested origins: "
                f"paths={sorted(missing_paths)}, hashes={sorted(missing_hashes)}")
    panel_cache_identity = _panel_identity(panel)
    code_cache_identity = _code_identity(adapter)
    out: dict[str, Forecast] = {}
    for origin in origins:
        target = out_root / adapter / origin
        sidecar = target.with_suffix(".json")
        grid_path = eligible_grids[origin] if eligible_grids is not None else None
        grid_hash = eligible_grid_hashes[origin] if eligible_grid_hashes is not None else None
        if grid_hash is not None and (not isinstance(grid_hash, str)
                                      or len(grid_hash) != 64
                                      or any(c not in "0123456789abcdef" for c in grid_hash)):
            raise SelectionError(
                f"eligible_grid_hashes[{origin!r}] must be a lowercase SHA-256 digest")
        extra = dict(run_options(adapter, origin, target)) if run_options is not None else {}
        reserved = sorted(set(extra) & {"spark", "out_dir"})
        if reserved:
            raise SelectionError(
                f"run_options for {adapter!r} tried to replace harness-owned arguments "
                f"{reserved}. The output location and Spark session define the evaluated run")
        try:
            options_json = _canonical_json(extra)
        except (TypeError, ValueError) as exc:
            raise SelectionError(
                f"run_options for {adapter!r} are not deterministically serialisable: {exc}") \
                from exc
        # Round-trip once so the in-memory comparison uses exactly the representation persisted
        # in JSON (for example a pathlib.Path is a string on both sides, not merely hash-equal).
        identity = json.loads(_canonical_json(
            _cache_identity(adapter, origin, horizon, panel_cache_identity,
                            json.loads(options_json), manifest_hash, code_cache_identity,
                            missing_forecast_policy, grid_hash)))
        if reuse and sidecar.exists():
            reused = _reuse_record(sidecar, identity, adapter, origin, horizon)
            if reused.frequency != panel.frequency:
                raise SelectionError(
                    f"cannot reuse {adapter}/{origin}: cached frequency "
                    f"{reused.frequency!r} differs from panel frequency {panel.frequency!r}")
            out[origin] = reused
            continue
        fc = registry.run(adapter, panel, origin, horizon, spark=spark, out_dir=str(target),
                          **extra)
        actual_span = (fc.adapter, fc.origin, fc.horizon, fc.frequency)
        expected_span = (adapter, origin, horizon, panel.frequency)
        if actual_span != expected_span:
            raise SelectionError(
                f"adapter result identifies {actual_span}, expected request {expected_span}")
        reconciliation: dict[str, object] | None = None
        if missing_forecast_policy == "zero":
            # Both are guaranteed above; the assertions keep the narrowing local and make an
            # impossible partial configuration fail loudly if this function is later refactored.
            assert grid_path is not None and grid_hash is not None
            fc, reconciliation = finalize_forecast_on_grid(
                fc, grid_path, grid_hash, spark,
                target.parent / f"{target.name}.canonical",
                missing_forecast_policy=missing_forecast_policy)
        else:
            fc = fc.validate()
            if fc.n_emitted > 0 and _forecast_content_hash(fc.path) is None:
                raise SelectionError(
                    f"adapter {adapter!r} claims {fc.n_emitted} emitted cells but did not "
                    f"materialise its forecast artifact at {fc.path!r}")
        forecast_content_hash = _forecast_content_hash(fc.path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        record = fc.as_record() | {
            "cache_key": _sha256_json(identity),
            "cache_identity": identity,
            "forecast_content_hash": forecast_content_hash,
        }
        if reconciliation is not None:
            record["grid_reconciliation"] = reconciliation
            record["grid_reconciliation_hash"] = _sha256_json(reconciliation)
        sidecar.write_text(json.dumps(record, indent=2))
        out[origin] = fc
    return out


def pool_scores(
    panel: Panel,
    forecasts: dict[str, Forecast],
    spark: SparkSession,
    *,
    missing_actual: MissingActual,
    restrict: dict[str, str] | None = None,
    eligible_grids: Mapping[str, str | pathlib.Path] | None = None,
) -> Score:
    """Score one adapter's forecasts across origins and pool them exactly.

    Pooling uses `Score.combine`, which recomputes ratios from summed raw counts. Averaging
    per-origin MAEs would be wrong unless every origin scored the same number of cells, and they
    do not once an adapter declines.

    Args:
        panel: Data the actuals come from.
        forecasts: origin -> `Forecast`, from `run_over_origins`.
        spark: Active session.
        missing_actual: Passed to `score_forecast`; see there, it has no default.
        restrict: origin -> common-cells Parquet, so every adapter is scored on the same cells.
        eligible_grids: origin -> canonical eligible-grid Parquet. Required for any origin where
            an absent panel row is to be classified as a sparse zero.

    Returns:
        The pooled `Score`.
    """
    parts = [score_forecast(panel, fc, spark, missing_actual=missing_actual,
                            restrict_to=(restrict or {}).get(origin),
                            eligible_grid=(eligible_grids or {}).get(origin))
             for origin, fc in sorted(forecasts.items())]
    return Score.combine(parts)


def rank(
    panel: Panel,
    manifest: EvaluationManifest,
    adapters: tuple[str, ...],
    horizon: int,
    spark: SparkSession,
    out_root: pathlib.Path,
    *,
    missing_actual: MissingActual,
    metric: str = "mae",
    reuse: bool = False,
    run_options: RunOptions | None = None,
    eligible_grids: Mapping[str, str | pathlib.Path] | None = None,
) -> Ranking:
    """Rank `adapters` on the manifest's development origins, then report on its acceptance ones.

    The order of operations is the point: development is scored and the winner picked BEFORE any
    acceptance origin is touched. Reversing it would make the reported number the maximum of N
    draws rather than an estimate.

    Args:
        panel: Data.
        manifest: Supplies the origins and the development/acceptance split, and has already
            refused a development origin whose encoder window reaches the acceptance period.
        adapters: Registry names to compare. Each must be registered.
        horizon: Periods per origin.
        spark: Active session.
        out_root: Where forecasts and cell intersections are written.
        missing_actual: See `score_forecast`; no default, on purpose.
        metric: A field from `RANKABLE`.
        reuse: Passed to `run_over_origins`; see there. Off by default.
        eligible_grids: origin -> canonical eligible-grid Parquet. Supplying this opts into a
            governed comparison: keys must be exactly ``manifest.origins``. Every finalized
            forecast key set must equal the corresponding grid before scoring; raw omissions are
            either rejected or labelled and zero-filled according to
            ``manifest.missing_forecast_policy``. ``None`` retains intersection-based behaviour
            only for explicitly exploratory comparisons.

    Returns:
        A `Ranking`.

    Raises:
        SelectionError: fewer than two adapters.
        RegistryError: an adapter is not registered.
    """
    if len(adapters) < 2:
        raise SelectionError(
            f"ranking needs at least two adapters to compare, got {adapters}. One adapter is a "
            "measurement, not a ranking")
    if panel.frequency != "D":
        raise SelectionError(
            "EvaluationManifest.horizon_days and its split windows are daily; governed ranking "
            f"cannot reinterpret them for panel frequency {panel.frequency!r}")
    if horizon != manifest.horizon_days:
        raise SelectionError(
            f"requested horizon {horizon} does not match manifest horizon "
            f"{manifest.horizon_days}; a manifest cannot govern a different forecast window")
    if panel.target_estimand != manifest.target_estimand:
        raise SelectionError(
            f"panel target_estimand {panel.target_estimand!r} does not match manifest "
            f"target_estimand {manifest.target_estimand!r}. Ranking different quantities is "
            "meaningless even when their columns share a name")
    if manifest.target_estimand == "unconstrained_demand":
        raise SelectionError(
            "governed unconstrained_demand scoring is not implemented: latent_demand_source is "
            "registered as a description but is not yet a content-bound actuals artifact read "
            "by score_forecast. Refusing prevents suppressed observed sales from selecting a "
            "latent-demand model")
    if eligible_grids is None and manifest.eligible_grid_hashes:
        raise SelectionError(
            "the manifest binds eligible_grid_hashes, so rank requires their materialised "
            "eligible_grids; intersection-based scoring would ignore the frozen contract")
    if manifest.missing_forecast_policy == "zero" and eligible_grids is None:
        raise SelectionError(
            "the manifest requires zero fallback for missing forecast keys, but no eligible "
            "grids were supplied to define those keys")
    grids = (_require_complete_grid_mapping(manifest, eligible_grids, spark)
             if eligible_grids is not None else None)
    for a in adapters:
        registry.get(a)
    # `development_origins` is never empty for a manifest that constructed: EvaluationManifest
    # already refuses one that reserved everything for acceptance. Re-checking it here would put
    # the same rule in two homes, making the extra branch unreachable.
    dev = manifest.development_origins

    dev_runs = {a: run_over_origins(a, panel, dev, horizon, spark, out_root, reuse=reuse,
                                    run_options=run_options, manifest_hash=manifest.hash,
                                    missing_forecast_policy=manifest.missing_forecast_policy,
                                    eligible_grids=grids,
                                    eligible_grid_hashes=(manifest.eligible_grid_hashes
                                                          if grids is not None else None))
                for a in adapters}
    if grids is None:
        dev_restrict, n_common = _intersections(dev_runs, dev, spark, out_root)
    else:
        dev_restrict, n_common = _exact_grid_restrictions(dev_runs, dev, grids, spark)
    development = {a: pool_scores(panel, runs, spark, missing_actual=missing_actual,
                                  restrict=dev_restrict, eligible_grids=grids)
                   for a, runs in dev_runs.items()}

    # The winner is fixed HERE, before a single acceptance origin has been read.
    chosen = _best(development, metric)

    acceptance: dict[str, Score] = {}
    acc_runs: dict[str, dict[str, Forecast]] = {}
    if manifest.acceptance_origins:
        acc = manifest.acceptance_origins
        acc_runs = {a: run_over_origins(a, panel, acc, horizon, spark, out_root, reuse=reuse,
                                        run_options=run_options, manifest_hash=manifest.hash,
                                        missing_forecast_policy=manifest.missing_forecast_policy,
                                        eligible_grids=grids,
                                        eligible_grid_hashes=(manifest.eligible_grid_hashes
                                                              if grids is not None else None))
                    for a in adapters}
        if grids is None:
            acc_restrict, _ = _intersections(acc_runs, acc, spark, out_root)
        else:
            acc_restrict, _ = _exact_grid_restrictions(acc_runs, acc, grids, spark)
        acceptance = {a: pool_scores(panel, runs, spark, missing_actual=missing_actual,
                                     restrict=acc_restrict, eligible_grids=grids)
                      for a, runs in acc_runs.items()}

    return Ranking(manifest_hash=manifest.hash, metric=metric, development=development,
                   acceptance=acceptance, chosen=chosen,
                   handicaps=registry.handicaps(adapters), n_common_cells=n_common,
                   acceptance_forecasts=acc_runs)


def _intersections(
    runs: dict[str, dict[str, Forecast]], origins: tuple[str, ...],
    spark: SparkSession, out_root: pathlib.Path,
) -> tuple[dict[str, str], int]:
    """Per origin, the cells every adapter emitted. Returns (origin -> path, smallest count)."""
    paths: dict[str, str] = {}
    counts: list[int] = []
    for origin in origins:
        fcs = [runs[a][origin] for a in sorted(runs)]
        path, n = common_cells(fcs, spark, out_root / "_common" / origin)
        paths[origin] = path
        counts.append(n)
    return paths, min(counts) if counts else 0


__all__ = ["MissingForecastPolicy", "RANKABLE", "Ranking", "RunOptions", "SelectionError",
           "common_cells", "finalize_forecast_on_grid", "pool_scores", "rank",
           "run_over_origins"]
