"""Detect the machine and size Spark to it. The system configures itself; a human may override.

WHY THIS EXISTS, stated plainly because it was a design error rather than a tuning miss. The Spark
settings were constants chosen on a 16 GB laptop and passed as defaults. Shipping that to a 34 GB
box meant the box ran at a 16 GB machine's limits -- measured: 10.4 GB used of 33.7 available,
with the JVM holding 2.3 GB, while the run took an hour per origin. **A default sized for one
machine is a ceiling for every other machine**, and no amount of documenting it fixes that, because
the failure is silent: the run works, it is just small.

So the configuration is DERIVED. The environment variables remain, and now they override a
measured default rather than substituting for the absence of one.

THE THING THAT MAKES THIS NON-OBVIOUS: in Spark LOCAL mode the JVM is also the executor, so the
heap and the Python workers come out of the SAME memory. A heap sized at 80% of the machine leaves
nothing for the workers, and the OS kills them -- which is exactly how three earlier runs died,
with no traceback, because a killed worker cannot raise. The split below is the whole point of the
module.

CONTAINERS LIE. `os.cpu_count()` reports the HOST's cores inside a container, and
`/proc/meminfo` reports the host's RAM. A cgroup limit is the real budget, and reading it wrong
fails in the dangerous direction -- sizing a 12 GB heap inside a 4 GB container gets the process
killed at the first shuffle. Both cgroup v2 and v1 are read before falling back.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache

from autoforecast._numeric import is_finite_real

__all__ = ["MachineProfile", "ResourceError", "SparkProfile", "accelerator_log_level",
           "configure_local_spark_runtime", "detect", "forecast_scoring_profile",
           "isolated_spark_profiles",
           "require_rapids_explain_enabled",
           "spark_accelerator_evidence", "spark_profile", "torch_accelerator_evidence"]

_GB = 1024 ** 3

_LOCAL_SPARK_LOG4J2 = """\
rootLogger.level = error
rootLogger.appenderRef.stderr.ref = stderr
appender.stderr.type = Console
appender.stderr.name = stderr
appender.stderr.target = SYSTEM_ERR
appender.stderr.layout.type = PatternLayout
appender.stderr.layout.pattern = %d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n%ex
"""

OS_RESERVE_GB = 2.0
"""Memory left to the operating system and everything not part of this job.

Not a guess: below about 2 GB free, the Linux OOM killer starts choosing victims, and the victim
it chooses is the largest RSS -- which is either the JVM or a Python worker holding a partition.
Either way the job dies with no traceback.
"""

HEAP_SHARE_WITH_PYTHON = 0.40
"""Share of usable memory given to the JVM heap when the workload runs PYTHON UDFs.

The rest goes to the Python workers, which hold a pandas partition each. statsforecast and
AutoGluon are in this category; the naive adapter is not -- it is native Catalyst with no Python
in the loop, which is why it never OOMed while the others did.
"""

TASKS_PER_CORE = 8
"""How many tasks each core should have to chew through.

Not a rule of thumb picked from a blog. With too few tasks the final wave leaves cores idle, and
the effect is large when each task is long: an AutoETS partition of 2,000 series takes minutes, so
one straggler holds three cores doing nothing. Eight per core makes the tail a small fraction of
the whole.
"""

MAX_SERIES_PER_PARTITION = 400
"""Hard ceiling on a Python-UDF partition, whatever the memory arithmetic allows.

MEASURED: the same AutoETS job over 30,490 series took 1,690s at 250 series per partition and over
4,960s at 2,000 -- three times slower, on the same machine, from a change the memory budget
permitted. A large partition is a long task holding a large frame, and neither helps a CPU-bound
per-series fit.
"""

HEAP_SHARE_NATIVE_ONLY = 0.70
"""Share for a workload with no Python UDFs. The JVM can take most of the machine because nothing
else needs it -- only a small driver-side Python process remains.
"""

_SPARK_ACCELERATOR_CONFIG_KEYS = (
    "spark.plugins",
    "spark.jars.packages",
    "spark.rapids.sql.enabled",
    "spark.rapids.sql.explain",
    "spark.driver.resource.gpu.amount",
    "spark.driver.resource.gpu.discoveryScript",
    "spark.driver.resource.gpu.vendor",
    "spark.executor.resource.gpu.amount",
    "spark.executor.resource.gpu.discoveryScript",
    "spark.executor.resource.gpu.vendor",
    "spark.task.resource.gpu.amount",
)
"""Startup configuration recorded for GPU-capable Spark deployments.

The allowlist is deliberately closed: dumping an entire SparkConf can copy credentials and
unrelated infrastructure details into experiment artifacts. These are the NVIDIA SQL-plugin and
Spark generic-resource settings needed to explain how GPU work was requested and discovered.
"""


class ResourceError(ValueError):
    """The machine cannot support the requested configuration."""


@lru_cache(maxsize=1)
def _local_spark_log4j2_path() -> pathlib.Path:
    """Materialize one process-private startup logger config for the local Spark JVM."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="autoforecast-spark-log-"))
    path = root / "log4j2.properties"
    path.write_text(_LOCAL_SPARK_LOG4J2, encoding="utf-8")
    path.chmod(0o600)
    return path


def configure_local_spark_runtime() -> dict[str, str]:
    """Bind a local-only Spark JVM to a resolvable loopback address.

    Spark resolves a host before its driver configuration can be read. A malformed or
    non-resolvable macOS hostname therefore fails during ``SparkContext`` construction, before
    data admission or the runner can report a governed incident. These runners use ``local[N]``
    exclusively, so loopback is the correct isolated address for both the initial resolution and
    the driver bind/advertise settings.
    """
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"
    return {
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.host": "127.0.0.1",
        # macOS does not ship Hadoop's optional JNI library.  Logging ERROR from JVM startup
        # removes that expected NativeCodeLoader warning while retaining actual Spark failures.
        "spark.driver.extraJavaOptions": (
            f"-Dlog4j2.configurationFile={_local_spark_log4j2_path().as_uri()}"
        ),
    }


@dataclass(frozen=True)
class MachineProfile:
    """What the machine actually has, after container limits are taken into account.

    Attributes:
        total_ram_bytes: The real budget -- a cgroup limit where one exists, the physical total
            otherwise.
        cores: Usable cores, likewise after any CPU quota.
        source: Where the numbers came from, so a surprising configuration can be traced rather
            than argued about.
    """

    total_ram_bytes: int
    cores: int
    source: str

    def __post_init__(self) -> None:
        """Refuse an impossible machine before deriving apparently valid Spark settings."""
        if (not isinstance(self.total_ram_bytes, int)
                or isinstance(self.total_ram_bytes, bool) or self.total_ram_bytes < 1):
            raise ResourceError(
                f"total_ram_bytes must be a positive integer, got {self.total_ram_bytes!r}")
        if not isinstance(self.cores, int) or isinstance(self.cores, bool) or self.cores < 1:
            raise ResourceError(f"machine cores must be a positive integer, got {self.cores!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ResourceError("machine resource source must be a non-blank string")

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / _GB

    def __str__(self) -> str:
        return f"{self.total_ram_gb:.1f} GB, {self.cores} cores (from {self.source})"


def _cgroup_memory_limit() -> int | None:
    """Container memory limit in bytes, or None outside a limited container.

    Both cgroup versions, because reading only v2 silently falls back to the host's RAM on an
    older runtime -- and sizing a heap to the host inside a smaller container is the failure that
    kills the process at the first shuffle rather than at start-up, where it would be obvious.
    """
    for path, unlimited in (("/sys/fs/cgroup/memory.max", "max"),                       # v2
                            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None)):     # v1
        p = pathlib.Path(path)
        if not p.exists():
            continue
        raw = p.read_text().strip()
        if raw == unlimited:
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1 reports an enormous sentinel rather than a word when unlimited.
        if value <= 0 or value > 2 ** 60:
            return None
        return value
    return None


def _cgroup_cpu_limit() -> int | None:
    """Container CPU quota in whole cores, or None when unlimited."""
    v2 = pathlib.Path("/sys/fs/cgroup/cpu.max")
    if v2.exists():
        parts = v2.read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if period > 0:
                return max(1, quota // period)
    quota_p = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_p = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_p.exists() and period_p.exists():
        quota, period = int(quota_p.read_text()), int(period_p.read_text())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    return None


def _physical_memory() -> tuple[int, str]:
    """Read physical RAM, falling back when an OS probe is unavailable or denied."""
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
                if total > 0:
                    return total, "/proc/meminfo"
    except (OSError, ValueError, IndexError):
        pass
    if shutil.which("sysctl"):  # macOS; restricted processes may be denied this query.
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                total = int(out.stdout.strip())
                if total > 0:
                    return total, "sysctl hw.memsize"
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and pages > 0:
            return page_size * pages, "os.sysconf"
    except (AttributeError, OSError, ValueError):
        pass
    try:
        import psutil

        total = psutil.virtual_memory().total
        if type(total) is int and total > 0:
            return total, "psutil.virtual_memory"
    except (ImportError, OSError, ValueError, NotImplementedError):
        pass
    raise ResourceError(
        "cannot determine total physical memory from /proc/meminfo, sysctl, os.sysconf, or "
        "psutil. Allow a system memory probe or pass a verified "
        "MachineProfile(total_ram_bytes=..., cores=..., source=...) to spark_profile(machine=...). "
        "FORECAST_SPARK_MEMORY sets only the JVM heap; it cannot supply the total memory budget"
    )


def detect() -> MachineProfile:
    """Measure the machine, preferring a container's limits over the host's totals."""
    physical, source = _physical_memory()
    limit = _cgroup_memory_limit()
    ram, ram_source = (limit, "cgroup") if limit and limit < physical else (physical, source)

    host_cores = os.cpu_count() or 1
    quota = _cgroup_cpu_limit()
    cores = min(host_cores, quota) if quota else host_cores
    return MachineProfile(total_ram_bytes=ram, cores=max(1, cores),
                          source=f"{ram_source}, cpu={'cgroup' if quota else 'os'}")


@dataclass(frozen=True)
class SparkProfile:
    """Spark settings derived from a machine, ready to apply to a session builder.

    Attributes:
        cores: `local[N]`.
        driver_memory: JVM heap, e.g. `"12g"`.
        shuffle_partitions: Partition count for shuffles.
        worker_reuse: Whether a Python worker survives between tasks.
        series_per_partition: How many series one Python worker fits at a time.
        python_worker_budget_gb: What is LEFT for the Python workers after the heap. Carried so
            the number that actually decides whether a run survives is visible rather than
            implied.
    """

    cores: int
    driver_memory: str
    shuffle_partitions: int
    worker_reuse: bool
    series_per_partition: int
    python_worker_budget_gb: float

    def __post_init__(self) -> None:
        if isinstance(self.cores, bool) or not isinstance(self.cores, int) or self.cores < 1:
            raise ResourceError(f"Spark cores must be a positive integer, got {self.cores!r}")
        if (not isinstance(self.driver_memory, str)
                or not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?[gGmM]", self.driver_memory)):
            raise ResourceError(
                f"Spark driver_memory must be a positive g/m quantity, got {self.driver_memory!r}")
        _memory_gb(self.driver_memory)
        for name, value in (("shuffle_partitions", self.shuffle_partitions),
                            ("series_per_partition", self.series_per_partition)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ResourceError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.worker_reuse, bool):
            raise ResourceError(f"worker_reuse must be boolean, got {self.worker_reuse!r}")
        if (not is_finite_real(self.python_worker_budget_gb)
                or self.python_worker_budget_gb <= 0):
            raise ResourceError(
                "python_worker_budget_gb must be a finite positive number, got "
                f"{self.python_worker_budget_gb!r}")

    def as_config(self) -> dict[str, str]:
        """The Spark options, ready for `SparkSession.builder.config`."""
        return {"spark.driver.memory": self.driver_memory,
                "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
                "spark.python.worker.reuse": "true" if self.worker_reuse else "false"}

    def __str__(self) -> str:
        return (f"local[{self.cores}] heap={self.driver_memory} "
                f"shuffle={self.shuffle_partitions} reuse={self.worker_reuse} "
                f"series/partition={self.series_per_partition} "
                f"(python workers get {self.python_worker_budget_gb:.1f} GB)")


def _memory_gb(value: str) -> float:
    """Parse the strict subset of Spark memory strings accepted by ``SparkProfile``."""
    match = re.fullmatch(r"([1-9][0-9]*(?:\.[0-9]+)?)([gGmM])", value)
    if not match:
        raise ResourceError(
            f"FORECAST_SPARK_MEMORY must be a positive g/m quantity, got {value!r}")
    amount = float(match.group(1))
    if not math.isfinite(amount):
        raise ResourceError(
            f"FORECAST_SPARK_MEMORY must fit a finite binary64 g/m quantity, got {value!r}")
    return amount if match.group(2).lower() == "g" else amount / 1024.0


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ResourceError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ResourceError(f"{name} must be a positive integer, got {raw!r}")
    return value


def spark_profile(machine: MachineProfile | None = None, *, python_udfs: bool = True,
                  honor_env: bool = True, os_reserve_gb: float = OS_RESERVE_GB,
                  ) -> SparkProfile:
    """Derive Spark settings from the machine. Environment variables override any of them.

    Args:
        machine: Measured profile. Detected when omitted.
        python_udfs: Whether this workload runs Python UDFs. **This is the decision that matters.**
            With them, the heap gets 40% and the workers get the rest; without them the heap can
            take 70% because only a small driver-side Python process remains. The naive adapter is
            native Catalyst and never OOMed; the statsforecast and AutoGluon adapters run a Python
            UDF per series and killed three runs.
        os_reserve_gb: Memory to leave outside Spark and Python. Composite allocators subtract
            this once before splitting the remaining bytes and pass zero to each child profile.

    Returns:
        A `SparkProfile`.

    Raises:
        ResourceError: the machine has less usable memory than the OS reserve, so there is
            nothing to allocate.
    """
    machine = machine or detect()
    if not is_finite_real(os_reserve_gb) or float(os_reserve_gb) < 0:
        raise ResourceError("os_reserve_gb must be a finite non-negative number")
    usable_gb = machine.total_ram_gb - os_reserve_gb
    if usable_gb <= 1.0:
        raise ResourceError(
            f"{machine} leaves {usable_gb:.1f} GB after the {os_reserve_gb} GB OS reserve. "
            "There is nothing to allocate; this job needs a larger machine")

    share = HEAP_SHARE_WITH_PYTHON if python_udfs else HEAP_SHARE_NATIVE_ONLY
    derived_heap_gb = max(1, int(usable_gb * share))

    cores = _env_positive_int("FORECAST_SPARK_CORES", machine.cores) \
        if honor_env else machine.cores
    if cores > machine.cores:
        raise ResourceError(
            f"FORECAST_SPARK_CORES={cores} exceeds the {machine.cores} detected usable cores")
    heap = os.environ.get("FORECAST_SPARK_MEMORY", f"{derived_heap_gb}g") \
        if honor_env else f"{derived_heap_gb}g"
    actual_heap_gb = _memory_gb(heap)
    if actual_heap_gb >= usable_gb:
        raise ResourceError(
            f"Spark heap {heap} consumes {actual_heap_gb:.1f} GB of {usable_gb:.1f} GB usable "
            "memory, leaving no room for the Python process/workers")
    worker_gb = usable_gb - actual_heap_gb
    # Enough partitions to keep every core busy through a shuffle, without so many that task
    # overhead dominates. Four per core is the usual rule of thumb and nothing here contradicts it.
    partitions = (_env_positive_int("FORECAST_SHUFFLE_PARTITIONS", max(8, cores * 16))
                  if honor_env else max(8, cores * 16))

    # Reuse is worth having WHENEVER the workers have room, and they nearly always do once the
    # heap is sized correctly. Turning it off costs a fresh interpreter plus a numpy and
    # statsforecast import per task -- measured at 4.91s, against tasks that take tens of seconds.
    reuse_default = worker_gb >= 2.0
    if honor_env:
        raw_reuse = os.environ.get(
            "FORECAST_WORKER_REUSE", "true" if reuse_default else "false").lower()
        if raw_reuse not in ("true", "false"):
            raise ResourceError(
                f"FORECAST_WORKER_REUSE must be true or false, got {raw_reuse!r}")
        reuse = raw_reuse == "true"
    else:
        reuse = reuse_default

    # Limit partition size by both worker memory and parallelism. The adapter can narrow
    # this ceiling once the actual series count is known.
    bytes_per_series = 2000 * 8 * 4          # ~2000 days, float64, x4 for frame overhead
    per_worker_bytes = (worker_gb * _GB) / max(cores, 1)
    memory_ceiling = int(per_worker_bytes / bytes_per_series)
    derived_series = max(50, min(MAX_SERIES_PER_PARTITION, memory_ceiling))
    series_per_partition = (_env_positive_int("FORECAST_SERIES_PER_PARTITION", derived_series)
        if honor_env else derived_series)

    return SparkProfile(cores=cores, driver_memory=heap, shuffle_partitions=partitions,
                        worker_reuse=reuse, series_per_partition=series_per_partition,
                        python_worker_budget_gb=worker_gb)


def forecast_scoring_profile(
    metric: str, adapters: tuple[str, ...], machine: MachineProfile | None = None
) -> SparkProfile:
    """Use the ordinary native scoring profile."""
    return (
        spark_profile(python_udfs=False)
        if machine is None
        else spark_profile(machine, python_udfs=False)
    )


def isolated_spark_profiles(
    machine: MachineProfile | None = None,
) -> tuple[SparkProfile, SparkProfile]:
    """Disjoint parent/worker Spark budgets for process-isolated adapter evaluation.

    The scoring parent and one adapter worker are alive at the same time. Giving both a profile
    derived from the whole machine double-counts RAM and cores: on the measured 16 GB host that
    meant two 5 GB JVMs plus nine GB of claimed Python-worker room. Split the physical budget
    first, then derive each profile inside its own share. Isolation workers run serially, so one
    child share is sufficient.

    Generic FORECAST_SPARK_* overrides are deliberately not applied twice. A caller needing a
    bespoke split should construct and pass two explicit SparkProfile values.
    """
    machine = machine or detect()
    if machine.cores < 2:
        raise ResourceError(
            "process-isolated comparison needs at least 2 usable cores so the live parent and "
            f"worker do not both claim the same core; detected {machine.cores}")
    # Reserve for the OS exactly once. Each sub-profile then divides only its already-reserved
    # share between the JVM and Python instead of subtracting the same reserve a second time.
    usable_bytes = machine.total_ram_bytes - int(OS_RESERVE_GB * _GB)
    minimum_share_bytes = 2 * _GB
    if usable_bytes <= 2 * minimum_share_bytes:
        raise ResourceError(
            "process-isolated comparison cannot derive safe concurrent parent and worker "
            f"budgets from {machine}: no more than {usable_bytes / _GB:.1f} GB remains after "
            f"the {OS_RESERVE_GB} GB OS reserve"
        )
    parent_bytes = max(minimum_share_bytes, int(usable_bytes * 0.30))
    parent_bytes = min(parent_bytes, usable_bytes - minimum_share_bytes)
    worker_bytes = usable_bytes - parent_bytes
    parent_cores = min(2, machine.cores)
    worker_cores = max(1, machine.cores - parent_cores)
    parent_machine = MachineProfile(parent_bytes, parent_cores,
                                    f"isolated parent share of {machine.source}")
    worker_machine = MachineProfile(worker_bytes, worker_cores,
                                    f"isolated worker share of {machine.source}")
    try:
        parent = spark_profile(
            parent_machine, python_udfs=False, honor_env=False, os_reserve_gb=0.0
        )
        worker = spark_profile(
            worker_machine, python_udfs=True, honor_env=False, os_reserve_gb=0.0
        )
    except ResourceError as error:
        raise ResourceError(
            "process-isolated comparison cannot derive safe concurrent parent and worker "
            f"budgets from {machine}: {error}"
        ) from error
    return parent, worker


def spark_accelerator_evidence(spark) -> dict[str, object]:
    """Read accelerator startup evidence from an existing Spark session without mutating it.

    ``nvidia_rapids_configured`` means the NVIDIA SQL plugin was requested at startup and was not
    explicitly disabled. It does not claim that a particular physical plan ran on a GPU; that
    requires plan/runtime evidence. Missing or unreadable state fails closed to
    ``cpu_or_unconfigured``. The output has a fixed, JSON-safe shape so run artifacts are
    comparable across CPU and GPU hosts.
    """
    configs: dict[str, str | None] = {
        key: None for key in _SPARK_ACCELERATOR_CONFIG_KEYS
    }
    probe_errors: list[str] = []
    failed_config_keys: set[str] = set()

    try:
        context = spark.sparkContext
    except Exception:  # Py4J/session teardown failures are evidence, not runner failures.
        context = None
        probe_errors.append("spark_context:unreadable")

    if context is not None:
        conf_get = None
        try:
            spark_conf = context.getConf()
        except Exception:
            spark_conf = None
            probe_errors.append("spark_conf:unreadable")
        else:
            try:
                conf_get = spark_conf.get
            except Exception:
                conf_get = None
            if not callable(conf_get):
                probe_errors.append("spark_conf:malformed")
        if spark_conf is not None and callable(conf_get):
            for key in _SPARK_ACCELERATOR_CONFIG_KEYS:
                try:
                    value = conf_get(key, None)
                except Exception:
                    failed_config_keys.add(key)
                    probe_errors.append(f"spark_conf:{key}:unreadable")
                    continue
                if value is None:
                    continue
                if type(value) is not str or not value.strip():
                    failed_config_keys.add(key)
                    probe_errors.append(f"spark_conf:{key}:malformed")
                    continue
                configs[key] = value

    driver_gpu_addresses: list[str] = []
    if context is not None:
        try:
            assigned = context.resources
        except Exception:
            probe_errors.append("driver_resources:unreadable")
        else:
            if not isinstance(assigned, dict):
                probe_errors.append("driver_resources:malformed")
            else:
                gpu = assigned.get("gpu")
                if gpu is not None:
                    try:
                        addresses = gpu.addresses
                    except Exception:
                        probe_errors.append("driver_gpu_addresses:unreadable")
                    else:
                        if (not isinstance(addresses, list)
                                or any(type(address) is not str or not address.strip()
                                       for address in addresses)):
                            probe_errors.append("driver_gpu_addresses:malformed")
                        else:
                            driver_gpu_addresses = sorted(
                                {address.strip() for address in addresses})

    plugins = configs["spark.plugins"]
    rapids_enabled = configs["spark.rapids.sql.enabled"]
    plugin_names = ({name.strip() for name in plugins.split(",")}
                    if plugins is not None else set())
    rapids_not_disabled = (
        rapids_enabled is None or rapids_enabled.strip().lower() == "true"
    )
    required_readable = not {
        "spark.plugins", "spark.rapids.sql.enabled"
    }.intersection(failed_config_keys)
    sql_acceleration = (
        "nvidia_rapids_configured"
        if ("com.nvidia.spark.SQLPlugin" in plugin_names
            and rapids_not_disabled and required_readable)
        else "cpu_or_unconfigured"
    )
    return {
        "spark_sql_acceleration": sql_acceleration,
        "spark_configs": configs,
        "driver_gpu_addresses": driver_gpu_addresses,
        "probe_errors": probe_errors,
    }


def accelerator_log_level(evidence: object) -> str:
    """Keep RAPIDS placement evidence visible without making CPU runs noisy."""
    if not isinstance(evidence, dict):
        return "ERROR"
    configs = evidence.get("spark_configs")
    if (evidence.get("spark_sql_acceleration") != "nvidia_rapids_configured"
            or not isinstance(configs, dict)):
        return "ERROR"
    return "WARN"


def require_rapids_explain_enabled(evidence: object) -> None:
    """Require the RAPIDS mode that reports both placed and fallback operators."""
    if not isinstance(evidence, dict) or (
            evidence.get("spark_sql_acceleration") != "nvidia_rapids_configured"):
        return
    configs = evidence.get("spark_configs")
    if not isinstance(configs, dict):
        raise ResourceError("RAPIDS configuration evidence is malformed")
    explain = configs.get("spark.rapids.sql.explain")
    if not isinstance(explain, str) or explain.strip().upper() != "ALL":
        raise ResourceError(
            "RAPIDS-configured evidence runs require spark.rapids.sql.explain=ALL so the "
            "captured log proves GPU placement or an explicit CPU fallback")


def torch_accelerator_evidence(torch_module) -> dict[str, object]:
    """Report Torch accelerator state using an injected module and the model's device rule.

    The adapter selects ``cuda`` when ``torch.cuda.is_available()`` is true and ``cpu`` otherwise;
    MPS is reported but deliberately does not change that existing rule. Torch remains an
    optional dependency of this resource module because the caller supplies the imported module.
    """
    probe_errors: list[str] = []
    try:
        cuda = torch_module.cuda
    except Exception:
        cuda = None
    try:
        if cuda is None:
            raise AttributeError("torch.cuda is unavailable")
        raw_cuda_available = cuda.is_available()
    except Exception:
        raw_cuda_available = False
        probe_errors.append("cuda_available:unreadable")
    if type(raw_cuda_available) is not bool:
        probe_errors.append("cuda_available:malformed")
        cuda_available = False
    else:
        cuda_available = raw_cuda_available

    try:
        if cuda is None:
            raise AttributeError("torch.cuda is unavailable")
        raw_cuda_count = cuda.device_count()
    except Exception:
        raw_cuda_count = 0
        probe_errors.append("cuda_device_count:unreadable")
    if type(raw_cuda_count) is not int or raw_cuda_count < 0:
        probe_errors.append("cuda_device_count:malformed")
        cuda_device_count = 0
    else:
        cuda_device_count = raw_cuda_count

    try:
        raw_cuda_version = torch_module.version.cuda
    except Exception:
        raw_cuda_version = None
        probe_errors.append("cuda_version:unreadable")
    if (raw_cuda_version is not None
            and (type(raw_cuda_version) is not str or not raw_cuda_version.strip())):
        probe_errors.append("cuda_version:malformed")
        cuda_version = None
    else:
        cuda_version = raw_cuda_version

    cuda_device_name = None
    if cuda_available and cuda_device_count > 0:
        try:
            if cuda is None:
                raise AttributeError("torch.cuda is unavailable")
            raw_cuda_name = cuda.get_device_name(0)
        except Exception:
            raw_cuda_name = None
            probe_errors.append("cuda_device_name:unreadable")
        if (raw_cuda_name is not None
                and (type(raw_cuda_name) is not str or not raw_cuda_name.strip())):
            probe_errors.append("cuda_device_name:malformed")
        else:
            cuda_device_name = raw_cuda_name

    try:
        mps = torch_module.backends.mps
        raw_mps_available = mps.is_available()
    except Exception:
        raw_mps_available = False
        probe_errors.append("mps_available:unreadable")
    if type(raw_mps_available) is not bool:
        probe_errors.append("mps_available:malformed")
        mps_available = False
    else:
        mps_available = raw_mps_available

    return {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_version": cuda_version,
        "cuda_device_name": cuda_device_name,
        "mps_available": mps_available,
        "selected_device": "cuda" if cuda_available else "cpu",
        "probe_errors": probe_errors,
    }


def _require_fresh_spark_runtime(session_type, context_type) -> None:
    """Refuse every process-global runtime that ``SparkSession.builder`` can silently reuse."""
    existing = []
    get_active = getattr(session_type, "getActiveSession", lambda: None)
    if get_active() is not None:
        existing.append("thread-active session")
    if getattr(session_type, "_instantiatedSession", None) is not None:
        existing.append("inactive default session")
    if getattr(context_type, "_active_spark_context", None) is not None:
        existing.append("bare SparkContext")
    if existing:
        raise RuntimeError(
            "this run requires a fresh Spark runtime; detected " + ", ".join(existing)
            + ". SparkSession.getOrCreate() can reuse its immutable JVM settings")


def _spark_creation_locks(session_type, context_type):
    """Return PySpark's process-global creation locks in its own acquisition order."""
    builder_type = getattr(session_type, "Builder", None)
    session_lock = getattr(builder_type, "_lock", None)
    context_lock = getattr(context_type, "_lock", None)
    if session_lock is None or context_lock is None:
        raise ResourceError(
            "this PySpark runtime exposes no atomic session/context creation locks; refusing "
            "ambiguous Spark ownership")
    return session_lock, context_lock


def _acquire_fresh_spark_runtime(
    session_type,
    context_type,
    *,
    master: str,
    app_name: str,
    config: dict[str, str],
):
    """Atomically create and return a Spark runtime that the caller exclusively owns.

    PySpark's builder silently reuses process-global sessions and contexts. Checking for one and
    then configuring a builder outside PySpark's own locks lets another thread install its runtime
    between those actions; the nominal creator can then mutate and later stop that competitor.
    Holding both re-entrant creation locks, in PySpark's order, through the repeated absence check,
    all builder configuration, and ``getOrCreate`` makes the returned runtime caller-owned.
    """
    if type(master) is not str or not master.strip():
        raise ResourceError("fresh Spark master must be a non-blank string")
    if type(app_name) is not str or not app_name.strip():
        raise ResourceError("fresh Spark app_name must be a non-blank string")
    if not isinstance(config, dict) or any(
            type(key) is not str or not key.strip() or type(value) is not str
            for key, value in config.items()):
        raise ResourceError("fresh Spark config must map non-blank string keys to strings")
    session_lock, context_lock = _spark_creation_locks(session_type, context_type)
    with session_lock:
        with context_lock:
            _require_fresh_spark_runtime(session_type, context_type)
            builder = session_type.builder.master(master).appName(app_name)
            for key, value in config.items():
                builder = builder.config(key, value)
            return builder.getOrCreate()


def acquire_local_runtime(session_type, context_type, *, profile, app_name: str):
    """Acquire an owned local runtime; refusal leaves caller globals and environment alone."""
    session_lock, context_lock = _spark_creation_locks(session_type, context_type)
    with session_lock, context_lock:
        _require_fresh_spark_runtime(session_type, context_type)
        config = {
            **profile.as_config(),
            **configure_local_spark_runtime(),
            "spark.ui.showConsoleProgress": "false",
        }
        os.environ["PYSPARK_PYTHON"] = sys.executable
        return _acquire_fresh_spark_runtime(
            session_type, context_type, master=f"local[{profile.cores}]",
            app_name=app_name, config=config,
        )


def _acquire_embedded_spark_runtime(
    session_type,
    context_type,
    *,
    app_name: str,
    profile_factory,
):
    """Atomically reuse caller Spark unchanged or create one runtime owned by this command.

    PySpark serialises classic session/context creation with these two process-global re-entrant
    locks, in this order. Holding both across state inspection and ``getOrCreate`` closes the
    check-then-act window: a competing conforming thread cannot create a caller-owned runtime
    after we decide ownership. Existing sessions are returned directly, without passing builder
    options that PySpark would apply to (and therefore mutate on) the caller's session.
    """
    if type(app_name) is not str or not app_name.strip():
        raise ResourceError("embedded Spark app_name must be a non-blank string")
    session_lock, context_lock = _spark_creation_locks(session_type, context_type)
    with session_lock:
        with context_lock:
            active = getattr(session_type, "getActiveSession", lambda: None)()
            default = getattr(session_type, "_instantiatedSession", None)
            if active is not None or default is not None:
                return (active if active is not None else default), False
            if getattr(context_type, "_active_spark_context", None) is not None:
                # Create only a SQL facade over the caller's context. Supplying even an app name
                # here would become a builder option and can mutate a reused session.
                return session_type.builder.getOrCreate(), False

            profile = profile_factory()
            if not isinstance(profile, SparkProfile):
                raise ResourceError("embedded Spark profile factory returned no SparkProfile")
            builder = session_type.builder.master(f"local[{profile.cores}]").appName(app_name)
            for key, value in profile.as_config().items():
                builder = builder.config(key, value)
            spark = builder.config("spark.ui.showConsoleProgress", "false").getOrCreate()
            return spark, True


def describe() -> str:
    """One line for a run log: the machine and what was derived from it.

    Printed by every runner. A run whose configuration cannot be read back from its own log is a
    run whose numbers cannot be explained later.
    """
    machine = detect()
    return f"machine: {machine}\nspark:   {spark_profile(machine)}"


def sanity_check(profile: SparkProfile, machine: MachineProfile | None = None) -> list[str]:
    """Warnings a caller should print rather than swallow. Empty means nothing looks wrong.

    Exists because the failure mode this module addresses was SILENT: a run configured for a
    quarter of the machine works perfectly and is simply slow, and nobody notices for two hours.
    """
    machine = machine or detect()
    notes = []
    heap_gb = float(profile.driver_memory.rstrip("gG"))
    if heap_gb + profile.python_worker_budget_gb < machine.total_ram_gb * 0.5:
        notes.append(
            f"using {heap_gb + profile.python_worker_budget_gb:.1f} GB of "
            f"{machine.total_ram_gb:.1f} GB -- under half the machine. Check the overrides")
    if profile.cores < machine.cores:
        notes.append(f"using {profile.cores} of {machine.cores} cores")
    if not profile.worker_reuse:
        notes.append("worker reuse is OFF: every task pays a fresh interpreter and its imports "
                     "(measured 4.91s for numpy + statsforecast)")
    return notes
