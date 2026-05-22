"""Tests for machine detection and Spark sizing.

The failure this module fixes was SILENT: a run configured for a quarter of the machine works
perfectly and is merely slow, and nobody notices for two hours. So the tests are mostly about the
two ways the sizing can be wrong without anything crashing -- under-using a big machine, and
over-committing inside a container that is smaller than its host.

Run: python3 autoforecast/test_resources.py
"""

from __future__ import annotations

import importlib
import os
import pathlib
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import autoforecast
from autoforecast.resources import (
    HEAP_SHARE_NATIVE_ONLY,
    HEAP_SHARE_WITH_PYTHON,
    OS_RESERVE_GB,
    MachineProfile,
    ResourceError,
    SparkProfile,
    accelerator_log_level,
    configure_local_spark_runtime,
    describe,
    detect,
    forecast_scoring_profile,
    isolated_spark_profiles,
    require_rapids_explain_enabled,
    sanity_check,
    spark_accelerator_evidence,
    spark_profile,
    torch_accelerator_evidence,
)

_GB = 1024 ** 3


def test_package_initialization_does_not_mutate_local_spark_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPARK_LOCAL_IP", raising=False)
    monkeypatch.delenv("SPARK_LOCAL_HOSTNAME", raising=False)

    importlib.reload(autoforecast)

    assert "SPARK_LOCAL_IP" not in os.environ
    assert "SPARK_LOCAL_HOSTNAME" not in os.environ


def test_local_spark_runtime_forces_loopback_before_jvm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARK_LOCAL_IP", "unresolvable-host")
    monkeypatch.setenv("SPARK_LOCAL_HOSTNAME", "unresolvable-host")

    config = configure_local_spark_runtime()
    assert config["spark.driver.bindAddress"] == "127.0.0.1"
    assert config["spark.driver.host"] == "127.0.0.1"
    option = config["spark.driver.extraJavaOptions"]
    assert option.startswith("-Dlog4j2.configurationFile=file:")
    log_config = pathlib.Path(option.split("=", 1)[1].removeprefix("file://"))
    assert log_config.read_text(encoding="utf-8").startswith("rootLogger.level = error")
    assert os.environ["SPARK_LOCAL_IP"] == "127.0.0.1"
    assert os.environ["SPARK_LOCAL_HOSTNAME"] == "localhost"


def machine(gb: float, cores: int) -> MachineProfile:
    return MachineProfile(total_ram_bytes=int(gb * _GB), cores=cores, source="test")


def clear_env() -> None:
    for k in ("FORECAST_SPARK_CORES", "FORECAST_SPARK_MEMORY", "FORECAST_SHUFFLE_PARTITIONS",
              "FORECAST_WORKER_REUSE", "FORECAST_SERIES_PER_PARTITION"):
        os.environ.pop(k, None)


def test_spark_accelerator_evidence_records_only_the_fixed_startup_allowlist() -> None:
    values = {
        "spark.plugins": "example.Plugin, com.nvidia.spark.SQLPlugin",
        "spark.jars.packages": "com.nvidia:rapids-4-spark_2.13:VERSION",
        "spark.rapids.sql.enabled": "TRUE",
        "spark.rapids.sql.explain": "ALL",
        "spark.driver.resource.gpu.amount": "1",
        "spark.driver.resource.gpu.discoveryScript": "/opt/spark/get-gpus.sh",
        "spark.driver.resource.gpu.vendor": "nvidia.com",
        "spark.executor.resource.gpu.amount": "2",
        "spark.executor.resource.gpu.discoveryScript": "/opt/spark/get-gpus.sh",
        "spark.executor.resource.gpu.vendor": "nvidia.com",
        "spark.task.resource.gpu.amount": "0.5",
        "spark.hadoop.fs.s3a.secret.key": "must-not-enter-evidence",
    }
    expected_configs = {
        "spark.plugins": values["spark.plugins"],
        "spark.jars.packages": values["spark.jars.packages"],
        "spark.rapids.sql.enabled": "TRUE",
        "spark.rapids.sql.explain": "ALL",
        "spark.driver.resource.gpu.amount": "1",
        "spark.driver.resource.gpu.discoveryScript": "/opt/spark/get-gpus.sh",
        "spark.driver.resource.gpu.vendor": "nvidia.com",
        "spark.executor.resource.gpu.amount": "2",
        "spark.executor.resource.gpu.discoveryScript": "/opt/spark/get-gpus.sh",
        "spark.executor.resource.gpu.vendor": "nvidia.com",
        "spark.task.resource.gpu.amount": "0.5",
    }

    class ReadOnlyConf:
        def __init__(self) -> None:
            self.read: list[str] = []

        def get(self, key, default=None):
            self.read.append(key)
            return values.get(key, default)

        def set(self, *_args):
            raise AssertionError("accelerator evidence mutated the Spark startup configuration")

    conf = ReadOnlyConf()
    spark = SimpleNamespace(sparkContext=SimpleNamespace(
        getConf=lambda: conf,
        resources={"gpu": SimpleNamespace(addresses=["GPU-b", " GPU-a ", "GPU-b"])},
    ))
    assert spark_accelerator_evidence(spark) == {
        "spark_sql_acceleration": "nvidia_rapids_configured",
        "spark_configs": expected_configs,
        "driver_gpu_addresses": ["GPU-a", "GPU-b"],
        "probe_errors": [],
    }
    assert conf.read == list(expected_configs)


def test_spark_accelerator_evidence_has_an_explicit_cpu_fallback_and_changes_no_profile() -> None:
    spark = SimpleNamespace(sparkContext=SimpleNamespace(
        getConf=lambda: SimpleNamespace(get=lambda _key, default=None: default),
        resources={},
    ))
    evidence = spark_accelerator_evidence(spark)
    assert evidence == {
        "spark_sql_acceleration": "cpu_or_unconfigured",
        "spark_configs": {
            "spark.plugins": None,
            "spark.jars.packages": None,
            "spark.rapids.sql.enabled": None,
            "spark.rapids.sql.explain": None,
            "spark.driver.resource.gpu.amount": None,
            "spark.driver.resource.gpu.discoveryScript": None,
            "spark.driver.resource.gpu.vendor": None,
            "spark.executor.resource.gpu.amount": None,
            "spark.executor.resource.gpu.discoveryScript": None,
            "spark.executor.resource.gpu.vendor": None,
            "spark.task.resource.gpu.amount": None,
        },
        "driver_gpu_addresses": [],
        "probe_errors": [],
    }
    profile_config = SparkProfile(
        cores=2, driver_memory="2g", shuffle_partitions=4, worker_reuse=True,
        series_per_partition=50, python_worker_budget_gb=2.0,
    ).as_config()
    assert not any("gpu" in key.lower() or "rapids" in key.lower() for key in profile_config)

    disabled = SimpleNamespace(sparkContext=SimpleNamespace(
        getConf=lambda: SimpleNamespace(get=lambda key, default=None: {
            "spark.plugins": "com.nvidia.spark.SQLPlugin",
            "spark.rapids.sql.enabled": "false",
        }.get(key, default)),
        resources={"gpu": SimpleNamespace(addresses=["0"])},
    ))
    assert spark_accelerator_evidence(disabled)["spark_sql_acceleration"] == \
        "cpu_or_unconfigured"

    default_enabled = SimpleNamespace(sparkContext=SimpleNamespace(
        getConf=lambda: SimpleNamespace(get=lambda key, default=None: {
            "spark.plugins": "com.nvidia.spark.SQLPlugin",
        }.get(key, default)),
        resources={},
    ))
    assert spark_accelerator_evidence(default_enabled)["spark_sql_acceleration"] == \
        "nvidia_rapids_configured"


def test_accelerator_log_level_preserves_rapids_explanations_only_when_configured() -> None:
    cpu = {"spark_sql_acceleration": "cpu_or_unconfigured", "spark_configs": {}}
    configured = {
        "spark_sql_acceleration": "nvidia_rapids_configured",
        "spark_configs": {"spark.rapids.sql.explain": "ALL"},
    }
    default_explain = {
        "spark_sql_acceleration": "nvidia_rapids_configured",
        "spark_configs": {"spark.rapids.sql.explain": None},
    }
    disabled = {
        "spark_sql_acceleration": "nvidia_rapids_configured",
        "spark_configs": {"spark.rapids.sql.explain": " none "},
    }
    assert accelerator_log_level(cpu) == "ERROR"
    assert accelerator_log_level(configured) == "WARN"
    assert accelerator_log_level(default_explain) == "WARN"
    assert accelerator_log_level(disabled) == "WARN"
    assert accelerator_log_level(None) == "ERROR"
    require_rapids_explain_enabled(cpu)
    require_rapids_explain_enabled(configured)
    for rejected_mode in (
        None, "", "NOT_ON_GPU", "garbage",
        "none", "nonE", "noNe", "noNE", "nOne", "nOnE", "nONe", "nONE",
        "None", "NonE", "NoNe", "NoNE", "NOne", "NOnE", "NONe", "NONE",
    ):
        value = rejected_mode if rejected_mode is None else f" \t{rejected_mode}\n"
        rejected = {**disabled, "spark_configs": {"spark.rapids.sql.explain": value}}
        try:
            require_rapids_explain_enabled(rejected)
        except ResourceError as exc:
            assert "spark.rapids.sql.explain=ALL" in str(exc)
        else:
            raise AssertionError(f"non-ALL RAPIDS explain mode was accepted: {rejected_mode!r}")


def test_spark_accelerator_evidence_fails_closed_on_malformed_or_exceptional_getters() -> None:
    class BrokenConf:
        def get(self, key, default=None):
            if key == "spark.plugins":
                return object()
            if key == "spark.rapids.sql.enabled":
                raise RuntimeError("gateway closed")
            return default

    class BrokenContext:
        def getConf(self):
            return BrokenConf()

        @property
        def resources(self):
            raise RuntimeError("gateway closed")

    evidence = spark_accelerator_evidence(SimpleNamespace(sparkContext=BrokenContext()))
    assert evidence["spark_sql_acceleration"] == "cpu_or_unconfigured"
    assert evidence["driver_gpu_addresses"] == []
    assert evidence["probe_errors"] == [
        "spark_conf:spark.plugins:malformed",
        "spark_conf:spark.rapids.sql.enabled:unreadable",
        "driver_resources:unreadable",
    ]
    assert all(value is None for value in evidence["spark_configs"].values())

    malformed_addresses = SimpleNamespace(sparkContext=SimpleNamespace(
        getConf=lambda: SimpleNamespace(get=lambda _key, default=None: default),
        resources={"gpu": SimpleNamespace(addresses="0")},
    ))
    malformed = spark_accelerator_evidence(malformed_addresses)
    assert malformed["driver_gpu_addresses"] == []
    assert malformed["probe_errors"] == ["driver_gpu_addresses:malformed"]

    malformed_context = SimpleNamespace(sparkContext=SimpleNamespace(
        getConf=lambda: None,
        resources=None,
    ))
    assert spark_accelerator_evidence(malformed_context)["probe_errors"] == [
        "spark_conf:malformed",
        "driver_resources:malformed",
    ]


def test_torch_accelerator_evidence_matches_the_existing_cuda_first_selection() -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: "NVIDIA T4" if index == 0 else "unused",
        ),
        version=SimpleNamespace(cuda="12.4"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    assert torch_accelerator_evidence(torch_module) == {
        "cuda_available": True,
        "cuda_device_count": 2,
        "cuda_version": "12.4",
        "cuda_device_name": "NVIDIA T4",
        "mps_available": False,
        "selected_device": "cuda",
        "probe_errors": [],
    }


def test_torch_accelerator_evidence_reports_mps_but_keeps_the_cpu_fallback() -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda _index: (_ for _ in ()).throw(
                AssertionError("queried a CUDA device name when CUDA is unavailable")),
        ),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    assert torch_accelerator_evidence(torch_module) == {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "cuda_device_name": None,
        "mps_available": True,
        "selected_device": "cpu",
        "probe_errors": [],
    }


def test_torch_accelerator_evidence_normalizes_malformed_and_exceptional_probes() -> None:
    class BrokenCuda:
        @staticmethod
        def is_available():
            return "yes"

        @staticmethod
        def device_count():
            raise RuntimeError("driver unavailable")

    class BrokenVersion:
        @property
        def cuda(self):
            raise RuntimeError("build metadata unavailable")

    torch_module = SimpleNamespace(
        cuda=BrokenCuda(),
        version=BrokenVersion(),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: 1)),
    )
    assert torch_accelerator_evidence(torch_module) == {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "cuda_device_name": None,
        "mps_available": False,
        "selected_device": "cpu",
        "probe_errors": [
            "cuda_available:malformed",
            "cuda_device_count:unreadable",
            "cuda_version:unreadable",
            "mps_available:malformed",
        ],
    }

    exceptional_name = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: (_ for _ in ()).throw(
                RuntimeError("device query failed")),
        ),
        version=SimpleNamespace(cuda=""),
        backends=SimpleNamespace(mps=SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("backend query failed")))),
    )
    exceptional = torch_accelerator_evidence(exceptional_name)
    assert exceptional["selected_device"] == "cuda"
    assert exceptional["cuda_device_name"] is None
    assert exceptional["probe_errors"] == [
        "cuda_version:malformed",
        "cuda_device_name:unreadable",
        "mps_available:unreadable",
    ]


def _other_thread_can_acquire(lock) -> bool:
    """Probe a lock without using elapsed time as evidence that acquisition was blocked."""
    result: list[bool] = []
    completed = threading.Event()

    def probe() -> None:
        acquired = lock.acquire(blocking=False)
        result.append(acquired)
        if acquired:
            lock.release()
        completed.set()

    thread = threading.Thread(target=probe)
    thread.start()
    assert completed.wait(2), "context-lock probe did not complete"
    thread.join(2)
    assert not thread.is_alive() and len(result) == 1
    return result[0]


def test_the_configuration_scales_with_the_machine() -> None:
    """THE point of the module. A 34 GB box must not run at a 16 GB box's settings -- which is
    exactly what happened: 10.4 GB used of 33.7 available, an hour per origin."""
    clear_env()
    small = spark_profile(machine(16, 4))
    large = spark_profile(machine(34, 4))
    assert int(large.driver_memory.rstrip("g")) > int(small.driver_memory.rstrip("g"))
    assert large.python_worker_budget_gb > small.python_worker_budget_gb
    assert large.series_per_partition >= small.series_per_partition


def test_forecast_scoring_preserves_the_native_resource_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_env()
    for gb in (16, 64):
        host = machine(gb, 10)
        for metric in ("wrmsse", "rmse"):
            assert forecast_scoring_profile(metric, ("hmfnet",), host) == (
                spark_profile(host, python_udfs=False)
            )


def test_machine_profile_refuses_impossible_or_mistyped_physical_resources() -> None:
    valid = {"total_ram_bytes": _GB, "cores": 1, "source": "probe"}
    for change in (
        {"total_ram_bytes": 0},
        {"total_ram_bytes": True},
        {"cores": 0},
        {"cores": True},
        {"source": ""},
    ):
        try:
            MachineProfile(**(valid | change))
        except ResourceError:
            continue
        raise AssertionError(f"accepted malformed machine profile {change!r}")


def test_most_of_the_machine_is_actually_allocated() -> None:
    """The silent failure is under-use, so the test is that the allocation is not small."""
    clear_env()
    for gb in (16, 34, 64):
        p = spark_profile(machine(gb, 8))
        allocated = int(p.driver_memory.rstrip("g")) + p.python_worker_budget_gb
        assert allocated >= gb - OS_RESERVE_GB - 1, \
            f"{gb} GB machine: only {allocated:.1f} GB allocated"


def test_the_heap_never_takes_the_whole_machine_when_python_workers_need_room() -> None:
    """In local mode the JVM IS the executor, so heap and Python workers share one machine. A
    heap at 80% leaves nothing for the workers and the OS kills them -- with no traceback,
    because a killed worker cannot raise. Three runs died exactly this way."""
    clear_env()
    p = spark_profile(machine(34, 4), python_udfs=True)
    heap = int(p.driver_memory.rstrip("g"))
    assert p.python_worker_budget_gb > heap, \
        "the workers got less than the heap on a Python-UDF workload"
    assert heap / (34 - OS_RESERVE_GB) < HEAP_SHARE_WITH_PYTHON + 0.05


def test_a_native_only_workload_gives_the_heap_more() -> None:
    """The naive adapter is native Catalyst with no Python in the loop -- which is why it never
    OOMed while the statsforecast adapters did. It should not be sized as if it had workers."""
    clear_env()
    m = machine(34, 4)
    with_py = int(spark_profile(m, python_udfs=True).driver_memory.rstrip("g"))
    native = int(spark_profile(m, python_udfs=False).driver_memory.rstrip("g"))
    assert native > with_py
    assert native / (34 - OS_RESERVE_GB) <= HEAP_SHARE_NATIVE_ONLY + 0.05


def test_worker_reuse_is_on_whenever_the_workers_have_room() -> None:
    """Measured: 4.91s for a cold numpy + statsforecast import, paid per task with reuse off."""
    clear_env()
    assert spark_profile(machine(34, 4)).worker_reuse is True
    assert spark_profile(machine(16, 4)).worker_reuse is True


def test_partition_size_is_capped_by_parallelism_not_only_by_memory() -> None:
    """The measured mistake this cap exists to prevent. The same 30,490-series AutoETS job on the
    same box took 1,690s at 250 series per partition and over 4,960s at 2,000 -- three times
    slower, from a change the MEMORY arithmetic permitted. The workload is CPU-bound (all cores at
    98% with 29% of RAM used), so the binding constraint is how many tasks there are, not how big
    each one may be."""
    from autoforecast.resources import MAX_SERIES_PER_PARTITION
    clear_env()
    for gb in (34, 64, 256):                     # more memory must not mean bigger partitions
        p = spark_profile(machine(gb, 4))
        assert p.series_per_partition <= MAX_SERIES_PER_PARTITION, (
            f"{gb} GB machine derived {p.series_per_partition} series per partition; memory "
            "cannot be allowed to set this on a CPU-bound per-series fit")


def test_a_tiny_machine_still_gets_a_workable_partition() -> None:
    """The floor matters as much as the ceiling: a partition of one series is all overhead."""
    clear_env()
    assert spark_profile(machine(4, 2)).series_per_partition >= 50


def test_an_environment_variable_overrides_the_derived_value() -> None:
    """The variables remain -- they now override a MEASURED default rather than substituting for
    the absence of one."""
    clear_env()
    os.environ["FORECAST_SPARK_MEMORY"] = "3g"
    os.environ["FORECAST_SPARK_CORES"] = "2"
    try:
        p = spark_profile(machine(34, 8))
        assert p.driver_memory == "3g" and p.cores == 2
    finally:
        clear_env()


def test_memory_override_recomputes_the_real_worker_budget() -> None:
    """Reporting the old derived remainder after changing the heap double-counts memory in logs
    and in the partition-size calculation."""
    clear_env()
    os.environ["FORECAST_SPARK_MEMORY"] = "20g"
    try:
        p = spark_profile(machine(34, 8))
        assert p.python_worker_budget_gb == 12.0  # 34 - 2 OS - 20 heap
    finally:
        clear_env()


def test_overcommitting_environment_overrides_are_refused() -> None:
    bad = (
        ("FORECAST_SPARK_MEMORY", "40g"),
        ("FORECAST_SPARK_MEMORY", "not-memory"),
        ("FORECAST_SPARK_CORES", "9"),
        ("FORECAST_SHUFFLE_PARTITIONS", "0"),
        ("FORECAST_SERIES_PER_PARTITION", "-1"),
        ("FORECAST_WORKER_REUSE", "sometimes"),
    )
    for name, value in bad:
        clear_env()
        os.environ[name] = value
        try:
            spark_profile(machine(34, 8))
        except ResourceError:
            continue
        finally:
            clear_env()
        raise AssertionError(f"accepted unsafe resource override {name}={value}")


def test_isolated_profiles_split_one_machine_instead_of_counting_it_twice() -> None:
    """The parent JVM remains live while an adapter worker owns another JVM. Their advertised
    budgets must fit one physical machine, not each independently claim all of it."""
    clear_env()
    m = machine(34, 8)
    parent, worker = isolated_spark_profiles(m)
    allocated = (
        int(parent.driver_memory.rstrip("g")) + parent.python_worker_budget_gb
        + int(worker.driver_memory.rstrip("g")) + worker.python_worker_budget_gb
    )
    assert allocated == pytest.approx(m.total_ram_gb - OS_RESERVE_GB)
    assert parent.cores + worker.cores <= m.cores
    assert parent.cores >= 1 and worker.cores >= 1


def test_isolated_profiles_do_not_apply_the_same_generic_override_twice() -> None:
    """A generic 20g override cannot silently become two concurrent 20g JVMs."""
    clear_env()
    os.environ["FORECAST_SPARK_MEMORY"] = "20g"
    os.environ["FORECAST_SPARK_CORES"] = "7"
    try:
        parent, worker = isolated_spark_profiles(machine(34, 8))
        assert parent.driver_memory != "20g" and worker.driver_memory != "20g"
        assert parent.cores + worker.cores <= 8
    finally:
        clear_env()


def test_isolated_profiles_refuse_a_machine_that_cannot_host_two_jvms() -> None:
    for profile, message in ((machine(6, 4), "cannot derive safe concurrent"),
                             (machine(16, 1), "at least 2 usable cores")):
        try:
            isolated_spark_profiles(profile)
        except ResourceError as exc:
            assert message in str(exc)
            continue
        raise AssertionError("split an undersized machine into two nominal Spark budgets")


def test_isolated_profiles_accept_an_eight_decimal_gb_runner() -> None:
    """CI providers advertise 8 GB runners whose binary capacity is about 7.8 GiB."""
    parent, worker = isolated_spark_profiles(machine(7.8, 4))
    assert parent.cores + worker.cores <= 4
    assert int(parent.driver_memory.rstrip("g")) >= 1
    assert int(worker.driver_memory.rstrip("g")) >= 1


def test_isolated_profiles_subtract_the_os_reserve_only_once() -> None:
    m = machine(8, 4)
    parent, worker = isolated_spark_profiles(m)
    allocated = sum(
        int(profile.driver_memory.rstrip("g")) + profile.python_worker_budget_gb
        for profile in (parent, worker)
    )
    assert allocated == pytest.approx(m.total_ram_gb - OS_RESERVE_GB)


def test_a_machine_too_small_to_run_on_is_refused() -> None:
    try:
        spark_profile(machine(2.5, 2))
    except ResourceError as e:
        assert "needs a larger machine" in str(e); return
    raise AssertionError("sized a job onto a machine with nothing left after the OS reserve")


@pytest.mark.parametrize("reserve", [True, 10**400])
def test_os_reserve_rejects_boolean_and_unrepresentable_numbers(reserve: object) -> None:
    with pytest.raises(ResourceError, match="finite non-negative number"):
        spark_profile(machine(8, 4), os_reserve_gb=reserve)  # type: ignore[arg-type]


@pytest.fixture
def blocked_memory_probes(monkeypatch: pytest.MonkeyPatch):
    """Make every host probe unavailable unless the individual test enables it."""
    import autoforecast.resources as res

    read_text = pathlib.Path.read_text

    def missing_meminfo(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise FileNotFoundError("not a Linux host")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", missing_meminfo)
    monkeypatch.setattr(res.shutil, "which", lambda _: "/usr/sbin/sysctl")
    monkeypatch.setattr(res.subprocess, "run", mock.Mock(
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="Operation not permitted"),
    ))
    monkeypatch.setattr(res.os, "sysconf", mock.Mock(side_effect=ValueError("unsupported")),
                        raising=False)
    monkeypatch.setitem(sys.modules, "psutil", None)
    return res


@pytest.mark.parametrize("failure", [
    SimpleNamespace(returncode=1, stdout="", stderr="Operation not permitted"),
    PermissionError("sysctl execution denied"),
    subprocess.TimeoutExpired("sysctl", 5),
    SimpleNamespace(returncode=0, stdout="0", stderr=""),
    SimpleNamespace(returncode=0, stdout="not memory", stderr=""),
])
def test_physical_memory_falls_back_when_sysctl_is_denied(
    blocked_memory_probes, monkeypatch: pytest.MonkeyPatch, failure,
) -> None:
    res = blocked_memory_probes
    sysctl = mock.Mock(side_effect=failure) if isinstance(failure, Exception) else mock.Mock(
        return_value=failure)
    monkeypatch.setattr(res.subprocess, "run", sysctl)
    sysconf = mock.Mock(side_effect={"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2 * 1024**2}.__getitem__)
    monkeypatch.setattr(res.os, "sysconf", sysconf)
    assert res._physical_memory() == (8 * _GB, "os.sysconf")
    sysctl.assert_called_once()
    assert sysconf.call_args_list == [mock.call("SC_PAGE_SIZE"), mock.call("SC_PHYS_PAGES")]


@pytest.mark.parametrize("unavailable", [ValueError("unsupported"), OSError("denied"), -1, 0])
def test_physical_memory_uses_psutil_when_sysconf_is_unavailable(
    blocked_memory_probes, monkeypatch: pytest.MonkeyPatch, unavailable,
) -> None:
    res = blocked_memory_probes
    sysconf = mock.Mock(side_effect=unavailable) if isinstance(unavailable, Exception) else mock.Mock(
        return_value=unavailable)
    monkeypatch.setattr(res.os, "sysconf", sysconf)
    virtual_memory = mock.Mock(return_value=SimpleNamespace(total=12 * _GB))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=virtual_memory))
    assert res._physical_memory() == (12 * _GB, "psutil.virtual_memory")
    virtual_memory.assert_called_once_with()


@pytest.mark.parametrize("probe", ["proc", "sysctl"])
def test_physical_memory_prefers_a_valid_primary_probe(
    blocked_memory_probes, monkeypatch: pytest.MonkeyPatch, probe: str,
) -> None:
    res = blocked_memory_probes
    if probe == "proc":
        monkeypatch.setattr(pathlib.Path, "read_text", lambda _: "MemTotal:       8388608 kB\n")
    else:
        monkeypatch.setattr(res.subprocess, "run", mock.Mock(
            return_value=SimpleNamespace(returncode=0, stdout=str(8 * _GB)),
        ))
    monkeypatch.setattr(res.os, "sysconf", mock.Mock(side_effect=AssertionError("unneeded fallback")))
    expected_source = "/proc/meminfo" if probe == "proc" else "sysctl hw.memsize"
    assert res._physical_memory() == (8 * _GB, expected_source)


@pytest.mark.parametrize("psutil_value", [None, 0, -1, True, OSError("denied")])
def test_spark_profile_override_does_not_emit_an_impossible_remediation(
    blocked_memory_probes, monkeypatch: pytest.MonkeyPatch, psutil_value,
) -> None:
    res = blocked_memory_probes
    if psutil_value is not None:
        probe = (mock.Mock(side_effect=psutil_value) if isinstance(psutil_value, Exception)
                 else mock.Mock(return_value=SimpleNamespace(total=psutil_value)))
        monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=probe))
    monkeypatch.setenv("FORECAST_SPARK_MEMORY", "1g")
    with pytest.raises(ResourceError, match="cannot determine total physical memory") as caught:
        res.spark_profile()
    message = str(caught.value)
    assert "MachineProfile(total_ram_bytes=" in message
    assert "FORECAST_SPARK_MEMORY sets only the JVM heap" in message
    assert "set FORECAST_SPARK_MEMORY explicitly instead" not in message


def test_fallback_memory_detection_still_honors_cgroup_limits(
    blocked_memory_probes, monkeypatch: pytest.MonkeyPatch,
) -> None:
    res = blocked_memory_probes
    monkeypatch.setattr(res.os, "sysconf", {"SC_PAGE_SIZE": 4096,
                                          "SC_PHYS_PAGES": 8 * 1024**2}.__getitem__)
    monkeypatch.setattr(res, "_cgroup_memory_limit", lambda: 4 * _GB)
    monkeypatch.setattr(res, "_cgroup_cpu_limit", lambda: 2)
    monkeypatch.setattr(res.os, "cpu_count", lambda: 16)
    actual = res.detect()
    assert actual == MachineProfile(4 * _GB, 2, "cgroup, cpu=cgroup")
    sized = res.spark_profile(actual, honor_env=False)
    assert sized.python_worker_budget_gb + float(sized.driver_memory.rstrip("g")) == 2.0


def test_spark_profile_override_works_after_a_denied_probe(
    blocked_memory_probes, monkeypatch: pytest.MonkeyPatch,
) -> None:
    res = blocked_memory_probes
    monkeypatch.setattr(res.os, "sysconf", {"SC_PAGE_SIZE": 4096,
                                          "SC_PHYS_PAGES": 2 * 1024**2}.__getitem__)
    monkeypatch.setattr(res, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(res, "_cgroup_cpu_limit", lambda: None)
    monkeypatch.setattr(res.os, "cpu_count", lambda: 2)
    monkeypatch.setenv("FORECAST_SPARK_MEMORY", "1g")
    monkeypatch.setenv("FORECAST_SPARK_CORES", "1")
    profile = res.spark_profile()
    assert profile.driver_memory == "1g"
    assert profile.cores == 1
    assert profile.python_worker_budget_gb == 5.0


def test_explicit_machine_profile_does_not_need_memory_probes(
    blocked_memory_probes,
) -> None:
    res = blocked_memory_probes
    profile = res.spark_profile(MachineProfile(8 * _GB, 2, "verified total"), honor_env=False)
    assert profile.python_worker_budget_gb + float(profile.driver_memory.rstrip("g")) == 6.0


def test_detection_returns_something_plausible_for_this_machine() -> None:
    m = detect()
    assert m.total_ram_bytes > 1 * _GB and m.cores >= 1
    assert m.source
    assert "GB" in str(m) and "cores" in str(m)


def test_total_ram_gb_converts_from_bytes_not_from_a_rounded_figure() -> None:
    """Reported in GB everywhere -- logs, warnings, the sizing arithmetic -- so a wrong divisor
    would misreport the machine consistently and look self-consistent while doing it."""
    m = machine(34, 4)
    assert abs(m.total_ram_gb - 34.0) < 1e-9
    assert m.total_ram_bytes == 34 * 1024 ** 3, "GiB, not GB -- the divisor must match the source"


def test_describe_names_both_the_machine_and_what_was_derived() -> None:
    """A run whose configuration cannot be read back from its own log is a run whose numbers
    cannot be explained later."""
    text = describe()
    assert "machine:" in text and "spark:" in text
    assert "heap=" in text and "series/partition=" in text


def test_sanity_check_warns_about_the_exact_failure_that_happened() -> None:
    """Under-using the machine, fewer cores than exist, and reuse switched off -- the three things
    that were true of the run that took an hour per origin."""
    starved = SparkProfile(cores=2, driver_memory="4g", shuffle_partitions=64,
                           worker_reuse=False, series_per_partition=100,
                           python_worker_budget_gb=2.0)
    notes = sanity_check(starved, machine(34, 8))
    joined = " ".join(notes)
    assert "under half the machine" in joined
    assert "2 of 8 cores" in joined
    assert "worker reuse is OFF" in joined


def test_spark_profile_refuses_unrepresentable_worker_budget() -> None:
    try:
        SparkProfile(
            cores=2, driver_memory="4g", shuffle_partitions=8,
            worker_reuse=True, series_per_partition=10,
            python_worker_budget_gb=10**400,
        )
    except ResourceError as exc:
        assert "finite positive" in str(exc)
    else:
        raise AssertionError("an unrepresentable Python-worker budget entered Spark evidence")


def test_spark_profile_refuses_unrepresentable_driver_memory() -> None:
    huge_memory = "9" * 400 + "g"
    try:
        SparkProfile(
            cores=2, driver_memory=huge_memory, shuffle_partitions=8,
            worker_reuse=True, series_per_partition=10, python_worker_budget_gb=1.0)
    except ResourceError:
        return
    raise AssertionError("an infinite binary64 driver heap entered Spark configuration evidence")


def test_fresh_spark_guard_covers_all_get_or_create_reuse_states() -> None:
    import autoforecast.resources as res

    states = (
        ("thread-active session", object(), None, None),
        ("inactive default session", None, object(), None),
        ("bare SparkContext", None, None, object()),
    )
    for label, active, instantiated, context in states:
        session_type = SimpleNamespace(
            getActiveSession=lambda active=active: active,
            _instantiatedSession=instantiated,
        )
        context_type = SimpleNamespace(_active_spark_context=context)
        try:
            res._require_fresh_spark_runtime(session_type, context_type)
        except RuntimeError as exc:
            assert label in str(exc), (label, exc)
        else:
            raise AssertionError(f"fresh-runtime guard accepted {label}")
    res._require_fresh_spark_runtime(
        SimpleNamespace(getActiveSession=lambda: None, _instantiatedSession=None),
        SimpleNamespace(_active_spark_context=None),
    )


def test_fresh_spark_acquisition_holds_both_pyspark_locks_through_creation() -> None:
    import autoforecast.resources as res

    session_lock, context_lock = threading.RLock(), threading.RLock()
    state = {"session": None}
    owned_spark, competing_spark = object(), object()
    builder = mock.Mock(name="builder")
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    attempted, completed = threading.Event(), threading.Event()

    def competitor():
        attempted.set()
        with session_lock:
            with context_lock:
                if state["session"] is None:
                    state["session"] = competing_spark
        completed.set()

    thread = None

    def create_owned():
        nonlocal thread
        thread = threading.Thread(target=competitor)
        thread.start()
        assert attempted.wait(2)
        assert not completed.wait(0.05), "competitor entered the fresh ownership boundary"
        state["session"] = owned_spark
        return owned_spark

    builder.getOrCreate.side_effect = create_owned
    session_type = SimpleNamespace(
        Builder=SimpleNamespace(_lock=session_lock),
        getActiveSession=lambda: state["session"],
        _instantiatedSession=None,
        builder=builder,
    )
    context_type = SimpleNamespace(_lock=context_lock, _active_spark_context=None)
    acquired = res._acquire_fresh_spark_runtime(
        session_type,
        context_type,
        master="local[1]",
        app_name="fresh-test",
        config={"spark.driver.memory": "1g"},
    )
    assert acquired is owned_spark
    assert thread is not None
    thread.join(2)
    assert completed.is_set() and state["session"] is owned_spark


def test_fresh_spark_acquisition_blocks_a_context_only_creator() -> None:
    """A direct SparkContext creator must not enter during fresh session creation."""
    import autoforecast.resources as res

    session_lock, context_lock = threading.RLock(), threading.RLock()
    state = {"session": None}
    owned_spark = object()
    builder = mock.Mock(name="builder")
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder

    def create_owned():
        assert not _other_thread_can_acquire(context_lock), (
            "a context-only competitor entered the fresh ownership boundary")
        state["session"] = owned_spark
        return owned_spark

    builder.getOrCreate.side_effect = create_owned
    session_type = SimpleNamespace(
        Builder=SimpleNamespace(_lock=session_lock),
        getActiveSession=lambda: state["session"],
        _instantiatedSession=None,
        builder=builder,
    )
    context_type = SimpleNamespace(_lock=context_lock, _active_spark_context=None)
    acquired = res._acquire_fresh_spark_runtime(
        session_type,
        context_type,
        master="local[1]",
        app_name="fresh-context-lock-test",
        config={"spark.driver.memory": "1g"},
    )
    assert acquired is owned_spark and state["session"] is owned_spark


def test_embedded_spark_acquisition_is_atomic_and_does_not_mutate_caller_config() -> None:
    import autoforecast.resources as res

    session_lock, context_lock = threading.RLock(), threading.RLock()
    state = {"session": None}
    owned_spark, competing_spark = object(), object()
    builder = SimpleNamespace()
    builder.master = lambda _value: builder
    builder.appName = lambda _value: builder
    builder.config = lambda *_args: builder

    def create_owned():
        state["session"] = owned_spark
        return owned_spark

    builder.getOrCreate = create_owned
    session_type = SimpleNamespace(
        Builder=SimpleNamespace(_lock=session_lock),
        getActiveSession=lambda: state["session"],
        _instantiatedSession=None,
        builder=builder,
    )
    context_type = SimpleNamespace(_lock=context_lock, _active_spark_context=None)
    attempted, completed = threading.Event(), threading.Event()

    def competitor():
        attempted.set()
        with session_lock:
            with context_lock:
                if state["session"] is None:
                    state["session"] = competing_spark
        completed.set()

    thread = None

    def profile_factory():
        nonlocal thread
        thread = threading.Thread(target=competitor)
        thread.start()
        assert attempted.wait(2)
        assert not completed.wait(0.05), "competitor entered the atomic ownership boundary"
        return SparkProfile(
            cores=1, driver_memory="1g", shuffle_partitions=1,
            worker_reuse=False, series_per_partition=1, python_worker_budget_gb=1.0)

    acquired, owned = res._acquire_embedded_spark_runtime(
        session_type, context_type, app_name="test", profile_factory=profile_factory)
    assert acquired is owned_spark and owned
    assert thread is not None
    thread.join(2)
    assert completed.is_set() and state["session"] is owned_spark

    # Once caller state exists, return it directly. None of the configured-builder methods may
    # run, because PySpark applies those options to an existing session.
    poisoned_builder = mock.Mock()
    session_type.builder = poisoned_builder
    acquired, owned = res._acquire_embedded_spark_runtime(
        session_type, context_type, app_name="test",
        profile_factory=lambda: (_ for _ in ()).throw(
            AssertionError("profiled a caller-owned runtime")))
    assert acquired is owned_spark and not owned
    poisoned_builder.master.assert_not_called()
    poisoned_builder.appName.assert_not_called()
    poisoned_builder.config.assert_not_called()
    poisoned_builder.getOrCreate.assert_not_called()


def test_embedded_spark_acquisition_blocks_a_context_only_creator() -> None:
    """Embedded ownership cannot be decided while a bare context creator can enter."""
    import autoforecast.resources as res

    session_lock, context_lock = threading.RLock(), threading.RLock()
    owned_spark = object()
    builder = mock.Mock(name="builder")
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = owned_spark
    session_type = SimpleNamespace(
        Builder=SimpleNamespace(_lock=session_lock),
        getActiveSession=lambda: None,
        _instantiatedSession=None,
        builder=builder,
    )
    context_type = SimpleNamespace(_lock=context_lock, _active_spark_context=None)

    def profile_factory():
        assert not _other_thread_can_acquire(context_lock), (
            "a context-only competitor entered the embedded ownership boundary")
        return SparkProfile(
            cores=1, driver_memory="1g", shuffle_partitions=1,
            worker_reuse=False, series_per_partition=1, python_worker_budget_gb=1.0)

    acquired, owned = res._acquire_embedded_spark_runtime(
        session_type, context_type, app_name="embedded-context-lock-test",
        profile_factory=profile_factory)
    assert acquired is owned_spark and owned


def test_sanity_check_is_silent_when_the_configuration_is_right() -> None:
    clear_env()
    m = machine(34, 4)
    assert sanity_check(spark_profile(m), m) == []


def test_as_config_produces_the_keys_spark_expects() -> None:
    clear_env()
    cfg = spark_profile(machine(34, 4)).as_config()
    assert cfg["spark.driver.memory"].endswith("g")
    assert int(cfg["spark.sql.shuffle.partitions"]) > 0
    assert cfg["spark.python.worker.reuse"] in ("true", "false")


def test_a_container_limit_beats_the_hosts_physical_total() -> None:
    """Containers lie: /proc/meminfo reports the HOST. Sizing a 12 GB heap inside a 4 GB container
    gets the process killed at the first shuffle rather than at start-up, where it would be
    obvious. Checked through the real reader on this machine plus a constructed profile, since a
    cgroup cannot be faked without root."""
    import autoforecast.resources as res
    real = res._cgroup_memory_limit()
    assert real is None or real > 0, "a cgroup limit must be positive or absent, never zero"
    # The consequence, tested directly: a small profile must produce a small heap.
    clear_env()
    tiny = spark_profile(machine(4, 2))
    assert int(tiny.driver_memory.rstrip("g")) <= 2, \
        f"a 4 GB container was given a {tiny.driver_memory} heap"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
