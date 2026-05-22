"""Protocol tests for the durable local campaign supervisor."""

from __future__ import annotations

import builtins
import contextlib
import fcntl
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest
import yaml

import autoforecast.campaign as campaign
import autoforecast.campaign_child as campaign_child
import autoforecast.training_lease as training_lease_module
from autoforecast.training_lease import (
    HOST_TRAINING_LEASE_FD,
    TrainingLeaseError,
    host_training_lease_path,
    inherited_training_lease_pass_fds,
    open_host_training_lease,
    training_lease,
    verify_inherited_host_training_lease,
    with_training_lease,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_campaign_numbers_reject_values_outside_binary64() -> None:
    with pytest.raises(campaign.CampaignError, match="finite number"):
        campaign._finite_number(10**400, "memory", positive=True)  # noqa: SLF001


def test_campaign_child_refuses_a_symlinked_worker_request(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "worker-request.json"
    target.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "worker-request-link.json"
    linked.symlink_to(target)

    with pytest.raises(campaign.CampaignError, match="regular non-symlink"):
        campaign_child.main(str(linked))


def test_process_tree_memory_refuses_when_no_measurement_backend_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_psutil(name: str, *args: object, **kwargs: object) -> object:
        if name == "psutil":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_psutil)
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    )
    with pytest.raises(campaign.CampaignError, match="cannot measure worker memory"):
        campaign._process_tree_rss(os.getpid())  # noqa: SLF001


def test_process_tree_memory_refuses_malformed_ps_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_psutil(name: str, *args: object, **kwargs: object) -> object:
        if name == "psutil":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_psutil)
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=f"{os.getpid()} 1 1024\ninvalid row\n", stderr=""
        ),
    )
    with pytest.raises(campaign.CampaignError, match="malformed ps row"):
        campaign._process_tree_rss(os.getpid())  # noqa: SLF001


def test_process_tree_memory_ignores_a_descendant_that_exits_during_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedDescendant:
        def memory_info(self) -> object:
            raise psutil.NoSuchProcess(12345)

    class LiveParent:
        def memory_info(self) -> object:
            return SimpleNamespace(rss=4096)

        def children(self, *, recursive: bool) -> list[ExitedDescendant]:
            assert recursive
            return [ExitedDescendant()]

    monkeypatch.setattr(psutil, "Process", lambda _pid: LiveParent())
    assert campaign._process_tree_rss(os.getpid()) == 4096  # noqa: SLF001


def test_terminate_process_group_kills_a_live_child_when_group_signalling_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_signal(_pid: int, _signal: signal.Signals) -> None:
        raise PermissionError("process group is no longer ours")

    class LiveChild:
        pid = 12345
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout: float) -> int:
            assert timeout == 1
            return -signal.SIGKILL

    child = LiveChild()
    monkeypatch.setattr(campaign.os, "killpg", deny_signal)
    campaign._terminate_process_group(child)  # type: ignore[arg-type]  # noqa: SLF001
    assert child.killed


def test_terminate_process_group_does_not_kill_an_exited_child_after_permission_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign.os,
        "killpg",
        lambda _pid, _signal: (_ for _ in ()).throw(PermissionError("denied")),
    )
    child = SimpleNamespace(pid=12345, poll=lambda: 0, kill=lambda: pytest.fail("kill called"))
    campaign._terminate_process_group(child)  # type: ignore[arg-type]  # noqa: SLF001


def _stage(
    stage_id: str,
    code: str,
    *,
    needs: list[str] | None = None,
    outputs: list[str] | None = None,
    attempts: int = 1,
    timeout: int = 10,
) -> dict[str, object]:
    return {
        "id": stage_id,
        "needs": needs or [],
        "command": [sys.executable, "-c", code],
        "timeout_seconds": timeout,
        "max_attempts": attempts,
        "retry_mode": "restart",
        "required_outputs": outputs or [],
        "resources": {"device": "cpu", "memory_gib": 0.25, "min_free_disk_gib": 0.0},
    }


def _plan(tmp_path: pathlib.Path, stages: list[dict[str, object]], **extra: object) -> pathlib.Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign": {"name": "nightly-test", "stages": stages, **extra},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _store(tmp_path: pathlib.Path, stages: list[dict[str, object]]) -> campaign.CampaignStore:
    return campaign.CampaignStore.open_or_create(
        _plan(tmp_path, stages),
        output_root=tmp_path / "runs",
        working_directory=ROOT,
    )


def _assert_centered_hash_banner(value: str, message: str) -> None:
    assert value.strip("# ") == message
    left = len(value) - len(value.lstrip("#"))
    right = len(value) - len(value.rstrip("#"))
    assert min(left, right) >= 6
    assert abs(left - right) <= 1


def test_live_stage_banner_has_position_and_human_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stages = [
        _stage("v0fs0o0", "pass", attempts=2),
        _stage("v1fs0o0", "pass", attempts=2),
        _stage("aggregate_selection", "pass", needs=["v0fs0o0", "v1fs0o0"]),
        _stage("final_fit", "pass", needs=["aggregate_selection"]),
        _stage("final_score", "pass", needs=["final_fit"]),
        _stage("replay_report", "pass", needs=["final_score"]),
    ]
    supervisor = campaign.CampaignSupervisor(_store(tmp_path, stages), poll_seconds=0.01)
    monkeypatch.setenv("AUTOFORECAST_LIVE_PROGRESS", "1")
    for stage in supervisor.store.plan.stages:
        supervisor._announce_stage(stage, 1)  # noqa: SLF001
    messages = [
        "Stage 1/6: Candidate 1 validation "
        "(fresh initialization, seed 1, origin 1) [v0fs0o0] - attempt 1/2",
        "Stage 2/6: Candidate 2 validation "
        "(fresh initialization, seed 1, origin 1) [v1fs0o0] - attempt 1/2",
        "Stage 3/6: Select the best validated candidate " "[aggregate_selection] - attempt 1/1",
        "Stage 4/6: Fresh fit on training + validation data " "[final_fit] - attempt 1/1",
        "Stage 5/6: Evaluate the frozen model on test data " "[final_score] - attempt 1/1",
        "Stage 6/6: Verify artifacts and publish the final report " "[replay_report] - attempt 1/1",
    ]
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == len(messages)
    for line, message in zip(lines, messages, strict=True):
        _assert_centered_hash_banner(line, message)


def test_single_model_benchmark_groups_checkpoints_into_three_user_stages(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stages = [
        _stage("v0fs0o0", "pass", attempts=2),
        _stage("aggregate_selection", "pass", needs=["v0fs0o0"]),
        _stage("final_fit", "pass", needs=["aggregate_selection"]),
        _stage("final_score", "pass", needs=["final_fit"]),
        _stage("replay_report", "pass", needs=["final_score"]),
    ]
    supervisor = campaign.CampaignSupervisor(_store(tmp_path, stages), poll_seconds=0.01)
    monkeypatch.setenv("AUTOFORECAST_LIVE_PROGRESS", "1")
    for stage in supervisor.store.plan.stages:
        supervisor._announce_stage(stage, 1)  # noqa: SLF001
    lines = capsys.readouterr().err.splitlines()
    _assert_centered_hash_banner(
        lines[0],
        "Stage 1/3: Train on training data and evaluate on validation data "
        "[v0fs0o0] - attempt 1/2",
    )
    assert len(lines) == 5
    assert lines[1] == (
        "Stage 1/3 | Phase 6/6 — Freeze the validation result and model configuration "
        "[aggregate_selection] - attempt 1/1"
    )
    assert lines[4] == (
        "Stage 3/3 | Phase 3/3 — Verify and seal the test report " "[replay_report] - attempt 1/1"
    )
    _assert_centered_hash_banner(
        lines[2],
        "Stage 2/3: Fresh fit on training + validation data " "[final_fit] - attempt 1/1",
    )
    _assert_centered_hash_banner(
        lines[3],
        "Stage 3/3: Evaluate once on test data " "[final_score] - attempt 1/1",
    )


def test_campaign_creation_refuses_a_dangling_link_at_the_computed_root(
    tmp_path: pathlib.Path,
) -> None:
    plan_path = _plan(tmp_path, [_stage("train", "pass")])
    plan = campaign.load_campaign_plan(plan_path)
    output = tmp_path / "runs"
    output.mkdir()
    claimed = output / f"{plan.name}-{plan.source_sha256[:12]}"
    claimed.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(campaign.CampaignError, match="not a regular directory"):
        campaign.CampaignStore.open_or_create(
            plan_path,
            output_root=output,
            working_directory=ROOT,
        )


def test_plan_is_strict_hash_bound_and_acyclic(tmp_path: pathlib.Path) -> None:
    path = _plan(
        tmp_path,
        [
            _stage("prepare", "pass"),
            _stage("train", "pass", needs=["prepare"]),
        ],
    )
    loaded = campaign.load_campaign_plan(path)
    assert loaded.name == "nightly-test"
    assert loaded.stages[1].needs == ("prepare",)
    assert loaded.source_sha256 == campaign.hashlib.sha256(path.read_bytes()).hexdigest()

    cycle = _plan(
        tmp_path,
        [
            _stage("prepare", "pass", needs=["train"]),
            _stage("train", "pass", needs=["prepare"]),
        ],
    )
    with pytest.raises(campaign.CampaignError, match="cycle"):
        campaign.load_campaign_plan(cycle)


def test_plan_refuses_duplicate_yaml_keys_and_boolean_schema_version(
    tmp_path: pathlib.Path,
) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="duplicate YAML mapping key"):
        campaign.load_campaign_plan(duplicate)

    path = _plan(tmp_path, [_stage("train", "pass")])
    path.write_text(path.read_text().replace("schema_version: 1", "schema_version: true"))
    with pytest.raises(campaign.CampaignError, match="schema_version 1"):
        campaign.load_campaign_plan(path)


@pytest.mark.parametrize(
    "reserved",
    (
        "stage-result.json",
        "STAGE-RESULT.JSON",
        "worker-token",
        "Worker-Token",
        ".child.lock",
        ".CHILD.LOCK",
    ),
)
def test_plan_refuses_wrapper_owned_required_outputs(tmp_path: pathlib.Path, reserved: str) -> None:
    path = _plan(tmp_path, [_stage("train", "pass", outputs=[reserved])])
    with pytest.raises(campaign.CampaignError, match="wrapper-owned control paths"):
        campaign.load_campaign_plan(path)


def test_plan_refuses_required_outputs_that_differ_only_by_case(tmp_path: pathlib.Path) -> None:
    path = _plan(tmp_path, [_stage("train", "pass", outputs=["model.bin", "MODEL.BIN"])])
    with pytest.raises(campaign.CampaignError, match="unique, including by case"):
        campaign.load_campaign_plan(path)


@pytest.mark.parametrize(
    ("outputs", "message"),
    (
        (["STAGE-RESULT.JSON"], "wrapper-owned control path"),
        (["model.bin", "MODEL.BIN"], "unique, including by case"),
    ),
)
def test_worker_request_refuses_case_aliased_outputs(
    tmp_path: pathlib.Path, outputs: list[str], message: str
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    request_path = campaign.CampaignSupervisor(store)._prepare_attempt(  # noqa: SLF001
        store.plan.stages[0], 1
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["required_outputs"] = outputs

    with pytest.raises(campaign.CampaignError, match=message):
        campaign._validate_worker_request(request, request_path)  # noqa: SLF001


def test_run_stage_worker_refuses_a_linked_request(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "worker-request.json"
    target.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "worker-request-link.json"
    linked.symlink_to(target)
    with pytest.raises(campaign.CampaignError, match="may not be a symlink"):
        campaign.run_stage_worker(linked)


def test_worker_internal_failure_publishes_a_valid_nonzero_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    monkeypatch.setattr(
        campaign,
        "_process_tree_rss",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("measurement backend failed")),
    )

    assert campaign.run_stage_worker(request) == 2
    result, _sha = campaign._validate_worker_result(  # noqa: SLF001
        request.parent / "stage-result.json", request, store.plan.stages[0], 1
    )
    assert result["status"] == "failed"
    assert result["exit_code"] == 70


def test_worker_wrapper_log_is_appendable_after_an_interrupted_launch(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    (request.parent / "worker-wrapper.log").write_bytes(b"previous launch\n")

    wrapper = supervisor._launch_worker(request)  # noqa: SLF001

    assert wrapper.wait(timeout=15) == 0
    assert (request.parent / "worker-wrapper.log").read_bytes().startswith(b"previous launch\n")


def test_worker_waits_for_a_brief_supervisor_lock_probe(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001

    with campaign._open_lock(request.parent / ".worker.lock") as probe:  # noqa: SLF001
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        wrapper = supervisor._launch_worker(request)  # noqa: SLF001
        time.sleep(0.25)

    assert wrapper.wait(timeout=15) == 0
    result, _sha = campaign._validate_worker_result(  # noqa: SLF001
        request.parent / "stage-result.json", request, store.plan.stages[0], 1
    )
    assert result["status"] == "completed"


def test_worker_launch_pins_the_checkout_when_working_directory_is_elsewhere(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    working = tmp_path / "external-working-directory"
    working.mkdir()
    store = campaign.CampaignStore.open_or_create(
        _plan(tmp_path, [_stage("train", "pass")]),
        output_root=tmp_path / "runs",
        working_directory=working,
    )
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update({"command": command, **kwargs})
        return SimpleNamespace()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    supervisor._launch_worker(request)  # noqa: SLF001

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)
    assert environment["AUTOFORECAST_CAMPAIGN_STAGE_INDEX"] == "1"
    assert environment["AUTOFORECAST_CAMPAIGN_STAGE_TOTAL"] == "1"
    assert captured["command"][:4] == [
        sys.executable,
        "-P",
        "-m",
        "autoforecast.campaign_worker",
    ]
    assert captured["cwd"] == working


def test_single_model_worker_receives_three_stage_display_total(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = [
        _stage("v0fs0o0", "pass"),
        _stage("aggregate_selection", "pass", needs=["v0fs0o0"]),
        _stage("final_fit", "pass", needs=["aggregate_selection"]),
        _stage("final_score", "pass", needs=["final_fit"]),
        _stage("replay_report", "pass", needs=["final_score"]),
    ]
    store = _store(tmp_path, stages)
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update({"command": command, **kwargs})
        return SimpleNamespace()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    supervisor._launch_worker(request)  # noqa: SLF001

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["AUTOFORECAST_CAMPAIGN_STAGE_INDEX"] == "1"
    assert environment["AUTOFORECAST_CAMPAIGN_STAGE_TOTAL"] == "3"
    assert environment["AUTOFORECAST_WORKFLOW_LAYOUT"] == "single_model_benchmark_v1"
    assert environment["AUTOFORECAST_WORKFLOW_STAGE_ROLE"] == "validation"


def test_resume_placeholder_cannot_be_split_across_arguments(tmp_path: pathlib.Path) -> None:
    stage = _stage("train", "pass", attempts=2)
    stage["command"] = [sys.executable, "{previous_", "attempt_dir}"]
    stage["retry_mode"] = "command_managed_resume"
    with pytest.raises(campaign.CampaignError, match="must receive"):
        campaign.load_campaign_plan(_plan(tmp_path, [stage]))


def test_non_retryable_data_failure_does_not_repeat_paid_training(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("train", "pass", attempts=2)])
    store.append(
        "stage_launching",
        stage_id="train",
        attempt=1,
        worker_request_sha256="a" * 64,
    )
    store.append(
        "stage_failed",
        stage_id="train",
        attempt=1,
        worker_result_sha256="b" * 64,
        reason="non-retryable: validation labels are unavailable",
    )
    state = store.state()
    assert state["stages"][0]["status"] == "failed"
    assert state["stages"][0]["attempts"] == 1


def test_placeholder_expansion_does_not_reinterpret_replacement_text(
    tmp_path: pathlib.Path,
) -> None:
    stage = campaign.load_campaign_plan(
        _plan(tmp_path, [_stage("train", "{campaign_root}")])
    ).stages[0]
    expanded = campaign._expand_command(  # noqa: SLF001
        stage,
        campaign_root=pathlib.Path("/literal-{attempt}"),
        attempt_dir=pathlib.Path("/attempt"),
        attempt=7,
    )
    assert expanded[-1] == "/literal-{attempt}"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["campaign"].update({"surprise": True}), "exactly name and stages"),
        (
            lambda value: value["campaign"]["stages"][0].update({"surprise": True}),
            "requires exactly",
        ),
        (
            lambda value: value["campaign"]["stages"][0].update({"needs": ["missing"]}),
            "unknown dependencies",
        ),
        (
            lambda value: value["campaign"]["stages"][0].update(
                {"required_outputs": ["../escape"]}
            ),
            "without '..'",
        ),
    ],
)
def test_plan_rejects_unknown_fields_dependencies_and_unsafe_outputs(
    tmp_path: pathlib.Path, mutate, message: str
) -> None:
    path = _plan(tmp_path, [_stage("prepare", "pass")])
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match=message):
        campaign.load_campaign_plan(path)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (("resources.device", []), ("retry_mode", [])),
)
def test_plan_rejects_unhashable_enum_values_without_leaking_type_error(
    tmp_path: pathlib.Path, field: str, invalid: object
) -> None:
    stage = _stage("train", "pass")
    if field == "resources.device":
        stage["resources"]["device"] = invalid  # type: ignore[index]
    else:
        stage[field] = invalid
    with pytest.raises(campaign.CampaignError):
        campaign.load_campaign_plan(_plan(tmp_path, [stage]))


def test_store_persists_exact_plan_and_recovers_only_a_valid_stale_projection(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("prepare", "pass")])
    first_state = store.state()
    request_sha = "a" * 64
    second_state = store.append(
        "stage_launching",
        stage_id="prepare",
        attempt=1,
        worker_request_sha256=request_sha,
    )
    assert (store.root / "plan-source.yaml").read_bytes() == store.plan.source_bytes

    (store.root / "state.json").write_text(
        json.dumps(first_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert store.reconcile_state() == second_state

    (store.root / "state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="not a valid journal projection"):
        store.reconcile_state()


def test_event_chain_tampering_is_refused(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("prepare", "pass")])
    first = store.root / "events/00000001.json"
    value = json.loads(first.read_text(encoding="utf-8"))
    value["time_ns"] += 1
    first.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="journal projection"):
        store.append(
            "stage_launching",
            stage_id="prepare",
            attempt=1,
            worker_request_sha256="b" * 64,
        )


def test_event_sequence_refuses_boolean_integer_alias(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("prepare", "pass")])
    first = store.root / "events/00000001.json"
    value = json.loads(first.read_text(encoding="utf-8"))
    value["sequence"] = True
    first.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="invalid identity or chain"):
        store.events()


def test_supervisor_runs_dependency_order_and_reopens_completed_evidence(
    tmp_path: pathlib.Path,
) -> None:
    first = (
        "from pathlib import Path; "
        "Path(r'{attempt_dir}/first.txt').write_text('first', encoding='utf-8')"
    )
    second = (
        "from pathlib import Path; "
        "source=Path(r'{campaign_root}/stages/prepare/attempt-0001/first.txt'); "
        "Path(r'{attempt_dir}/second.txt').write_text(source.read_text() + '-second')"
    )
    store = _store(
        tmp_path,
        [
            _stage("prepare", first, outputs=["first.txt"]),
            _stage("train", second, needs=["prepare"], outputs=["second.txt"]),
        ],
    )
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "completed"
    assert (store.root / "stages/train/attempt-0001/second.txt").read_text() == "first-second"
    assert [event["kind"] for event in store.events()] == [
        "campaign_created",
        "stage_launching",
        "stage_completed",
        "stage_launching",
        "stage_completed",
        "campaign_completed",
    ]
    reopened = campaign.CampaignStore.from_root(store.root)
    assert reopened.state()["status"] == "completed"


def test_failed_attempt_retries_as_a_new_attempt_without_rewriting_history(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import sys; "
        "attempt=int('{attempt}'); "
        "sys.exit(7) if attempt == 1 else "
        "Path(r'{attempt_dir}/done.txt').write_text('done')"
    )
    store = _store(tmp_path, [_stage("train", code, outputs=["done.txt"], attempts=2)])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    stage = state["stages"][0]
    assert state["status"] == "completed"
    assert stage["attempts"] == 2
    assert (store.root / "stages/train/attempt-0001/stage-result.json").is_file()
    assert (store.root / "stages/train/attempt-0002/done.txt").read_text() == "done"


def test_command_managed_resume_receives_only_the_previous_attempt_path(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import sys; "
        "attempt=int('{attempt}'); previous=r'{previous_attempt_dir}'; "
        "sys.exit(7) if attempt == 1 else "
        "Path(r'{attempt_dir}/resume.txt').write_text(previous)"
    )
    stage = _stage("train", code, outputs=["resume.txt"], attempts=2)
    stage["retry_mode"] = "command_managed_resume"
    store = _store(tmp_path, [stage])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "completed"
    resumed_from = (store.root / "stages/train/attempt-0002/resume.txt").read_text()
    assert resumed_from == str(store.root / "stages/train/attempt-0001")


def test_command_managed_resume_must_receive_its_previous_attempt(tmp_path: pathlib.Path) -> None:
    stage = _stage("train", "pass", attempts=2)
    stage["retry_mode"] = "command_managed_resume"
    with pytest.raises(campaign.CampaignError, match="must receive"):
        campaign.load_campaign_plan(_plan(tmp_path, [stage]))


def test_restart_mode_may_not_receive_a_previous_attempt(tmp_path: pathlib.Path) -> None:
    stage = _stage("train", "print('{previous_attempt_dir}')")
    with pytest.raises(campaign.CampaignError, match="restart may not receive"):
        campaign.load_campaign_plan(_plan(tmp_path, [stage]))


def test_exhausted_failure_blocks_dependants_and_terminates_campaign(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(
        tmp_path,
        [
            _stage("prepare", "raise SystemExit(9)"),
            _stage("train", "raise AssertionError('must not run')", needs=["prepare"]),
        ],
    )
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    assert [stage["status"] for stage in state["stages"]] == ["failed", "blocked"]
    assert not (store.root / "stages/train/attempt-0001").exists()
    assert store.events()[-1]["kind"] == "campaign_failed"


def test_timeout_is_a_failed_terminal_result(tmp_path: pathlib.Path) -> None:
    store = _store(
        tmp_path,
        [_stage("slow", "import time; time.sleep(5)", timeout=2)],
    )
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/slow/attempt-0001/stage-result.json").read_text())
    assert result["status"] == "failed"
    assert "exceeded timeout_seconds=2" in result["reason"]


def test_worker_holds_a_shared_device_lease_and_times_out_instead_of_overlapping(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass", timeout=2)])
    lease_path = ROOT / ".autoforecast/device-leases/cpu.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with campaign._open_lock(lease_path) as lease:  # noqa: SLF001 - cross-process lock is behavior
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/train/attempt-0001/stage-result.json").read_text())
    assert "timed out waiting for exclusive cpu device" in result["reason"]


def test_host_lease_blocks_a_campaign_from_another_worktree(
    tmp_path: pathlib.Path,
) -> None:
    other_worktree = tmp_path / "other-worktree"
    other_worktree.mkdir()
    plan = _plan(tmp_path, [_stage("train", "pass", timeout=2)])
    store = campaign.CampaignStore.open_or_create(
        plan,
        output_root=tmp_path / "runs",
        working_directory=other_worktree,
    )
    with training_lease():
        state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/train/attempt-0001/stage-result.json").read_text())
    assert "timed out waiting for exclusive cpu device" in result["reason"]


def test_campaign_child_reuses_the_host_lease_without_deadlocking(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from autoforecast.training_lease import training_lease; "
        "from pathlib import Path; "
        "exec('with training_lease():\\n Path(r\"{attempt_dir}/done.txt\").write_text(\"ok\")')"
    )
    store = _store(tmp_path, [_stage("train", code, outputs=["done.txt"])])
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"


def test_host_lease_rejects_a_forged_inherited_descriptor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unrelated = tmp_path / "unrelated.lock"
    with unrelated.open("a+b") as handle:
        monkeypatch.setenv(HOST_TRAINING_LEASE_FD, str(handle.fileno()))
        with pytest.raises(TrainingLeaseError, match="unsafe"):
            with training_lease():
                pytest.fail("forged descriptor acquired the host lease")


def test_host_lease_path_uses_the_os_account_home(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_lease_module.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(tmp_path)),
    )
    assert host_training_lease_path() == tmp_path / ".autoforecast/training.lock"


def test_host_lease_public_descriptor_helpers_refuse_forgery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(HOST_TRAINING_LEASE_FD, raising=False)
    monkeypatch.setattr(
        training_lease_module.pwd, "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(tmp_path)),
    )
    assert inherited_training_lease_pass_fds() == ()
    with open_host_training_lease() as handle:
        verify_inherited_host_training_lease(handle.fileno())
        monkeypatch.setenv(HOST_TRAINING_LEASE_FD, str(handle.fileno()))
        assert inherited_training_lease_pass_fds() == (handle.fileno(),)
        with (tmp_path / "unrelated.lock").open("a+b") as unrelated:
            with pytest.raises(TrainingLeaseError, match="unsafe"):
                verify_inherited_host_training_lease(unrelated.fileno())
    monkeypatch.setenv(HOST_TRAINING_LEASE_FD, "not-an-fd")
    with pytest.raises(TrainingLeaseError, match="invalid"):
        inherited_training_lease_pass_fds()


def test_host_lease_open_refuses_a_symlink(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_lease_module.pwd, "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(tmp_path)),
    )
    path = host_training_lease_path()
    path.symlink_to(tmp_path / "unrelated.lock")
    with pytest.raises((OSError, TrainingLeaseError)):
        open_host_training_lease()


def test_with_training_lease_wraps_the_complete_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextlib.contextmanager
    def observed_lease():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(training_lease_module, "training_lease", observed_lease)

    @with_training_lease
    def work(value: int) -> int:
        events.append("work")
        return value + 1

    assert work(2) == 3
    assert events == ["enter", "work", "exit"]


def test_worker_and_child_identity_survive_a_local_hostname_change(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_sha = "a" * 64
    supervisor = campaign.CampaignSupervisor
    for filename, schema, pid_field, validator in (
        (
            "worker-identity.json",
            campaign.WORKER_IDENTITY_SCHEMA,
            "wrapper_pid",
            supervisor._worker_identity,  # noqa: SLF001
        ),
        (
            "child-identity.json",
            campaign.CHILD_IDENTITY_SCHEMA,
            "child_pid",
            supervisor._child_identity,  # noqa: SLF001
        ),
    ):
        path = tmp_path / filename
        path.write_text(
            json.dumps(
                {
                    "schema": schema,
                    "worker_request_sha256": request_sha,
                    pid_field: 123,
                    "process_create_time_ns": 1,
                    "host": "old-local-name",
                    "started_at_ns": 1,
                }
            )
            + "\n"
        )
        monkeypatch.setattr(campaign.socket, "gethostname", lambda: "new-local-name")
        assert validator(path, request_sha)["host"] == "old-local-name"
        with pytest.raises(campaign.CampaignError, match="identity is invalid"):
            validator(path, "b" * 64)


def test_worker_enforces_and_records_declared_peak_memory_ceiling(
    tmp_path: pathlib.Path,
) -> None:
    stage = _stage(
        "train",
        (
            "import time; payload=bytearray(80 * 1024 * 1024); "
            "payload[::4096]=b'x' * (len(payload) // 4096); time.sleep(2)"
        ),
        timeout=5,
    )
    stage["resources"] = {"device": "cpu", "memory_gib": 0.05, "min_free_disk_gib": 0.0}
    store = _store(tmp_path, [stage])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/train/attempt-0001/stage-result.json").read_text())
    assert "exceeded memory_gib" in result["reason"]
    assert result["peak_rss_bytes"] > 0.05 * 1024**3


def test_second_supervisor_is_refused_by_the_stable_lease(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("prepare", "pass")])
    lock_path = store.root / ".supervisor.lock"
    with campaign._CampaignLease(store.root):  # noqa: SLF001 - the process lock is the behavior
        with pytest.raises(campaign.CampaignError, match="active supervisor"):
            campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert lock_path.is_file()


def test_completed_output_tampering_is_refused_before_reuse(tmp_path: pathlib.Path) -> None:
    code = "from pathlib import Path; Path(r'{attempt_dir}/result.txt').write_text('original')"
    store = _store(tmp_path, [_stage("train", code, outputs=["result.txt"])])
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    output = store.root / "stages/train/attempt-0001/result.txt"
    output.write_text("changed", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="does not match current output bytes"):
        store.verify_completed_outputs()
    with pytest.raises(campaign.CampaignError, match="does not match current output bytes"):
        campaign.CampaignStore.from_root(store.root)


def test_terminal_notification_is_persisted_and_not_repeated(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def notify(*, campaign_name: str, status: str, path: str) -> dict[str, object]:
        calls.append(status)
        return {
            "schema": "autoforecast.campaign-terminal-notification.v1",
            "status": "sent",
            "campaign_name": campaign_name,
            "campaign_status": status,
            "configured_sinks": ["test"],
            "delivered_sinks": ["test"],
            "failures": [],
            "artifact_path": path,
        }

    monkeypatch.setattr(campaign, "campaign_notification_from_environment", notify)
    store = _store(tmp_path, [_stage("train", "pass")])
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    assert calls == ["completed"]
    record = json.loads((store.root / "terminal-notification.json").read_text())
    assert record["status"] == "sent"


def test_terminal_notification_is_not_retried_after_delivery_result_write_is_interrupted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original_write = campaign._write_new_json  # noqa: SLF001
    fail_result_once = True

    def notify(*, campaign_name: str, status: str, path: str) -> dict[str, object]:
        calls.append(status)
        return {
            "schema": "autoforecast.campaign-terminal-notification.v1",
            "status": "sent",
            "campaign_name": campaign_name,
            "campaign_status": status,
            "configured_sinks": ["test"],
            "delivered_sinks": ["test"],
            "failures": [],
            "artifact_path": path,
        }

    def interrupted_write(path: pathlib.Path, value: dict[str, object]) -> str:
        nonlocal fail_result_once
        if path.name == "terminal-notification.json" and fail_result_once:
            fail_result_once = False
            raise OSError("simulated crash after network delivery")
        return original_write(path, value)

    monkeypatch.setattr(campaign, "campaign_notification_from_environment", notify)
    monkeypatch.setattr(campaign, "_write_new_json", interrupted_write)
    store = _store(tmp_path, [_stage("train", "pass")])
    with pytest.raises(OSError, match="simulated crash"):
        campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    assert calls == ["completed"]
    result = json.loads((store.root / "terminal-notification.json").read_text())
    assert result["status"] == "indeterminate_after_interruption"


def test_missing_required_output_turns_a_zero_exit_into_failure(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("train", "pass", outputs=["missing.txt"])])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/train/attempt-0001/stage-result.json").read_text())
    assert result["exit_code"] == 0
    assert "required stage output is missing" in result["reason"]


def test_stage_cannot_forge_its_wrappers_terminal_result(tmp_path: pathlib.Path) -> None:
    code = r"""
from pathlib import Path
import hashlib, json, os
from autoforecast import campaign

attempt = Path(r"{attempt_dir}")
request_path = attempt / "worker-request.json"
request = json.loads(request_path.read_text(encoding="utf-8"))
identity = json.loads((attempt / "worker-identity.json").read_text(encoding="utf-8"))
proof = attempt / "proof.txt"
proof.write_text("child-controlled", encoding="utf-8")
result = {
    "schema": campaign.WORKER_RESULT_SCHEMA,
    "worker_request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
    "plan_sha256": request["plan_sha256"],
    "stage_id": request["stage_id"],
    "attempt": request["attempt"],
    "status": "completed",
    "exit_code": 0,
    "reason": None,
    "wrapper_pid": identity["wrapper_pid"],
    "child_pid": os.getpid(),
    "started_at_ns": identity["started_at_ns"],
    "finished_at_ns": identity["started_at_ns"] + 1,
    "peak_rss_bytes": 0,
    "artifacts": [{
        "path": "proof.txt",
        "kind": "file",
        "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    }],
}
if "worker_token_sha256" in request:
    result["worker_token"] = "0" * 64
(attempt / "stage-result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
raise SystemExit(17)
"""
    store = _store(tmp_path, [_stage("train", code, outputs=["proof.txt"])])

    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()

    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/train/attempt-0001/stage-result.json").read_text())
    assert result["status"] == "failed"
    assert result["exit_code"] == 17
    assert result["worker_token"] != "0" * 64


def test_terminal_supervisor_heartbeat_is_visible_and_status_reads_do_not_mutate_it(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    heartbeat = store.root / "supervisor-heartbeat.json"
    before = heartbeat.read_bytes()
    observation = campaign.CampaignStore.from_root(store.root).supervisor_observation()
    assert observation is not None
    assert observation["observed_state"] == "completed"
    assert observation["running_stage"] is None
    assert heartbeat.read_bytes() == before


def test_malformed_supervisor_heartbeat_is_not_reported_as_liveness(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    (store.root / "supervisor-heartbeat.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="heartbeat is invalid"):
        store.supervisor_observation()


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("session", ""),
        ("pid", -1),
        ("time_ns", -1),
        ("detail", "contradiction"),
        ("observed_state", []),
        ("running_stage", []),
    ),
)
def test_supervisor_heartbeat_requires_consistent_positive_identity(
    tmp_path: pathlib.Path,
    field: str,
    invalid: object,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    path = store.root / "supervisor-heartbeat.json"
    record = json.loads(path.read_text())
    record[field] = invalid
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="heartbeat is invalid"):
        store.supervisor_observation()


def test_restarted_supervisor_adopts_a_surviving_detached_worker(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import time; time.sleep(0.4); "
        "Path(r'{attempt_dir}/done.txt').write_text('survived')"
    )
    store = _store(tmp_path, [_stage("train", code, outputs=["done.txt"])])
    original = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = original._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    original._launch_worker(request)  # noqa: SLF001 - simulate death after worker launch
    identity = store.root / "stages/train/attempt-0001/worker-identity.json"
    deadline = time.monotonic() + 5
    while not identity.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert identity.is_file()

    recovered = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert recovered["status"] == "completed"
    assert (store.root / "stages/train/attempt-0001/done.txt").read_text() == "survived"
    assert sum(event["kind"] == "stage_launching" for event in store.events()) == 1


def test_restart_waits_for_a_token_consuming_wrapper_before_identity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    attempt = request.parent
    (attempt / "worker-token").unlink()
    worker_lock = attempt / ".worker.lock"
    worker_lock.touch()
    held = True

    def lock_is_held(path: pathlib.Path) -> bool:
        assert path == worker_lock
        return held

    def release_lock(_seconds: float) -> None:
        nonlocal held
        held = False

    monkeypatch.setattr(campaign, "_lock_is_held", lock_is_held)
    monkeypatch.setattr(campaign.time, "sleep", release_lock)

    supervisor._observe_running(store.plan.stages[0], 1)  # noqa: SLF001

    assert store.state()["stages"][0]["status"] == "failed"
    assert store.events()[-1]["reason"] == (
        "worker wrapper exited before publishing identity (status=detached)"
    )


def test_rss_measurement_failure_fails_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    store = _store(tmp_path, [_stage("train", "import time; time.sleep(2)")])
    request = campaign.CampaignSupervisor(store)._prepare_attempt(  # noqa: SLF001
        store.plan.stages[0], 1
    )
    monkeypatch.setattr(
        campaign,
        "_process_tree_rss",
        lambda _pid: (_ for _ in ()).throw(campaign.CampaignError("measurement denied")),
    )
    assert campaign.run_stage_worker(request) == 2
    result, _sha = campaign._validate_worker_result(  # noqa: SLF001
        request.parent / "stage-result.json", request, store.plan.stages[0], 1
    )
    assert result["status"] == "failed"
    assert "stage memory measurement failed: measurement denied" in result["reason"]


def test_worker_publishes_identity_before_consuming_its_launch_token(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    store = _store(tmp_path, [_stage("train", "pass")])
    request = campaign.CampaignSupervisor(store)._prepare_attempt(  # noqa: SLF001
        store.plan.stages[0], 1
    )
    real_load = campaign._load_worker_token  # noqa: SLF001

    def load_after_identity(path: pathlib.Path, *, consume: bool) -> str:
        assert (path.parent / "worker-identity.json").is_file()
        return real_load(path, consume=consume)

    monkeypatch.setattr(campaign, "_load_worker_token", load_after_identity)

    assert campaign.run_stage_worker(request) == 0


def test_restart_reconstructs_a_missing_request_after_launch_intent(
    tmp_path: pathlib.Path,
) -> None:
    code = "from pathlib import Path; Path(r'{attempt_dir}/done.txt').write_text('recovered')"
    store = _store(tmp_path, [_stage("train", code, outputs=["done.txt"])])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    request.unlink()

    recovered = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert recovered["status"] == "completed"
    assert request.is_file()
    assert (request.parent / "done.txt").read_text() == "recovered"


def test_restart_terminates_a_lock_verified_worker_with_a_stale_heartbeat(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass", timeout=40)])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    digest = campaign._digest_path(request)  # noqa: SLF001
    attempt = request.parent
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,sys,time; f=open(sys.argv[1], 'a+b'); "
                "fcntl.flock(f.fileno(), fcntl.LOCK_EX); time.sleep(60)"
            ),
            str(attempt / ".worker.lock"),
        ],
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while not campaign._lock_is_held(attempt / ".worker.lock") and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.01)
    (attempt / "worker-identity.json").write_text(
        json.dumps(
            {
                "schema": campaign.WORKER_IDENTITY_SCHEMA,
                "worker_request_sha256": digest,
                "wrapper_pid": holder.pid,
                "process_create_time_ns": campaign._observed_process_create_time_ns(  # noqa: SLF001
                    holder.pid
                ),
                "host": socket.gethostname(),
                "started_at_ns": time.time_ns() - 100_000_000_000,
            },
            indent=2,
        )
        + "\n"
    )
    (attempt / "heartbeat.json").write_text(
        json.dumps(
            {
                "schema": campaign.WORKER_HEARTBEAT_SCHEMA,
                "worker_request_sha256": digest,
                "wrapper_pid": holder.pid,
                "child_pid": None,
                "phase": "waiting_for_device",
                "rss_bytes": 1,
                "time_ns": time.time_ns() - 100_000_000_000,
            },
            indent=2,
        )
        + "\n"
    )
    try:
        state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
        assert state["status"] == "failed"
        assert holder.wait(timeout=5) == -signal.SIGTERM
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=5)


def test_forged_result_does_not_suppress_stale_worker_recovery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, [_stage("train", "pass", timeout=40)])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    digest = campaign._digest_path(request)  # noqa: SLF001
    attempt = request.parent
    worker_lock = attempt / ".worker.lock"
    worker_lock.touch()
    (attempt / "worker-identity.json").write_text(
        json.dumps(
            {
                "schema": campaign.WORKER_IDENTITY_SCHEMA,
                "worker_request_sha256": digest,
                "wrapper_pid": 123,
                "process_create_time_ns": 1,
                "host": socket.gethostname(),
                "started_at_ns": 1,
            }
        )
        + "\n"
    )
    (attempt / "stage-result.json").write_text("{}\n", encoding="utf-8")
    held = True
    terminated = False

    def lock_is_held(path: pathlib.Path) -> bool:
        assert path == worker_lock
        return held

    def terminate(_pid: int, _created_at_ns: int) -> None:
        nonlocal held, terminated
        held = False
        terminated = True

    monkeypatch.setattr(campaign, "_lock_is_held", lock_is_held)
    monkeypatch.setattr(campaign, "_require_lease_holder", lambda *_args: None)
    monkeypatch.setattr(campaign, "_terminate_pid_group", terminate)
    monkeypatch.setattr(supervisor, "_terminate_stranded_child", lambda *_args: None)
    monkeypatch.setattr(
        campaign.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("supervisor spun")),
    )

    with pytest.raises(campaign.CampaignError, match="worker result fields or schema differ"):
        supervisor._observe_running(store.plan.stages[0], 1)  # noqa: SLF001

    assert terminated


def test_stale_worker_identity_cannot_target_a_process_that_does_not_hold_its_lease(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, [_stage("train", "pass", timeout=40)])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    digest = campaign._digest_path(request)  # noqa: SLF001
    attempt = request.parent
    (attempt / ".worker.lock").touch()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    monkeypatch.setattr(campaign, "_lock_is_held", lambda _path: True)
    monkeypatch.setattr(campaign, "_lease_holder_pids", lambda _path: {holder.pid})
    (attempt / "worker-identity.json").write_text(
        json.dumps(
            {
                "schema": campaign.WORKER_IDENTITY_SCHEMA,
                "worker_request_sha256": digest,
                "wrapper_pid": victim.pid,
                "process_create_time_ns": campaign._observed_process_create_time_ns(  # noqa: SLF001
                    victim.pid
                ),
                "host": socket.gethostname(),
                "started_at_ns": 1,
            }
        )
        + "\n"
    )
    try:
        with pytest.raises(campaign.CampaignError, match="does not own its campaign lease"):
            campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
        assert victim.poll() is None
    finally:
        for process in (holder, victim):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_stranded_child_identity_cannot_target_a_process_without_the_child_lease(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    digest = campaign._digest_path(request)  # noqa: SLF001
    attempt = request.parent
    (attempt / ".child.lock").touch()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    victim = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    monkeypatch.setattr(campaign, "_lock_is_held", lambda _path: True)
    monkeypatch.setattr(campaign, "_lease_holder_pids", lambda _path: {holder.pid})
    (attempt / "child-identity.json").write_text(
        json.dumps(
            {
                "schema": campaign.CHILD_IDENTITY_SCHEMA,
                "worker_request_sha256": digest,
                "child_pid": victim.pid,
                "process_create_time_ns": campaign._observed_process_create_time_ns(  # noqa: SLF001
                    victim.pid
                ),
                "host": socket.gethostname(),
                "started_at_ns": 1,
            }
        )
        + "\n"
    )
    try:
        with pytest.raises(campaign.CampaignError, match="does not own its campaign lease"):
            supervisor._terminate_stranded_child(attempt, digest)  # noqa: SLF001
        assert victim.poll() is None
    finally:
        for process in (holder, victim):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_detached_process_cleanup_refuses_a_reused_pid_identity(
    tmp_path: pathlib.Path,
) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        observed = campaign._observed_process_create_time_ns(holder.pid)  # noqa: SLF001
        assert observed is not None
        with pytest.raises(campaign.CampaignError, match="reused PID"):
            campaign._terminate_pid_group(holder.pid, observed + 1)  # noqa: SLF001
        assert holder.poll() is None
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_wrapper_sigkill_keeps_device_leased_until_supervisor_kills_child_and_retries(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import time; attempt=int('{attempt}'); "
        "time.sleep(60) if attempt == 1 else "
        "Path(r'{attempt_dir}/done.txt').write_text('recovered')"
    )
    store = _store(
        tmp_path,
        [_stage("train", code, outputs=["done.txt"], attempts=2, timeout=90)],
    )
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    wrapper = supervisor._launch_worker(request)  # noqa: SLF001
    attempt = request.parent
    child_identity = attempt / "child-identity.json"
    deadline = time.monotonic() + 10
    while not child_identity.exists() and time.monotonic() < deadline:
        if wrapper.poll() is not None:
            raise AssertionError("worker exited before publishing child identity")
        time.sleep(0.01)
    assert child_identity.is_file()
    first_child_pid = json.loads(child_identity.read_text())["child_pid"]
    os.killpg(wrapper.pid, signal.SIGKILL)
    wrapper.wait(timeout=5)
    lease_path = ROOT / ".autoforecast/device-leases/cpu.lock"
    with campaign._open_lock(lease_path) as lease:  # noqa: SLF001
        with pytest.raises(BlockingIOError):
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with campaign._open_lock(host_training_lease_path()) as lease:  # noqa: SLF001
        with pytest.raises(BlockingIOError):
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "completed"
    assert state["stages"][0]["attempts"] == 2
    assert not campaign._pid_alive(first_child_pid)  # noqa: SLF001
    assert (store.root / "stages/train/attempt-0002/done.txt").read_text() == "recovered"


def test_successful_stage_kills_daemonized_descendants_and_releases_inherited_leases(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import os,time; pid=os.fork(); "
        "(time.sleep(60), os._exit(0)) if pid == 0 else "
        "Path(r'{attempt_dir}/background.pid').write_text(str(pid))"
    )
    store = _store(tmp_path, [_stage("train", code, outputs=["background.pid"])])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "completed"
    background = int((store.root / "stages/train/attempt-0001/background.pid").read_text())
    deadline = time.monotonic() + 5
    while campaign._pid_alive(background) and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.05)
    assert not campaign._pid_alive(background)  # noqa: SLF001
    lease_path = ROOT / ".autoforecast/device-leases/cpu.lock"
    with campaign._open_lock(lease_path) as lease:  # noqa: SLF001
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_successful_stage_kills_descendant_that_escaped_its_process_group(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import os,time; pid=os.fork(); "
        "(os.setsid(), time.sleep(60), os._exit(0)) if pid == 0 else "
        "Path(r'{attempt_dir}/background.pid').write_text(str(pid))"
    )
    store = _store(tmp_path, [_stage("train", code, outputs=["background.pid"])])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "completed"
    background = int((store.root / "stages/train/attempt-0001/background.pid").read_text())
    deadline = time.monotonic() + 5
    while campaign._pid_alive(background) and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.05)
    assert not campaign._pid_alive(background)  # noqa: SLF001
    with campaign._open_lock(ROOT / ".autoforecast/device-leases/cpu.lock") as lease:  # noqa: SLF001
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_restart_refuses_a_malformed_terminal_result(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    supervisor = campaign.CampaignSupervisor(store, poll_seconds=0.01)
    request = supervisor._prepare_attempt(store.plan.stages[0], 1)  # noqa: SLF001
    (request.parent / "stage-result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="fields or schema differ"):
        campaign.CampaignSupervisor(store, poll_seconds=0.01).run()


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("worker_token", "0" * 64),
        ("attempt", True),
        ("status", []),
        ("exit_code", False),
        ("wrapper_pid", "not-a-pid"),
        ("child_pid", []),
        ("started_at_ns", "yesterday"),
        ("finished_at_ns", None),
        ("peak_rss_bytes", -1),
        ("artifacts", {}),
    ),
)
def test_worker_result_refuses_every_malformed_terminal_field(
    tmp_path: pathlib.Path,
    field: str,
    invalid: object,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    stage = store.plan.stages[0]
    attempt = store.root / "stages/train/attempt-0001"
    result_path = attempt / "stage-result.json"
    result = json.loads(result_path.read_text())
    result[field] = invalid
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="worker result"):
        campaign._validate_worker_result(  # noqa: SLF001
            result_path,
            attempt / "worker-request.json",
            stage,
            1,
        )


def test_resource_admission_allows_available_memory_below_stage_limit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    assert campaign.CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"


def test_resource_admission_still_refuses_insufficient_disk(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _stage("train", "pass")
    stage["resources"]["min_free_disk_gib"] = 2.0
    store = _store(tmp_path, [stage])
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))
    with pytest.raises(campaign.CampaignError, match="free disk"):
        campaign.CampaignSupervisor(store, poll_seconds=0.01).run()


def test_real_supervisor_process_restart_adopts_its_surviving_worker(
    tmp_path: pathlib.Path,
) -> None:
    code = (
        "from pathlib import Path; import time; time.sleep(1.0); "
        "Path(r'{attempt_dir}/done.txt').write_text('survived-real-restart')"
    )
    plan_path = _plan(tmp_path, [_stage("train", code, outputs=["done.txt"])])
    plan = campaign.load_campaign_plan(plan_path)
    output_root = tmp_path / "runs"
    campaign_root = output_root / f"{plan.name}-{plan.source_sha256[:12]}"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_campaign.py"),
        "run",
        str(plan_path),
        "--output-root",
        str(output_root),
        "--working-directory",
        str(ROOT),
        "--poll-seconds",
        "0.01",
    ]
    original = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        identity = campaign_root / "stages/train/attempt-0001/worker-identity.json"
        deadline = time.monotonic() + 10
        while not identity.exists() and time.monotonic() < deadline:
            if original.poll() is not None:
                stdout, stderr = original.communicate()
                raise AssertionError(f"supervisor ended before worker identity: {stdout}{stderr}")
            time.sleep(0.01)
        assert identity.is_file()
        original.terminate()
        original.wait(timeout=5)

        recovered = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        assert json.loads(recovered.stdout)["status"] == "completed"
        assert (
            campaign_root / "stages/train/attempt-0001/done.txt"
        ).read_text() == "survived-real-restart"
    finally:
        if original.poll() is None:
            original.terminate()
            original.wait(timeout=5)


def test_supervisor_lock_refuses_a_symlink_without_touching_its_target(
    tmp_path: pathlib.Path,
) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    target = tmp_path / "outside.lock"
    target.write_text("untouched", encoding="utf-8")
    lock = store.root / ".supervisor.lock"
    lock.unlink()
    lock.symlink_to(target)
    with pytest.raises(campaign.CampaignError, match="lock path is unsafe"):
        campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert target.read_text(encoding="utf-8") == "untouched"


def test_campaign_store_refuses_a_symlinked_root(tmp_path: pathlib.Path) -> None:
    store = _store(tmp_path, [_stage("train", "pass")])
    link = tmp_path / "campaign-link"
    link.symlink_to(store.root, target_is_directory=True)
    with pytest.raises(campaign.CampaignError, match="not a regular directory"):
        campaign.CampaignStore.from_root(link)


def test_required_output_refuses_a_symlinked_parent(tmp_path: pathlib.Path) -> None:
    outside = tmp_path / "outside"
    code = (
        "from pathlib import Path; import os; "
        f"outside=Path(r'{outside}'); outside.mkdir(); "
        "(outside/'value.txt').write_text('outside'); "
        "os.symlink(outside, Path(r'{attempt_dir}/linked'))"
    )
    store = _store(tmp_path, [_stage("train", code, outputs=["linked/value.txt"])])
    state = campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert state["status"] == "failed"
    result = json.loads((store.root / "stages/train/attempt-0001/stage-result.json").read_text())
    assert "missing or linked" in result["reason"]
    assert (outside / "value.txt").read_text() == "outside"


def test_worker_cannot_advance_after_modifying_the_frozen_plan(tmp_path: pathlib.Path) -> None:
    code = (
        "from pathlib import Path; Path(r'{campaign_root}/plan-source.yaml').write_text('changed')"
    )
    store = _store(tmp_path, [_stage("train", code)])
    with pytest.raises(campaign.CampaignError, match="plan bytes differ"):
        campaign.CampaignSupervisor(store, poll_seconds=0.01).run()
    assert store.events()[-1]["kind"] == "stage_launching"
