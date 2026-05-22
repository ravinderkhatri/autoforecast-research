#!/usr/bin/env python3
"""Tests for inference benchmark identity, timing scope and comparison rules."""

from __future__ import annotations

import hashlib
import math
import pathlib
import subprocess
import sys
import tempfile
import unicodedata
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import bench_inference as bench  # noqa: E402


def test_sampler_stage_timers_scope_exactly_the_declared_work() -> None:
    """A work-based clock: each timer must contain exactly the stage it names, nothing else."""
    clock = [0.0]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.apply_calls = 0

        def encode_base(self, **inputs):
            return SimpleNamespace(group_index=torch.tensor([0, 1]), horizon=2)

        def draw_factor_paths(self, n_groups, horizon, n_draws, *, generator):
            clock[0] += 0.125
            return torch.rand(n_draws, n_groups, horizon, 1, generator=generator)

        def apply_factor(self, base, *, n_draws, factor_path):
            clock[0] += 0.125
            self.apply_calls += 1
            shape = (n_draws, 2, base.horizon)
            return (
                torch.full(shape, 0.5),
                torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
                + float(self.apply_calls),
                torch.full(shape, 1.5),
            )

    def fake_perf_counter():
        return clock[0]

    def fake_time(fn, reps=5):
        fn()
        return {"median_s": 0.5, "p95_s": 0.5, "min_s": 0.5, "n_reps": reps}

    def fake_predicted_paths(p, mu, k, n_draws, *, generator):
        clock[0] += 1.0
        return torch.rand(*mu.shape, n_draws, generator=generator)

    def fake_full_paths(received_model, received_inputs, n_paths, **kwargs):
        clock[0] += 7.0
        return torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    real_fingerprint = bench._tensor_fingerprint  # noqa: SLF001

    def slow_fingerprint(values):
        clock[0] += 100.0  # publication work must stay OUTSIDE every timer
        return real_fingerprint(values)

    def fake_support_size(mu, _k):
        return int(torch.ceil(mu.max()).item()) + 1

    def fake_support_sizes(mu, _k):
        return torch.ceil(mu).to(torch.long) + 1

    with (
        mock.patch.object(bench, "PATH_CELLS", 2),
        mock.patch.object(bench, "SAMPLER_FACTOR_FIXTURES", 2),
        mock.patch.object(bench, "benchmark_inputs",
                          return_value=({"history": torch.ones(2, 2)}, {"tensor_sha256": "f"})),
        mock.patch.object(bench, "_time", side_effect=fake_time),
        mock.patch.object(bench.time, "perf_counter", side_effect=fake_perf_counter),
        mock.patch.object(bench.ours, "predicted_paths", side_effect=fake_predicted_paths),
        mock.patch.object(bench.ours, "draw_paths_over_factor", side_effect=fake_full_paths),
        mock.patch.object(bench, "_full_stochastic_input_fingerprints",
                          return_value={"sealed": True}),
        mock.patch.object(bench, "_tensor_fingerprint", side_effect=slow_fingerprint),
        mock.patch.object(bench, "support_size", side_effect=fake_support_size),
        mock.patch.object(bench, "support_sizes", side_effect=fake_support_sizes),
    ):
        result = bench.sampler_stage(
            FakeModel(), {"horizon": 2, "head_kind": "hurdle_negbin"}, {})

    assert result["factor_and_head_one_draw"]["mean_s"] == 0.25, result["factor_and_head_one_draw"]
    assert result["sample_one_path"]["mean_s"] == 1.0, result["sample_one_path"]
    assert result["paired_one_path"]["mean_s"] == 1.25, result["paired_one_path"]
    assert result["measured_1000_paths"]["elapsed_s"] == 7.0, result["measured_1000_paths"]
    assert result["projected_1000_draws_s"] == 0.5 + 1000 * 1.25
    # Hurdle parameters (p present) must use the hurdle support rule, per fixture and overall.
    assert result["support_by_factor_fixture"] == [6, 7], result["support_by_factor_fixture"]
    assert result["support_required"] == 7
    # Shared table price: PMF and CDF, both float64, PATH_CELLS x horizon x support.
    for row in result["batch_max_mu_sweep"]:
        if row["shared_support"] is not None:
            assert row["shared_table_gb"] == 2 * 2 * row["shared_support"] * 2 * 8 / 1e9, row


def test_comparison_refuses_a_different_benchmark_contract_version() -> None:
    mismatches = bench._comparison_mismatches(  # noqa: SLF001
        {"benchmark_contract": 1}, {"benchmark_contract": bench.BENCHMARK_CONTRACT})
    assert any(line.startswith("benchmark_contract") for line in mismatches), mismatches
    legacy = {"point_loop": {}, "path_loop": {}}
    assert any(line.startswith("benchmark_contract")
               for line in bench._comparison_mismatches(  # noqa: SLF001
                   legacy, {"benchmark_contract": bench.BENCHMARK_CONTRACT}))


def test_tensor_fingerprint_binds_dtype_when_payload_bytes_collide() -> None:
    as_int = torch.tensor([1, 2], dtype=torch.int32)
    as_float = as_int.view(torch.float32)  # identical bytes, different dtype
    assert torch.equal(as_float.view(torch.int32), as_int)
    assert bench._tensor_fingerprint({"a": as_int}) != bench._tensor_fingerprint({"a": as_float})  # noqa: SLF001


def test_git_identity_seals_untracked_file_content_and_excludes_only_the_output() -> None:
    probe = bench.ROOT / "bench_identity_untracked_probe.txt"
    if probe.exists() or probe.is_symlink():
        raise AssertionError(f"identity probe path already exists: {probe}")
    try:
        before = bench.identity()
        probe.write_text("generation one\n")
        first = bench.identity()
        probe.write_text("generation two\n")
        second = bench.identity()
        excluded = bench.identity(probe)
    finally:
        probe.unlink(missing_ok=True)
    assert first["git_untracked_files"] == before["git_untracked_files"] + 1
    assert first["git_dirty_sha256"] is not None
    assert first["git_dirty_sha256"] != second["git_dirty_sha256"], "untracked bytes not sealed"
    assert excluded["git_untracked_files"] == before["git_untracked_files"]
    assert excluded["git_dirty_sha256"] == before["git_dirty_sha256"]
    assert excluded["git_identity_excluded_output"] == "bench_identity_untracked_probe.txt"


def test_git_identity_refuses_non_sha_and_tracked_symlink() -> None:
    def non_sha_git(command, **kwargs):
        args = command[1:]
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=b"not-a-sha\n", stderr=b"")
        raise AssertionError(f"identity continued past a non-SHA head: {args}")

    with mock.patch.object(bench.subprocess, "run", side_effect=non_sha_git):
        try:
            bench.identity()
        except RuntimeError as exc:
            assert "non-SHA" in str(exc)
        else:
            raise AssertionError("a non-SHA git identity was accepted")

    sha = ("a" * 40 + "\n").encode()

    def tracked_symlink_git(command, **kwargs):
        args = command[1:]
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=sha, stderr=b"")
        if args[0] == "diff" or args[:2] == ["ls-files", "--others"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if args[:2] == ["ls-files", "-s"]:
            return SimpleNamespace(
                returncode=0, stdout=b"120000 " + b"b" * 40 + b" 0\tlinked.py\0", stderr=b"")
        raise AssertionError(f"identity continued past a tracked symlink: {args}")

    with mock.patch.object(bench.subprocess, "run", side_effect=tracked_symlink_git):
        try:
            bench.identity()
        except RuntimeError as exc:
            assert "tracked symlink" in str(exc)
        else:
            raise AssertionError("a tracked symlink was sealed by its readlink text only")


def test_publish_json_refuses_symlink_target_and_readback_mismatch() -> None:
    with tempfile.TemporaryDirectory() as directory:
        real = pathlib.Path(directory) / "real.json"
        real.write_text("{}")
        link = pathlib.Path(directory) / "link.json"
        link.symlink_to(real)
        try:
            bench._publish_json(link, {"x": 1})  # noqa: SLF001
        except RuntimeError as exc:
            assert "must not be a symlink" in str(exc)
        else:
            raise AssertionError("a symlink output was replaced")
        assert real.read_text() == "{}" and link.is_symlink()

        target = pathlib.Path(directory) / "record.json"
        target.write_text('{"generation":"OLD"}\n')
        original_load = bench.load_exact_json

        def corrupt_readback(path):
            if path.resolve() == target.resolve():
                return {"generation": "CORRUPT"}
            return original_load(path)

        with mock.patch.object(bench, "load_exact_json", side_effect=corrupt_readback):
            try:
                bench._publish_json(target, {"generation": "NEW"})  # noqa: SLF001
            except RuntimeError as exc:
                assert "differs from the fsynced payload" in str(exc)
            else:
                raise AssertionError("a readback that differs from the payload was published")
        assert target.read_text() == '{"generation":"OLD"}\n'
        assert sorted(p.name for p in pathlib.Path(directory).iterdir()) == [
            "link.json", "real.json", "record.json"]


def test_benchmark_publication_and_baseline_refuse_ambiguous_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        target = root / "record.json"
        real_replace = bench.os.replace

        def poison_after_replace(source, destination):
            real_replace(source, destination)
            if pathlib.Path(source).suffix == ".tmp":
                pathlib.Path(destination).write_text('{"metric":1,"metric":2}')

        with mock.patch.object(bench.os, "replace", side_effect=poison_after_replace):
            try:
                bench._publish_json(target, {"metric": 1})  # noqa: SLF001
            except RuntimeError as exc:
                assert "duplicate JSON object key" in str(exc), str(exc)
            else:
                raise AssertionError("ambiguous post-replace benchmark readback was accepted")
        assert not target.exists()

        for index, payload in enumerate((
                '{"metric":1,"metric":2}', '{"metric":NaN}', '["not", "a", "record"]')):
            baseline = root / f"baseline-{index}.json"
            baseline.write_text(payload)
            try:
                bench._load_benchmark_baseline(baseline)  # noqa: SLF001
            except RuntimeError as exc:
                assert "benchmark baseline" in str(exc), str(exc)
            else:
                raise AssertionError(f"ambiguous benchmark baseline was accepted: {payload}")


def test_benchmark_inputs_refuses_zero_cells() -> None:
    cfg = {"encoder_days": 2, "horizon": 2, "n_attributes": 1, "n_series": 2, "n_products": 2,
           "n_stores": 1, "n_known": 1, "n_commercial": 0, "text_dim": 0}
    try:
        bench.benchmark_inputs(cfg, 0, {})
    except ValueError as exc:
        assert ">= 1" in str(exc)
    else:
        raise AssertionError("zero benchmark cells were accepted")


def test_main_refuses_output_equal_to_checkpoint_and_seals_identity_before_load() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = pathlib.Path(directory) / "model.pt"
        checkpoint.write_bytes(b"ckpt")
        order: list[str] = []

        def fake_identity(output=None):
            order.append("identity")
            return {"git_sha": "a" * 40}

        def fake_load(path):
            order.append("load")
            raise RuntimeError("stop after load")

        with (
            mock.patch.object(bench.sys, "argv",
                              ["bench_inference.py", str(checkpoint), "--json", str(checkpoint)]),
            mock.patch.object(bench, "identity", side_effect=fake_identity),
            mock.patch.object(bench.ours, "load_checkpoint", side_effect=fake_load),
        ):
            try:
                bench.main()
            except RuntimeError as exc:
                assert "must not overwrite" in str(exc), exc
            else:
                raise AssertionError("the benchmark overwrote its own checkpoint")
        assert order == []

        with (
            mock.patch.object(bench.sys, "argv", ["bench_inference.py", str(checkpoint)]),
            mock.patch.object(bench, "identity", side_effect=fake_identity),
            mock.patch.object(bench.ours, "load_checkpoint", side_effect=fake_load),
        ):
            try:
                bench.main()
            except RuntimeError as exc:
                assert str(exc) == "stop after load"
        assert order == ["identity", "load"], order


def test_checkpoint_output_refuses_samefile_aliases_supported_by_this_filesystem() -> None:
    """Hard links always alias; case and Unicode aliases are checked when the volume has them."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        checkpoint = root / "model-café.pt"
        checkpoint.write_bytes(b"checkpoint")
        hard_link = root / "record.json"
        hard_link.hardlink_to(checkpoint)
        candidates = [hard_link]
        for candidate in (
            root / checkpoint.name.swapcase(),
            root / unicodedata.normalize("NFD", checkpoint.name),
        ):
            if candidate != checkpoint and candidate.exists() and candidate.samefile(checkpoint):
                candidates.append(candidate)

        for alias in candidates:
            try:
                bench._require_output_separate_from_checkpoint(alias, checkpoint)  # noqa: SLF001
            except RuntimeError as exc:
                assert "must not overwrite" in str(exc), str(exc)
            else:
                raise AssertionError(f"checkpoint filesystem alias {alias} was accepted")


def test_baseline_output_refuses_literal_symlink_and_hardlink_aliases() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        baseline = root / "baseline.json"
        baseline.write_text('{"sealed":true}\n')
        hard_link = root / "hard-link.json"
        hard_link.hardlink_to(baseline)
        symbolic_link = root / "symbolic-link.json"
        symbolic_link.symlink_to(baseline)

        for output in (baseline, hard_link, symbolic_link):
            try:
                bench._require_output_separate_from_baseline(output, baseline)  # noqa: SLF001
            except RuntimeError as exc:
                assert "comparison baseline" in str(exc), str(exc)
            else:
                raise AssertionError(f"baseline filesystem alias {output} was accepted")


def test_baseline_snapshot_binds_exact_bytes_and_detects_midrun_replacement() -> None:
    with tempfile.TemporaryDirectory() as directory:
        baseline = pathlib.Path(directory) / "baseline.json"
        original = b'{"identity":{"git_sha":"old"}}\n'
        baseline.write_bytes(original)

        record, digest = bench._load_benchmark_baseline_snapshot(baseline)  # noqa: SLF001
        assert record == {"identity": {"git_sha": "old"}}
        assert digest == hashlib.sha256(original).hexdigest()

        baseline.write_bytes(b'{"identity":{"git_sha":"new"}}\n')
        try:
            bench._require_baseline_unchanged(baseline, digest)  # noqa: SLF001
        except RuntimeError as exc:
            assert "baseline changed during the run" in str(exc), str(exc)
        else:
            raise AssertionError("a replaced benchmark baseline retained its old authority")


def test_git_identity_marks_a_clean_tracked_tree_dirty_when_untracked_files_exist() -> None:
    """With git reporting NO tracked diff, one untracked file alone must make the seal dirty."""
    probe = bench.ROOT / "bench_identity_untracked_only_probe.txt"
    if probe.exists() or probe.is_symlink():
        raise AssertionError(f"identity probe path already exists: {probe}")
    original_run = bench.subprocess.run

    def fake_git(command, **kwargs):
        if command[0] != "git":
            return original_run(command, **kwargs)
        args = command[1:]
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=("c" * 40 + "\n").encode(), stderr=b"")
        if args[0] == "diff" or args[:2] == ["ls-files", "-s"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if args[:2] == ["ls-files", "--others"]:
            return SimpleNamespace(returncode=0, stdout=probe.name.encode() + b"\0", stderr=b"")
        raise AssertionError(f"unexpected git call {args}")

    probe.write_text("untracked only\n")
    try:
        with mock.patch.object(bench.subprocess, "run", side_effect=fake_git):
            sealed = bench.identity()
    finally:
        probe.unlink(missing_ok=True)
    assert sealed["git_tracked_diff_bytes"] == 0
    assert sealed["git_untracked_files"] == 1
    assert sealed["git_dirty_sha256"] is not None, "an untracked executable left the seal clean"


def test_git_identity_uses_literal_output_paths_and_refuses_tracked_aliases() -> None:
    """Every Git metacharacter is a filename; no filesystem alias may replace source."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory).resolve()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                       cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
        tracked = root / "source.py"
        tracked.write_text("ORIGINAL = True\n")
        unicode_tracked = root / "café.py"
        unicode_tracked.write_text("ORIGINAL = True\n")
        (root / "nested").mkdir()
        (root / "nested" / "source.py").write_text("ORIGINAL = True\n")
        for name in ("a", "b", "m", "z"):
            (root / name).mkdir()
            (root / name / "source.py").write_text("ORIGINAL = True\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        tracked.write_text("MUTATED = True\n")

        with (
            mock.patch.object(bench, "ROOT", root),
            mock.patch.object(bench, "_machine_hardware", return_value={"fixture": True}),
        ):
            baseline = bench.identity()
            assert baseline["git_tracked_diff_bytes"] > 0
            for spelling in ("*", "**", "?", "[a]", "[a-z]", "[!a]", "[^a]",
                             "[[:alpha:]]"):
                sealed = bench.identity(root / spelling)
                assert sealed["git_tracked_diff_bytes"] == baseline["git_tracked_diff_bytes"], (
                    f"literal output name {spelling!r} hid a tracked source modification")
            for invalid in (tracked, root / "nested"):
                try:
                    bench.identity(invalid)
                except RuntimeError as exc:
                    assert ("tracked" in str(exc) or "directory" in str(exc)), str(exc)
                else:
                    raise AssertionError(f"benchmark output replaced invalid path {invalid}")

            hard_link = root / "hard-link-output.json"
            hard_link.hardlink_to(tracked)
            aliases = [hard_link]
            for source, candidate in (
                (tracked, root / "SOURCE.PY"),
                (unicode_tracked, root / unicodedata.normalize("NFD", unicode_tracked.name)),
            ):
                if candidate != source and candidate.exists() and candidate.samefile(source):
                    aliases.append(candidate)
            for alias in aliases:
                try:
                    bench.identity(alias)
                except RuntimeError as exc:
                    assert "tracked" in str(exc) and "alias" in str(exc), str(exc)
                else:
                    raise AssertionError(f"tracked filesystem alias {alias} was accepted")


def _init_git_fixture(root: pathlib.Path) -> None:
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    (root / "source.py").write_text("ORIGINAL = True\n")
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)


def _git_absolute(root: pathlib.Path, option: str) -> pathlib.Path:
    value = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", option], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return pathlib.Path(value)


def _assert_git_admin_outputs_refused(root: pathlib.Path) -> None:
    git_dir = _git_absolute(root, "--git-dir")
    common_dir = _git_absolute(root, "--git-common-dir")
    targets = {
        root / ".git",
        git_dir,
        git_dir / "index",
        git_dir / "HEAD",
        git_dir / "future" / "record.json",
        common_dir,
        common_dir / "HEAD",
        common_dir / "refs" / "heads" / "main",
    }
    with (
        mock.patch.object(bench, "ROOT", root),
        mock.patch.object(bench, "_machine_hardware", return_value={"fixture": True}),
    ):
        for target in targets:
            try:
                bench.identity(target)
            except RuntimeError as exc:
                assert "Git administrative" in str(exc), (target, str(exc))
            else:
                raise AssertionError(f"Git administrative output target was accepted: {target}")


def test_git_identity_refuses_normal_repository_administrative_paths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory) / "main"
        _init_git_fixture(root)
        root = root.resolve()
        _assert_git_admin_outputs_refused(root)


def test_git_identity_refuses_linked_worktree_and_common_administrative_paths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = pathlib.Path(directory)
        main = base / "main"
        linked = base / "linked"
        _init_git_fixture(main)
        subprocess.run(
            ["git", "worktree", "add", "-q", "-b", "linked-fixture", str(linked)],
            cwd=main, check=True,
        )
        linked = linked.resolve()
        assert (linked / ".git").is_file(), "fixture did not create a linked-worktree gitfile"
        assert _git_absolute(linked, "--git-dir") != _git_absolute(linked, "--git-common-dir")
        _assert_git_admin_outputs_refused(linked)


def test_git_identity_fails_closed_when_admin_directories_cannot_be_resolved() -> None:
    sha = ("a" * 40 + "\n").encode()

    def failing_git(command, **_kwargs):
        args = command[1:]
        if args == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=sha, stderr=b"")
        if args == ["rev-parse", "--path-format=absolute", "--git-dir"]:
            return SimpleNamespace(returncode=128, stdout=b"", stderr=b"admin unavailable")
        raise AssertionError(f"unexpected command after Git-admin failure: {args}")

    with mock.patch.object(bench.subprocess, "run", side_effect=failing_git):
        try:
            bench.identity(bench.ROOT / "benchmark-output.json")
        except RuntimeError as exc:
            assert "admin unavailable" in str(exc), str(exc)
        else:
            raise AssertionError("a failed Git administrative-directory lookup was accepted")


def test_git_identity_excludes_untracked_output_filesystem_alias() -> None:
    """Use Git's actual spelling when APFS resolves a requested case/Unicode alias."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory) / "main"
        _init_git_fixture(root)
        root = root.resolve()
        output = root / "result-café.json"
        output.write_text('{"old":true}\n')
        candidates = (
            root / output.name.upper(),
            root / unicodedata.normalize("NFD", output.name),
        )
        supported = [candidate for candidate in candidates
                     if candidate != output and candidate.exists()
                     and candidate.samefile(output)]
        with (
            mock.patch.object(bench, "ROOT", root),
            mock.patch.object(bench, "_machine_hardware", return_value={"fixture": True}),
        ):
            canonical = bench.identity(output)
            assert canonical["git_untracked_files"] == 0
            assert canonical["git_identity_excluded_output"] == output.name
            for alias in supported:
                sealed = bench.identity(alias)
                assert sealed["git_untracked_files"] == 0, alias
                assert sealed["git_dirty_sha256"] == canonical["git_dirty_sha256"], alias
                assert sealed["git_identity_excluded_output"] == output.name, alias


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
