#!/usr/bin/env python3
"""Fast contract tests for ``bench_inference``; the full timing run remains a manual benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import bench_inference as bench  # noqa: E402


def _fixture() -> tuple[dict, dict]:
    cfg = {
        "encoder_days": 4,
        "horizon": 3,
        "n_attributes": 2,
        "n_series": 4,
        "n_products": 4,
        "n_stores": 3,
        "n_known": 2,
        "n_commercial": 0,
        "text_dim": 0,
        "price_index": None,
        "discount_available_index": None,
        "event_distance_index": (),
        "event_present_index": (),
    }
    record = {"run": {
        "series_ids": ["a", "b", "c"],
        "tensors": {
            "attributes": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "group_index": torch.tensor([10, 20, 10]),
            "store_index": torch.tensor([0, 1, 2]),
            "product_index": torch.tensor([2, 1, 0]),
            "text": None,
        },
    }}
    return cfg, record


def test_machine_hardware_records_supported_platforms_and_fails_closed() -> None:
    darwin_values = {
        "machdep.cpu.brand_string": "Test CPU",
        "hw.physicalcpu": "8",
        "hw.logicalcpu": "10",
        "hw.memsize": "17179869184",
    }

    def sysctl(command, **_kwargs):
        return SimpleNamespace(
            returncode=0, stdout=darwin_values[command[-1]] + "\n", stderr="")

    with mock.patch.object(bench.sys, "platform", "darwin"), \
         mock.patch.object(bench.subprocess, "run", side_effect=sysctl):
        assert bench._machine_hardware() == {  # noqa: SLF001
            "cpu_model": "Test CPU", "physical_cores": 8, "logical_cores": 10,
            "memory_bytes": 17179869184,
        }

    cpuinfo = (
        "processor : 0\nmodel name : Test Linux CPU\nphysical id : 0\ncore id : 0\n\n"
        "processor : 1\nmodel name : Test Linux CPU\nphysical id : 0\ncore id : 1\n"
    )
    meminfo = "MemTotal:       16384 kB\n"
    with mock.patch.object(bench.sys, "platform", "linux"), \
         mock.patch.object(bench.pathlib.Path, "read_text",
                           side_effect=[cpuinfo, meminfo]), \
         mock.patch.object(bench.os, "cpu_count", return_value=4):
        assert bench._machine_hardware() == {  # noqa: SLF001
            "cpu_model": "Test Linux CPU", "physical_cores": 2, "logical_cores": 4,
            "memory_bytes": 16384 * 1024,
        }

    failing_darwin = SimpleNamespace(returncode=1, stdout="", stderr="not available")
    with mock.patch.object(bench.sys, "platform", "darwin"), \
         mock.patch.object(bench.subprocess, "run", return_value=failing_darwin):
        try:
            bench._machine_hardware()  # noqa: SLF001
        except RuntimeError as exc:
            assert "cannot establish hardware identity" in str(exc)
        else:
            raise AssertionError("an unreadable Darwin hardware identity was accepted")

    with mock.patch.object(bench.sys, "platform", "linux"), \
         mock.patch.object(bench.pathlib.Path, "read_text", return_value=""), \
         mock.patch.object(bench.os, "cpu_count", return_value=4):
        try:
            bench._machine_hardware()  # noqa: SLF001
        except RuntimeError as exc:
            assert "does not contain CPU model and MemTotal" in str(exc)
        else:
            raise AssertionError("an empty Linux hardware identity was accepted")

    with mock.patch.object(bench.sys, "platform", "unsupported"):
        try:
            bench._machine_hardware()  # noqa: SLF001
        except RuntimeError as exc:
            assert "unsupported benchmark platform" in str(exc)
        else:
            raise AssertionError("an unsupported benchmark platform was accepted")


def test_checkpoint_conditioning_is_used_and_replayable() -> None:
    cfg, record = _fixture()
    first, first_meta = bench.benchmark_inputs(cfg, 3, record, seed=7)
    second, second_meta = bench.benchmark_inputs(cfg, 3, record, seed=7)
    assert first_meta["static_source"] == "checkpoint run.tensors"
    assert first_meta["factor_blocks"] == 2
    assert set(first["series_index"].tolist()) == {0, 1, 2}
    assert torch.equal(first["attributes"], record["run"]["tensors"]["attributes"])
    assert first_meta["tensor_sha256"] == second_meta["tensor_sha256"]
    assert all(torch.equal(first[k], second[k]) for k in first if first[k] is not None)


def test_checkpoint_rows_preserve_alignment_for_samples_and_repetition() -> None:
    _, record = _fixture()
    sampled, index, evidence = bench._checkpoint_rows(record, 2, seed=19)  # noqa: SLF001
    expected = torch.tensor(
        bench.ours._sample_series_indices(record["run"]["series_ids"], 2, 19),  # noqa: SLF001
        dtype=torch.long,
    )
    assert torch.equal(index, expected)
    assert torch.equal(
        sampled["product_index"],
        record["run"]["tensors"]["product_index"].index_select(0, expected),
    )
    assert sampled["text"] is None
    assert evidence["selection"] == "production hash sample"
    assert evidence["selected_row_indices_sha256"] == bench._tensor_fingerprint(  # noqa: SLF001
        {"row": expected}
    )

    repeated, repeated_index, repeated_evidence = bench._checkpoint_rows(  # noqa: SLF001
        record, 5, seed=19
    )
    assert repeated_index.tolist() == [0, 1, 2, 0, 1]
    assert repeated["group_index"].tolist() == [10, 20, 10, 10, 20]
    assert "repeated" in repeated_evidence["selection"]

    malformed = copy.deepcopy(record)
    malformed["run"]["tensors"]["store_index"] = torch.tensor([0, 1])
    try:
        bench._checkpoint_rows(malformed, 2, seed=0)  # noqa: SLF001
    except RuntimeError as exc:
        assert "not aligned" in str(exc)
    else:
        raise AssertionError("misaligned checkpoint conditioning rows were accepted")

    missing = copy.deepcopy(record)
    del missing["run"]["tensors"]["product_index"]
    assert bench._checkpoint_rows(missing, 2, seed=0) is None  # noqa: SLF001


def test_tensor_fingerprints_bind_names_values_dtype_shape_and_none() -> None:
    base = torch.tensor([[1, 2]], dtype=torch.int64)
    expected = bench._tensor_fingerprint({"a": base, "b": None})  # noqa: SLF001
    assert expected == bench._tensor_fingerprint({"b": None, "a": base})  # noqa: SLF001
    assert expected != bench._tensor_fingerprint({"a": base.clone(), "c": None})  # noqa: SLF001
    assert expected != bench._tensor_fingerprint({"a": base.to(torch.float32), "b": None})  # noqa: SLF001
    assert expected != bench._tensor_fingerprint({"a": base.reshape(2, 1), "b": None})  # noqa: SLF001
    changed = base.clone()
    changed[0, 0] = 9
    assert expected != bench._tensor_fingerprint({"a": changed, "b": None})  # noqa: SLF001

    digest = hashlib.sha256()
    try:
        bench._update_tensor_fingerprint(digest, "bad", object())  # noqa: SLF001
    except TypeError as exc:
        assert "non-tensor" in str(exc)
    else:
        raise AssertionError("a non-tensor fingerprint input was accepted")


def test_uniform_seal_is_seed_and_head_sensitive() -> None:
    p = torch.tensor([[0.4, 0.8]])
    mu = torch.tensor([[1.0, 2.0]])
    same = bench._uniform_fingerprint(p, mu, 1, 11)  # noqa: SLF001
    assert same == bench._uniform_fingerprint(p, mu, 1, 11)  # noqa: SLF001
    assert same != bench._uniform_fingerprint(p, mu, 1, 12)  # noqa: SLF001
    assert same != bench._uniform_fingerprint(None, mu, 1, 11)  # noqa: SLF001


def test_uniform_fingerprint_does_not_advance_the_production_generator() -> None:
    p = torch.tensor([[0.4, 0.8]], dtype=torch.float32)
    mu = torch.tensor([[1.0, 2.0]])
    generator = torch.Generator().manual_seed(37)
    before = generator.get_state().clone()
    fingerprint = bench._uniform_fingerprint_from_generator(  # noqa: SLF001
        p, mu, 3, generator
    )
    assert fingerprint == bench._uniform_fingerprint(p, mu, 3, 37)  # noqa: SLF001
    assert torch.equal(generator.get_state(), before)

    control = torch.Generator().manual_seed(37)
    assert torch.equal(
        torch.rand(5, generator=generator),
        torch.rand(5, generator=control),
    )


def test_full_stochastic_seal_covers_the_declared_stream() -> None:
    model = torch.nn.Linear(1, 1)
    model.factor = SimpleNamespace(rank=2, loading_occurrence=SimpleNamespace(weight=model.weight))
    first = bench._full_stochastic_input_fingerprints(  # noqa: SLF001
        model, n_groups=2, n_cells=3, horizon=4, p_dtype=torch.float32,
        factor_seed=11, count_seed=22, n_paths=3)
    second = bench._full_stochastic_input_fingerprints(  # noqa: SLF001
        model, n_groups=2, n_cells=3, horizon=4, p_dtype=torch.float32,
        factor_seed=11, count_seed=22, n_paths=3)
    changed = bench._full_stochastic_input_fingerprints(  # noqa: SLF001
        model, n_groups=2, n_cells=3, horizon=4, p_dtype=torch.float32,
        factor_seed=12, count_seed=22, n_paths=3)
    assert first == second
    assert first["n_paths"] == 3
    assert first["factor_innovations_sha256"] != changed["factor_innovations_sha256"]

    factorless = torch.nn.Linear(1, 1)
    factorless.factor = None
    no_hurdle = bench._full_stochastic_input_fingerprints(  # noqa: SLF001
        factorless, n_groups=2, n_cells=3, horizon=4, p_dtype=None,
        factor_seed=11, count_seed=22, n_paths=3)
    with_hurdle = bench._full_stochastic_input_fingerprints(  # noqa: SLF001
        factorless, n_groups=2, n_cells=3, horizon=4, p_dtype=torch.float32,
        factor_seed=11, count_seed=22, n_paths=3)
    assert no_hurdle["factor_innovation_shape_per_draw"] is None
    assert no_hurdle["count_uniform_shape_per_draw"] == [3, 4, 3]
    assert no_hurdle["count_uniforms_sha256"] != with_hurdle["count_uniforms_sha256"]


def test_full_stochastic_seal_matches_the_production_stream() -> None:
    """The precomputed seal must describe what ``draw_paths_over_factor`` consumes."""
    factor_inputs: list[torch.Tensor] = []
    count_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    class RecordingFactor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rank = 2
            self.loading_occurrence = torch.nn.Embedding(1, self.rank)

        def draw(self, n_groups, horizon, n_draws=1, generator=None):
            innovations = torch.randn(
                n_draws, n_groups, horizon, self.rank,
                dtype=self.loading_occurrence.weight.dtype, generator=generator)
            factor_inputs.append(innovations.detach().clone())
            return innovations

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.factor = RecordingFactor()

        def encode_base(self, **_inputs):
            return SimpleNamespace(n_groups=2, n_cells=3, horizon=4)

        def apply_factor(self, base, *, n_draws):
            self.factor.draw(base.n_groups, base.horizon, n_draws)
            shape = (n_draws, base.n_cells, base.horizon)
            return (
                torch.full(shape, 0.4),
                torch.full(shape, 2.0),
                torch.full(shape, 1.5),
            )

    model = RecordingModel()
    n_paths, seed = 3, 19
    factor_seed = bench.ours._domain_seed(  # noqa: SLF001
        seed, "factor", bench.ours.RNG_DOMAIN_SCHEME)
    count_seed = bench.ours._domain_seed(  # noqa: SLF001
        seed, "count", bench.ours.RNG_DOMAIN_SCHEME)
    declared = bench._full_stochastic_input_fingerprints(  # noqa: SLF001
        model, n_groups=2, n_cells=3, horizon=4, p_dtype=torch.float32,
        factor_seed=factor_seed, count_seed=count_seed, n_paths=n_paths)

    original_predicted_paths = bench.ours.predicted_paths

    def recording_predicted_paths(p, mu, k, n_draws, *, generator):
        replay = torch.Generator(device=generator.device)
        replay.set_state(generator.get_state())
        tail = (*mu.shape, n_draws)
        occurrence = torch.rand(tail, dtype=p.dtype, generator=replay)
        size = torch.rand(tail, dtype=torch.float64, generator=replay)
        expected_final_state = replay.get_state().clone()
        count_inputs.append((occurrence, size))
        paths = original_predicted_paths(p, mu, k, n_draws, generator=generator)
        assert torch.equal(generator.get_state(), expected_final_state)
        return paths

    with mock.patch.object(
        bench.ours, "predicted_paths", side_effect=recording_predicted_paths
    ):
        paths = bench.ours.draw_paths_over_factor(
            model, {"fixture": torch.ones(3, 1)}, n_paths,
            factor_draws=n_paths, seed=seed,
            rng_domain_scheme=bench.ours.RNG_DOMAIN_SCHEME)

    assert paths.shape == (n_paths, 3, 4)
    assert len(factor_inputs) == len(count_inputs) == n_paths
    factor_digest = hashlib.sha256()
    count_digest = hashlib.sha256()
    for draw, innovations in enumerate(factor_inputs):
        bench._update_tensor_fingerprint(  # noqa: SLF001
            factor_digest, f"factor_innovations_{draw:04d}", innovations)
    for draw, (occurrence, size) in enumerate(count_inputs):
        bench._update_tensor_fingerprint(  # noqa: SLF001
            count_digest, f"occurrence_uniforms_{draw:04d}", occurrence)
        bench._update_tensor_fingerprint(  # noqa: SLF001
            count_digest, f"size_uniforms_{draw:04d}", size)
    assert declared["factor_innovations_sha256"] == factor_digest.hexdigest()
    assert declared["count_uniforms_sha256"] == count_digest.hexdigest()


def test_summarise_reports_the_declared_statistics() -> None:
    summary = bench._summarise([4.0, 1.0, 3.0, 2.0])  # noqa: SLF001
    assert summary == {
        "median_s": 2.5,
        "mean_s": 2.5,
        "stdev_s": math.sqrt(5.0 / 3.0),
        "p95_s": 4.0,
        "min_s": 1.0,
        "n_reps": 4,
    }
    assert bench._summarise([7.0])["stdev_s"] == 0.0  # noqa: SLF001


def test_model_config_record_is_json_shaped_and_excludes_runtime_tensors() -> None:
    cfg = {
        "cycles": (7, 28),
        "nested": {"indices": (1, 3)},
        "runtime_tensor": torch.tensor([99]),
    }
    record = bench._model_config_record(cfg)  # noqa: SLF001
    assert record == {"cycles": [7, 28], "nested": {"indices": [1, 3]}}
    assert cfg["cycles"] == (7, 28)


def test_forward_stages_measures_each_route_and_seals_outputs() -> None:
    calls: dict[str, object] = {}

    class FakeModel(torch.nn.Module):
        def encode_base(self, **inputs):
            calls.setdefault("encode_inputs", inputs)
            return SimpleNamespace(group_index=torch.tensor([0, 0, 1]), horizon=2)

        def draw_factor_paths(self, n_groups, horizon, n_draws, *, generator):
            calls["factor_args"] = (n_groups, horizon, n_draws, generator.initial_seed())
            return torch.arange(n_groups * horizon * n_draws, dtype=torch.float32).reshape(
                n_draws, n_groups, horizon, 1
            )

        def apply_factor(self, cached, *, n_draws, generator):
            calls["apply_args"] = (cached, n_draws, generator.initial_seed())
            shape = (n_draws, 3, cached.horizon)
            return (
                torch.full(shape, 0.5),
                torch.full(shape, 2.0),
                torch.full(shape, 1.5),
            )

    model = FakeModel()
    inputs = {"history": torch.ones(3, 2)}
    fixture = {"tensor_sha256": "fixture"}
    timing_values = iter((3.0, 5.0, 20.0, 4.0))

    def fake_time(fn, reps=5):
        fn()
        value = next(timing_values)
        return {"median_s": value, "p95_s": value, "min_s": value, "n_reps": reps}

    def fake_mc(received_model, received_inputs, *, factor_draws, seed):
        calls["mc"] = (received_model, received_inputs, factor_draws, seed)
        return torch.tensor([11.0])

    def fake_gh(received_model, received_inputs):
        calls["gh"] = (received_model, received_inputs)
        return torch.tensor([13.0])

    with (
        mock.patch.object(bench, "benchmark_inputs", return_value=(inputs, fixture)),
        mock.patch.object(bench, "_time", side_effect=fake_time),
        mock.patch.object(bench.ours, "mean_over_factor", side_effect=fake_mc),
        mock.patch.object(bench.ours, "mean_over_factor_quadrature", side_effect=fake_gh),
    ):
        result = bench.forward_stages(model, {"unused": True}, {}, cells=3)

    assert calls["encode_inputs"] == inputs
    assert calls["factor_args"] == (2, 2, bench.DRAW_CHUNK, 104729)
    assert calls["apply_args"][1:] == (bench.DRAW_CHUNK, 104729)
    assert calls["mc"][2:] == (1000, 0)
    assert calls["gh"] == (model, inputs)
    assert result["cells"] == 3
    assert result["input_fixture"] is fixture
    assert result["deterministic_base_s"] == 3.0
    assert result["per_factor_draw_s"] == 5.0 / bench.DRAW_CHUNK
    assert result["synthetic_removed_loop_projection_s"] == 320.0
    assert result["ratio_vs_synthetic_removed_loop_projection"] == {
        "cached_mc_1000": 16.0,
        "quadrature": 80.0,
    }
    assert result["measured_output_sha256"] == {
        "mc_1000_draws": bench._tensor_fingerprint({"mean": torch.tensor([11.0])}),  # noqa: SLF001
        "gauss_hermite": bench._tensor_fingerprint({"mean": torch.tensor([13.0])}),  # noqa: SLF001
    }


def test_sampler_stage_pairs_sequential_factor_and_count_streams() -> None:
    calls: dict[str, object] = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.apply_calls = 0

        def encode_base(self, **inputs):
            calls.setdefault("encode_inputs", inputs)
            return SimpleNamespace(group_index=torch.tensor([0, 1]), horizon=2)

        def draw_factor_paths(self, n_groups, horizon, n_draws, *, generator):
            return torch.rand(n_draws, n_groups, horizon, 1, generator=generator)

        def apply_factor(self, base, *, n_draws, factor_path):
            self.apply_calls += 1
            shape = (n_draws, 2, base.horizon)
            offset = float(self.apply_calls)
            return (
                torch.full(shape, 0.5),
                torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + offset,
                torch.full(shape, 1.5),
            )

    model = FakeModel()
    inputs = {"history": torch.ones(2, 2)}
    fixture = {"tensor_sha256": "path-fixture"}
    clock = [0.0]

    def fake_perf_counter():
        clock[0] += 0.25
        return clock[0]

    def fake_time(fn, reps=5):
        fn()
        return {"median_s": 0.5, "p95_s": 0.5, "min_s": 0.5, "n_reps": reps}

    def fake_predicted_paths(p, mu, k, n_draws, *, generator):
        calls.setdefault("predicted_shapes", []).append((p.shape, mu.shape, k.shape, n_draws))
        return torch.rand(*mu.shape, n_draws, generator=generator)

    def fake_full_paths(received_model, received_inputs, n_paths, **kwargs):
        calls["full_paths"] = (received_model, received_inputs, n_paths, kwargs)
        return torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    def fake_full_fingerprints(received_model, **kwargs):
        calls["full_fingerprints"] = (received_model, kwargs)
        return {"sealed": True}

    def fake_support_size(mu, _k):
        return int(torch.ceil(mu.max()).item()) + 1

    def fake_support_sizes(mu, _k):
        return torch.ceil(mu).to(torch.long) + 1

    with (
        mock.patch.object(bench, "PATH_CELLS", 2),
        mock.patch.object(bench, "SAMPLER_FACTOR_FIXTURES", 2),
        mock.patch.object(bench, "benchmark_inputs", return_value=(inputs, fixture)),
        mock.patch.object(bench, "_time", side_effect=fake_time),
        mock.patch.object(bench.time, "perf_counter", side_effect=fake_perf_counter),
        mock.patch.object(bench.ours, "predicted_paths", side_effect=fake_predicted_paths),
        mock.patch.object(bench.ours, "draw_paths_over_factor", side_effect=fake_full_paths),
        mock.patch.object(
            bench, "_full_stochastic_input_fingerprints", side_effect=fake_full_fingerprints
        ),
        mock.patch.object(bench, "support_size", side_effect=fake_support_size),
        mock.patch.object(bench, "support_sizes", side_effect=fake_support_sizes),
    ):
        result = bench.sampler_stage(model, {"horizon": 2, "head_kind": "hurdle_negbin"}, {})

    assert calls["encode_inputs"] == inputs
    assert len(calls["predicted_shapes"]) == 3  # one warm-up plus two recorded fixtures
    assert calls["full_paths"][2:] == (
        1000,
        {
            "factor_draws": 1000,
            "seed": bench.SAMPLER_FACTOR_SEED,
            "rng_domain_scheme": bench.ours.RNG_DOMAIN_SCHEME,
        },
    )
    full_fingerprint_args = calls["full_fingerprints"][1]
    assert full_fingerprint_args["n_groups"] == 2
    assert full_fingerprint_args["n_cells"] == 2
    assert full_fingerprint_args["p_dtype"] == torch.float32
    assert result["cells"] == 2
    assert result["input_fixture"] is fixture
    assert result["factor_fixture_draws"] == 2
    assert len(result["fixture_fingerprints"]) == 2
    assert result["fixture_fingerprints"][0]["factor_generator_state_sha256"] != (
        result["fixture_fingerprints"][1]["factor_generator_state_sha256"]
    )
    assert result["full_stochastic_input_fingerprints"] == {"sealed": True}
    assert result["factor_and_head_one_draw"]["mean_s"] == 0.25
    assert result["sample_one_path"]["mean_s"] == 0.25
    assert result["paired_one_path"]["mean_s"] == 0.5
    assert result["projected_1000_draws_s"] == 500.5
    assert result["measured_1000_paths"]["elapsed_s"] == 0.25
    assert result["measured_1000_paths"]["shape"] == [2, 2, 2]
    assert [row["max_mu"] for row in result["batch_max_mu_sweep"]] == [
        2.0,
        20.0,
        100.0,
        300.0,
    ]


def test_comparison_refuses_every_execution_identity_mismatch() -> None:
    current = {
        "benchmark_contract": bench.BENCHMARK_CONTRACT,
        "checkpoint_sha256": "checkpoint",
        "model_config": {"d_model": 8},
        "benchmark_config": {
            "draw_chunk": 25,
            "benchmark_source_sha256": "source",
            "measurement_protocol": {"full_paths": {"reps": 1, "scope": "end to end"}},
        },
        "point_loop": {"cells": 3, "input_fixture": {"tensor_sha256": "point"},
                       "per_draw_factor_path_sha256": "point-factor",
                       "per_draw_output_sha256": "point-output",
                       "measured_output_sha256": {"gauss_hermite": "gh", "mc": "mc"},
                       "per_draw_timings": {"factor": {"n_reps": 3}},
                       "measured": {"gauss_hermite": {"n_reps": 2}}},
        "path_loop": {"cells": 2, "input_fixture": {"tensor_sha256": "path"},
                      "fixture_fingerprints": [{"factor_path_sha256": "factor",
                                                "count_uniforms_sha256": "uniform",
                                                "parameters_sha256": "parameters",
                                                "paths_sha256": "paths"}],
                      "full_stochastic_input_fingerprints": {
                          "factor_innovations_sha256": "all-factor",
                          "count_uniforms_sha256": "all-count"},
                      "encode_base_once": {"n_reps": 3},
                      "factor_and_head_one_draw": {"n_reps": 50},
                      "sample_one_path": {"n_reps": 50},
                      "paired_one_path": {"n_reps": 50},
                      "projection_method": "mean of paired work",
                      "measured_1000_paths": {
                          "output_sha256": "all-paths", "n_reps": 1,
                          "scope": "end to end including output"}},
        "resources": {"scope": "whole process"},
        "identity": {"threads": 6, "interop_threads": 2, "platform": "machine",
                     "torch": "v", "python": "v",
                     "hardware": {"cpu_model": "Apple M2 Pro", "physical_cores": 10,
                                  "logical_cores": 10, "memory_bytes": 17_179_869_184}},
    }
    assert bench._comparison_mismatches(current, copy.deepcopy(current)) == []  # noqa: SLF001
    mutations = (
        lambda r: r.__setitem__("checkpoint_sha256", "different"),
        lambda r: r["model_config"].__setitem__("d_model", 9),
        lambda r: r["benchmark_config"].__setitem__("draw_chunk", 7),
        lambda r: r["benchmark_config"].__setitem__("benchmark_source_sha256", "different"),
        lambda r: r["point_loop"].__setitem__("cells", 4),
        lambda r: r["path_loop"]["input_fixture"].__setitem__("tensor_sha256", "different"),
        lambda r: r["point_loop"].__setitem__("per_draw_factor_path_sha256", "different"),
        lambda r: r["point_loop"].__setitem__("per_draw_output_sha256", "different"),
        lambda r: r["point_loop"]["measured_output_sha256"].__setitem__(
            "gauss_hermite", "different"),
        lambda r: r["path_loop"]["fixture_fingerprints"][0].__setitem__(
            "count_uniforms_sha256", "different"),
        lambda r: r["path_loop"]["fixture_fingerprints"][0].__setitem__(
            "parameters_sha256", "different"),
        lambda r: r["path_loop"]["full_stochastic_input_fingerprints"].__setitem__(
            "factor_innovations_sha256", "different"),
        lambda r: r["path_loop"]["measured_1000_paths"].__setitem__(
            "output_sha256", "different"),
        lambda r: r["point_loop"]["measured"]["gauss_hermite"].__setitem__("n_reps", 1),
        lambda r: r["path_loop"]["measured_1000_paths"].__setitem__(
            "scope", "kernel only; allocation excluded"),
        lambda r: r["identity"].__setitem__("threads", 7),
        lambda r: r["identity"].__setitem__("interop_threads", 3),
        lambda r: r["identity"].__setitem__("platform", "different"),
        lambda r: r["identity"]["hardware"].__setitem__("cpu_model", "Apple M4"),
    )
    for mutate in mutations:
        earlier = copy.deepcopy(current)
        mutate(earlier)
        assert bench._comparison_mismatches(earlier, current)  # noqa: SLF001

    numeric_type_aliases = (
        lambda r: r.__setitem__("benchmark_contract", 3.0),
        lambda r: r["model_config"].__setitem__("d_model", 8.0),
        lambda r: r["benchmark_config"].__setitem__("draw_chunk", 25.0),
        lambda r: r["point_loop"].__setitem__("cells", 3.0),
        lambda r: r["point_loop"]["per_draw_timings"]["factor"].__setitem__("n_reps", 3.0),
        lambda r: r["path_loop"]["measured_1000_paths"].__setitem__("n_reps", True),
        lambda r: r["path_loop"]["measured_1000_paths"].__setitem__("n_reps", 1.0),
        lambda r: r["identity"].__setitem__("threads", 6.0),
        lambda r: r["identity"]["hardware"].__setitem__("physical_cores", 10.0),
    )
    for alias in numeric_type_aliases:
        earlier = copy.deepcopy(current)
        alias(earlier)
        assert bench._comparison_mismatches(earlier, current), (  # noqa: SLF001
            "a numeric type alias made incompatible benchmark evidence comparable", earlier)


def test_compatible_baseline_ratios_compare_only_the_same_measured_stage() -> None:
    record = {
        "point_loop": {"measured": {
            "mc_1000_draws": {"median_s": 63.0},
            "gauss_hermite": {"median_s": 17.0},
        }},
        "path_loop": {"measured_1000_paths": {"elapsed_s": 442.0}},
    }
    self_comparisons = bench._same_stage_comparisons(record, copy.deepcopy(record))  # noqa: SLF001
    assert [(label, old / new) for label, old, new in self_comparisons] == [
        ("point MC(1,000)", 1.0),
        ("point Gauss-Hermite", 1.0),
        ("paths 1,000/1,000", 1.0),
    ], "a benchmark compared with itself must never report an algorithm-switch speedup"
    assert bench._speed_change_text(10.0, 20.0) == "2.00x slowdown"  # noqa: SLF001
    assert bench._speed_change_text(20.0, 10.0) == "2.00x speedup"  # noqa: SLF001
    assert bench._speed_change_text(10.0, 10.0) == "unchanged"  # noqa: SLF001

    for invalid in (None, True, 0.0, -1.0, 10**400, float("nan"), float("inf")):
        broken = copy.deepcopy(record)
        broken["point_loop"]["measured"]["mc_1000_draws"]["median_s"] = invalid
        try:
            bench._same_stage_comparisons(broken, record)  # noqa: SLF001
        except RuntimeError as exc:
            assert "invalid same-stage timing" in str(exc)
        else:
            raise AssertionError(f"invalid baseline timing {invalid!r} was compared")

        broken = copy.deepcopy(record)
        broken["path_loop"]["measured_1000_paths"]["elapsed_s"] = invalid
        try:
            bench._same_stage_comparisons(record, broken)  # noqa: SLF001
        except RuntimeError as exc:
            assert "invalid same-stage timing" in str(exc)
        else:
            raise AssertionError(f"invalid current timing {invalid!r} was compared")


def test_main_runs_mocked_stages_rechecks_inputs_and_publishes_sealed_record() -> None:
    point = {
        "cells": 7,
        "deterministic_base_s": 1.0,
        "per_draw_below_timing_noise": False,
        "per_factor_draw_s": 0.1,
        "base_over_draw_ratio": 10.0,
        "synthetic_removed_loop_projection_s": 5.0,
        "ratio_vs_synthetic_removed_loop_projection": {
            "cached_mc_1000": 2.5,
            "quadrature": 5.0,
        },
        "measured": {
            "mc_1000_draws": {"median_s": 2.0},
            "gauss_hermite": {"median_s": 1.0},
        },
    }
    path = {
        "cells": 2,
        "encode_base_once": {"median_s": 0.5},
        "factor_and_head_one_draw": {"median_s": 0.25},
        "sample_one_path": {"median_s": 0.75},
        "sampler_over_factor_head_ratio": 3.0,
        "support_required": 17,
        "mu_range": [0.5, 3.0],
        "batch_max_mu_sweep": [{
            "max_mu": 2.0,
            "shared_refused": False,
            "shared_support": 7,
            "shared_table_gb": 0.01,
            "refused_per_cell": False,
            "per_cell_median_support": 4,
            "per_cell_max_support": 7,
        }],
        "projected_1000_draws_s": 60.0,
        "measured_1000_paths": {"elapsed_s": 120.0},
    }
    checkpoint_check = mock.Mock()
    identity_check = mock.Mock()
    baseline_check = mock.Mock()
    baseline_alias_check = mock.Mock()
    publisher = mock.Mock()
    forward = mock.Mock(return_value=point)
    sampler = mock.Mock(return_value=path)
    model = object()
    model_config = {
        "cycles": (7, 28),
        "runtime_tensor": torch.tensor([1]),
    }

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = pathlib.Path(directory) / "model.pt"
        output = pathlib.Path(directory) / "result.json"
        baseline = pathlib.Path(directory) / "baseline.json"
        earlier = {"identity": {"git_sha": "b" * 40, "platform": "test-platform"}}
        comparisons = [
            ("point MC(1,000)", 4.0, 2.0),
            ("point Gauss-Hermite", 2.0, 1.0),
            ("paths 1,000/1,000", 240.0, 120.0),
        ]

        def fake_file_hash(pathlib_path):
            return "checkpoint-sha" if pathlib_path == checkpoint else "source-sha"

        with (
            mock.patch.object(
                bench.sys,
                "argv",
                [
                    "bench_inference.py",
                    str(checkpoint),
                    "--json",
                    str(output),
                    "--point-cells",
                    "7",
                    "--baseline",
                    str(baseline),
                ],
            ),
            mock.patch.object(bench, "identity", return_value={"git_sha": "a" * 40}),
            mock.patch.object(bench, "_file_sha256", side_effect=fake_file_hash),
            mock.patch.object(
                bench.ours,
                "load_checkpoint",
                return_value=(model, {"model_config": model_config}),
            ),
            mock.patch.object(bench, "_require_checkpoint_unchanged", checkpoint_check),
            mock.patch.object(bench, "_require_identity_unchanged", identity_check),
            mock.patch.object(
                bench,
                "_load_benchmark_baseline_snapshot",
                return_value=(earlier, "baseline-sha"),
            ),
            mock.patch.object(bench, "_require_baseline_unchanged", baseline_check),
            mock.patch.object(
                bench, "_require_output_separate_from_baseline", baseline_alias_check),
            mock.patch.object(bench, "_comparison_mismatches", return_value=[]),
            mock.patch.object(bench, "_same_stage_comparisons", return_value=comparisons),
            mock.patch.object(bench, "forward_stages", forward),
            mock.patch.object(bench, "sampler_stage", sampler),
            mock.patch.object(bench, "_whole_process_peak_rss_bytes", return_value=1234),
            mock.patch.object(bench, "_publish_json", publisher),
        ):
            assert bench.main() == 0

    forward.assert_called_once_with(model, model_config, {"model_config": model_config}, 7)
    sampler.assert_called_once_with(model, model_config, {"model_config": model_config})
    assert checkpoint_check.call_count == 3
    assert identity_check.call_count == 3
    assert baseline_alias_check.call_count == 2
    assert baseline_check.call_count == 2
    publisher.assert_called_once()
    published_path, sealed = publisher.call_args.args
    assert published_path == output
    assert sealed["checkpoint_sha256"] == "checkpoint-sha"
    assert sealed["benchmark_config"]["benchmark_source_sha256"] == "source-sha"
    assert sealed["benchmark_config"]["point_cells"] == 7
    assert sealed["model_config"] == {"cycles": [7, 28]}
    assert sealed["point_loop"] is point
    assert sealed["path_loop"] is path
    assert sealed["resources"]["whole_process_peak_rss_bytes"] == 1234
    assert sealed["comparison"] == {
        "baseline_path": str(baseline.resolve()),
        "baseline_sha256": "baseline-sha",
        "same_stage": [
            {
                "stage": label,
                "baseline_seconds": was,
                "current_seconds": now,
                "baseline_over_current_ratio": was / now,
                "direction": "2.00x speedup",
            }
            for label, was, now in comparisons
        ],
        "scope": "only computationally compatible, like-for-like measured stages",
    }


def test_git_identity_fails_closed() -> None:
    original = bench.subprocess.run
    bench.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
        returncode=128, stdout=b"", stderr=b"not a repository")
    try:
        try:
            bench.identity()
        except RuntimeError as exc:
            assert "not a repository" in str(exc)
        else:
            raise AssertionError("git failure was accepted as a benchmark identity")
    finally:
        bench.subprocess.run = original


def test_git_identity_refuses_untracked_symlink() -> None:
    probe = bench.ROOT / "bench_identity_symlink_probe.py"
    if probe.exists() or probe.is_symlink():
        raise AssertionError(f"identity probe path already exists: {probe}")
    probe.symlink_to(pathlib.Path("scripts") / "test_bench_inference.py")
    try:
        try:
            bench.identity()
        except RuntimeError as exc:
            assert "untracked symlink" in str(exc)
        else:
            raise AssertionError("symlink target bytes were not bound or refused")
    finally:
        probe.unlink(missing_ok=True)


def test_peak_rss_units_are_platform_correct() -> None:
    original_getrusage = bench.resource.getrusage
    original_platform = bench.sys.platform
    bench.resource.getrusage = lambda _: SimpleNamespace(ru_maxrss=123)
    try:
        bench.sys.platform = "darwin"
        assert bench._whole_process_peak_rss_bytes() == 123  # noqa: SLF001
        bench.sys.platform = "linux"
        assert bench._whole_process_peak_rss_bytes() == 123 * 1024  # noqa: SLF001
        bench.resource.getrusage = lambda _: SimpleNamespace(ru_maxrss=10**400)
        try:
            bench._whole_process_peak_rss_bytes()  # noqa: SLF001
        except RuntimeError as exc:
            assert "invalid ru_maxrss" in str(exc)
        else:
            raise AssertionError("an unrepresentable RSS measurement entered benchmark evidence")
    finally:
        bench.resource.getrusage = original_getrusage
        bench.sys.platform = original_platform


def test_atomic_json_refuses_nan_and_verifies_readback() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = pathlib.Path(directory) / "record.json"
        bench._publish_json(target, {"finite": 1.25})  # noqa: SLF001
        assert json.loads(target.read_text()) == {"finite": 1.25}
        assert list(pathlib.Path(directory).iterdir()) == [target]
        refused = pathlib.Path(directory) / "refused.json"
        try:
            bench._publish_json(refused, {"bad": float("nan")})  # noqa: SLF001
        except RuntimeError as exc:
            assert "not finite JSON" in str(exc)
        else:
            raise AssertionError("NaN was published in a supposedly sealed benchmark record")
        assert not refused.exists()


def test_atomic_json_restores_old_or_removes_new_on_verification_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = pathlib.Path(directory) / "record.json"
        original_load = bench.load_exact_json

        def fail_readback(path):
            if path.resolve() == target.resolve():
                raise bench.ExactJSONError(
                    "injected readback failure", kind="read", cause_type_name="OSError")
            return original_load(path)

        with mock.patch.object(bench, "load_exact_json", side_effect=fail_readback):
            try:
                bench._publish_json(target, {"finite": 1})  # noqa: SLF001
            except RuntimeError as exc:
                assert "injected readback failure" in str(exc)
            else:
                raise AssertionError("injected readback failure was accepted")
        assert not target.exists()

        old_payload = '{"generation":"OLD-VALID"}\n'
        target.write_text(old_payload)
        with mock.patch.object(bench, "load_exact_json", side_effect=fail_readback):
            try:
                bench._publish_json(target, {"generation": "NEW"})  # noqa: SLF001
            except RuntimeError as exc:
                assert "injected readback failure" in str(exc)
            else:
                raise AssertionError("injected replacement readback failure was accepted")
        assert target.read_text() == old_payload
        assert list(pathlib.Path(directory).iterdir()) == [target]


def test_checkpoint_and_source_toctou_refuse_publication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = pathlib.Path(directory) / "model.pt"
        checkpoint.write_bytes(b"version one")
        expected = bench._file_sha256(checkpoint)  # noqa: SLF001
        checkpoint.write_bytes(b"version two")
        try:
            bench._require_checkpoint_unchanged(checkpoint, expected)  # noqa: SLF001
        except RuntimeError as exc:
            assert "changed during the run" in str(exc)
        else:
            raise AssertionError("checkpoint replacement was accepted")

    original_identity = bench.identity
    bench.identity = lambda output=None: {"git_sha": "after"}
    try:
        try:
            bench._require_identity_unchanged({"git_sha": "before"}, None)  # noqa: SLF001
        except RuntimeError as exc:
            assert "changed during the benchmark" in str(exc)
        else:
            raise AssertionError("mid-run source mutation was accepted")
    finally:
        bench.identity = original_identity


def test_tensor_fingerprint_seals_the_factorless_empty_path_sentinel() -> None:
    reference = torch.nn.Linear(1, 1).weight
    sentinel = torch.empty(
        (bench.DRAW_CHUNK, 0, 3, 0), device=reference.device, dtype=reference.dtype)
    sealed = bench._tensor_fingerprint({"factor_path": sentinel})  # noqa: SLF001
    assert len(sealed) == 64
    assert sealed == bench._tensor_fingerprint({"factor_path": sentinel.clone()})  # noqa: SLF001
    assert sealed != bench._tensor_fingerprint(  # noqa: SLF001
        {"factor_path": torch.empty((1, 0, 3, 0), dtype=reference.dtype)})
    assert sealed != bench._tensor_fingerprint(  # noqa: SLF001
        {"factor_path": torch.empty((bench.DRAW_CHUNK, 0, 3, 0), dtype=torch.float64)})


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
