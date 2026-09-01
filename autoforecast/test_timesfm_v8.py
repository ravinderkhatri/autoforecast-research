"""Public V8 selection/refit/replay with the real source architecture at fixture size."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from forecast.models.timesfm_v8 import EpochChoices, OnePassProfileDataset, OnlineWindows

from autoforecast.adapters import timesfm_v8 as adapter
from autoforecast.test_timesfm_hnb import ORIGIN, _campaign_result, _saved, panel_fixture
from autoforecast.timesfm_hnb_data import groups, load_data, missing_profiles
from autoforecast.timesfm_hnb_source import SourceRequest
from autoforecast.timesfm_v8_source import LoRASource, state_digest


def test_one_pass_sampler_balances_all_origin_profile_choices():
    choices = EpochChoices(rows=24, batch_size=2, origins=3, profiles=4, seed=7)
    first = [choices.for_row(index) for index in range(24)]
    assert all(first[index] == first[index + 1] for index in range(0, 24, 2))
    assert len(set(first)) == 12
    copy = EpochChoices(24, 2, 3, 4, 7)
    assert first == [copy.for_row(index) for index in range(24)]
    choices.set_epoch(1)
    assert len(set(choices.for_row(index) for index in range(24))) == 12


def test_online_window_epoch_visits_each_eligible_series_once(tmp_path):
    panel = panel_fixture(tmp_path)
    origins = ("2024-02-04", "2024-02-01", "2024-01-29")
    data = load_data(panel, ORIGIN, 3, 12, history_days=24)
    observed = data.observed.copy()
    earliest = data.position(origins[-1])
    observed[0, earliest + 1 : earliest + 4] = False
    data = replace(data, observed=observed)
    grouping = groups(data, 2, 32)
    profiles = missing_profiles(
        data.fields, tuple(tuple(family) for family in data.schema.get("families", ()))
    )
    dataset = OnlineWindows(data, origins, 3, 12, grouping, profiles, 2, 0)
    assert len(dataset) == 3
    fallback_seen = False
    for epoch in range(64):
        dataset.set_epoch(epoch)
        fallback_seen |= any(
            not dataset.availability[dataset.choices.for_row(index)[0], dataset.rows[index]]
            for index in range(len(dataset))
        )
        examples = [dataset[index] for index in range(len(dataset))]
        assert sorted(row for _, _, _, row in examples) == [0, 1, 2]
        assert all(
            dataset.availability[origins.index(origin), row]
            for origin, _, _, row in examples
        )
        assert all(absent in profiles for _, absent, _, _ in examples)
    assert fallback_seen
    copy = OnlineWindows(data, origins, 3, 12, grouping, profiles, 2, 0)
    copy.set_epoch(63)
    assert [copy[index] for index in range(len(copy))] == examples


def test_precomputed_profiles_do_not_multiply_epoch_length():
    datasets = [[(profile, row) for row in range(6)] for profile in range(3)]
    sampled = OnePassProfileDataset(datasets, batch_size=2, seed=11)
    assert len(sampled) == 6
    assert [sampled[index][1] for index in range(len(sampled))] == list(range(6))
    assert len({sampled[index][0] for index in range(0, 6, 2)}) == 3
    sampled.set_epoch(1)
    assert [sampled[index][1] for index in range(len(sampled))] == list(range(6))
    with pytest.raises(ValueError, match="same eligible series"):
        OnePassProfileDataset([datasets[0], datasets[1][:-1]], 2, 11)


class TinyLoRA(LoRASource):
    instances = []

    def __init__(self, device="cpu"):
        TimesFM3Torch = pytest.importorskip("timesfm3.model").TimesFM3Torch

        self.model = TimesFM3Torch(
            input_patch_len=4,
            output_patch_len=8,
            residual_block_config={
                "hidden_dims": 16,
                "output_dims": 16,
                "use_bias": False,
                "activation": "relu",
            },
            transformer_config={
                "num_layers": 1,
                "transformer": {
                    "model_dims": 16,
                    "hidden_dims": 16,
                    "num_heads": 2,
                    "use_rope_var": False,
                    "use_rope_seq": True,
                    "use_bias": False,
                    "attention_norm": "rms",
                    "feedforward_norm": "rms",
                    "qk_norm": "rms",
                    "ff_activation": "relu",
                    "deterministic": True,
                },
            },
        )
        self.model.pre_transformer_resblock.set_input_dims(24)
        # The pretrained fixture must be constant independently of the head/LoRA seed.
        with torch.random.fork_rng():
            torch.manual_seed(123)
            for parameter in self.model.parameters():
                torch.nn.init.normal_(parameter, std=0.05)
        self.model.eval().requires_grad_(False)
        self.provider = SimpleNamespace(device=torch.device(device))
        self.identity = {"source": "tiny-real-architecture", "device": device}
        self.enable_lora()
        self.initial_adapter = state_digest(self.adapter_state())
        self.initial_base = {
            n: p.clone() for n, p in self.model.named_parameters() if not p.requires_grad
        }
        self.instances.append(self)


def test_zero_lora_decode_parity_and_only_adapter_changes():
    source = TinyLoRA()
    target = np.arange(22, dtype=np.float32).reshape(2, 11) % 7
    request = SourceRequest(target, np.zeros(target.shape, bool), 5, ("target:a", "target:b"))
    source.model.eval()
    reference = source.model.decode(
        target=torch.tensor(target)[None],
        target_mask=torch.zeros((1, 2, 11), dtype=torch.bool),
        horizon=5,
    )
    torch.testing.assert_close(source.tensor_quantiles(request), reference[0].sort(-1).values)
    optimizer = torch.optim.AdamW(
        [p for p in source.model.parameters() if p.requires_grad], lr=0.01
    )
    for _ in range(3):
        optimizer.zero_grad()
        (source.tensor_quantiles(request) - 2).square().mean().backward()
        optimizer.step()
    assert state_digest(source.adapter_state()) != source.initial_adapter
    assert all(
        torch.equal(value, dict(source.model.named_parameters())[name])
        for name, value in source.initial_base.items()
    )
    source.seal()
    expected = source.predict(request)
    restored = TinyLoRA()
    restored.load_adapter(source.adapter_state())
    np.testing.assert_array_equal(expected, restored.predict(request))
    broken = source.adapter_state()
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="architecture"):
        restored.load_adapter(broken)


@pytest.mark.parametrize("constant_feeds", [False, True])
def test_v8_select_refit_replay_and_budget(tmp_path, monkeypatch, constant_feeds):
    from autoforecast import timesfm_v8_source
    from autoforecast.forecast_campaign import _selected_update_budget, _verify_fixed_refit_budget

    monkeypatch.setattr(timesfm_v8_source, "LoRASource", TinyLoRA)
    panel = panel_fixture(tmp_path)
    if constant_feeds:
        frame = pd.read_parquet(panel.path)
        frame["price"] = 0.0
        frame["competitor"] = 1.0
        frame.loc[frame["id"] == "a", "qty"] = 0
        frame.to_parquet(panel.path, index=False)

    def fit(root, **kwargs):
        return adapter.fit_predict(
            panel,
            ORIGIN,
            3,
            out_dir=root / "forecast",
            checkpoint_out=root / "model/model.pt",
            context_days=12,
            group_size=2,
            head_width=8,
            batch_size=2,
            max_epochs=2,
            mean_loss_weight=0.2,
            **kwargs,
        )

    root = tmp_path / "selection"
    forecast = fit(root)
    saved = _saved(root)
    assert forecast.adapter == "timesfm_v8"
    assert saved["schema"] == adapter.SCHEMA
    assert saved["training"]["selection_metric"] == "count_nll_mean"
    assert state_digest(saved["source_lora_state"]) != TinyLoRA.instances[-1].initial_adapter
    assert any(
        torch.count_nonzero(value)
        for name, value in saved["source_lora_state"].items()
        if "lora_B" in name
    )  # B starts at zero: weight decay alone cannot pass.
    assert all(
        torch.equal(value, dict(TinyLoRA.instances[-1].model.named_parameters())[name])
        for name, value in TinyLoRA.instances[-1].initial_base.items()
    )
    result = _campaign_result(root, forecast)
    result["adapter"] = "timesfm_v8"
    updates = _selected_update_budget(result, "wrmsse", 2)
    assert updates in (2, 4)
    replay = adapter.predict_from_checkpoint(
        panel,
        ORIGIN,
        3,
        checkpoint=root / "model/model.pt",
        checkpoint_sha256=result["checkpoint"]["sha256"],
        out_dir=tmp_path / "replay",
    )
    pd.testing.assert_frame_equal(pd.read_parquet(forecast.path), pd.read_parquet(replay.path))
    assert fit(root).as_record() == forecast.as_record()
    finalroot = tmp_path / "final"
    final = fit(finalroot, training_mode="fixed_refit", fixed_update_budget=7)
    finalresult = _campaign_result(finalroot, final)
    finalresult["adapter"] = "timesfm_v8"
    _verify_fixed_refit_budget(finalresult, 7, batch_size=2)
    assert _saved(finalroot)["training"]["epoch_log"][-1]["updates"] == 7


def test_v8_default_artifact_path_agrees_with_isolated_recovery(tmp_path, monkeypatch):
    from autoforecast import timesfm_v8_source

    monkeypatch.setattr(timesfm_v8_source, "LoRASource", TinyLoRA)
    panel = panel_fixture(tmp_path)
    output = tmp_path / "default"
    adapter.fit_predict(
        panel,
        ORIGIN,
        3,
        out_dir=output,
        context_days=12,
        group_size=2,
        head_width=8,
        batch_size=2,
        max_epochs=1,
        training_mode="fixed_refit",
        fixed_update_budget=1,
    )
    assert (tmp_path / "default-v8-artifacts/model.pt").is_file()


@pytest.mark.parametrize(
    "case",
    ["zero_target", "constant_target", "zero_future", "constant_future",
     "zero_past", "constant_prefix", "masked_prefix", "masked_future"],
)
def test_sparse_source_forward_parity_and_finite_gradients(case):
    source = TinyLoRA()
    source.model.eval()
    target = np.arange(22, dtype=np.float32).reshape(2, 11) % 7
    mask = np.zeros_like(target, dtype=bool)
    kwargs = {}
    reference_kwargs = {}
    if case in {"zero_target", "constant_target"}:
        target[0] = 0 if case == "zero_target" else 1
    elif case in {"constant_prefix", "masked_prefix"}:
        target[:, :8] = 0
        mask[:, :8] = case == "masked_prefix"
    else:
        past = case == "zero_past"
        name = "past_only" if past else "past_future"
        cov = np.full((1, 11 if past else 16), case == "constant_future", np.float32)
        cov_mask = np.full(cov.shape, case == "masked_future", bool)
        kwargs = {name: cov, name + "_mask": cov_mask}
        reference_kwargs = {
            name + "_covariates": torch.tensor(cov)[None],
            name + "_mask": torch.tensor(cov_mask)[None],
        }
    schema = ("a", "b", "covariate") if kwargs else ("a", "b")
    request = SourceRequest(target, mask, 5, schema, **kwargs)
    reference = source.model.decode(
        target=torch.tensor(target)[None], target_mask=torch.tensor(mask)[None],
        horizon=5, **reference_kwargs,
    )[0, :2].sort(-1).values
    output = source.tensor_quantiles(request)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    (output - 2).square().mean().backward()
    gradients = [p.grad for p in source.model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


def test_zero_variance_policy_preserves_positive_derivatives_and_is_scoped():
    from autoforecast.timesfm_v8_source import _ZeroSafeSqrt

    values = torch.tensor([0., 0.25, 1., 4.], requires_grad=True)
    with _ZeroSafeSqrt():
        output = torch.sqrt(values)
    output.sum().backward()
    torch.testing.assert_close(output, torch.tensor([0., 0.5, 1., 2.]))
    torch.testing.assert_close(values.grad, torch.tensor([0., 1., 0.5, 0.25]))
    native = torch.tensor(0., requires_grad=True)
    torch.sqrt(native).backward()
    assert torch.isinf(native.grad)  # Native behavior outside the source context is unchanged.
