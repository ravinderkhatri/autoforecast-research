"""LoRA source and count head trained together, one source group in memory at a time."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from autoforecast.timesfm_hnb_data import features
from autoforecast.timesfm_hnb_source import cached_quantiles
from autoforecast.timesfm_v8_source import LORA_RECIPE
from forecast.models.timesfm_hnb import CountModule
from torch.utils.data import Dataset

EPOCH_DEFINITION = "one_pass_eligible_series_sampled_origin_profile_v1"


class EpochChoices:
    """Draw every declared origin/profile combination evenly across series batches."""

    def __init__(self, rows: int, batch_size: int, origins: int, profiles: int, seed: int):
        self.batch_size = batch_size
        self.batches = math.ceil(rows / batch_size)
        self.origins = origins
        self.options = origins * profiles
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        self.schedule = np.concatenate(
            [rng.permutation(self.options) for _ in range(math.ceil(self.batches / self.options))]
        )[: self.batches]

    def for_row(self, index: int) -> tuple[int, int]:
        choice = int(self.schedule[index // self.batch_size])
        return choice % self.origins, choice // self.origins


class OnePassProfileDataset(Dataset):
    """Use one precomputed missing-feed profile for each series batch per epoch."""

    def __init__(self, datasets, batch_size: int, seed: int):
        if (
            not datasets
            or not len(datasets[0])
            or any(len(d) != len(datasets[0]) for d in datasets)
        ):
            raise ValueError("TimesFM-HNB profile datasets must contain the same eligible series")
        self.datasets = datasets
        self.choices = EpochChoices(len(datasets[0]), batch_size, 1, len(datasets), seed)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        _, profile = self.choices.for_row(index)
        return self.datasets[profile][index]

    def set_epoch(self, epoch: int) -> None:
        self.choices.set_epoch(epoch)


class OnlineWindows(Dataset):
    """Visit each eligible series once while sampling rolling windows and missing feeds."""

    def __init__(self, data, origins, horizon, context, grouping, profiles, batch_size, seed):
        if data.panel.plan is not None:
            raise ValueError("V8 currently requires materialized known-future snapshots")
        self.data = data
        if isinstance(origins, str):
            origins = (origins,)
        self.horizon, self.context, self.grouping = horizon, context, grouping
        self.origins, self.profiles = origins, profiles
        self.group_index = {row: index for index, group in enumerate(grouping) for row in group}
        availability = []
        for origin in origins:
            position = data.position(origin)
            admitted = data.observed[:, position - context + 1 : position + 1].any(1)
            admitted &= data.observed[:, position + 1 : position + horizon + 1].any(1)
            availability.append(admitted)
        self.availability = np.stack(availability)
        self.rows = [row for group in grouping for row in group if self.availability[:, row].any()]
        if not self.rows:
            raise ValueError("V8 training window has no observed history/target pairs")
        self.choices = EpochChoices(len(self.rows), batch_size, len(origins), len(profiles), seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        origin_index, profile_index = self.choices.for_row(index)
        if not self.availability[origin_index, row]:
            eligible = np.flatnonzero(self.availability[:, row])
            rng = np.random.default_rng(
                np.random.SeedSequence([self.choices.seed, self.choices.epoch, index, row])
            )
            origin_index = int(rng.choice(eligible))
        return (
            self.origins[origin_index], self.profiles[profile_index],
            self.group_index[row], row,
        )

    def set_epoch(self, epoch: int) -> None:
        self.choices.set_epoch(epoch)

    def collate(self, samples):
        buckets = {}
        for origin, absent, group_index, row in samples:
            buckets.setdefault((origin, absent, group_index), []).append(row)
        microbatches = []
        data = self.data
        for (origin, absent, group_index), selected in buckets.items():
            group = self.grouping[group_index]
            rows = list(group)
            subset = replace(
                data,
                series=tuple(data.series[i] for i in rows),
                target=data.target[rows],
                observed=data.observed[rows],
                covariates=data.covariates[rows],
                static=data.static[rows],
                categories=data.categories[rows],
                metadata={key: [values[i] for i in rows] for key, values in data.metadata.items()},
            )
            requests = []
            inputs, _ = features(
                subset,
                origin,
                self.horizon,
                self.context,
                (tuple(range(len(rows))),),
                None,
                Path("."),
                absent=absent,
                requests_out=requests,
                progress=False,
            )
            positions = [rows.index(row) for row in selected]
            admitted_rows = np.flatnonzero(inputs["admitted"]).tolist()
            take = [admitted_rows.index(index) for index in positions]
            tensors = {
                name: torch.from_numpy(value[positions].copy())
                for name, value in inputs.items()
                if name != "admitted"
            }
            tensors["series_index"] = torch.tensor(selected, dtype=torch.long)
            start = data.position(origin) + 1
            tensors["target"] = torch.from_numpy(
                data.target[selected, start : start + self.horizon].copy()
            )
            tensors["observed"] = torch.from_numpy(
                data.observed[selected, start : start + self.horizon].copy()
            )
            microbatches.append({"inputs": tensors, "request": requests[0], "take": take})
        return {"microbatches": microbatches, "n_samples": len(samples)}


class LoRACountModule(CountModule):
    """Lightning manual optimization releases each group's graph before the next group."""

    def __init__(
        self, architecture, learning_rate, weight_decay, mean_loss_weight, source, cache_dir=None
    ):
        super().__init__(architecture, learning_rate, weight_decay, mean_loss_weight)
        # The provider is not an nn.Module: keep its device independent of the CPU head.
        self.source = source
        self.cache_dir = cache_dir
        self.lora = cache_dir is None
        self.automatic_optimization = False
        self.last_progress = 0.0

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Inputs already live on CPU; immutable NumPy source requests keep their identity.
        return batch

    def _condition(self, microbatch):
        batch = dict(microbatch["inputs"])
        if self.lora:
            quantiles = self.source.tensor_quantiles(microbatch["request"])[
                microbatch["take"]
            ].cpu()
        else:
            values, _ = cached_quantiles(self.source, microbatch["request"], self.cache_dir)
            quantiles = torch.from_numpy(values[microbatch["take"]])
        adapted = torch.asinh(quantiles / batch["source_scale"][:, None, None])
        batch["features"] = torch.cat((adapted, batch["features"][..., 9:]), dim=-1)
        return batch

    def _trainable(self):
        return [
            *self.head.parameters(),
            *([p for p in self.source.model.parameters() if p.requires_grad] if self.lora else []),
        ]

    def training_step(self, batch, batch_idx):
        if self.lora:
            self.source.model.train()
            self.source.sealed = False
        optimizer = self.optimizers()
        optimizer.zero_grad()
        total = torch.zeros((), dtype=torch.float64)
        for microbatch in batch["microbatches"]:
            inputs = self._condition(microbatch)
            contribution = self._loss(inputs) * len(inputs["target"]) / batch["n_samples"]
            self.manual_backward(contribution)
            total = total + contribution.detach().cpu()
        torch.nn.utils.clip_grad_norm_(
            self._trainable(), 1.0, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
        now = time.monotonic()
        if now - self.last_progress >= 30:
            label = "V8" if self.lora else "TimesFM-HNB"
            print(
                f"{label} epoch={self.current_epoch + 1} batch={batch_idx + 1}/"
                f"{self.trainer.num_training_batches} updates={self.global_step} "
                f"loss={float(total):.6f}",
                flush=True,
            )
            self.last_progress = now
        return {"loss": total, "n_samples": batch["n_samples"]}

    def validation_step(self, batch, batch_idx):
        if self.lora:
            self.source.model.eval()
        total = torch.zeros((), dtype=torch.float64)
        for microbatch in batch["microbatches"]:
            inputs = self._condition(microbatch)
            total = total + self._loss(inputs) * len(inputs["target"]) / batch["n_samples"]
        self.log("val_loss", total, on_step=False, on_epoch=True, batch_size=batch["n_samples"])

    def configure_optimizers(self):
        groups = [
            {"params": self.head.parameters(), "lr": self.learning_rate},
            {
                "params": [p for p in self.source.model.parameters() if p.requires_grad]
                if self.lora
                else [],
                "lr": LORA_RECIPE["learning_rate"],
            },
        ]
        if not self.lora:
            groups = groups[:1]
        return torch.optim.AdamW(groups, weight_decay=self.weight_decay)

    def on_save_checkpoint(self, checkpoint):
        if self.lora:
            checkpoint["source_lora_state"] = self.source.adapter_state()

    def on_load_checkpoint(self, checkpoint):
        if self.lora:
            self.source.load_adapter(checkpoint["source_lora_state"])
