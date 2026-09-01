"""TimesFM-HNB's supervised count head; the pretrained TimesFM network lives outside autograd."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from forecast.models.mft.sft import (
    SFT_LOG_K_MAX,
    SFT_LOG_K_MIN,
    ShiftedHurdlePrediction,
)
from torch import nn


class CountHead(nn.Module):
    """Quantile/covariate features to hurdle counts with static/series dispersion."""

    def __init__(
        self,
        *,
        n_features: int,
        n_static: int,
        cardinalities: list[int],
        n_series: int,
        width: int,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size + 1, embedding_dim, padding_idx=0) for size in cardinalities
        )
        static_width = n_static + len(cardinalities) * embedding_dim
        self.network = nn.Sequential(
            nn.Linear(n_features + static_width, width), nn.SiLU(), nn.Linear(width, 2)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.dispersion_bias = nn.Parameter(torch.zeros(()))
        self.static_dispersion = nn.Linear(static_width, 1, bias=False) if static_width else None
        if self.static_dispersion is not None:
            nn.init.zeros_(self.static_dispersion.weight)
        self.series_dispersion = nn.Embedding(n_series, 1)
        nn.init.zeros_(self.series_dispersion.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> ShiftedHurdlePrediction:
        attributes = torch.cat(
            [
                batch["static"],
                *(
                    embedding(batch["categories"][:, i])
                    for i, embedding in enumerate(self.embeddings)
                ),
            ],
            dim=-1,
        )
        features = batch["features"]
        static = attributes[:, None, :].expand(-1, features.shape[1], -1)
        parameters = self.network(torch.cat([features, static], dim=-1))
        parameters = parameters + batch["baseline"][:, None, :]
        log_k = self.dispersion_bias + self.series_dispersion(batch["series_index"]).squeeze(-1)
        if self.static_dispersion is not None:
            log_k = log_k + self.static_dispersion(attributes).squeeze(-1)
        dispersion = log_k.double().clamp(SFT_LOG_K_MIN, SFT_LOG_K_MAX).exp()
        return ShiftedHurdlePrediction(
            logit=parameters[..., 0],
            excess=parameters[..., 1].double().exp(),
            dispersion=dispersion[:, None].expand(-1, features.shape[1]),
        )


class CountModule(pl.LightningModule):
    """Reuse Lightning's optimizer, checkpoint, epoch, and exact max-step machinery."""

    def __init__(
        self,
        architecture: dict,
        learning_rate: float,
        weight_decay: float,
        mean_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = CountHead(**architecture)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mean_loss_weight = mean_loss_weight

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = self.head(batch)
        observed = batch["observed"]
        per_cell = prediction.nll(batch["target"], observed)
        per_series = per_cell.sum(dim=-1) / observed.sum(dim=-1)
        loss = per_series.mean()
        if self.mean_loss_weight:
            mean_error = ((prediction.mean() - batch["target"]).square() * observed).mean(-1)
            loss = loss + self.mean_loss_weight * (mean_error / batch["mean_scale"]).mean()
        if not torch.isfinite(loss) or loss < 0:
            raise FloatingPointError("TimesFM-HNB count likelihood is nonfinite or negative")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        return {"loss": loss, "n_samples": len(batch["target"])}

    def validation_step(self, batch, batch_idx):
        self.log(
            "val_loss",
            self._loss(batch),
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["target"]),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
