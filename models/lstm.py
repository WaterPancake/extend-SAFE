"""SAFE-LSTM benchmark failure detector."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .safe_losses import safe_cumulative_loss, timestep_bce_loss


class SafeLSTMModel(BaseModel):
    """Small causal LSTM baseline from SAFE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 1,
        dropout: float = 0.0,
        cumsum: bool = False,
        running_mean: bool = False,
        loss_type: str = "bce",
        threshold: float = 1.0,
        use_threshold: bool = False,
        use_time_weighting: bool = False,
    ) -> None:
        super().__init__(input_dim)
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if loss_type not in {"safe", "bce"}:
            raise ValueError(f"Unsupported loss_type={loss_type!r}")

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cumsum = cumsum
        self.running_mean = running_mean
        self.loss_type = loss_type
        self.threshold = threshold
        self.use_threshold = use_threshold
        self.use_time_weighting = use_time_weighting

        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3:
            raise ValueError(f"features must be 3-D, got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {features.shape[-1]}, expected {self.input_dim}")

        hidden, _ = self.lstm(features)
        hidden = self.dropout(hidden)
        scores = torch.sigmoid(self.head(hidden))
        if self.cumsum or self.running_mean:
            scores = scores.cumsum(dim=1)
        if self.running_mean:
            normalizer = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=scores.dtype)
            scores = scores / normalizer.view(1, -1, 1)
        return scores

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] | tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        scores = self(batch)
        class_weights = tuple(weights) if weights is not None else None
        if self.loss_type == "safe":
            return safe_cumulative_loss(
                scores,
                batch,
                threshold=self.threshold,
                use_threshold=self.use_threshold,
                use_time_weighting=self.use_time_weighting,
                class_weights=class_weights,
            )
        return timestep_bce_loss(
            scores,
            batch,
            use_time_weighting=self.use_time_weighting,
            class_weights=class_weights,
        )

    forward_compute_loss = forward_loss


LstmModel = SafeLSTMModel
