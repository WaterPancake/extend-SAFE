"""Multi-layer VLA failure detector with learned language-layer mixing."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .safe_losses import safe_cumulative_loss, timestep_bce_loss


class LayerMixLSTMModel(BaseModel):
    """Project each selected LM layer, learn a layer mix, then run an LSTM."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim_per_layer: int,
        projection_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        loss_type: str = "bce",
        threshold: float = 1.0,
        use_time_weighting: bool = False,
    ) -> None:
        super().__init__(input_dim)
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if input_dim != n_layers * hidden_dim_per_layer:
            raise ValueError(
                f"input_dim must equal n_layers * hidden_dim_per_layer, got "
                f"{input_dim} != {n_layers} * {hidden_dim_per_layer}"
            )
        if loss_type not in {"safe", "bce"}:
            raise ValueError(f"Unsupported loss_type={loss_type!r}")

        self.n_layers = n_layers
        self.hidden_dim_per_layer = hidden_dim_per_layer
        self.projection_dim = projection_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.loss_type = loss_type
        self.threshold = threshold
        self.use_time_weighting = use_time_weighting

        self.layer_projection = nn.Sequential(
            nn.Linear(hidden_dim_per_layer, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            projection_dim,
            lstm_hidden_dim,
            lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden_dim, 1)

    @property
    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits.detach(), dim=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3:
            raise ValueError(f"features must be 3-D, got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {features.shape[-1]}, expected {self.input_dim}")

        batch_size, seq_len, _ = features.shape
        layered = features.reshape(batch_size, seq_len, self.n_layers, self.hidden_dim_per_layer)
        projected = self.layer_projection(layered)
        weights = torch.softmax(self.layer_logits, dim=0).view(1, 1, self.n_layers, 1)
        mixed = (projected * weights).sum(dim=2)
        hidden, _ = self.lstm(mixed)
        hidden = self.dropout(hidden)
        return torch.sigmoid(self.head(hidden))

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
                use_time_weighting=self.use_time_weighting,
                class_weights=class_weights,
            )
        loss, logs = timestep_bce_loss(
            scores,
            batch,
            use_time_weighting=self.use_time_weighting,
            class_weights=class_weights,
        )
        for idx, value in enumerate(self.layer_weights.cpu().tolist()):
            logs[f"layer_weight_{idx}"] = value
        return loss, logs

    forward_compute_loss = forward_loss
