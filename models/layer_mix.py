"""Multi-layer VLA failure detectors with learned language-layer structure."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .safe_losses import safe_cumulative_loss, timestep_bce_loss


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax transform from Martins & Astudillo, implemented dependency-free."""

    logits = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    cssv = sorted_logits.cumsum(dim) - 1
    rhos = torch.arange(
        1,
        logits.shape[dim] + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    view = [1] * logits.ndim
    view[dim] = -1
    rhos = rhos.view(view)

    support = rhos * sorted_logits > cssv
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cssv.gather(dim, support_size - 1) / support_size.to(logits.dtype)
    return torch.clamp(logits - tau, min=0.0)


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
        mix_activation: str = "softmax",
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
        if mix_activation not in {"softmax", "sparsemax"}:
            raise ValueError(f"Unsupported mix_activation={mix_activation!r}")

        self.n_layers = n_layers
        self.hidden_dim_per_layer = hidden_dim_per_layer
        self.projection_dim = projection_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.loss_type = loss_type
        self.threshold = threshold
        self.use_time_weighting = use_time_weighting
        self.mix_activation = mix_activation

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
        return self._mix_weights(self.layer_logits.detach())

    def _mix_weights(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mix_activation == "softmax":
            return torch.softmax(logits, dim=0)
        return sparsemax(logits, dim=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3:
            raise ValueError(f"features must be 3-D, got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {features.shape[-1]}, expected {self.input_dim}")

        batch_size, seq_len, _ = features.shape
        layered = features.reshape(batch_size, seq_len, self.n_layers, self.hidden_dim_per_layer)
        projected = self.layer_projection(layered)
        weights = self._mix_weights(self.layer_logits).view(1, 1, self.n_layers, 1)
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


class SparseLayerMixLSTMModel(LayerMixLSTMModel):
    """Layer mixer with sparsemax gates so unused layers can receive zero mass."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["mix_activation"] = "sparsemax"
        super().__init__(*args, **kwargs)


class LayerTokenTransformerLSTMModel(BaseModel):
    """Encode LM layers as tokens per timestep, then model time with an LSTM."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim_per_layer: int,
        projection_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        encoder_layers: int = 1,
        encoder_heads: int = 4,
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
        if projection_dim % encoder_heads != 0:
            raise ValueError(
                f"projection_dim must be divisible by encoder_heads, got "
                f"{projection_dim} and {encoder_heads}"
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

        self.layer_projection = nn.Linear(hidden_dim_per_layer, projection_dim)
        self.layer_positional_embedding = nn.Parameter(torch.zeros(1, n_layers, projection_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, projection_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=encoder_heads,
            dim_feedforward=projection_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.layer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
        )

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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3:
            raise ValueError(f"features must be 3-D, got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {features.shape[-1]}, expected {self.input_dim}")

        batch_size, seq_len, _ = features.shape
        layered = features.reshape(batch_size, seq_len, self.n_layers, self.hidden_dim_per_layer)
        layer_tokens = self.layer_projection(layered)
        layer_tokens = layer_tokens + self.layer_positional_embedding.view(
            1, 1, self.n_layers, self.projection_dim
        )
        layer_tokens = layer_tokens.reshape(batch_size * seq_len, self.n_layers, self.projection_dim)
        cls = self.cls_token.expand(batch_size * seq_len, -1, -1)
        encoded = self.layer_encoder(torch.cat([cls, layer_tokens], dim=1))
        timestep_features = encoded[:, 0].reshape(batch_size, seq_len, self.projection_dim)
        hidden, _ = self.lstm(timestep_features)
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
        return timestep_bce_loss(
            scores,
            batch,
            use_time_weighting=self.use_time_weighting,
            class_weights=class_weights,
        )

    forward_compute_loss = forward_loss
