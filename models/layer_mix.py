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


class DynamicLayerMixLSTMModel(BaseModel):
    """Causally gate VLA layers at every timestep before temporal modeling.

    The projection, gate, LSTM, and head are shared across layers. Therefore
    the trainable parameter count is independent of ``n_layers``: comparisons
    between one and multiple layers control for model capacity.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim_per_layer: int,
        projection_dim: int = 32,
        lstm_hidden_dim: int = 64,
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
                "input_dim must equal n_layers * hidden_dim_per_layer, got "
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

        self.layer_norm = nn.LayerNorm(hidden_dim_per_layer)
        self.layer_projection = nn.Sequential(
            nn.Linear(hidden_dim_per_layer, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_gate = nn.Linear(projection_dim, 1, bias=False)

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

    def _project_and_gate(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = features.shape
        layered = features.reshape(
            batch_size, seq_len, self.n_layers, self.hidden_dim_per_layer
        )
        projected = self.layer_projection(self.layer_norm(layered))
        weights = torch.softmax(self.layer_gate(projected), dim=2)
        return (projected * weights).sum(dim=2), weights

    def gate_weights(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-timestep layer weights for diagnostics."""

        _, weights = self._project_and_gate(batch["features"])
        return weights.squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3:
            raise ValueError(f"features must be 3-D, got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"input_dim mismatch: got {features.shape[-1]}, expected {self.input_dim}"
            )

        mixed, _ = self._project_and_gate(features)
        hidden, _ = self.lstm(mixed)
        return torch.sigmoid(self.head(self.dropout(hidden)))

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


class ResidualAuxLayerLSTMModel(BaseModel):
    """Add a zero-initialized auxiliary-layer correction to a frozen monitor.

    Inputs concatenate the auxiliary layer first and the final layer second.
    The final-layer monitor is frozen, so the auxiliary branch cannot erase its
    representation; at initialization this model is exactly the baseline.
    """

    def __init__(
        self,
        hidden_dim_per_layer: int,
        base_projection_dim: int = 32,
        base_lstm_hidden_dim: int = 64,
        aux_projection_dim: int = 16,
        aux_lstm_hidden_dim: int = 32,
        dropout: float = 0.0,
        use_time_weighting: bool = False,
    ) -> None:
        super().__init__(2 * hidden_dim_per_layer)
        self.hidden_dim_per_layer = hidden_dim_per_layer
        self.use_time_weighting = use_time_weighting
        self.base = DynamicLayerMixLSTMModel(
            input_dim=hidden_dim_per_layer,
            n_layers=1,
            hidden_dim_per_layer=hidden_dim_per_layer,
            projection_dim=base_projection_dim,
            lstm_hidden_dim=base_lstm_hidden_dim,
            dropout=dropout,
            loss_type="bce",
        )
        self.aux_norm = nn.LayerNorm(hidden_dim_per_layer)
        self.aux_projection = nn.Sequential(
            nn.Linear(hidden_dim_per_layer, aux_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_lstm = nn.LSTM(
            aux_projection_dim, aux_lstm_hidden_dim, batch_first=True
        )
        self.aux_head = nn.Linear(aux_lstm_hidden_dim, 1)
        nn.init.zeros_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def residual_logits(self, features: torch.Tensor) -> torch.Tensor:
        auxiliary = features[:, :, : self.hidden_dim_per_layer]
        projected = self.aux_projection(self.aux_norm(auxiliary))
        hidden, _ = self.aux_lstm(projected)
        return self.aux_head(hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected features (*, *, {self.input_dim}), got {tuple(features.shape)}"
            )
        final_layer = features[:, :, self.hidden_dim_per_layer :]
        with torch.no_grad():
            base_scores = self.base({"features": final_layer})
        base_logits = torch.logit(base_scores.clamp(1e-5, 1.0 - 1e-5))
        return torch.sigmoid(base_logits + self.residual_logits(features))

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] | tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return timestep_bce_loss(
            self(batch),
            batch,
            use_time_weighting=self.use_time_weighting,
            class_weights=tuple(weights) if weights is not None else None,
        )

    forward_compute_loss = forward_loss


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
