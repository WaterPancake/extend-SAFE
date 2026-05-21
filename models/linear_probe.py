"""Linear probe failure detector for frozen VLA hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel
from .safe_losses import timestep_bce_loss


class LinearProbeModel(BaseModel):
    """Per-timestep logistic regression baseline.

    This is the lowest-capacity sanity check: one affine map from frozen VLA
    features to a scalar failure probability at each timestep.
    """

    def __init__(
        self,
        input_dim: int,
        use_time_weighting: bool = False,
    ) -> None:
        super().__init__(input_dim)
        self.use_time_weighting = use_time_weighting
        self.head = nn.Linear(input_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if features.ndim != 3:
            raise ValueError(f"features must be 3-D, got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {features.shape[-1]}, expected {self.input_dim}")
        return torch.sigmoid(self.head(features))

    def forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] | tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        class_weights = tuple(weights) if weights is not None else None
        return timestep_bce_loss(
            self(batch),
            batch,
            use_time_weighting=self.use_time_weighting,
            class_weights=class_weights,
        )

    forward_compute_loss = forward_loss
