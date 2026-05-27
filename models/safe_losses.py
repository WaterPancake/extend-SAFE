"""Loss helpers for SAFE-style failure prediction models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_failure_labels(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return rollout labels where 1 means failure and 0 means success."""

    if "labels" in batch:
        return batch["labels"].float()
    if "failure_labels" in batch:
        return batch["failure_labels"].float()
    if "success_labels" in batch:
        return 1.0 - batch["success_labels"].float()
    if "success" in batch:
        return 1.0 - batch["success"].float()
    raise KeyError("Batch must include labels, failure_labels, success_labels, or success")


def get_valid_masks(batch: dict[str, torch.Tensor], scores: torch.Tensor) -> torch.Tensor:
    """Return boolean valid timestep masks, defaulting to all valid."""

    if "valid_masks" in batch:
        return batch["valid_masks"].bool()
    return torch.ones_like(scores, dtype=torch.bool)


def time_weights(valid_masks: torch.Tensor, use_time_weighting: bool = False) -> torch.Tensor:
    """Create optional SAFE-style per-timestep weights."""

    weights = valid_masks.float()
    if not use_time_weighting:
        return weights

    batch_size, seq_len = valid_masks.shape
    lengths = valid_masks.sum(dim=1).clamp_min(1).float()
    t = torch.arange(seq_len, device=valid_masks.device, dtype=torch.float32)
    normalized_t = t.unsqueeze(0).expand(batch_size, -1) / lengths.unsqueeze(1)
    weights = 5.0 * torch.exp(-3.0 * normalized_t) + 1.0
    weights = weights * valid_masks.float()
    normalizer = (weights.sum(dim=1) / lengths).clamp_min(1e-8)
    return weights / normalizer.unsqueeze(1)


def aggregate_masked_loss(
    losses: torch.Tensor,
    valid_masks: torch.Tensor,
    failure_labels: torch.Tensor,
    class_weights: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average timestep losses per rollout, then average over the batch.

    class_weights is `(failure_weight, success_weight)`.
    """

    valid = valid_masks.float()
    per_rollout = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    failure_mask = failure_labels.bool()
    success_mask = ~failure_mask

    if class_weights is None:
        class_weights = (1.0, 1.0)

    weighted = per_rollout.clone()
    weighted[failure_mask] *= class_weights[0]
    weighted[success_mask] *= class_weights[1]
    monitor_loss = weighted.mean()

    failure_loss = per_rollout[failure_mask].mean() if failure_mask.any() else per_rollout.new_tensor(0.0)
    success_loss = per_rollout[success_mask].mean() if success_mask.any() else per_rollout.new_tensor(0.0)

    return monitor_loss, {
        "monitor_loss": float(monitor_loss.detach().cpu()),
        "failure_loss": float(failure_loss.detach().cpu()),
        "success_loss": float(success_loss.detach().cpu()),
    }


def safe_cumulative_loss(
    scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    threshold: float = 1.0,
    use_threshold: bool = True,
    use_time_weighting: bool = False,
    class_weights: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SAFE-style cumulative score loss.

    Successful rollouts are pushed toward low scores. Failed rollouts are pushed
    above `threshold`, with optional time weighting.
    """

    if scores.ndim == 3:
        scores = scores.squeeze(-1)
    failure_labels = get_failure_labels(batch).to(scores.device)
    valid_masks = get_valid_masks(batch, scores).to(scores.device)
    weights = time_weights(valid_masks, use_time_weighting).to(scores)

    success_losses = F.relu(scores)
    if use_threshold:
        failure_losses = weights * F.relu(threshold - scores)
    else:
        failure_losses = weights * (-scores)
    losses = torch.where(failure_labels[:, None].bool(), failure_losses, success_losses)
    return aggregate_masked_loss(losses, valid_masks, failure_labels, class_weights)


def timestep_bce_loss(
    scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    use_time_weighting: bool = False,
    class_weights: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-timestep binary cross entropy with failure as the positive class."""

    if scores.ndim == 3:
        scores = scores.squeeze(-1)
    failure_labels = get_failure_labels(batch).to(scores.device)
    valid_masks = get_valid_masks(batch, scores).to(scores.device)
    targets = failure_labels[:, None].expand_as(scores)

    losses = F.binary_cross_entropy(scores.clamp(1e-6, 1.0 - 1e-6), targets, reduction="none")
    weights = time_weights(valid_masks, use_time_weighting).to(scores)
    losses = torch.where(failure_labels[:, None].bool(), losses * weights, losses)
    return aggregate_masked_loss(losses, valid_masks, failure_labels, class_weights)
