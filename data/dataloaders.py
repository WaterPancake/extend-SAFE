"""PyTorch dataloaders for SAFE-style OpenVLA rollout latent artifacts."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


TokenPool = Literal["mean", "first", "last", "first_last", "none"]

ROLLOUT_NAME_RE = re.compile(
    r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])\.pkl$"
)


@dataclass(frozen=True)
class RolloutInfo:
    path: Path
    task_id: int | None
    episode_idx: int | None
    success: bool | None


class OpenVLARolloutDataset(Dataset):
    """Dataset for rollout `.pkl` files with concatenated VLA hidden states.

    expected shape:

        hidden_states: (steps, action_tokens, n_saved_layers * hidden_dim)

    `layers` are model-layer ids from the artifact metadata, not zero-based
    positions. For example, OpenVLA artifacts captured every fourth layer may
    contain `[1, 4, 8, 12, 16, 20, 24, 28, 32]`. Passing `layers=[32, 28]`
    returns those two layers in that order. Passing `layers=None` uses only the
    final saved layer.
    """

    def __init__(
        self,
        root: str | Path,
        layers: Sequence[int] | None = None,
        token_pool: TokenPool = "mean",
        recursive: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.requested_layers = tuple(layers) if layers is not None else None
        self.token_pool = token_pool

        if token_pool not in {"mean", "first", "last", "first_last", "none"}:
            raise ValueError(f"Unsupported token_pool={token_pool!r}")

        pattern = "**/*.pkl" if recursive else "*.pkl"
        self.rollouts = [
            self._parse_rollout_path(path) for path in sorted(self.root.glob(pattern))
        ]
        if not self.rollouts:
            raise FileNotFoundError(f"No .pkl rollout files found under {self.root}")

        first = self._load_pickle(self.rollouts[0].path)
        self.saved_layers = self._read_saved_layers(first, self.rollouts[0].path)
        self.hidden_dim = self._read_hidden_dim(first, self.rollouts[0].path)
        self.selected_layers = self._resolve_layers(
            self.requested_layers, self.saved_layers
        )
        self.layer_positions = tuple(
            self.saved_layers.index(layer) for layer in self.selected_layers
        )
        self.input_dim = self._compute_input_dim()
        self.task_min_steps = self._compute_task_min_steps()

        self._validate_artifact_schema(first, self.rollouts[0].path)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.rollouts[index]
        artifact = self._load_pickle(info.path)
        hidden_states = self._extract_features(artifact, info.path)
        success = self._read_success(artifact, info)
        task_id = self._read_task_id(artifact, info)
        length = int(hidden_states.shape[0])

        return {
            "features": hidden_states,
            "label": torch.tensor(float(not success), dtype=torch.float32),
            "success": torch.tensor(float(success), dtype=torch.float32),
            "length": torch.tensor(length, dtype=torch.long),
            "task_min_step": torch.tensor(
                self._task_min_step(task_id, length), dtype=torch.long
            ),
            "path": str(info.path),
            "task_id": task_id,
            "episode_idx": self._read_episode_idx(artifact, info),
            "selected_layers": self.selected_layers,
            "saved_layers": tuple(self.saved_layers),
        }

    def _extract_features(self, artifact: dict[str, Any], path: Path) -> torch.Tensor:
        self._validate_artifact_schema(artifact, path)
        hidden_states = (
            artifact["hidden_states"].detach().to(dtype=torch.float32, device="cpu")
        )
        steps, action_tokens, _ = hidden_states.shape
        n_saved_layers = len(self.saved_layers)

        hidden_states = hidden_states.reshape(
            steps, action_tokens, n_saved_layers, self.hidden_dim
        )
        hidden_states = hidden_states[:, :, self.layer_positions, :]

        if self.token_pool == "mean":
            features = hidden_states.mean(dim=1)
        elif self.token_pool == "first":
            features = hidden_states[:, 0]
        elif self.token_pool == "last":
            features = hidden_states[:, -1]
        elif self.token_pool == "first_last":
            features = torch.cat([hidden_states[:, 0], hidden_states[:, -1]], dim=-1)
        else:
            features = hidden_states

        return features.reshape(steps, -1)

    def _compute_input_dim(self) -> int:
        token_multiplier = 2 if self.token_pool == "first_last" else 1
        if self.token_pool == "none":
            token_multiplier = self._peek_action_tokens()
        return len(self.selected_layers) * self.hidden_dim * token_multiplier

    def _peek_action_tokens(self) -> int:
        artifact = self._load_pickle(self.rollouts[0].path)
        return int(artifact["hidden_states"].shape[1])

    def _compute_task_min_steps(self) -> dict[int, int]:
        """Per-task min length over all rollouts (matches SAFE set_task_min_step)."""

        task_min_steps: dict[int, int] = {}
        for info in self.rollouts:
            artifact = self._load_pickle(info.path)
            self._validate_artifact_schema(artifact, info.path)
            task_id = self._read_task_id(artifact, info)
            if task_id is None:
                continue
            length = int(artifact["hidden_states"].shape[0])
            current = task_min_steps.get(task_id)
            if current is None or length < current:
                task_min_steps[task_id] = length
        return task_min_steps

    def _task_min_step(self, task_id: int | None, length: int) -> int:
        if task_id is None:
            return length
        return min(self.task_min_steps.get(task_id, length), length)

    def _validate_artifact_schema(self, artifact: dict[str, Any], path: Path) -> None:
        if "hidden_states" not in artifact:
            raise KeyError(f"{path} is missing 'hidden_states'")
        hidden_states = artifact["hidden_states"]
        if hidden_states.ndim != 3:
            raise ValueError(
                f"{path} hidden_states must be 3-D, got shape {tuple(hidden_states.shape)}"
            )

        saved_layers = self._read_saved_layers(artifact, path)
        hidden_dim = self._read_hidden_dim(artifact, path)
        expected_width = len(saved_layers) * hidden_dim
        actual_width = int(hidden_states.shape[-1])
        if actual_width != expected_width:
            raise ValueError(
                f"{path} hidden width mismatch: got {actual_width}, expected "
                f"{len(saved_layers)} layers * {hidden_dim} hidden = {expected_width}"
            )
        if list(saved_layers) != list(self.saved_layers):
            raise ValueError(
                f"{path} saved layers {list(saved_layers)} do not match dataset saved layers {self.saved_layers}"
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"{path} hidden_dim {hidden_dim} does not match dataset hidden_dim {self.hidden_dim}"
            )

    def _resolve_layers(
        self, requested: tuple[int, ...] | None, saved_layers: list[int]
    ) -> tuple[int, ...]:
        if requested is None:
            return (saved_layers[-1],)

        missing = [layer for layer in requested if layer not in saved_layers]
        if missing:
            raise ValueError(
                f"Requested layers {missing} are not in saved artifact layers {saved_layers}. "
                "Pass model-layer ids that were actually captured."
            )
        if not requested:
            raise ValueError("layers must contain at least one layer id")
        return requested

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, dict):
            raise TypeError(
                f"{path} must contain a dict, got {type(artifact).__name__}"
            )
        return artifact

    @staticmethod
    def _read_saved_layers(artifact: dict[str, Any], path: Path) -> list[int]:
        layers = artifact.get(
            "hidden_state_layers", artifact.get("hidden_states_layers")
        )
        if layers is None:
            raise KeyError(f"{path} is missing 'hidden_state_layers'")
        return [int(layer) for layer in layers]

    @staticmethod
    def _read_hidden_dim(artifact: dict[str, Any], path: Path) -> int:
        if "hidden_state_dim_per_layer" not in artifact:
            raise KeyError(f"{path} is missing 'hidden_state_dim_per_layer'")
        return int(artifact["hidden_state_dim_per_layer"])

    @staticmethod
    def _parse_rollout_path(path: Path) -> RolloutInfo:
        match = ROLLOUT_NAME_RE.search(path.name)
        if match is None:
            return RolloutInfo(path=path, task_id=None, episode_idx=None, success=None)
        return RolloutInfo(
            path=path,
            task_id=int(match.group("task_id")),
            episode_idx=int(match.group("episode_idx")),
            success=bool(int(match.group("success"))),
        )

    @staticmethod
    def _read_success(artifact: dict[str, Any], info: RolloutInfo) -> bool:
        if "episode_success" in artifact:
            return bool(artifact["episode_success"])
        if info.success is not None:
            return info.success
        raise KeyError(f"Could not infer success label for {info.path}")

    @staticmethod
    def _read_task_id(artifact: dict[str, Any], info: RolloutInfo) -> int | None:
        if "task_id" in artifact:
            return int(artifact["task_id"])
        return info.task_id

    @staticmethod
    def _read_episode_idx(artifact: dict[str, Any], info: RolloutInfo) -> int | None:
        if "episode_idx" in artifact:
            return int(artifact["episode_idx"])
        if "eposide_idx" in artifact:
            return int(artifact["eposide_idx"])
        return info.episode_idx


def collate_rollouts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length rollout features into a batch."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    lengths = torch.tensor(
        [item["features"].shape[0] for item in batch], dtype=torch.long
    )
    input_dim = int(batch[0]["features"].shape[-1])
    max_len = int(lengths.max().item())

    features = torch.zeros(len(batch), max_len, input_dim, dtype=torch.float32)
    valid_masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for row, item in enumerate(batch):
        seq = item["features"]
        if seq.shape[-1] != input_dim:
            raise ValueError(
                f"Batch item {row} has input_dim {seq.shape[-1]}, expected {input_dim}"
            )
        length = seq.shape[0]
        features[row, :length] = seq
        valid_masks[row, :length] = True

    return {
        "features": features,
        "valid_masks": valid_masks,
        "lengths": lengths,
        "task_min_steps": torch.stack([item["task_min_step"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "success": torch.stack([item["success"] for item in batch]),
        "paths": [item["path"] for item in batch],
        "task_ids": [item["task_id"] for item in batch],
        "episode_indices": [item["episode_idx"] for item in batch],
        "selected_layers": batch[0]["selected_layers"],
        "saved_layers": batch[0]["saved_layers"],
    }


def create_rollout_dataloader(
    root: str | Path,
    layers: Sequence[int] | None = None,
    token_pool: TokenPool = "mean",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    recursive: bool = True,
) -> DataLoader:
    """Create a DataLoader for OpenVLA rollout `.pkl` artifacts."""

    dataset = OpenVLARolloutDataset(
        root=root,
        layers=layers,
        token_pool=token_pool,
        recursive=recursive,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_rollouts,
    )
