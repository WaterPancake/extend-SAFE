from __future__ import annotations

import pickle
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from scripts.score_layer_pooled_macro import MixedSchemaLayerDataset
from scripts.prepare_openvla_layer_subset import (
    select_hidden_subset,
    validate_output_audit,
)


def write_rollout(
    path: Path,
    *,
    task: int,
    episode: int,
    success: bool,
    length: int,
    pooled: bool = False,
) -> None:
    layers = [1, 32]
    hidden_dim = 4
    values = torch.arange(length * 2 * len(layers) * hidden_dim).reshape(
        length, 2, len(layers) * hidden_dim
    )
    hidden = values[:, -1] if pooled else values
    artifact = {
        "hidden_states": hidden.to(torch.bfloat16),
        "hidden_state_layers": layers,
        "hidden_state_dim_per_layer": hidden_dim,
        "task_id": task,
        "episode_idx": episode,
        "episode_success": success,
    }
    if pooled:
        artifact["hidden_state_token_index"] = -1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def test_openvla_dataset_selects_layer_and_masks_task_min(tmp_path: Path) -> None:
    write_rollout(
        tmp_path / "task0--ep0--succ1.pkl",
        task=0,
        episode=0,
        success=True,
        length=3,
    )
    write_rollout(
        tmp_path / "task0--ep1--succ0.pkl",
        task=0,
        episode=1,
        success=False,
        length=5,
    )
    dataset = OpenVLARolloutDataset(
        tmp_path, layers=(32,), token_pool="last", cache=False
    )

    assert len(dataset) == 2
    assert dataset.hidden_dim == 4
    assert dataset.task_min_steps == {0: 3}
    assert dataset[0]["features"].shape == (3, 4)
    assert int(dataset[1]["task_min_step"]) == 3

    batch = collate_rollouts([dataset[0], dataset[1]])
    assert batch["features"].shape == (2, 5, 4)
    assert batch["valid_masks"].sum(dim=1).tolist() == [3, 5]
    assert batch["labels"].tolist() == [0.0, 1.0]


def test_mixed_schema_dataset_preserves_root_order(tmp_path: Path) -> None:
    root_a = tmp_path / "shard-a"
    root_b = tmp_path / "shard-b"
    write_rollout(
        root_a / "task0--ep0--succ1.pkl",
        task=0,
        episode=0,
        success=True,
        length=3,
    )
    write_rollout(
        root_b / "task1--ep0--succ0.pkl",
        task=1,
        episode=0,
        success=False,
        length=4,
        pooled=True,
    )

    dataset = MixedSchemaLayerDataset([root_a, root_b], layer=32, token_pool="last")
    assert len(dataset) == 2
    assert dataset[0]["path"].startswith(str(root_a))
    assert dataset[1]["path"].startswith(str(root_b))
    assert dataset[0]["features"].shape == (3, 4)
    assert dataset[1]["features"].shape == (4, 4)


def test_layer_subset_normalizes_3d_and_prepooled_2d_sources() -> None:
    layers = [1, 20, 32]
    hidden = torch.arange(4 * 2 * 3 * 2).reshape(4, 2, 6)
    common = {
        "hidden_state_layers": layers,
        "hidden_state_dim_per_layer": 2,
    }
    token_preserving = {**common, "hidden_states": hidden}
    prepooled = {
        **common,
        "hidden_states": hidden[:, -1],
        "hidden_state_token_index": -1,
    }

    from_3d, token_3d = select_hidden_subset(
        token_preserving, (20, 32), token_index=-1, temporal_stride=2
    )
    from_2d, token_2d = select_hidden_subset(
        prepooled, (20, 32), token_index=-1, temporal_stride=2
    )

    assert from_3d.shape == (2, 1, 4)
    torch.testing.assert_close(from_3d, from_2d)
    assert token_3d == 1
    assert token_2d == -1


def test_layer_subset_rejects_token_selection_from_prepooled_source() -> None:
    artifact = {
        "hidden_states": torch.zeros(3, 4),
        "hidden_state_layers": [1, 32],
        "hidden_state_dim_per_layer": 2,
    }
    with pytest.raises(ValueError, match="no action-token axis"):
        select_hidden_subset(artifact, (32,), token_index=0, temporal_stride=1)


def test_layer_subset_output_contract_rejects_short_collection() -> None:
    audit = {
        "homogeneous_schema": True,
        "rollouts": 500,
        "task_counts": {str(task): 50 for task in range(10)},
        "minimum_length": 147,
    }
    args = Namespace(
        expected_rollouts=500,
        expected_tasks=tuple(range(10)),
        expected_rollouts_per_task=50,
        minimum_length=148,
    )
    with pytest.raises(ValueError, match="minimum length"):
        validate_output_audit(audit, args)
