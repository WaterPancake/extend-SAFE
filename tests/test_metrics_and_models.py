from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from models import DynamicLayerMixLSTMModel, ResidualAuxLayerLSTMModel
from scripts.analyze_pooled_macro_decomposition import decompose
from scripts.evaluate_dynamic_layer_loto import shuffled_donors


def test_pair_decomposition_reconstructs_pooled_auc() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    tasks = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    # Within each task the ranking is imperfect, while task-dependent scale
    # makes cross-task pairs easier.
    scores = np.asarray([0.1, 0.4, 0.5, 0.3, 0.7, 0.9, 0.8, 1.0])
    result = decompose(labels, scores, tasks)
    reconstructed = (
        result.n_within_pairs * result.within_micro_auc
        + result.n_between_pairs * result.between_task_auc
    ) / (result.n_within_pairs + result.n_between_pairs)
    assert np.isclose(result.pooled_auc, reconstructed)
    assert np.isclose(result.between_pair_fraction, 0.5)


def test_dynamic_layer_model_is_capacity_matched() -> None:
    one = DynamicLayerMixLSTMModel(
        input_dim=8, n_layers=1, hidden_dim_per_layer=8, projection_dim=4, lstm_hidden_dim=6
    )
    three = DynamicLayerMixLSTMModel(
        input_dim=24, n_layers=3, hidden_dim_per_layer=8, projection_dim=4, lstm_hidden_dim=6
    )
    assert sum(parameter.numel() for parameter in one.parameters()) == sum(
        parameter.numel() for parameter in three.parameters()
    )
    weights = three.gate_weights({"features": torch.randn(2, 5, 24)})
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))


def test_residual_model_starts_as_exact_baseline() -> None:
    model = ResidualAuxLayerLSTMModel(
        hidden_dim_per_layer=8,
        base_projection_dim=4,
        base_lstm_hidden_dim=6,
        aux_projection_dim=3,
        aux_lstm_hidden_dim=5,
    ).eval()
    features = torch.randn(2, 4, 16)
    combined = model({"features": features})
    baseline = model.base({"features": features[:, :, 8:]})
    torch.testing.assert_close(combined, baseline, atol=1e-6, rtol=1e-6)


def test_shuffled_control_is_within_partition_and_has_no_fixed_points() -> None:
    base = SimpleNamespace(
        rollouts=[
            SimpleNamespace(task_id=task, success=bool(index % 2), path=f"r{index}")
            for task in (0, 1)
            for index in range(6)
        ]
    )
    split = {
        "train": [0, 1, 6, 7],
        "val": [2, 3, 8, 9],
        "test": [4, 5, 10, 11],
    }
    donors = shuffled_donors(base, split, seed=17)
    partition_by_index = {
        index: partition for partition, indices in split.items() for index in indices
    }
    assert set(donors) == set(partition_by_index)
    for index, donor in donors.items():
        assert donor != index
        assert partition_by_index[donor] == partition_by_index[index]
        assert base.rollouts[donor].task_id == base.rollouts[index].task_id
