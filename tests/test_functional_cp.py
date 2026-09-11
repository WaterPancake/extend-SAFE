from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_conformal import RolloutScores, functional_cp_thresholds
from scripts.evaluate_functional_cp_loto import evaluate_family


def record(
    *, task: int, episode: int, label: int, scores: list[float], task_min: int
) -> RolloutScores:
    return RolloutScores(
        path=f"task{task}--ep{episode}--succ{1-label}.pkl",
        task_id=task,
        episode_idx=episode,
        label=label,
        success=not bool(label),
        length=len(scores),
        task_min_step=task_min,
        scores=np.asarray(scores, dtype=np.float64),
    )


def synthetic_folds() -> tuple[
    dict[int, dict[int, RolloutScores]], dict[int, dict[str, list[int]]]
]:
    records = {}
    splits = {}
    for fold in range(10):
        seen_task = (fold + 1) % 10
        fold_records: dict[int, RolloutScores] = {}
        for index in range(6):
            label = int(index >= 4)
            base = 0.15 + 0.02 * index
            fold_records[index] = record(
                task=seen_task,
                episode=index,
                label=label,
                scores=[base + 0.01 * step for step in range(8)],
                task_min=6,
            )
        for index in range(6, 10):
            label = int(index >= 8)
            base = (0.18 if label == 0 else 0.45) + 0.01 * index
            fold_records[index] = record(
                task=fold,
                episode=index,
                label=label,
                scores=[base + 0.015 * step for step in range(8)],
                task_min=6,
            )
        records[fold] = fold_records
        splits[fold] = {"val": list(range(6)), "test": list(range(6, 10))}
    return records, splits


def test_functional_cp_thresholds_are_calibration_only() -> None:
    calibration = [
        record(task=1, episode=i, label=0, scores=[0.1 + i * 0.01] * 6, task_min=5)
        for i in range(6)
    ]
    low_test = [record(task=0, episode=0, label=1, scores=[0.0] * 6, task_min=5)]
    high_test = [record(task=0, episode=0, label=1, scores=[100.0] * 6, task_min=5)]
    low_thresholds, low_info = functional_cp_thresholds(
        calibration, low_test, alpha=0.2, seed=3, horizon=6
    )
    high_thresholds, high_info = functional_cp_thresholds(
        calibration, high_test, alpha=0.2, seed=3, horizon=6
    )
    np.testing.assert_allclose(low_thresholds, high_thresholds)
    assert low_info == high_info


def test_loto_fixed_horizon_evaluation_has_no_extension() -> None:
    records, splits = synthetic_folds()
    raw = evaluate_family(
        "mlp",
        records,
        splits,
        alphas=[0.2],
        cp_seeds=[0],
        horizon=8,
        fixed_horizons=[4],
    )
    assert len(raw) == 30
    assert set(raw["eval_time"]) == {
        "by final end",
        "by earliest stop",
        "fixed horizon 4",
    }
    fixed = raw[raw["eval_time"] == "fixed horizon 4"]
    assert len(fixed) == 10
    assert fixed["roc_auc_max_score"].eq(1.0).all()

    with pytest.raises(ValueError, match="must not extend or censor"):
        evaluate_family(
            "mlp",
            records,
            splits,
            alphas=[0.2],
            cp_seeds=[0],
            horizon=8,
            fixed_horizons=[9],
        )


def test_loto_rejects_heldout_task_in_calibration() -> None:
    records, splits = synthetic_folds()
    records[0][0].task_id = 0
    with pytest.raises(ValueError, match="leaks"):
        evaluate_family(
            "mlp",
            records,
            splits,
            alphas=[0.2],
            cp_seeds=[0],
            horizon=8,
        )


def test_loto_rejects_fold_task_mismatch() -> None:
    records, splits = synthetic_folds()
    for index in splits[0]["test"]:
        records[0][index].task_id = 2
    with pytest.raises(ValueError, match="expected held-out task 0, found task 2"):
        evaluate_family(
            "mlp",
            records,
            splits,
            alphas=[0.2],
            cp_seeds=[0],
            horizon=8,
        )
