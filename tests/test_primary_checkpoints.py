from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.verify_primary_checkpoints import (
    audit_checkpoint_dirs,
    expected_test_indices,
)


def write_family(root: Path, model: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    all_indices = set(range(1_000))
    for fold in range(10):
        test = expected_test_indices(fold)
        seen = sorted(all_indices - test)
        split = {
            "train": seen[:540],
            "val": seen[540:],
            "test": sorted(test),
        }
        torch.save(
            {
                "experiment": {"model_type": model, "seed": fold},
                "result": {"model_type": model, "seed": fold, "n_rollouts": 1_000},
                "split": split,
                "selected_layers": (32,),
                "token_pool": "last",
            },
            root / f"safe_{model}_seed-{fold}.pt",
        )


def test_primary_checkpoint_audit_accepts_exact_contract(tmp_path: Path) -> None:
    mlp = tmp_path / "mlp"
    lstm = tmp_path / "lstm"
    write_family(mlp, "mlp")
    write_family(lstm, "lstm")

    result = audit_checkpoint_dirs(mlp, lstm)

    assert result["status"] == "pass"
    assert result["split_sizes"] == {"train": 540, "val": 360, "test": 100}
    assert set(result["checkpoint_manifest"]) == {"mlp", "lstm"}


def test_primary_checkpoint_audit_rejects_wrong_fold_indices(tmp_path: Path) -> None:
    mlp = tmp_path / "mlp"
    lstm = tmp_path / "lstm"
    write_family(mlp, "mlp")
    write_family(lstm, "lstm")
    path = lstm / "safe_lstm_seed-0.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["split"]["test"][0], checkpoint["split"]["val"][0] = (
        checkpoint["split"]["val"][0],
        checkpoint["split"]["test"][0],
    )
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="test indices do not match"):
        audit_checkpoint_dirs(mlp, lstm)
