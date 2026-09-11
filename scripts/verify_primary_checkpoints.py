"""Verify the finalized primary MLP/LSTM checkpoint and split contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_functional_cp_loto import checkpoint_map
from scripts.score_bundle import sha256_file


def expected_test_indices(fold: int) -> set[int]:
    """Indices for task ``fold`` under the ordered two-shard data contract."""

    first = range(fold * 50, (fold + 1) * 50)
    second = range(500 + fold * 50, 500 + (fold + 1) * 50)
    return set(first) | set(second)


def inspect_family(model: str, directory: Path) -> tuple[dict[int, dict], dict]:
    paths = checkpoint_map(directory)
    splits = {}
    manifest = {}
    expected_all = set(range(1_000))
    for fold, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        experiment = checkpoint.get("experiment", {})
        result = checkpoint.get("result", {})
        if experiment.get("model_type") != model or result.get("model_type") != model:
            raise ValueError(f"{path}: expected model_type={model!r}")
        if int(experiment.get("seed", -1)) != fold or int(result.get("seed", -1)) != fold:
            raise ValueError(f"{path}: seed metadata does not match fold {fold}")
        if int(result.get("n_rollouts", -1)) != 1_000:
            raise ValueError(f"{path}: expected n_rollouts=1000")
        if tuple(checkpoint.get("selected_layers", ())) != (32,):
            raise ValueError(f"{path}: expected selected layer 32")
        if checkpoint.get("token_pool") != "last":
            raise ValueError(f"{path}: expected last-token pooling")
        raw_split = checkpoint.get("split")
        if not isinstance(raw_split, dict) or set(raw_split) != {"train", "val", "test"}:
            raise ValueError(f"{path}: split must contain train, val, and test")
        split = {name: [int(index) for index in values] for name, values in raw_split.items()}
        expected_sizes = {"train": 540, "val": 360, "test": 100}
        for name, expected_size in expected_sizes.items():
            values = split[name]
            if len(values) != expected_size or len(set(values)) != expected_size:
                raise ValueError(
                    f"{path}: {name} must contain {expected_size} unique indices"
                )
        train, validation, test = (set(split[name]) for name in ("train", "val", "test"))
        if train & validation or train & test or validation & test:
            raise ValueError(f"{path}: train/validation/test indices overlap")
        if train | validation | test != expected_all:
            raise ValueError(f"{path}: split does not partition indices 0..999")
        if test != expected_test_indices(fold):
            raise ValueError(
                f"{path}: test indices do not match task/fold {fold} under the "
                "ordered two-shard contract"
            )
        splits[fold] = split
        manifest[str(fold)] = {"file": path.name, "sha256": sha256_file(path)}
    return splits, manifest


def audit_checkpoint_dirs(mlp_dir: Path, lstm_dir: Path) -> dict[str, Any]:
    mlp_splits, mlp_manifest = inspect_family("mlp", mlp_dir)
    lstm_splits, lstm_manifest = inspect_family("lstm", lstm_dir)
    if mlp_splits != lstm_splits:
        raise ValueError("MLP and LSTM checkpoint split mappings differ")
    normalized_splits = {
        str(fold): {
            role: sorted(indices) for role, indices in sorted(split.items())
        }
        for fold, split in sorted(mlp_splits.items())
    }
    split_sha256 = hashlib.sha256(
        json.dumps(normalized_splits, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "pass",
        "n_rollouts": 1_000,
        "folds": list(range(10)),
        "split_sizes": {"train": 540, "val": 360, "test": 100},
        "ordered_root_rollouts": [500, 500],
        "split_sha256": split_sha256,
        "checkpoint_manifest": {
            "mlp": mlp_manifest,
            "lstm": lstm_manifest,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlp-checkpoint-dir",
        type=Path,
        default=Path("runs/openvla_mlp_loto_l32"),
    )
    parser.add_argument(
        "--lstm-checkpoint-dir",
        type=Path,
        default=Path("runs/openvla_lstm_loto_l32"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_checkpoint_dirs(args.mlp_checkpoint_dir, args.lstm_checkpoint_dir)
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
