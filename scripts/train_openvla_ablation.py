"""Train SAFE-style OpenVLA failure-prediction ablations.

Default data root:

    data/rollouts/openvla

The script uses the rollout dataloader in `data/dataloaders.py`, selects the
last action token by default, and runs the ablation set discussed in ROADMAP.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import (
    LayerMixLSTMModel,
    LayerTokenTransformerLSTMModel,
    LinearProbeModel,
    SafeLSTMModel,
    SafeMLPModel,
    SparseLayerMixLSTMModel,
)


DEFAULT_ROOT = Path("data/rollouts/openvla")
SAFE_OPENVLA_LIBERO_UNSEEN_ROC_AUC = {
    "lstm": 0.7247,
    "mlp": 0.7347,
}
# SAFE reports no openvla-mini number (different backbone, 896-d hidden, and
# these rollouts are libero_90, not libero_10). We reuse the published SAFE
# OpenVLA-LIBERO unseen ROC-AUCs as a rough cross-model yardstick so the delta
# columns stay populated; treat the delta as indicative only, not an
# apples-to-apples comparison.
SAFE_OPENVLA_MINI_LIBERO_UNSEEN_ROC_AUC = {
    "lstm": 0.7247,
    "mlp": 0.7347,
}

SAFE_OPTIMAL_LR = 1e-4
SAFE_OPTIMAL_LAMBDA_REG_BY_MODEL = {
    "mlp": 1e-2,
    "lstm": 1.0,
    # LSTM-like layer-fusion models use the LSTM prior unless explicitly swept.
    "layer_mix": 1.0,
    "sparse_layer_mix": 1.0,
    "layer_token": 1.0,
    # Keep the historical linear-probe default at reg=1 for checkpoint-name compatibility.
    "linear_probe": 1.0,
}


def default_lambda_reg_for_model(model_type: str) -> float:
    return SAFE_OPTIMAL_LAMBDA_REG_BY_MODEL.get(model_type, 1.0)


def resolve_lambda_reg(model_type: str, exp_value: float | None, cli_value: float | None) -> float:
    if exp_value is not None:
        return exp_value
    if cli_value is not None:
        return cli_value
    return default_lambda_reg_for_model(model_type)


def safe_paper_baseline(root: Path | str, model_type: str) -> float:
    """Pick the SAFE reference ROC-AUC for the dataset under ``root``."""

    table = (
        SAFE_OPENVLA_MINI_LIBERO_UNSEEN_ROC_AUC
        if "openvla-mini" in str(root)
        else SAFE_OPENVLA_LIBERO_UNSEEN_ROC_AUC
    )
    return table.get(model_type, float("nan"))


@dataclass(frozen=True)
class Experiment:
    name: str
    model_type: str
    layers: tuple[int, ...]
    token_pool: str = "last"
    lr: float | None = None
    lambda_reg: float | None = None
    seed: int | None = None
    n_history_steps: int = 1


def parse_layers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def normalize_token_pool(value: str) -> str:
    """Map SAFE token_idx_rel values to local action-token pooling names."""

    normalized = value.strip().lower()
    aliases = {
        "0": "first",
        "0.0": "first",
        "first": "first",
        "1": "last",
        "1.0": "last",
        "last": "last",
        "mean": "mean",
        "first+last": "first_last",
        "first-last": "first_last",
        "first_last": "first_last",
        "none": "none",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported token pool {value!r}; use mean, 0.0/first, 1.0/last, first_last, or none"
        )
    return aliases[normalized]


def token_pool_tag(value: str) -> str:
    return normalize_token_pool(value).replace("_", "-")


def discover_saved_layers(root: Path) -> tuple[int, ...]:
    dataset = OpenVLARolloutDataset(
        root, layers=None, token_pool="last", recursive=True
    )
    return tuple(dataset.saved_layers)


def build_default_experiments(saved_layers: tuple[int, ...]) -> list[Experiment]:
    final_layer = (saved_layers[-1],)
    late_layers = tuple(layer for layer in (24, 28, 32) if layer in saved_layers)
    if len(late_layers) < 2:
        late_layers = tuple(saved_layers[-min(3, len(saved_layers)) :])

    experiments = [
        Experiment("linear_probe_last_layer", "linear_probe", final_layer),
        Experiment("safe_mlp_last_layer", "mlp", final_layer),
        Experiment("safe_lstm_last_layer", "lstm", final_layer),
        Experiment("linear_probe_late_concat", "linear_probe", late_layers),
        Experiment("safe_mlp_late_concat", "mlp", late_layers),
        Experiment("safe_lstm_late_concat", "lstm", late_layers),
        Experiment("linear_probe_all_captured", "linear_probe", saved_layers),
        Experiment("layer_mix_late", "layer_mix", late_layers),
        Experiment("layer_mix_all_captured", "layer_mix", saved_layers),
        Experiment("sparse_layer_mix_late", "sparse_layer_mix", late_layers),
        Experiment("sparse_layer_mix_all_captured", "sparse_layer_mix", saved_layers),
        Experiment("layer_token_encoder_late", "layer_token", late_layers),
        Experiment("layer_token_encoder_all_captured", "layer_token", saved_layers),
    ]
    experiments.extend(
        Experiment(f"linear_probe_layer_{layer}", "linear_probe", (layer,))
        for layer in saved_layers
    )
    experiments.extend(
        Experiment(f"safe_lstm_layer_{layer}", "lstm", (layer,))
        for layer in saved_layers
    )
    return experiments


def filter_experiments(
    experiments: Iterable[Experiment],
    include: set[str] | None,
    skip_single_layer_sweep: bool,
) -> list[Experiment]:
    out = []
    for exp in experiments:
        if include is not None and exp.name not in include:
            continue
        if skip_single_layer_sweep and (
            exp.name.startswith("safe_lstm_layer_")
            or exp.name.startswith("linear_probe_layer_")
        ):
            continue
        out.append(exp)
    return out


def build_safe_openvla_libero_experiments(
    saved_layers: tuple[int, ...],
    seeds: tuple[int, ...],
    token_pools: tuple[str, ...],
    lrs: tuple[float, ...],
    lambda_regs: tuple[float, ...],
    models: tuple[str, ...],
    history_steps: tuple[int, ...],
    layers: tuple[int, ...] | None = None,
) -> list[Experiment]:
    """Grid from SAFE's OpenVLA LIBERO batch-training script.

    By default each experiment uses only the final saved layer (matching SAFE).
    Pass ``layers`` to override (e.g. the last three captured layers).
    """

    final_layer = tuple(layers) if layers else (saved_layers[-1],)
    experiments: list[Experiment] = []
    for seed, token_pool_raw, lr, lambda_reg, model in product(
        seeds, token_pools, lrs, lambda_regs, models
    ):
        token_pool = normalize_token_pool(token_pool_raw)
        model_name = model.strip().lower()
        if model_name in {"indep", "mlp", "safe_mlp"}:
            model_type = "mlp"
            model_tag = "safe_mlp"
        elif model_name in {"lstm", "safe_lstm"}:
            model_type = "lstm"
            model_tag = "safe_lstm"
        elif model_name in {"linear_probe", "probe", "linear"}:
            model_type = "linear_probe"
            model_tag = "linear_probe"
        elif model_name in {"layer_mix", "layermix", "mix"}:
            model_type = "layer_mix"
            model_tag = "layer_mix"
        elif model_name in {"sparse_layer_mix", "sparse_layermix", "sparse_mix"}:
            model_type = "sparse_layer_mix"
            model_tag = "sparse_layer_mix"
        elif model_name in {"layer_token", "layer_tokens", "layer_token_encoder", "token_as_layer"}:
            model_type = "layer_token"
            model_tag = "layer_token"
        else:
            raise ValueError(
                f"Unsupported SAFE sweep model {model!r}; use "
                "lstm, mlp/indep, linear_probe, layer_mix, sparse_layer_mix, or layer_token"
            )

        # Layer-relationship models operate over multiple captured layers. With
        # no explicit override they default to ALL captured layers. An explicit
        # `layers` override is honored as-is.
        if model_type in {"layer_mix", "sparse_layer_mix", "layer_token"}:
            exp_layers = tuple(layers) if layers else tuple(saved_layers)
        else:
            exp_layers = final_layer

        model_history_steps = history_steps if model_type == "mlp" else (1,)
        for n_history_steps in model_history_steps:
            history_tag = f"_hist-{n_history_steps}" if n_history_steps != 1 else ""
            name = (
                f"{model_tag}_tok-{token_pool_tag(token_pool)}"
                f"_lr-{lr:g}_reg-{lambda_reg:g}{history_tag}_seed-{seed}"
            )
            experiments.append(
                Experiment(
                    name=name,
                    model_type=model_type,
                    layers=exp_layers,
                    token_pool=token_pool,
                    lr=lr,
                    lambda_reg=lambda_reg,
                    seed=seed,
                    n_history_steps=n_history_steps,
                )
            )
    return experiments


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(
    labels: list[int],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    indices = np.arange(len(labels))
    labels_array = np.asarray(labels)

    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    try:
        train_idx, tmp_idx, y_train, y_tmp = train_test_split(
            indices,
            labels_array,
            train_size=train_frac,
            random_state=seed,
            stratify=labels_array,
        )
        relative_val = val_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(
            tmp_idx,
            train_size=relative_val,
            random_state=seed,
            stratify=y_tmp,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        n_train = int(round(len(shuffled) * train_frac))
        n_val = int(round(len(shuffled) * val_frac))
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train : n_train + n_val]
        test_idx = shuffled[n_train + n_val :]

    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def task_ids_for_dataset(dataset: OpenVLARolloutDataset) -> list[int]:
    task_ids: list[int] = []
    for info in dataset.rollouts:
        artifact = dataset._load_pickle(info.path)
        task_id = dataset._read_task_id(artifact, info)
        if task_id is None:
            raise KeyError(f"Could not infer task_id for {info.path}")
        task_ids.append(task_id)
    return task_ids


def split_indices_by_task(
    dataset: OpenVLARolloutDataset,
    unseen_task_ratio: float,
    seen_train_ratio: float,
    seed: int,
    held_out_tasks: list[int] | None = None,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """SAFE repo split: shuffle task IDs, hold out unseen tasks, split seen rollouts.

    When ``held_out_tasks`` is given, those exact task IDs become the unseen
    (test) set and ``unseen_task_ratio`` is ignored for selection -- this is the
    leave-one-task-out (LOTO) mode. ``seed`` still controls the train/val shuffle
    of the seen rollouts, so a fixed seed across folds isolates the held-out-task
    effect.
    """

    if not 0.0 <= unseen_task_ratio < 1.0:
        raise ValueError("unseen_task_ratio must be in [0, 1)")
    if not 0.0 < seen_train_ratio < 1.0:
        raise ValueError("seen_train_ratio must be in (0, 1)")

    task_ids = task_ids_for_dataset(dataset)
    task_id_order = list(set(task_ids))
    unique_task_ids = sorted(task_id_order)
    if len(unique_task_ids) < 2:
        raise ValueError("Task-level split requires at least two task IDs")

    rng = np.random.RandomState(seed)
    shuffled_tasks = list(task_id_order)
    rng.shuffle(shuffled_tasks)

    if held_out_tasks is not None:
        held = [int(t) for t in held_out_tasks]
        missing = [t for t in held if t not in unique_task_ids]
        if missing:
            raise ValueError(f"held_out_tasks {missing} not in dataset tasks {unique_task_ids}")
        if not 0 < len(held) < len(unique_task_ids):
            raise ValueError("held_out_tasks must hold out between 1 and n-1 tasks")
        unseen_set = set(held)
        seen_task_ids = [int(t) for t in shuffled_tasks if int(t) not in unseen_set]
        unseen_task_ids = [int(t) for t in shuffled_tasks if int(t) in unseen_set]
    else:
        n_unseen = round(unseen_task_ratio * len(unique_task_ids))
        if unseen_task_ratio > 0 and n_unseen == 0:
            n_unseen = 1
        if n_unseen >= len(unique_task_ids):
            n_unseen = len(unique_task_ids) - 1
        n_seen = len(unique_task_ids) - n_unseen
        seen_task_ids = [int(task_id) for task_id in shuffled_tasks[:n_seen]]
        unseen_task_ids = [int(task_id) for task_id in shuffled_tasks[n_seen:]]
        unseen_set = set(unseen_task_ids)
    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    per_task_counts: dict[str, dict[str, int]] = {}

    for task_id in shuffled_tasks:
        indices = [
            idx
            for idx, rollout_task_id in enumerate(task_ids)
            if rollout_task_id == task_id
        ]
        if task_id in unseen_set:
            test_indices.extend(indices)
            per_task_counts[str(task_id)] = {"train": 0, "val": 0, "test": len(indices)}
            continue

        permuted_positions = torch.randperm(len(indices), generator=torch_generator)
        indices = [int(indices[int(position)]) for position in permuted_positions]
        n_train = int(seen_train_ratio * len(indices))
        if len(indices) > 1:
            n_train = min(max(n_train, 1), len(indices) - 1)
        train_part = indices[:n_train]
        val_part = indices[n_train:]
        train_indices.extend(train_part)
        val_indices.extend(val_part)
        per_task_counts[str(task_id)] = {
            "train": len(train_part),
            "val": len(val_part),
            "test": 0,
        }

    split = {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }
    metadata = {
        "split_mode": "task",
        "task_split_algorithm": "safe_repo_np_shuffle_seen_first",
        "unseen_task_ratio": unseen_task_ratio,
        "seen_train_ratio": seen_train_ratio,
        "shuffled_task_ids": [int(task_id) for task_id in shuffled_tasks],
        "seen_task_ids": seen_task_ids,
        "unseen_task_ids": unseen_task_ids,
        "seen_task_ids_sorted": sorted(seen_task_ids),
        "unseen_task_ids_sorted": sorted(unseen_task_ids),
        "per_task_counts": per_task_counts,
    }
    return split, metadata


def labels_for_dataset(dataset: OpenVLARolloutDataset) -> list[int]:
    labels: list[int] = []
    for info in dataset.rollouts:
        artifact = dataset._load_pickle(info.path)
        success = dataset._read_success(artifact, info)
        labels.append(int(not success))
    return labels


# Memoize datasets by (root, layers, token_pool, cache) so a single-process
# grid (many experiments sharing the same feature view) only reads/unpickles the
# rollouts once. The populated in-memory cache is then reused across every
# experiment with that layer/token combo -- the difference between 1 disk pass
# and one-per-cell. Splits vary per seed but only via the Subset wrapper, so the
# underlying dataset object is safe to share.
_DATASET_CACHE: dict[tuple, OpenVLARolloutDataset] = {}


def get_dataset(
    root: Path,
    layers: tuple[int, ...],
    token_pool: str,
    cache: bool = True,
) -> OpenVLARolloutDataset:
    key = (str(root), tuple(layers), token_pool, cache)
    dataset = _DATASET_CACHE.get(key)
    if dataset is None:
        dataset = OpenVLARolloutDataset(
            root, layers=layers, token_pool=token_pool, recursive=True, cache=cache
        )
        _DATASET_CACHE[key] = dataset
    return dataset


def make_loaders(
    root: Path,
    layers: tuple[int, ...],
    token_pool: str,
    split: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
    cache: bool = True,
) -> tuple[OpenVLARolloutDataset, dict[str, DataLoader]]:
    dataset = get_dataset(root, layers, token_pool, cache)
    loaders = {
        name: DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            collate_fn=collate_rollouts,
        )
        for name, indices in split.items()
        if indices
    }
    return dataset, loaders


def class_weights_for_split(
    dataset: OpenVLARolloutDataset, indices: list[int]
) -> tuple[float, float]:
    """Return SAFE-style `(failure_weight, success_weight)` for a training split."""

    if not indices:
        return (1.0, 1.0)

    n_success = 0
    for idx in indices:
        info = dataset.rollouts[idx]
        artifact = dataset._load_pickle(info.path)
        n_success += int(dataset._read_success(artifact, info))
    n_total = len(indices)
    n_failure = n_total - n_success

    freq_failure = (n_failure + 1) / n_total
    freq_success = (n_success + 1) / n_total
    return (1.0 / freq_failure, 1.0 / freq_success)


def regularization_loss(model: torch.nn.Module, lambda_reg: float) -> torch.Tensor:
    """SAFE-style L2 regularization over non-bias parameters."""

    if lambda_reg <= 0:
        return next(model.parameters()).new_tensor(0.0)
    reg = next(model.parameters()).new_tensor(0.0)
    for name, param in model.named_parameters():
        if "bias" not in name:
            reg = reg + torch.sum(param**2)
    return lambda_reg * reg


def build_model(
    exp: Experiment,
    dataset: OpenVLARolloutDataset,
    hidden_dim: int,
    projection_dim: int,
    dropout: float,
    device: torch.device,
) -> torch.nn.Module:
    if exp.model_type == "linear_probe":
        model = LinearProbeModel(input_dim=dataset.input_dim)
    elif exp.model_type == "mlp":
        model = SafeMLPModel(
            input_dim=dataset.input_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=dropout,
            cumsum=True,
            n_history_steps=exp.n_history_steps,
            loss_type="safe",
            use_threshold=False,
        )
    elif exp.model_type == "lstm":
        model = SafeLSTMModel(
            input_dim=dataset.input_dim,
            hidden_dim=hidden_dim,
            n_layers=1,
            dropout=dropout,
            loss_type="bce",
            use_threshold=False,
        )
    elif exp.model_type == "layer_mix":
        model = LayerMixLSTMModel(
            input_dim=dataset.input_dim,
            n_layers=len(dataset.selected_layers),
            hidden_dim_per_layer=dataset.hidden_dim,
            projection_dim=projection_dim,
            lstm_hidden_dim=hidden_dim,
            dropout=dropout,
            loss_type="bce",
        )
    elif exp.model_type == "sparse_layer_mix":
        model = SparseLayerMixLSTMModel(
            input_dim=dataset.input_dim,
            n_layers=len(dataset.selected_layers),
            hidden_dim_per_layer=dataset.hidden_dim,
            projection_dim=projection_dim,
            lstm_hidden_dim=hidden_dim,
            dropout=dropout,
            loss_type="bce",
        )
    elif exp.model_type == "layer_token":
        model = LayerTokenTransformerLSTMModel(
            input_dim=dataset.input_dim,
            n_layers=len(dataset.selected_layers),
            hidden_dim_per_layer=dataset.hidden_dim,
            projection_dim=projection_dim,
            lstm_hidden_dim=hidden_dim,
            dropout=dropout,
            loss_type="bce",
        )
    else:
        raise ValueError(f"Unknown model_type={exp.model_type!r}")
    return model.to(device)


def sanitize_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [sanitize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_config_value(item) for key, item in value.items()}
    return value


def init_wandb_run(
    args: argparse.Namespace,
    exp: Experiment,
    dataset: OpenVLARolloutDataset,
    split: dict[str, list[int]],
):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested with --wandb, but wandb is not installed. "
            "Install it or run through uv with `--with wandb`."
        ) from exc

    base_config = {
        key: sanitize_config_value(value) for key, value in vars(args).items()
    }
    base_config.update(
        {
            "experiment": asdict(exp),
            "model_type": exp.model_type,
            "selected_layers": list(dataset.selected_layers),
            "saved_layers": list(dataset.saved_layers),
            "token_pool": dataset.token_pool,
            "input_dim": dataset.input_dim,
            "hidden_dim_per_layer": dataset.hidden_dim,
            "n_rollouts": len(dataset),
            "split_sizes": {name: len(indices) for name, indices in split.items()},
        }
    )

    if args.wandb_run_name:
        run_name = f"{args.wandb_run_name}-{exp.name}"
    else:
        run_name = f"{args.wandb_run_prefix}{exp.name}"

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        job_type=exp.model_type,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=base_config,
        reinit="finish_previous",
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def masked_rollout_scores(
    scores: torch.Tensor,
    valid_masks: torch.Tensor,
    stop_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = scores.squeeze(-1)
    mask = valid_masks.bool()
    if stop_lengths is not None:
        stop_lengths = stop_lengths.to(scores.device).clamp_min(1)
        timesteps = torch.arange(scores.shape[1], device=scores.device)
        stop_mask = timesteps.unsqueeze(0) < stop_lengths.unsqueeze(1)
        mask = mask & stop_mask
    masked = scores.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


def ranking_metrics(labels: list[int], rollout_scores: list[float]) -> dict[str, float]:
    if len(set(labels)) <= 1:
        return {"roc_auc": float("nan"), "tpr_at_5_fpr": float("nan")}

    roc_auc = float(roc_auc_score(labels, rollout_scores))
    fpr, tpr, _ = roc_curve(labels, rollout_scores)
    valid = np.where(fpr <= 0.05)[0]
    tpr_at_5_fpr = float(tpr[valid].max()) if len(valid) else 0.0
    return {"roc_auc": roc_auc, "tpr_at_5_fpr": tpr_at_5_fpr}


def metadata_baselines_for_split(
    dataset: OpenVLARolloutDataset, indices: list[int]
) -> dict[str, float | int | bool]:
    labels: list[int] = []
    lengths: list[float] = []
    progress_ratios: list[float] = []
    task_min_steps: list[float] = []

    for idx in indices:
        info = dataset.rollouts[idx]
        artifact = dataset._load_pickle(info.path)
        success = dataset._read_success(artifact, info)
        task_id = dataset._read_task_id(artifact, info)
        length = int(artifact["hidden_states"].shape[0])
        task_min_step = dataset._task_min_step(task_id, length)

        labels.append(int(not success))
        lengths.append(float(length))
        progress_ratios.append(float(length / max(task_min_step, 1)))
        task_min_steps.append(float(task_min_step))

    def auc(values: list[float]) -> float:
        if len(set(labels)) <= 1:
            return float("nan")
        return float(roc_auc_score(labels, values))

    length_auc = auc(lengths)
    return {
        "n_rollouts": len(indices),
        "n_failed": int(sum(labels)),
        "n_success": int(len(labels) - sum(labels)),
        "length_only_roc_auc": length_auc,
        "progress_ratio_roc_auc": auc(progress_ratios),
        "task_min_step_roc_auc": auc(task_min_steps),
        "length_leakage_flag": bool(np.isfinite(length_auc) and length_auc >= 0.99),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    losses = []
    labels = []
    early_rollout_scores = []
    end_rollout_scores = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss, _ = model.forward_loss(batch)
        scores = model(batch)
        losses.append(float(loss.detach().cpu()))
        labels.extend(batch["labels"].detach().cpu().int().tolist())
        early_rollout_scores.extend(
            masked_rollout_scores(
                scores, batch["valid_masks"], batch.get("task_min_steps")
            )
            .detach()
            .cpu()
            .tolist()
        )
        end_rollout_scores.extend(
            masked_rollout_scores(scores, batch["valid_masks"]).detach().cpu().tolist()
        )

    metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
    early_metrics = ranking_metrics(labels, early_rollout_scores)
    end_metrics = ranking_metrics(labels, end_rollout_scores)
    metrics["falert_early_roc_auc"] = early_metrics["roc_auc"]
    metrics["falert_early_tpr_at_5_fpr"] = early_metrics["tpr_at_5_fpr"]
    metrics["falert_end_roc_auc"] = end_metrics["roc_auc"]
    metrics["falert_end_tpr_at_5_fpr"] = end_metrics["tpr_at_5_fpr"]
    # Backward-compatible primary metrics now match SAFE's earliest-stop setting.
    metrics["roc_auc"] = early_metrics["roc_auc"]
    metrics["tpr_at_5_fpr"] = early_metrics["tpr_at_5_fpr"]
    return metrics


def train_one_experiment(
    exp: Experiment,
    root: Path,
    split: dict[str, list[int]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    dataset, loaders = make_loaders(
        root=root,
        layers=exp.layers,
        token_pool=exp.token_pool,
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache=not args.no_cache,
    )
    metadata_baselines = {
        name: metadata_baselines_for_split(dataset, indices)
        for name, indices in split.items()
        if indices
    }
    model = build_model(
        exp=exp,
        dataset=dataset,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        device=device,
    )
    lr = exp.lr if exp.lr is not None else args.lr
    lambda_reg = resolve_lambda_reg(exp.model_type, exp.lambda_reg, args.lambda_reg)
    class_weights = class_weights_for_split(dataset, split["train"])
    class_weights = (
        class_weights[0] * args.lambda_fail,
        class_weights[1] * args.lambda_success,
    )

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer={args.optimizer!r}")
    wandb_run = init_wandb_run(args, exp, dataset, split)

    progress_path = args.output_dir / f"{exp.name}.progress.pt"
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience
    start_epoch = 1

    if args.resume_progress and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        completed_epoch = int(progress.get("epoch", 0))
        if completed_epoch >= args.epochs:
            print(
                f"ignoring stale complete progress checkpoint for {exp.name}: "
                f"epoch={completed_epoch}"
            )
        else:
            model.load_state_dict(progress["model_state_dict"])
            optimizer.load_state_dict(progress["optimizer_state_dict"])
            best_state = progress.get("best_state_dict")
            best_val = float(progress.get("best_val", best_val))
            best_epoch = int(progress.get("best_epoch", best_epoch))
            patience_left = int(progress.get("patience_left", patience_left))
            start_epoch = completed_epoch + 1
            print(
                f"resumed {exp.name} from {progress_path} "
                f"at epoch={completed_epoch}; continuing at epoch={start_epoch}"
            )

    accumulation_steps = max(1, int(args.gradient_accumulation_steps))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        total_train_batches = len(loaders["train"])
        for batch_idx, batch in enumerate(loaders["train"], start=1):
            batch = move_batch(batch, device)
            loss, _ = model.forward_loss(batch, weights=class_weights)
            reg_loss = regularization_loss(model, lambda_reg)
            total_loss = loss + reg_loss
            group_start = ((batch_idx - 1) // accumulation_steps) * accumulation_steps + 1
            group_end = min(group_start + accumulation_steps - 1, total_train_batches)
            group_size = group_end - group_start + 1
            (total_loss / group_size).backward()
            if batch_idx % accumulation_steps == 0 or batch_idx == total_train_batches:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(total_loss.detach().cpu()))

        val_metrics = (
            evaluate(model, loaders["val"], device)
            if "val" in loaders
            else {"loss": float("nan")}
        )
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = val_metrics["loss"]
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"{exp.name} epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_falert_early_auc={val_metrics.get('roc_auc', float('nan')):.4f} "
                f"val_falert_end_auc={val_metrics.get('falert_end_roc_auc', float('nan')):.4f}"
            )

        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/roc_auc": val_metrics.get("roc_auc", float("nan")),
                "val/tpr_at_5_fpr": val_metrics.get("tpr_at_5_fpr", float("nan")),
                "val/falert_early_roc_auc": val_metrics.get(
                    "falert_early_roc_auc", float("nan")
                ),
                "val/falert_early_tpr_at_5_fpr": val_metrics.get(
                    "falert_early_tpr_at_5_fpr", float("nan")
                ),
                "val/falert_end_roc_auc": val_metrics.get(
                    "falert_end_roc_auc", float("nan")
                ),
                "val/falert_end_tpr_at_5_fpr": val_metrics.get(
                    "falert_end_tpr_at_5_fpr", float("nan")
                ),
                "model/lr": lr,
                "model/lambda_reg": lambda_reg,
                "train/class_weight_failure": class_weights[0],
                "train/class_weight_success": class_weights[1],
            }
            if hasattr(model, "layer_weights"):
                for layer, weight in zip(
                    dataset.selected_layers, model.layer_weights.cpu().tolist()
                ):
                    log_payload[f"layer_weights/layer_{layer}"] = weight
            wandb_run.log(log_payload, step=epoch)

        if (
            args.progress_save_every > 0
            and (epoch % args.progress_save_every == 0 or epoch == args.epochs)
        ):
            tmp_progress_path = progress_path.with_suffix(
                progress_path.suffix + ".tmp"
            )
            torch.save(
                {
                    "experiment": asdict(exp),
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_state_dict": best_state,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "patience_left": patience_left,
                    "selected_layers": dataset.selected_layers,
                    "saved_layers": dataset.saved_layers,
                    "token_pool": dataset.token_pool,
                    "lr": lr,
                    "lambda_reg": lambda_reg,
                    "n_history_steps": exp.n_history_steps,
                },
                tmp_progress_path,
            )
            tmp_progress_path.replace(progress_path)

        if not args.no_early_stopping and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = (
        evaluate(model, loaders["test"], device)
        if "test" in loaders
        else {"loss": float("nan")}
    )
    val_metrics = (
        evaluate(model, loaders["val"], device)
        if "val" in loaders
        else {"loss": float("nan")}
    )
    safe_paper_roc_auc = safe_paper_baseline(dataset.root, exp.model_type)
    test_early_roc_auc = test_metrics.get("falert_early_roc_auc", float("nan"))
    test_end_roc_auc = test_metrics.get("falert_end_roc_auc", float("nan"))
    test_early_delta = (
        test_early_roc_auc - safe_paper_roc_auc
        if np.isfinite(test_early_roc_auc) and np.isfinite(safe_paper_roc_auc)
        else float("nan")
    )
    test_end_delta = (
        test_end_roc_auc - safe_paper_roc_auc
        if np.isfinite(test_end_roc_auc) and np.isfinite(safe_paper_roc_auc)
        else float("nan")
    )

    result = {
        "name": exp.name,
        "model_type": exp.model_type,
        "layers": list(exp.layers),
        "token_pool": dataset.token_pool,
        "input_dim": dataset.input_dim,
        "hidden_dim_per_layer": dataset.hidden_dim,
        "n_rollouts": len(dataset),
        "lr": lr,
        "lambda_reg": lambda_reg,
        "seed": exp.seed if exp.seed is not None else args.seed,
        "n_history_steps": exp.n_history_steps,
        "best_epoch": best_epoch,
        "val_loss": val_metrics["loss"],
        "val_roc_auc": val_metrics.get("roc_auc", float("nan")),
        "val_tpr_at_5_fpr": val_metrics.get("tpr_at_5_fpr", float("nan")),
        "val_safe_early_stop_roc_auc": val_metrics.get(
            "falert_early_roc_auc", float("nan")
        ),
        "val_full_rollout_roc_auc": val_metrics.get("falert_end_roc_auc", float("nan")),
        "val_falert_early_roc_auc": val_metrics.get(
            "falert_early_roc_auc", float("nan")
        ),
        "val_falert_early_tpr_at_5_fpr": val_metrics.get(
            "falert_early_tpr_at_5_fpr", float("nan")
        ),
        "val_falert_end_roc_auc": val_metrics.get("falert_end_roc_auc", float("nan")),
        "val_falert_end_tpr_at_5_fpr": val_metrics.get(
            "falert_end_tpr_at_5_fpr", float("nan")
        ),
        "test_loss": test_metrics["loss"],
        "test_roc_auc": test_metrics.get("roc_auc", float("nan")),
        "test_tpr_at_5_fpr": test_metrics.get("tpr_at_5_fpr", float("nan")),
        "test_safe_early_stop_roc_auc": test_early_roc_auc,
        "test_full_rollout_roc_auc": test_end_roc_auc,
        "test_falert_early_roc_auc": test_metrics.get(
            "falert_early_roc_auc", float("nan")
        ),
        "test_falert_early_tpr_at_5_fpr": test_metrics.get(
            "falert_early_tpr_at_5_fpr", float("nan")
        ),
        "test_falert_end_roc_auc": test_metrics.get("falert_end_roc_auc", float("nan")),
        "test_falert_end_tpr_at_5_fpr": test_metrics.get(
            "falert_end_tpr_at_5_fpr", float("nan")
        ),
        "val_length_only_roc_auc": metadata_baselines.get("val", {}).get(
            "length_only_roc_auc", float("nan")
        ),
        "val_progress_ratio_roc_auc": metadata_baselines.get("val", {}).get(
            "progress_ratio_roc_auc", float("nan")
        ),
        "val_task_min_step_roc_auc": metadata_baselines.get("val", {}).get(
            "task_min_step_roc_auc", float("nan")
        ),
        "val_length_leakage_flag": metadata_baselines.get("val", {}).get(
            "length_leakage_flag", False
        ),
        "test_length_only_roc_auc": metadata_baselines.get("test", {}).get(
            "length_only_roc_auc", float("nan")
        ),
        "test_progress_ratio_roc_auc": metadata_baselines.get("test", {}).get(
            "progress_ratio_roc_auc", float("nan")
        ),
        "test_task_min_step_roc_auc": metadata_baselines.get("test", {}).get(
            "task_min_step_roc_auc", float("nan")
        ),
        "test_length_leakage_flag": metadata_baselines.get("test", {}).get(
            "length_leakage_flag", False
        ),
        "safe_paper_openvla_libero_unseen_roc_auc": safe_paper_roc_auc,
        "test_safe_early_stop_delta_vs_safe_paper": test_early_delta,
        "test_full_rollout_delta_vs_safe_paper": test_end_delta,
        "metadata_baselines": metadata_baselines,
    }
    if hasattr(model, "layer_weights"):
        result["layer_weights"] = {
            str(layer): weight
            for layer, weight in zip(
                dataset.selected_layers, model.layer_weights.cpu().tolist()
            )
        }

    checkpoint_path = args.output_dir / f"{exp.name}.pt"
    torch.save(
        {
            "experiment": asdict(exp),
            "result": result,
            "model_state_dict": model.state_dict(),
            # The exact rollout indices this model was trained/evaluated with.
            # Downstream consumers (conformal sweeps, score analyses) must use
            # this rather than split_indices_*.json files, which can go stale
            # when a sweep directory mixes checkpoints from several invocations.
            "split": {name: [int(i) for i in indices] for name, indices in split.items()},
            "selected_layers": dataset.selected_layers,
            "saved_layers": dataset.saved_layers,
            "token_pool": dataset.token_pool,
            "lr": lr,
            "lambda_reg": lambda_reg,
            "n_history_steps": exp.n_history_steps,
        },
        checkpoint_path,
    )
    result["checkpoint_path"] = str(checkpoint_path)
    progress_path.unlink(missing_ok=True)

    if wandb_run is not None:
        wandb_run.log(
            {
                "best_epoch": best_epoch,
                "final/val_loss": result["val_loss"],
                "final/val_roc_auc": result["val_roc_auc"],
                "final/val_tpr_at_5_fpr": result["val_tpr_at_5_fpr"],
                "final/val_safe_early_stop_roc_auc": result[
                    "val_safe_early_stop_roc_auc"
                ],
                "final/val_full_rollout_roc_auc": result["val_full_rollout_roc_auc"],
                "final/val_falert_early_roc_auc": result["val_falert_early_roc_auc"],
                "final/val_falert_early_tpr_at_5_fpr": result[
                    "val_falert_early_tpr_at_5_fpr"
                ],
                "final/val_falert_end_roc_auc": result["val_falert_end_roc_auc"],
                "final/val_falert_end_tpr_at_5_fpr": result[
                    "val_falert_end_tpr_at_5_fpr"
                ],
                "test/loss": result["test_loss"],
                "test/roc_auc": result["test_roc_auc"],
                "test/tpr_at_5_fpr": result["test_tpr_at_5_fpr"],
                "test/safe_early_stop_roc_auc": result["test_safe_early_stop_roc_auc"],
                "test/full_rollout_roc_auc": result["test_full_rollout_roc_auc"],
                "test/falert_early_roc_auc": result["test_falert_early_roc_auc"],
                "test/falert_early_tpr_at_5_fpr": result[
                    "test_falert_early_tpr_at_5_fpr"
                ],
                "test/falert_end_roc_auc": result["test_falert_end_roc_auc"],
                "test/falert_end_tpr_at_5_fpr": result["test_falert_end_tpr_at_5_fpr"],
                "baseline/val_length_only_roc_auc": result["val_length_only_roc_auc"],
                "baseline/val_progress_ratio_roc_auc": result[
                    "val_progress_ratio_roc_auc"
                ],
                "baseline/test_length_only_roc_auc": result["test_length_only_roc_auc"],
                "baseline/test_progress_ratio_roc_auc": result[
                    "test_progress_ratio_roc_auc"
                ],
                "safe_paper/openvla_libero_unseen_roc_auc": result[
                    "safe_paper_openvla_libero_unseen_roc_auc"
                ],
                "safe_paper/test_safe_early_stop_delta": result[
                    "test_safe_early_stop_delta_vs_safe_paper"
                ],
                "safe_paper/test_full_rollout_delta": result[
                    "test_full_rollout_delta_vs_safe_paper"
                ],
            }
        )
        if args.wandb_log_checkpoints:
            artifact = wandb_run.Artifact(f"{exp.name}-checkpoint", type="model")
            artifact.add_file(str(checkpoint_path))
            wandb_run.log_artifact(artifact)
        wandb_run.finish()
    return result


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_results.json"
    csv_path = output_dir / "ablation_results.csv"

    with json_path.open("w") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = [
        "name",
        "model_type",
        "layers",
        "token_pool",
        "input_dim",
        "lr",
        "lambda_reg",
        "seed",
        "n_history_steps",
        "best_epoch",
        "val_loss",
        "val_roc_auc",
        "val_tpr_at_5_fpr",
        "val_safe_early_stop_roc_auc",
        "val_full_rollout_roc_auc",
        "val_falert_early_roc_auc",
        "val_falert_early_tpr_at_5_fpr",
        "val_falert_end_roc_auc",
        "val_falert_end_tpr_at_5_fpr",
        "test_loss",
        "test_roc_auc",
        "test_tpr_at_5_fpr",
        "test_safe_early_stop_roc_auc",
        "test_full_rollout_roc_auc",
        "test_falert_early_roc_auc",
        "test_falert_early_tpr_at_5_fpr",
        "test_falert_end_roc_auc",
        "test_falert_end_tpr_at_5_fpr",
        "val_length_only_roc_auc",
        "val_progress_ratio_roc_auc",
        "val_task_min_step_roc_auc",
        "val_length_leakage_flag",
        "test_length_only_roc_auc",
        "test_progress_ratio_roc_auc",
        "test_task_min_step_roc_auc",
        "test_length_leakage_flag",
        "safe_paper_openvla_libero_unseen_roc_auc",
        "test_safe_early_stop_delta_vs_safe_paper",
        "test_full_rollout_delta_vs_safe_paper",
        "checkpoint_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fieldnames}
            row["layers"] = " ".join(str(layer) for layer in result["layers"])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/openvla_ablation")
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N dataloader batches before optimizer.step().",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable in-memory feature caching. The cache makes repeat-epoch "
            "access ~instant but holds every selected layer for all rollouts in "
            "RAM (layer_mix over all 9 OpenVLA layers is ~38 GB). Use on "
            "low-memory hosts to trade speed for footprint."
        ),
    )
    parser.add_argument(
        "--token-pool", default="last", help="Action-token pooling for non-sweep runs."
    )
    parser.add_argument(
        "--n-history-steps",
        type=int,
        default=1,
        help="Causal history window for MLP inputs.",
    )
    parser.add_argument("--lr", type=float, default=SAFE_OPTIMAL_LR)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=None,
        help=(
            "Override L2 regularization. By default this is model-aware: "
            "MLP=1e-2, LSTM=1, layer-fusion/linear defaults=1."
        ),
    )
    parser.add_argument("--lambda-success", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Max gradient norm. Use 0 to disable clipping, matching SAFE defaults.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        default="task",
        choices=["task", "random"],
        help="Use SAFE-style task holdout by default; random is rollout-level.",
    )
    parser.add_argument(
        "--unseen-task-ratio",
        type=float,
        default=0.3,
        help="Fraction of task IDs held out for unseen-task test when split-mode=task.",
    )
    parser.add_argument(
        "--seen-train-ratio",
        type=float,
        default=0.6,
        help="Fraction of seen-task rollouts used for training when split-mode=task.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training progress to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project", default="extend-safe", help="W&B project name."
    )
    parser.add_argument(
        "--wandb-rollout-project",
        default="extend-safe-rollout",
        help="W&B project used automatically for --safe-openvla-libero-sweep unless --wandb-project is overridden.",
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="Optional W&B entity or team."
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional base run name; experiment name is appended.",
    )
    parser.add_argument(
        "--wandb-run-prefix",
        default="",
        help="Optional prefix for per-ablation W&B run names.",
    )
    parser.add_argument(
        "--wandb-group", default=None, help="Shared W&B group for all ablation runs."
    )
    parser.add_argument(
        "--wandb-tags", nargs="*", default=[], help="Optional W&B tags."
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use offline to log locally without uploading.",
    )
    parser.add_argument(
        "--wandb-log-checkpoints",
        action="store_true",
        help="Log model checkpoints as W&B artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain experiments even if the final checkpoint already exists.",
    )
    parser.add_argument(
        "--resume-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-experiment .progress.pt checkpoints when present.",
    )
    parser.add_argument(
        "--progress-save-every",
        type=int,
        default=5,
        help="Save a resumable .progress.pt checkpoint every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run only these experiment names. By default, runs the full ablation set.",
    )
    parser.add_argument(
        "--skip-single-layer-sweep",
        action="store_true",
        help="Skip per-layer LSTM sweep experiments.",
    )
    parser.add_argument(
        "--custom-layers",
        type=parse_layers,
        help="Optional comma-separated layer list for one extra layer_mix_custom experiment.",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Train for all requested epochs, matching SAFE's fixed-epoch training loop.",
    )
    parser.add_argument(
        "--safe-openvla-libero-sweep",
        action="store_true",
        help=(
            "Run the SAFE OpenVLA LIBERO MLP/LSTM grid from submit_openvla_libero.bash: "
            "seeds 0,1,2; token_idx_rel mean,0.0,1.0; lr 1e-4,3e-4,1e-3; "
            "lambda_reg 1e-3,1e-2,1e-1,1; batch size 64; Adam."
        ),
    )
    parser.add_argument("--sweep-seeds", type=parse_csv_ints, default=(0, 1, 2))
    parser.add_argument(
        "--sweep-token-pools", type=parse_csv_strings, default=("mean", "0.0", "1.0")
    )
    parser.add_argument(
        "--sweep-lrs", type=parse_csv_floats, default=(1e-4, 3e-4, 1e-3)
    )
    parser.add_argument(
        "--sweep-lambda-reg", type=parse_csv_floats, default=(1e-3, 1e-2, 1e-1, 1.0)
    )
    parser.add_argument(
        "--sweep-models", type=parse_csv_strings, default=("lstm", "mlp")
    )
    parser.add_argument("--sweep-history-steps", type=parse_csv_ints, default=(1,))
    parser.add_argument(
        "--sweep-layers",
        type=parse_layers,
        default=None,
        help=(
            "Override the layers used in the SAFE sweep (comma-separated model-layer "
            "ids, e.g. 18,21,24). Defaults to the final saved layer only."
        ),
    )
    parser.add_argument(
        "--safe-sweep-epochs",
        type=int,
        default=50,
        help="Epoch hard cap for the SAFE OpenVLA LIBERO sweep.",
    )
    parser.add_argument(
        "--safe-sweep-batch-size",
        type=int,
        default=64,
        help="Batch size for --safe-openvla-libero-sweep. Default matches SAFE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved splits/experiments and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.safe_openvla_libero_sweep:
        args.batch_size = args.safe_sweep_batch_size
        args.epochs = args.safe_sweep_epochs
        args.optimizer = "adam"
        if args.wandb_project == "extend-safe":
            args.wandb_project = args.wandb_rollout_project

    seed_everything(args.seed)
    if args.wandb and args.wandb_group is None:
        group_prefix = (
            "safe-openvla-libero"
            if args.safe_openvla_libero_sweep
            else "openvla-ablation"
        )
        args.wandb_group = f"{group_prefix}-{int(time.time())}"

    root = args.root.expanduser().resolve()
    device = torch.device(args.device)

    saved_layers = discover_saved_layers(root)
    # Route through the memo so the split-computation pass is reused by any
    # experiment that also uses the last layer with last-token pooling.
    base_dataset = get_dataset(
        root, layers=(saved_layers[-1],), token_pool="last", cache=not args.no_cache
    )

    if args.safe_openvla_libero_sweep:
        experiments = build_safe_openvla_libero_experiments(
            saved_layers=saved_layers,
            seeds=args.sweep_seeds,
            token_pools=args.sweep_token_pools,
            lrs=args.sweep_lrs,
            lambda_regs=args.sweep_lambda_reg,
            models=args.sweep_models,
            history_steps=args.sweep_history_steps,
            layers=args.sweep_layers,
        )
        experiments = filter_experiments(
            experiments,
            include=set(args.only) if args.only else None,
            skip_single_layer_sweep=False,
        )
        split_seeds = sorted(
            {int(exp.seed) for exp in experiments if exp.seed is not None}
        )
    else:
        token_pool = normalize_token_pool(args.token_pool)
        experiments = [
            replace(
                exp,
                token_pool=token_pool,
                seed=args.seed,
                n_history_steps=args.n_history_steps,
            )
            for exp in build_default_experiments(saved_layers)
        ]
        if args.custom_layers is not None:
            experiments.append(
                Experiment(
                    "layer_mix_custom",
                    "layer_mix",
                    args.custom_layers,
                    token_pool=token_pool,
                    seed=args.seed,
                    n_history_steps=args.n_history_steps,
                )
            )
        experiments = filter_experiments(
            experiments,
            include=set(args.only) if args.only else None,
            skip_single_layer_sweep=args.skip_single_layer_sweep,
        )
        split_seeds = [args.seed]

    splits: dict[int, dict[str, list[int]]] = {}
    split_metadata_by_seed: dict[int, dict[str, object]] = {}
    for split_seed in split_seeds:
        if args.split_mode == "task":
            split, split_metadata = split_indices_by_task(
                base_dataset,
                unseen_task_ratio=args.unseen_task_ratio,
                seen_train_ratio=args.seen_train_ratio,
                seed=split_seed,
            )
        else:
            labels = labels_for_dataset(base_dataset)
            split = split_indices(labels, args.train_frac, args.val_frac, split_seed)
            split_metadata = {
                "split_mode": "random",
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": 1.0 - args.train_frac - args.val_frac,
            }

        split_metadata["seed"] = split_seed
        splits[split_seed] = split
        split_metadata_by_seed[split_seed] = split_metadata

        suffix = "" if len(split_seeds) == 1 else f"_seed{split_seed}"
        with (args.output_dir / f"split_indices{suffix}.json").open("w") as handle:
            json.dump(split, handle, indent=2)
        with (args.output_dir / f"split_metadata{suffix}.json").open("w") as handle:
            json.dump(split_metadata, handle, indent=2)

    if len(split_seeds) > 1:
        with (args.output_dir / "split_metadata_by_seed.json").open("w") as handle:
            json.dump(split_metadata_by_seed, handle, indent=2)

    print(f"root={root}")
    print(f"saved_layers={saved_layers}")
    print(f"safe_openvla_libero_sweep={args.safe_openvla_libero_sweep}")
    print(f"split_metadata_by_seed={split_metadata_by_seed}")
    print(
        f"split_sizes_by_seed={ {seed: {name: len(idx) for name, idx in split.items()} for seed, split in splits.items()} }"
    )
    print(f"wandb_project={args.wandb_project}")
    print(f"wandb_group={args.wandb_group}")
    print(f"running {len(experiments)} experiments on {device}")
    if args.dry_run:
        for exp in experiments:
            print(
                f"dry_run {exp.name}: model={exp.model_type} layers={exp.layers} "
                f"token_pool={exp.token_pool} seed={exp.seed} "
                f"lr={exp.lr if exp.lr is not None else args.lr} "
                f"lambda_reg={resolve_lambda_reg(exp.model_type, exp.lambda_reg, args.lambda_reg)} "
                f"n_history_steps={exp.n_history_steps}"
            )
        return

    results = []
    for exp in experiments:
        exp_seed = exp.seed if exp.seed is not None else args.seed
        checkpoint_path = args.output_dir / f"{exp.name}.pt"
        if checkpoint_path.exists() and not args.force:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            result = dict(checkpoint["result"])
            result["checkpoint_path"] = str(checkpoint_path)
            results.append(result)
            print(f"\n== {exp.name}: SKIP existing checkpoint={checkpoint_path} ==")
            continue

        seed_everything(exp_seed)
        split = splits[exp_seed]
        print(
            f"\n== {exp.name}: {exp.model_type}, layers={exp.layers}, "
            f"token_pool={exp.token_pool}, seed={exp_seed}, "
            f"lr={exp.lr if exp.lr is not None else args.lr}, "
            f"lambda_reg={resolve_lambda_reg(exp.model_type, exp.lambda_reg, args.lambda_reg)}, "
            f"n_history_steps={exp.n_history_steps} =="
        )
        result = train_one_experiment(exp, root, split, args, device)
        results.append(result)
        print(
            f"done {exp.name}: val_falert_early_auc={result['val_roc_auc']:.4f} "
            f"test_falert_early_auc={result['test_roc_auc']:.4f} "
            f"test_falert_end_auc={result['test_falert_end_roc_auc']:.4f} "
            f"test_length_only_auc={result['test_length_only_roc_auc']:.4f} "
            f"checkpoint={result['checkpoint_path']}"
        )

    write_results(results, args.output_dir)
    print(f"\nwrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
