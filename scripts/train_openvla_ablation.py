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
from dataclasses import asdict, dataclass
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
from models import LayerMixLSTMModel, LinearProbeModel, SafeLSTMModel, SafeMLPModel


DEFAULT_ROOT = Path("data/rollouts/openvla")


@dataclass(frozen=True)
class Experiment:
    name: str
    model_type: str
    layers: tuple[int, ...]


def parse_layers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def discover_saved_layers(root: Path) -> tuple[int, ...]:
    dataset = OpenVLARolloutDataset(root, layers=None, token_pool="last", recursive=True)
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
    ]
    experiments.extend(
        Experiment(f"linear_probe_layer_{layer}", "linear_probe", (layer,)) for layer in saved_layers
    )
    experiments.extend(
        Experiment(f"safe_lstm_layer_{layer}", "lstm", (layer,)) for layer in saved_layers
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
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """SAFE-style split: hold out task IDs, then split seen-task rollouts."""

    if not 0.0 <= unseen_task_ratio < 1.0:
        raise ValueError("unseen_task_ratio must be in [0, 1)")
    if not 0.0 < seen_train_ratio < 1.0:
        raise ValueError("seen_train_ratio must be in (0, 1)")

    task_ids = task_ids_for_dataset(dataset)
    unique_task_ids = sorted(set(task_ids))
    if len(unique_task_ids) < 2:
        raise ValueError("Task-level split requires at least two task IDs")

    rng = np.random.default_rng(seed)
    shuffled_tasks = list(rng.permutation(unique_task_ids))
    n_unseen = round(unseen_task_ratio * len(unique_task_ids))
    if unseen_task_ratio > 0 and n_unseen == 0:
        n_unseen = 1
    if n_unseen >= len(unique_task_ids):
        n_unseen = len(unique_task_ids) - 1

    unseen_task_ids = sorted(int(task_id) for task_id in shuffled_tasks[:n_unseen])
    seen_task_ids = sorted(int(task_id) for task_id in shuffled_tasks[n_unseen:])
    unseen_set = set(unseen_task_ids)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    per_task_counts: dict[str, dict[str, int]] = {}

    for task_id in unique_task_ids:
        indices = [idx for idx, rollout_task_id in enumerate(task_ids) if rollout_task_id == task_id]
        indices = [int(idx) for idx in rng.permutation(indices)]
        if task_id in unseen_set:
            test_indices.extend(indices)
            per_task_counts[str(task_id)] = {"train": 0, "val": 0, "test": len(indices)}
            continue

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
        "unseen_task_ratio": unseen_task_ratio,
        "seen_train_ratio": seen_train_ratio,
        "seen_task_ids": seen_task_ids,
        "unseen_task_ids": unseen_task_ids,
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


def make_loaders(
    root: Path,
    layers: tuple[int, ...],
    split: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
) -> tuple[OpenVLARolloutDataset, dict[str, DataLoader]]:
    dataset = OpenVLARolloutDataset(root, layers=layers, token_pool="last", recursive=True)
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
            loss_type="safe",
        )
    elif exp.model_type == "lstm":
        model = SafeLSTMModel(
            input_dim=dataset.input_dim,
            hidden_dim=hidden_dim,
            n_layers=1,
            dropout=dropout,
            loss_type="bce",
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

    base_config = {key: sanitize_config_value(value) for key, value in vars(args).items()}
    base_config.update(
        {
            "experiment": asdict(exp),
            "model_type": exp.model_type,
            "selected_layers": list(dataset.selected_layers),
            "saved_layers": list(dataset.saved_layers),
            "token_pool": "last",
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


def masked_rollout_scores(scores: torch.Tensor, valid_masks: torch.Tensor) -> torch.Tensor:
    scores = scores.squeeze(-1)
    masked = scores.masked_fill(~valid_masks.bool(), float("-inf"))
    return masked.max(dim=1).values


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    labels = []
    rollout_scores = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss, _ = model.forward_loss(batch)
        scores = model(batch)
        losses.append(float(loss.detach().cpu()))
        labels.extend(batch["labels"].detach().cpu().int().tolist())
        rollout_scores.extend(
            masked_rollout_scores(scores, batch["valid_masks"]).detach().cpu().tolist()
        )

    metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, rollout_scores))
        fpr, tpr, _ = roc_curve(labels, rollout_scores)
        valid = np.where(fpr <= 0.05)[0]
        metrics["tpr_at_5_fpr"] = float(tpr[valid].max()) if len(valid) else 0.0
    else:
        metrics["roc_auc"] = float("nan")
        metrics["tpr_at_5_fpr"] = float("nan")
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
        split=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(
        exp=exp,
        dataset=dataset,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb_run = init_wandb_run(args, exp, dataset, split)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.forward_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, loaders["val"], device) if "val" in loaders else {"loss": float("nan")}
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = val_metrics["loss"]
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"{exp.name} epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auc={val_metrics.get('roc_auc', float('nan')):.4f}"
            )

        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/roc_auc": val_metrics.get("roc_auc", float("nan")),
                "val/tpr_at_5_fpr": val_metrics.get("tpr_at_5_fpr", float("nan")),
            }
            if isinstance(model, LayerMixLSTMModel):
                for layer, weight in zip(dataset.selected_layers, model.layer_weights.cpu().tolist()):
                    log_payload[f"layer_weights/layer_{layer}"] = weight
            wandb_run.log(log_payload, step=epoch)

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, loaders["test"], device) if "test" in loaders else {"loss": float("nan")}
    val_metrics = evaluate(model, loaders["val"], device) if "val" in loaders else {"loss": float("nan")}

    result = {
        "name": exp.name,
        "model_type": exp.model_type,
        "layers": list(exp.layers),
        "input_dim": dataset.input_dim,
        "hidden_dim_per_layer": dataset.hidden_dim,
        "n_rollouts": len(dataset),
        "best_epoch": best_epoch,
        "val_loss": val_metrics["loss"],
        "val_roc_auc": val_metrics.get("roc_auc", float("nan")),
        "val_tpr_at_5_fpr": val_metrics.get("tpr_at_5_fpr", float("nan")),
        "test_loss": test_metrics["loss"],
        "test_roc_auc": test_metrics.get("roc_auc", float("nan")),
        "test_tpr_at_5_fpr": test_metrics.get("tpr_at_5_fpr", float("nan")),
    }
    if isinstance(model, LayerMixLSTMModel):
        result["layer_weights"] = {
            str(layer): weight
            for layer, weight in zip(dataset.selected_layers, model.layer_weights.cpu().tolist())
        }

    checkpoint_path = args.output_dir / f"{exp.name}.pt"
    torch.save(
        {
            "experiment": asdict(exp),
            "result": result,
            "model_state_dict": model.state_dict(),
            "selected_layers": dataset.selected_layers,
            "saved_layers": dataset.saved_layers,
            "token_pool": "last",
        },
        checkpoint_path,
    )
    result["checkpoint_path"] = str(checkpoint_path)

    if wandb_run is not None:
        wandb_run.log(
            {
                "best_epoch": best_epoch,
                "final/val_loss": result["val_loss"],
                "final/val_roc_auc": result["val_roc_auc"],
                "final/val_tpr_at_5_fpr": result["val_tpr_at_5_fpr"],
                "test/loss": result["test_loss"],
                "test/roc_auc": result["test_roc_auc"],
                "test/tpr_at_5_fpr": result["test_tpr_at_5_fpr"],
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
        "input_dim",
        "best_epoch",
        "val_loss",
        "val_roc_auc",
        "val_tpr_at_5_fpr",
        "test_loss",
        "test_roc_auc",
        "test_tpr_at_5_fpr",
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/openvla_ablation"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Log training progress to Weights & Biases.")
    parser.add_argument("--wandb-project", default="extend-safe", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity or team.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional base run name; experiment name is appended.")
    parser.add_argument("--wandb-run-prefix", default="", help="Optional prefix for per-ablation W&B run names.")
    parser.add_argument("--wandb-group", default=None, help="Shared W&B group for all ablation runs.")
    parser.add_argument("--wandb-tags", nargs="*", default=[], help="Optional W&B tags.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb and args.wandb_group is None:
        args.wandb_group = f"openvla-ablation-{int(time.time())}"

    root = args.root.expanduser().resolve()
    device = torch.device(args.device)

    saved_layers = discover_saved_layers(root)
    base_dataset = OpenVLARolloutDataset(root, layers=(saved_layers[-1],), token_pool="last", recursive=True)
    if args.split_mode == "task":
        split, split_metadata = split_indices_by_task(
            base_dataset,
            unseen_task_ratio=args.unseen_task_ratio,
            seen_train_ratio=args.seen_train_ratio,
            seed=args.seed,
        )
    else:
        labels = labels_for_dataset(base_dataset)
        split = split_indices(labels, args.train_frac, args.val_frac, args.seed)
        split_metadata = {
            "split_mode": "random",
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": 1.0 - args.train_frac - args.val_frac,
        }
    with (args.output_dir / "split_indices.json").open("w") as handle:
        json.dump(split, handle, indent=2)
    with (args.output_dir / "split_metadata.json").open("w") as handle:
        json.dump(split_metadata, handle, indent=2)

    experiments = build_default_experiments(saved_layers)
    if args.custom_layers is not None:
        experiments.append(Experiment("layer_mix_custom", "layer_mix", args.custom_layers))
    experiments = filter_experiments(
        experiments,
        include=set(args.only) if args.only else None,
        skip_single_layer_sweep=args.skip_single_layer_sweep,
    )

    print(f"root={root}")
    print(f"saved_layers={saved_layers}")
    print(f"token_pool=last")
    print(f"split_metadata={split_metadata}")
    print(f"split_sizes={ {name: len(idx) for name, idx in split.items()} }")
    print(f"running {len(experiments)} experiments on {device}")

    results = []
    for exp in experiments:
        print(f"\n== {exp.name}: {exp.model_type}, layers={exp.layers} ==")
        result = train_one_experiment(exp, root, split, args, device)
        results.append(result)
        print(
            f"done {exp.name}: val_auc={result['val_roc_auc']:.4f} "
            f"test_auc={result['test_roc_auc']:.4f} checkpoint={result['checkpoint_path']}"
        )

    write_results(results, args.output_dir)
    print(f"\nwrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
