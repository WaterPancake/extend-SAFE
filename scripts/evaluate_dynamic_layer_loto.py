"""LOTO gate for episode-aligned information beyond OpenVLA's final layer.

This experiment compares a capacity-matched layer-32 monitor with a dynamic
layer-20+32 monitor.  A within-task shuffled-layer-20 control preserves task
and early-progress structure while destroying episode alignment.  All models
train and evaluate only through each task's minimum rollout length, avoiding
the known timeout/rollout-length shortcut.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import DynamicLayerMixLSTMModel, ResidualAuxLayerLSTMModel


VARIANTS = (
    "baseline32",
    "layer20_32",
    "shuffled20_32",
    "late24_28_32",
    "shuffled_late24_28_32",
    "residual20_32",
    "shuffled_residual20_32",
)
DEFAULT_VARIANTS = VARIANTS[:3]


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset-layers",
        type=parse_csv_ints,
        default=(20, 32),
        help="Layers stored in --root and exposed to the experiment.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/openvla_dynamic_layer_loto_500")
    )
    parser.add_argument("--folds", type=parse_csv_ints, default=None)
    parser.add_argument("--variants", type=parse_csv_strings, default=DEFAULT_VARIANTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}; choose from {VARIANTS}")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyLayerView(Dataset):
    """Crop to the SAFE early window and select/shuffle requested layers."""

    def __init__(
        self,
        base: OpenVLARolloutDataset,
        variant: str,
        donor_by_index: dict[int, int] | None = None,
    ) -> None:
        self.base = base
        self.variant = variant
        self.donor_by_index = donor_by_index or {}
        self.hidden_dim = base.hidden_dim
        self.position_by_layer = {
            layer: position for position, layer in enumerate(base.selected_layers)
        }
        if variant == "baseline32":
            self.selected_layers = (32,)
        elif "late24_28_32" in variant:
            self.selected_layers = (24, 28, 32)
        else:
            self.selected_layers = (20, 32)
        missing = [
            layer for layer in self.selected_layers if layer not in self.position_by_layer
        ]
        if missing:
            raise ValueError(
                f"variant {variant} needs layers {missing}, but dataset has "
                f"{base.selected_layers}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def _layer(self, features: torch.Tensor, position: int) -> torch.Tensor:
        start = position * self.hidden_dim
        return features[:, start : start + self.hidden_dim]

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        stop = int(item["task_min_step"])
        layer32 = self._layer(
            item["features"], self.position_by_layer[32]
        )[:stop]
        if self.variant == "baseline32":
            features = layer32
        else:
            feature_parts = []
            if self.variant.startswith("shuffled"):
                donor_index = self.donor_by_index[index]
                donor = self.base[donor_index]
                auxiliary_source = donor["features"]
                if len(auxiliary_source) < stop:
                    raise RuntimeError(
                        f"donor {donor_index} is shorter than task early window {stop}"
                    )
            else:
                auxiliary_source = item["features"]
            for layer in self.selected_layers[:-1]:
                feature_parts.append(
                    self._layer(
                        auxiliary_source, self.position_by_layer[layer]
                    )[:stop]
                )
            feature_parts.append(layer32)
            features = torch.cat(feature_parts, dim=-1)

        out = dict(item)
        out["features"] = features
        out["length"] = torch.tensor(stop, dtype=torch.long)
        out["task_min_step"] = torch.tensor(stop, dtype=torch.long)
        out["selected_layers"] = self.selected_layers
        return out


def task_id(base: OpenVLARolloutDataset, index: int) -> int:
    value = base.rollouts[index].task_id
    if value is None:
        raise ValueError(f"missing task id for {base.rollouts[index].path}")
    return int(value)


def failure_label(base: OpenVLARolloutDataset, index: int) -> int:
    success = base.rollouts[index].success
    if success is None:
        item = base[index]
        return int(item["label"])
    return int(not success)


def build_loto_split(
    base: OpenVLARolloutDataset, held_out_task: int, seed: int
) -> dict[str, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(base)):
        grouped[task_id(base, index)].append(index)

    split = {"train": [], "val": [], "test": []}
    for task, indices in sorted(grouped.items()):
        if task == held_out_task:
            split["test"].extend(indices)
            continue
        rng = np.random.default_rng(seed + 1009 * task)
        shuffled = np.asarray(indices)[rng.permutation(len(indices))].tolist()
        n_train = min(max(int(0.8 * len(shuffled)), 1), len(shuffled) - 1)
        split["train"].extend(shuffled[:n_train])
        split["val"].extend(shuffled[n_train:])
    return {name: sorted(indices) for name, indices in split.items()}


def shuffled_donors(
    base: OpenVLARolloutDataset,
    split: dict[str, list[int]],
    seed: int,
) -> dict[int, int]:
    """Derange layer-20 donors within task and within data partition."""

    donors: dict[int, int] = {}
    for partition_number, indices in enumerate(split.values()):
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            grouped[task_id(base, index)].append(index)
        for task, group in grouped.items():
            if len(group) < 2:
                raise ValueError(f"cannot derange singleton task {task}")
            rng = np.random.default_rng(seed + 7919 * task + partition_number)
            order = np.asarray(group)[rng.permutation(len(group))].tolist()
            shifted = order[1:] + order[:1]
            donors.update(zip(order, shifted))
    if set(donors) != set(itertools.chain.from_iterable(split.values())):
        raise RuntimeError("shuffled donor map does not cover the split")
    if any(index == donor for index, donor in donors.items()):
        raise RuntimeError("shuffled donor map contains a fixed point")
    return donors


def make_loaders(
    dataset: Dataset,
    split: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for offset, (name, indices) in enumerate(split.items()):
        generator = torch.Generator().manual_seed(seed + offset)
        loaders[name] = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=(name == "train"),
            generator=generator,
            num_workers=num_workers,
            collate_fn=collate_rollouts,
        )
    return loaders


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def class_weights(base: OpenVLARolloutDataset, indices: list[int]) -> tuple[float, float]:
    labels = np.asarray([failure_label(base, index) for index in indices])
    n_failure = int(labels.sum())
    n_success = len(labels) - n_failure
    return (len(labels) / (2 * n_failure), len(labels) / (2 * n_success))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    records: list[dict[str, object]] = []
    losses: list[float] = []
    gate_sums = np.zeros(getattr(model, "n_layers", 0), dtype=np.float64)
    gate_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        loss, _ = model.forward_loss(batch)
        scores = model(batch).squeeze(-1)
        masks = batch["valid_masks"].bool()
        rollout_scores = scores.masked_fill(~masks, float("-inf")).max(dim=1).values
        if hasattr(model, "gate_weights"):
            gates = model.gate_weights(batch)
            gate_sums += (
                gates * masks.unsqueeze(-1)
            ).sum(dim=(0, 1)).detach().cpu().numpy()
            gate_count += int(masks.sum())
        losses.append(float(loss.detach().cpu()))
        for row, score in enumerate(rollout_scores.detach().cpu().tolist()):
            records.append(
                {
                    "path": raw_batch["paths"][row],
                    "task_id": int(raw_batch["task_ids"][row]),
                    "episode_idx": int(raw_batch["episode_indices"][row]),
                    "label": int(raw_batch["labels"][row]),
                    "score": float(score),
                }
            )

    labels = [int(row["label"]) for row in records]
    scores = [float(row["score"]) for row in records]
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else math.nan
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(fpr <= 0.05)
    tpr5 = float(tpr[valid].max()) if len(valid) else 0.0
    metrics = {
        "loss": float(np.mean(losses)),
        "roc_auc": auc,
        "tpr_at_5_fpr": tpr5,
        "n": len(records),
        "n_failure": int(sum(labels)),
        "n_success": int(len(labels) - sum(labels)),
    }
    if len(gate_sums):
        for position, value in enumerate(gate_sums / max(gate_count, 1)):
            metrics[f"mean_gate_position_{position}"] = float(value)
    return metrics, records


def train_variant(
    base: OpenVLARolloutDataset,
    split: dict[str, list[int]],
    variant: str,
    args: argparse.Namespace,
    fold: int,
) -> dict[str, object]:
    result_path = args.output_dir / f"fold_{fold}_{variant}.json"
    checkpoint_path = args.output_dir / f"fold_{fold}_{variant}.pt"
    if result_path.exists() and checkpoint_path.exists() and not args.force:
        with result_path.open() as handle:
            return json.load(handle)

    fold_seed = args.seed + 104729 * fold
    donors = (
        shuffled_donors(base, split, fold_seed)
        if variant.startswith("shuffled")
        else None
    )
    dataset = EarlyLayerView(base, variant, donors)
    loaders = make_loaders(
        dataset, split, args.batch_size, args.num_workers, fold_seed
    )
    n_layers = len(dataset.selected_layers)
    seed_everything(fold_seed)
    if "residual" in variant:
        baseline_path = args.output_dir / f"fold_{fold}_baseline32.pt"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"{baseline_path} is required before training residual variants"
            )
        baseline_checkpoint = torch.load(
            baseline_path, map_location="cpu", weights_only=False
        )
        baseline_config = baseline_checkpoint["config"]
        if (
            int(baseline_config["projection_dim"]) != args.projection_dim
            or int(baseline_config["hidden_dim"]) != args.hidden_dim
        ):
            raise ValueError("residual and baseline dimensions must match")
        model = ResidualAuxLayerLSTMModel(
            hidden_dim_per_layer=base.hidden_dim,
            base_projection_dim=args.projection_dim,
            base_lstm_hidden_dim=args.hidden_dim,
            aux_projection_dim=max(args.projection_dim // 2, 1),
            aux_lstm_hidden_dim=max(args.hidden_dim // 2, 1),
            dropout=0.0,
        )
        model.base.load_state_dict(baseline_checkpoint["model_state_dict"])
        model.freeze_base()
        model = model.to(args.device)
    else:
        model = DynamicLayerMixLSTMModel(
            input_dim=n_layers * base.hidden_dim,
            n_layers=n_layers,
            hidden_dim_per_layer=base.hidden_dim,
            projection_dim=args.projection_dim,
            lstm_hidden_dim=args.hidden_dim,
            dropout=0.0,
            loss_type="bce",
        ).to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    weights = class_weights(base, split["train"])

    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    patience_left = args.patience
    if "residual" in variant:
        initial_metrics, _ = evaluate(
            model, loaders["val"], torch.device(args.device)
        )
        best_val_loss = initial_metrics["loss"]
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    for epoch in range(1, args.epochs + 1):
        model.train()
        if "residual" in variant:
            model.base.eval()
        train_losses = []
        for raw_batch in loaders["train"]:
            batch = move_batch(raw_batch, torch.device(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.forward_loss(batch, weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, loaders["val"], torch.device(args.device))
        if val_metrics["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"fold={fold} variant={variant} epoch={epoch} "
                f"train_loss={np.mean(train_losses):.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['roc_auc']:.4f}",
                flush=True,
            )
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("training never produced a checkpoint")
    model.load_state_dict(best_state)
    val_metrics, _ = evaluate(model, loaders["val"], torch.device(args.device))
    test_metrics, test_records = evaluate(
        model, loaders["test"], torch.device(args.device)
    )
    result = {
        "fold": fold,
        "variant": variant,
        "selected_layers": list(dataset.selected_layers),
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "split_sizes": {name: len(indices) for name, indices in split.items()},
        "class_weights": list(weights),
        "val": val_metrics,
        "test": test_metrics,
    }
    torch.save(
        {
            "result": result,
            "model_state_dict": best_state,
            "split": split,
            "config": vars(args),
        },
        checkpoint_path,
    )
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2, default=str)
    with (args.output_dir / f"fold_{fold}_{variant}_scores.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=test_records[0].keys())
        writer.writeheader()
        writer.writerows(test_records)
    return result


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def sign_flip_pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict:
    by_variant = {
        variant: sorted(
            [row for row in results if row["variant"] == variant],
            key=lambda row: row["fold"],
        )
        for variant in args.variants
    }
    summary: dict[str, object] = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "n_source_rollouts": len(results) and sum(
            by_variant[args.variants[0]][0]["split_sizes"].values()
        ),
        "variants": {},
        "comparisons": {},
    }
    for variant, rows in by_variant.items():
        aucs = np.asarray([row["test"]["roc_auc"] for row in rows], dtype=float)
        summary["variants"][variant] = {
            "macro_task_auc": float(aucs.mean()),
            "task_auc_std": float(aucs.std()),
            "task_aucs": {str(row["fold"]): row["test"]["roc_auc"] for row in rows},
            "parameter_counts": sorted({row["parameter_count"] for row in rows}),
            "mean_gate_weights": {
                str(position): float(np.mean([
                    row["test"][f"mean_gate_position_{position}"] for row in rows
                ]))
                for position in range(len(rows[0]["selected_layers"]))
                if all(
                    f"mean_gate_position_{position}" in row["test"] for row in rows
                )
            },
        }

    comparisons = []
    if "baseline32" in by_variant:
        comparisons.extend(
            (variant, "baseline32")
            for variant in args.variants
            if variant != "baseline32"
        )
    if {"layer20_32", "shuffled20_32"}.issubset(by_variant):
        comparisons.append(("layer20_32", "shuffled20_32"))
    if {"late24_28_32", "shuffled_late24_28_32"}.issubset(by_variant):
        comparisons.append(("late24_28_32", "shuffled_late24_28_32"))
    if {"residual20_32", "shuffled_residual20_32"}.issubset(by_variant):
        comparisons.append(("residual20_32", "shuffled_residual20_32"))

    for left, right in comparisons:
        left_rows = by_variant[left]
        right_rows = by_variant[right]
        left_folds = [row["fold"] for row in left_rows]
        if left_folds != [row["fold"] for row in right_rows]:
            continue
        left_auc = np.asarray([row["test"]["roc_auc"] for row in left_rows])
        right_auc = np.asarray([row["test"]["roc_auc"] for row in right_rows])
        delta = left_auc - right_auc
        low, high = bootstrap_interval(delta, args.seed)
        summary["comparisons"][f"{left}_minus_{right}"] = {
            "mean_delta_auc": float(delta.mean()),
            "bootstrap_95_ci": [low, high],
            "positive_tasks": int((delta > 0).sum()),
            "n_tasks": len(delta),
            "sign_flip_pvalue_two_sided": sign_flip_pvalue(delta),
            "per_task_delta": {
                str(fold): float(value) for fold, value in zip(left_folds, delta)
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = OpenVLARolloutDataset(
        args.root,
        layers=tuple(args.dataset_layers),
        token_pool="last",
        recursive=True,
        cache=True,
    )
    tasks = sorted({task_id(base, index) for index in range(len(base))})
    folds = tuple(tasks) if args.folds is None else tuple(args.folds)
    missing = sorted(set(folds) - set(tasks))
    if missing:
        raise ValueError(f"requested folds are not dataset tasks: {missing}")
    print(
        f"root={args.root} rollouts={len(base)} tasks={tasks} folds={folds} "
        f"device={args.device}",
        flush=True,
    )

    results = []
    for fold in folds:
        split = build_loto_split(base, fold, args.seed)
        for variant in args.variants:
            result = train_variant(base, split, variant, args, fold)
            results.append(result)
            print(
                f"done fold={fold} variant={variant} "
                f"test_auc={result['test']['roc_auc']:.4f}",
                flush=True,
            )
    summary = summarize(results, args)
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
