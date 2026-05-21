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
                f"train_loss={np.mean(train_losses):.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_auc={val_metrics.get('roc_auc', float('nan')):.4f}"
            )

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    root = args.root.expanduser().resolve()
    device = torch.device(args.device)

    saved_layers = discover_saved_layers(root)
    base_dataset = OpenVLARolloutDataset(root, layers=(saved_layers[-1],), token_pool="last", recursive=True)
    labels = labels_for_dataset(base_dataset)
    split = split_indices(labels, args.train_frac, args.val_frac, args.seed)
    with (args.output_dir / "split_indices.json").open("w") as handle:
        json.dump(split, handle, indent=2)

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
