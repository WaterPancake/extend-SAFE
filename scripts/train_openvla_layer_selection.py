"""Train SAFE LSTM layer-selection experiments for OpenVLA latents.

This is a focused follow-up to ``train_openvla_ablation.py``:

1. train one SAFE LSTM per captured layer,
2. rank layers by validation metric,
3. train SAFE LSTMs on concatenated top-k layer features.

Defaults use the current best LSTM hyperparameters from the OpenVLA sweep:
last action token, lr=1e-4, lambda_reg=1, SAFE-style task holdout, Adam, and
fixed-epoch training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from train_openvla_ablation import (
    DEFAULT_ROOT,
    Experiment,
    _DATASET_CACHE,
    discover_saved_layers,
    get_dataset,
    labels_for_dataset,
    normalize_token_pool,
    parse_csv_ints,
    parse_layers,
    seed_everything,
    split_indices,
    split_indices_by_task,
    task_ids_for_dataset,
    train_one_experiment,
    write_results,
)


DEFAULT_RANKING_METRIC = "val_falert_early_roc_auc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/openvla_lstm_layer_selection"),
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=None,
        help=(
            "Optional comma-separated captured layer IDs to sweep. Defaults to every "
            "layer advertised by the rollout artifacts."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_ints,
        default=(0, 1, 2),
        help="Comma-separated train/split seeds.",
    )
    parser.add_argument(
        "--ranking-metric",
        default=DEFAULT_RANKING_METRIC,
        help=(
            "Validation metric used to rank single layers before top-k concat. "
            "Default is SAFE-style earliest-stop ROC-AUC."
        ),
    )
    parser.add_argument(
        "--max-top-k",
        type=int,
        default=None,
        help=(
            "Largest top-k concat set to train. Defaults to ceil(n_layers / 2), "
            "the 'best half' hard stop."
        ),
    )
    parser.add_argument(
        "--min-top-k",
        type=int,
        default=2,
        help="Smallest top-k concat set to train after ranking.",
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm", "mlp", "linear_probe"],
        default="lstm",
        help="Probe architecture for every experiment in the sweep.",
    )
    parser.add_argument(
        "--skip-topk",
        action="store_true",
        help="Stop after the single-layer sweep (no top-k concat phase).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N dataloader batches before optimizer.step().",
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset feature caching to reduce RAM at the cost of speed.",
    )
    parser.add_argument(
        "--keep-dataset-cache",
        action="store_true",
        help=(
            "Keep cached feature views across experiments. Faster on high-RAM hosts, "
            "but single-layer plus top-k runs can accumulate many GB."
        ),
    )
    parser.add_argument(
        "--token-pool",
        default="last",
        help="Action-token pooling. Default: last.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lambda-success", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Max gradient norm. Use 0 to disable clipping.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--early-stopping",
        action="store_false",
        dest="no_early_stopping",
        help="Enable patience-based early stopping instead of fixed-epoch training.",
    )
    parser.set_defaults(no_early_stopping=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        default="task",
        choices=["task", "random"],
        help="Use SAFE-style task holdout by default; random is rollout-level.",
    )
    parser.add_argument("--unseen-task-ratio", type=float, default=0.3)
    parser.add_argument("--seen-train-ratio", type=float, default=0.6)
    parser.add_argument(
        "--loto",
        action="store_true",
        help=(
            "Leave-one-task-out: run one fold per task, holding that task out as "
            "the test set and training on the rest. Overrides --seeds (each fold "
            "is labelled _seed-<held_out_task_id>) and forces --skip-topk."
        ),
    )
    parser.add_argument(
        "--loto-seed",
        type=int,
        default=0,
        help="Seed for the seen-task train/val shuffle in --loto mode (fixed across folds).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain experiments even if the checkpoint already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved layer plan and exit before training.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", default="safe-replicate-openvla-layer-selection"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-prefix", default="")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    return parser.parse_args()


def experiment_name(prefix: str, layers: Iterable[int], args: argparse.Namespace) -> str:
    layer_tag = "-".join(str(layer) for layer in layers)
    token_tag = normalize_token_pool(args.token_pool).replace("_", "-")
    return (
        f"safe_{args.model_type}_{prefix}_layers_{layer_tag}"
        f"_tok-{token_tag}_lr-{args.lr:g}_reg-{args.lambda_reg:g}"
    )


def single_layer_experiments(
    layers: tuple[int, ...], seed: int, args: argparse.Namespace
) -> list[Experiment]:
    token_pool = normalize_token_pool(args.token_pool)
    return [
        Experiment(
            name=f"{experiment_name('single', (layer,), args)}_seed-{seed}",
            model_type=args.model_type,
            layers=(layer,),
            token_pool=token_pool,
            lr=args.lr,
            lambda_reg=args.lambda_reg,
            seed=seed,
        )
        for layer in layers
    ]


def topk_experiment(
    k: int, layers: tuple[int, ...], seed: int, args: argparse.Namespace
) -> Experiment:
    token_pool = normalize_token_pool(args.token_pool)
    return Experiment(
        name=f"{experiment_name(f'top{k:02d}', layers, args)}_seed-{seed}",
        model_type=args.model_type,
        layers=layers,
        token_pool=token_pool,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        seed=seed,
    )


def finite_metric(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def aggregate_single_layer_ranking(
    results: list[dict[str, object]], metric: str
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        layers = result.get("layers")
        if not isinstance(layers, list) or len(layers) != 1:
            continue
        grouped[int(layers[0])].append(result)

    ranking = []
    for layer, layer_results in grouped.items():
        values = [finite_metric(result.get(metric)) for result in layer_results]
        finite_values = [value for value in values if math.isfinite(value)]
        ranking.append(
            {
                "layer": layer,
                "metric": metric,
                "mean": float(np.mean(finite_values)) if finite_values else float("nan"),
                "std": float(np.std(finite_values)) if finite_values else float("nan"),
                "n": len(finite_values),
                "seeds": [int(result["seed"]) for result in layer_results],
                "values": values,
            }
        )

    ranking.sort(
        key=lambda row: (
            not math.isfinite(float(row["mean"])),
            -float(row["mean"]) if math.isfinite(float(row["mean"])) else 0.0,
            int(row["layer"]),
        )
    )
    return ranking


def load_existing_result(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    result = checkpoint.get("result") if isinstance(checkpoint, dict) else None
    return result if isinstance(result, dict) else None


def existing_checkpoint_split(path: Path) -> dict[str, list[int]] | None:
    """Return the split stored in a checkpoint, or None for older checkpoints."""

    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    split = checkpoint.get("split") if isinstance(checkpoint, dict) else None
    return split if isinstance(split, dict) else None


def train_or_load(
    exp: Experiment,
    root: Path,
    split: dict[str, list[int]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = args.output_dir / f"{exp.name}.pt"
    if not args.force:
        existing = load_existing_result(checkpoint_path)
        if existing is not None:
            stored_split = existing_checkpoint_split(checkpoint_path)
            if stored_split is not None and {
                k: list(v) for k, v in stored_split.items()
            } != {k: list(v) for k, v in split.items()}:
                raise RuntimeError(
                    f"{checkpoint_path} was trained with a different split than "
                    f"this invocation computed; refusing to reuse it. Use a "
                    f"fresh --output-dir or rerun with --force."
                )
            print(f"skip existing {exp.name}: checkpoint={checkpoint_path}")
            return existing

    print(
        f"\n== {exp.name}: layers={exp.layers}, token_pool={exp.token_pool}, "
        f"seed={exp.seed}, lr={exp.lr}, lambda_reg={exp.lambda_reg} =="
    )
    try:
        return train_one_experiment(exp, root, split, args, device)
    finally:
        if not args.keep_dataset_cache:
            _DATASET_CACHE.clear()


def write_layer_selection_summary(
    output_dir: Path,
    ranking: list[dict[str, object]],
    topk_results: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    summary = {
        "ranking_metric": args.ranking_metric,
        "ranking": ranking,
        "topk_results": topk_results,
        "config": {
            "token_pool": normalize_token_pool(args.token_pool),
            "lr": args.lr,
            "lambda_reg": args.lambda_reg,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "seeds": list(args.seeds),
            "split_mode": args.split_mode,
            "max_top_k": args.max_top_k,
            "min_top_k": args.min_top_k,
        },
    }
    with (output_dir / "layer_selection_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    ranking_path = output_dir / "single_layer_ranking.csv"
    with ranking_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rank", "layer", "metric", "mean", "std", "n"]
        )
        writer.writeheader()
        for rank, row in enumerate(ranking, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "layer": row["layer"],
                    "metric": row["metric"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "n": row["n"],
                }
            )

    topk_path = output_dir / "topk_concat_results.csv"
    fieldnames = [
        "name",
        "layers",
        "seed",
        "val_falert_early_roc_auc",
        "test_falert_early_roc_auc",
        "test_falert_end_roc_auc",
        "test_length_only_roc_auc",
        "checkpoint_path",
    ]
    with topk_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in topk_results:
            row = {key: result.get(key) for key in fieldnames}
            row["layers"] = " ".join(str(layer) for layer in result["layers"])
            writer.writerow(row)


def build_splits(
    args: argparse.Namespace,
    root: Path,
    saved_layers: tuple[int, ...],
) -> tuple[dict[int, dict[str, list[int]]], dict[int, dict[str, object]]]:
    base_dataset = get_dataset(
        root,
        layers=(saved_layers[-1],),
        token_pool="last",
        cache=not args.no_cache,
    )

    splits: dict[int, dict[str, list[int]]] = {}
    metadata_by_seed: dict[int, dict[str, object]] = {}
    for seed in args.seeds:
        if args.loto:
            # In LOTO mode each "seed" slot is the held-out task id; the
            # train/val shuffle uses the fixed --loto-seed across all folds.
            split, metadata = split_indices_by_task(
                base_dataset,
                unseen_task_ratio=args.unseen_task_ratio,
                seen_train_ratio=args.seen_train_ratio,
                seed=args.loto_seed,
                held_out_tasks=[int(seed)],
            )
            metadata["loto_held_out_task"] = int(seed)
            metadata["loto_seed"] = args.loto_seed
        elif args.split_mode == "task":
            split, metadata = split_indices_by_task(
                base_dataset,
                unseen_task_ratio=args.unseen_task_ratio,
                seen_train_ratio=args.seen_train_ratio,
                seed=seed,
            )
        else:
            labels = labels_for_dataset(base_dataset)
            split = split_indices(labels, args.train_frac, args.val_frac, seed)
            metadata = {
                "split_mode": "random",
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": 1.0 - args.train_frac - args.val_frac,
            }
        metadata["seed"] = seed
        splits[seed] = split
        metadata_by_seed[seed] = metadata

    if not args.keep_dataset_cache:
        _DATASET_CACHE.clear()

    return splits, metadata_by_seed


def write_splits(
    output_dir: Path,
    splits: dict[int, dict[str, list[int]]],
    metadata_by_seed: dict[int, dict[str, object]],
) -> None:
    for seed, split in splits.items():
        suffix = "" if len(splits) == 1 else f"_seed{seed}"
        indices_path = output_dir / f"split_indices{suffix}.json"
        if indices_path.exists():
            with indices_path.open() as handle:
                existing = json.load(handle)
            if {k: list(v) for k, v in existing.items()} != {
                k: list(v) for k, v in split.items()
            }:
                raise RuntimeError(
                    f"{indices_path} already exists with a different split than "
                    f"this invocation computed. The directory contains outputs "
                    f"from an earlier run with different splits; reusing its "
                    f"checkpoints would mix splits silently. Use a fresh "
                    f"--output-dir, or delete the stale checkpoints and split "
                    f"files."
                )
        with indices_path.open("w") as handle:
            json.dump(split, handle, indent=2)
        with (output_dir / f"split_metadata{suffix}.json").open("w") as handle:
            json.dump(metadata_by_seed[seed], handle, indent=2)

    if len(splits) > 1:
        with (output_dir / "split_metadata_by_seed.json").open("w") as handle:
            json.dump(metadata_by_seed, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.token_pool = normalize_token_pool(args.token_pool)
    if args.wandb and args.wandb_group is None:
        args.wandb_group = f"openvla-lstm-layer-selection-{int(time.time())}"

    root = args.root.expanduser().resolve()
    device = torch.device(args.device)
    saved_layers = discover_saved_layers(root)
    layers = tuple(args.layers) if args.layers is not None else tuple(saved_layers)
    missing = [layer for layer in layers if layer not in saved_layers]
    if missing:
        raise ValueError(f"Requested layers {missing} are not in saved layers {saved_layers}")

    if args.loto:
        # One fold per task: relabel the seed dimension as the held-out task id.
        base_dataset = get_dataset(
            root, layers=(saved_layers[-1],), token_pool="last", cache=not args.no_cache
        )
        task_ids = sorted({int(t) for t in task_ids_for_dataset(base_dataset)})
        if not args.keep_dataset_cache:
            _DATASET_CACHE.clear()
        args.seeds = tuple(task_ids)
        args.skip_topk = True
        print(f"LOTO mode: {len(task_ids)} folds, held-out tasks={task_ids}, "
              f"train/val shuffle seed={args.loto_seed}")

    if args.skip_topk:
        # No top-k concat phase: any layer count is fine (incl. a single layer).
        max_top_k = len(layers)
        args.max_top_k = max_top_k
    else:
        max_top_k = args.max_top_k
        if max_top_k is None:
            max_top_k = max(args.min_top_k, math.ceil(len(layers) / 2))
        max_top_k = min(max_top_k, len(layers))
        if args.min_top_k > max_top_k:
            raise ValueError("--min-top-k must be <= resolved --max-top-k")
        args.max_top_k = max_top_k

    splits, split_metadata_by_seed = build_splits(args, root, saved_layers)
    write_splits(args.output_dir, splits, split_metadata_by_seed)

    print(f"root={root}")
    print(f"saved_layers={saved_layers}")
    print(f"swept_layers={layers}")
    print(f"seeds={args.seeds}")
    print(f"ranking_metric={args.ranking_metric}")
    print(f"top_k_range={list(range(args.min_top_k, max_top_k + 1))}")
    print(f"split_metadata_by_seed={split_metadata_by_seed}")
    print(
        f"split_sizes_by_seed={ {seed: {name: len(idx) for name, idx in split.items()} for seed, split in splits.items()} }"
    )
    print(
        f"lstm_defaults: token_pool={args.token_pool} lr={args.lr:g} "
        f"lambda_reg={args.lambda_reg:g} optimizer={args.optimizer} "
        f"epochs={args.epochs} no_early_stopping={args.no_early_stopping}"
    )
    if args.dry_run:
        for seed in args.seeds:
            for exp in single_layer_experiments(layers, seed, args):
                print(f"dry_run single {exp.name}: layers={exp.layers}")
        print(
            "dry_run top-k experiments are resolved after single-layer ranking; "
            f"will train k={args.min_top_k}..{max_top_k}."
        )
        return

    all_results: list[dict[str, object]] = []
    single_results: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_everything(seed)
        for exp in single_layer_experiments(layers, seed, args):
            result = train_or_load(exp, root, splits[seed], args, device)
            single_results.append(result)
            all_results.append(result)
            print(
                f"done {exp.name}: {args.ranking_metric}="
                f"{finite_metric(result.get(args.ranking_metric)):.4f} "
                f"test_falert_early_auc={finite_metric(result.get('test_falert_early_roc_auc')):.4f}"
            )

    ranking = aggregate_single_layer_ranking(single_results, args.ranking_metric)
    if not ranking:
        raise RuntimeError("No single-layer results were available for ranking")
    ranked_layers = tuple(int(row["layer"]) for row in ranking)
    print("\n== single-layer ranking ==")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank:02d}. layer={row['layer']} "
            f"{row['metric']}={float(row['mean']):.4f} +/- {float(row['std']):.4f} "
            f"n={row['n']}"
        )

    topk_results: list[dict[str, object]] = []
    if args.skip_topk:
        write_results(all_results, args.output_dir)
        write_layer_selection_summary(args.output_dir, ranking, topk_results, args)
        print(f"\nskip-topk set; wrote single-layer results to {args.output_dir}")
        return

    for k in range(args.min_top_k, max_top_k + 1):
        top_layers = ranked_layers[:k]
        for seed in args.seeds:
            seed_everything(seed)
            exp = topk_experiment(k, top_layers, seed, args)
            result = train_or_load(exp, root, splits[seed], args, device)
            topk_results.append(result)
            all_results.append(result)
            print(
                f"done {exp.name}: {args.ranking_metric}="
                f"{finite_metric(result.get(args.ranking_metric)):.4f} "
                f"test_falert_early_auc={finite_metric(result.get('test_falert_early_roc_auc')):.4f}"
            )

    write_results(all_results, args.output_dir)
    write_layer_selection_summary(args.output_dir, ranking, topk_results, args)
    print(f"\nwrote results to {args.output_dir}")
    print(f"ranking: {args.output_dir / 'single_layer_ranking.csv'}")
    print(f"top-k:   {args.output_dir / 'topk_concat_results.csv'}")


if __name__ == "__main__":
    main()
