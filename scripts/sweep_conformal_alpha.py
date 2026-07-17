"""Sweep conformal alpha values for trained SAFE/OpenVLA detectors.

This script is seed-aware: checkpoint names ending in ``_seed-N.pt`` are paired
with ``split_indices_seedN.json``. It loads each checkpoint once, collects
validation/test score traces once, then evaluates all requested alpha values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.evaluate_conformal import (
    baseline_metrics,
    build_thresholds,
    collect_scores,
    evaluate_thresholds,
    load_checkpoint,
    load_split,
    make_dataset,
    model_from_checkpoint,
)


SEED_RE = re.compile(r"_seed-(?P<seed>\d+)$")


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    for alpha in alphas:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return sorted(dict.fromkeys(alphas))


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoint:
        paths.extend(path.expanduser() for path in args.checkpoint)
    if args.checkpoint_dir:
        paths.extend(sorted(args.checkpoint_dir.expanduser().glob(args.checkpoint_glob)))
    if not paths:
        raise ValueError("Pass --checkpoint or --checkpoint-dir")
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def seed_from_checkpoint(path: Path) -> int:
    match = SEED_RE.search(path.stem)
    if match is None:
        raise ValueError(
            f"Could not infer seed from {path.name}; expected a name ending in _seed-N.pt"
        )
    return int(match.group("seed"))


def split_path_for_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    seed = seed_from_checkpoint(checkpoint_path)
    split_dir = (
        args.split_dir.expanduser()
        if args.split_dir is not None
        else checkpoint_path.parent
    )
    return split_dir / args.split_template.format(seed=seed)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def finite_std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.std(finite))


def raw_accuracy(metrics: dict[str, float]) -> float:
    n_failed = int(metrics["n_failed"])
    n_success = int(metrics["n_success"])
    n_total = n_failed + n_success
    if n_total == 0:
        return float("nan")
    return float(
        (metrics["tpr"] * n_failed + metrics["tnr"] * n_success) / n_total
    )


def write_thresholds(path: Path, thresholds: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "thresholds.npy", thresholds)
    with (path / "thresholds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep_index", "threshold"])
        writer.writerows((idx, float(value)) for idx, value in enumerate(thresholds))


def evaluate_checkpoint_alphas(
    args: argparse.Namespace,
    checkpoint_path: Path,
    alphas: list[float],
    device: torch.device,
) -> list[dict[str, Any]]:
    seed = seed_from_checkpoint(checkpoint_path)
    split_path = split_path_for_checkpoint(args, checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    layers = tuple(int(layer) for layer in checkpoint["selected_layers"])
    token_pool = str(checkpoint.get("token_pool", "last"))

    dataset = make_dataset(args.root.expanduser().resolve(), layers, token_pool)
    model = model_from_checkpoint(
        checkpoint,
        dataset,
        device,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    )

    # Prefer the split embedded in the checkpoint (the indices the model was
    # actually trained/evaluated with). split_indices_*.json files can go
    # stale when a sweep directory mixes outputs from several invocations.
    if "split" in checkpoint:
        split = checkpoint["split"]
    else:
        split = load_split(split_path)
        print(
            f"warning: {checkpoint_path.name} has no embedded split; "
            f"falling back to {split_path} (verify it matches training)"
        )
    val_scores = collect_scores(
        model, dataset, split["val"], args.batch_size, args.num_workers, device
    )
    test_scores = collect_scores(
        model, dataset, split["test"], args.batch_size, args.num_workers, device
    )
    val_baselines = baseline_metrics(val_scores)
    test_baselines = baseline_metrics(test_scores)
    csv_root = args.csv_root.expanduser().resolve() if args.csv_root else None
    align_extend = args.calibration_method == "functional_cp"

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        thresholds, threshold_info = build_thresholds(
            val_scores, test_scores, alpha, args.calibration_method, args.cp_seed
        )
        val_by_time, _ = evaluate_thresholds(
            val_scores, thresholds, csv_root, align_extend
        )
        test_by_time, _ = evaluate_thresholds(
            test_scores, thresholds, csv_root, align_extend
        )
        val_metrics = val_by_time["by earliest stop"]
        test_metrics = test_by_time["by earliest stop"]

        alpha_tag = f"alpha_{alpha:.3f}".replace(".", "p")
        per_alpha_dir = args.output_dir / alpha_tag / checkpoint_path.stem
        write_thresholds(per_alpha_dir, thresholds)

        summary = {
            "checkpoint": str(checkpoint_path),
            "split_indices": str(split_path),
            "seed": seed,
            "alpha": alpha,
            "calibration_method": args.calibration_method,
            "cp_seed": args.cp_seed,
            "selected_layers": layers,
            "token_pool": token_pool,
            "primary_time": "by earliest stop",
            "threshold_info": threshold_info,
            "threshold_length": int(len(thresholds)),
            "val": val_metrics,
            "test": test_metrics,
            "val_by_time": val_by_time,
            "test_by_time": test_by_time,
            "val_baselines": val_baselines,
            "test_baselines": test_baselines,
        }
        with (per_alpha_dir / "conformal_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)

        row = {
            "checkpoint": checkpoint_path.stem,
            "seed": seed,
            "alpha": alpha,
            "calibration_method": args.calibration_method,
            "threshold_length": int(len(thresholds)),
            "band_width": threshold_info.get("band_width", float("nan")),
            "calibration_success_count": threshold_info.get(
                "calibration_success_count", float("nan")
            ),
            "regression_calibration_count": threshold_info.get(
                "regression_calibration_count", float("nan")
            ),
            "modulation_calibration_count": threshold_info.get(
                "modulation_calibration_count", float("nan")
            ),
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_raw_accuracy": raw_accuracy(val_metrics),
            "val_tpr": val_metrics["tpr"],
            "val_tnr": val_metrics["tnr"],
            "val_fpr": val_metrics["fpr"],
            "val_avg_det_time": val_metrics["avg_det_time"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_raw_accuracy": raw_accuracy(test_metrics),
            "test_tpr": test_metrics["tpr"],
            "test_tnr": test_metrics["tnr"],
            "test_fpr": test_metrics["fpr"],
            "test_any_alarm_rate": test_metrics["any_alarm_rate"],
            "test_mean_first_alarm_fraction": test_metrics[
                "mean_first_alarm_fraction"
            ],
            "test_avg_det_time": test_metrics["avg_det_time"],
            "test_mean_success_false_alarm_fraction": test_metrics[
                "mean_success_false_alarm_fraction"
            ],
            "test_roc_auc_max_score": test_metrics["roc_auc_max_score"],
            "test_length_only_roc_auc": test_baselines["length_only_roc_auc"],
            "test_progress_ratio_roc_auc": test_baselines[
                "progress_ratio_roc_auc"
            ],
            "test_length_leakage_flag": test_baselines["length_leakage_flag"],
        }
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = [
        "test_balanced_accuracy",
        "test_raw_accuracy",
        "test_tpr",
        "test_tnr",
        "test_fpr",
        "test_any_alarm_rate",
        "test_mean_first_alarm_fraction",
        "test_avg_det_time",
        "test_mean_success_false_alarm_fraction",
        "test_roc_auc_max_score",
        "val_balanced_accuracy",
        "val_raw_accuracy",
        "val_tpr",
        "val_tnr",
        "val_fpr",
        "val_avg_det_time",
        "band_width",
    ]
    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_alpha[float(row["alpha"])].append(row)

    out: list[dict[str, Any]] = []
    for alpha, alpha_rows in sorted(by_alpha.items()):
        agg: dict[str, Any] = {
            "alpha": alpha,
            "n_seeds": len(alpha_rows),
            "seeds": ",".join(str(row["seed"]) for row in alpha_rows),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in alpha_rows]
            agg[f"{key}_mean"] = finite_mean(values)
            agg[f"{key}_std"] = finite_std(values)
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(
    output_dir: Path, per_seed_rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]
) -> None:
    fig, (ax_tradeoff, ax_rates) = plt.subplots(1, 2, figsize=(13, 5.2))

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        by_seed[int(row["seed"])].append(row)

    for seed, seed_rows in sorted(by_seed.items()):
        ordered = sorted(seed_rows, key=lambda row: float(row["alpha"]))
        ax_tradeoff.plot(
            [float(row["test_avg_det_time"]) for row in ordered],
            [float(row["test_balanced_accuracy"]) for row in ordered],
            marker="o",
            linewidth=1.0,
            alpha=0.35,
            label=f"seed {seed}",
        )

    mean_x = [float(row["test_avg_det_time_mean"]) for row in aggregate]
    mean_y = [float(row["test_balanced_accuracy_mean"]) for row in aggregate]
    alphas = [float(row["alpha"]) for row in aggregate]
    ax_tradeoff.plot(
        mean_x,
        mean_y,
        marker="o",
        linewidth=2.6,
        color="#111827",
        label="mean",
    )
    best_alpha = float(
        max(aggregate, key=lambda row: float(row["test_balanced_accuracy_mean"]))[
            "alpha"
        ]
    )
    labeled_alphas = {min(alphas), max(alphas), best_alpha}
    if 0.05 in alphas:
        labeled_alphas.add(0.05)

    for x_coord, y_coord, alpha in zip(mean_x, mean_y, alphas):
        if alpha not in labeled_alphas:
            continue
        if math.isfinite(x_coord) and math.isfinite(y_coord):
            ax_tradeoff.annotate(
                f"{alpha:g}",
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax_tradeoff.set_xlabel("Average detection time fraction (lower is earlier)")
    ax_tradeoff.set_ylabel("Balanced accuracy")
    ax_tradeoff.set_title("CP Accuracy vs Detection Time")
    ax_tradeoff.grid(True, alpha=0.25)
    ax_tradeoff.set_xlim(-0.03, 1.03)
    ax_tradeoff.set_ylim(0.45, 1.0)
    ax_tradeoff.legend(frameon=False, fontsize=8)

    ax_rates.plot(
        alphas,
        [float(row["test_tpr_mean"]) for row in aggregate],
        marker="o",
        label="TPR",
    )
    ax_rates.plot(
        alphas,
        [float(row["test_tnr_mean"]) for row in aggregate],
        marker="o",
        label="TNR",
    )
    ax_rates.plot(
        alphas,
        [float(row["test_fpr_mean"]) for row in aggregate],
        marker="o",
        label="FPR",
    )
    ax_rates.set_xlabel("alpha")
    ax_rates.set_ylabel("Rate")
    ax_rates.set_title("Detection and False Alarm Rates")
    ax_rates.grid(True, alpha=0.25)
    ax_rates.set_ylim(-0.03, 1.03)
    ax_rates.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "alpha_tradeoff.png", dpi=180)
    fig.savefig(output_dir / "alpha_tradeoff.svg")
    plt.close(fig)


def format_metric(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "nan"
    return f"{mean:.4f} +/- {std:.4f}"


def write_report(
    output_dir: Path,
    aggregate: list[dict[str, Any]],
    checkpoints: list[Path],
    args: argparse.Namespace,
) -> None:
    best_balanced = max(
        aggregate, key=lambda row: float(row["test_balanced_accuracy_mean"])
    )
    report = [
        "# Conformal Alpha Sweep",
        "",
        f"- Calibration method: `{args.calibration_method}`",
        f"- CP seed: `{args.cp_seed}`",
        f"- Checkpoints: `{len(checkpoints)}`",
        f"- Primary evaluation: `by earliest stop`",
        f"- Best mean balanced accuracy: alpha `{best_balanced['alpha']:g}` "
        f"with `{best_balanced['test_balanced_accuracy_mean']:.4f}`",
        "",
        "![Alpha tradeoff](alpha_tradeoff.png)",
        "",
        "| alpha | balanced acc | raw acc | TPR | TNR | FPR | avg det time | any alarm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            "| "
            f"{float(row['alpha']):g} | "
            f"{format_metric(float(row['test_balanced_accuracy_mean']), float(row['test_balanced_accuracy_std']))} | "
            f"{format_metric(float(row['test_raw_accuracy_mean']), float(row['test_raw_accuracy_std']))} | "
            f"{format_metric(float(row['test_tpr_mean']), float(row['test_tpr_std']))} | "
            f"{format_metric(float(row['test_tnr_mean']), float(row['test_tnr_std']))} | "
            f"{format_metric(float(row['test_fpr_mean']), float(row['test_fpr_std']))} | "
            f"{format_metric(float(row['test_avg_det_time_mean']), float(row['test_avg_det_time_std']))} | "
            f"{format_metric(float(row['test_any_alarm_rate_mean']), float(row['test_any_alarm_rate_std']))} |"
        )
    report.extend(
        [
            "",
            "Notes:",
            "- `avg det time` is averaged over failed rollouts, with missed detections counted at the evaluation cutoff.",
            "- `raw acc` uses the observed held-out success/failure mix; SAFE comparisons usually emphasize balanced accuracy for the CP tradeoff.",
            "- Lower `avg det time` is earlier detection; higher `alpha` generally lowers the threshold and increases both TPR and FPR.",
        ]
    )
    (output_dir / "alpha_tradeoff_report.md").write_text("\n".join(report) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/rollouts/openvla"))
    parser.add_argument(
        "--csv-root", type=Path, default=Path("data/rollouts/openvla_csv")
    )
    parser.add_argument("--checkpoint", type=Path, nargs="*")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-glob", default="*.pt")
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split-template", default="split_indices_seed{seed}.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default="0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5",
        help="Comma-separated conformal alpha values.",
    )
    parser.add_argument(
        "--calibration-method",
        default="functional_cp",
        choices=["functional_cp", "functional_envelope", "pointwise"],
    )
    parser.add_argument("--cp-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = parse_alphas(args.alphas)
    paths = checkpoint_paths(args)
    device = torch.device(args.device)

    all_rows: list[dict[str, Any]] = []
    for path in paths:
        split_path = split_path_for_checkpoint(args, path)
        print(f"Evaluating {path.name} with {split_path.name}")
        checkpoint_rows = evaluate_checkpoint_alphas(args, path, alphas, device)
        all_rows.extend(checkpoint_rows)
        best_row = max(
            checkpoint_rows, key=lambda row: float(row["test_balanced_accuracy"])
        )
        print(
            "  best alpha="
            f"{best_row['alpha']:g} "
            f"bal_acc={best_row['test_balanced_accuracy']:.4f} "
            f"tpr={best_row['test_tpr']:.4f} "
            f"tnr={best_row['test_tnr']:.4f} "
            f"avg_det_time={best_row['test_avg_det_time']:.4f}"
        )

    aggregate = aggregate_rows(all_rows)
    write_csv(args.output_dir / "alpha_sweep_per_seed.csv", all_rows)
    write_csv(args.output_dir / "alpha_sweep_summary.csv", aggregate)
    with (args.output_dir / "alpha_sweep_per_seed.json").open("w") as handle:
        json.dump(all_rows, handle, indent=2)
    with (args.output_dir / "alpha_sweep_summary.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2)

    plot_tradeoff(args.output_dir, all_rows, aggregate)
    write_report(args.output_dir, aggregate, paths, args)
    print(f"Wrote alpha sweep artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
