"""Evaluate SAFE-style conformal thresholds for trained OpenVLA detectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from scripts.train_openvla_ablation import Experiment, build_model


@dataclass
class RolloutScores:
    path: str
    task_id: int | None
    episode_idx: int | None
    label: int
    success: bool
    length: int
    task_min_step: int
    scores: np.ndarray


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Finite-sample upper conformal quantile with conservative rounding."""

    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values)
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), values.size - 1)
    return float(sorted_values[rank])


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoint:
        paths.extend(path.expanduser() for path in args.checkpoint)
    if args.checkpoint_dir:
        paths.extend(sorted(args.checkpoint_dir.expanduser().glob("*.pt")))
    if not paths:
        raise ValueError("Pass --checkpoint or --checkpoint-dir")
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def load_split(path: Path) -> dict[str, list[int]]:
    with path.expanduser().open() as handle:
        split = json.load(handle)
    return {name: [int(idx) for idx in indices] for name, indices in split.items()}


def make_dataset(
    root: Path, layers: tuple[int, ...], token_pool: str
) -> OpenVLARolloutDataset:
    return OpenVLARolloutDataset(
        root, layers=layers, token_pool=token_pool, recursive=True
    )


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    dataset: OpenVLARolloutDataset,
    device: torch.device,
    hidden_dim: int,
    projection_dim: int,
    dropout: float,
) -> torch.nn.Module:
    exp_payload = checkpoint["experiment"]
    exp = Experiment(
        name=str(exp_payload["name"]),
        model_type=str(exp_payload["model_type"]),
        layers=tuple(int(layer) for layer in exp_payload["layers"]),
        token_pool=str(exp_payload.get("token_pool", checkpoint.get("token_pool", "last"))),
        lr=exp_payload.get("lr", checkpoint.get("lr")),
        lambda_reg=exp_payload.get("lambda_reg", checkpoint.get("lambda_reg")),
        seed=exp_payload.get("seed"),
        n_history_steps=int(exp_payload.get("n_history_steps", checkpoint.get("n_history_steps", 1))),
    )
    model = build_model(
        exp=exp,
        dataset=dataset,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        dropout=dropout,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_scores(
    model: torch.nn.Module,
    dataset: OpenVLARolloutDataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[RolloutScores]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_rollouts,
    )
    out: list[RolloutScores] = []
    for batch in loader:
        features = batch["features"].to(device)
        valid_masks = batch["valid_masks"].to(device)
        scores = (
            model({"features": features, "valid_masks": valid_masks})
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        masks = batch["valid_masks"].cpu().numpy().astype(bool)
        labels = batch["labels"].cpu().numpy().astype(int)
        success = batch["success"].cpu().numpy().astype(bool)
        task_min_steps = batch["task_min_steps"].cpu().numpy().astype(int)

        for row in range(scores.shape[0]):
            length = int(masks[row].sum())
            out.append(
                RolloutScores(
                    path=batch["paths"][row],
                    task_id=batch["task_ids"][row],
                    episode_idx=batch["episode_indices"][row],
                    label=int(labels[row]),
                    success=bool(success[row]),
                    length=length,
                    task_min_step=int(min(task_min_steps[row], length)),
                    scores=scores[row, :length].astype(np.float64),
                )
            )
    return out


def pointwise_thresholds(
    success_rollouts: list[RolloutScores], alpha: float
) -> np.ndarray:
    max_len = max(rollout.length for rollout in success_rollouts)
    thresholds = np.empty(max_len, dtype=np.float64)
    for timestep in range(max_len):
        values = np.asarray(
            [
                rollout.scores[timestep]
                for rollout in success_rollouts
                if rollout.length > timestep
            ],
            dtype=np.float64,
        )
        thresholds[timestep] = conformal_quantile(values, alpha)
    return thresholds


def functional_envelope_thresholds(
    success_rollouts: list[RolloutScores], alpha: float
) -> np.ndarray:
    """Upper trajectory band that drops the most extreme calibration curves.

    This controls rollout-level false alarms more directly than independent
    pointwise quantiles: a successful trajectory alarms only if it exits the
    calibrated score envelope at any timestep.
    """

    max_len = max(rollout.length for rollout in success_rollouts)
    n_success = len(success_rollouts)
    keep_count = int(np.ceil((n_success + 1) * (1.0 - alpha)))
    keep_count = min(max(keep_count, 1), n_success)

    ranked = sorted(success_rollouts, key=lambda rollout: float(np.max(rollout.scores)))
    kept = ranked[:keep_count]

    thresholds = np.empty(max_len, dtype=np.float64)
    for timestep in range(max_len):
        values = np.asarray(
            [rollout.scores[timestep] for rollout in kept if rollout.length > timestep],
            dtype=np.float64,
        )
        if values.size == 0:
            thresholds[timestep] = thresholds[timestep - 1] if timestep > 0 else 0.0
        else:
            thresholds[timestep] = float(np.max(values))
    return thresholds


def extend_score(score: np.ndarray, length: int) -> np.ndarray:
    if score.size == 0:
        raise ValueError("Cannot extend an empty score curve")
    if score.size >= length:
        return score[:length].astype(np.float64, copy=False)
    return np.pad(score, (0, length - score.size), mode="edge").astype(np.float64)


def extend_scores(rollouts: list[RolloutScores], length: int) -> np.ndarray:
    return np.asarray([extend_score(rollout.scores, length) for rollout in rollouts])


def tfunc_modulation(
    training_data: np.ndarray, prediction_trajectory: np.ndarray, alpha: float
) -> np.ndarray:
    """SAFE FunctionalPredictor Tfunc modulation."""

    eps = 1e-8
    train_size = training_data.shape[0]
    deviations = np.abs(training_data - prediction_trajectory)
    rank = int(np.ceil((train_size + 1) * (1.0 - alpha)))
    if rank > train_size:
        return np.max(deviations, axis=0, keepdims=True) + eps

    max_deviations = np.max(deviations, axis=1)
    gamma = np.sort(max_deviations)[rank - 1]
    return np.max(deviations[max_deviations <= gamma], axis=0, keepdims=True) + eps


def functional_cp_thresholds(
    calibration: list[RolloutScores],
    test_rollouts: list[RolloutScores],
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """SAFE functional CP upper band calibrated on successful rollouts.

    This mirrors SAFE's `eval_functional_conformal`: align variable-length score
    curves by extending the last value, use successful calibration rollouts,
    split those curves 30/70 for mean-regression/modulation, then build a
    one-sided upper prediction band with Tfunc modulation.
    """

    success_rollouts = [rollout for rollout in calibration if rollout.success]
    if not success_rollouts:
        raise ValueError(
            "Functional CP needs at least one successful calibration rollout"
        )

    max_length = max(rollout.length for rollout in calibration + test_rollouts)
    cal_success = extend_scores(success_rollouts, max_length)
    if len(cal_success) == 1:
        cal_scores_1 = cal_success
        cal_scores_2 = cal_success
    else:
        rng = np.random.default_rng(seed)
        shuffled = cal_success[rng.permutation(len(cal_success))]
        n_cal_1 = int(len(shuffled) * 0.3)
        n_cal_1 = min(max(n_cal_1, 1), len(shuffled) - 1)
        cal_scores_1 = shuffled[:n_cal_1]
        cal_scores_2 = shuffled[n_cal_1:]

    prediction_trajectory = np.mean(cal_scores_1, axis=0, keepdims=True)
    modulation_trajectory = tfunc_modulation(
        cal_scores_1, prediction_trajectory, alpha
    )
    calibration_scores = np.max(
        (cal_scores_2 - prediction_trajectory) / modulation_trajectory, axis=1
    )
    band_width = float(np.quantile(calibration_scores, 1.0 - alpha))
    thresholds = (
        prediction_trajectory + band_width * modulation_trajectory
    ).reshape(-1)
    return thresholds, {
        "calibration_method": "functional_cp",
        "alignment": "extend",
        "calibration_success_count": int(len(success_rollouts)),
        "regression_calibration_count": int(len(cal_scores_1)),
        "modulation_calibration_count": int(len(cal_scores_2)),
        "threshold_length": int(len(thresholds)),
        "band_width": band_width,
    }


def build_thresholds(
    calibration: list[RolloutScores],
    test_rollouts: list[RolloutScores],
    alpha: float,
    method: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    success_rollouts = [rollout for rollout in calibration if rollout.success]
    if not success_rollouts:
        raise ValueError(
            "Conformal calibration needs at least one successful calibration rollout"
        )

    if method == "functional_cp":
        return functional_cp_thresholds(calibration, test_rollouts, alpha, seed)
    if method == "pointwise":
        thresholds = pointwise_thresholds(success_rollouts, alpha)
        return thresholds, {
            "calibration_method": "pointwise",
            "threshold_length": int(len(thresholds)),
            "calibration_success_count": int(len(success_rollouts)),
        }
    if method == "functional_envelope":
        thresholds = functional_envelope_thresholds(success_rollouts, alpha)
        return thresholds, {
            "calibration_method": "functional_envelope",
            "threshold_length": int(len(thresholds)),
            "calibration_success_count": int(len(success_rollouts)),
        }
    raise ValueError(f"Unknown calibration method: {method}")


def threshold_for_length(thresholds: np.ndarray, length: int) -> np.ndarray:
    if length <= len(thresholds):
        return thresholds[:length]
    tail = np.full(length - len(thresholds), thresholds[-1], dtype=np.float64)
    return np.concatenate([thresholds, tail])


def first_alarm(scores: np.ndarray, thresholds: np.ndarray) -> int | None:
    active = scores >= threshold_for_length(thresholds, len(scores))
    alarm_indices = np.flatnonzero(active)
    if alarm_indices.size == 0:
        return None
    return int(alarm_indices[0])


def scores_for_eval_time(
    rollout: RolloutScores, thresholds: np.ndarray, eval_time: str, align_extend: bool
) -> tuple[np.ndarray, int]:
    if eval_time == "by final end":
        if align_extend:
            scores = extend_score(rollout.scores, len(thresholds))
            return scores, len(scores)
        return rollout.scores, rollout.length
    if eval_time == "by earliest stop":
        if align_extend:
            scores = extend_score(rollout.scores, len(thresholds))
        else:
            scores = rollout.scores
        stop = min(rollout.task_min_step, len(scores))
        return scores[:stop], stop
    raise ValueError(f"Unknown eval_time={eval_time!r}")


def csv_timestep(
    csv_root: Path | None, rollout_path: str, index: int | None
) -> int | None:
    if csv_root is None or index is None:
        return None
    path = csv_root / (Path(rollout_path).stem + ".csv")
    if not path.exists():
        return None
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            if row_idx == index:
                try:
                    return int(float(row["action/timestep"]))
                except (KeyError, ValueError):
                    return None
    return None


def baseline_metrics(rollouts: list[RolloutScores]) -> dict[str, float | int | bool]:
    labels = [rollout.label for rollout in rollouts]

    def auc_for(values: list[float]) -> float:
        if len(set(labels)) <= 1:
            return float("nan")
        return float(roc_auc_score(labels, values))

    lengths = [float(rollout.length) for rollout in rollouts]
    progress_ratio = [
        float(rollout.length / max(rollout.task_min_step, 1)) for rollout in rollouts
    ]
    task_min_steps = [float(rollout.task_min_step) for rollout in rollouts]
    length_auc = auc_for(lengths)
    return {
        "n_rollouts": len(rollouts),
        "n_failed": int(sum(labels)),
        "n_success": int(len(labels) - sum(labels)),
        "length_only_roc_auc": length_auc,
        "progress_ratio_roc_auc": auc_for(progress_ratio),
        "task_min_step_roc_auc": auc_for(task_min_steps),
        "length_leakage_flag": bool(np.isfinite(length_auc) and length_auc >= 0.99),
    }


def evaluate_thresholds_for_time(
    rollouts: list[RolloutScores],
    thresholds: np.ndarray,
    csv_root: Path | None,
    eval_time: str,
    align_extend: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    labels = []
    rollout_scores = []
    detections = []
    alarm_fractions = []
    failed_alarm_fractions = []
    success_alarm_fractions = []
    failed_detection_time_fractions = []

    for rollout in rollouts:
        eval_scores, eval_length = scores_for_eval_time(
            rollout, thresholds, eval_time, align_extend
        )
        alarm_idx = first_alarm(eval_scores, thresholds)
        detected = alarm_idx is not None
        alarm_fraction = (
            None if alarm_idx is None else alarm_idx / max(eval_length, 1)
        )
        alarm_timestep = csv_timestep(csv_root, rollout.path, alarm_idx)

        labels.append(rollout.label)
        rollout_scores.append(float(np.max(eval_scores)))
        detections.append(detected)
        if alarm_fraction is not None:
            alarm_fractions.append(float(alarm_fraction))
            if rollout.label == 1:
                failed_alarm_fractions.append(float(alarm_fraction))
            else:
                success_alarm_fractions.append(float(alarm_fraction))
        if rollout.label == 1:
            detection_index = alarm_idx if alarm_idx is not None else eval_length
            failed_detection_time_fractions.append(
                float(detection_index / max(eval_length, 1))
            )

        rows.append(
            {
                "path": rollout.path,
                "task_id": rollout.task_id,
                "episode_idx": rollout.episode_idx,
                "label": rollout.label,
                "success": rollout.success,
                "length": rollout.length,
                "task_min_step": rollout.task_min_step,
                "eval_time": eval_time,
                "eval_length": eval_length,
                "max_score": float(np.max(eval_scores)),
                "detected": detected,
                "first_alarm_index": alarm_idx,
                "first_alarm_fraction": alarm_fraction,
                "first_alarm_action_timestep": alarm_timestep,
            }
        )

    labels_array = np.asarray(labels, dtype=int)
    detections_array = np.asarray(detections, dtype=bool)
    failed_mask = labels_array == 1
    success_mask = labels_array == 0

    tpr = (
        float(detections_array[failed_mask].mean())
        if failed_mask.any()
        else float("nan")
    )
    tnr = (
        float((~detections_array[success_mask]).mean())
        if success_mask.any()
        else float("nan")
    )
    fpr = (
        float(detections_array[success_mask].mean())
        if success_mask.any()
        else float("nan")
    )
    balanced_accuracy = float(np.nanmean([tpr, tnr]))
    roc_auc = (
        float(roc_auc_score(labels, rollout_scores))
        if len(set(labels)) > 1
        else float("nan")
    )

    metrics = {
        "n_rollouts": len(rollouts),
        "n_failed": int(failed_mask.sum()),
        "n_success": int(success_mask.sum()),
        "roc_auc_max_score": roc_auc,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "any_alarm_rate": float(detections_array.mean())
        if len(detections_array)
        else float("nan"),
        "mean_first_alarm_fraction": float(np.mean(alarm_fractions))
        if alarm_fractions
        else float("nan"),
        "mean_failed_first_alarm_fraction": (
            float(np.mean(failed_alarm_fractions))
            if failed_alarm_fractions
            else float("nan")
        ),
        "avg_det_time": (
            float(np.mean(failed_detection_time_fractions))
            if failed_detection_time_fractions
            else float("nan")
        ),
        "mean_success_false_alarm_fraction": (
            float(np.mean(success_alarm_fractions))
            if success_alarm_fractions
            else float("nan")
        ),
    }
    return metrics, rows


def evaluate_thresholds(
    rollouts: list[RolloutScores],
    thresholds: np.ndarray,
    csv_root: Path | None,
    align_extend: bool,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    by_time: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for eval_time in ("by final end", "by earliest stop"):
        metrics, time_rows = evaluate_thresholds_for_time(
            rollouts, thresholds, csv_root, eval_time, align_extend
        )
        by_time[eval_time] = metrics
        rows.extend(time_rows)
    return by_time, rows


def write_rollout_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    args: argparse.Namespace, checkpoint_path: Path
) -> dict[str, Any]:
    device = torch.device(args.device)
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

    split = load_split(args.split_indices)
    val_scores = collect_scores(
        model, dataset, split["val"], args.batch_size, args.num_workers, device
    )
    test_scores = collect_scores(
        model, dataset, split["test"], args.batch_size, args.num_workers, device
    )
    thresholds, threshold_info = build_thresholds(
        val_scores, test_scores, args.alpha, args.calibration_method, args.cp_seed
    )
    align_extend = args.calibration_method == "functional_cp"

    csv_root = args.csv_root.expanduser().resolve() if args.csv_root else None
    val_by_time, val_rows = evaluate_thresholds(
        val_scores, thresholds, csv_root, align_extend
    )
    test_by_time, test_rows = evaluate_thresholds(
        test_scores, thresholds, csv_root, align_extend
    )
    val_metrics = val_by_time["by earliest stop"]
    test_metrics = test_by_time["by earliest stop"]

    out_dir = args.output_dir.expanduser().resolve() / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "thresholds.npy", thresholds)
    with (out_dir / "thresholds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep_index", "threshold"])
        writer.writerows((idx, float(value)) for idx, value in enumerate(thresholds))
    write_rollout_rows(out_dir / "val_rollout_conformal.csv", val_rows)
    write_rollout_rows(out_dir / "test_rollout_conformal.csv", test_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "alpha": args.alpha,
        "calibration_method": args.calibration_method,
        "cp_seed": args.cp_seed,
        "selected_layers": layers,
        "primary_time": "by earliest stop",
        "threshold_info": threshold_info,
        "threshold_length": int(len(thresholds)),
        "n_calibration_success": int(sum(rollout.success for rollout in val_scores)),
        "val": val_metrics,
        "test": test_metrics,
        "val_by_time": val_by_time,
        "test_by_time": test_by_time,
        "val_baselines": baseline_metrics(val_scores),
        "test_baselines": baseline_metrics(test_scores),
    }
    with (out_dir / "conformal_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/rollouts/openvla"))
    parser.add_argument(
        "--csv-root", type=Path, default=Path("data/rollouts/openvla_csv")
    )
    parser.add_argument(
        "--checkpoint", type=Path, nargs="*", help="Specific checkpoint(s) to evaluate."
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Directory of .pt checkpoints to evaluate."
    )
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/conformal_eval"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--calibration-method",
        default="functional_cp",
        choices=["functional_cp", "functional_envelope", "pointwise"],
        help=(
            "functional_cp follows SAFE: extend-align curves, calibrate on successes, "
            "use a mean/Tfunc one-sided functional band. The older local "
            "functional_envelope and pointwise modes are kept for comparison."
        ),
    )
    parser.add_argument(
        "--cp-seed",
        type=int,
        default=0,
        help="Seed for SAFE functional CP's 30/70 successful-calibration split.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in checkpoint_paths(args):
        print(f"Evaluating {path}")
        summaries.append(evaluate_checkpoint(args, path))
        test = summaries[-1]["test"]
        print(
            f"  test balanced_accuracy={test['balanced_accuracy']:.4f} "
            f"tpr={test['tpr']:.4f} tnr={test['tnr']:.4f} "
            f"mean_alarm_frac={test['mean_first_alarm_fraction']:.4f}"
        )

    with (args.output_dir / "conformal_summaries.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)


if __name__ == "__main__":
    main()
