"""Evaluate dynamic layer LOTO checkpoints with SAFE functional conformal CP."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import DynamicLayerMixLSTMModel, ResidualAuxLayerLSTMModel
from scripts.evaluate_conformal import (
    RolloutScores,
    evaluate_thresholds,
    functional_cp_thresholds,
)
from scripts.evaluate_dynamic_layer_loto import (
    EarlyLayerView,
    parse_csv_ints,
    shuffled_donors,
    task_id,
)
from scripts.evaluate_functional_cp_loto import (
    operating_points,
    summarize_macro,
    summarize_per_task,
)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-layers", type=parse_csv_ints, default=(20, 32))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", default="0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3"
    )
    parser.add_argument("--cp-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--horizon", type=int, default=130)
    parser.add_argument(
        "--fixed-horizons",
        default="",
        help=(
            "Comma-separated common horizons in processed-sample units. Every "
            "validation and test trajectory must reach each horizon."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--task-bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    return parser.parse_args()


def result_row(
    variant: str,
    fold: int,
    cp_seed: int,
    alpha: float,
    eval_time: str,
    validation: list[RolloutScores],
    info: dict,
    metrics: dict,
) -> dict:
    caught_alarm = metrics["mean_failed_first_alarm_fraction"]
    effective_alarm = metrics["avg_det_time"]
    return {
        "model": variant,
        "fold": fold,
        "heldout_task": fold,
        "cp_seed": cp_seed,
        "alpha": alpha,
        "eval_time": eval_time,
        "n_validation_success": sum(record.success for record in validation),
        "regression_calibration_count": info["regression_calibration_count"],
        "modulation_calibration_count": info["modulation_calibration_count"],
        "band_width": info["band_width"],
        "realized_fpr": metrics["fpr"],
        "catch_rate": metrics["tpr"],
        "tnr": metrics["tnr"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "caught_alarm_fraction": caught_alarm,
        "effective_alarm_fraction": effective_alarm,
        "caught_lead_fraction": (
            1.0 - caught_alarm if np.isfinite(caught_alarm) else np.nan
        ),
        "effective_lead_fraction": 1.0 - effective_alarm,
        "success_false_alarm_fraction": metrics[
            "mean_success_false_alarm_fraction"
        ],
        "any_alarm_rate": metrics["any_alarm_rate"],
        "roc_auc_max_score": metrics["roc_auc_max_score"],
    }


@torch.no_grad()
def collect_scores(
    model: torch.nn.Module,
    dataset: EarlyLayerView,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> list[RolloutScores]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rollouts,
    )
    records = []
    for raw_batch in loader:
        features = raw_batch["features"].to(device)
        scores = model({"features": features}).squeeze(-1).detach().cpu().numpy()
        for row, path in enumerate(raw_batch["paths"]):
            length = int(raw_batch["lengths"][row])
            records.append(
                RolloutScores(
                    path=path,
                    task_id=int(raw_batch["task_ids"][row]),
                    episode_idx=int(raw_batch["episode_indices"][row]),
                    label=int(raw_batch["labels"][row]),
                    success=bool(raw_batch["success"][row]),
                    length=length,
                    task_min_step=length,
                    scores=scores[row, :length].astype(np.float64),
                )
            )
    return records


def load_fold(
    path: Path,
    base: OpenVLARolloutDataset,
    device: torch.device,
) -> tuple[torch.nn.Module, EarlyLayerView, dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result = checkpoint["result"]
    config = checkpoint["config"]
    variant = result["variant"]
    fold = int(result["fold"])
    split = {
        name: [int(index) for index in indices]
        for name, indices in checkpoint["split"].items()
    }
    donors = (
        shuffled_donors(base, split, int(config["seed"]) + 104729 * fold)
        if variant.startswith("shuffled")
        else None
    )
    dataset = EarlyLayerView(base, variant, donors)
    n_layers = len(result["selected_layers"])
    if "residual" in variant:
        model = ResidualAuxLayerLSTMModel(
            hidden_dim_per_layer=base.hidden_dim,
            base_projection_dim=int(config["projection_dim"]),
            base_lstm_hidden_dim=int(config["hidden_dim"]),
            aux_projection_dim=max(int(config["projection_dim"]) // 2, 1),
            aux_lstm_hidden_dim=max(int(config["hidden_dim"]) // 2, 1),
            dropout=0.0,
        )
        model.freeze_base()
    else:
        model = DynamicLayerMixLSTMModel(
            input_dim=n_layers * base.hidden_dim,
            n_layers=n_layers,
            hidden_dim_per_layer=base.hidden_dim,
            projection_dim=int(config["projection_dim"]),
            lstm_hidden_dim=int(config["hidden_dim"]),
            dropout=0.0,
            loss_type="bce",
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), dataset, split, result


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)
    cp_seeds = parse_int_list(args.cp_seeds)
    fixed_horizons = parse_int_list(args.fixed_horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base = OpenVLARolloutDataset(
        args.root.expanduser().resolve(),
        layers=tuple(args.dataset_layers),
        token_pool="last",
        recursive=True,
        cache=True,
    )

    rows = []
    checkpoint_manifest = {}
    paths = sorted(args.checkpoint_dir.glob("fold_*_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no fold checkpoints under {args.checkpoint_dir}")
    for path in paths:
        model, dataset, split, result = load_fold(path, base, device)
        variant = result["variant"]
        fold = int(result["fold"])
        validation = collect_scores(
            model, dataset, split["val"], args.batch_size, device
        )
        test = collect_scores(model, dataset, split["test"], args.batch_size, device)
        heldout_tasks = sorted({int(record.task_id) for record in test})
        if heldout_tasks != [fold]:
            raise ValueError(f"fold {fold} test tasks are {heldout_tasks}")
        minimum_length = min(record.length for record in validation + test)
        invalid_horizons = [
            fixed_horizon
            for fixed_horizon in fixed_horizons
            if fixed_horizon > minimum_length
        ]
        if invalid_horizons:
            raise ValueError(
                f"fold {fold} fixed horizons {invalid_horizons} exceed minimum "
                f"validation/test length {minimum_length}"
            )
        checkpoint_manifest[f"{fold}:{variant}"] = str(path)
        for cp_seed in cp_seeds:
            for alpha in alphas:
                thresholds, info = functional_cp_thresholds(
                    validation, test, alpha, cp_seed, horizon=args.horizon
                )
                metrics = evaluate_thresholds(
                    test, thresholds, csv_root=None, align_extend=True
                )[0]["by earliest stop"]
                rows.append(
                    result_row(
                        variant,
                        fold,
                        cp_seed,
                        alpha,
                        "by earliest stop",
                        validation,
                        info,
                        metrics,
                    )
                )
                for fixed_horizon in fixed_horizons:
                    fixed_thresholds, fixed_info = functional_cp_thresholds(
                        validation,
                        test,
                        alpha,
                        cp_seed,
                        horizon=fixed_horizon,
                    )
                    fixed_test = [
                        replace(
                            record,
                            length=fixed_horizon,
                            task_min_step=fixed_horizon,
                            scores=record.scores[:fixed_horizon],
                        )
                        for record in test
                    ]
                    fixed_metrics = evaluate_thresholds(
                        fixed_test,
                        fixed_thresholds,
                        csv_root=None,
                        align_extend=False,
                    )[0]["by final end"]
                    rows.append(
                        result_row(
                            variant,
                            fold,
                            cp_seed,
                            alpha,
                            f"fixed horizon {fixed_horizon}",
                            validation,
                            fixed_info,
                            fixed_metrics,
                        )
                    )
        print(f"evaluated fold={fold} variant={variant}", flush=True)

    raw = pd.DataFrame(rows)
    per_task = summarize_per_task(raw)
    macro = summarize_macro(per_task, args.bootstrap_seed, args.task_bootstrap)
    points = operating_points(macro)
    raw.to_csv(args.output_dir / "functional_cp_raw.csv", index=False)
    per_task.to_csv(args.output_dir / "functional_cp_per_task.csv", index=False)
    macro.to_csv(args.output_dir / "functional_cp_macro.csv", index=False)
    points.to_csv(args.output_dir / "functional_cp_operating_points.csv", index=False)
    fixed_mask = raw["eval_time"].str.startswith("fixed horizon ")
    if fixed_mask.any():
        raw[fixed_mask].to_csv(
            args.output_dir / "functional_cp_fixed_horizon_raw.csv", index=False
        )
        per_task[per_task["eval_time"].str.startswith("fixed horizon ")].to_csv(
            args.output_dir / "functional_cp_fixed_horizon_per_task.csv", index=False
        )
        macro[macro["eval_time"].str.startswith("fixed horizon ")].to_csv(
            args.output_dir / "functional_cp_fixed_horizon_macro.csv", index=False
        )
        points[points["eval_time"].str.startswith("fixed horizon ")].to_csv(
            args.output_dir / "functional_cp_fixed_horizon_operating_points.csv",
            index=False,
        )

    fpr5 = points[
        (points["constraint_type"] == "realized_fpr_at_most")
        & (points["constraint_value"] == 0.05)
    ].sort_values("model")
    summary = {
        "n_rollouts": len(base),
        "tasks": sorted({task_id(base, index) for index in range(len(base))}),
        "horizon": args.horizon,
        "fixed_horizons": fixed_horizons,
        "alphas": alphas,
        "cp_seeds": cp_seeds,
        "fpr_at_most_5_percent": fpr5.to_dict(orient="records"),
        "checkpoint_manifest": checkpoint_manifest,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print("\nFPR <= 5% operating points")
    print(
        fpr5[
            [
                "model",
                "eval_time",
                "constraint_met",
                "alpha",
                "realized_fpr_mean",
                "catch_rate_mean",
                "effective_lead_fraction_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
