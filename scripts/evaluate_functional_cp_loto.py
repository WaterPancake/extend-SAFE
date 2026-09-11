"""Faithful SAFE functional-CP detection-time evaluation under LOTO.

This evaluator is specialized for the finalized layer-32 OpenVLA audit:

* ten leave-one-task-out folds per model family;
* successful seen-task validation rollouts calibrate SAFE's time-varying
  mean/Tfunc functional conformal band;
* held-out tasks are evaluated at both SAFE's earliest-stop cap and the full
  rollout horizon;
* multiple functional-CP 30/70 calibration splits quantify CP-seed variation;
* model trajectories are scored once and reused for every alpha and CP seed.

The ordered rollout roots may mix token-preserving 3-D artifacts and 2-D
last-token-preprocessed artifacts. Source data are never rewritten.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.dataloaders import collate_rollouts
from scripts.evaluate_conformal import (
    RolloutScores,
    evaluate_thresholds,
    functional_cp_thresholds,
)
from scripts.score_layer_pooled_macro import MixedSchemaLayerDataset, build_probe
from scripts.score_bundle import load_score_bundle, sha256_file, write_score_bundle


SEED_RE = re.compile(r"_seed-(\d+)\.pt$")
SUMMARY_METRICS = [
    "roc_auc_max_score",
    "realized_fpr",
    "catch_rate",
    "balanced_accuracy",
    "caught_alarm_fraction",
    "effective_alarm_fraction",
    "caught_lead_fraction",
    "effective_lead_fraction",
]


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise ValueError("alpha values must be comma-separated numbers in (0, 1)")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("CP seeds must not be empty")
    return values


def parse_optional_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    values = sorted(set(parse_int_list(value)))
    if any(item <= 0 for item in values):
        raise ValueError("fixed horizons must be positive integers")
    return values


def checkpoint_map(directory: Path) -> dict[int, Path]:
    out = {}
    for path in sorted(directory.glob("*.pt")):
        match = SEED_RE.search(path.name)
        if match is not None:
            out[int(match.group(1))] = path
    expected = set(range(10))
    if set(out) != expected:
        raise ValueError(
            f"{directory} must contain finalized folds 0..9; found {sorted(out)}"
        )
    return out


def load_family(
    model_type: str, checkpoint_dir: Path, device: torch.device
) -> tuple[dict[int, torch.nn.Module], dict[int, dict[str, list[int]]], dict[int, Path]]:
    paths = checkpoint_map(checkpoint_dir)
    models = {}
    splits = {}
    for fold, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_probe(model_type)
        model.load_state_dict(checkpoint["model_state_dict"])
        models[fold] = model.to(device).eval()
        split = checkpoint.get("split")
        if split is None:
            raise ValueError(f"{path} has no embedded split")
        splits[fold] = {
            name: [int(index) for index in indices]
            for name, indices in split.items()
        }
    return models, splits, paths


@torch.no_grad()
def score_family(
    model_type: str,
    checkpoint_dir: Path,
    dataset: MixedSchemaLayerDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[
    dict[int, list[RolloutScores]],
    dict[int, dict[str, list[int]]],
    dict[int, Path],
]:
    models, splits, paths = load_family(model_type, checkpoint_dir, device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rollouts,
    )
    records: dict[int, list[RolloutScores | None]] = {
        fold: [None] * len(dataset) for fold in models
    }
    offset = 0
    for batch in loader:
        features = batch["features"].to(device)
        masks = batch["valid_masks"].to(device)
        batch_size_actual = len(batch["paths"])
        for fold, model in models.items():
            scores = model({"features": features}).squeeze(-1)
            scores = scores.detach().cpu().numpy()
            for row in range(batch_size_actual):
                length = int(batch["lengths"][row])
                records[fold][offset + row] = RolloutScores(
                    path=batch["paths"][row],
                    task_id=int(batch["task_ids"][row]),
                    episode_idx=int(batch["episode_indices"][row]),
                    label=int(batch["labels"][row]),
                    success=bool(batch["success"][row]),
                    length=length,
                    task_min_step=int(batch["task_min_steps"][row]),
                    scores=scores[row, :length].astype(np.float64),
                )
        offset += batch_size_actual
        if offset % 100 == 0 or offset == len(dataset):
            print(f"{model_type}: scored {offset}/{len(dataset)} rollouts", flush=True)

    del models
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    finalized: dict[int, list[RolloutScores]] = {}
    for fold, fold_records in records.items():
        if any(record is None for record in fold_records):
            raise AssertionError(f"missing scored rollouts for {model_type} fold {fold}")
        finalized[fold] = [record for record in fold_records if record is not None]
    return finalized, splits, paths


def evaluate_family(
    model_type: str,
    records: dict[
        int, Sequence[RolloutScores] | Mapping[int, RolloutScores]
    ],
    splits: dict[int, dict[str, list[int]]],
    alphas: list[float],
    cp_seeds: list[int],
    horizon: int,
    fixed_horizons: list[int] | None = None,
) -> pd.DataFrame:
    fixed_horizons = fixed_horizons or []
    rows = []
    for fold in range(10):
        validation = [records[fold][index] for index in splits[fold]["val"]]
        test = [records[fold][index] for index in splits[fold]["test"]]
        heldout_tasks = sorted({int(record.task_id) for record in test})
        if len(heldout_tasks) != 1:
            raise ValueError(
                f"{model_type} fold {fold} is not LOTO: tasks={heldout_tasks}"
            )
        heldout_task = heldout_tasks[0]
        if heldout_task != fold:
            raise ValueError(
                f"{model_type} fold {fold}: expected held-out task {fold}, "
                f"found task {heldout_task}"
            )
        validation_tasks = {int(record.task_id) for record in validation}
        if heldout_task in validation_tasks:
            raise ValueError(
                f"{model_type} fold {fold}: held-out task {heldout_task} leaks "
                "into the validation/calibration split"
            )
        n_validation_success = sum(record.success for record in validation)
        minimum_length = min(record.length for record in validation + test)
        invalid_horizons = [value for value in fixed_horizons if value > minimum_length]
        if invalid_horizons:
            raise ValueError(
                f"{model_type} fold {fold}: fixed horizons {invalid_horizons} exceed "
                f"the minimum validation/test rollout length {minimum_length}; fixed "
                "evaluation must not extend or censor trajectories"
            )
        for cp_seed in cp_seeds:
            for alpha in alphas:
                thresholds, threshold_info = functional_cp_thresholds(
                    validation,
                    test,
                    alpha,
                    cp_seed,
                    horizon=horizon,
                )
                by_time, _ = evaluate_thresholds(
                    test, thresholds, csv_root=None, align_extend=True
                )
                for eval_time, metrics in by_time.items():
                    caught_alarm_fraction = metrics["mean_failed_first_alarm_fraction"]
                    effective_alarm_fraction = metrics["avg_det_time"]
                    rows.append(
                        {
                            "model": model_type,
                            "fold": fold,
                            "heldout_task": heldout_task,
                            "cp_seed": cp_seed,
                            "alpha": alpha,
                            "eval_time": eval_time,
                            "n_validation_success": n_validation_success,
                            "regression_calibration_count": threshold_info[
                                "regression_calibration_count"
                            ],
                            "modulation_calibration_count": threshold_info[
                                "modulation_calibration_count"
                            ],
                            "band_width": threshold_info["band_width"],
                            "realized_fpr": metrics["fpr"],
                            "catch_rate": metrics["tpr"],
                            "tnr": metrics["tnr"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "caught_alarm_fraction": caught_alarm_fraction,
                            "effective_alarm_fraction": effective_alarm_fraction,
                            "caught_lead_fraction": (
                                1.0 - caught_alarm_fraction
                                if np.isfinite(caught_alarm_fraction)
                                else float("nan")
                            ),
                            "effective_lead_fraction": 1.0
                            - effective_alarm_fraction,
                            "success_false_alarm_fraction": metrics[
                                "mean_success_false_alarm_fraction"
                            ],
                            "any_alarm_rate": metrics["any_alarm_rate"],
                            "roc_auc_max_score": metrics["roc_auc_max_score"],
                        }
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
                    fixed_by_time, _ = evaluate_thresholds(
                        fixed_test,
                        fixed_thresholds,
                        csv_root=None,
                        align_extend=False,
                    )
                    metrics = fixed_by_time["by final end"]
                    caught_alarm_fraction = metrics[
                        "mean_failed_first_alarm_fraction"
                    ]
                    effective_alarm_fraction = metrics["avg_det_time"]
                    rows.append(
                        {
                            "model": model_type,
                            "fold": fold,
                            "heldout_task": heldout_task,
                            "cp_seed": cp_seed,
                            "alpha": alpha,
                            "eval_time": f"fixed horizon {fixed_horizon}",
                            "n_validation_success": n_validation_success,
                            "regression_calibration_count": fixed_info[
                                "regression_calibration_count"
                            ],
                            "modulation_calibration_count": fixed_info[
                                "modulation_calibration_count"
                            ],
                            "band_width": fixed_info["band_width"],
                            "realized_fpr": metrics["fpr"],
                            "catch_rate": metrics["tpr"],
                            "tnr": metrics["tnr"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "caught_alarm_fraction": caught_alarm_fraction,
                            "effective_alarm_fraction": effective_alarm_fraction,
                            "caught_lead_fraction": (
                                1.0 - caught_alarm_fraction
                                if np.isfinite(caught_alarm_fraction)
                                else float("nan")
                            ),
                            "effective_lead_fraction": 1.0
                            - effective_alarm_fraction,
                            "success_false_alarm_fraction": metrics[
                                "mean_success_false_alarm_fraction"
                            ],
                            "any_alarm_rate": metrics["any_alarm_rate"],
                            "roc_auc_max_score": metrics["roc_auc_max_score"],
                        }
                    )
        print(
            f"{model_type}: evaluated fold {fold} (held-out task {heldout_task})",
            flush=True,
        )
    return pd.DataFrame(rows)


def summarize_per_task(raw: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in SUMMARY_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std_cp_seed"] = (metric, "std")
    aggregations["band_width_mean"] = ("band_width", "mean")
    return (
        raw.groupby(
            ["model", "fold", "heldout_task", "alpha", "eval_time"],
            as_index=False,
        )
        .agg(**aggregations)
        .sort_values(["model", "eval_time", "alpha", "heldout_task"])
    )


def bootstrap_task_ci(
    values: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_macro(
    per_task: pd.DataFrame, bootstrap_seed: int, n_bootstrap: int
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for (model, alpha, eval_time), group in per_task.groupby(
        ["model", "alpha", "eval_time"], sort=True
    ):
        row: dict[str, Any] = {
            "model": model,
            "alpha": alpha,
            "eval_time": eval_time,
            "n_tasks": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = group[f"{metric}_mean"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{metric}_std_task"] = (
                float(np.std(finite)) if len(finite) else np.nan
            )
            if len(finite):
                low, high = bootstrap_task_ci(finite, rng, n_bootstrap)
            else:
                low, high = np.nan, np.nan
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "eval_time", "alpha"])


def summarize_cp_seed(raw: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        f"{metric}_macro": (metric, "mean") for metric in SUMMARY_METRICS
    }
    return (
        raw.groupby(["model", "cp_seed", "alpha", "eval_time"], as_index=False)
        .agg(**aggregations)
        .sort_values(["model", "eval_time", "alpha", "cp_seed"])
    )


def operating_points(macro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, eval_time), group in macro.groupby(["model", "eval_time"]):
        group = group.copy()
        for cap in (0.01, 0.05, 0.10):
            eligible = group[group["realized_fpr_mean"] <= cap]
            constraint_met = not eligible.empty
            candidates = eligible if constraint_met else group.nsmallest(1, "realized_fpr_mean")
            chosen = candidates.sort_values(
                ["catch_rate_mean", "effective_alarm_fraction_mean"],
                ascending=[False, True],
            ).iloc[0]
            rows.append(
                {
                    "model": model,
                    "eval_time": eval_time,
                    "constraint_type": "realized_fpr_at_most",
                    "constraint_value": cap,
                    "constraint_met": constraint_met,
                    **{
                        key: chosen[key]
                        for key in (
                            "alpha",
                            "realized_fpr_mean",
                            "catch_rate_mean",
                            "caught_alarm_fraction_mean",
                            "effective_alarm_fraction_mean",
                            "effective_lead_fraction_mean",
                        )
                    },
                }
            )
        for target in (0.25, 0.50, 0.75):
            eligible = group[group["catch_rate_mean"] >= target]
            constraint_met = not eligible.empty
            candidates = eligible if constraint_met else group.nlargest(1, "catch_rate_mean")
            chosen = candidates.sort_values(
                ["realized_fpr_mean", "effective_alarm_fraction_mean"],
                ascending=[True, True],
            ).iloc[0]
            rows.append(
                {
                    "model": model,
                    "eval_time": eval_time,
                    "constraint_type": "catch_rate_at_least",
                    "constraint_value": target,
                    "constraint_met": constraint_met,
                    **{
                        key: chosen[key]
                        for key in (
                            "alpha",
                            "realized_fpr_mean",
                            "catch_rate_mean",
                            "caught_alarm_fraction_mean",
                            "effective_alarm_fraction_mean",
                            "effective_lead_fraction_mean",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def plot_tradeoff(macro: pd.DataFrame, path: Path) -> None:
    models = sorted(macro["model"].unique())
    eval_times = ["by earliest stop", "by final end"]
    figure, axes = plt.subplots(
        len(eval_times), len(models), figsize=(6.4 * len(models), 5.0 * len(eval_times)),
        squeeze=False,
    )
    color_norm = plt.Normalize(0.0, 1.0)
    for row_index, eval_time in enumerate(eval_times):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            selected = macro[
                (macro["model"] == model) & (macro["eval_time"] == eval_time)
            ].sort_values("alpha")
            scatter = axis.scatter(
                selected["realized_fpr_mean"],
                selected["catch_rate_mean"],
                c=selected["effective_alarm_fraction_mean"],
                cmap="viridis_r",
                norm=color_norm,
                s=64,
                zorder=3,
            )
            axis.plot(
                selected["realized_fpr_mean"],
                selected["catch_rate_mean"],
                color="0.45",
                linewidth=1.0,
                zorder=2,
            )
            for result in selected.itertuples(index=False):
                axis.annotate(
                    f"{result.alpha:g}",
                    (result.realized_fpr_mean, result.catch_rate_mean),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            axis.set_xlim(-0.01, max(0.12, selected["realized_fpr_mean"].max() + 0.03))
            axis.set_ylim(-0.03, 1.03)
            axis.grid(alpha=0.2)
            axis.set_title(f"{model.upper()} — {eval_time}")
            axis.set_xlabel("Realized held-out-task FPR")
            axis.set_ylabel("Failure catch rate")
            figure.colorbar(
                scatter,
                ax=axis,
                label="Effective alarm fraction (misses = 1)",
            )
    figure.suptitle("SAFE functional CP under leave-one-task-out evaluation")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    macro: pd.DataFrame, points: pd.DataFrame, args: argparse.Namespace
) -> None:
    lines = [
        "# SAFE Functional CP Under LOTO",
        "",
        f"- Alphas: `{','.join(str(value) for value in args.alpha_values)}`",
        f"- CP seeds: `{','.join(str(value) for value in args.cp_seed_values)}`",
        f"- Horizon: `{args.horizon}`",
        f"- Fixed horizons: `{','.join(str(value) for value in args.fixed_horizon_values) or 'none'}`",
        "- Calibration: successful seen-task validation rollouts only",
        "- Uncertainty: CP-seed mean within task, then task-macro; 95% task-bootstrap CI",
        "",
        "## Macro tradeoff",
        "",
        "| model | eval | alpha | realized FPR | catch | caught alarm frac | effective alarm frac |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in macro.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.eval_time} | {row.alpha:g} | "
            f"{row.realized_fpr_mean:.3f} | {row.catch_rate_mean:.3f} | "
            f"{row.caught_alarm_fraction_mean:.3f} | "
            f"{row.effective_alarm_fraction_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Constraint-selected operating points",
            "",
            "| model | eval | constraint | met | alpha | FPR | catch | effective alarm frac |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in points.itertuples(index=False):
        symbol = "<=" if row.constraint_type == "realized_fpr_at_most" else ">="
        name = "FPR" if row.constraint_type == "realized_fpr_at_most" else "catch"
        lines.append(
            f"| {row.model} | {row.eval_time} | {name} {symbol} {row.constraint_value:.0%} | "
            f"{row.constraint_met} | {row.alpha:g} | {row.realized_fpr_mean:.3f} | "
            f"{row.catch_rate_mean:.3f} | {row.effective_alarm_fraction_mean:.3f} |"
        )
    (args.output_dir / "functional_cp_loto_report.md").write_text(
        "\n".join(lines) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, nargs="+")
    parser.add_argument("--mlp-checkpoint-dir", type=Path)
    parser.add_argument("--lstm-checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", default="0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3"
    )
    parser.add_argument("--cp-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--horizon", type=int, default=520)
    parser.add_argument(
        "--fixed-horizons",
        default="",
        help=(
            "Comma-separated common observation horizons. Every validation and "
            "test rollout must be at least this long; no extension or censoring "
            "is allowed. The closeout analysis preregisters 50,100,148."
        ),
    )
    parser.add_argument(
        "--score-bundle-in",
        type=Path,
        help="Replay evaluation from a published score bundle without raw latents.",
    )
    parser.add_argument(
        "--score-bundle-out",
        type=Path,
        help="Write validation/test trajectories after checkpoint inference.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--task-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.alpha_values = parse_float_list(args.alphas)
    args.cp_seed_values = parse_int_list(args.cp_seeds)
    args.fixed_horizon_values = parse_optional_int_list(args.fixed_horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.score_bundle_in is not None:
        if any(
            value is not None
            for value in (
                args.root,
                args.mlp_checkpoint_dir,
                args.lstm_checkpoint_dir,
                args.score_bundle_out,
            )
        ):
            raise ValueError(
                "--score-bundle-in is mutually exclusive with raw roots, "
                "checkpoint directories, and --score-bundle-out"
            )
        family_records, family_splits, bundle_manifest = load_score_bundle(
            args.score_bundle_in
        )
        expected_models = {"mlp", "lstm"}
        if set(family_records) != expected_models:
            raise ValueError(
                f"Score bundle must contain {sorted(expected_models)}, found "
                f"{sorted(family_records)}"
            )
        checkpoint_manifest = bundle_manifest.get("checkpoint_manifest", {})
        root_manifest: Any = bundle_manifest.get("source_root_aliases", {})
        n_rollouts = int(bundle_manifest["unique_rollout_count"])
        if n_rollouts != 1000:
            raise ValueError(f"Expected 1,000 unique rollouts, found {n_rollouts}")
        print(
            f"score bundle: {n_rollouts} rollouts, horizon={args.horizon}", flush=True
        )
    else:
        missing = [
            name
            for name, value in (
                ("--root", args.root),
                ("--mlp-checkpoint-dir", args.mlp_checkpoint_dir),
                ("--lstm-checkpoint-dir", args.lstm_checkpoint_dir),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Raw inference requires " + ", ".join(missing)
            )
        device = torch.device(args.device)
        dataset = MixedSchemaLayerDataset(args.root, layer=32, token_pool="last")
        if len(dataset) != 1000:
            raise ValueError(f"Expected 1,000 rollouts, found {len(dataset)}")
        print(
            f"dataset: {len(dataset)} rollouts, horizon={args.horizon}, device={device}",
            flush=True,
        )
        family_records = {}
        family_splits = {}
        family_paths = {}
        for model_type, checkpoint_dir in (
            ("mlp", args.mlp_checkpoint_dir),
            ("lstm", args.lstm_checkpoint_dir),
        ):
            records, splits, paths = score_family(
                model_type, checkpoint_dir, dataset, args.batch_size, device
            )
            family_records[model_type] = records
            family_splits[model_type] = splits
            family_paths[model_type] = paths
        checkpoint_manifest = {
            model: {
                str(fold): {"file": path.name, "sha256": sha256_file(path)}
                for fold, path in paths.items()
            }
            for model, paths in family_paths.items()
        }
        root_manifest = {
            f"root-{index}": path.name for index, path in enumerate(args.root)
        }
        n_rollouts = len(dataset)
        if args.score_bundle_out is not None:
            command = " ".join(shlex.quote(value) for value in sys.argv)
            write_score_bundle(
                args.score_bundle_out,
                family_records,
                family_splits,
                args.root,
                family_paths,
                generator_paths=[
                    Path(__file__),
                    Path(__file__).with_name("evaluate_conformal.py"),
                    Path(__file__).with_name("prepare_openvla_layer_subset.py"),
                    Path(__file__).with_name("score_bundle.py"),
                    Path(__file__).with_name("score_layer_pooled_macro.py"),
                    Path(__file__).with_name("verify_primary_prepared_data.py"),
                    REPO_ROOT / "data" / "dataloaders.py",
                    *sorted((REPO_ROOT / "models").glob("*.py")),
                ],
                command=command,
            )
            print(f"wrote score bundle to {args.score_bundle_out}", flush=True)

    raw_frames = []
    for model_type in ("mlp", "lstm"):
        records = family_records[model_type]
        splits = family_splits[model_type]
        raw = evaluate_family(
            model_type,
            records,
            splits,
            args.alpha_values,
            args.cp_seed_values,
            args.horizon,
            args.fixed_horizon_values,
        )
        raw.to_csv(args.output_dir / f"functional_cp_{model_type}_raw.csv", index=False)
        raw_frames.append(raw)
        final_end = raw[raw["eval_time"] == "by final end"]
        quick = (
            final_end.groupby("alpha")[["realized_fpr", "catch_rate", "effective_alarm_fraction"]]
            .mean()
            .reset_index()
        )
        print(f"\n{model_type.upper()} full-rollout macro over task x CP seed:")
        print(quick.to_string(index=False), flush=True)
        gc.collect()

    raw = pd.concat(raw_frames, ignore_index=True)
    per_task = summarize_per_task(raw)
    macro = summarize_macro(per_task, args.bootstrap_seed, args.task_bootstrap)
    cp_seed_summary = summarize_cp_seed(raw)
    points = operating_points(macro)
    raw.to_csv(args.output_dir / "functional_cp_loto_raw.csv", index=False)
    per_task.to_csv(args.output_dir / "functional_cp_loto_per_task.csv", index=False)
    macro.to_csv(args.output_dir / "functional_cp_loto_macro.csv", index=False)
    cp_seed_summary.to_csv(
        args.output_dir / "functional_cp_loto_cp_seed_sensitivity.csv", index=False
    )
    points.to_csv(args.output_dir / "functional_cp_loto_operating_points.csv", index=False)
    fixed_mask = raw["eval_time"].str.startswith("fixed horizon ")
    if fixed_mask.any():
        fixed_raw = raw[fixed_mask].copy()
        fixed_per_task = per_task[
            per_task["eval_time"].str.startswith("fixed horizon ")
        ].copy()
        fixed_macro = macro[macro["eval_time"].str.startswith("fixed horizon ")].copy()
        fixed_points = points[
            points["eval_time"].str.startswith("fixed horizon ")
        ].copy()
        fixed_raw.to_csv(
            args.output_dir / "functional_cp_fixed_horizon_raw.csv", index=False
        )
        fixed_per_task.to_csv(
            args.output_dir / "functional_cp_fixed_horizon_per_task.csv", index=False
        )
        fixed_macro.to_csv(
            args.output_dir / "functional_cp_fixed_horizon_macro.csv", index=False
        )
        fixed_points.to_csv(
            args.output_dir / "functional_cp_fixed_horizon_operating_points.csv",
            index=False,
        )
    plot_tradeoff(macro, args.output_dir / "functional_cp_loto_tradeoff.png")
    write_report(macro, points, args)
    manifest = {
        "roots": root_manifest,
        "n_rollouts": n_rollouts,
        "horizon": args.horizon,
        "fixed_horizons": args.fixed_horizon_values,
        "alphas": args.alpha_values,
        "cp_seeds": args.cp_seed_values,
        "task_bootstrap": args.task_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "checkpoint_manifest": checkpoint_manifest,
    }
    with (args.output_dir / "functional_cp_loto_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nWrote functional CP LOTO artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
