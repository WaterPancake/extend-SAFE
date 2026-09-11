"""Aggregate dynamic-layer LOTO and functional-CP results across train seeds."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_functional_cp_loto import (
    operating_points,
    summarize_macro,
    summarize_per_task,
)


COMPARISONS = (
    ("layer20_32", "baseline32"),
    ("shuffled20_32", "baseline32"),
    ("layer20_32", "shuffled20_32"),
    ("late24_28_32", "baseline32"),
    ("shuffled_late24_28_32", "baseline32"),
    ("late24_28_32", "shuffled_late24_28_32"),
    ("residual20_32", "baseline32"),
    ("shuffled_residual20_32", "baseline32"),
    ("residual20_32", "shuffled_residual20_32"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--cp-raw", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    return parser.parse_args()


def bootstrap_interval(
    values: np.ndarray, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def sign_flip_pvalue(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def paired_cp_row(
    per_task: pd.DataFrame,
    *,
    left: str,
    right: str,
    eval_time: str,
    left_alpha: float,
    right_alpha: float,
    comparison_type: str,
    bootstrap_seed: int,
    n_bootstrap: int,
) -> dict:
    keys = ["heldout_task"]
    metric_columns = (
        "realized_fpr_mean",
        "catch_rate_mean",
        "effective_lead_fraction_mean",
    )
    left_rows = per_task[
        (per_task.model == left)
        & (per_task.eval_time == eval_time)
        & np.isclose(per_task.alpha, left_alpha)
    ][keys + list(metric_columns)]
    right_rows = per_task[
        (per_task.model == right)
        & (per_task.eval_time == eval_time)
        & np.isclose(per_task.alpha, right_alpha)
    ][keys + list(metric_columns)]
    paired = left_rows.merge(right_rows, on=keys, suffixes=("_left", "_right"))
    if paired.empty:
        raise ValueError(
            f"no paired functional-CP rows for {left} versus {right} at {eval_time}"
        )
    row = {
        "comparison_type": comparison_type,
        "left": left,
        "right": right,
        "eval_time": eval_time,
        "left_alpha": left_alpha,
        "right_alpha": right_alpha,
        "n_tasks": int(len(paired)),
    }
    for metric in metric_columns:
        delta = (
            paired[f"{metric}_left"] - paired[f"{metric}_right"]
        ).to_numpy(dtype=float)
        finite = delta[np.isfinite(delta)]
        if len(finite):
            low, high = bootstrap_interval(finite, bootstrap_seed, n_bootstrap)
            mean = float(finite.mean())
            positive_tasks = int((finite > 0).sum())
        else:
            low = high = mean = float("nan")
            positive_tasks = 0
        prefix = metric.removesuffix("_mean")
        row[f"delta_{prefix}"] = mean
        row[f"delta_{prefix}_ci_low"] = low
        row[f"delta_{prefix}_ci_high"] = high
        row[f"delta_{prefix}_positive_tasks"] = positive_tasks
    catch_delta = (
        paired["catch_rate_mean_left"] - paired["catch_rate_mean_right"]
    ).to_numpy(dtype=float)
    catch_delta = catch_delta[np.isfinite(catch_delta)]
    row["catch_rate_sign_flip_pvalue_two_sided"] = (
        sign_flip_pvalue(catch_delta) if len(catch_delta) else float("nan")
    )
    return row


def paired_cp_comparisons(
    per_task: pd.DataFrame,
    points: pd.DataFrame,
    bootstrap_seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    available_models = set(per_task.model.unique())
    eval_times = sorted(per_task.eval_time.unique())
    common_alphas = sorted(per_task.alpha.unique())
    for left, right in COMPARISONS:
        if left not in available_models or right not in available_models:
            continue
        for eval_time in eval_times:
            for alpha in common_alphas:
                rows.append(
                    paired_cp_row(
                        per_task,
                        left=left,
                        right=right,
                        eval_time=eval_time,
                        left_alpha=float(alpha),
                        right_alpha=float(alpha),
                        comparison_type="same nominal alpha",
                        bootstrap_seed=bootstrap_seed,
                        n_bootstrap=n_bootstrap,
                    )
                )
            selected = points[
                (points.eval_time == eval_time)
                & (points.constraint_type == "realized_fpr_at_most")
                & np.isclose(points.constraint_value, 0.05)
            ].set_index("model")
            if left not in selected.index or right not in selected.index:
                continue
            rows.append(
                paired_cp_row(
                    per_task,
                    left=left,
                    right=right,
                    eval_time=eval_time,
                    left_alpha=float(selected.loc[left, "alpha"]),
                    right_alpha=float(selected.loc[right, "alpha"]),
                    comparison_type="independent FPR <= 0.05 operating points",
                    bootstrap_seed=bootstrap_seed,
                    n_bootstrap=n_bootstrap,
                )
            )
    return pd.DataFrame(rows)


def aggregate_ranking(args: argparse.Namespace) -> dict:
    rows = []
    gate_rows = []
    for seed_index, path in enumerate(args.summary):
        payload = json.loads(path.read_text())
        train_seed = int(payload["config"]["seed"])
        for variant, result in payload["variants"].items():
            for task, auc in result["task_aucs"].items():
                rows.append(
                    {
                        "train_seed": train_seed,
                        "variant": variant,
                        "task": int(task),
                        "auc": float(auc),
                    }
                )
            for position, weight in result.get("mean_gate_weights", {}).items():
                gate_rows.append(
                    {
                        "train_seed": train_seed,
                        "variant": variant,
                        "position": int(position),
                        "weight": float(weight),
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(args.summary)
    counts = frame.groupby(["variant", "task"])["train_seed"].nunique()
    if not (counts == expected).all():
        raise ValueError("not every variant/task has every training seed")

    per_task = (
        frame.groupby(["variant", "task"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_sd_train_seed=("auc", "std"))
        .sort_values(["variant", "task"])
    )
    per_seed = (
        frame.groupby(["variant", "train_seed"], as_index=False)
        .agg(macro_task_auc=("auc", "mean"))
        .sort_values(["variant", "train_seed"])
    )
    per_task.to_csv(args.output_dir / "ranking_per_task_seed_aggregated.csv", index=False)
    per_seed.to_csv(args.output_dir / "ranking_per_train_seed.csv", index=False)

    summary = {
        "n_train_seeds": len(args.summary),
        "train_seeds": sorted(frame["train_seed"].unique().astype(int).tolist()),
        "n_tasks": int(frame["task"].nunique()),
        "tasks": sorted(frame["task"].unique().astype(int).tolist()),
        "variants": {},
        "comparisons": {},
    }
    for variant, group in per_task.groupby("variant"):
        seed_values = per_seed[per_seed["variant"] == variant]["macro_task_auc"]
        summary["variants"][variant] = {
            "macro_task_auc": float(group["auc_mean"].mean()),
            "task_auc_sd": float(group["auc_mean"].std(ddof=0)),
            "macro_auc_by_train_seed": [float(value) for value in seed_values],
        }
    if gate_rows:
        gates = pd.DataFrame(gate_rows)
        for (variant, position), group in gates.groupby(["variant", "position"]):
            summary["variants"][variant].setdefault("mean_gate_weights", {})[
                str(position)
            ] = float(group["weight"].mean())

    pivot = per_task.pivot(index="task", columns="variant", values="auc_mean")
    for left, right in COMPARISONS:
        if left not in pivot or right not in pivot:
            continue
        delta = (pivot[left] - pivot[right]).to_numpy(dtype=float)
        low, high = bootstrap_interval(delta, args.bootstrap_seed, args.n_bootstrap)
        seed_delta = (
            per_seed[per_seed.variant == left]
            .set_index("train_seed")["macro_task_auc"]
            - per_seed[per_seed.variant == right]
            .set_index("train_seed")["macro_task_auc"]
        )
        summary["comparisons"][f"{left}_minus_{right}"] = {
            "mean_delta_auc": float(delta.mean()),
            "bootstrap_95_ci": [low, high],
            "positive_tasks": int((delta > 0).sum()),
            "n_tasks": len(delta),
            "sign_flip_pvalue_two_sided": sign_flip_pvalue(delta),
            "macro_delta_by_train_seed": [float(value) for value in seed_delta],
            "per_task_delta": {
                str(task): float(value) for task, value in zip(pivot.index, delta)
            },
        }
    return summary


def aggregate_cp(args: argparse.Namespace) -> dict | None:
    if not args.cp_raw:
        return None
    train_seeds = [
        int(json.loads(path.read_text())["config"]["seed"])
        for path in args.summary
    ]
    if len(args.cp_raw) % len(train_seeds) != 0:
        raise ValueError(
            "--cp-raw must contain the same number of files per training seed"
        )
    frames = []
    for path_index, path in enumerate(args.cp_raw):
        frame = pd.read_csv(path)
        # Files are supplied in complete seed-ordered sets. This permits, for
        # example, one set of retrospective rows followed by one set of fixed-
        # horizon rows while retaining the actual training-seed identity.
        frame["train_seed"] = train_seeds[path_index % len(train_seeds)]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    per_task = summarize_per_task(raw)
    macro = summarize_macro(per_task, args.bootstrap_seed, args.n_bootstrap)
    points = operating_points(macro)
    comparisons = paired_cp_comparisons(
        per_task, points, args.bootstrap_seed, args.n_bootstrap
    )
    per_task.to_csv(args.output_dir / "functional_cp_per_task_seed_aggregated.csv", index=False)
    macro.to_csv(args.output_dir / "functional_cp_macro_seed_aggregated.csv", index=False)
    points.to_csv(
        args.output_dir / "functional_cp_operating_points_seed_aggregated.csv",
        index=False,
    )
    comparisons.to_csv(
        args.output_dir / "functional_cp_comparisons_seed_aggregated.csv",
        index=False,
    )
    fpr5 = points[
        (points.constraint_type == "realized_fpr_at_most")
        & (points.constraint_value == 0.05)
    ].sort_values("model")
    fpr5_comparisons = comparisons[
        comparisons.comparison_type
        == "independent FPR <= 0.05 operating points"
    ]
    return {
        "n_train_seeds": len(train_seeds),
        "train_seeds": train_seeds,
        "n_cp_artifact_sets": len(args.cp_raw) // len(train_seeds),
        "fpr_at_most_5_percent": fpr5.to_dict(orient="records"),
        "paired_fpr_at_most_5_percent_comparisons": fpr5_comparisons.to_dict(
            orient="records"
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_ranking(args)
    summary["functional_cp"] = aggregate_cp(args)
    with (args.output_dir / "summary_seed_aggregated.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
