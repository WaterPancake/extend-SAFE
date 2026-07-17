"""Decompose pooled ROC-AUC and task-bootstrap its gap from macro AUC.

The input is a long-form CSV with one row per held-out rollout/model/split and
the columns ``model, split_id, rollout_idx, task_id, label, score``. Generate it
from finalized checkpoints with ``score_layer_pooled_macro.py --output ...``.

For each model/split, pooled AUC is decomposed exactly into comparisons where
the failed and successful rollouts come from the same task and comparisons
where they come from different tasks. Confidence intervals use a paired,
hierarchical bootstrap: resample task blocks, then resample failures and
successes separately within each selected task.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REQUIRED_COLUMNS = {
    "model",
    "split_id",
    "rollout_idx",
    "task_id",
    "label",
    "score",
}


@dataclass(frozen=True)
class Decomposition:
    pooled_auc: float
    macro_auc: float
    within_micro_auc: float
    between_task_auc: float
    pooled_minus_macro: float
    n_tasks: int
    n_rollouts: int
    n_success: int
    n_failure: int
    n_within_pairs: int
    n_between_pairs: int
    between_pair_fraction: float


def pair_sum(labels: np.ndarray, scores: np.ndarray) -> tuple[float, int]:
    """Return Mann-Whitney correct-pair sum (ties=0.5) and pair count."""

    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan"), 0
    comparisons = positive[:, None] - negative[None, :]
    correct = float(np.count_nonzero(comparisons > 0))
    correct += 0.5 * float(np.count_nonzero(comparisons == 0))
    return correct, int(comparisons.size)


def decompose(
    labels: np.ndarray, scores: np.ndarray, task_ids: np.ndarray
) -> Decomposition:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    task_ids = np.asarray(task_ids)
    if not (len(labels) == len(scores) == len(task_ids)):
        raise ValueError("labels, scores, and task_ids must have equal length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("each model/split must contain both labels 0 and 1")

    total_correct, total_pairs = pair_sum(labels, scores)
    per_task_aucs = []
    within_correct = 0.0
    within_pairs = 0
    for task_id in np.unique(task_ids):
        active = task_ids == task_id
        task_correct, task_pairs = pair_sum(labels[active], scores[active])
        if task_pairs == 0:
            raise ValueError(f"task {task_id!r} does not contain both classes")
        per_task_aucs.append(task_correct / task_pairs)
        within_correct += task_correct
        within_pairs += task_pairs

    between_correct = total_correct - within_correct
    between_pairs = total_pairs - within_pairs
    if between_pairs <= 0:
        raise ValueError("decomposition needs at least two tasks")

    pooled_auc = total_correct / total_pairs
    macro_auc = float(np.mean(per_task_aucs))
    within_auc = within_correct / within_pairs
    between_auc = between_correct / between_pairs
    reconstructed = (
        within_pairs * within_auc + between_pairs * between_auc
    ) / total_pairs
    sklearn_auc = float(roc_auc_score(labels, scores))
    if not np.isclose(pooled_auc, sklearn_auc, atol=1e-12):
        raise AssertionError((pooled_auc, sklearn_auc))
    if not np.isclose(pooled_auc, reconstructed, atol=1e-12):
        raise AssertionError((pooled_auc, reconstructed))

    return Decomposition(
        pooled_auc=pooled_auc,
        macro_auc=macro_auc,
        within_micro_auc=within_auc,
        between_task_auc=between_auc,
        pooled_minus_macro=pooled_auc - macro_auc,
        n_tasks=int(len(np.unique(task_ids))),
        n_rollouts=int(len(labels)),
        n_success=int(np.count_nonzero(labels == 0)),
        n_failure=int(np.count_nonzero(labels == 1)),
        n_within_pairs=int(within_pairs),
        n_between_pairs=int(between_pairs),
        between_pair_fraction=float(between_pairs / total_pairs),
    )


def hierarchical_bootstrap(
    frame: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator
) -> pd.DataFrame:
    tasks = frame["task_id"].unique()
    task_scores = {}
    for task in tasks:
        task_frame = frame[frame["task_id"] == task]
        task_scores[task] = {
            label: task_frame.loc[task_frame["label"] == label, "score"].to_numpy(
                dtype=np.float64
            )
            for label in (0, 1)
        }
        if any(values.size == 0 for values in task_scores[task].values()):
            raise ValueError(f"task {task!r} does not contain both classes")

    rows = []
    for bootstrap_id in range(n_bootstrap):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        positive_parts = []
        negative_parts = []
        task_aucs = []
        within_correct = 0.0
        within_pairs = 0
        for task in sampled_tasks:
            source_negative = task_scores[task][0]
            source_positive = task_scores[task][1]
            negative = source_negative[
                rng.integers(0, len(source_negative), size=len(source_negative))
            ]
            positive = source_positive[
                rng.integers(0, len(source_positive), size=len(source_positive))
            ]
            comparisons = positive[:, None] - negative[None, :]
            task_correct = float(np.count_nonzero(comparisons > 0))
            task_correct += 0.5 * float(np.count_nonzero(comparisons == 0))
            task_pairs = int(comparisons.size)
            task_aucs.append(task_correct / task_pairs)
            within_correct += task_correct
            within_pairs += task_pairs
            positive_parts.append(positive)
            negative_parts.append(negative)

        positive = np.concatenate(positive_parts)
        negative = np.concatenate(negative_parts)
        comparisons = positive[:, None] - negative[None, :]
        total_correct = float(np.count_nonzero(comparisons > 0))
        total_correct += 0.5 * float(np.count_nonzero(comparisons == 0))
        total_pairs = int(comparisons.size)
        between_pairs = total_pairs - within_pairs
        between_correct = total_correct - within_correct
        pooled_auc = total_correct / total_pairs
        macro_auc = float(np.mean(task_aucs))
        result = Decomposition(
            pooled_auc=pooled_auc,
            macro_auc=macro_auc,
            within_micro_auc=within_correct / within_pairs,
            between_task_auc=between_correct / between_pairs,
            pooled_minus_macro=pooled_auc - macro_auc,
            n_tasks=int(len(sampled_tasks)),
            n_rollouts=int(len(positive) + len(negative)),
            n_success=int(len(negative)),
            n_failure=int(len(positive)),
            n_within_pairs=int(within_pairs),
            n_between_pairs=int(between_pairs),
            between_pair_fraction=float(between_pairs / total_pairs),
        )
        rows.append({"bootstrap_id": bootstrap_id, **asdict(result)})
    return pd.DataFrame(rows)


def percentile_interval(values: pd.Series) -> tuple[float, float]:
    low, high = np.quantile(values.to_numpy(dtype=float), [0.025, 0.975])
    return float(low), float(high)


def plot_decomposition(by_split: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("macro_auc", "Macro within-task"),
        ("within_micro_auc", "Within-task micro"),
        ("between_task_auc", "Between-task"),
        ("pooled_auc", "Pooled"),
    ]
    models = list(dict.fromkeys(by_split["model"].astype(str)))
    figure, axes = plt.subplots(
        len(models), 1, figsize=(10, max(3.4, 3.1 * len(models))), squeeze=False
    )
    for axis, model in zip(axes[:, 0], models):
        selected = by_split[by_split["model"].astype(str) == model].copy()
        selected = selected.sort_values("split_id", key=lambda x: x.astype(str))
        x = np.arange(len(selected))
        width = 0.19
        for metric_idx, (column, label) in enumerate(metrics):
            offset = (metric_idx - (len(metrics) - 1) / 2) * width
            axis.bar(x + offset, selected[column], width=width, label=label)
        axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axis.set_ylim(0.4, 1.0)
        axis.set_ylabel("ROC-AUC")
        axis.set_title(str(model).upper())
        axis.set_xticks(x, [f"split {value}" for value in selected["split_id"]])
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper center")
    axes[-1, 0].set_xlabel("Original held-out-task split")
    figure.suptitle("Where pooled ROC-AUC gets its pairwise comparisons", y=1.01)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        nargs="+",
        required=True,
        help="One or more compatible long-form score CSVs (for example MLP and LSTM).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/results_audit"))
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/pooled_auc_decomposition.png")
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for score_path in args.scores:
        score_frame = pd.read_csv(score_path)
        score_frame["score_source"] = str(score_path)
        frames.append(score_frame)
    frame = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"score inputs are missing columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required score columns contain missing values")
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be binary with failure=1 and success=0")
    duplicate_key = ["model", "split_id", "rollout_idx"]
    if frame.duplicated(duplicate_key).any():
        raise ValueError(f"duplicate rows for key {duplicate_key}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    point_rows = []
    task_summary_rows = []
    cross_pair_rows = []
    bootstrap_frames = []
    summary_rows = []
    for (model, split_id), group in frame.groupby(["model", "split_id"], sort=True):
        point = decompose(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            group["task_id"].to_numpy(),
        )
        heldout_tasks = ",".join(
            str(task) for task in sorted(group["task_id"].unique())
        )
        point_rows.append(
            {
                "model": model,
                "split_id": split_id,
                "heldout_tasks": heldout_tasks,
                **asdict(point),
            }
        )
        tasks = sorted(group["task_id"].unique())
        for task_id in tasks:
            task_group = group[group["task_id"] == task_id]
            task_labels = task_group["label"].to_numpy(dtype=np.int8)
            task_scores = task_group["score"].to_numpy(dtype=np.float64)
            task_correct, task_pairs = pair_sum(task_labels, task_scores)
            success_scores = task_scores[task_labels == 0]
            failure_scores = task_scores[task_labels == 1]
            task_summary_rows.append(
                {
                    "model": model,
                    "split_id": split_id,
                    "task_id": task_id,
                    "n_rollouts": len(task_group),
                    "n_success": len(success_scores),
                    "n_failure": len(failure_scores),
                    "failure_rate": float(np.mean(task_labels)),
                    "within_task_auc": task_correct / task_pairs,
                    "mean_success_score": float(np.mean(success_scores)),
                    "mean_failure_score": float(np.mean(failure_scores)),
                    "median_success_score": float(np.median(success_scores)),
                    "median_failure_score": float(np.median(failure_scores)),
                }
            )
        for failure_task in tasks:
            failure_scores = group.loc[
                (group["task_id"] == failure_task) & (group["label"] == 1),
                "score",
            ].to_numpy(dtype=np.float64)
            for success_task in tasks:
                success_scores = group.loc[
                    (group["task_id"] == success_task) & (group["label"] == 0),
                    "score",
                ].to_numpy(dtype=np.float64)
                comparisons = failure_scores[:, None] - success_scores[None, :]
                correct = float(np.count_nonzero(comparisons > 0))
                correct += 0.5 * float(np.count_nonzero(comparisons == 0))
                cross_pair_rows.append(
                    {
                        "model": model,
                        "split_id": split_id,
                        "failure_task": failure_task,
                        "success_task": success_task,
                        "same_task": failure_task == success_task,
                        "pair_auc": correct / comparisons.size,
                        "n_pairs": int(comparisons.size),
                    }
                )
        bootstrap = hierarchical_bootstrap(group, args.n_bootstrap, rng)
        bootstrap.insert(0, "split_id", split_id)
        bootstrap.insert(0, "model", model)
        bootstrap_frames.append(bootstrap)
        delta_low, delta_high = percentile_interval(bootstrap["pooled_minus_macro"])
        between_low, between_high = percentile_interval(bootstrap["between_task_auc"])
        summary_rows.append(
            {
                "model": model,
                "split_id": split_id,
                "heldout_tasks": heldout_tasks,
                "pooled_minus_macro": point.pooled_minus_macro,
                "pooled_minus_macro_ci_low": delta_low,
                "pooled_minus_macro_ci_high": delta_high,
                "prob_delta_le_zero": float(
                    np.mean(bootstrap["pooled_minus_macro"] <= 0)
                ),
                "between_task_auc": point.between_task_auc,
                "between_task_auc_ci_low": between_low,
                "between_task_auc_ci_high": between_high,
                "between_pair_fraction": point.between_pair_fraction,
                "n_bootstrap": args.n_bootstrap,
            }
        )

    by_split = pd.DataFrame(point_rows)
    bootstraps = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_summary = pd.DataFrame(summary_rows)
    descriptive = (
        by_split.groupby("model", as_index=False)
        .agg(
            n_splits=("split_id", "nunique"),
            mean_pooled_auc=("pooled_auc", "mean"),
            mean_macro_auc=("macro_auc", "mean"),
            mean_within_micro_auc=("within_micro_auc", "mean"),
            mean_between_task_auc=("between_task_auc", "mean"),
            mean_pooled_minus_macro=("pooled_minus_macro", "mean"),
            min_pooled_minus_macro=("pooled_minus_macro", "min"),
            max_pooled_minus_macro=("pooled_minus_macro", "max"),
            mean_between_pair_fraction=("between_pair_fraction", "mean"),
        )
    )

    by_split.to_csv(output_dir / "pair_decomposition_by_split.csv", index=False)
    pd.DataFrame(task_summary_rows).to_csv(
        output_dir / "task_score_summary.csv", index=False
    )
    pd.DataFrame(cross_pair_rows).to_csv(
        output_dir / "ordered_task_pair_auc.csv", index=False
    )
    bootstrap_summary.to_csv(
        output_dir / "pair_decomposition_bootstrap_summary.csv", index=False
    )
    bootstraps.to_csv(output_dir / "pair_decomposition_bootstrap_samples.csv", index=False)
    descriptive.to_csv(output_dir / "pair_decomposition_descriptive.csv", index=False)
    plot_decomposition(by_split, args.figure)
    manifest = {
        "scores": [str(path) for path in args.scores],
        "n_score_rows": int(len(frame)),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "task, then label-stratified rollouts within task",
        "outputs": [
            "pair_decomposition_by_split.csv",
            "pair_decomposition_bootstrap_summary.csv",
            "pair_decomposition_bootstrap_samples.csv",
            "pair_decomposition_descriptive.csv",
            "task_score_summary.csv",
            "ordered_task_pair_auc.csv",
            str(args.figure),
        ],
    }
    with (output_dir / "pair_decomposition_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(descriptive.to_string(index=False))
    print(f"wrote decomposition outputs to {output_dir}")
    print(f"wrote figure to {args.figure}")


if __name__ == "__main__":
    main()
