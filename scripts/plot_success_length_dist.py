"""Plot per-task SUCCESS rollout-length distributions.

Failures always run to the 520-step timeout, so only successful rollouts carry
length structure. Produces:
  1) a cross-task violin plot (compare distributions side by side), and
  2) a per-task histogram grid (distribution shape per task), and
  3) pooled successful lengths and the distribution of per-task means,
with the global-min cap (148) and the 520 timeout marked.

Lengths are cached to runs/openvla_layer_correlations/success_lengths.json so
re-plotting does not re-read the 1000 pickles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("data/rollouts")
OUTDIR = Path("docs")
CACHE = Path("runs/openvla_layer_correlations/success_lengths.json")


def load_success_lengths() -> dict[int, list[int]]:
    if CACHE.exists():
        return {int(k): v for k, v in json.loads(CACHE.read_text()).items()}
    from data.dataloaders import OpenVLARolloutDataset

    ds = OpenVLARolloutDataset(ROOT, layers=(32,), token_pool="last", recursive=True, cache=False)
    succ: dict[int, list[int]] = {}
    for info in ds.rollouts:
        art = ds._load_pickle(info.path)
        if ds._read_success(art, info):
            t = int(ds._read_task_id(art, info))
            succ.setdefault(t, []).append(int(art["hidden_states"].shape[0]))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({str(k): v for k, v in succ.items()}))
    return succ


def main() -> None:
    succ = load_success_lengths()
    tasks = sorted(succ)
    data = [sorted(succ[t]) for t in tasks]
    global_min = min(min(d) for d in data)

    # ---- Figure 1: cross-task violin ----
    fig, ax = plt.subplots(figsize=(11, 6))
    parts = ax.violinplot(data, positions=tasks, widths=0.8, showmedians=True,
                          showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0"); pc.set_alpha(0.65)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_edgecolor("#1f2d3d"); parts[key].set_linewidth(1.0)
    ax.axhline(520, color="#c0392b", ls="--", lw=1.3, label="failure length / timeout = 520")
    ax.axhline(global_min, color="#2e7d32", ls="--", lw=1.3,
               label=f"global min cap = {global_min}")
    for t, d in zip(tasks, data):
        ax.text(t, max(d) + 6, f"n={len(d)}", ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_xlabel("task id"); ax.set_ylabel("rollout length (steps)")
    ax.set_title("Per-task SUCCESS rollout-length distribution (failures all = 520)")
    ax.set_xticks(tasks); ax.set_ylim(0, 560)
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p1 = OUTDIR / "success_length_violin.png"
    fig.savefig(p1, dpi=160); plt.close(fig)

    # ---- Figure 2: per-task histogram grid ----
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    bins = np.linspace(100, 520, 36)
    for ax, t, d in zip(axes.ravel(), tasks, data):
        arr = np.array(d)
        ax.hist(arr, bins=bins, color="#4C72B0", alpha=0.8)
        ax.axvline(np.min(arr), color="#2e7d32", ls="--", lw=1.0)
        ax.axvline(np.median(arr), color="#e67e22", ls="-", lw=1.2)
        ax.axvline(520, color="#c0392b", ls="--", lw=1.0)
        ax.set_title(f"task {t}  (n={len(d)})\nmin={int(arr.min())} med={int(np.median(arr))}",
                     fontsize=9)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(handles=[
        plt.Line2D([], [], color="#2e7d32", ls="--", label="min"),
        plt.Line2D([], [], color="#e67e22", label="median"),
        plt.Line2D([], [], color="#c0392b", ls="--", label="520 timeout"),
    ], fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Per-task SUCCESS rollout-length histograms", y=1.0, fontsize=13)
    fig.supxlabel("rollout length (steps)"); fig.supylabel("count")
    fig.tight_layout()
    p2 = OUTDIR / "success_length_hist_grid.png"
    fig.savefig(p2, dpi=160); plt.close(fig)

    # ---- Figure 3: pooled rollouts vs equally weighted task means ----
    pooled = np.concatenate([np.asarray(values) for values in data])
    task_means = np.asarray([np.mean(values) for values in data])
    pooled_mean = float(np.mean(pooled))
    pooled_variance = float(np.var(pooled, ddof=1))
    task_mean = float(np.mean(task_means))
    task_mean_variance = float(np.var(task_means, ddof=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(
        pooled,
        bins=np.arange(140, 541, 20),
        color="#4C72B0",
        edgecolor="white",
        alpha=0.9,
    )
    axes[0].axvline(
        pooled_mean, color="#c0392b", lw=2, label=f"mean = {pooled_mean:.2f}"
    )
    axes[0].set_title(f"All successful rollouts (n={len(pooled)})")
    axes[0].set_xlabel("rollout length (steps)")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(
        task_means,
        bins=np.arange(175, 476, 25),
        color="#55A868",
        edgecolor="white",
        alpha=0.9,
    )
    axes[1].axvline(
        task_mean, color="#c0392b", lw=2, label=f"mean = {task_mean:.2f}"
    )
    for task, value in zip(tasks, task_means):
        axes[1].annotate(
            str(task),
            xy=(value, 0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_title("Mean successful length per task (10 tasks)")
    axes[1].set_xlabel("task mean rollout length (steps)")
    axes[1].set_ylabel("number of tasks")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Successful rollout-length distributions")
    fig.tight_layout()
    p3 = OUTDIR / "success_length_histogram.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)

    print(
        f"wrote {p1}\n"
        f"wrote {p2}\n"
        f"wrote {p3}\n"
        f"pooled: mean={pooled_mean:.4f} variance={pooled_variance:.4f} "
        f"std={np.sqrt(pooled_variance):.4f}\n"
        f"task means: mean={task_mean:.4f} variance={task_mean_variance:.4f} "
        f"std={np.sqrt(task_mean_variance):.4f}\n"
        f"cached lengths -> {CACHE}"
    )


if __name__ == "__main__":
    main()
