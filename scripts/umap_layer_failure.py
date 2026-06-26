"""UMAP of per-layer rollout-pooled features: does FAILURE separate, or do
latents just organize by task/time? (nonlinear robustness check on the linear
complementarity gate, + paper figure.)

Uses the cached pooled features ([0,task_min_step]-mean per layer). Produces:
  - docs/umap_by_label.png  (per-layer UMAP, colored fail vs success)
  - docs/umap_by_task.png    (per-layer UMAP, colored by task)
and a quantitative companion: cross-task kNN label-separability per layer
(train kNN on 9 tasks' features, AUC on the held-out task; mean over 10 tasks).
A nonlinear analog of the linear probe -- if it's ~chance on held-out tasks too,
there is no nonlinear failure structure the linear gate missed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/rollouts")
CACHE = Path("runs/openvla_layer_correlations/complementarity_cache.npz")
LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 32]
NAME_RE = re.compile(r"task(?P<t>\d+)--ep(?P<e>\d+)--succ(?P<s>[01])\.pkl$")


def main() -> None:
    d = np.load(CACHE)
    pooled, labels = d["pooled"], d["labels"]  # (n,9,4096), (n,)
    paths = sorted(ROOT.glob("**/*.pkl"))
    task_ids = np.array([int(NAME_RE.search(p.name).group("t")) for p in paths])
    succ = np.array([1 - int(NAME_RE.search(p.name).group("s")) for p in paths], float)  # 1=fail
    assert len(paths) == len(labels), f"{len(paths)} files vs {len(labels)} cached"
    assert np.allclose(succ, labels), "filename labels do not match cache order!"
    tasks = sorted(set(task_ids))

    # ---- UMAP per layer ----
    embeds = {}
    for li, L in enumerate(LAYERS):
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0)
        embeds[L] = reducer.fit_transform(pooled[:, li, :])
        print(f"umap layer {L} done", flush=True)

    for tag, color_by in [("by_label", labels), ("by_task", task_ids)]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        for ax, L in zip(axes.ravel(), LAYERS):
            e = embeds[L]
            if tag == "by_label":
                ax.scatter(e[labels == 0, 0], e[labels == 0, 1], s=6, c="#4C72B0", alpha=0.5, label="success")
                ax.scatter(e[labels == 1, 0], e[labels == 1, 1], s=6, c="#c0392b", alpha=0.5, label="failure")
            else:
                sc = ax.scatter(e[:, 0], e[:, 1], s=6, c=color_by, cmap="tab10", alpha=0.6)
            ax.set_title(f"layer {L}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        if tag == "by_label":
            axes[0, 0].legend(fontsize=8, markerscale=2, loc="upper right")
        fig.suptitle(f"UMAP of rollout-pooled features, {tag.replace('_',' ')}", fontsize=14)
        fig.tight_layout()
        out = Path("docs") / f"umap_{tag}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"wrote {out}")

    # ---- quantitative companion: cross-task kNN label AUC (LOTO) ----
    print("\n=== Cross-task kNN (k=15) label-separability per layer (mean over 10 held-out tasks) ===")
    print(f"{'layer':>6} {'kNN AUC':>8} {'std':>7}")
    for li, L in enumerate(LAYERS):
        aucs = []
        for t in tasks:
            tr = task_ids != t; te = task_ids == t
            if len(np.unique(labels[te])) < 2:
                continue
            sc = StandardScaler().fit(pooled[tr, li, :])
            knn = KNeighborsClassifier(n_neighbors=15).fit(sc.transform(pooled[tr, li, :]), labels[tr])
            p = knn.predict_proba(sc.transform(pooled[te, li, :]))[:, 1]
            aucs.append(roc_auc_score(labels[te], p))
        print(f"{L:>6} {np.mean(aucs):>8.3f} {np.std(aucs):>7.3f}")


if __name__ == "__main__":
    main()
