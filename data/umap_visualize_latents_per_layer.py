"""Per-layer UMAP of VLA hidden states.

Usage:
    python umap_visualize_latents_per_layer.py --dir INPUT_DIR --model MODEL_NAME [--dataset DATASET_NAME] [--tokens {mean,first,last,first+last}] [--n-neighbors N [N ...]]

INPUT_DIR must contain .pkl rollout files named `task<id>--ep<idx>--succ{0,1}.pkl`.
Output PNGs are written to data/visualization by default, tagged with MODEL_NAME,
optional DATASET_NAME, token choice, and n_neighbors.

Pipeline:
1. load every rollout from INPUT_DIR,
2. select/pool the per-step hidden state across action tokens with --tokens,
3. split the concatenated layer features into (n_layers, feature_dim),
4. PCA-pre-reduce each layer to 50d when needed (matching the per-layer t-SNE pipeline),
5. fit ONE UMAP per layer across all rollouts,
6. plot two panels per layer:
      (a) success rollouts in solid blue, failure rollouts blue->red by timestep
      (b) same projection, colored by task id.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from umap import UMAP

try:
    from .action_token_pooling import (
        TOKEN_CHOICES,
        normalize_tokens,
        pool_action_tokens,
        token_pool_description,
        token_pool_tag,
    )
except ImportError:
    from action_token_pooling import (
        TOKEN_CHOICES,
        normalize_tokens,
        pool_action_tokens,
        token_pool_description,
        token_pool_tag,
    )

DATA_DIR = Path(__file__).parent
VISUALIZATION_DIR = DATA_DIR / "visualization"

NAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])\.pkl$")

SUCCESS_BLUE = "#0000ff"
FAIL_CMAP = LinearSegmentedColormap.from_list("fail_grad", ["#0000ff", "#ff0000"])

PCA_DIMS = 50
DEFAULT_N_NEIGHBORS = [15]
MIN_DIST = 0.1
METRIC = "euclidean"
RANDOM_STATE = 0


def load_rollouts(folder: Path, tokens: str):
    """Return list of dicts with keys: feats (steps, n_layers, feature_dim), task_id, success."""
    rollouts = []
    for path in sorted(folder.glob("*.pkl")):
        m = NAME_RE.search(path.name)
        if not m:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        h = (
            d["hidden_states"].to(torch.float32).numpy()
        )  # expect: (steps, 7, n_layers*hidden_dim)
        n_layers = len(d["hidden_state_layers"])
        hidden_dim = int(d["hidden_state_dim_per_layer"])
        assert h.shape[-1] == n_layers * hidden_dim, (h.shape, n_layers, hidden_dim)
        # (steps, action_tokens, n_layers*hidden_dim) -> (steps, n_layers, feature_dim)
        feats = pool_action_tokens(h, n_layers, hidden_dim, tokens)
        rollouts.append(
            {
                "feats": feats,
                "task_id": int(d["task_id"]),
                "success": bool(d["episode_success"]),
                "name": path.stem,
                "layer_indices": d["hidden_state_layers"],
            }
        )
    return rollouts


def effective_n_neighbors(n_neighbors: int, n_samples: int) -> int:
    if n_neighbors <= 1:
        raise SystemExit(f"n_neighbors must be > 1, got {n_neighbors}")
    if n_samples <= 2:
        raise SystemExit(f"UMAP needs at least 3 samples, got {n_samples}")
    eff = min(n_neighbors, n_samples - 1)
    if eff != n_neighbors:
        print(f"  requested n_neighbors={n_neighbors}; using {eff} for N={n_samples}")
    return eff


def pca_pre_reduce_layer(X: np.ndarray, random_state: int = RANDOM_STATE):
    """Apply the same per-layer PCA rule as the t-SNE per-layer script."""
    if X.shape[1] <= PCA_DIMS:
        return X.astype(np.float32), X.shape[1]
    pca_dims = min(PCA_DIMS, X.shape[0], X.shape[1])
    X_reduced = PCA(n_components=pca_dims, random_state=random_state).fit_transform(X)
    return X_reduced.astype(np.float32), pca_dims


def umap_per_layer(
    rollouts,
    n_neighbors: int = 15,
    min_dist: float = MIN_DIST,
    metric: str = METRIC,
    random_state: int = RANDOM_STATE,
):
    """Fit UMAP per layer on the concatenation of all rollouts.

    Returns (embeds, lengths, pca_dims_by_layer), where embeds maps layer slot -> (N, 2).
    """
    n_layers = rollouts[0]["feats"].shape[1]
    lengths = [r["feats"].shape[0] for r in rollouts]

    out = {}
    pca_dims_by_layer = {}
    for layer in range(n_layers):
        X = np.concatenate(
            [r["feats"][:, layer, :] for r in rollouts], axis=0
        )  # (sum_steps, hidden_dim)
        n_samples = X.shape[0]
        X_reduced, pca_dims = pca_pre_reduce_layer(X, random_state=random_state)
        pca_dims_by_layer[layer] = pca_dims
        nn = effective_n_neighbors(n_neighbors, n_samples)
        emb = UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        ).fit_transform(X_reduced)
        out[layer] = emb
        print(
            f"  layer {layer}: UMAP done "
            f"(N={n_samples}, PCA-{pca_dims}, n_neighbors={nn}, min_dist={min_dist:g})"
        )
    return out, lengths, pca_dims_by_layer


def plot_model(
    model_name,
    rollouts,
    embeds,
    lengths,
    save_path: Path,
    dataset_name: str | None = None,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    token_desc: str = "mean over action tokens",
):
    n_layers = len(embeds)
    fig, axes = plt.subplots(n_layers, 2, figsize=(9, 3.4 * n_layers), squeeze=False)

    # boundaries that split the global embed array into per-rollout slices
    starts = np.cumsum([0] + lengths)

    # global x/y limits so each pair of axes shares the same view per layer
    task_ids = sorted({r["task_id"] for r in rollouts})
    task_cmap = plt.get_cmap("tab20", max(len(task_ids), 2))
    task_color = {tid: task_cmap(i) for i, tid in enumerate(task_ids)}

    for layer, emb in embeds.items():
        ax_a, ax_b = axes[layer, 0], axes[layer, 1]

        # ---- (a) success solid blue, failure blue->red by timestep ---------
        for i, r in enumerate(rollouts):
            s, e = starts[i], starts[i + 1]
            xy = emb[s:e]
            steps = np.arange(xy.shape[0])
            t = steps / max(steps[-1], 1)
            if r["success"]:
                ax_a.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=SUCCESS_BLUE,
                    s=8,
                    alpha=0.55,
                    edgecolors="none",
                )
            else:
                ax_a.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=t,
                    cmap=FAIL_CMAP,
                    s=10,
                    alpha=0.85,
                    edgecolors="none",
                    vmin=0,
                    vmax=1,
                )
        ax_a.set_title(
            f"layer idx {rollouts[0]['layer_indices'][layer]}  |  success(blue) / fail(blue→red)"
        )
        ax_a.set_xticks([])
        ax_a.set_yticks([])

        # ---- (b) colored by task id ----------------------------------------
        for i, r in enumerate(rollouts):
            s, e = starts[i], starts[i + 1]
            xy = emb[s:e]
            ax_b.scatter(
                xy[:, 0],
                xy[:, 1],
                color=task_color[r["task_id"]],
                s=8,
                alpha=0.7,
                edgecolors="none",
            )
        ax_b.set_title(
            f"layer idx {rollouts[0]['layer_indices'][layer]}  |  by task id"
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([])

    # task-id legend on the last task-id panel
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=task_color[t],
            markersize=6,
            label=f"task {t}",
        )
        for t in task_ids
    ]
    axes[-1, 1].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )

    # legend for panel (a) on the last (a) panel
    a_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SUCCESS_BLUE,
            markersize=6,
            label="success",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FAIL_CMAP(0.0),
            markersize=6,
            label="fail t=0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FAIL_CMAP(1.0),
            markersize=6,
            label="fail t=T",
        ),
    ]
    axes[-1, 0].legend(
        handles=a_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )

    n_succ = sum(r["success"] for r in rollouts)
    n_fail = len(rollouts) - n_succ
    title_prefix = (
        model_name if dataset_name is None else f"{model_name} / {dataset_name}"
    )
    neighbor_text = "" if n_neighbors is None else f", n_neighbors={n_neighbors}"
    min_dist_text = "" if min_dist is None else f", min_dist={min_dist:g}"
    fig.suptitle(
        f"{title_prefix}: per-layer UMAP of hidden states "
        f"({len(rollouts)} rollouts: {n_succ} success / {n_fail} fail, "
        f"{token_desc}, PCA≤{PCA_DIMS}{neighbor_text}{min_dist_text})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.985))
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"saved -> {save_path}")


def slugify(value: str) -> str:
    """Make a stable filename tag from a model name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-") or "model"


def validate_data_dir(data_dir: Path) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"input directory is not a directory: {data_dir}")

    paths = sorted(p for p in data_dir.glob("*.pkl") if NAME_RE.search(p.name))
    if not paths:
        raise SystemExit(
            f"no rollouts (task<id>--ep<idx>--succ{{0,1}}.pkl) found in {data_dir}"
        )
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help="directory of .pkl rollouts; equivalent to --dir",
    )
    p.add_argument(
        "--dir",
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=Path,
        help="directory of .pkl rollouts",
    )
    p.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        required=True,
        help="model name to use in the plot title and output filename",
    )
    p.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset_name",
        default=None,
        help="optional dataset name to include in the plot title and output filename",
    )
    p.add_argument(
        "--tokens",
        type=normalize_tokens,
        choices=TOKEN_CHOICES,
        default="mean",
        help="action-token selection/pooling: mean, first, last, or first+last (default: mean)",
    )
    p.add_argument(
        "--n-neighbors",
        "--n_neighbors",
        "--neighbors",
        dest="n_neighbors",
        type=int,
        nargs="+",
        default=DEFAULT_N_NEIGHBORS,
        help="one or more UMAP n_neighbors values to try, e.g. --n-neighbors 5 15 50",
    )
    p.add_argument(
        "--min-dist",
        "--min_dist",
        dest="min_dist",
        type=float,
        default=MIN_DIST,
        help=f"UMAP min_dist (default: {MIN_DIST})",
    )
    p.add_argument(
        "--metric",
        default=METRIC,
        help=f"UMAP metric (default: {METRIC})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALIZATION_DIR,
        help=f"where to save PNGs (default: {VISUALIZATION_DIR})",
    )
    args = p.parse_args()

    if args.input_dir is not None and args.data_dir is not None:
        raise SystemExit(
            "provide either positional input_dir or --dir/--data-dir, not both"
        )
    rollout_dir = args.data_dir if args.data_dir is not None else args.input_dir
    if rollout_dir is None:
        raise SystemExit("missing rollout directory: pass --dir INPUT_DIR")

    paths = validate_data_dir(rollout_dir)
    rollout_dir = rollout_dir.expanduser().resolve()
    print(f"== {args.model_name} ==  folder: {rollout_dir}")
    print(f"found {len(paths)} matching .pkl rollout files")

    rollouts = load_rollouts(rollout_dir, args.tokens)
    token_desc = token_pool_description(args.tokens)
    token_tag = token_pool_tag(args.tokens)
    print(f"  token selection: {token_desc}")
    print(
        f"  loaded {len(rollouts)} rollouts; layer dims = "
        f"{rollouts[0]['feats'].shape[1]} layers x {rollouts[0]['feats'].shape[2]} features"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for n_neighbors in args.n_neighbors:
        print(f"running per-layer UMAP with n_neighbors={n_neighbors}")
        embeds, lengths, _ = umap_per_layer(
            rollouts,
            n_neighbors=n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
        )

        filename_parts = ["latent_umap_per_layer", slugify(args.model_name)]
        if args.dataset_name:
            filename_parts.append(slugify(args.dataset_name))
        filename_parts.append(f"tokens-{token_tag}")
        filename_parts.append(f"nn{n_neighbors}")
        save = output_dir / ("_".join(filename_parts) + ".png")
        plot_model(
            args.model_name,
            rollouts,
            embeds,
            lengths,
            save,
            args.dataset_name,
            n_neighbors,
            args.min_dist,
            token_desc,
        )


if __name__ == "__main__":
    main()
