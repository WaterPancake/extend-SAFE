"""UMAP visualization of all rollouts in a directory (SAFE Fig. 1 style).

Per step: select/pool the action-token dimension of `hidden_states` with --tokens
({mean,first,last,first+last}) -> one per-layer feature vector. Stride within each
rollout to keep total points tractable, PCA-pre-reduce to 50d (matching the t-SNE
pooled pipeline), then fit one UMAP across all rollouts.

Usage:
    python umap_visualize_latents_pooled.py [INPUT_DIR] [--layer {all,last,<int>}] [--tokens {mean,first,last,first+last}] [--n-neighbors N [N ...]]

INPUT_DIR must contain .pkl rollout files named `task<id>--ep<idx>--succ{0,1}.pkl`.
Output PNGs are written to data/visualization, tagged with the input dir name, layer
choice, token choice, and n_neighbors value.
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
DEFAULT_ROLLOUT_DIR = DATA_DIR / "openvla_mini_rollout" / "LIBERO_90"

NAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])\.pkl$")
SUCCESS_BLUE = "#1f4e8c"
FAIL_CMAP = LinearSegmentedColormap.from_list("fail_grad", ["#1f4e8c", "#c0392b"])

STRIDE = 4  # keep every 4th step within each rollout
PCA_DIMS = 50  # pre-reduction before UMAP, matching t-SNE pooled script
DEFAULT_N_NEIGHBORS = [15]
MIN_DIST = 0.1
METRIC = "euclidean"
RANDOM_STATE = 0


def load_rollout(path: Path, stride: int, tokens: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    h = d["hidden_states"].to(torch.float32).numpy()
    n_layers = len(d["hidden_state_layers"])
    hidden_dim = int(d["hidden_state_dim_per_layer"])
    # (steps, action_tokens, n_layers*hidden_dim) -> (steps, n_layers, feature_dim)
    feats = pool_action_tokens(h, n_layers, hidden_dim, tokens)
    T = feats.shape[0]
    keep_idx = np.arange(0, T, stride)
    return {
        "feats": feats[keep_idx],
        "step_frac": keep_idx / max(T - 1, 1),
        "task_id": int(d["task_id"]),
        "success": bool(d["episode_success"]),
        "name": path.stem,
        "layer_indices": list(d["hidden_state_layers"]),
        "hidden_dim": int(feats.shape[-1]),
    }


def slice_features(rollouts, layer_arg: str):
    """Return (X_per_rollout: list[(T_i, D)], description: str, tag: str)."""
    n_layers = rollouts[0]["feats"].shape[1]
    layer_indices = rollouts[0]["layer_indices"]
    hidden_dim = rollouts[0]["hidden_dim"]
    if layer_arg == "all":
        sliced = [r["feats"].reshape(r["feats"].shape[0], -1) for r in rollouts]
        desc = f"all {n_layers} layers concatenated = {n_layers * hidden_dim} dims"
        tag = "all"
    elif layer_arg == "last":
        sliced = [r["feats"][:, -1, :] for r in rollouts]
        desc = (
            f"last layer only (model layer idx {layer_indices[-1]}, {hidden_dim} dims)"
        )
        tag = f"last_layer{layer_indices[-1]}"
    else:
        idx = int(layer_arg)
        sliced = [r["feats"][:, idx, :] for r in rollouts]
        desc = f"layer slot {idx} (model layer idx {layer_indices[idx]}, {hidden_dim} dims)"
        tag = f"layer{layer_indices[idx]}"
    return sliced, desc, tag


def pca_pre_reduce(X: np.ndarray, random_state: int = RANDOM_STATE):
    """PCA-pre-reduce to the same target dimensionality used by t-SNE pooled."""
    pca_dims = min(PCA_DIMS, X.shape[1], X.shape[0])
    print(f"  PCA -> {pca_dims}d ...")
    X_reduced = PCA(n_components=pca_dims, random_state=random_state).fit_transform(X)
    return X_reduced.astype(np.float32), pca_dims


def effective_n_neighbors(n_neighbors: int, n_samples: int) -> int:
    if n_neighbors <= 1:
        raise SystemExit(f"n_neighbors must be > 1, got {n_neighbors}")
    if n_samples <= 2:
        raise SystemExit(f"UMAP needs at least 3 samples, got {n_samples}")
    eff = min(n_neighbors, n_samples - 1)
    if eff != n_neighbors:
        print(f"  requested n_neighbors={n_neighbors}; using {eff} for N={n_samples}")
    return eff


def plot_embedding(
    xy: np.ndarray,
    rollouts,
    starts: np.ndarray,
    step_frac: np.ndarray,
    feat_desc: str,
    token_desc: str,
    dir_tag: str,
    pca_dims: int,
    n_neighbors: int,
    min_dist: float,
    save_path: Path,
):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    # ---- (a) success solid blue, failure blue->red by normalized timestep -------
    for i, r in enumerate(rollouts):
        s, e = starts[i], starts[i + 1]
        pts = xy[s:e]
        if r["success"]:
            ax_a.scatter(
                pts[:, 0], pts[:, 1], c=SUCCESS_BLUE, s=4, alpha=0.30, edgecolors="none"
            )
        else:
            ax_a.scatter(
                pts[:, 0],
                pts[:, 1],
                c=step_frac[s:e],
                cmap=FAIL_CMAP,
                s=5,
                alpha=0.65,
                edgecolors="none",
                vmin=0,
                vmax=1,
            )
    n_succ = sum(r["success"] for r in rollouts)
    n_fail = len(rollouts) - n_succ
    ax_a.set_title(f"success ({n_succ}, blue) / fail ({n_fail}, blue→red by timestep)")
    ax_a.set_xlabel("UMAP 1")
    ax_a.set_ylabel("UMAP 2")
    ax_a.legend(
        handles=[
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
        ],
        loc="best",
        fontsize=8,
        frameon=False,
    )

    # ---- (b) colored by task id ------------------------------------------------
    task_ids = sorted({r["task_id"] for r in rollouts})
    cmap = plt.get_cmap("tab20", max(len(task_ids), 2))
    task_color = {tid: cmap(i) for i, tid in enumerate(task_ids)}
    for i, r in enumerate(rollouts):
        s, e = starts[i], starts[i + 1]
        pts = xy[s:e]
        ax_b.scatter(
            pts[:, 0],
            pts[:, 1],
            color=task_color[r["task_id"]],
            s=4,
            alpha=0.40,
            edgecolors="none",
        )
    ax_b.set_title(f"by task id ({len(task_ids)} tasks)")
    ax_b.set_xlabel("UMAP 1")
    if len(task_ids) <= 20:
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
        ax_b.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=False,
            ncol=1,
        )

    fig.suptitle(
        f"{dir_tag}: per-step hidden state ({token_desc}, "
        f"{feat_desc}) → PCA-{pca_dims} → UMAP-2D\n"
        f"{len(rollouts)} rollouts, {xy.shape[0]} sampled steps (stride {STRIDE}), "
        f"n_neighbors={n_neighbors}, min_dist={min_dist:g}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.93))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_ROLLOUT_DIR),
        help=f"directory of .pkl rollouts (default: {DEFAULT_ROLLOUT_DIR})",
    )
    p.add_argument(
        "--layer",
        default="all",
        help="'all' (default), 'last', or an integer slot into selected layers",
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

    rollout_dir = Path(args.input_dir).expanduser().resolve()
    if not rollout_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {rollout_dir}")

    paths = sorted(pp for pp in rollout_dir.glob("*.pkl") if NAME_RE.search(pp.name))
    if not paths:
        raise SystemExit(
            f"no rollouts (task<id>--ep<idx>--succ{{0,1}}.pkl) found in {rollout_dir}"
        )
    print(f"loading {len(paths)} rollouts from {rollout_dir} (stride={STRIDE})")
    rollouts = [load_rollout(pp, STRIDE, args.tokens) for pp in paths]
    sliced, feat_desc, tag = slice_features(rollouts, args.layer)
    token_desc = token_pool_description(args.tokens)
    token_tag = token_pool_tag(args.tokens)
    print(f"  token selection: {token_desc}")
    print(f"  feature slice: {feat_desc}")
    dir_tag = rollout_dir.name
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = [s.shape[0] for s in sliced]
    starts = np.cumsum([0] + lengths)
    X = np.concatenate(sliced, axis=0)
    step_frac = np.concatenate([r["step_frac"] for r in rollouts], axis=0)
    print(f"  pooled matrix: {X.shape}")

    X_pca, pca_dims = pca_pre_reduce(X, RANDOM_STATE)

    for requested_neighbors in args.n_neighbors:
        nn = effective_n_neighbors(requested_neighbors, X_pca.shape[0])
        print(f"  UMAP (n_neighbors={nn}, min_dist={args.min_dist:g}, metric={args.metric}) ...")
        xy = UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=RANDOM_STATE,
        ).fit_transform(X_pca)

        out_path = output_dir / f"latent_space_viz_umap__{dir_tag}__{tag}__tokens-{token_tag}__nn{requested_neighbors}.png"
        plot_embedding(
            xy,
            rollouts,
            starts,
            step_frac,
            feat_desc,
            token_desc,
            dir_tag,
            pca_dims,
            requested_neighbors,
            args.min_dist,
            out_path,
        )


if __name__ == "__main__":
    main()
