"""t-SNE visualization of all rollouts in a directory (SAFE Fig. 1 style).

Per step: mean-pool the 7 action tokens of `hidden_states` -> one (n_layers*hidden_dim)
vector that concatenates the selected layers. Stride within each rollout to keep total
points tractable, PCA-pre-reduce to 50d, then fit one t-SNE across all rollouts.

Usage:
    python visualize_latents_tsne.py [INPUT_DIR] [--layer {all,last,<int>}]

INPUT_DIR must contain .pkl rollout files named `task<id>--ep<idx>--succ{0,1}.pkl`.
Output PNG is written to data/visualization, tagged with the input dir name and layer choice.
"""

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

try:
    from cuml.manifold import TSNE

    _TSNE_KW = dict(learning_rate=200.0)
except ImportError:
    from sklearn.manifold import TSNE

    _TSNE_KW = dict(init="pca", learning_rate="auto")

DATA_DIR = Path(__file__).parent
VISUALIZATION_DIR = DATA_DIR / "visualization"
DEFAULT_ROLLOUT_DIR = DATA_DIR / "openvla_mini_rollout" / "LIBERO_90"

NAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])\.pkl$")
SUCCESS_BLUE = "#1f4e8c"
FAIL_CMAP = LinearSegmentedColormap.from_list("fail_grad", ["#1f4e8c", "#c0392b"])

STRIDE = 4  # keep every 4th step within each rollout
PCA_DIMS = 50  # pre-reduction before t-SNE
PERPLEXITY = 20
RANDOM_STATE = 0


def load_rollout(path: Path, stride: int):
    with open(path, "rb") as f:
        d = pickle.load(f)
    # (steps, action_tokens=7, n_layers*hidden_dim) -> mean over action tokens
    h = d["hidden_states"].to(torch.float32).numpy().mean(axis=1)
    n_layers = len(d["hidden_state_layers"])
    hidden_dim = int(d["hidden_state_dim_per_layer"])
    # (steps, n_layers, hidden_dim)
    feats = h.reshape(h.shape[0], n_layers, hidden_dim)
    T = feats.shape[0]
    keep_idx = np.arange(0, T, stride)
    return {
        "feats": feats[keep_idx],
        "step_frac": keep_idx / max(T - 1, 1),
        "task_id": int(d["task_id"]),
        "success": bool(d["episode_success"]),
        "name": path.stem,
        "layer_indices": list(d["hidden_state_layers"]),
        "hidden_dim": hidden_dim,
    }


def slice_features(rollouts, layer_arg: str):
    """Return (X_per_rollout: list[(T_i, D)], description: str) for the chosen layer slice."""
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
    rollouts = [load_rollout(pp, STRIDE) for pp in paths]
    sliced, feat_desc, tag = slice_features(rollouts, args.layer)
    print(f"  feature slice: {feat_desc}")
    dir_tag = rollout_dir.name
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH = VISUALIZATION_DIR / f"latent_space_viz_tsne__{dir_tag}__{tag}.png"

    lengths = [s.shape[0] for s in sliced]
    starts = np.cumsum([0] + lengths)
    X = np.concatenate(sliced, axis=0)
    step_frac = np.concatenate([r["step_frac"] for r in rollouts], axis=0)
    print(f"  pooled matrix: {X.shape}")

    pca_dims = min(PCA_DIMS, X.shape[1])
    print(f"  PCA -> {pca_dims}d ...")
    X50 = PCA(n_components=pca_dims, random_state=RANDOM_STATE).fit_transform(X)

    print(f"  t-SNE (perplexity={PERPLEXITY}) ...")
    xy = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        random_state=RANDOM_STATE,
        **_TSNE_KW,
    ).fit_transform(X50.astype(np.float32))

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
    ax_a.set_xlabel("t-SNE 1")
    ax_a.set_ylabel("t-SNE 2")
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
    ax_b.set_xlabel("t-SNE 1")
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
        f"{dir_tag}: per-step hidden state (mean over action tokens, "
        f"{feat_desc}) → PCA-{pca_dims} → t-SNE-2D\n"
        f"{len(rollouts)} rollouts, {X.shape[0]} sampled steps (stride {STRIDE}), "
        f"perplexity={PERPLEXITY}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.93))
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
