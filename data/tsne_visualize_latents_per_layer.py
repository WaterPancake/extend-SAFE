"""Per-layer t-SNE of VLA hidden states.

Usage:
    python visualize_latents_per_layer.py --dir INPUT_DIR --model MODEL_NAME [--dataset DATASET_NAME] [--perplexity P [P ...]]

INPUT_DIR must contain .pkl rollout files named `task<id>--ep<idx>--succ{0,1}.pkl`.
Output PNGs are written to data/visualization by default, tagged with MODEL_NAME, optional DATASET_NAME, and perplexity.

Pipeline:
1. load every rollout from INPUT_DIR,
2. mean-pool the per-step hidden state across the 7 action tokens,
3. split the concatenated layer features into (n_layers, hidden_dim),
4. fit ONE t-SNE per layer across all rollouts,
5. plot two panels per layer:
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
from sklearn.manifold import TSNE

DATA_DIR = Path(__file__).parent
VISUALIZATION_DIR = DATA_DIR / "visualization"

NAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ([01])\.pkl$")

SUCCESS_BLUE = "#0000ff"
FAIL_CMAP = LinearSegmentedColormap.from_list("fail_grad", ["#0000ff", "#ff0000"])


def load_rollouts(folder: Path):
    """Return list of dicts with keys: feats (steps, n_layers, hidden_dim), task_id, success."""
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
        hidden_dim = d["hidden_state_dim_per_layer"]
        assert h.shape[-1] == n_layers * hidden_dim, (h.shape, n_layers, hidden_dim)
        # mean over action tokens -> (steps, n_layers*hidden_dim) -> (steps, n_layers, hidden_dim)
        feats = h.mean(axis=1).reshape(-1, n_layers, hidden_dim)
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


def tsne_per_layer(rollouts, perplexity=30.0, random_state=0):
    """Fit t-SNE per layer on the concatenation of all rollouts. Returns dict layer_idx -> (N, 2)."""
    n_layers = rollouts[0]["feats"].shape[1]
    lengths = [r["feats"].shape[0] for r in rollouts]

    out = {}
    for layer in range(n_layers):
        X = np.concatenate(
            [r["feats"][:, layer, :] for r in rollouts], axis=0
        )  # (sum_steps, hidden_dim)
        n_samples = X.shape[0]
        # PCA pre-reduce (standard t-SNE practice) when dim > 50
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=random_state).fit_transform(X)
        ppl = min(perplexity, max(5.0, (n_samples - 1) / 3))
        emb = TSNE(
            n_components=2,
            perplexity=ppl,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(X)
        out[layer] = emb
        print(f"  layer {layer}: t-SNE done (N={n_samples}, perplexity={ppl:.1f})")
    return out, lengths


def plot_model(
    model_name,
    rollouts,
    embeds,
    lengths,
    save_path: Path,
    dataset_name: str | None = None,
    perplexity: float | None = None,
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
    perplexity_text = "" if perplexity is None else f", perplexity={perplexity:g}"
    fig.suptitle(
        f"{title_prefix}: per-layer t-SNE of hidden states "
        f"({len(rollouts)} rollouts: {n_succ} success / {n_fail} fail, mean-pooled action tokens{perplexity_text})",
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
        "--perplexity",
        type=float,
        nargs="+",
        default=[30.0],
        help="one or more t-SNE perplexities to try, e.g. --perplexity 5 10 30 50",
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

    rollouts = load_rollouts(rollout_dir)
    print(
        f"  loaded {len(rollouts)} rollouts; layer dims = "
        f"{rollouts[0]['feats'].shape[1]} layers x {rollouts[0]['feats'].shape[2]} hidden"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for perplexity in args.perplexity:
        print(f"running per-layer t-SNE with perplexity={perplexity:g}")
        embeds, lengths = tsne_per_layer(rollouts, perplexity=perplexity)

        filename_parts = ["latent_tsne_per_layer", slugify(args.model_name)]
        if args.dataset_name:
            filename_parts.append(slugify(args.dataset_name))
        filename_parts.append(f"perp{perplexity:g}")
        save = output_dir / ("_".join(filename_parts) + ".png")
        plot_model(
            args.model_name,
            rollouts,
            embeds,
            lengths,
            save,
            args.dataset_name,
            perplexity,
        )


if __name__ == "__main__":
    main()
