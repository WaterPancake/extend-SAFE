"""Supplemental RBF-kernel CKA between the 9 OpenVLA layers' rollout-pooled features.

The linear CKA table (docs/results_audit/linear_cka_table.csv) shows that
representations de-correlate with depth (L1-L32 CKA = 0.447, L24-L32 = 0.960).
Linear CKA only captures second-order (covariance) similarity; RBF-kernel CKA
captures nonlinear structure via a Gaussian kernel on pairwise distances.

This script computes both linear and RBF CKA on the same 1,000-rollout,
9-layer, early-window-pooled features and writes a combined table to
docs/results_audit/rbf_cka_table.csv.

Usage:
    uv run python scripts/compute_rbf_cka.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 32]
LAYER_OFFSETS = {l: i * 4096 for i, l in enumerate(LAYERS)}
SHARD_PATHS = [
    Path("data/processed/shard1"),
    Path("data/processed/OpenVLA_500_2"),
]
OUTPUT_CSV = Path("docs/results_audit/rbf_cka_table.csv")


def linear_cka(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Linear CKA = <K_a, K_b>_F / (||K_a||_F * ||K_b||_F) where K = X X^T."""
    Xa = Xa - Xa.mean(0, keepdims=True)
    Xb = Xb - Xb.mean(0, keepdims=True)
    Ka = Xa @ Xa.T
    Kb = Xb @ Xb.T
    return float((Ka * Kb).sum() / (np.linalg.norm(Ka) * np.linalg.norm(Kb)))


def rbf_cka(Xa: np.ndarray, Xb: np.ndarray, gamma: float | None = None) -> float:
    """RBF-kernel CKA.

    Uses a Gaussian (RBF) kernel: K(x, y) = exp(-gamma * ||x - y||^2).
    If gamma is None, uses the median heuristic: gamma = 1 / median(||x_i - x_j||^2).

    CKA = <K_a, K_b>_F / (||K_a||_F * ||K_b||_F) where K = rbf_kernel(X).
    """
    Xa = Xa - Xa.mean(0, keepdims=True)
    Xb = Xb - Xb.mean(0, keepdims=True)

    # Compute squared Euclidean distances for each set
    def sq_dist(X: np.ndarray) -> np.ndarray:
        sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        return sq + sq.T - 2 * (X @ X.T)

    Da = sq_dist(Xa)
    Db = sq_dist(Xb)

    # Median heuristic for gamma
    if gamma is None:
        # Use the combined median of both distance matrices
        median_a = np.median(Da[Da > 0]) if np.any(Da > 0) else 1.0
        median_b = np.median(Db[Db > 0]) if np.any(Db > 0) else 1.0
        gamma = 1.0 / max(median_a, median_b)

    Ka = np.exp(-gamma * np.maximum(Da, 0))
    Kb = np.exp(-gamma * np.maximum(Db, 0))

    # Center the kernel matrices (HSIC normalization)
    n = Xa.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Ka_c = H @ Ka @ H
    Kb_c = H @ Kb @ H

    hsic_ab = (Ka_c * Kb_c).sum()
    hsic_aa = (Ka_c * Ka_c).sum()
    hsic_bb = (Kb_c * Kb_c).sum()

    if hsic_aa <= 0 or hsic_bb <= 0:
        return 0.0

    return float(hsic_ab / np.sqrt(hsic_aa * hsic_bb))


def load_and_pool_rollouts(shard_path: Path, layers: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load all rollouts from a shard, pool over early window, return (features, labels, metadata).

    Returns:
        pooled: (n_rollouts, n_layers, 4096) — mean over [0, task_min_step]
        labels: (n_rollouts,) — 0 = success, 1 = failure
        metadata: dict with task_id, episode_idx, success, length, task_min_step
    """
    files = sorted(shard_path.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files under {shard_path}")

    pooled_list = []
    labels_list = []
    meta_list = []

    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        hs = data["hidden_states"]  # (T, tokens, 36864) or (T, 36864)
        layers_saved = data["hidden_state_layers"]
        dim_per_layer = data.get("hidden_state_dim_per_layer", 4096)
        token_idx = data.get("hidden_state_token_index", None)
        success = data.get("episode_success", None)
        if success is None:
            # Parse from filename
            name = fp.stem
            success = int("succ1" in name)

        # If token axis present, select last token
        if hs.ndim == 3 and token_idx is None:
            # (T, tokens, layers*dim) — select last token
            hs = hs[:, -1, :]
        elif hs.ndim == 3 and token_idx is not None:
            # Already has token axis but with a specific index — use last
            if hs.shape[1] > 1:
                hs = hs[:, -1, :]

        # Now hs is (T, layers*dim) — extract each layer
        T = hs.shape[0]
        task_min_step = T  # For CKA, pool over the full trajectory (all rollouts alive)

        layer_features = np.zeros((len(layers), dim_per_layer), dtype=np.float32)
        for li, L in enumerate(layers):
            if L not in layers_saved:
                raise ValueError(f"Layer {L} not in saved layers {layers_saved} for {fp}")
            idx = layers_saved.index(L)
            start = idx * dim_per_layer
            end = start + dim_per_layer
            layer_hs_raw = hs[:, start:end]
            if hasattr(layer_hs_raw, 'float'):
                layer_hs_raw = layer_hs_raw.float()
            if hasattr(layer_hs_raw, 'numpy'):
                layer_hs = layer_hs_raw.numpy().astype(np.float32)
            else:
                layer_hs = np.asarray(layer_hs_raw, dtype=np.float32)
            # Pool over full trajectory
            layer_features[li] = layer_hs.mean(0)

        pooled_list.append(layer_features)
        labels_list.append(0 if success else 1)
        meta_list.append({
            "task": data.get("task_id", fp.stem.split("--")[0]),
            "episode": data.get("eposide_idx", None),
            "success": success,
            "length": T,
        })

    pooled = np.stack(pooled_list, axis=0)  # (n, n_layers, 4096)
    labels = np.array(labels_list, dtype=np.float32)
    return pooled, labels, {"n_rollouts": len(files), "shard": str(shard_path)}


def main() -> None:
    print("Loading shards and pooling over full trajectories...")
    all_pooled = []
    all_labels = []
    for shard_path in SHARD_PATHS:
        if not shard_path.exists():
            print(f"  WARNING: {shard_path} not found, skipping")
            continue
        pooled, labels, meta = load_and_pool_rollouts(shard_path, LAYERS)
        print(f"  {shard_path.name}: {meta['n_rollouts']} rollouts, shape {pooled.shape}")
        all_pooled.append(pooled)
        all_labels.append(labels)

    if not all_pooled:
        raise FileNotFoundError("No shard data found")

    pooled = np.concatenate(all_pooled, axis=0)  # (1000, 9, 4096)
    labels = np.concatenate(all_labels)
    n = pooled.shape[0]
    print(f"\nTotal: {n} rollouts, {len(LAYERS)} layers, {pooled.shape[2]} dims per layer")

    # Standardize each layer's features
    for li in range(len(LAYERS)):
        scaler = StandardScaler()
        pooled[:, li, :] = scaler.fit_transform(pooled[:, li, :])

    # Compute linear CKA (for reference — should match existing table)
    print("\n=== Linear CKA ===")
    lin_cka = np.eye(len(LAYERS))
    for i in range(len(LAYERS)):
        for j in range(i + 1, len(LAYERS)):
            lin_cka[i, j] = lin_cka[j, i] = linear_cka(pooled[:, i, :], pooled[:, j, :])

    print("      " + "  ".join(f"L{L:>2}" for L in LAYERS))
    for i, L in enumerate(LAYERS):
        print(f"L{L:>2}  " + "  ".join(f"{lin_cka[i,j]:.3f}" for j in range(len(LAYERS))))

    # Compute RBF-kernel CKA
    print("\n=== RBF-kernel CKA (median heuristic gamma) ===")
    rbf_cka_mat = np.eye(len(LAYERS))
    for i in range(len(LAYERS)):
        for j in range(i + 1, len(LAYERS)):
            rbf_cka_mat[i, j] = rbf_cka_mat[j, i] = rbf_cka(pooled[:, i, :], pooled[:, j, :])

    print("      " + "  ".join(f"L{L:>2}" for L in LAYERS))
    for i, L in enumerate(LAYERS):
        print(f"L{L:>2}  " + "  ".join(f"{rbf_cka_mat[i,j]:.3f}" for j in range(len(LAYERS))))

    # Key comparisons
    print(f"\n  L1-L32:  linear={lin_cka[0, -1]:.3f}  rbf={rbf_cka_mat[0, -1]:.3f}")
    print(f"  L4-L32:  linear={lin_cka[1, -1]:.3f}  rbf={rbf_cka_mat[1, -1]:.3f}")
    print(f"  L8-L32:  linear={lin_cka[2, -1]:.3f}  rbf={rbf_cka_mat[2, -1]:.3f}")
    print(f"  L20-L32: linear={lin_cka[5, -1]:.3f}  rbf={rbf_cka_mat[5, -1]:.3f}")
    print(f"  L24-L32: linear={lin_cka[7, -1]:.3f}  rbf={rbf_cka_mat[7, -1]:.3f}")
    print(f"  L28-L32: linear={lin_cka[8, -1]:.3f}  rbf={rbf_cka_mat[8, -1]:.3f}")

    # Write combined CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ",linear_" + ",linear_".join(f"L{L}" for L in LAYERS) + ",rbf_" + ",rbf_".join(f"L{L}" for L in LAYERS)
    lines = [",".join(f"L{L}" for L in LAYERS)]
    for i, L in enumerate(LAYERS):
        row = [f"L{L}"]
        row.extend(f"{lin_cka[i,j]:.6f}" for j in range(len(LAYERS)))
        row.extend(f"{rbf_cka_mat[i,j]:.6f}" for j in range(len(LAYERS)))
        lines.append(",".join(row))

    # Write as proper CSV with layer index columns
    import csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"L{L}" for L in LAYERS])
        for i, L in enumerate(LAYERS):
            writer.writerow([f"L{L}"] + [f"{lin_cka[i,j]:.6f}" for j in range(len(LAYERS))])
        writer.writerow([])
        writer.writerow(["RBF"] + [f"L{L}" for L in LAYERS])
        for i, L in enumerate(LAYERS):
            writer.writerow([f"L{L}"] + [f"{rbf_cka_mat[i,j]:.6f}" for j in range(len(LAYERS))])

    print(f"\nWrote {OUTPUT_CSV}")

    # Also write a combined comparison CSV
    comparison_path = Path("docs/results_audit/cka_linear_vs_rbf.csv")
    with open(comparison_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_a", "layer_b", "linear_cka", "rbf_cka"])
        for i in range(len(LAYERS)):
            for j in range(i + 1, len(LAYERS)):
                writer.writerow([f"L{LAYERS[i]}", f"L{LAYERS[j]}", f"{lin_cka[i,j]:.6f}", f"{rbf_cka_mat[i,j]:.6f}"])
    print(f"Wrote {comparison_path}")


if __name__ == "__main__":
    main()
