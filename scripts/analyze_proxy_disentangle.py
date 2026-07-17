"""Disentangle genuine early failure detection from a length/time proxy.

Two tests, run per probe family (linear_probe / mlp) over all (layer, seed)
single-layer checkpoints:

  Test A -- fixed *absolute* timestep, instantaneous score, macro per-task.
    Among rollouts still alive at absolute step t0 (so elapsed time == t0 for
    every rollout compared), take the per-step score AT t0 (not max-up-to) and
    compute macro per-task ROC-AUC. With elapsed time held constant across all
    rollouts, any AUC > 0.5 is genuine state-based discrimination, not a clock.
    The linear probe (instantaneous sigmoid per step) is the clean version; the
    cumsum MLP accumulates, shown for reference.

  Test B -- progress-proxy check. Among SUCCESSFUL rollouts only (label fixed),
    within-task Spearman correlation of the rollout's falert_early score (max
    over [0, task_min_step]) with how long that success eventually took. A
    strong positive correlation means the probe is reading "this rollout is
    slow / not done yet" -- a progress/time signal -- rather than failure.

Scores the checkpoints on their own training data and self-validates: the
recomputed falert_early test AUC must match the stored metric.

Usage:
    uv run python scripts/analyze_proxy_disentangle.py \
        --model-type linear_probe \
        --checkpoint-dir runs/openvla_linear_layer_selection \
        --root data/rollouts/openvla_last_token \
        --output runs/openvla_layer_correlations/linear_proxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import LinearProbeModel, SafeMLPModel

ABS_TIMESTEPS = [5, 10, 20, 30, 40, 60, 80, 120, 160]
DEFAULT_REG_TAG_BY_MODEL = {
    "mlp": "0.01",
    "linear_probe": "1",
}


def default_reg_tag(model_type: str) -> str:
    return DEFAULT_REG_TAG_BY_MODEL[model_type]


def build_probe(model_type: str) -> torch.nn.Module:
    if model_type == "linear_probe":
        return LinearProbeModel(input_dim=4096)
    if model_type == "mlp":
        return SafeMLPModel(
            input_dim=4096, hidden_dim=256, n_layers=2, dropout=0.0,
            cumsum=True, n_history_steps=1, loss_type="safe", use_threshold=False,
        )
    raise ValueError(f"unsupported model_type {model_type!r}")


def macro_auc(frame: pd.DataFrame, score_col: str) -> float:
    """Mean per-task ROC-AUC over tasks that have both classes present."""
    per_task = [
        roc_auc_score(g["label"], g[score_col])
        for _, g in frame.groupby("task_id")
        if g["label"].nunique() > 1 and len(g) > 1
    ]
    return float(np.mean(per_task)) if per_task else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=["linear_probe", "mlp"], required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20, 24, 28, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--token-pool", default="last")
    parser.add_argument(
        "--reg-tag",
        default=None,
        help="lambda_reg tag in checkpoint filenames. Default is model-aware: MLP=0.01, linear=1.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    reg_tag = args.reg_tag if args.reg_tag is not None else default_reg_tag(args.model_type)

    models, splits, stored_auc = {}, {}, {}
    for layer in args.layers:
        for seed in args.seeds:
            name = (
                f"safe_{args.model_type}_single_layers_{layer}"
                f"_tok-{args.token_pool}_lr-0.0001_reg-{reg_tag}_seed-{seed}"
            )
            ckpt = torch.load(
                args.checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=False
            )
            model = build_probe(args.model_type)
            model.load_state_dict(ckpt["model_state_dict"])
            models[(layer, seed)] = model.to(device).eval()
            if "split" not in ckpt:
                raise SystemExit(f"{name}.pt has no embedded split; use a post-fix checkpoint dir")
            splits[(layer, seed)] = ckpt["split"]
            stored_auc[(layer, seed)] = ckpt["result"]["test_falert_early_roc_auc"]

    dataset = OpenVLARolloutDataset(
        args.root, layers=tuple(args.layers), token_pool=args.token_pool,
        recursive=True, cache=False,
    )
    layer_pos = {layer: i for i, layer in enumerate(dataset.selected_layers)}
    dim = dataset.hidden_dim
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_rollouts,
    )

    # one row per rollout, with columns per (layer,seed) for each test
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            masks = batch["valid_masks"].to(device)
            stops = batch["task_min_steps"].to(device).clamp_min(1)
            lengths = batch["lengths"]
            n, max_len = masks.shape
            batch_rows = [
                {"rollout_idx": idx + r, "task_id": batch["task_ids"][r],
                 "label": float(batch["labels"][r]),
                 "success": float(batch["success"][r]),
                 "length": int(lengths[r])}
                for r in range(n)
            ]
            for layer in args.layers:
                pos = layer_pos[layer]
                lf = features[:, :, pos * dim : (pos + 1) * dim]
                for seed in args.seeds:
                    raw = models[(layer, seed)]({"features": lf}).squeeze(-1)  # (n, T)
                    # falert_early = max of per-step score up to task_min_step
                    tmask = torch.arange(max_len, device=device).unsqueeze(0) < stops.unsqueeze(1)
                    early = raw.masked_fill(~(masks & tmask), float("-inf")).max(dim=1).values.cpu()
                    for r in range(n):
                        batch_rows[r][f"early_l{layer}_s{seed}"] = float(early[r])
                    # instantaneous score at each fixed absolute timestep (if alive)
                    for t0 in ABS_TIMESTEPS:
                        if t0 >= max_len:
                            continue
                        vals = raw[:, t0].cpu()
                        for r in range(n):
                            alive = lengths[r] > t0
                            batch_rows[r][f"abs{t0}_l{layer}_s{seed}"] = (
                                float(vals[r]) if alive else float("nan")
                            )
            rows.extend(batch_rows)
            idx += n
            print(f"scored {idx}/{len(dataset)} rollouts", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output / "proxy_scores.csv", index=False)

    # --- self-validation: falert_early test AUC vs stored ---
    max_err = 0.0
    for layer in args.layers:
        for seed in args.seeds:
            test = df[df.rollout_idx.isin(splits[(layer, seed)]["test"])]
            rec = roc_auc_score(test["label"], test[f"early_l{layer}_s{seed}"])
            max_err = max(max_err, abs(rec - stored_auc[(layer, seed)]))
    print(f"\nValidation: max |recomputed falert_early - stored| = {max_err:.4f}")
    if max_err > 0.01:
        raise SystemExit("ABORT: scores do not reproduce stored metrics; root mismatch")
    print("  OK\n")

    # ============ Test A: fixed absolute timestep, macro per-task ============
    # Evaluated over ALL rollouts grouped by task (max diagnostic power: does
    # any instantaneous early signal exist), and over test-task rollouts only
    # (generalization). AUC per (layer,seed) then averaged over seeds.
    a_rows = []
    for layer in args.layers:
        for seed in args.seeds:
            test_idx = set(splits[(layer, seed)]["test"])
            for t0 in ABS_TIMESTEPS:
                col = f"abs{t0}_l{layer}_s{seed}"
                if col not in df.columns:
                    continue
                for scope, frame in [
                    ("all", df.dropna(subset=[col])),
                    ("test", df[df.rollout_idx.isin(test_idx)].dropna(subset=[col])),
                ]:
                    n_alive = len(frame)
                    n_fail_alive = int((frame["label"] == 1).sum())
                    a_rows.append({
                        "layer": layer, "seed": seed, "t0": t0, "scope": scope,
                        "macro_auc": macro_auc(frame, col),
                        "n_alive": n_alive, "n_fail_alive": n_fail_alive,
                    })
    a = pd.DataFrame(a_rows)
    a.to_csv(args.output / "testA_fixed_abs_timestep.csv", index=False)

    for scope in ["all", "test"]:
        piv = (a[a.scope == scope].groupby(["layer", "t0"]).macro_auc.mean()
               .unstack("t0").round(3))
        print(f"\n=== TEST A ({scope} rollouts) macro per-task AUC: layer x absolute timestep ===")
        print(piv.to_string())
        nalive = a[a.scope == scope].groupby("t0")[["n_alive", "n_fail_alive"]].mean().round(0)
        print("mean rollouts alive at t0 (total / failures):")
        print(nalive.to_string())

    # ============ Test B: progress proxy among successes ============
    succ = df[df.success == 1.0]
    b_rows = []
    for layer in args.layers:
        for seed in args.seeds:
            col = f"early_l{layer}_s{seed}"
            per_task = []
            for _, g in succ.groupby("task_id"):
                if g["length"].nunique() > 1 and len(g) > 2:
                    rho, _ = spearmanr(g[col], g["length"])
                    if not np.isnan(rho):
                        per_task.append(rho)
            b_rows.append({
                "layer": layer, "seed": seed,
                "mean_spearman_score_vs_successlen": float(np.mean(per_task)) if per_task else float("nan"),
                "n_tasks": len(per_task),
            })
    b = pd.DataFrame(b_rows)
    b.to_csv(args.output / "testB_progress_proxy.csv", index=False)
    print("\n=== TEST B: within-task Spearman(falert_early score, success length) "
          "among SUCCESSES (mean over tasks, then seeds) ===")
    print(b.groupby("layer").mean_spearman_score_vs_successlen.mean().round(3).to_string())
    print("\n(Positive => probe reads slowness/progress, not failure semantics.)")


if __name__ == "__main__":
    main()
