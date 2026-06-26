"""Pooled and macro-per-task test falert_early AUC for one layer, any family.

For a single model layer across all seeds of a probe family, scores rollouts,
recomputes falert_early (max per-step score up to task_min_step) on each
checkpoint's embedded test split, and reports pooled and macro per-task ROC-AUC.
Self-validates pooled vs the stored metric.

Usage:
    uv run python scripts/score_layer_pooled_macro.py --model-type lstm \
        --checkpoint-dir runs/openvla_lstm_layer_selection_1k \
        --root data/rollouts --layer 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import LinearProbeModel, SafeLSTMModel, SafeMLPModel


def build_probe(model_type: str) -> torch.nn.Module:
    if model_type == "linear_probe":
        return LinearProbeModel(input_dim=4096)
    if model_type == "lstm":
        return SafeLSTMModel(input_dim=4096, hidden_dim=256, n_layers=1,
                             dropout=0.0, loss_type="bce", use_threshold=False)
    if model_type == "mlp":
        return SafeMLPModel(input_dim=4096, hidden_dim=256, n_layers=2, dropout=0.0,
                            cumsum=True, n_history_steps=1, loss_type="safe", use_threshold=False)
    raise ValueError(model_type)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=["lstm", "mlp", "linear_probe"], required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--layer", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--reg-tag", default="1",
                   help="lambda_reg tag in the checkpoint filename (e.g. '1' or '0.01').")
    p.add_argument("--all-layers", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20, 24, 28, 32])
    p.add_argument("--token-pool", default="last")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    prefix = {"lstm": "safe_lstm", "mlp": "safe_mlp", "linear_probe": "safe_linear_probe"}[args.model_type]

    models, splits, stored = {}, {}, {}
    for s in args.seeds:
        name = f"{prefix}_single_layers_{args.layer}_tok-{args.token_pool}_lr-0.0001_reg-{args.reg_tag}_seed-{s}"
        c = torch.load(args.checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        m = build_probe(args.model_type)
        m.load_state_dict(c["model_state_dict"])
        models[s] = m.to(device).eval()
        splits[s] = c["split"]
        stored[s] = c["result"]["test_falert_early_roc_auc"]

    ds = OpenVLARolloutDataset(args.root, layers=tuple(args.all_layers),
                               token_pool=args.token_pool, recursive=True, cache=False)
    pos = {l: i for i, l in enumerate(ds.selected_layers)}[args.layer]
    dim = ds.hidden_dim
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_rollouts)

    rows, idx = [], 0
    with torch.no_grad():
        for b in loader:
            feats = b["features"].to(device)
            masks = b["valid_masks"].to(device)
            stops = b["task_min_steps"].to(device).clamp_min(1)
            n, T = masks.shape
            lf = feats[:, :, pos * dim:(pos + 1) * dim]
            tmask = torch.arange(T, device=device).unsqueeze(0) < stops.unsqueeze(1)
            br = [{"idx": idx + r, "task_id": b["task_ids"][r], "label": float(b["labels"][r])} for r in range(n)]
            for s in args.seeds:
                raw = models[s]({"features": lf}).squeeze(-1)
                early = raw.masked_fill(~(masks & tmask), float("-inf")).max(dim=1).values.cpu()
                for r in range(n):
                    br[r][f"early_s{s}"] = float(early[r])
            rows.extend(br)
            idx += n
    df = pd.DataFrame(rows)

    print(f"\n{args.model_type} layer {args.layer}  (n_rollouts={len(df)})")
    print(f"{'seed':>5} {'pooled':>8} {'macro':>8} {'stored':>8} {'|err|':>7}")
    pooled_all, macro_all = [], []
    for s in args.seeds:
        test = df[df["idx"].isin(splits[s]["test"])]
        col = f"early_s{s}"
        pooled = roc_auc_score(test["label"], test[col])
        per_task = [roc_auc_score(g["label"], g[col]) for _, g in test.groupby("task_id")
                    if g["label"].nunique() > 1 and len(g) > 1]
        macro = float(np.mean(per_task))
        err = abs(pooled - stored[s])
        pooled_all.append(pooled); macro_all.append(macro)
        flag = "  <-- MISMATCH" if err > 0.01 else ""
        print(f"{s:>5} {pooled:>8.4f} {macro:>8.4f} {stored[s]:>8.4f} {err:>7.4f}{flag}")
    print(f"{'mean':>5} {np.mean(pooled_all):>8.4f} {np.mean(macro_all):>8.4f}")
    print(f"  pooled {np.mean(pooled_all):.3f}±{np.std(pooled_all):.3f}   "
          f"macro {np.mean(macro_all):.3f}±{np.std(macro_all):.3f}")
    if max(abs(p - stored[s]) for s, p in zip(args.seeds, pooled_all)) > 0.01:
        raise SystemExit("ABORT: pooled does not match stored; root/split mismatch")


if __name__ == "__main__":
    main()
