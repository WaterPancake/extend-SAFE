"""Sharper gate: does layer-k carry FAILURE-relevant signal layer-32 misses,
found by a SUPERVISED projection (not variance-ranked PCA)?

For each LOTO fold (layer-32 MLP trained on 9 tasks, 1 held out), two augmentations
of the layer-32 score s32, both evaluated on the held-out task:
  (A) PCA-100(feat_k)  -- unsupervised robustness vs the PCA-30 test (more variance).
  (B) PLS-5(feat_k -> layer-32 residual) -- supervised: directions in layer-k most
      correlated with what layer-32 gets WRONG (catches low-variance failure dirs).
base = logistic(s32); aug = logistic(s32 + proj). Report mean ΔAUC over 10 folds.

Pooled features + s32 scores are cached to npz so this can be re-run cheaply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import SafeMLPModel

ROOT = Path("data/rollouts")
CKPT_DIR = Path("runs/openvla_mlp_loto_l32")
CACHE = Path("runs/openvla_layer_correlations/complementarity_cache.npz")
LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 32]
FOLDS = list(range(10))


def build_mlp():
    return SafeMLPModel(input_dim=4096, hidden_dim=256, n_layers=2, dropout=0.0,
                        cumsum=True, n_history_steps=1, loss_type="safe", use_threshold=False)


def load_probes_splits():
    probes, splits = {}, {}
    for f in FOLDS:
        name = f"safe_mlp_single_layers_32_tok-last_lr-0.0001_reg-0.01_seed-{f}"
        c = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
        m = build_mlp(); m.load_state_dict(c["model_state_dict"]); probes[f] = m.eval()
        splits[f] = c["split"]
    return probes, splits


def build_cache(splits):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probes, _ = load_probes_splits()
    for f in FOLDS:
        probes[f] = probes[f].to(device)
    ds = OpenVLARolloutDataset(ROOT, layers=tuple(LAYERS), token_pool="last", recursive=True, cache=False)
    pos = {l: i for i, l in enumerate(ds.selected_layers)}; dim = ds.hidden_dim
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_rollouts)
    n = len(ds)
    pooled = np.zeros((n, len(LAYERS), dim), np.float32)
    labels = np.zeros(n, np.float32)
    s32 = np.zeros((len(FOLDS), n), np.float32)
    idx = 0
    with torch.no_grad():
        for b in loader:
            feats = b["features"].to(device); masks = b["valid_masks"].to(device)
            stops = b["task_min_steps"].to(device).clamp_min(1)
            nb, T = masks.shape
            tmask = (torch.arange(T, device=device).unsqueeze(0) < stops.unsqueeze(1)) & masks
            denom = tmask.sum(1).clamp_min(1).unsqueeze(1).float()
            for li, L in enumerate(LAYERS):
                lf = feats[:, :, pos[L]*dim:(pos[L]+1)*dim]
                pooled[idx:idx+nb, li, :] = ((lf*tmask.unsqueeze(-1)).sum(1)/denom).cpu().numpy()
            l32 = feats[:, :, pos[32]*dim:(pos[32]+1)*dim]
            for f in FOLDS:
                raw = probes[f]({"features": l32}).squeeze(-1)
                s32[f, idx:idx+nb] = raw.masked_fill(~tmask, float("-inf")).max(1).values.cpu().numpy()
            labels[idx:idx+nb] = b["labels"].numpy()
            idx += nb; print(f"scored {idx}/{n}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, pooled=pooled, labels=labels, s32=s32)
    return pooled, labels, s32


def main():
    _, splits = load_probes_splits()
    if CACHE.exists():
        d = np.load(CACHE); pooled, labels, s32 = d["pooled"], d["labels"], d["s32"]
        print(f"loaded cache {CACHE}")
    else:
        pooled, labels, s32 = build_cache(splits)

    methods = ["PCA100", "PLS5resid"]
    deltas = {m: {L: [] for L in LAYERS} for m in methods}
    base_aucs = []
    for fi, f in enumerate(FOLDS):
        tr = np.array(splits[f]["train"] + splits[f]["val"]); te = np.array(splits[f]["test"])
        ytr, yte = labels[tr], labels[te]
        if len(np.unique(yte)) < 2:
            continue
        scs = StandardScaler().fit(s32[fi][tr].reshape(-1, 1))
        s_tr, s_te = scs.transform(s32[fi][tr].reshape(-1, 1)), scs.transform(s32[fi][te].reshape(-1, 1))
        base = LogisticRegression(max_iter=3000).fit(s_tr, ytr)
        base_auc = roc_auc_score(yte, base.predict_proba(s_te)[:, 1]); base_aucs.append(base_auc)
        resid = ytr - base.predict_proba(s_tr)[:, 1]   # what layer-32 misses on seen tasks
        for li, L in enumerate(LAYERS):
            sc = StandardScaler().fit(pooled[tr, li, :])
            Ftr, Fte = sc.transform(pooled[tr, li, :]), sc.transform(pooled[te, li, :])
            # (A) PCA-100 unsupervised
            pca = PCA(100).fit(Ftr)
            for m, (Ptr, Pte) in {
                "PCA100": (pca.transform(Ftr), pca.transform(Fte)),
                "PLS5resid": _pls(Ftr, resid, Fte),
            }.items():
                Atr, Ate = np.hstack([s_tr, Ptr]), np.hstack([s_te, Pte])
                aug = LogisticRegression(max_iter=3000).fit(Atr, ytr)
                deltas[m][L].append(roc_auc_score(yte, aug.predict_proba(Ate)[:, 1]) - base_auc)

    print(f"\nbase (layer-32 score alone) held-out AUC = {np.mean(base_aucs):.3f} ± {np.std(base_aucs):.3f}\n")
    for m in methods:
        print(f"=== {m}: mean ΔAUC over folds (held-out task), per layer ===")
        print(f" {'layer':>6} {'ΔAUC':>8} {'std':>7} {'folds>0':>8}")
        for L in LAYERS:
            dd = np.array(deltas[m][L])
            print(f" {L:>6} {dd.mean():>+8.3f} {dd.std():>7.3f} {int((dd>0).sum()):>5}/{len(dd)}")
        print()


def _pls(Ftr, target, Fte):
    pls = PLSRegression(n_components=5).fit(Ftr, target)
    return pls.transform(Ftr), pls.transform(Fte)


if __name__ == "__main__":
    main()
