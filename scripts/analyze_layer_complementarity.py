"""Gate for the multi-layer hypothesis: is there COMPLEMENTARY, cross-task
failure signal across layers that the best single (deep) layer misses?

Two diagnostics, all on features pooled over [0, task_min_step] (the falert_early
window, so all rollouts are alive -> no survival leakage):

  1) Linear CKA between the 9 layers' (rollout-pooled) features. We previously
     measured *score* redundancy (deep block 0.88-0.92); this measures
     *representation* redundancy. If features decorrelate faster than scores,
     there is headroom for feature-level fusion.

  2) Incremental-value test (the decisive one): for each of the 10 LOTO folds
     (layer-32 MLP trained on 9 tasks, 1 task held out), does adding layer-k
     features improve AUC on the HELD-OUT task beyond the layer-32 probe's own
     score? base = logistic(s32); aug = logistic(s32 + PCA30(feat_k)). Report
     mean ΔAUC over folds, per layer k. Shallow layers (1/8) with ΔAUC>0 while
     the deep control (24) ~0 => complementary, generalizing signal => build the
     combiner. All ~0 => multi-layer is dead.

Uses the LOTO MLP layer-32 checkpoints (runs/openvla_mlp_loto_l32, reg-0.01).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
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
LAYERS = [1, 4, 8, 12, 16, 20, 24, 28, 32]
FOLDS = list(range(10))  # held-out task id == seed label
PCA_DIM = 30


def build_mlp() -> torch.nn.Module:
    return SafeMLPModel(input_dim=4096, hidden_dim=256, n_layers=2, dropout=0.0,
                        cumsum=True, n_history_steps=1, loss_type="safe", use_threshold=False)


def linear_cka(Xa: np.ndarray, Xb: np.ndarray) -> float:
    Xa = Xa - Xa.mean(0, keepdims=True)
    Xb = Xb - Xb.mean(0, keepdims=True)
    Ka, Kb = Xa @ Xa.T, Xb @ Xb.T
    return float((Ka * Kb).sum() / (np.linalg.norm(Ka) * np.linalg.norm(Kb)))


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # load the 10 LOTO layer-32 probes + their embedded splits
    probes, splits = {}, {}
    for f in FOLDS:
        name = f"safe_mlp_single_layers_32_tok-last_lr-0.0001_reg-0.01_seed-{f}"
        c = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
        m = build_mlp(); m.load_state_dict(c["model_state_dict"]); probes[f] = m.to(device).eval()
        splits[f] = c["split"]

    ds = OpenVLARolloutDataset(ROOT, layers=tuple(LAYERS), token_pool="last", recursive=True, cache=False)
    pos = {l: i for i, l in enumerate(ds.selected_layers)}
    dim = ds.hidden_dim
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_rollouts)

    n = len(ds)
    pooled = np.zeros((n, len(LAYERS), dim), np.float32)   # mean over [0,cap] per layer
    labels = np.zeros(n, np.float32)
    s32 = {f: np.zeros(n, np.float32) for f in FOLDS}       # layer-32 falert_early score per fold
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
                pooled[idx:idx+nb, li, :] = ((lf * tmask.unsqueeze(-1)).sum(1) / denom).cpu().numpy()
            l32 = feats[:, :, pos[32]*dim:(pos[32]+1)*dim]
            for f in FOLDS:
                raw = probes[f]({"features": l32}).squeeze(-1)
                early = raw.masked_fill(~tmask, float("-inf")).max(1).values
                s32[f][idx:idx+nb] = early.cpu().numpy()
            labels[idx:idx+nb] = b["labels"].numpy()
            idx += nb
            print(f"scored {idx}/{n}", flush=True)

    # ---- (1) linear CKA matrix ----
    print("\n=== (1) Linear CKA between rollout-pooled layer features ===")
    cka = np.eye(len(LAYERS))
    for i in range(len(LAYERS)):
        for j in range(i+1, len(LAYERS)):
            cka[i, j] = cka[j, i] = linear_cka(pooled[:, i, :], pooled[:, j, :])
    print("      " + " ".join(f"L{L:>2}" for L in LAYERS))
    for i, L in enumerate(LAYERS):
        print(f"L{L:>2} " + " ".join(f"{cka[i,j]:.2f}" for j in range(len(LAYERS))))
    print(f"\n shallow<->deep CKA (L8<->L32) = {cka[LAYERS.index(8), LAYERS.index(32)]:.3f}"
          f"   (L1<->L32) = {cka[LAYERS.index(1), LAYERS.index(32)]:.3f}"
          f"   deep block (L24<->L32) = {cka[LAYERS.index(24), LAYERS.index(32)]:.3f}")

    # ---- (2) incremental-value test over LOTO folds ----
    print("\n=== (2) Incremental held-out AUC from adding layer-k beyond layer-32 score ===")
    deltas = {L: [] for L in LAYERS}
    base_aucs = []
    for f in FOLDS:
        tr = np.array(splits[f]["train"] + splits[f]["val"])
        te = np.array(splits[f]["test"])
        ytr, yte = labels[tr], labels[te]
        if len(np.unique(yte)) < 2:
            continue
        # base: logistic on s32 alone (same pipeline as augmented for fairness)
        sc_s = StandardScaler().fit(s32[f][tr].reshape(-1, 1))
        base = LogisticRegression(max_iter=2000).fit(sc_s.transform(s32[f][tr].reshape(-1,1)), ytr)
        base_auc = roc_auc_score(yte, base.predict_proba(sc_s.transform(s32[f][te].reshape(-1,1)))[:,1])
        base_aucs.append(base_auc)
        for li, L in enumerate(LAYERS):
            scf = StandardScaler().fit(pooled[tr, li, :])
            pca = PCA(PCA_DIM).fit(scf.transform(pooled[tr, li, :]))
            Ptr = pca.transform(scf.transform(pooled[tr, li, :]))
            Pte = pca.transform(scf.transform(pooled[te, li, :]))
            Atr = np.hstack([sc_s.transform(s32[f][tr].reshape(-1,1)), Ptr])
            Ate = np.hstack([sc_s.transform(s32[f][te].reshape(-1,1)), Pte])
            aug = LogisticRegression(max_iter=2000).fit(Atr, ytr)
            aug_auc = roc_auc_score(yte, aug.predict_proba(Ate)[:,1])
            deltas[L].append(aug_auc - base_auc)

    print(f"\n base (layer-32 score alone) held-out AUC = {np.mean(base_aucs):.3f} ± {np.std(base_aucs):.3f}")
    print(f"\n {'layer':>6} {'mean ΔAUC':>10} {'std':>7} {'#folds>0':>9}")
    for L in LAYERS:
        d = np.array(deltas[L])
        print(f" {L:>6} {d.mean():>+10.3f} {d.std():>7.3f} {int((d>0).sum()):>6}/{len(d)}")
    print("\n Interpretation: shallow layers (1/8) with clearly +ΔAUC, while the deep")
    print(" control (24/28) ~0, => complementary cross-task signal => build combiner.")


if __name__ == "__main__":
    main()
