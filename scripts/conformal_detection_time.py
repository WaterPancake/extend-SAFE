"""Per-task calibrated early-warning analysis on held-out tasks.

Task-holdout split => train/val are SEEN tasks, test are UNSEEN tasks. We:
  1. Calibrate a constant alarm threshold tau on seen-task VAL via split
     conformal so that the false-positive rate (fraction of successful val
     rollouts that ever cross tau) equals a target FPR.
  2. Evaluate on held-out TEST tasks, per task then macro-averaged:
       - realized FPR  (does the calibrated FPR transfer to unseen tasks?)
       - catch rate / TPR (fraction of failures that alarm before timeout)
       - lead time (steps between first alarm and the rollout's end; for
         failures the end is the 520-step timeout -> how early we warned).

An alarm fires at the first timestep where the running-max failure score
crosses tau. Scores are read over each rollout's full valid length.

Usage:
    uv run python scripts/conformal_detection_time.py --model-type mlp \
        --checkpoint-dir runs/openvla_mlp_layer_selection_1k \
        --root data/rollouts --layer 32 --fpr 0.01 0.05 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def conformal_threshold(success_max_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest tau s.t. fraction of success scores exceeding tau <= target_fpr,
    with the split-conformal finite-sample correction (rank (1-a)(n+1))."""
    n = len(success_max_scores)
    s = np.sort(success_max_scores)
    rank = int(np.ceil((1.0 - target_fpr) * (n + 1)))
    if rank > n:  # not enough calibration data to guarantee the FPR -> +inf (never alarm)
        return float("inf")
    return float(s[rank - 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=["lstm", "mlp", "linear_probe"], required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--layer", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--reg-tag", default="1",
                   help="lambda_reg tag in the checkpoint filename (e.g. '1' or '0.01').")
    p.add_argument("--fpr", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    p.add_argument("--all-layers", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20, 24, 28, 32])
    p.add_argument("--token-pool", default="last")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    prefix = {"lstm": "safe_lstm", "mlp": "safe_mlp", "linear_probe": "safe_linear_probe"}[args.model_type]

    models, splits = {}, {}
    for s in args.seeds:
        name = f"{prefix}_single_layers_{args.layer}_tok-{args.token_pool}_lr-0.0001_reg-{args.reg_tag}_seed-{s}"
        c = torch.load(args.checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        m = build_probe(args.model_type)
        m.load_state_dict(c["model_state_dict"])
        models[s] = m.to(device).eval()
        splits[s] = c["split"]

    ds = OpenVLARolloutDataset(args.root, layers=tuple(args.all_layers),
                               token_pool=args.token_pool, recursive=True, cache=False)
    pos = {l: i for i, l in enumerate(ds.selected_layers)}[args.layer]
    dim = ds.hidden_dim
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_rollouts)

    # one scoring pass: per rollout, per seed -> running-max trace summary
    # store: max_score (overall), and the running-max trace + length so we can
    # compute first-alarm step for any tau.
    recs = {s: {} for s in args.seeds}  # seed -> idx -> dict
    idx = 0
    with torch.no_grad():
        for b in loader:
            feats = b["features"].to(device)
            masks = b["valid_masks"].to(device)
            lengths = b["lengths"]
            n, T = masks.shape
            lf = feats[:, :, pos * dim:(pos + 1) * dim]
            for s in args.seeds:
                raw = models[s]({"features": lf}).squeeze(-1)
                raw = raw.masked_fill(~masks, float("-inf"))
                runmax = torch.cummax(raw, dim=1).values.cpu().numpy()
                for r in range(n):
                    L = int(lengths[r])
                    trace = runmax[r, :L]
                    recs[s][idx + r] = {
                        "task_id": b["task_ids"][r],
                        "label": int(b["labels"][r]),
                        "length": L,
                        "trace": trace,
                        "max_score": float(trace[-1]),
                    }
            idx += n
            print(f"scored {idx}/{len(ds)}", flush=True)

    out_rows = []
    print(f"\n=== {args.model_type} layer {args.layer}: per-task calibrated early-warning "
          f"(calibrate on seen-task val, evaluate on held-out test tasks) ===")
    for target in args.fpr:
        # per seed: calibrate tau on val successes, evaluate on test tasks
        macro = {"realized_fpr": [], "catch": [], "lead_steps": [], "lead_frac": [], "alarm_frac": []}
        per_seed_tau = []
        for s in args.seeds:
            val_idx, test_idx = set(splits[s]["val"]), set(splits[s]["test"])
            val_succ = [recs[s][i]["max_score"] for i in val_idx if recs[s][i]["label"] == 0]
            tau = conformal_threshold(np.array(val_succ), target)
            per_seed_tau.append(tau)
            # group test rollouts by task
            by_task = {}
            for i in test_idx:
                by_task.setdefault(recs[s][i]["task_id"], []).append(recs[s][i])
            task_fpr, task_catch, task_lead, task_leadfrac, task_alarmfrac = [], [], [], [], []
            for tid, rs in by_task.items():
                succ = [r for r in rs if r["label"] == 0]
                fail = [r for r in rs if r["label"] == 1]
                if succ:
                    task_fpr.append(np.mean([r["max_score"] >= tau for r in succ]))
                if fail:
                    catches, leads, leadfracs, alarmfracs = [], [], [], []
                    for r in fail:
                        crossed = np.where(r["trace"] >= tau)[0]
                        if len(crossed):
                            t0 = int(crossed[0])
                            catches.append(1)
                            leads.append(r["length"] - t0)            # steps before end
                            leadfracs.append((r["length"] - t0) / r["length"])
                            alarmfracs.append(t0 / r["length"])
                        else:
                            catches.append(0)
                    task_catch.append(np.mean(catches))
                    if leads:
                        task_lead.append(np.mean(leads))
                        task_leadfrac.append(np.mean(leadfracs))
                        task_alarmfrac.append(np.mean(alarmfracs))
            macro["realized_fpr"].append(np.mean(task_fpr) if task_fpr else np.nan)
            macro["catch"].append(np.mean(task_catch) if task_catch else np.nan)
            macro["lead_steps"].append(np.mean(task_lead) if task_lead else np.nan)
            macro["lead_frac"].append(np.mean(task_leadfrac) if task_leadfrac else np.nan)
            macro["alarm_frac"].append(np.mean(task_alarmfrac) if task_alarmfrac else np.nan)

        row = {
            "model": args.model_type, "layer": args.layer, "target_fpr": target,
            "realized_fpr": np.nanmean(macro["realized_fpr"]),
            "catch_rate": np.nanmean(macro["catch"]),
            "lead_steps": np.nanmean(macro["lead_steps"]),
            "lead_frac": np.nanmean(macro["lead_frac"]),
            "alarm_frac_of_rollout": np.nanmean(macro["alarm_frac"]),
            "catch_std": np.nanstd(macro["catch"]),
        }
        out_rows.append(row)
        print(f"\n target FPR = {target:.0%}   (tau per seed: "
              f"{', '.join(f'{t:.3f}' for t in per_seed_tau)})")
        print(f"   realized FPR (held-out, macro):  {row['realized_fpr']:.3f}")
        print(f"   catch rate / TPR (held-out):     {row['catch_rate']:.3f} +/- {row['catch_std']:.3f}")
        print(f"   mean lead time:                  {row['lead_steps']:.0f} steps "
              f"({row['lead_frac']:.2f} of rollout) before end")
        print(f"   first alarm at:                  {row['alarm_frac_of_rollout']:.2f} of rollout")

    df = pd.DataFrame(out_rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
