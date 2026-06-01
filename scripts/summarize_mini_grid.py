#!/usr/bin/env python
"""Aggregate the openvla-mini grid-search checkpoints into a ranked summary.

Each grid cell (scripts/train_mini_grid.sh) trains one (model, token, lr,
lambda, seed) combination and saves a checkpoint under
    runs/mini_grid/{lstm,mlp}/<exp_name>.pt
Per-cell `ablation_results.csv` files get overwritten, so this script reads the
checkpoints directly, aggregates across seeds, and reports the best
hyperparameters per model.

Selection is done on the *validation* leakage-resistant metric
(`val_falert_early_roc_auc`, mean over seeds) to avoid peeking at the test set;
the corresponding unseen-test numbers are reported alongside. Works on a partial
grid, so it doubles as a live progress check while the sweep is still running.

Usage:
    .venv/bin/python scripts/summarize_mini_grid.py
    .venv/bin/python scripts/summarize_mini_grid.py --root runs/mini_grid --top 10
    .venv/bin/python scripts/summarize_mini_grid.py --csv runs/mini_grid/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

TOTAL_CELLS = 216  # 2 models x 3 tokens x 4 lambdas x 3 lrs x 3 seeds
SAFE_PAPER = {"lstm": 0.7247, "mlp": 0.7347}  # unseen ROC-AUC reference

# metrics pulled from each checkpoint's result dict
VAL_SEL = "val_falert_early_roc_auc"          # selection metric (leakage-resistant)
TEST_EARLY = "test_falert_early_roc_auc"      # headline unseen metric
TEST_END = "test_falert_end_roc_auc"          # leaky full-rollout reference
TEST_TPR = "test_falert_early_tpr_at_5_fpr"   # SAFE secondary metric
TEST_LEAK = "test_length_only_roc_auc"        # leak presence check


def load_cells(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pt in sorted(root.rglob("*.pt")):
        try:
            ck = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip unreadable {pt}: {exc}")
            continue
        exp = ck.get("experiment", {})
        r = ck.get("result", {})
        rows.append(
            {
                "model": exp.get("model_type", r.get("model_type")),
                "token": exp.get("token_pool", r.get("token_pool")),
                "lr": float(exp.get("lr", r.get("lr"))),
                "lambda": float(exp.get("lambda_reg", r.get("lambda_reg"))),
                "seed": int(exp.get("seed", r.get("seed"))),
                "val_sel": r.get(VAL_SEL, float("nan")),
                "test_early": r.get(TEST_EARLY, float("nan")),
                "test_end": r.get(TEST_END, float("nan")),
                "test_tpr5": r.get(TEST_TPR, float("nan")),
                "test_leak": r.get(TEST_LEAK, float("nan")),
                "best_epoch": r.get("best_epoch", float("nan")),
                "path": str(pt),
            }
        )
    return rows


def _agg(values: list[float]) -> tuple[float, float, int]:
    """mean, population-std, count over the non-NaN values."""
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return mean(clean), (pstdev(clean) if len(clean) > 1 else 0.0), len(clean)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["token"], row["lr"], row["lambda"])].append(row)

    out: list[dict[str, Any]] = []
    for (model, token, lr, lam), cells in groups.items():
        val_m, val_s, _ = _agg([c["val_sel"] for c in cells])
        te_m, te_s, n = _agg([c["test_early"] for c in cells])
        end_m, _, _ = _agg([c["test_end"] for c in cells])
        tpr_m, _, _ = _agg([c["test_tpr5"] for c in cells])
        leak_m, _, _ = _agg([c["test_leak"] for c in cells])
        ep_m, _, _ = _agg([c["best_epoch"] for c in cells])
        out.append(
            {
                "model": model,
                "token": token,
                "lr": lr,
                "lambda": lam,
                "n_seeds": n,
                "val_early_mean": val_m,
                "val_early_std": val_s,
                "test_early_mean": te_m,
                "test_early_std": te_s,
                "test_end_mean": end_m,
                "test_tpr5_mean": tpr_m,
                "test_leak_mean": leak_m,
                "best_epoch_mean": ep_m,
            }
        )
    return out


def fmt(x: float, nd: int = 4) -> str:
    return "  nan " if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"


def print_model_table(model: str, configs: list[dict[str, Any]], top: int) -> None:
    rows = [c for c in configs if c["model"] == model]
    if not rows:
        return
    # rank by validation selection metric (mean over seeds), nan last
    rows.sort(key=lambda c: (-(c["val_early_mean"]) if not math.isnan(c["val_early_mean"]) else 1e9))
    baseline = SAFE_PAPER.get(model, float("nan"))
    print(f"\n================ {model.upper()}  (ranked by val falert_early; SAFE unseen ref={baseline}) ================")
    print(f"{'token':>6} {'lr':>7} {'lambda':>7} {'n':>2} | {'val_early':>16} | {'test_early':>16} {'Δpaper':>8} | {'test_end':>9} {'tpr@5':>7} {'leak':>6} {'ep':>4}")
    print("-" * 118)
    for c in rows[:top]:
        delta = c["test_early_mean"] - baseline if not math.isnan(c["test_early_mean"]) else float("nan")
        val = f"{fmt(c['val_early_mean'])}±{fmt(c['val_early_std'],3).strip()}"
        tst = f"{fmt(c['test_early_mean'])}±{fmt(c['test_early_std'],3).strip()}"
        print(
            f"{c['token']:>6} {c['lr']:>7g} {c['lambda']:>7g} {c['n_seeds']:>2} | "
            f"{val:>16} | {tst:>16} {fmt(delta,4):>8} | "
            f"{fmt(c['test_end_mean']):>9} {fmt(c['test_tpr5_mean'],3):>7} {fmt(c['test_leak_mean'],2):>6} {fmt(c['best_epoch_mean'],0):>4}"
        )
    best = rows[0]
    print(
        f"\n  >>> best {model}: token={best['token']} lr={best['lr']:g} lambda={best['lambda']:g} "
        f"| val_early={fmt(best['val_early_mean'])} test_early={fmt(best['test_early_mean'])} "
        f"(Δpaper={fmt(best['test_early_mean']-baseline)}, n={best['n_seeds']} seeds)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/mini_grid"))
    ap.add_argument("--top", type=int, default=12, help="rows per model table")
    ap.add_argument("--csv", type=Path, default=Path("runs/mini_grid/summary.csv"))
    ap.add_argument("--per-seed-csv", type=Path, default=Path("runs/mini_grid/summary_per_seed.csv"))
    args = ap.parse_args()

    rows = load_cells(args.root)
    print(f"found {len(rows)}/{TOTAL_CELLS} grid cells under {args.root}")
    if not rows:
        return
    by_model = defaultdict(int)
    for r in rows:
        by_model[r["model"]] += 1
    print("  per-model cells:", dict(by_model))

    configs = aggregate(rows)
    for model in sorted({r["model"] for r in rows if r["model"]}):
        print_model_table(model, configs, args.top)

    # write CSVs
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    agg_fields = [
        "model", "token", "lr", "lambda", "n_seeds",
        "val_early_mean", "val_early_std",
        "test_early_mean", "test_early_std",
        "test_end_mean", "test_tpr5_mean", "test_leak_mean", "best_epoch_mean",
    ]
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=agg_fields)
        w.writeheader()
        for c in sorted(configs, key=lambda c: (c["model"], -(_safe(c["val_early_mean"])))):
            w.writerow({k: c[k] for k in agg_fields})
    with args.per_seed_csv.open("w", newline="") as fh:
        seed_fields = ["model", "token", "lr", "lambda", "seed",
                       "val_sel", "test_early", "test_end", "test_tpr5", "test_leak", "best_epoch", "path"]
        w = csv.DictWriter(fh, fieldnames=seed_fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["token"], r["lr"], r["lambda"], r["seed"])):
            w.writerow({k: r[k] for k in seed_fields})
    print(f"\nwrote {args.csv}  and  {args.per_seed_csv}")


def _safe(x: float) -> float:
    return -1e9 if (x is None or (isinstance(x, float) and math.isnan(x))) else x


if __name__ == "__main__":
    main()
