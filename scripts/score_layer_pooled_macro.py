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
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataloaders import OpenVLARolloutDataset, collate_rollouts
from models import LinearProbeModel, SafeLSTMModel, SafeMLPModel


DEFAULT_REG_TAG_BY_MODEL = {
    "mlp": "0.01",
    "lstm": "1",
    "linear_probe": "1",
}

ROLLOUT_RE = re.compile(
    r"task(?P<task_id>\d+)--ep(?P<episode_idx>\d+)--succ(?P<success>[01])"
)


class MixedSchemaLayerDataset(Dataset):
    """One-layer view over ordered roots with 3-D or pre-pooled 2-D artifacts.

    The 1k OpenVLA audit data are split across two historical schemas: the
    original shard preserves the action-token axis, while the second shard has
    already selected the last action token. This adapter reads both without
    rewriting the rollout files and computes task_min_step over the union.
    """

    def __init__(self, roots: list[Path], layer: int, token_pool: str) -> None:
        self.paths = [
            path
            for root in roots
            for path in sorted(root.expanduser().rglob("*.pkl"))
        ]
        if not self.paths:
            raise FileNotFoundError(f"No rollout files under {roots}")
        self.layer = int(layer)
        self.token_pool = token_pool
        self.selected_layers = (self.layer,)
        first_artifact = self._load(self.paths[0])
        self.hidden_dim = int(first_artifact["hidden_state_dim_per_layer"])
        self.task_min_steps: dict[int, int] = {}
        for path in self.paths:
            artifact = self._load(path)
            task_id, _, _ = self._metadata(path, artifact)
            length = int(artifact["hidden_states"].shape[0])
            current = self.task_min_steps.get(task_id)
            if current is None or length < current:
                self.task_min_steps[task_id] = length

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _load(path: Path) -> dict:
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, dict) or "hidden_states" not in artifact:
            raise ValueError(f"Invalid rollout artifact: {path}")
        return artifact

    @staticmethod
    def _metadata(path: Path, artifact: dict) -> tuple[int, int, bool]:
        match = ROLLOUT_RE.search(path.name)
        if match is None:
            raise ValueError(f"Cannot parse task/episode/success from {path}")
        task_id = int(artifact.get("task_id", match.group("task_id")))
        episode_idx = int(
            artifact.get(
                "episode_idx",
                artifact.get("eposide_idx", match.group("episode_idx")),
            )
        )
        success = bool(artifact.get("episode_success", int(match.group("success"))))
        return task_id, episode_idx, success

    def _features(self, path: Path, artifact: dict) -> torch.Tensor:
        hidden = artifact["hidden_states"].detach().to(dtype=torch.float32, device="cpu")
        saved_layers = [int(value) for value in artifact["hidden_state_layers"]]
        hidden_dim = int(artifact["hidden_state_dim_per_layer"])
        if hidden_dim != self.hidden_dim or self.layer not in saved_layers:
            raise ValueError(
                f"{path}: layer={self.layer}, hidden_dim={hidden_dim}, "
                f"saved_layers={saved_layers}"
            )
        position = saved_layers.index(self.layer)
        if hidden.ndim == 3:
            steps, action_tokens, width = hidden.shape
            expected = len(saved_layers) * hidden_dim
            if width != expected:
                raise ValueError(f"{path}: width={width}, expected={expected}")
            hidden = hidden.reshape(steps, action_tokens, len(saved_layers), hidden_dim)
            hidden = hidden[:, :, position, :]
            if self.token_pool == "last":
                return hidden[:, -1, :]
            if self.token_pool == "first":
                return hidden[:, 0, :]
            if self.token_pool == "mean":
                return hidden.mean(dim=1)
            raise ValueError(f"Unsupported token pool for mixed schema: {self.token_pool}")
        if hidden.ndim == 2:
            steps, width = hidden.shape
            expected = len(saved_layers) * hidden_dim
            if width != expected:
                raise ValueError(f"{path}: width={width}, expected={expected}")
            token_index = int(artifact.get("hidden_state_token_index", -999))
            compatible = (self.token_pool == "last" and token_index == -1) or (
                self.token_pool == "first" and token_index == 0
            )
            if not compatible:
                raise ValueError(
                    f"{path}: pre-pooled token index {token_index} is incompatible "
                    f"with token_pool={self.token_pool!r}"
                )
            hidden = hidden.reshape(steps, len(saved_layers), hidden_dim)
            return hidden[:, position, :]
        raise ValueError(f"{path}: expected 2-D or 3-D hidden states, got {hidden.shape}")

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        artifact = self._load(path)
        task_id, episode_idx, success = self._metadata(path, artifact)
        features = self._features(path, artifact)
        length = int(features.shape[0])
        return {
            "features": features,
            "label": torch.tensor(float(not success), dtype=torch.float32),
            "success": torch.tensor(float(success), dtype=torch.float32),
            "length": torch.tensor(length, dtype=torch.long),
            "task_min_step": torch.tensor(
                min(self.task_min_steps[task_id], length), dtype=torch.long
            ),
            "path": str(path),
            "task_id": task_id,
            "episode_idx": episode_idx,
            "selected_layers": self.selected_layers,
            "saved_layers": self.selected_layers,
        }


def default_reg_tag(model_type: str) -> str:
    return DEFAULT_REG_TAG_BY_MODEL[model_type]


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
    p.add_argument(
        "--root",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One rollout root, or multiple roots in the exact shard ordering used "
            "to create the checkpoint split indices. Multiple roots may mix 3-D "
            "token-preserving and 2-D pre-pooled artifact schemas."
        ),
    )
    p.add_argument("--layer", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--reg-tag",
        default=None,
        help=(
            "lambda_reg tag in the checkpoint filename. Default is model-aware: "
            "MLP=0.01, LSTM/linear=1."
        ),
    )
    p.add_argument("--all-layers", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20, 24, 28, 32])
    p.add_argument("--token-pool", default="last")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional long-form CSV of test-only per-rollout scores. The output "
            "is suitable for analyze_pooled_macro_decomposition.py."
        ),
    )
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    prefix = {"lstm": "safe_lstm", "mlp": "safe_mlp", "linear_probe": "safe_linear_probe"}[args.model_type]
    reg_tag = args.reg_tag if args.reg_tag is not None else default_reg_tag(args.model_type)

    models, splits, stored = {}, {}, {}
    for s in args.seeds:
        name = f"{prefix}_single_layers_{args.layer}_tok-{args.token_pool}_lr-0.0001_reg-{reg_tag}_seed-{s}"
        c = torch.load(args.checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        m = build_probe(args.model_type)
        m.load_state_dict(c["model_state_dict"])
        models[s] = m.to(device).eval()
        splits[s] = c["split"]
        stored[s] = c["result"]["test_falert_early_roc_auc"]

    if len(args.root) == 1:
        ds = OpenVLARolloutDataset(
            args.root[0], layers=tuple(args.all_layers),
            token_pool=args.token_pool, recursive=True, cache=False
        )
    else:
        ds = MixedSchemaLayerDataset(args.root, args.layer, args.token_pool)
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
    test_score_rows = []
    for s in args.seeds:
        test = df[df["idx"].isin(splits[s]["test"])]
        col = f"early_s{s}"
        pooled = roc_auc_score(test["label"], test[col])
        per_task = [roc_auc_score(g["label"], g[col]) for _, g in test.groupby("task_id")
                    if g["label"].nunique() > 1 and len(g) > 1]
        macro = float(np.mean(per_task))
        err = abs(pooled - stored[s])
        pooled_all.append(pooled); macro_all.append(macro)
        test_score_rows.extend(
            {
                "model": args.model_type,
                "split_id": s,
                "rollout_idx": int(row.idx),
                "task_id": int(row.task_id),
                "label": int(row.label),
                "score": float(getattr(row, col)),
                "layer": args.layer,
                "token_pool": args.token_pool,
                "checkpoint_dir": str(args.checkpoint_dir),
            }
            for row in test.itertuples(index=False)
        )
        flag = "  <-- MISMATCH" if err > 0.01 else ""
        print(f"{s:>5} {pooled:>8.4f} {macro:>8.4f} {stored[s]:>8.4f} {err:>7.4f}{flag}")
    print(f"{'mean':>5} {np.mean(pooled_all):>8.4f} {np.mean(macro_all):>8.4f}")
    print(f"  pooled {np.mean(pooled_all):.3f}±{np.std(pooled_all):.3f}   "
          f"macro {np.mean(macro_all):.3f}±{np.std(macro_all):.3f}")
    if max(abs(p - stored[s]) for s, p in zip(args.seeds, pooled_all)) > 0.01:
        raise SystemExit("ABORT: pooled does not match stored; root/split mismatch")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_score_rows).to_csv(args.output, index=False)
        print(f"wrote {args.output} ({len(test_score_rows)} test-score rows)")


if __name__ == "__main__":
    main()
