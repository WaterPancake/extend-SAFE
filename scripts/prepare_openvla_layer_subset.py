"""Extract a compact, time-resolved subset of OpenVLA rollout layers.

Source rollouts may preserve every action token or may already be pooled to one
token. This utility normalizes both schemas while keeping selected layers, one
action token, and an optional temporal stride, so controlled experiments can
run without repeatedly streaming the full rollout collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Prefer repository-relative paths in manifests and derived metadata."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, default=(20, 32))
    parser.add_argument("--token-index", type=int, default=-1)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument(
        "--skip-corrupt",
        action="store_true",
        help="Record and skip unreadable source artifacts instead of failing.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--expected-rollouts", type=int)
    parser.add_argument("--expected-tasks", type=parse_layers)
    parser.add_argument("--expected-rollouts-per-task", type=int)
    parser.add_argument("--minimum-length", type=int)
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        help=(
            "Optional earlier preparation manifest to retain when extending an "
            "existing processed collection with another source batch."
        ),
    )
    return parser.parse_args()


def load_artifact(path: Path) -> dict:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "hidden_states" not in artifact:
        raise ValueError(f"{path} is not a rollout artifact")
    return artifact


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_hidden_subset(
    artifact: dict,
    layers: tuple[int, ...],
    token_index: int,
    temporal_stride: int,
) -> tuple[torch.Tensor, int]:
    """Normalize token-preserving or pre-pooled sources to one token axis."""

    hidden = artifact["hidden_states"]
    saved_layers = list(
        artifact.get("hidden_state_layers", artifact.get("hidden_states_layers", []))
    )
    hidden_dim = int(artifact.get("hidden_state_dim_per_layer", 0))
    if not saved_layers or hidden_dim <= 0:
        raise ValueError("artifact lacks layer layout metadata")
    missing = [layer for layer in layers if layer not in saved_layers]
    if missing:
        raise ValueError(f"artifact lacks requested layers {missing}")
    expected_width = len(saved_layers) * hidden_dim
    positions = [saved_layers.index(layer) for layer in layers]

    if hidden.ndim == 3:
        if hidden.shape[-1] != expected_width:
            raise ValueError(
                f"unexpected hidden shape {tuple(hidden.shape)}; expected final "
                f"width {expected_width}"
            )
        selected_token = token_index % int(hidden.shape[1])
        selected = hidden.reshape(
            hidden.shape[0], hidden.shape[1], len(saved_layers), hidden_dim
        )[:, selected_token : selected_token + 1, positions, :]
    elif hidden.ndim == 2:
        if hidden.shape[-1] != expected_width:
            raise ValueError(
                f"unexpected hidden shape {tuple(hidden.shape)}; expected final "
                f"width {expected_width}"
            )
        if token_index != -1:
            raise ValueError(
                "a pre-pooled 2-D source has no action-token axis; use --token-index -1"
            )
        selected_token = int(artifact.get("hidden_state_token_index", -1))
        selected = hidden.reshape(
            hidden.shape[0], 1, len(saved_layers), hidden_dim
        )[:, :, positions, :]
    else:
        raise ValueError(
            f"unexpected hidden shape {tuple(hidden.shape)}; expected 2-D or 3-D"
        )

    compact = selected[::temporal_stride].reshape(
        -1, 1, len(layers) * hidden_dim
    ).contiguous()
    return compact, selected_token


def audit_output(output_root: Path) -> dict[str, Any]:
    """Summarize the complete processed collection, not only this input batch."""

    paths = sorted(output_root.rglob("*.pkl"))
    task_counts: Counter[int] = Counter()
    outcome_counts: Counter[tuple[int, bool]] = Counter()
    episodes: dict[int, set[int]] = defaultdict(set)
    schemas: Counter[tuple[Any, ...]] = Counter()
    source_token_indices: Counter[int] = Counter()
    lengths: list[int] = []
    inventory = hashlib.sha256()

    for path in paths:
        artifact = load_artifact(path)
        hidden = artifact["hidden_states"]
        task_id = int(artifact["task_id"])
        episode_idx = int(
            artifact.get("episode_idx", artifact.get("eposide_idx"))
        )
        success = bool(artifact["episode_success"])
        schema = (
            tuple(int(layer) for layer in artifact["hidden_state_layers"]),
            int(artifact["hidden_state_dim_per_layer"]),
            int(hidden.ndim),
            tuple(int(value) for value in hidden.shape[1:]),
            artifact.get("temporal_stride"),
            artifact.get("hidden_state_layout"),
        )
        task_counts[task_id] += 1
        outcome_counts[(task_id, success)] += 1
        episodes[task_id].add(episode_idx)
        schemas[schema] += 1
        source_token_indices[int(artifact.get("source_action_token_index", -1))] += 1
        lengths.append(int(hidden.shape[0]))
        relative = path.relative_to(output_root).as_posix()
        inventory.update(relative.encode())
        inventory.update(str(path.stat().st_size).encode())

    schema_rows = []
    for schema, count in sorted(schemas.items(), key=lambda item: repr(item[0])):
        (
            layers,
            hidden_dim,
            hidden_ndim,
            hidden_shape_suffix,
            temporal_stride,
            layout,
        ) = schema
        schema_rows.append(
            {
                "count": count,
                "layers": list(layers),
                "hidden_dim_per_layer": hidden_dim,
                "hidden_ndim": hidden_ndim,
                "hidden_shape_suffix": list(hidden_shape_suffix),
                "temporal_stride": temporal_stride,
                "hidden_state_layout": layout,
            }
        )
    return {
        "rollouts": len(paths),
        "task_counts": {
            str(task): count for task, count in sorted(task_counts.items())
        },
        "episode_coverage": {
            str(task): {
                "minimum": min(task_episodes),
                "maximum": max(task_episodes),
                "unique_count": len(task_episodes),
            }
            for task, task_episodes in sorted(episodes.items())
        },
        "outcome_counts": {
            f"task{task}_success{int(success)}": count
            for (task, success), count in sorted(outcome_counts.items())
        },
        "minimum_length": min(lengths) if lengths else None,
        "maximum_length": max(lengths) if lengths else None,
        "homogeneous_schema": len(schemas) == 1,
        "source_action_token_index_counts": {
            str(index): count for index, count in sorted(source_token_indices.items())
        },
        "schemas": schema_rows,
        "inventory_sha256": inventory.hexdigest(),
    }


def validate_output_audit(audit: dict[str, Any], args: argparse.Namespace) -> None:
    if not audit["homogeneous_schema"]:
        raise ValueError("processed output does not have one homogeneous tensor schema")
    if args.expected_rollouts is not None and audit["rollouts"] != args.expected_rollouts:
        raise ValueError(
            f"expected {args.expected_rollouts} output rollouts, found {audit['rollouts']}"
        )
    task_counts = {int(task): int(count) for task, count in audit["task_counts"].items()}
    if args.expected_tasks is not None and set(task_counts) != set(args.expected_tasks):
        raise ValueError(
            f"expected tasks {list(args.expected_tasks)}, found {sorted(task_counts)}"
        )
    if args.expected_rollouts_per_task is not None:
        mismatched = {
            task: count
            for task, count in task_counts.items()
            if count != args.expected_rollouts_per_task
        }
        if mismatched:
            raise ValueError(
                f"expected {args.expected_rollouts_per_task} rollouts per task; "
                f"mismatches={mismatched}"
            )
    if args.minimum_length is not None and (
        audit["minimum_length"] is None
        or int(audit["minimum_length"]) < args.minimum_length
    ):
        raise ValueError(
            f"expected minimum length >= {args.minimum_length}, found "
            f"{audit['minimum_length']}"
        )


def main() -> None:
    args = parse_args()
    if args.temporal_stride < 1:
        raise ValueError("--temporal-stride must be >= 1")

    paths = sorted(args.input_root.expanduser().resolve().rglob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no .pkl rollouts under {args.input_root}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inventory = hashlib.sha256()
    written = 0
    skipped = 0
    corrupt: list[dict[str, str]] = []
    for index, source_path in enumerate(paths, start=1):
        relative_path = source_path.relative_to(args.input_root.expanduser().resolve())
        output_path = output_root / relative_path
        inventory.update(str(relative_path).encode())
        inventory.update(str(source_path.stat().st_size).encode())
        if output_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            artifact = load_artifact(source_path)
        except Exception as exc:
            if not args.skip_corrupt:
                raise RuntimeError(f"failed to read {source_path}") from exc
            corrupt.append({"path": portable_path(source_path), "error": repr(exc)})
            print(f"skipping corrupt artifact {source_path}: {exc!r}", flush=True)
            continue
        hidden = artifact["hidden_states"]
        hidden_dim = int(artifact.get("hidden_state_dim_per_layer", 0))
        try:
            selected, token_index = select_hidden_subset(
                artifact,
                args.layers,
                args.token_index,
                args.temporal_stride,
            )
        except ValueError as error:
            raise ValueError(f"{source_path}: {error}") from error

        compact = dict(artifact)
        compact["hidden_states"] = selected
        compact["hidden_state_layers"] = list(args.layers)
        compact["hidden_state_layer_stride"] = None
        compact["hidden_state_dim_per_layer"] = hidden_dim
        compact["hidden_state_token_index"] = 0
        compact["hidden_state_layout"] = "action_token_axis_then_concatenated_selected_layers"
        compact["source_rollout_path"] = portable_path(source_path)
        compact["source_hidden_state_ndim"] = int(hidden.ndim)
        compact["source_action_token_index"] = int(token_index)
        compact["temporal_stride"] = int(args.temporal_stride)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(compact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(output_path)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"processed {index}/{len(paths)}", flush=True)

    output_audit = audit_output(output_root)
    validate_output_audit(output_audit, args)
    manifest = {
        "input_root": portable_path(args.input_root),
        "output_root": portable_path(output_root),
        "source_rollouts": len(paths),
        "written": written,
        "skipped_existing": skipped,
        "corrupt_source_artifacts": corrupt,
        "layers": list(args.layers),
        "token_index": args.token_index,
        "temporal_stride": args.temporal_stride,
        "source_inventory_sha256": inventory.hexdigest(),
        "output_audit": output_audit,
    }
    if args.prior_manifest is not None:
        prior_path = args.prior_manifest.expanduser().resolve()
        manifest["prior_preparation_manifest"] = {
            "file": prior_path.name,
            "sha256": sha256_file(prior_path),
            "manifest": json.loads(prior_path.read_text()),
        }
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
