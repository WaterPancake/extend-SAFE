"""Portable, non-pickle storage for LOTO score trajectories.

The full OpenVLA hidden-state artifacts and model checkpoints are too large for
the publication repository.  A score bundle preserves exactly the validation
and held-out-test trajectories needed to replay functional conformal
evaluation, including fixed-horizon sensitivity analyses.

Bundle layout::

    score_bundle/
      manifest.json
      records.csv
      trajectories.npz

``trajectories.npz`` contains a single concatenated float32 score array and
integer offsets.  ``records.csv`` contains no machine-local absolute paths;
source files are identified as ``root-<n>/<relative-path>``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_conformal import RolloutScores


SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECORD_FIELDS = [
    "record_id",
    "model",
    "fold",
    "split_role",
    "dataset_index",
    "source_key",
    "task_id",
    "episode_idx",
    "label",
    "success",
    "length",
    "task_min_step",
    "n_scores",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _record_at(
    records: Sequence[RolloutScores] | Mapping[int, RolloutScores], index: int
) -> RolloutScores:
    record = records[index]
    if record is None:
        raise ValueError(f"No score trajectory for dataset index {index}")
    return record


def _source_key(path: str, roots: Sequence[Path]) -> str:
    candidate = Path(path).expanduser().resolve()
    for root_index, root in enumerate(roots):
        resolved_root = root.expanduser().resolve()
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError:
            continue
        return f"root-{root_index}/{relative.as_posix()}"
    raise ValueError(
        f"Rollout path {candidate.name!r} is not under a declared source root"
    )


def _portable_generator_key(path: Path, index: int) -> str:
    """Return a stable generator name without exposing a machine-local path."""

    candidate = path.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return candidate.relative_to(repo_root).as_posix()
    except ValueError:
        return f"external-{index}/{candidate.name}"


def _command_has_absolute_path(command: str | None) -> bool:
    if not command:
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return True
    for token in tokens:
        candidate = token.split("=", 1)[-1]
        if Path(candidate).is_absolute():
            return True
    return False


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute()
    if isinstance(value, dict):
        return any(
            _contains_absolute_path(key) or _contains_absolute_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_score_bundle(
    output_dir: Path,
    family_records: Mapping[
        str, Mapping[int, Sequence[RolloutScores] | Mapping[int, RolloutScores]]
    ],
    family_splits: Mapping[str, Mapping[int, Mapping[str, Sequence[int]]]],
    source_roots: Sequence[Path],
    checkpoint_paths: Mapping[str, Mapping[int, Path]],
    *,
    generator_paths: Sequence[Path] = (),
    command: str | None = None,
) -> dict[str, Any]:
    """Write a replayable score bundle and return its manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    offsets = [0]
    inventory_rows: set[tuple[str, int | None, int | None, int, int]] = set()

    for model in sorted(family_records):
        if model not in family_splits:
            raise ValueError(f"Missing splits for model family {model!r}")
        for fold in sorted(family_records[model]):
            records = family_records[model][fold]
            split = family_splits[model][fold]
            for role in ("val", "test"):
                if role not in split:
                    raise ValueError(f"{model} fold {fold} has no {role!r} split")
                for dataset_index in split[role]:
                    record = _record_at(records, int(dataset_index))
                    scores = np.asarray(record.scores, dtype=np.float32)
                    if scores.ndim != 1 or scores.size != int(record.length):
                        raise ValueError(
                            f"{model} fold {fold} index {dataset_index}: score "
                            f"shape {scores.shape} does not match length {record.length}"
                        )
                    source_key = _source_key(record.path, source_roots)
                    record_id = len(rows)
                    rows.append(
                        {
                            "record_id": record_id,
                            "model": model,
                            "fold": int(fold),
                            "split_role": role,
                            "dataset_index": int(dataset_index),
                            "source_key": source_key,
                            "task_id": "" if record.task_id is None else int(record.task_id),
                            "episode_idx": (
                                "" if record.episode_idx is None else int(record.episode_idx)
                            ),
                            "label": int(record.label),
                            "success": int(bool(record.success)),
                            "length": int(record.length),
                            "task_min_step": int(record.task_min_step),
                            "n_scores": int(scores.size),
                        }
                    )
                    arrays.append(scores)
                    offsets.append(offsets[-1] + scores.size)
                    inventory_rows.add(
                        (
                            source_key,
                            record.task_id,
                            record.episode_idx,
                            int(record.label),
                            int(record.length),
                        )
                    )

    if not rows:
        raise ValueError("Cannot write an empty score bundle")

    records_path = output_dir / "records.csv"
    trajectories_path = output_dir / "trajectories.npz"
    _write_records(records_path, rows)
    np.savez_compressed(
        trajectories_path,
        scores=np.concatenate(arrays).astype(np.float32, copy=False),
        offsets=np.asarray(offsets, dtype=np.int64),
    )

    inventory_text = "\n".join(
        "|".join("" if value is None else str(value) for value in row)
        for row in sorted(inventory_rows)
    )
    checkpoint_manifest = {
        model: {
            str(fold): {
                "file": path.name,
                "sha256": sha256_file(path),
            }
            for fold, path in sorted(paths.items())
        }
        for model, paths in sorted(checkpoint_paths.items())
    }
    generator_manifest = {
        _portable_generator_key(path, index): sha256_file(path)
        for index, path in enumerate(sorted(generator_paths))
        if path.exists()
    }
    source_preparation_manifests = {}
    for root_index, root in enumerate(source_roots):
        preparation_path = root / "manifest.json"
        if not preparation_path.exists():
            continue
        preparation = json.loads(preparation_path.read_text())
        if _contains_absolute_path(preparation):
            raise ValueError(
                f"Source preparation manifest contains an absolute path: "
                f"{preparation_path}"
            )
        canonical = json.dumps(
            preparation, sort_keys=True, separators=(",", ":")
        ).encode()
        source_preparation_manifests[f"root-{root_index}"] = {
            "file": preparation_path.name,
            "file_sha256": sha256_file(preparation_path),
            "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            "manifest": preparation,
        }
    if _command_has_absolute_path(command):
        raise ValueError(
            "Score-bundle generation command must use relative paths or aliases"
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Replay LOTO ranking and functional conformal evaluation",
        "path_policy": "root aliases and relative paths only; no absolute source paths",
        "score_dtype": "float32",
        "record_count": len(rows),
        "score_value_count": int(offsets[-1]),
        "unique_rollout_count": len(inventory_rows),
        "models": sorted(family_records),
        "folds_by_model": {
            model: sorted(int(fold) for fold in folds)
            for model, folds in family_records.items()
        },
        "included_split_roles": ["val", "test"],
        "source_root_aliases": {
            f"root-{index}": root.name for index, root in enumerate(source_roots)
        },
        "source_preparation_manifests": source_preparation_manifests,
        "source_inventory_sha256": hashlib.sha256(
            inventory_text.encode("utf-8")
        ).hexdigest(),
        "checkpoint_manifest": checkpoint_manifest,
        "generator_sha256": generator_manifest,
        "command": command,
        "files": {
            "records.csv": sha256_file(records_path),
            "trajectories.npz": sha256_file(trajectories_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_score_bundle(output_dir)
    return manifest


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def verify_score_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate hashes, schema, offsets, metadata, and path sanitization."""

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing score-bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    required_manifest_fields = {
        "schema_version",
        "record_count",
        "score_value_count",
        "unique_rollout_count",
        "models",
        "folds_by_model",
        "included_split_roles",
        "source_root_aliases",
        "source_preparation_manifests",
        "source_inventory_sha256",
        "checkpoint_manifest",
        "generator_sha256",
        "files",
    }
    missing_manifest_fields = sorted(required_manifest_fields - set(manifest))
    if missing_manifest_fields:
        raise ValueError(
            f"Score-bundle manifest is missing fields: {missing_manifest_fields}"
        )
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported score-bundle schema {manifest.get('schema_version')!r}"
        )
    required_files = {"records.csv", "trajectories.npz"}
    if set(manifest["files"]) != required_files:
        raise ValueError(
            f"Score-bundle files must be {sorted(required_files)}, found "
            f"{sorted(manifest['files'])}"
        )
    for name, expected in manifest["files"].items():
        if not SHA256_RE.fullmatch(str(expected)):
            raise ValueError(f"Invalid SHA-256 digest for {name}")
        actual = sha256_file(bundle_dir / name)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: {actual} != {expected}")

    aliases = manifest["source_root_aliases"]
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError("source_root_aliases must be a non-empty mapping")
    for alias, root_name in aliases.items():
        if not re.fullmatch(r"root-\d+", str(alias)):
            raise ValueError(f"Invalid source-root alias {alias!r}")
        if (
            not root_name
            or Path(str(root_name)).is_absolute()
            or len(Path(str(root_name)).parts) != 1
        ):
            raise ValueError(f"Unsafe source-root name {root_name!r}")
    if _command_has_absolute_path(manifest.get("command")):
        raise ValueError("Generation command contains an absolute path")

    preparation_manifests = manifest["source_preparation_manifests"]
    if not isinstance(preparation_manifests, dict):
        raise ValueError("source_preparation_manifests must be a mapping")
    if not set(preparation_manifests) <= set(aliases):
        raise ValueError("Source preparation manifest aliases are not declared roots")
    for alias, detail in preparation_manifests.items():
        if detail.get("file") != "manifest.json":
            raise ValueError(f"Unsafe preparation-manifest file for {alias}")
        for field in ("file_sha256", "canonical_sha256"):
            if not SHA256_RE.fullmatch(str(detail.get(field, ""))):
                raise ValueError(f"Invalid {field} for {alias}")
        preparation = detail.get("manifest")
        if not isinstance(preparation, dict) or _contains_absolute_path(preparation):
            raise ValueError(f"Unsafe embedded preparation manifest for {alias}")
        canonical = json.dumps(
            preparation, sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != detail["canonical_sha256"]:
            raise ValueError(f"Preparation manifest digest mismatch for {alias}")

    checkpoint_manifest = manifest["checkpoint_manifest"]
    generator_manifest = manifest["generator_sha256"]
    for generator, digest in generator_manifest.items():
        if Path(generator).is_absolute() or ".." in Path(generator).parts:
            raise ValueError(f"Unsafe generator path {generator!r}")
        if not SHA256_RE.fullmatch(str(digest)):
            raise ValueError(f"Invalid generator SHA-256 for {generator!r}")

    with (bundle_dir / "records.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RECORD_FIELDS:
            raise ValueError(
                f"Unexpected records.csv fields: {reader.fieldnames!r}"
            )
        rows = list(reader)
    with np.load(bundle_dir / "trajectories.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"scores", "offsets"}:
            raise ValueError(
                f"Unexpected trajectory arrays: {sorted(arrays.files)}"
            )
        scores = arrays["scores"].copy()
        offsets = arrays["offsets"].copy()
    if len(rows) != int(manifest["record_count"]):
        raise ValueError("record_count does not match records.csv")
    if offsets.shape != (len(rows) + 1,) or offsets[0] != 0:
        raise ValueError("Invalid score offsets")
    if np.any(np.diff(offsets) <= 0) or offsets[-1] != scores.size:
        raise ValueError("Score offsets do not span the trajectory array")
    if scores.dtype != np.float32 or offsets.dtype != np.int64:
        raise ValueError("Unexpected score-bundle array dtypes")
    if scores.size != int(manifest["score_value_count"]):
        raise ValueError("score_value_count does not match trajectories.npz")
    if not np.isfinite(scores).all():
        raise ValueError("Score trajectories contain non-finite values")

    seen_keys: set[tuple[str, int, str, int]] = set()
    seen_dataset_indices: dict[tuple[str, int, int], str] = {}
    source_inventory: set[tuple[str, int, int, int, int]] = set()
    observed_folds: dict[str, set[int]] = {}
    observed_roles: dict[tuple[str, int], set[str]] = {}
    for row_index, row in enumerate(rows):
        if int(row["record_id"]) != row_index:
            raise ValueError("record_id must match records.csv row order")
        source_key = row["source_key"]
        source_parts = Path(source_key).parts
        if (
            Path(source_key).is_absolute()
            or ".." in source_parts
            or len(source_parts) < 2
            or source_parts[0] not in aliases
        ):
            raise ValueError(f"Unsafe source_key {source_key!r}")
        if row["task_id"] == "" or row["episode_idx"] == "":
            raise ValueError(f"Incomplete rollout metadata in record {row_index}")
        task_id = int(row["task_id"])
        episode_idx = int(row["episode_idx"])
        label = int(row["label"])
        success = int(row["success"])
        length = int(row["length"])
        task_min_step = int(row["task_min_step"])
        if task_id < 0 or episode_idx < 0:
            raise ValueError(f"Negative rollout metadata in record {row_index}")
        if label not in {0, 1} or success not in {0, 1} or label != 1 - success:
            raise ValueError(f"Inconsistent outcome metadata in record {row_index}")
        if length <= 0 or not 1 <= task_min_step <= length:
            raise ValueError(f"Invalid rollout lengths in record {row_index}")
        key = (
            row["model"],
            int(row["fold"]),
            row["split_role"],
            int(row["dataset_index"]),
        )
        if key in seen_keys:
            raise ValueError(f"Duplicate score-bundle record {key}")
        seen_keys.add(key)
        dataset_key = (
            row["model"],
            int(row["fold"]),
            int(row["dataset_index"]),
        )
        previous_role = seen_dataset_indices.get(dataset_key)
        if previous_role is not None:
            raise ValueError(
                f"Dataset index {dataset_key[2]} appears in both {previous_role!r} "
                f"and {row['split_role']!r} for {dataset_key[0]} fold "
                f"{dataset_key[1]}"
            )
        seen_dataset_indices[dataset_key] = row["split_role"]
        expected_length = int(offsets[row_index + 1] - offsets[row_index])
        if int(row["n_scores"]) != expected_length:
            raise ValueError(f"Score length mismatch in record {row_index}")
        if int(row["length"]) != expected_length:
            raise ValueError(f"Rollout length mismatch in record {row_index}")
        if row["split_role"] not in {"val", "test"}:
            raise ValueError(f"Unexpected split role {row['split_role']!r}")
        source_inventory.add(
            (source_key, task_id, episode_idx, label, length)
        )
        model = row["model"]
        fold = int(row["fold"])
        observed_folds.setdefault(model, set()).add(fold)
        observed_roles.setdefault((model, fold), set()).add(row["split_role"])

    models = sorted(observed_folds)
    if models != sorted(manifest["models"]):
        raise ValueError("Manifest models do not match records.csv")
    expected_folds = {
        model: sorted(folds) for model, folds in observed_folds.items()
    }
    normalized_manifest_folds = {
        model: sorted(int(fold) for fold in folds)
        for model, folds in manifest["folds_by_model"].items()
    }
    if expected_folds != normalized_manifest_folds:
        raise ValueError("Manifest folds do not match records.csv")
    if sorted(manifest["included_split_roles"]) != ["test", "val"]:
        raise ValueError("Score bundle must include validation and test roles")
    for key, roles in observed_roles.items():
        if roles != {"val", "test"}:
            raise ValueError(f"Incomplete split roles for {key}: {sorted(roles)}")

    if int(manifest["unique_rollout_count"]) != len(source_inventory):
        raise ValueError("unique_rollout_count does not match records.csv")
    inventory_text = "\n".join(
        "|".join(str(value) for value in row)
        for row in sorted(source_inventory)
    )
    inventory_digest = hashlib.sha256(inventory_text.encode("utf-8")).hexdigest()
    if inventory_digest != manifest["source_inventory_sha256"]:
        raise ValueError("source_inventory_sha256 does not match records.csv")

    if set(checkpoint_manifest) != set(models):
        raise ValueError("Checkpoint manifest models do not match score records")
    for model, folds in expected_folds.items():
        model_checkpoints = checkpoint_manifest[model]
        if {int(fold) for fold in model_checkpoints} != set(folds):
            raise ValueError(f"Incomplete checkpoint manifest for {model}")
        for fold, detail in model_checkpoints.items():
            file_name = str(detail.get("file", ""))
            digest = str(detail.get("sha256", ""))
            if (
                not file_name
                or Path(file_name).is_absolute()
                or len(Path(file_name).parts) != 1
            ):
                raise ValueError(
                    f"Unsafe checkpoint file for {model} fold {fold}: {file_name!r}"
                )
            if not SHA256_RE.fullmatch(digest):
                raise ValueError(f"Invalid checkpoint SHA-256 for {model} fold {fold}")
    return manifest


def load_score_bundle(
    bundle_dir: Path,
) -> tuple[
    dict[str, dict[int, dict[int, RolloutScores]]],
    dict[str, dict[int, dict[str, list[int]]]],
    dict[str, Any],
]:
    """Load a verified bundle into records and split mappings."""

    manifest = verify_score_bundle(bundle_dir)
    with (bundle_dir / "records.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(bundle_dir / "trajectories.npz", allow_pickle=False) as arrays:
        scores = arrays["scores"].copy()
        offsets = arrays["offsets"].copy()

    records: dict[str, dict[int, dict[int, RolloutScores]]] = {}
    splits: dict[str, dict[int, dict[str, list[int]]]] = {}
    for row_index, row in enumerate(rows):
        model = row["model"]
        fold = int(row["fold"])
        role = row["split_role"]
        dataset_index = int(row["dataset_index"])
        start, stop = int(offsets[row_index]), int(offsets[row_index + 1])
        record = RolloutScores(
            path=row["source_key"],
            task_id=_optional_int(row["task_id"]),
            episode_idx=_optional_int(row["episode_idx"]),
            label=int(row["label"]),
            success=bool(int(row["success"])),
            length=int(row["length"]),
            task_min_step=int(row["task_min_step"]),
            scores=scores[start:stop].astype(np.float64),
        )
        fold_records = records.setdefault(model, {}).setdefault(fold, {})
        if dataset_index in fold_records:
            raise ValueError(
                f"Dataset index {dataset_index} duplicated for {model} fold {fold}"
            )
        fold_records[dataset_index] = record
        fold_splits = splits.setdefault(model, {}).setdefault(
            fold, {"val": [], "test": []}
        )
        fold_splits[role].append(dataset_index)

    for model in splits:
        for fold in splits[model]:
            for role in splits[model][fold]:
                splits[model][fold][role].sort()
    return records, splits, manifest
