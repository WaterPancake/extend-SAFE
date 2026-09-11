from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_conformal import RolloutScores
from scripts.evaluate_functional_cp_loto import evaluate_family
from scripts.score_bundle import (
    load_score_bundle,
    sha256_file,
    verify_score_bundle,
    write_score_bundle,
)
from scripts.verify_publication import (
    REPO_ROOT,
    REQUIRED_GENERATORS,
    verify_public_score_bundle,
)


def make_record(root: Path, index: int, task: int, label: int) -> RolloutScores:
    return RolloutScores(
        path=str(root / f"task{task}--ep{index}--succ{1-label}.pkl"),
        task_id=task,
        episode_idx=index,
        label=label,
        success=not bool(label),
        length=5,
        task_min_step=4,
        scores=np.linspace(0.1 + label, 0.5 + label, 5),
    )


def test_score_bundle_round_trip_and_path_sanitization(tmp_path: Path) -> None:
    source = tmp_path / "private" / "raw"
    records = {
        model: {
            0: {index: make_record(source, index, index // 2, index % 2) for index in range(4)}
        }
        for model in ("mlp", "lstm")
    }
    splits = {
        model: {0: {"val": [0, 1], "test": [2, 3]}}
        for model in ("mlp", "lstm")
    }
    checkpoints = {}
    for model in ("mlp", "lstm"):
        path = tmp_path / f"{model}.pt"
        path.write_bytes(f"checkpoint-{model}".encode())
        checkpoints[model] = {0: path}

    bundle = tmp_path / "bundle"
    manifest = write_score_bundle(
        bundle,
        records,
        splits,
        [source],
        checkpoints,
        generator_paths=[Path(__file__)],
        command="synthetic bundle",
    )
    assert manifest["unique_rollout_count"] == 4
    assert manifest["record_count"] == 8

    with (bundle / "records.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["source_key"].startswith("root-0/") for row in rows)
    assert str(tmp_path) not in (bundle / "records.csv").read_text()
    assert str(tmp_path) not in (bundle / "manifest.json").read_text()

    loaded_records, loaded_splits, loaded_manifest = load_score_bundle(bundle)
    assert loaded_manifest["files"] == manifest["files"]
    assert loaded_splits == splits
    np.testing.assert_allclose(
        loaded_records["mlp"][0][2].scores,
        records["mlp"][0][2].scores,
        rtol=1e-6,
    )


def test_score_bundle_detects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    records = {"mlp": {0: {0: make_record(source, 0, 0, 0), 1: make_record(source, 1, 1, 1)}}}
    splits = {"mlp": {0: {"val": [0], "test": [1]}}}
    checkpoint = tmp_path / "mlp.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = tmp_path / "bundle"
    write_score_bundle(bundle, records, splits, [source], {"mlp": {0: checkpoint}})

    with (bundle / "records.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_score_bundle(bundle)


def test_score_bundle_rejects_incomplete_rollout_metadata(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    records = {
        "mlp": {
            0: {
                0: make_record(source, 0, 1, 0),
                1: make_record(source, 1, 0, 1),
            }
        }
    }
    splits = {"mlp": {0: {"val": [0], "test": [1]}}}
    checkpoint = tmp_path / "mlp.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = tmp_path / "bundle"
    write_score_bundle(bundle, records, splits, [source], {"mlp": {0: checkpoint}})

    records_path = bundle / "records.csv"
    with records_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["task_id"] = ""
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["records.csv"] = sha256_file(records_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="Incomplete rollout metadata"):
        verify_score_bundle(bundle)


def test_score_bundle_rejects_validation_test_overlap(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    records = {"mlp": {0: {0: make_record(source, 0, 0, 0)}}}
    splits = {"mlp": {0: {"val": [0], "test": [0]}}}
    checkpoint = tmp_path / "mlp.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="appears in both"):
        write_score_bundle(
            tmp_path / "bundle",
            records,
            splits,
            [source],
            {"mlp": {0: checkpoint}},
        )


def test_score_bundle_rejects_unsafe_source_key(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    records = {
        "mlp": {
            0: {
                0: make_record(source, 0, 0, 0),
                1: make_record(source, 1, 1, 1),
            }
        }
    }
    splits = {"mlp": {0: {"val": [0], "test": [1]}}}
    checkpoint = tmp_path / "mlp.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = tmp_path / "bundle"
    write_score_bundle(bundle, records, splits, [source], {"mlp": {0: checkpoint}})

    records_path = bundle / "records.csv"
    with records_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["source_key"] = "../private.pkl"
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["records.csv"] = sha256_file(records_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="Unsafe source_key"):
        verify_score_bundle(bundle)


def test_score_bundle_rejects_invalid_offsets(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    records = {
        "mlp": {
            0: {
                0: make_record(source, 0, 0, 0),
                1: make_record(source, 1, 1, 1),
            }
        }
    }
    splits = {"mlp": {0: {"val": [0], "test": [1]}}}
    checkpoint = tmp_path / "mlp.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = tmp_path / "bundle"
    write_score_bundle(bundle, records, splits, [source], {"mlp": {0: checkpoint}})

    trajectories_path = bundle / "trajectories.npz"
    with np.load(trajectories_path, allow_pickle=False) as arrays:
        scores = arrays["scores"].copy()
        offsets = arrays["offsets"].copy()
    offsets[1] = offsets[0]
    np.savez_compressed(trajectories_path, scores=scores, offsets=offsets)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["trajectories.npz"] = sha256_file(trajectories_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(ValueError, match="offsets"):
        verify_score_bundle(bundle)


def replay_fixture(
    source: Path,
) -> tuple[
    dict[str, dict[int, dict[int, RolloutScores]]],
    dict[str, dict[int, dict[str, list[int]]]],
]:
    family_records = {}
    family_splits = {}
    for model_index, model in enumerate(("mlp", "lstm")):
        records_by_fold = {}
        splits_by_fold = {}
        for fold in range(10):
            seen_task = (fold + 1) % 10
            fold_records = {}
            for index in range(10):
                is_test = index >= 6
                label = int(index >= (8 if is_test else 4))
                task = fold if is_test else seen_task
                base = 0.10 + 0.01 * index + 0.005 * model_index
                if is_test and label:
                    base += 0.35
                fold_records[index] = RolloutScores(
                    path=str(
                        source
                        / f"fold-{fold}"
                        / f"task{task}--ep{index}--succ{1-label}.pkl"
                    ),
                    task_id=task,
                    episode_idx=index,
                    label=label,
                    success=not bool(label),
                    length=8,
                    task_min_step=6,
                    scores=np.asarray(
                        [
                            base
                            + 0.0125 * step
                            + 0.0005 * (index + 1) * step**2
                            for step in range(8)
                        ],
                        dtype=np.float32,
                    ).astype(np.float64),
                )
            records_by_fold[fold] = fold_records
            splits_by_fold[fold] = {
                "val": list(range(6)),
                "test": list(range(6, 10)),
            }
        family_records[model] = records_by_fold
        family_splits[model] = splits_by_fold
    return family_records, family_splits


def test_bundle_replay_reproduces_full_and_fixed_horizon_tables(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    records, splits = replay_fixture(source)
    checkpoints = {}
    for model in records:
        checkpoints[model] = {}
        for fold in range(10):
            checkpoint = tmp_path / f"{model}-fold-{fold}.pt"
            checkpoint.write_bytes(f"{model}-{fold}".encode())
            checkpoints[model][fold] = checkpoint

    expected = {
        model: evaluate_family(
            model,
            records[model],
            splits[model],
            alphas=[0.2],
            cp_seeds=[0, 1],
            horizon=8,
            fixed_horizons=[4],
        )
        for model in records
    }
    bundle = tmp_path / "bundle"
    write_score_bundle(bundle, records, splits, [source], checkpoints)
    loaded_records, loaded_splits, _ = load_score_bundle(bundle)

    for model in expected:
        replayed = evaluate_family(
            model,
            loaded_records[model],
            loaded_splits[model],
            alphas=[0.2],
            cp_seeds=[0, 1],
            horizon=8,
            fixed_horizons=[4],
        )
        assert set(replayed["eval_time"]) == {
            "by final end",
            "by earliest stop",
            "fixed horizon 4",
        }
        pd.testing.assert_frame_equal(
            replayed,
            expected[model],
            check_exact=False,
            rtol=1e-6,
            atol=1e-7,
        )


def test_public_bundle_contract_accepts_exact_primary_layout(tmp_path: Path) -> None:
    sources = [
        tmp_path / "openVLA-last-layer",
        tmp_path / "openVLA-last-layer-2",
    ]
    base_records = {
        index: RolloutScores(
            path=str(
                sources[index // 500]
                / f"task{index % 500 // 50}--ep{index % 50}--succ{1 - index % 2}.pkl"
            ),
            task_id=index % 500 // 50,
            episode_idx=index % 50,
            label=index % 2,
            success=not bool(index % 2),
            length=148,
            task_min_step=148,
            scores=np.linspace(0.0, 1.0 + index % 2, 148, dtype=np.float32),
        )
        for index in range(1_000)
    }
    records = {
        model: {fold: base_records for fold in range(10)}
        for model in ("mlp", "lstm")
    }
    splits = {model: {} for model in records}
    checkpoints = {model: {} for model in records}
    for model in records:
        for fold in range(10):
            test = list(range(fold * 50, (fold + 1) * 50)) + list(
                range(500 + fold * 50, 500 + (fold + 1) * 50)
            )
            validation = [
                index
                for task in range(10)
                if task != fold
                for index in range(task * 50, task * 50 + 40)
            ]
            splits[model][fold] = {"val": validation, "test": test}
            checkpoint = tmp_path / f"{model}-fold-{fold}.pt"
            checkpoint.write_bytes(f"{model}-{fold}".encode())
            checkpoints[model][fold] = checkpoint

    bundle = tmp_path / "bundle"
    command = (
        "scripts/evaluate_functional_cp_loto.py "
        "--mlp-checkpoint-dir runs/openvla_mlp_loto_l32 "
        "--lstm-checkpoint-dir runs/openvla_lstm_loto_l32 --horizon 520 "
        "--fixed-horizons 50,100,148 "
        "--alphas 0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3 "
        "--cp-seeds 0,1,2,3,4,5,6,7,8,9 --task-bootstrap 10000 "
        "--bootstrap-seed 20260716 --score-bundle-out bundle"
    )
    manifest = write_score_bundle(
        bundle,
        records,
        splits,
        sources,
        checkpoints,
        generator_paths=[REPO_ROOT / path for path in REQUIRED_GENERATORS],
        command=command,
    )

    assert manifest["record_count"] == 9_200
    assert manifest["unique_rollout_count"] == 1_000
    assert verify_public_score_bundle(bundle) == []
