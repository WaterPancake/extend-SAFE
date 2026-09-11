"""Verify links, tabular artifacts, manifests, and the public score bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_bundle import load_score_bundle, sha256_file
from scripts.compare_functional_cp_outputs import REQUIRED_TABLES
LOCAL_LINK_RE = re.compile(r"\[[^]]*\]\(([^)]+)\)")
PRIMARY_ALPHAS = {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3}
PRIMARY_EVAL_TIMES = {
    "by final end",
    "by earliest stop",
    "fixed horizon 50",
    "fixed horizon 100",
    "fixed horizon 148",
}
FIXED_EVAL_TIMES = {
    "fixed horizon 50",
    "fixed horizon 100",
    "fixed horizon 148",
}
DYNAMIC_MODELS = {
    "baseline32",
    "layer20_32",
    "shuffled20_32",
    "residual20_32",
    "shuffled_residual20_32",
}
DYNAMIC_EVAL_TIMES = {
    "by earliest stop",
    "fixed horizon 13",
    "fixed horizon 25",
    "fixed horizon 37",
}
DYNAMIC_COMPARISONS = {
    ("layer20_32", "baseline32"),
    ("shuffled20_32", "baseline32"),
    ("layer20_32", "shuffled20_32"),
    ("residual20_32", "baseline32"),
    ("shuffled_residual20_32", "baseline32"),
    ("residual20_32", "shuffled_residual20_32"),
}
REQUIRED_GENERATORS = {
    "data/dataloaders.py",
    "models/__init__.py",
    "models/base.py",
    "models/layer_mix.py",
    "models/linear_probe.py",
    "models/lstm.py",
    "models/mlp.py",
    "models/safe_losses.py",
    "scripts/evaluate_conformal.py",
    "scripts/evaluate_functional_cp_loto.py",
    "scripts/prepare_openvla_layer_subset.py",
    "scripts/score_bundle.py",
    "scripts/score_layer_pooled_macro.py",
    "scripts/verify_primary_prepared_data.py",
}


def verify_links(document: Path) -> list[str]:
    failures = []
    for raw_target in LOCAL_LINK_RE.findall(document.read_text()):
        target = raw_target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if not (document.parent / target).exists():
            failures.append(f"{document}: broken local link {raw_target!r}")
    return failures


def verify_tables(result_root: Path) -> list[str]:
    failures = []
    for path in sorted(result_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                json.loads(path.read_text())
            elif path.suffix == ".csv":
                with path.open(newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, None)
                    if not header:
                        raise ValueError("empty CSV")
                    width = len(header)
                    for row_number, row in enumerate(reader, start=2):
                        if len(row) != width:
                            raise ValueError(
                                f"row {row_number} has {len(row)} fields, expected {width}"
                            )
        except Exception as error:  # reported with the artifact path below
            failures.append(f"{path}: {error}")
    return failures


def verify_publication_manifest(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing publication manifest: {path}"]
    failures = []
    manifest = json.loads(path.read_text())
    for section in ("reports", "result_artifacts", "analysis_code_and_environment"):
        if section not in manifest:
            failures.append(f"{path}: missing section {section!r}")
            continue
        for relative, expected in manifest[section].items():
            artifact = REPO_ROOT / relative
            if not artifact.exists():
                failures.append(f"{path}: missing hashed file {relative}")
                continue
            actual = sha256_file(artifact)
            if actual != expected:
                failures.append(
                    f"{path}: checksum mismatch for {relative}: {actual} != {expected}"
                )
    return failures


def command_value(command: str, option: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return None


def verify_primary_result_tables(result_root: Path) -> list[str]:
    """Check exact row domains/cardinalities for the primary release tables."""

    specs = {
        "functional_cp_loto_per_task.csv": (
            800,
            PRIMARY_EVAL_TIMES,
            ("model", "fold", "alpha", "eval_time"),
        ),
        "functional_cp_loto_macro.csv": (
            80,
            PRIMARY_EVAL_TIMES,
            ("model", "alpha", "eval_time"),
        ),
        "functional_cp_loto_cp_seed_sensitivity.csv": (
            800,
            PRIMARY_EVAL_TIMES,
            ("model", "cp_seed", "alpha", "eval_time"),
        ),
        "functional_cp_loto_operating_points.csv": (
            60,
            PRIMARY_EVAL_TIMES,
            ("model", "eval_time", "constraint_type", "constraint_value"),
        ),
        "functional_cp_fixed_horizon_raw.csv": (
            4_800,
            FIXED_EVAL_TIMES,
            ("model", "fold", "cp_seed", "alpha", "eval_time"),
        ),
        "functional_cp_fixed_horizon_per_task.csv": (
            480,
            FIXED_EVAL_TIMES,
            ("model", "fold", "alpha", "eval_time"),
        ),
        "functional_cp_fixed_horizon_macro.csv": (
            48,
            FIXED_EVAL_TIMES,
            ("model", "alpha", "eval_time"),
        ),
        "functional_cp_fixed_horizon_operating_points.csv": (
            36,
            FIXED_EVAL_TIMES,
            ("model", "eval_time", "constraint_type", "constraint_value"),
        ),
    }
    failures = []
    for name, (expected_count, expected_times, key_columns) in specs.items():
        path = result_root / name
        if not path.exists():
            failures.append(f"missing required primary result table: {path}")
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count:
            failures.append(
                f"{path}: expected {expected_count} rows, found {len(rows)}"
            )
            continue
        fields = set(rows[0]) if rows else set()
        missing_fields = sorted(
            {"model", "eval_time", *key_columns} - fields
        )
        if missing_fields:
            failures.append(f"{path}: missing columns {missing_fields}")
            continue
        models = {row["model"] for row in rows}
        eval_times = {row["eval_time"] for row in rows}
        if models != {"mlp", "lstm"}:
            failures.append(f"{path}: models are {sorted(models)}")
        if eval_times != expected_times:
            failures.append(f"{path}: eval_time values are {sorted(eval_times)}")
        if "alpha" in fields and "operating_points" not in name:
            alphas = {float(row["alpha"]) for row in rows}
            if alphas != PRIMARY_ALPHAS:
                failures.append(f"{path}: alpha grid is {sorted(alphas)}")
        keys = {tuple(row[column] for column in key_columns) for row in rows}
        if len(keys) != len(rows):
            failures.append(f"{path}: duplicate result keys")
        if "fold" in fields:
            folds = {int(row["fold"]) for row in rows}
            if folds != set(range(10)):
                failures.append(f"{path}: folds are {sorted(folds)}")
            if "heldout_task" in fields and any(
                int(row["heldout_task"]) != int(row["fold"]) for row in rows
            ):
                failures.append(f"{path}: heldout_task does not match fold")
        if "cp_seed" in fields:
            cp_seeds = {int(row["cp_seed"]) for row in rows}
            if cp_seeds != set(range(10)):
                failures.append(f"{path}: CP seeds are {sorted(cp_seeds)}")
        if "n_tasks" in fields and any(int(float(row["n_tasks"])) != 10 for row in rows):
            failures.append(f"{path}: n_tasks must equal 10")
    return failures


def verify_dynamic_layer_result_tables(result_root: Path) -> list[str]:
    """Check the published 1,000-rollout multilayer artifact domains."""

    failures = []
    expected_task_counts = {str(task): 50 for task in range(10)}
    for shard in (0, 1):
        path = result_root / f"dynamic_layer_data_manifest_1000_shard{shard}.json"
        if not path.exists():
            failures.append(f"missing dynamic-layer shard manifest: {path}")
            continue
        payload = json.loads(path.read_text())
        audit = payload.get("output_audit", {})
        schemas = audit.get("schemas", [])
        if (
            payload.get("corrupt_source_artifacts") != []
            or audit.get("rollouts") != 500
            or audit.get("task_counts") != expected_task_counts
            or not audit.get("homogeneous_schema")
            or int(audit.get("minimum_length", 0)) < 37
            or len(schemas) != 1
            or schemas[0].get("layers") != [20, 32]
            or schemas[0].get("hidden_dim_per_layer") != 4096
            or schemas[0].get("hidden_shape_suffix") != [1, 8192]
            or schemas[0].get("temporal_stride") != 4
        ):
            failures.append(f"{path}: invalid 500-rollout compact-shard contract")

    summary_path = result_root / "dynamic_layer_loto_1000_summary.json"
    if not summary_path.exists():
        failures.append(f"missing dynamic-layer summary: {summary_path}")
    else:
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("n_train_seeds") != 3
            or summary.get("train_seeds") != [20260722, 20260723, 20260724]
            or summary.get("n_tasks") != 10
            or summary.get("tasks") != list(range(10))
            or set(summary.get("variants", {})) != DYNAMIC_MODELS
        ):
            failures.append(f"{summary_path}: invalid ranking summary domain")
        functional = summary.get("functional_cp") or {}
        points = functional.get("fpr_at_most_5_percent", [])
        point_domain = {
            (row.get("model"), row.get("eval_time")) for row in points
        }
        if (
            functional.get("n_train_seeds") != 3
            or functional.get("train_seeds") != [20260722, 20260723, 20260724]
            or functional.get("n_cp_artifact_sets") != 1
            or len(points) != 20
            or point_domain
            != {
                (model, eval_time)
                for model in DYNAMIC_MODELS
                for eval_time in DYNAMIC_EVAL_TIMES
            }
        ):
            failures.append(f"{summary_path}: invalid functional-CP summary domain")

    table_specs = {
        "dynamic_layer_loto_1000_per_task.csv": (
            50,
            ("variant", "task"),
        ),
        "dynamic_layer_functional_cp_1000_macro.csv": (
            160,
            ("model", "alpha", "eval_time"),
        ),
        "dynamic_layer_functional_cp_1000_operating_points.csv": (
            120,
            ("model", "eval_time", "constraint_type", "constraint_value"),
        ),
        "dynamic_layer_functional_cp_1000_comparisons.csv": (
            216,
            (
                "comparison_type",
                "left",
                "right",
                "eval_time",
                "left_alpha",
                "right_alpha",
            ),
        ),
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for name, (expected_rows, key_columns) in table_specs.items():
        path = result_root / name
        if not path.exists():
            failures.append(f"missing dynamic-layer result table: {path}")
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        tables[name] = rows
        if len(rows) != expected_rows:
            failures.append(
                f"{path}: expected {expected_rows} rows, found {len(rows)}"
            )
            continue
        if not set(key_columns) <= set(rows[0]):
            failures.append(f"{path}: missing result-key columns")
            continue
        keys = {tuple(row[column] for column in key_columns) for row in rows}
        if len(keys) != len(rows):
            failures.append(f"{path}: duplicate result keys")

    ranking = tables.get("dynamic_layer_loto_1000_per_task.csv", [])
    if ranking and (
        {row["variant"] for row in ranking} != DYNAMIC_MODELS
        or {int(row["task"]) for row in ranking} != set(range(10))
    ):
        failures.append("dynamic-layer ranking table has an invalid model/task domain")

    for name in (
        "dynamic_layer_functional_cp_1000_macro.csv",
        "dynamic_layer_functional_cp_1000_operating_points.csv",
    ):
        rows = tables.get(name, [])
        if rows and (
            {row["model"] for row in rows} != DYNAMIC_MODELS
            or {row["eval_time"] for row in rows} != DYNAMIC_EVAL_TIMES
        ):
            failures.append(f"{name}: invalid model/evaluation-window domain")
    macro = tables.get("dynamic_layer_functional_cp_1000_macro.csv", [])
    if macro and (
        {float(row["alpha"]) for row in macro} != PRIMARY_ALPHAS
        or any(int(float(row["n_tasks"])) != 10 for row in macro)
    ):
        failures.append("dynamic-layer macro table has an invalid alpha/task domain")

    comparisons = tables.get(
        "dynamic_layer_functional_cp_1000_comparisons.csv", []
    )
    if comparisons and (
        {(row["left"], row["right"]) for row in comparisons}
        != DYNAMIC_COMPARISONS
        or {row["eval_time"] for row in comparisons} != DYNAMIC_EVAL_TIMES
        or {row["comparison_type"] for row in comparisons}
        != {
            "same nominal alpha",
            "independent FPR <= 0.05 operating points",
        }
        or any(int(float(row["n_tasks"])) != 10 for row in comparisons)
    ):
        failures.append("dynamic-layer comparison table has an invalid domain")

    replication_models = {
        "baseline32",
        "late24_28_32",
        "shuffled_late24_28_32",
    }
    replication_manifest = (
        result_root / "dynamic_late_layer_replication_data_manifest.json"
    )
    if not replication_manifest.exists():
        failures.append(f"missing late-layer replication manifest: {replication_manifest}")
    else:
        payload = json.loads(replication_manifest.read_text())
        audit = payload.get("output_audit", {})
        schemas = audit.get("schemas", [])
        if (
            payload.get("corrupt_source_artifacts") != []
            or audit.get("rollouts") != 500
            or audit.get("task_counts") != expected_task_counts
            or not audit.get("homogeneous_schema")
            or int(audit.get("minimum_length", 0)) < 37
            or len(schemas) != 1
            or schemas[0].get("layers") != [24, 28, 32]
            or schemas[0].get("hidden_dim_per_layer") != 4096
            or schemas[0].get("hidden_shape_suffix") != [1, 12288]
            or schemas[0].get("temporal_stride") != 4
        ):
            failures.append(
                f"{replication_manifest}: invalid late-layer replication contract"
            )

    replication_summary_path = (
        result_root / "dynamic_late_layer_replication_loto_summary.json"
    )
    if not replication_summary_path.exists():
        failures.append(
            f"missing late-layer replication summary: {replication_summary_path}"
        )
    else:
        summary = json.loads(replication_summary_path.read_text())
        functional = summary.get("functional_cp") or {}
        points = functional.get("fpr_at_most_5_percent", [])
        if (
            summary.get("n_train_seeds") != 3
            or summary.get("train_seeds") != [20260722, 20260723, 20260724]
            or summary.get("n_tasks") != 10
            or summary.get("tasks") != list(range(10))
            or set(summary.get("variants", {})) != replication_models
            or functional.get("n_train_seeds") != 3
            or functional.get("n_cp_artifact_sets") != 1
            or len(points) != 12
            or {
                (row.get("model"), row.get("eval_time")) for row in points
            }
            != {
                (model, eval_time)
                for model in replication_models
                for eval_time in DYNAMIC_EVAL_TIMES
            }
        ):
            failures.append(
                f"{replication_summary_path}: invalid late-layer replication domain"
            )

    replication_specs = {
        "dynamic_late_layer_replication_loto_per_task.csv": (
            30,
            ("variant", "task"),
        ),
        "dynamic_late_layer_replication_functional_cp_macro.csv": (
            96,
            ("model", "alpha", "eval_time"),
        ),
        "dynamic_late_layer_replication_functional_cp_operating_points.csv": (
            72,
            ("model", "eval_time", "constraint_type", "constraint_value"),
        ),
        "dynamic_late_layer_replication_functional_cp_comparisons.csv": (
            108,
            (
                "comparison_type",
                "left",
                "right",
                "eval_time",
                "left_alpha",
                "right_alpha",
            ),
        ),
    }
    replication_tables: dict[str, list[dict[str, str]]] = {}
    for name, (expected_rows, key_columns) in replication_specs.items():
        path = result_root / name
        if not path.exists():
            failures.append(f"missing late-layer replication table: {path}")
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        replication_tables[name] = rows
        if len(rows) != expected_rows:
            failures.append(
                f"{path}: expected {expected_rows} rows, found {len(rows)}"
            )
        elif not rows or not set(key_columns) <= set(rows[0]):
            failures.append(f"{path}: missing result-key columns")
        elif len(
            {tuple(row[column] for column in key_columns) for row in rows}
        ) != len(rows):
            failures.append(f"{path}: duplicate result keys")

    replication_ranking = replication_tables.get(
        "dynamic_late_layer_replication_loto_per_task.csv", []
    )
    if replication_ranking and (
        {row["variant"] for row in replication_ranking} != replication_models
        or {int(row["task"]) for row in replication_ranking} != set(range(10))
    ):
        failures.append("late-layer replication ranking has an invalid domain")
    for name in (
        "dynamic_late_layer_replication_functional_cp_macro.csv",
        "dynamic_late_layer_replication_functional_cp_operating_points.csv",
    ):
        rows = replication_tables.get(name, [])
        if rows and (
            {row["model"] for row in rows} != replication_models
            or {row["eval_time"] for row in rows} != DYNAMIC_EVAL_TIMES
        ):
            failures.append(f"{name}: invalid model/evaluation-window domain")
    replication_macro = replication_tables.get(
        "dynamic_late_layer_replication_functional_cp_macro.csv", []
    )
    if replication_macro and (
        {float(row["alpha"]) for row in replication_macro} != PRIMARY_ALPHAS
        or any(int(float(row["n_tasks"])) != 10 for row in replication_macro)
    ):
        failures.append("late-layer replication macro table has an invalid domain")
    replication_comparisons = replication_tables.get(
        "dynamic_late_layer_replication_functional_cp_comparisons.csv", []
    )
    expected_replication_pairs = {
        ("late24_28_32", "baseline32"),
        ("shuffled_late24_28_32", "baseline32"),
        ("late24_28_32", "shuffled_late24_28_32"),
    }
    if replication_comparisons and (
        {
            (row["left"], row["right"]) for row in replication_comparisons
        }
        != expected_replication_pairs
        or {row["eval_time"] for row in replication_comparisons}
        != DYNAMIC_EVAL_TIMES
        or any(
            int(float(row["n_tasks"])) != 10
            for row in replication_comparisons
        )
    ):
        failures.append("late-layer replication comparisons have an invalid domain")
    return failures


def verify_public_score_bundle(
    bundle: Path, checkpoint_audit: Path | None = None
) -> list[str]:
    """Apply release-specific coverage checks after schema verification."""

    failures = []
    try:
        records, splits, manifest = load_score_bundle(bundle)
    except Exception as error:
        return [f"{bundle}: {error}"]

    if int(manifest["unique_rollout_count"]) != 1000:
        failures.append(
            f"{bundle}: expected 1,000 unique rollouts, found "
            f"{manifest['unique_rollout_count']}"
        )
    if sorted(manifest["models"]) != ["lstm", "mlp"]:
        failures.append(
            f"{bundle}: expected MLP and LSTM scores, found {manifest['models']}"
        )
    expected_folds = list(range(10))
    if int(manifest["record_count"]) != 9_200:
        failures.append(
            f"{bundle}: expected 9,200 model/fold/split records, found "
            f"{manifest['record_count']}"
        )
    expected_aliases = {
        "root-0": "openVLA-last-layer",
        "root-1": "openVLA-last-layer-2",
    }
    if manifest.get("source_root_aliases") != expected_aliases:
        failures.append(
            f"{bundle}: source root aliases are {manifest.get('source_root_aliases')}, "
            f"expected {expected_aliases}"
        )
    preparations = manifest.get("source_preparation_manifests", {})
    if preparations and set(preparations) != set(expected_aliases):
        failures.append(
            f"{bundle}: prepared-data provenance must cover both roots"
        )
    for alias, detail in preparations.items():
        audit = detail.get("manifest", {}).get("output_audit", {})
        expected_task_counts = {str(task): 50 for task in range(10)}
        schemas = audit.get("schemas", [])
        if (
            audit.get("rollouts") != 500
            or audit.get("task_counts") != expected_task_counts
            or not audit.get("homogeneous_schema")
            or int(audit.get("minimum_length", 0)) < 148
            or len(schemas) != 1
            or schemas[0].get("layers") != [32]
            or schemas[0].get("hidden_dim_per_layer") != 4096
            or schemas[0].get("hidden_shape_suffix") != [1, 4096]
            or schemas[0].get("temporal_stride") != 1
        ):
            failures.append(
                f"{bundle}: invalid prepared-data provenance for {alias}"
            )
    command = str(manifest.get("command") or "")
    expected_options = {
        "--mlp-checkpoint-dir": "runs/openvla_mlp_loto_l32",
        "--lstm-checkpoint-dir": "runs/openvla_lstm_loto_l32",
        "--horizon": "520",
        "--fixed-horizons": "50,100,148",
        "--alphas": "0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3",
        "--cp-seeds": "0,1,2,3,4,5,6,7,8,9",
        "--task-bootstrap": "10000",
        "--bootstrap-seed": "20260716",
    }
    for option, expected in expected_options.items():
        actual = command_value(command, option)
        if actual != expected:
            failures.append(
                f"{bundle}: generation command {option} is {actual!r}, "
                f"expected {expected!r}"
            )
    if command_value(command, "--score-bundle-out") is None:
        failures.append(f"{bundle}: generation command lacks --score-bundle-out")

    generators = manifest.get("generator_sha256", {})
    if not REQUIRED_GENERATORS <= set(generators):
        failures.append(
            f"{bundle}: missing generator hashes "
            f"{sorted(REQUIRED_GENERATORS - set(generators))}"
        )
    for relative in sorted(REQUIRED_GENERATORS & set(generators)):
        source = REPO_ROOT / relative
        if not source.exists():
            failures.append(f"{bundle}: missing generator source {relative}")
        elif sha256_file(source) != generators[relative]:
            failures.append(
                f"{bundle}: generator hash does not match current source {relative}"
            )
    if checkpoint_audit is not None:
        if not checkpoint_audit.exists():
            failures.append(f"missing primary checkpoint audit: {checkpoint_audit}")
        else:
            audit_payload = json.loads(checkpoint_audit.read_text())
            audited_checkpoints = audit_payload.get("checkpoint_manifest")
            if manifest.get("checkpoint_manifest") != audited_checkpoints:
                failures.append(
                    f"{bundle}: checkpoint hashes do not match {checkpoint_audit}"
                )
    identities_by_model = {}
    for model in ("mlp", "lstm"):
        folds = [
            int(fold)
            for fold in manifest.get("folds_by_model", {}).get(model, [])
        ]
        if sorted(folds) != expected_folds:
            failures.append(
                f"{bundle}: {model} must contain LOTO folds 0-9, found "
                f"{sorted(folds)}"
            )
            continue
        test_indices = []
        test_source_keys = []
        identity_by_index = {}
        for fold in expected_folds:
            fold_splits = splits[model][fold]
            validation = fold_splits["val"]
            test = fold_splits["test"]
            if len(validation) != 360 or len(test) != 100:
                failures.append(
                    f"{bundle}: {model} fold {fold} must have 360 validation "
                    f"and 100 test records, found {len(validation)} and {len(test)}"
                )
            if set(validation) & set(test):
                failures.append(
                    f"{bundle}: {model} fold {fold} validation/test indices overlap"
                )
            fold_records = records[model][fold]
            test_tasks = {int(fold_records[index].task_id) for index in test}
            validation_tasks = {
                int(fold_records[index].task_id) for index in validation
            }
            if test_tasks != {fold}:
                failures.append(
                    f"{bundle}: {model} fold {fold} test tasks are "
                    f"{sorted(test_tasks)}, expected [{fold}]"
                )
            if fold in validation_tasks:
                failures.append(
                    f"{bundle}: {model} fold {fold} leaks the held-out task "
                    "into validation"
                )
            for index in validation + test:
                record = fold_records[index]
                if record.length < 148:
                    failures.append(
                        f"{bundle}: {model} fold {fold} index {index} has length "
                        f"{record.length}, below fixed horizon 148"
                    )
                identity = (
                    record.path,
                    record.task_id,
                    record.episode_idx,
                    record.label,
                    record.length,
                    record.task_min_step,
                )
                previous = identity_by_index.setdefault(index, identity)
                if previous != identity:
                    failures.append(
                        f"{bundle}: {model} dataset index {index} has inconsistent "
                        "rollout metadata across folds"
                    )
            test_indices.extend(test)
            test_source_keys.extend(fold_records[index].path for index in test)
        if sorted(test_indices) != list(range(1_000)):
            failures.append(
                f"{bundle}: {model} held-out tests must cover dataset indices "
                "0..999 exactly once"
            )
        if len(set(test_source_keys)) != 1_000:
            failures.append(
                f"{bundle}: {model} held-out tests do not contain 1,000 unique "
                "source keys"
            )
        identities_by_model[model] = identity_by_index
    if set(splits) == {"mlp", "lstm"} and splits["mlp"] != splits["lstm"]:
        failures.append(f"{bundle}: MLP and LSTM split mappings differ")
    if checkpoint_audit is not None and checkpoint_audit.exists() and "mlp" in splits:
        normalized_splits = {
            str(fold): {
                role: sorted(indices) for role, indices in sorted(split.items())
            }
            for fold, split in sorted(splits["mlp"].items())
        }
        split_sha256 = hashlib.sha256(
            json.dumps(
                normalized_splits, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        expected_split_sha256 = json.loads(checkpoint_audit.read_text()).get(
            "split_sha256"
        )
        if split_sha256 != expected_split_sha256:
            failures.append(
                f"{bundle}: split mapping digest does not match {checkpoint_audit}"
            )
    if set(identities_by_model) == {"mlp", "lstm"} and (
        identities_by_model["mlp"] != identities_by_model["lstm"]
    ):
        failures.append(f"{bundle}: MLP and LSTM rollout metadata differ")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-score-bundle",
        action="store_true",
        help="Development-only escape hatch; release CI does not use this.",
    )
    args = parser.parse_args()
    result_root = REPO_ROOT / "docs" / "results_audit"
    required = [
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
        REPO_ROOT / "docs" / "safe_openvla_audit.md",
        REPO_ROOT / "docs" / "safe_openvla_audit.qmd",
    ]
    failures = [f"missing required file: {path}" for path in required if not path.exists()]
    failures.extend(verify_links(REPO_ROOT / "README.md"))
    failures.extend(verify_links(REPO_ROOT / "docs" / "safe_openvla_audit.md"))
    failures.extend(verify_tables(result_root))
    failures.extend(verify_dynamic_layer_result_tables(result_root))
    failures.extend(
        verify_publication_manifest(result_root / "publication_manifest.json")
    )

    bundle = result_root / "score_bundle"
    if bundle.exists():
        failures.extend(verify_primary_result_tables(result_root))
        failures.extend(
            verify_public_score_bundle(
                bundle, result_root / "primary_checkpoint_audit.json"
            )
        )
    elif not args.allow_missing_score_bundle:
        failures.append(f"missing required public score bundle: {bundle}")

    if failures:
        raise SystemExit("publication verification failed:\n- " + "\n- ".join(failures))
    print("publication verification passed")


if __name__ == "__main__":
    main()
