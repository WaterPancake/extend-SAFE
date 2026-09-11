from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from scripts.verify_publication import (
    REPO_ROOT,
    verify_dynamic_layer_result_tables,
    verify_primary_result_tables,
)


MODELS = ("mlp", "lstm")
FOLDS = tuple(range(10))
CP_SEEDS = tuple(range(10))
ALPHAS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
FIXED_TIMES = ("fixed horizon 50", "fixed horizon 100", "fixed horizon 148")
ALL_TIMES = ("by final end", "by earliest stop", *FIXED_TIMES)
CONSTRAINTS = (
    ("realized_fpr_at_most", 0.01),
    ("realized_fpr_at_most", 0.05),
    ("realized_fpr_at_most", 0.10),
    ("catch_rate_at_least", 0.25),
    ("catch_rate_at_least", 0.50),
    ("catch_rate_at_least", 0.75),
)


def write_primary_tables(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    def result_rows(times: tuple[str, ...], include_cp_seed: bool) -> list[dict]:
        axes = (MODELS, FOLDS, CP_SEEDS, ALPHAS, times) if include_cp_seed else (
            MODELS,
            FOLDS,
            ALPHAS,
            times,
        )
        rows = []
        for values in itertools.product(*axes):
            if include_cp_seed:
                model, fold, cp_seed, alpha, eval_time = values
            else:
                model, fold, alpha, eval_time = values
                cp_seed = None
            row = {
                "model": model,
                "fold": fold,
                "heldout_task": fold,
                "alpha": alpha,
                "eval_time": eval_time,
            }
            if cp_seed is not None:
                row["cp_seed"] = cp_seed
            rows.append(row)
        return rows

    pd.DataFrame(result_rows(ALL_TIMES, False)).to_csv(
        root / "functional_cp_loto_per_task.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model": model,
                "alpha": alpha,
                "eval_time": eval_time,
                "n_tasks": 10,
            }
            for model, alpha, eval_time in itertools.product(
                MODELS, ALPHAS, ALL_TIMES
            )
        ]
    ).to_csv(root / "functional_cp_loto_macro.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "cp_seed": cp_seed,
                "alpha": alpha,
                "eval_time": eval_time,
            }
            for model, cp_seed, alpha, eval_time in itertools.product(
                MODELS, CP_SEEDS, ALPHAS, ALL_TIMES
            )
        ]
    ).to_csv(root / "functional_cp_loto_cp_seed_sensitivity.csv", index=False)

    def point_rows(times: tuple[str, ...]) -> list[dict]:
        return [
            {
                "model": model,
                "eval_time": eval_time,
                "constraint_type": constraint_type,
                "constraint_value": constraint_value,
                "alpha": 0.05,
            }
            for model, eval_time, (constraint_type, constraint_value) in itertools.product(
                MODELS, times, CONSTRAINTS
            )
        ]

    pd.DataFrame(point_rows(ALL_TIMES)).to_csv(
        root / "functional_cp_loto_operating_points.csv", index=False
    )
    pd.DataFrame(result_rows(FIXED_TIMES, True)).to_csv(
        root / "functional_cp_fixed_horizon_raw.csv", index=False
    )
    pd.DataFrame(result_rows(FIXED_TIMES, False)).to_csv(
        root / "functional_cp_fixed_horizon_per_task.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model": model,
                "alpha": alpha,
                "eval_time": eval_time,
                "n_tasks": 10,
            }
            for model, alpha, eval_time in itertools.product(
                MODELS, ALPHAS, FIXED_TIMES
            )
        ]
    ).to_csv(root / "functional_cp_fixed_horizon_macro.csv", index=False)
    pd.DataFrame(point_rows(FIXED_TIMES)).to_csv(
        root / "functional_cp_fixed_horizon_operating_points.csv", index=False
    )


def test_primary_result_table_contract_accepts_exact_domains(tmp_path: Path) -> None:
    write_primary_tables(tmp_path)
    assert verify_primary_result_tables(tmp_path) == []


def test_primary_result_table_contract_rejects_missing_row(tmp_path: Path) -> None:
    write_primary_tables(tmp_path)
    path = tmp_path / "functional_cp_fixed_horizon_macro.csv"
    frame = pd.read_csv(path).iloc[:-1]
    frame.to_csv(path, index=False)

    failures = verify_primary_result_tables(tmp_path)
    assert any("expected 48 rows, found 47" in failure for failure in failures)


def test_published_dynamic_layer_tables_match_exact_contracts() -> None:
    result_root = REPO_ROOT / "docs" / "results_audit"
    assert verify_dynamic_layer_result_tables(result_root) == []
