from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.compare_functional_cp_outputs import REQUIRED_TABLES, compare_outputs


def write_tables(root: Path, value: float = 0.25) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_TABLES:
        pd.DataFrame(
            [{"model": "mlp", "eval_time": "fixed horizon 50", "value": value}]
        ).to_csv(root / name, index=False)


def test_compare_outputs_accepts_matching_tables(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    write_tables(expected)
    write_tables(actual, value=0.25 + 1e-10)

    compare_outputs(expected, actual)


def test_compare_outputs_rejects_changed_table(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    write_tables(expected)
    write_tables(actual)
    pd.DataFrame(
        [{"model": "mlp", "eval_time": "fixed horizon 50", "value": 0.5}]
    ).to_csv(actual / REQUIRED_TABLES[0], index=False)

    with pytest.raises(ValueError, match="does not match"):
        compare_outputs(expected, actual)
