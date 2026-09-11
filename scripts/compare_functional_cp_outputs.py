"""Compare published and replayed functional-CP result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_TABLES = (
    "functional_cp_loto_per_task.csv",
    "functional_cp_loto_macro.csv",
    "functional_cp_loto_cp_seed_sensitivity.csv",
    "functional_cp_loto_operating_points.csv",
    "functional_cp_fixed_horizon_raw.csv",
    "functional_cp_fixed_horizon_per_task.csv",
    "functional_cp_fixed_horizon_macro.csv",
    "functional_cp_fixed_horizon_operating_points.csv",
)


def compare_outputs(
    expected_root: Path,
    actual_root: Path,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    failures = []
    for name in REQUIRED_TABLES:
        expected_path = expected_root / name
        actual_path = actual_root / name
        missing = [str(path) for path in (expected_path, actual_path) if not path.exists()]
        if missing:
            failures.append(f"{name}: missing {', '.join(missing)}")
            continue
        expected = pd.read_csv(expected_path)
        actual = pd.read_csv(actual_path)
        try:
            pd.testing.assert_frame_equal(
                actual,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    if failures:
        raise ValueError(
            "functional-CP replay does not match published tables:\n- "
            + "\n- ".join(failures)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--actual-root", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    compare_outputs(
        args.expected_root,
        args.actual_root,
        rtol=args.rtol,
        atol=args.atol,
    )
    print("functional-CP replay matches published tables")


if __name__ == "__main__":
    main()
