from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prepare_openvla_layer_subset import REPO_ROOT, portable_path
from scripts.summarize_dynamic_layer_seeds import paired_cp_row


def test_portable_path_uses_repository_relative_form() -> None:
    path = REPO_ROOT / "data" / "processed" / "example.pkl"
    assert portable_path(path) == "data/processed/example.pkl"


def test_paired_cp_row_compares_matched_tasks() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": model,
                "heldout_task": task,
                "eval_time": "fixed horizon 13",
                "alpha": 0.05,
                "realized_fpr_mean": fpr,
                "catch_rate_mean": catch,
                "effective_lead_fraction_mean": lead,
            }
            for model, task, fpr, catch, lead in (
                ("layer20_32", 0, 0.04, 0.20, 0.10),
                ("layer20_32", 1, 0.06, 0.30, 0.12),
                ("baseline32", 0, 0.03, 0.10, 0.08),
                ("baseline32", 1, 0.05, 0.10, 0.09),
            )
        ]
    )

    result = paired_cp_row(
        frame,
        left="layer20_32",
        right="baseline32",
        eval_time="fixed horizon 13",
        left_alpha=0.05,
        right_alpha=0.05,
        comparison_type="same nominal alpha",
        bootstrap_seed=7,
        n_bootstrap=100,
    )

    assert result["n_tasks"] == 2
    assert np.isclose(result["delta_realized_fpr"], 0.01)
    assert np.isclose(result["delta_catch_rate"], 0.15)
    assert result["delta_catch_rate_positive_tasks"] == 2
    assert np.isclose(result["delta_effective_lead_fraction"], 0.025)
