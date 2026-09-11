from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.verify_primary_prepared_data as verifier


def valid_audit() -> dict:
    return {
        "rollouts": 500,
        "task_counts": {str(task): 50 for task in range(10)},
        "episode_coverage": {
            str(task): {"minimum": 0, "maximum": 49, "unique_count": 50}
            for task in range(10)
        },
        "outcome_counts": {},
        "minimum_length": 148,
        "maximum_length": 520,
        "homogeneous_schema": True,
        "source_action_token_index_counts": {"-1": 500},
        "schemas": [
            {
                "count": 500,
                "layers": [32],
                "hidden_dim_per_layer": 4096,
                "hidden_ndim": 3,
                "hidden_shape_suffix": [1, 4096],
                "temporal_stride": 1,
                "hidden_state_layout": "action_token_axis_then_concatenated_selected_layers",
            }
        ],
        "inventory_sha256": "a" * 64,
    }


def test_prepared_primary_verifier_accepts_two_ordered_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [
        tmp_path / "openVLA-last-layer",
        tmp_path / "openVLA-last-layer-2",
    ]
    audit = valid_audit()
    for root in roots:
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps({"corrupt_source_artifacts": [], "output_audit": audit})
        )
    monkeypatch.setattr(verifier, "audit_output", lambda root: audit)

    result = verifier.verify_prepared_roots(roots)

    assert result["status"] == "pass"
    assert result["source_root_aliases"] == {
        "root-0": "openVLA-last-layer",
        "root-1": "openVLA-last-layer-2",
    }


def test_prepared_primary_verifier_rejects_reversed_names(tmp_path: Path) -> None:
    roots = [tmp_path / "openVLA-last-layer-2", tmp_path / "openVLA-last-layer"]
    with pytest.raises(ValueError, match="root names"):
        verifier.verify_prepared_roots(roots)
