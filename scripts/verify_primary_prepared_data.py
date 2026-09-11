"""Verify staged full-rate layer-32 roots before primary checkpoint inference."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_openvla_layer_subset import (
    audit_output,
    portable_path,
    sha256_file,
    validate_output_audit,
)


EXPECTED_ROOT_NAMES = ("openVLA-last-layer", "openVLA-last-layer-2")


def verify_prepared_roots(roots: list[Path]) -> dict[str, Any]:
    if len(roots) != 2:
        raise ValueError(f"expected two ordered roots, found {len(roots)}")
    if tuple(root.name for root in roots) != EXPECTED_ROOT_NAMES:
        raise ValueError(
            f"root names must be {EXPECTED_ROOT_NAMES}, found "
            f"{tuple(root.name for root in roots)}"
        )

    root_results = {}
    for index, root in enumerate(roots):
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing preparation manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("corrupt_source_artifacts"):
            raise ValueError(f"{manifest_path}: corrupt source artifacts were skipped")
        audit = audit_output(root)
        validate_output_audit(
            audit,
            Namespace(
                expected_rollouts=500,
                expected_tasks=tuple(range(10)),
                expected_rollouts_per_task=50,
                minimum_length=148,
            ),
        )
        if manifest.get("output_audit") != audit:
            raise ValueError(f"{manifest_path}: stored output audit is stale")
        if len(audit["schemas"]) != 1:
            raise ValueError(f"{root}: expected one output schema")
        schema = audit["schemas"][0]
        expected_schema = {
            "layers": [32],
            "hidden_dim_per_layer": 4096,
            "hidden_ndim": 3,
            "hidden_shape_suffix": [1, 4096],
            "temporal_stride": 1,
            "hidden_state_layout": "action_token_axis_then_concatenated_selected_layers",
        }
        observed_schema = {key: schema.get(key) for key in expected_schema}
        if observed_schema != expected_schema:
            raise ValueError(
                f"{root}: schema {observed_schema} does not match {expected_schema}"
            )
        root_results[f"root-{index}"] = {
            "name": root.name,
            "path": portable_path(root),
            "preparation_manifest_sha256": sha256_file(manifest_path),
            "output_inventory_sha256": audit["inventory_sha256"],
            "rollouts": audit["rollouts"],
            "minimum_length": audit["minimum_length"],
        }
    return {
        "status": "pass",
        "source_root_aliases": {
            f"root-{index}": name for index, name in enumerate(EXPECTED_ROOT_NAMES)
        },
        "roots": root_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_prepared_roots(args.root)
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
