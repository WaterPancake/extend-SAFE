# Public score-bundle format

`docs/results_audit/score_bundle/` is the schema-v2 lightweight reproducibility
artifact for the primary 1,000-rollout evaluation.

It contains:

- `records.csv`: one validation or held-out-test record per model/fold/rollout;
- `trajectories.npz`: concatenated float32 trajectories plus int64 offsets;
- `manifest.json`: schema version, counts, source inventory digest, generator
  hashes, generation command, and SHA-256 hashes for bundle files and source
  checkpoints.

When inference uses the low-space prepared roots, the manifest also embeds both
portable preparation manifests and their file/canonical hashes. Direct use of
the original roots leaves that mapping empty. In either route, the scored source
inventory and root aliases are mandatory.

For this release, the exact contract is 9,200 rows: two model families, ten
folds per family, and 460 retained records per fold (360 validation plus 100
held-out test). The held-out rows cover each of dataset indices 0--999 exactly
once per model. The bundle has 1,000 unique rollout identities even though seen
validation rollouts recur across folds.

The NumPy archive contains numeric arrays only and is loaded with
`allow_pickle=False`. Source paths are rewritten as root aliases plus relative
paths, so the bundle does not expose machine-local directories.

The bundle intentionally omits probe-training episodes that are not in a fold's
validation split. Functional conformal replay needs only successful seen-task
validation trajectories and held-out-task test trajectories. Across all ten
folds, each of the 1,000 rollouts still appears as held-out test data once.

Run the integrity verifier directly with:

```bash
uv run python -c "from pathlib import Path; from scripts.score_bundle import verify_score_bundle; verify_score_bundle(Path('docs/results_audit/score_bundle'))"
```

Any checksum mismatch, unsafe path, missing task/outcome metadata, inconsistent
source inventory, incomplete fold/split/checkpoint coverage, non-finite score,
malformed offset, or trajectory-length disagreement is a hard error. The
publication verifier additionally requires both MLP and LSTM families, all ten
LOTO folds, identical family split mappings, fold-*k* tests restricted to task
*k*, fixed-horizon eligibility through step 148, and the exact row/rollout
counts above.

CI replays the conformal analysis from the bundle and compares the published
and replayed per-task, macro, CP-seed-sensitivity, operating-point, and
fixed-horizon tables, including the fixed-horizon raw metric rows, with:

```bash
uv run python scripts/compare_functional_cp_outputs.py \
  --expected-root docs/results_audit \
  --actual-root reproduced/functional_cp_loto
```
