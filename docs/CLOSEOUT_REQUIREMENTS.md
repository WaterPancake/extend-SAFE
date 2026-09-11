# Closeout requirements

This document turns the five publication tasks into verifiable acceptance
criteria. The project is complete only when every item below is satisfied in a
clean checkout.

## Completion semantics

An item is **passed** only when its listed artifacts exist and its verification
command exits zero in a fresh clone. A missing prerequisite is **blocked**, not
waived; a development-only flag such as `--allow-missing-score-bundle` cannot be
used as release evidence. Requirement 1 is deliberately last: the publication
commit is made only after requirements 2--5 pass.

Current status (2026-08-03):

| Requirement | Status | Evidence or blocker |
|---|---|---|
| 1. Publication commit | Pending | Publication files are still uncommitted; requirements 2 and 3 are not complete. |
| 2. Fixed-horizon sensitivity | Blocked on input | The 20 finalized checkpoints pass `primary_checkpoint_audit.json`, but the full-rate first primary shard is absent. The new 1,000-rollout stride-4 multilayer union cannot reconstruct the finalized probes' 50/100/148-step scores. |
| 3. Compact score release | Blocked on input | It must be generated during primary checkpoint inference; scalar AUC, aggregate CP tables, and newly trained dynamic-monitor scores cannot reconstruct the finalized MLP/LSTM trajectories. |
| 4. Environment/license/citation | Locally verified | `uv sync --frozen --extra dev` and CFF validation passed on 2026-08-03; a clean-checkout run remains part of the final commit gate. |
| 5. Tests and CI | Locally verified except release-data gate | Compilation and 31 tests pass; the waived publication audit passes, while the strict audit fails only because requirement 3 is absent. |

The blocker does not require keeping both full shards simultaneously. `DATA.md`
defines a staged layer-32/last-token preparation route that is lossless for the
finalized probes, preserves full temporal resolution, and validates each
500-rollout shard before the source copy is removed.

## 1. Publication commit

- The Quarto source, freshly rendered GitHub Markdown, final analysis code, and
  every report-linked compact artifact are tracked.
- Generated raw data, checkpoints, and development runs remain ignored.
- All local Markdown links resolve and all published CSV/JSON files parse.
- `docs/results_audit/publication_manifest.json` is regenerated after every
  other publication edit and verifies every listed hash.
- `uv run python scripts/verify_publication.py` exits zero without a waiver.
- The final commit leaves no intended publication changes uncommitted;
  `git status --porcelain` is empty after the commit.

## 2. Fixed-horizon sensitivity

- The ordered primary roots contain exactly 1,000 readable rollouts: 500 per
  root, 100 per task, with dataset indices 0--999 matching the checkpoint split
  contract. Each finalized checkpoint family contains folds 0--9; every fold
  has 540 train, 360 validation, and 100 held-out-test indices, and MLP/LSTM
  split mappings are identical.
- Evaluate the finalized layer-32 MLP and LSTM checkpoints under all ten LOTO
  folds at common horizons 50, 100, and 148 steps. These values were selected
  before the rerun; every validation and test trajectory must have at least 148
  observed scores. The evaluator must fail instead of padding, extending, or
  excluding a trajectory at a fixed horizon.
- At each horizon, rerun SAFE functional conformal calibration for the existing
  alpha grid `0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3` and CP split seeds 0--9.
- Aggregate CP seeds within held-out task, then report equal-task macro means
  and task-bootstrap intervals.
- Publish fixed-horizon raw, per-task, macro, and operating-point tables. Their
  expected row counts are 4,800, 480, 48, and 36 respectively.
- Update the technical report with the fixed-horizon low-FPR results and state
  explicitly whether the primary conclusion changes.

The authoritative generation command is the raw-inference command in
`DATA.md`. Passing this item requires that its output manifest records all three
fixed horizons and all 20 checkpoint hashes.

The checkpoint-only half of the prerequisite is independently checked with:

```bash
uv run python scripts/verify_primary_checkpoints.py
```

## 3. Compact score release and provenance

- Publish validation and held-out-test score trajectories for every MLP/LSTM
  LOTO fold in a non-pickle bundle. The bundle must contain exactly 9,200
  model/fold/split records and 1,000 unique rollout identities. Every model/fold
  has 360 validation and 100 test records; held-out test indices cover 0--999
  exactly once per model, and fold *k* tests only task *k*.
- Include rollout metadata, split role, task/outcome labels, lengths, and
  `task_min_step`, but no machine-local absolute paths or raw hidden states.
- Include SHA-256 checksums for bundle files and source checkpoints, a source
  inventory digest, schema version, generation command, and generator hashes.
- The bundle checkpoint hashes and normalized split digest must match the
  tracked `primary_checkpoint_audit.json`; generator hashes must match the
  current inference, model, dataloader, and conformal source files.
- A verifier must reject modified files, invalid offsets, unsafe paths, and
  incomplete metadata, overlapping validation/test indices, inconsistent
  cross-fold rollout identities, or divergent MLP/LSTM split mappings.
- Replaying from the bundle must reproduce both the original functional-CP
  tables and the fixed-horizon tables without raw latents or checkpoints. CI
  performs the replay and compares eight published result tables with
  `scripts/compare_functional_cp_outputs.py`.

## 4. Environment, license, and citation

- Direct runtime dependencies live in `pyproject.toml`; exact transitive
  versions live in `uv.lock`.
- `uv sync --frozen --extra dev` creates the tested environment.
- The repository contains an explicit software license and a valid citation
  file. Third-party papers/code/data remain governed by their own licenses.

Acceptance commands:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev cffconvert --validate --infile CITATION.cff
```

## 5. Tests and continuous integration

- Synthetic tests cover rollout schema loading, masking, LOTO split isolation,
  functional-CP calibration, fixed-horizon no-extension behavior, score-bundle
  round trips/checksums, pooled-AUC decomposition, and multilayer controls.
- CI installs from the frozen lock, compiles the Python surface, runs all tests,
  verifies the published bundle and artifacts, checks local report links,
  replays the primary conformal analysis, and compares it with the tracked
  tables.
- The exact local release gate is:

```bash
uv run --frozen --extra dev python -m compileall -q data models scripts tests
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev python scripts/verify_publication.py
```
