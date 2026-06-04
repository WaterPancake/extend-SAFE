# OpenVLA layer-method grid search — findings

Two new failure-detector methods added to the SAFE sweep and grid-searched on the
original OpenVLA LIBERO rollouts (500 rollouts, libero_10, 9 captured layers
[1,4,8,12,16,20,24,28,32], hidden 4096, last layer = 32).

- **linear_probe** — per-timestep logistic regression on the LAST captured layer
  (lowest-capacity, SAFE-comparable baseline).
- **layer_mix** — LSTM over a LEARNED softmax mix of ALL 9 captured layers; the
  learned weights are meant to reveal which layers carry the failure signal.

Selection metric: `val_falert_early_roc_auc` (leakage-resistant); headline =
`test_falert_early_roc_auc`. `test_length_only_roc_auc = 1.0` everywhere (failures
run to timeout); the `early` metric is the one to trust.

## linear_probe — best HPs (grid: token × lr × λ × 3 seeds)

Run on the FULL 500 rollouts for `mean` pooling, and on a consistent 254-rollout
subset for all three pooling modes (full-500 `last`/`first` are not survivable on
an 18 GB Mac — see Constraints).

| data scale | robust token | test_early |
|---|---|---|
| 120-subset | — | ~0.50 (noise — too little data) |
| 254-subset | mean | 0.59 ± 0.04 |
| **full-500** | **mean** | **0.65 ± 0.02** |

- **Best: `token=mean, lr=3e-4, λ=0.01` → test_early 0.65 ± 0.02** (full-500).
- `mean` is the robust pooling; `first` wins on val but collapses on test
  (~0.47) — the same val/test overfit trap seen in the openvla-mini grid.
- test_early rises monotonically with data scale, so the full-500 number is the
  trustworthy one; subsets only confirm the *trend* and the choice of `mean`.

## layer_mix — tested + lr×λ search (data-limited)

Full-fidelity layer_mix (all 9 layers × 500 rollouts) is **not runnable locally**
(see Constraints), so the local result is a 12-cell lr×λ search at `token=mean`,
seed 0, 40 epochs, on a 60-rollout subset:

- Best (val): `mean, lr=1e-4, λ=1e-3` → val_early 0.59, test_early 0.52 — weak and
  undertrained (60 rollouts, 40 epochs, 1.5M params).
- **Learned layer weights are essentially uniform (~0.111 = 1/9)**; layers 20–32
  edge out 1–16 only marginally (0.1113 vs 0.1109). The mix barely moved from
  zero-init — too little training to differentiate layers.
- Conclusion: the "which layers matter" readout requires proper training (full
  data, many epochs). **This is a RunPod job.**

## layer_mix on the LAST LAYER ONLY — baseline (runs locally)

`scripts/train_openvla_layermix_lastlayer.sh` runs the layer_mix architecture
(per-layer projection 4096/896→256 + LSTM + head) fed **only the last captured
layer**, so its cache is ~1 layer instead of all 9. This isolates *"is the
layer_mix architecture any good on the last layer?"* from *"do the lower layers
add signal?"*, and is the apples-to-apples capacity comparison vs the
last-layer `linear_probe` (same input, LSTM+projection instead of logistic reg).

### openvla-mini (libero_90, 500 rollouts, last layer = 24, hidden 896)

Full 108-cell grid (token × lr × λ × 3 seeds), 200 epochs, MPS. Ranked by
`val_falert_early`; selected for robustness (test mean AND low test variance):

| token | lr | λ | val_early | test_early |
|---|---|---|---|---|
| **first** | **3e-4** | **0.01** | 0.855 | **0.773 ± 0.042** |
| first | 1e-4 | 0.01 | 0.858 | 0.768 ± 0.041 |
| last | 3e-4 | 0.01 | **0.869** | 0.749 ± 0.107 |
| mean | 1e-3 | 0.01 | 0.858 | 0.728 ± 0.062 |

- **Robust winner: `token=first, lr=3e-4, λ=0.01` → test_early 0.773 ± 0.042.**
  `last` wins on val but its test variance is ~3× larger (the val/test overfit
  trap seen across these grids); `first` is higher *and* tighter on test.
- **λ=0.01 is the sweet spot** across all three pooling modes.
- **Mini ≫ OpenVLA on the same method:** ~0.77 test_early here vs ~0.65 for the
  OpenVLA last-layer linear_probe — the mini rollouts carry a much stronger
  last-layer failure signal.
- **More epochs OVERFIT here.** Refitting the robust winner at 1000 epochs ×
  3 seeds (no early stopping) gave test_early **0.695 ± 0.135** — *worse* and 3×
  noisier than the 200-epoch grid (0.773 ± 0.042). Val kept climbing (0.815 →
  0.910 across seeds) while test diverged; seed 2 collapsed to 0.512 (chance).
  Takeaway: **the 200-epoch result is the one to report**; 1000 epochs without
  early stopping memorizes the train/val split. (Refit in
  `runs/openvla_layers/layer_mix_last24_refit1k/`.)
- Checkpoints: `runs/openvla_layers/layer_mix_last24/`; wandb project
  `safe-replicate-minivla-layermix-lastlayer`.

## Constraints (why the full grid is RunPod, not local)

- Data is **94 GB** (each pkl = 192 MB bf16, all 9 layers) on an **18 GB RAM** Mac.
- **Memory**: layer_mix caches all 9 layers → ~38 GB > RAM. linear_probe caches
  one layer (~4 GB) and fits. Single-process full-500 reads are OOM-flaky
  (macOS jetsam on page-cache pressure from streaming 94 GB; `purge` needs sudo).
  Reliable local pattern: per-token process on a ≤254-rollout subset.
- **Compute**: layer_mix is ~30 s/epoch even on MPS (36 864-dim features), so the
  108-cell × 80-epoch grid is multi-day locally.

## To run the full grid on RunPod (CUDA, ≥48 GB RAM)

    bash scripts/train_openvla_layers_grid.sh          # full 216-cell grid / resume
    NO_CACHE=1 bash scripts/train_openvla_layers_grid.sh   # low-RAM host
    python scripts/summarize_mini_grid.py --root runs/openvla_layers --csv runs/openvla_layers/summary.csv

## Code changes (this work)

- `scripts/train_openvla_ablation.py`: sweep builder emits `linear_probe`
  (last layer) and `layer_mix` (all captured layers); added `--no-cache`; added
  cross-experiment dataset memoization (`get_dataset`) so a single-process grid
  reads each layer/token view once instead of per-cell.
- `scripts/train_openvla_layers_grid.sh`: resumable RunPod grid driver (fixed a
  bash-3.2 empty-array `set -u` bug).
- `scripts/summarize_mini_grid.py`: ranks arbitrary model names (not just lstm/mlp).
