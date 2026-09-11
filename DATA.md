# Data and checkpoint contract

The large rollout hidden states and trained checkpoints are not committed to
this repository. Once generated, the tracked score bundle is the supported
lightweight route for independently replaying the reported metrics.

## Primary 1,000-rollout audit

Full inference expects two ordered roots:

```text
data/rollouts/openVLA-last-layer/
data/rollouts/openVLA-last-layer-2/
```

Each root contains 500 pickle files named
`task<TASK>--ep<EPISODE>--succ<0|1>.pkl`. Together they must contain exactly 100
rollouts for each task 0 through 9. Root order is part of the checkpoint split
contract and must not be changed.

Each artifact is a dictionary with:

- `hidden_states`: `(steps, action_tokens, layers * hidden_dim)` for the first
  historical schema, or `(steps, layers * hidden_dim)` for an artifact already
  pooled to the last action token;
- `hidden_state_layers`: saved model-layer identifiers including layer 32;
- `hidden_state_dim_per_layer`: 4096;
- `hidden_state_token_index: -1` for the pre-pooled schema;
- task, episode, and success metadata, either in the dictionary or filename.

### Low-space staged preparation

The finalized probes need only the full-rate layer-32 last-token trajectory.
When both full source shards cannot coexist on disk, each 500-rollout shard can
be restored, normalized, verified, and removed from staging before restoring
the next one. The preparation script accepts both historical source schemas and
does not change tensor values or temporal resolution:

```bash
uv run python scripts/prepare_openvla_layer_subset.py \
  --input-root data/staging/openVLA-last-layer \
  --output-root data/processed/primary_l32/openVLA-last-layer \
  --layers 32 --token-index -1 --temporal-stride 1 \
  --expected-rollouts 500 --expected-tasks 0,1,2,3,4,5,6,7,8,9 \
  --expected-rollouts-per-task 50 --minimum-length 148

uv run python scripts/prepare_openvla_layer_subset.py \
  --input-root data/staging/openVLA-last-layer-2 \
  --output-root data/processed/primary_l32/openVLA-last-layer-2 \
  --layers 32 --token-index -1 --temporal-stride 1 \
  --expected-rollouts 500 --expected-tasks 0,1,2,3,4,5,6,7,8,9 \
  --expected-rollouts-per-task 50 --minimum-length 148

uv run python scripts/verify_primary_prepared_data.py \
  --root data/processed/primary_l32/openVLA-last-layer \
         data/processed/primary_l32/openVLA-last-layer-2 \
  --output runs/results_audit/primary_prepared_data_audit.json
```

Only remove a staging shard after its output manifest reports 500 rollouts,
50 per task, a homogeneous `[32]` schema, temporal stride 1, and minimum length
at least 148. Preserve both manifests for release provenance. For inference,
substitute the two `data/processed/primary_l32/...` directories for the raw
`--root` paths below, in the same order. The stride-4 500-rollout multilayer set
is not a valid substitute.

The finalized checkpoints are expected under:

```text
runs/openvla_mlp_loto_l32/
runs/openvla_lstm_loto_l32/
```

Each directory must contain folds 0 through 9, with the exact split indices
embedded in each checkpoint. The score-bundle manifest records SHA-256 hashes
for all 20 checkpoints.

Generate the public replay bundle and all conformal tables with:

```bash
uv run python scripts/verify_primary_checkpoints.py

uv run python scripts/evaluate_functional_cp_loto.py \
  --root data/rollouts/openVLA-last-layer data/rollouts/openVLA-last-layer-2 \
  --mlp-checkpoint-dir runs/openvla_mlp_loto_l32 \
  --lstm-checkpoint-dir runs/openvla_lstm_loto_l32 \
  --score-bundle-out docs/results_audit/score_bundle \
  --output-dir runs/results_audit/functional_cp_loto_closeout \
  --alphas 0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3 \
  --cp-seeds 0,1,2,3,4,5,6,7,8,9 \
  --horizon 520 --fixed-horizons 50,100,148 \
  --task-bootstrap 10000 --bootstrap-seed 20260716
```

After reviewing the generated tables, publish the eight replay-checked result
tables named by `scripts/compare_functional_cp_outputs.py` into
`docs/results_audit/`. Then replay from the bundle into a separate directory and
compare:

```bash
uv run python scripts/evaluate_functional_cp_loto.py \
  --score-bundle-in docs/results_audit/score_bundle \
  --output-dir reproduced/functional_cp_loto \
  --alphas 0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3 \
  --cp-seeds 0,1,2,3,4,5,6,7,8,9 \
  --horizon 520 --fixed-horizons 50,100,148 \
  --task-bootstrap 10000 --bootstrap-seed 20260716 --device cpu

uv run python scripts/compare_functional_cp_outputs.py \
  --expected-root docs/results_audit \
  --actual-root reproduced/functional_cp_loto
```

The fixed horizons were declared before this closeout rerun. The longest, 148,
is the observed global minimum rollout length. Every rollout is therefore
observed through each horizon without padding or exclusion; timesteps after the
selected common observation window are intentionally ignored.

## Multilayer follow-up

The published layer-20/32 time-resolved experiment uses two ordered,
homogeneous 500-rollout shards: 1,000 files total, 100 for each task, layers
20/32, the last action token, and temporal stride 4. The compact shards live at:

```text
data/processed/openVLA_500_l20_l32_last_s4/
data/processed/openVLA_500_2_l20_l32_last_s4/
```

Their portable schema and inventory records are tracked as
`docs/results_audit/dynamic_layer_data_manifest_1000_shard0.json` and
`docs/results_audit/dynamic_layer_data_manifest_1000_shard1.json`. The training
root preserves this order with hard-linked `shard0/` and `shard1/` directories
under `data/processed/openVLA_1000_l20_l32_last_s4`; hard links avoid another
copy of the tensors.

The first shard includes task 3 normalized from 50 restored full-layer
artifacts. Its nested prior manifest records the earlier 450-file boundary, in
which task 3 had one truncated file and 49 missing files. The second source
shard, `data/processed/OpenVLA_500_2`, contains nine already-last-token layers
at full temporal resolution; `scripts/prepare_openvla_layer_subset.py` selected
layers 20/32 and temporal stride 4 without token interpolation.

The derived directories under `data/processed/` do not substitute for the two
primary full-rate layer-32 inputs. The finalized primary checkpoints embed
split indices for 1,000 full-rate rollouts, and the MLP accumulates per-step
increments while the LSTM is sequence dependent. Upsampling the stride-4
multilayer set would therefore change detector outputs and cannot recover the
predeclared 50/100/148-step primary analysis.

The former local `openVLA_500_l24_l28_l32_last_s4` directory used for the
original 450-rollout, nine-task late-layer experiment is not present in this
checkout. The second shard has been preserved as the homogeneous ten-task
replication set `data/processed/openVLA_500_2_l24_l28_l32_last_s4`, with its
portable manifest tracked under `docs/results_audit/`. Combining the two into a
single 1,000-rollout late-layer analysis still requires restoring layers 24/28
for the first shard.

## Distribution and licensing

This repository does not grant rights to redistribute OpenVLA weights, LIBERO
assets, or rollout artifacts derived from them. Obtain those inputs under their
upstream terms. The compact score bundle contains detector outputs and metadata,
not raw observations, actions, hidden states, or model weights.
