#!/usr/bin/env bash
#
# layer_mix on the LAST captured layer ONLY -- a cheap baseline / sanity check
# for the full all-9-layer layer_mix grid.
#
# Why this exists:
#   The full layer_mix caches all 9 captured layers for 500 rollouts (~17 GB
#   bf16) and is ~30 s/epoch (36864-dim features). Restricting it to one layer
#   makes the cache ~2 GB (like linear_probe) and training fast, while keeping
#   the SAME architecture (per-layer projection 4096->256, LSTM, head). With one
#   layer the learned softmax mix is degenerate (weight = 1.0), so this isolates
#   "is the layer_mix architecture itself any good on the last layer?" from "do
#   the lower layers add signal?" -- compare its test_early to the full grid.
#
#   This is also the apples-to-apples capacity comparison for linear_probe:
#   same last-layer input, but LSTM+projection instead of logistic regression.
#
# It is just the layer-restricted grid driver underneath; the only difference
# from train_openvla_layers_grid.sh is MODELS=layer_mix and LAYERS=<last layer>.
#
# The last captured layer is auto-detected from the rollouts (so it works
# whether the final layer is 32, 24, ...). Override with LAYERS=... if needed.
#
# Usage:
#   bash scripts/train_openvla_layermix_lastlayer.sh                 # full lr x lambda x token x seed grid
#   DRY=1 bash scripts/train_openvla_layermix_lastlayer.sh           # preview cells
#   EPOCHS=2 TOKENS=last LRS=1e-4 LAMBDAS=1e-2 SEEDS=0 \
#     bash scripts/train_openvla_layermix_lastlayer.sh               # 1-cell smoke
#   LAYERS=24 bash scripts/train_openvla_layermix_lastlayer.sh       # pin a different single layer

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PY:-}" ]]; then :; elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python; fi

ROOT="${ROOT:-data/rollouts/openvla}"

# Auto-detect the last captured layer unless the caller pinned LAYERS.
if [[ -z "${LAYERS:-}" ]]; then
  LAYERS="$(
    "${PY}" - "${ROOT}" <<'PY'
import sys
from pathlib import Path
from data.dataloaders import OpenVLARolloutDataset
ds = OpenVLARolloutDataset(Path(sys.argv[1]), recursive=True, cache=False)
print(tuple(ds.saved_layers)[-1])
PY
  )" || { echo "could not auto-detect last layer from ${ROOT}; set LAYERS=..." >&2; exit 1; }
  echo "auto-detected last captured layer: ${LAYERS}"
fi

# Runs the python sweep directly (mirroring the per-token batching of the shared
# grid driver) into a SEPARATE output dir so these one-layer checkpoints don't
# collide with the all-9-layer grid in runs/openvla_layers/layer_mix.
PROJECT="${PROJECT:-safe-replicate-openvla-layermix-lastlayer}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
HIDDEN="${HIDDEN:-256}"
TOKENS="${TOKENS:-first last mean}"
LRS="${LRS:-1e-4 3e-4 1e-3}"
LAMBDAS="${LAMBDAS:-1e-3 1e-2 1e-1 1}"
SEEDS="${SEEDS:-0 1 2}"
NO_CACHE="${NO_CACHE:-0}"
FORCE="${FORCE:-0}"
DRY="${DRY:-0}"

outdir="runs/openvla_layers/layer_mix_last${LAYERS}"
mkdir -p "${outdir}"
group="layer_mix-last${LAYERS}-openvla-${EPOCHS}ep"

cache_flag=(); [[ "${NO_CACHE}" == "1" ]] && cache_flag=(--no-cache)
csv() { local IFS=,; echo "$*"; }

cells_per_group=0
for lr in ${LRS}; do for reg in ${LAMBDAS}; do for s in ${SEEDS}; do
  cells_per_group=$((cells_per_group+1)); done; done; done
n_tok=0; for t in ${TOKENS}; do n_tok=$((n_tok+1)); done
total=$((cells_per_group*n_tok))

echo "=== layer_mix (LAST LAYER ${LAYERS} only) hyperparameter grid ==="
echo "tokens=[${TOKENS}] lrs=[${LRS}] lambdas=[${LAMBDAS}] seeds=[${SEEDS}]"
echo "epochs=${EPOCHS} hidden=${HIDDEN} device=${DEVICE} root=${ROOT} no_cache=${NO_CACHE}"
echo "one process per token x ${cells_per_group} cells = ${total} cells -> ${outdir}"
echo

failed=(); skipped=(); ran=(); gidx=0
toktag_for() { case "$1" in first|0|0.0) echo first;; last|1|1.0) echo last;; mean) echo mean;; *) echo "$1";; esac; }

for tok in ${TOKENS}; do
  gidx=$((gidx+1))
  toktag="$(toktag_for "${tok}")"
  prefix="[token ${gidx}/${n_tok}]"
  have=$(find "${outdir}" -maxdepth 1 -name "layer_mix_tok-${toktag}_*.pt" 2>/dev/null | wc -l | tr -d ' ')

  if [[ "${FORCE}" != "1" && "${have}" -ge "${cells_per_group}" ]]; then
    echo ">>> ${prefix} SKIP layer_mix/${tok} (${have}/${cells_per_group} done)"; skipped+=("${tok}"); continue
  fi
  if [[ "${DRY}" == "1" ]]; then
    echo ">>> ${prefix} DRY layer_mix/${tok}: ${cells_per_group} cells (have ${have})"; continue
  fi

  echo ">>> ${prefix} RUN layer_mix/${tok}: ${cells_per_group} cells (have ${have})  ->  ${outdir}"
  log="${outdir}/train_layer_mix_tok-${toktag}.log"

  "${PY}" scripts/train_openvla_ablation.py \
    --root "${ROOT}" \
    --safe-openvla-libero-sweep \
    --safe-sweep-epochs "${EPOCHS}" \
    --no-early-stopping \
    --sweep-seeds "$(csv ${SEEDS})" \
    --sweep-token-pools "${tok}" \
    --sweep-lrs "$(csv ${LRS})" \
    --sweep-lambda-reg "$(csv ${LAMBDAS})" \
    --sweep-models "layer_mix" \
    --sweep-layers "${LAYERS}" \
    --hidden-dim "${HIDDEN}" \
    --device "${DEVICE}" \
    --output-dir "${outdir}" \
    ${cache_flag[@]+"${cache_flag[@]}"} \
    --wandb --wandb-project "${PROJECT}" \
    --wandb-group "${group}" \
    2>&1 | tee "${log}"

  status="${PIPESTATUS[0]}"
  now=$(find "${outdir}" -maxdepth 1 -name "layer_mix_tok-${toktag}_*.pt" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${status}" -ne 0 ]]; then
    echo "!!! ${prefix} FAILED (exit ${status}): layer_mix/${tok} (${now}/${cells_per_group}) -- re-run to resume" >&2; failed+=("${tok}")
  elif [[ "${now}" -ge "${cells_per_group}" ]]; then ran+=("${tok}")
  else echo "!!! ${prefix} PARTIAL: layer_mix/${tok} (${now}/${cells_per_group}) -- re-run to finish" >&2; failed+=("${tok}"); fi
done

echo
echo "================ summary ================"
echo "ran:     ${#ran[@]}"
echo "skipped: ${#skipped[@]}"
echo "failed:  ${#failed[@]}"
if [[ "${#failed[@]}" -gt 0 ]]; then
  printf '  failed -> %s\n' "${failed[@]}" >&2
  echo "Some cells failed; re-run the script to resume them." >&2
  exit 1
fi
echo "All requested cells complete. Aggregate with:"
echo "  ${PY} scripts/summarize_mini_grid.py --root runs/openvla_layers --csv runs/openvla_layers/summary.csv"
