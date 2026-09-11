#!/usr/bin/env bash
#
# Hyperparameter grid search for the LINEAR-PROBE and LAYER-MIX failure
# detectors on the ORIGINAL OpenVLA LIBERO rollouts. Intended for a RunPod
# (CUDA) instance -- these methods are the layer-analysis track of the SAFE
# replication, distinct from the lstm/mlp baselines.
#
#   linear_probe : per-timestep logistic regression on the LAST captured layer
#                  (the lowest-capacity, SAFE-comparable baseline).
#   layer_mix    : LSTM over a LEARNED softmax mix of ALL captured layers --
#                  the learned weights reveal which layers carry the signal,
#                  no powerset search required.
#
# Grid (override any axis via env):
#   token  in {first, last, mean}        (3)
#   lr     in {1e-4, 3e-4, 1e-3}         (3)
#   lambda in {1e-3, 1e-2, 1e-1, 1}      (4)
#   seed   in {0, 1, 2}                  (3)
#   model  in {linear_probe, layer_mix}  (2)
#   => 3*3*4*3 = 108 cells per model, 216 total.
#
# EXECUTION: one python process per (model, token) -- it sweeps all lr x lambda
#   x seed in a single run, so the 94 GB of rollouts is read & cached ONCE per
#   token and reused across its 36 cells, instead of re-reading per cell.
#
# MEMORY: the dataset caches every SELECTED layer for all 500 rollouts in RAM,
#   in bf16. linear_probe (1 layer) ~ 2 GB; layer_mix (all 9 layers) ~ 17 GB.
#   On a RAM-capped host (e.g. a ~47 GiB RunPod container) layer_mix fits with
#   headroom. If still tight: NO_CACHE=1 (slow: re-reads pkls every epoch) or
#   shrink layer_mix's footprint with LAYERS (see below).
#
# Resumable per (model, token): a token whose 36 checkpoints all exist is
#   skipped; an interrupted token re-runs its cells (cheap -- cache is warm).
#
# RunPod setup (once):
#   uv sync --frozen
#   wandb login                       # or:  export WANDB_MODE=offline
#   # rollouts at data/rollouts/openvla (override with ROOT=...)
#
# Usage:
#   bash scripts/train_openvla_layers_grid.sh                      # full grid / resume
#   MODELS="linear_probe" bash scripts/train_openvla_layers_grid.sh # one method
#   LAYERS="24,28,32" bash scripts/train_openvla_layers_grid.sh     # layer_mix over late layers only
#   NO_CACHE=1 bash scripts/train_openvla_layers_grid.sh            # low-RAM host
#   DRY=1 bash scripts/train_openvla_layers_grid.sh                 # preview cells
#   EPOCHS=50 SEEDS="0" bash scripts/train_openvla_layers_grid.sh   # quick screen

set -uo pipefail

# --- repo root (this script lives in scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- python: prefer local venv, else system python (RunPod) ---
if [[ -n "${PY:-}" ]]; then :; elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python; fi

# --- config (override via env) ---
ROOT="${ROOT:-data/rollouts/openvla}"
DEVICE="${DEVICE:-cuda}"
PROJECT="${PROJECT:-safe-replicate-openvla-layers}"
EPOCHS="${EPOCHS:-200}"
HIDDEN="${HIDDEN:-256}"
MODELS="${MODELS:-linear_probe layer_mix}"
TOKENS="${TOKENS:-first last mean}"
LRS="${LRS:-1e-4 3e-4 1e-3}"
LAMBDAS="${LAMBDAS:-1e-3 1e-2 1e-1 1}"
SEEDS="${SEEDS:-0 1 2}"
LAYERS="${LAYERS:-}"          # optional comma list; only meaningful for layer_mix
NO_CACHE="${NO_CACHE:-0}"
FORCE="${FORCE:-0}"
DRY="${DRY:-0}"

outdir_for() { echo "runs/openvla_layers/$1"; }

# format a number the way the python script tags run names ({x:g})
gfmt() { "${PY}" -c "import sys; print(format(float(sys.argv[1]), 'g'))" "$1"; }

# token tag (normalize_token_pool then '_'->'-')
toktag_for() { case "$1" in first|0|0.0) echo first;; last|1|1.0) echo last;;
               mean) echo mean;; *) echo "$1";; esac; }

# experiment name as built by build_safe_openvla_libero_experiments()
# (model_tag == model name for linear_probe / layer_mix; n_history=1 => no tag)
expname_for() {
  local model="$1" tok="$2" lr="$3" reg="$4" seed="$5"
  echo "${model}_tok-$(toktag_for "${tok}")_lr-$(gfmt "${lr}")_reg-$(gfmt "${reg}")_seed-${seed}"
}

# --- count ---
total=0
for m in ${MODELS}; do for t in ${TOKENS}; do for lr in ${LRS}; do
  for reg in ${LAMBDAS}; do for s in ${SEEDS}; do total=$((total+1)); done; done; done; done; done

echo "=== OpenVLA LIBERO: linear_probe + layer_mix hyperparameter grid ==="
echo "models=[${MODELS}] tokens=[${TOKENS}] lrs=[${LRS}] lambdas=[${LAMBDAS}] seeds=[${SEEDS}]"
echo "epochs=${EPOCHS} hidden=${HIDDEN} layers_override=[${LAYERS:-<defaults: probe=last, mix=all-captured>}]"
echo "device=${DEVICE} project=${PROJECT} root=${ROOT} python=${PY} no_cache=${NO_CACHE}"
echo "total cells=${total}  (FORCE=${FORCE} DRY=${DRY})"
echo

cache_flag=(); [[ "${NO_CACHE}" == "1" ]] && cache_flag=(--no-cache)
layers_flag=(); [[ -n "${LAYERS}" ]] && layers_flag=(--sweep-layers "${LAYERS}")

# Join a space-separated list into a comma list for the python sweep flags.
csv() { local IFS=,; echo "$*"; }

# One python PROCESS per (model, token): it sweeps all lr x lambda x seed in a
# single run, so the 94 GB of rollouts is read/cached ONCE per token and reused
# across all its cells -- instead of re-reading per cell. Resume granularity is
# therefore per (model, token): a token whose checkpoints are all present is
# skipped; an interrupted token re-runs its cells (cheap, cache is warm).
cells_per_group=0
for lr in ${LRS}; do for reg in ${LAMBDAS}; do for s in ${SEEDS}; do
  cells_per_group=$((cells_per_group+1)); done; done; done
n_groups=0
for m in ${MODELS}; do for t in ${TOKENS}; do n_groups=$((n_groups+1)); done; done
echo "running ${n_groups} per-token processes x ${cells_per_group} cells = ${total} cells"
echo "(one rollout read per (model,token); cache reused across its cells)"
echo

failed=(); skipped=(); ran=(); gidx=0

for model in ${MODELS}; do
  outdir="$(outdir_for "${model}")"
  mkdir -p "${outdir}"
  group="${model}-openvla-layers-${EPOCHS}ep"
  for tok in ${TOKENS}; do
    gidx=$((gidx+1))
    toktag="$(toktag_for "${tok}")"
    prefix="[group ${gidx}/${n_groups}]"
    have=$(find "${outdir}" -maxdepth 1 -name "${model}_tok-${toktag}_*.pt" 2>/dev/null | wc -l | tr -d ' ')

    if [[ "${FORCE}" != "1" && "${have}" -ge "${cells_per_group}" ]]; then
      echo ">>> ${prefix} SKIP ${model}/${tok} (${have}/${cells_per_group} done)"; skipped+=("${model}/${tok}"); continue
    fi
    if [[ "${DRY}" == "1" ]]; then
      echo ">>> ${prefix} DRY ${model}/${tok}: ${cells_per_group} cells (have ${have})"; continue
    fi

    echo ">>> ${prefix} RUN ${model}/${tok}: ${cells_per_group} cells (have ${have})  ->  ${outdir}"
    log="${outdir}/train_${model}_tok-${toktag}.log"

    "${PY}" scripts/train_openvla_ablation.py \
      --root "${ROOT}" \
      --safe-openvla-libero-sweep \
      --safe-sweep-epochs "${EPOCHS}" \
      --no-early-stopping \
      --sweep-seeds "$(csv ${SEEDS})" \
      --sweep-token-pools "${tok}" \
      --sweep-lrs "$(csv ${LRS})" \
      --sweep-lambda-reg "$(csv ${LAMBDAS})" \
      --sweep-models "${model}" \
      --hidden-dim "${HIDDEN}" \
      --device "${DEVICE}" \
      --output-dir "${outdir}" \
      ${cache_flag[@]+"${cache_flag[@]}"} ${layers_flag[@]+"${layers_flag[@]}"} \
      --wandb --wandb-project "${PROJECT}" \
      --wandb-group "${group}" \
      2>&1 | tee "${log}"

    status="${PIPESTATUS[0]}"
    now=$(find "${outdir}" -maxdepth 1 -name "${model}_tok-${toktag}_*.pt" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${status}" -ne 0 ]]; then
      echo "!!! ${prefix} FAILED (exit ${status}): ${model}/${tok} (${now}/${cells_per_group}) -- re-run to resume" >&2; failed+=("${model}/${tok}")
    elif [[ "${now}" -ge "${cells_per_group}" ]]; then ran+=("${model}/${tok}")
    else echo "!!! ${prefix} PARTIAL: ${model}/${tok} (${now}/${cells_per_group}) -- re-run to finish" >&2; failed+=("${model}/${tok}"); fi
  done
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
