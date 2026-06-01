#!/usr/bin/env bash
#
# Full SAFE-style hyperparameter GRID SEARCH for the openvla-mini backbone
# (a different LM than OpenVLA, so hyperparameters are re-tuned from scratch,
# matching the authors' sweep). Trains both LSTM and MLP on the last-layer
# feature and logs to the `safe-replicate-libero90-mini` W&B project.
#
# Grid (per model, per seed):
#   token pooling : first, last, mean
#   lambda_reg    : 1e-3, 1e-2, 1e-1, 1
#   learning rate : 1e-4, 3e-4, 1e-3
# => 3 * 4 * 3 = 36 configs/model/seed.
# Default models={lstm,mlp}, seeds={0,1,2} => 36 * 2 * 3 = 216 runs total.
#
# Resumable: each run writes a checkpoint
#   <output_dir>/<exp_name>.pt
# only after that experiment finishes training. On re-run, any experiment whose
# checkpoint already exists is skipped, so a crash / Ctrl-C is recovered simply
# by re-running this script. A crash *mid-experiment* re-runs just that one
# experiment from scratch (checkpoints are written end-of-training only).
#
# Usage:
#   bash scripts/train_mini_grid.sh                       # run/resume full grid (200 ep)
#   FORCE=1 bash scripts/train_mini_grid.sh               # ignore existing checkpoints
#   EPOCHS=1000 bash scripts/train_mini_grid.sh           # fully-converged pass
#   SEEDS="0" MODELS="lstm" bash scripts/train_mini_grid.sh    # subset
#   TOKENS="last" LAMBDAS="1e-2" LRS="3e-4" bash scripts/train_mini_grid.sh
#   DRY=1 bash scripts/train_mini_grid.sh                 # list runs, don't train

set -uo pipefail

# --- repo root (this script lives in scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- config (override via env) ---
PY="${PY:-.venv/bin/python}"
ROOT="${ROOT:-data/rollouts/openvla-mini}"
DEVICE="${DEVICE:-mps}"
PROJECT="${PROJECT:-safe-replicate-libero90-mini}"
EPOCHS="${EPOCHS:-200}"        # grid pass for ranking hyperparameters; refit winner at 1000
HIDDEN="${HIDDEN:-256}"

# grid axes (space-separated; override via env)
MODELS="${MODELS:-lstm mlp}"
SEEDS="${SEEDS:-0 1 2}"
TOKENS="${TOKENS:-first last mean}"
LAMBDAS="${LAMBDAS:-1e-3 1e-2 1e-1 1}"
LRS="${LRS:-1e-4 3e-4 1e-3}"

FORCE="${FORCE:-0}"
DRY="${DRY:-0}"

# format a numeric string the way the python script tags run names ({x:g})
gfmt() { "${PY}" -c "import sys; print(format(float(sys.argv[1]), 'g'))" "$1"; }

# model tag used in run names (safe_lstm / safe_mlp)
modeltag_for() {
  case "$1" in
    lstm) echo "safe_lstm" ;;
    mlp)  echo "safe_mlp"  ;;
    *) echo "unknown model: $1" >&2; return 1 ;;
  esac
}

# token tag (normalize_token_pool then '_'->'-'); first/last/mean map to themselves
toktag_for() {
  case "$1" in
    first|0|0.0) echo "first" ;;
    last|1|1.0)  echo "last"  ;;
    mean)        echo "mean"  ;;
    first_last|first+last|first-last) echo "first-last" ;;
    *) echo "$1" ;;
  esac
}

outdir_for() { echo "runs/mini_grid/$1"; }

# experiment name as built by build_safe_openvla_libero_experiments()
# (n_history_steps=1 => no history tag)
expname_for() {
  local model="$1" token="$2" lr="$3" lam="$4" seed="$5"
  echo "$(modeltag_for "${model}")_tok-$(toktag_for "${token}")_lr-$(gfmt "${lr}")_reg-$(gfmt "${lam}")_seed-${seed}"
}

# --- count grid size ---
total=0
for model in ${MODELS}; do for seed in ${SEEDS}; do
  for token in ${TOKENS}; do for lam in ${LAMBDAS}; do for lr in ${LRS}; do
    total=$((total + 1))
  done; done; done
done; done

echo "=== openvla-mini grid search ==="
echo "models=[${MODELS}] seeds=[${SEEDS}]"
echo "tokens=[${TOKENS}] lambdas=[${LAMBDAS}] lrs=[${LRS}]"
echo "epochs=${EPOCHS} hidden=${HIDDEN} device=${DEVICE} project=${PROJECT}"
echo "total runs=${total}  (FORCE=${FORCE} DRY=${DRY})"
echo

failed=(); skipped=(); ran=()
idx=0

for model in ${MODELS}; do
  outdir="$(outdir_for "${model}")"
  group="${model}-mini-grid-${EPOCHS}ep"
  mkdir -p "${outdir}"

  for seed in ${SEEDS}; do
    for token in ${TOKENS}; do
      for lam in ${LAMBDAS}; do
        for lr in ${LRS}; do
          idx=$((idx + 1))
          exp="$(expname_for "${model}" "${token}" "${lr}" "${lam}" "${seed}")"
          ckpt="${outdir}/${exp}.pt"
          prefix="[${idx}/${total}]"

          if [[ "${FORCE}" != "1" && -f "${ckpt}" ]]; then
            echo ">>> ${prefix} SKIP (exists): ${exp}"
            skipped+=("${exp}")
            continue
          fi
          if [[ "${DRY}" == "1" ]]; then
            echo ">>> ${prefix} DRY: ${exp}  ->  ${outdir}"
            continue
          fi

          echo ">>> ${prefix} RUN: ${exp}  ->  ${outdir}"
          log="${outdir}/train_${exp}.log"

          "${PY}" scripts/train_openvla_ablation.py \
            --root "${ROOT}" \
            --safe-openvla-libero-sweep \
            --safe-sweep-epochs "${EPOCHS}" \
            --no-early-stopping \
            --sweep-seeds "${seed}" \
            --sweep-token-pools "${token}" \
            --sweep-lrs "${lr}" \
            --sweep-lambda-reg "${lam}" \
            --sweep-models "${model}" \
            --hidden-dim "${HIDDEN}" \
            --device "${DEVICE}" \
            --output-dir "${outdir}" \
            --wandb --wandb-project "${PROJECT}" \
            --wandb-group "${group}" \
            2>&1 | tee "${log}"

          status="${PIPESTATUS[0]}"
          if [[ "${status}" -ne 0 ]]; then
            echo "!!! ${prefix} FAILED (exit ${status}): ${exp} — re-run script to retry" >&2
            failed+=("${exp}")
          elif [[ -f "${ckpt}" ]]; then
            ran+=("${exp}")
          else
            echo "!!! ${prefix} WARNING: exit 0 but no checkpoint at ${ckpt}" >&2
            failed+=("${exp}")
          fi
        done
      done
    done
  done
done

echo
echo "================ summary ================"
echo "ran:     ${#ran[@]}"
echo "skipped: ${#skipped[@]}"
echo "failed:  ${#failed[@]}"
if [[ "${#failed[@]}" -gt 0 ]]; then
  printf '  failed -> %s\n' "${failed[@]}" >&2
  echo "Some experiments failed; re-run the script to resume them." >&2
  exit 1
fi
echo "All requested experiments complete."
