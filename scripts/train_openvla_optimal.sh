#!/usr/bin/env bash
#
# Train the SAFE LSTM + MLP failure detectors on the ORIGINAL OpenVLA LIBERO
# rollouts using the authors' reported optimal hyperparameters, across 3 seeds.
# Intended for a RunPod (CUDA) instance.
#
# Optimal config (from the SAFE paper):
#   LSTM : token=last, lr=1e-4, lambda_reg=1     (hidden=256, 1 layer, BCE)
#   MLP  : token=last, lr=1e-4, lambda_reg=1e-2  (hidden=256, 2 layers, cumsum, n_history=1, "safe" loss)
# Common: last-layer feature (auto-discovered), seeds 0/1/2, 200 epochs,
#         no early stopping, batch 64, Adam  (the SAFE OpenVLA-LIBERO sweep setup).
#
# Resumable: each (model, seed) writes a checkpoint
#   <output_dir>/<exp_name>.pt
# only after that run finishes. On re-run, any run whose checkpoint already
# exists is skipped -- so a crashed / preempted spot instance just needs the
# script re-run to resume. A crash mid-run re-runs only that one (model, seed).
#
# RunPod setup (once):
#   pip install -r requirements.txt          # or your env's torch + numpy + scikit-learn + wandb
#   wandb login                              # or:  export WANDB_MODE=offline
#   # make sure the rollouts are at data/rollouts/openvla (override with ROOT=...)
#
# Usage:
#   bash scripts/train_openvla_optimal.sh                  # full run / resume
#   ROOT=/workspace/openvla bash scripts/train_openvla_optimal.sh
#   MODELS="lstm" SEEDS="0" bash scripts/train_openvla_optimal.sh   # subset
#   EPOCHS=1000 bash scripts/train_openvla_optimal.sh      # fully converged
#   DEVICE=cpu DRY=1 bash scripts/train_openvla_optimal.sh # preview only
#   FORCE=1 bash scripts/train_openvla_optimal.sh          # ignore existing checkpoints

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
PROJECT="${PROJECT:-safe-replicate-openvla}"
EPOCHS="${EPOCHS:-200}"
HIDDEN="${HIDDEN:-256}"
TOKEN="${TOKEN:-last}"        # paper-optimal pooling for both models
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-lstm mlp}"
FORCE="${FORCE:-0}"
DRY="${DRY:-0}"

# per-model paper-optimal hyperparameters
lr_for()     { echo "1e-4"; }                      # both models: lr=1e-4
lambda_for() { case "$1" in lstm) echo "1";; mlp) echo "1e-2";;
               *) echo "unknown model: $1" >&2; return 1;; esac; }
modeltag_for(){ case "$1" in lstm) echo "safe_lstm";; mlp) echo "safe_mlp";;
               *) echo "unknown model: $1" >&2; return 1;; esac; }
outdir_for() { echo "runs/openvla_optimal/$1"; }

# format a number the way the python script tags run names ({x:g})
gfmt() { "${PY}" -c "import sys; print(format(float(sys.argv[1]), 'g'))" "$1"; }

# token tag (normalize_token_pool then '_'->'-')
toktag_for() { case "$1" in first|0|0.0) echo first;; last|1|1.0) echo last;;
               mean) echo mean;; *) echo "$1";; esac; }

# experiment name as built by build_safe_openvla_libero_experiments() (n_history=1 => no tag)
expname_for() {
  local model="$1" seed="$2"
  echo "$(modeltag_for "$model")_tok-$(toktag_for "$TOKEN")_lr-$(gfmt "$(lr_for "$model")")_reg-$(gfmt "$(lambda_for "$model")")_seed-${seed}"
}

# --- count ---
total=0
for m in ${MODELS}; do for s in ${SEEDS}; do total=$((total+1)); done; done

echo "=== OpenVLA LIBERO: SAFE paper-optimal training ==="
echo "models=[${MODELS}] seeds=[${SEEDS}] token=${TOKEN} epochs=${EPOCHS} hidden=${HIDDEN}"
echo "  lstm: lr=$(lr_for lstm) lambda=$(lambda_for lstm)   mlp: lr=$(lr_for mlp) lambda=$(lambda_for mlp)"
echo "device=${DEVICE} project=${PROJECT} root=${ROOT} python=${PY}"
echo "total runs=${total}  (FORCE=${FORCE} DRY=${DRY})"
echo

failed=(); skipped=(); ran=(); idx=0

for model in ${MODELS}; do
  outdir="$(outdir_for "${model}")"
  group="${model}-openvla-optimal-${EPOCHS}ep"
  mkdir -p "${outdir}"
  lr="$(lr_for "${model}")"; lam="$(lambda_for "${model}")"

  for seed in ${SEEDS}; do
    idx=$((idx+1))
    exp="$(expname_for "${model}" "${seed}")"
    ckpt="${outdir}/${exp}.pt"
    prefix="[${idx}/${total}]"

    if [[ "${FORCE}" != "1" && -f "${ckpt}" ]]; then
      echo ">>> ${prefix} SKIP (exists): ${exp}"; skipped+=("${exp}"); continue
    fi
    if [[ "${DRY}" == "1" ]]; then
      echo ">>> ${prefix} DRY: ${exp}  (lr=${lr} lambda=${lam})  ->  ${outdir}"; continue
    fi

    echo ">>> ${prefix} RUN: ${exp}  ->  ${outdir}"
    log="${outdir}/train_${exp}.log"

    "${PY}" scripts/train_openvla_ablation.py \
      --root "${ROOT}" \
      --safe-openvla-libero-sweep \
      --safe-sweep-epochs "${EPOCHS}" \
      --no-early-stopping \
      --sweep-seeds "${seed}" \
      --sweep-token-pools "${TOKEN}" \
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
      echo "!!! ${prefix} FAILED (exit ${status}): ${exp} -- re-run script to retry" >&2; failed+=("${exp}")
    elif [[ -f "${ckpt}" ]]; then ran+=("${exp}")
    else echo "!!! ${prefix} WARNING: exit 0 but no checkpoint at ${ckpt}" >&2; failed+=("${exp}"); fi
  done
done

echo
echo "================ summary ================"
echo "ran:     ${#ran[@]}"
echo "skipped: ${#skipped[@]}"
echo "failed:  ${#failed[@]}"
if [[ "${#failed[@]}" -gt 0 ]]; then
  printf '  failed -> %s\n' "${failed[@]}" >&2
  echo "Some runs failed; re-run the script to resume them." >&2
  exit 1
fi
echo "All requested runs complete. Aggregate with:"
echo "  ${PY} scripts/summarize_mini_grid.py --root runs/openvla_optimal --csv runs/openvla_optimal/summary.csv"
