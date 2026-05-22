#!/usr/bin/env bash
set -euo pipefail

# Run a small grid of unbiased full-support time proposals for eval_elbo_bound.py.
# Intended to be launched on all TPU workers from the repo root.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src"

CONFIG="${CONFIG:-configs/training_configs/train_owt_ELF-B.yml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-embedded-language-flows/ELF-B-owt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/elbo_bound_grid/elf_b_owt}"
NUM_EXAMPLES="${NUM_EXAMPLES:-512}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
MC_SAMPLES="${MC_SAMPLES:-16}"
REPEATS="${REPEATS:-8}"
REFERENCE_MC_SAMPLES="${REFERENCE_MC_SAMPLES:-256}"
REFERENCE_REPEATS="${REFERENCE_REPEATS:-2}"
POSTERIOR_SIGMAS="${POSTERIOR_SIGMAS:-0.05 0.1 0.2 0.4 1.0}"
DISTRIBUTED="${DISTRIBUTED:-1}"
STREAMING="${STREAMING:-1}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

COMMON_ARGS=(
  --config "${CONFIG}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --num_examples "${NUM_EXAMPLES}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --mc_samples "${MC_SAMPLES}"
  --repeats "${REPEATS}"
  --reference_mc_samples "${REFERENCE_MC_SAMPLES}"
  --reference_repeats "${REFERENCE_REPEATS}"
)

if [[ "${DISTRIBUTED}" == "1" ]]; then
  COMMON_ARGS+=(--distributed)
fi
if [[ "${STREAMING}" == "1" ]]; then
  COMMON_ARGS+=(--streaming)
fi

run_one() {
  local name="$1"
  shift
  echo "=== ELBO estimator grid: ${name} ==="
  python eval_elbo_bound.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/${name}" \
    "$@"
}

for sigma in ${POSTERIOR_SIGMAS}; do
  # Symmetric logit-normal baseline.
  run_one "sigma${sigma}_sigmoid_loc0_scale2" \
    --posterior_sigma "${sigma}" \
    --time_proposal sigmoid_normal \
    --time_proposal_loc 0.0 \
    --time_proposal_scale 2.0

  # Right-shifted logit-normal proposals put more mass near the data endpoint.
  run_one "sigma${sigma}_sigmoid_loc2_scale2" \
    --posterior_sigma "${sigma}" \
    --time_proposal sigmoid_normal \
    --time_proposal_loc 2.0 \
    --time_proposal_scale 2.0

  run_one "sigma${sigma}_sigmoid_loc4_scale2" \
    --posterior_sigma "${sigma}" \
    --time_proposal sigmoid_normal \
    --time_proposal_loc 4.0 \
    --time_proposal_scale 2.0

  # Beta proposals with beta < 1 also emphasize t near 1 while keeping full support.
  run_one "sigma${sigma}_beta_a1_b0.5" \
    --posterior_sigma "${sigma}" \
    --time_proposal beta \
    --time_proposal_alpha 1.0 \
    --time_proposal_beta 0.5

  run_one "sigma${sigma}_beta_a1_b0.25" \
    --posterior_sigma "${sigma}" \
    --time_proposal beta \
    --time_proposal_alpha 1.0 \
    --time_proposal_beta 0.25
done
