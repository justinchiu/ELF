#!/usr/bin/env bash
set -euo pipefail

# Run a small grid of unbiased full-support time proposals for eval_elbo_bound.py.
# Intended to be launched on all TPU workers from the repo root.
# Set GCS_OUTPUT_ROOT=gs://... to sync each config's incremental progress to durable storage.
# With SKIP_COMPLETED=1, configs with an existing summary.json are skipped.

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
WEIGHT_MODE="${WEIGHT_MODE:-vdm_xpred}"
T_MAX="${T_MAX:-1.0}"
POSTERIOR_SIGMAS="${POSTERIOR_SIGMAS:-0.05 0.1 0.2 0.4 1.0}"
DISTRIBUTED="${DISTRIBUTED:-1}"
STREAMING="${STREAMING:-1}"
GCS_OUTPUT_ROOT="${GCS_OUTPUT_ROOT:-}"
GCS_SYNC_RETRIES="${GCS_SYNC_RETRIES:-3}"
GCS_SYNC_RETRY_SECONDS="${GCS_SYNC_RETRY_SECONDS:-15}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

if [[ -n "${GCS_OUTPUT_ROOT}" && "${GCS_OUTPUT_ROOT}" != gs://* ]]; then
  echo "GCS_OUTPUT_ROOT must be a gs:// path, got: ${GCS_OUTPUT_ROOT}" >&2
  exit 1
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
  --weight_mode "${WEIGHT_MODE}"
  --t_max "${T_MAX}"
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
  local status=0
  local gcs_args=()

  if should_skip "${name}"; then
    return 0
  fi

  if [[ -n "${GCS_OUTPUT_ROOT}" ]]; then
    gcs_args=(
      --gcs_output_dir "${GCS_OUTPUT_ROOT}/${name}"
      --gcs_sync_retries "${GCS_SYNC_RETRIES}"
      --gcs_sync_retry_seconds "${GCS_SYNC_RETRY_SECONDS}"
    )
  fi

  echo "=== ELBO estimator grid: ${name} ==="
  python eval_elbo_bound.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/${name}" \
    "${gcs_args[@]}" \
    "$@" || status=$?

  if ! sync_one "${name}"; then
    if [[ "${status}" -eq 0 ]]; then
      return 1
    fi
  fi
  return "${status}"
}

should_skip() {
  local name="$1"
  local local_summary="${OUTPUT_ROOT}/${name}/summary.json"
  local remote_summary="${GCS_OUTPUT_ROOT}/${name}/summary.json"

  if [[ "${SKIP_COMPLETED}" != "1" ]]; then
    return 1
  fi

  if [[ -f "${local_summary}" ]]; then
    echo "=== ELBO estimator grid: ${name} already complete locally; skipping ==="
    return 0
  fi

  if [[ -n "${GCS_OUTPUT_ROOT}" ]] && gcloud storage ls "${remote_summary}" >/dev/null 2>&1; then
    echo "=== ELBO estimator grid: ${name} already complete in GCS; skipping ==="
    return 0
  fi

  return 1
}

sync_one() {
  local name="$1"
  local local_dir="${OUTPUT_ROOT}/${name}"
  local remote_dir="${GCS_OUTPUT_ROOT}/${name}"
  local attempt

  if [[ -z "${GCS_OUTPUT_ROOT}" ]]; then
    return 0
  fi

  # Only the JAX process-0 host writes output files. Non-writing hosts run this
  # script too, but should not fail or create empty GCS directories.
  if [[ ! -d "${local_dir}" ]] || [[ -z "$(find "${local_dir}" -type f -print -quit)" ]]; then
    echo "No local outputs for ${name} on $(hostname); skipping GCS sync."
    return 0
  fi

  for attempt in $(seq 1 "${GCS_SYNC_RETRIES}"); do
    echo "Syncing ${local_dir} -> ${remote_dir} (attempt ${attempt}/${GCS_SYNC_RETRIES})"
    if gcloud storage rsync --recursive "${local_dir}" "${remote_dir}"; then
      echo "Synced ${name} to ${remote_dir}"
      return 0
    fi
    sleep "${GCS_SYNC_RETRY_SECONDS}"
  done

  echo "Failed to sync ${name} to ${remote_dir}" >&2
  return 1
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
