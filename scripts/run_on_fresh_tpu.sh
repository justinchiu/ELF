#!/usr/bin/env bash
# Create or reuse a spot TPU pod, bootstrap ELF, and run an experiment.
#
# Safe defaults:
#   - PREEMPTED nodes are deleted and recreated.
#   - READY nodes are reused, not deleted.
#   - Other existing states require FORCE_DELETE_EXISTING=1 before deletion.
#   - All workers are bootstrapped so distributed runs can work.
#   - The default experiment runs only on worker 0 for the cheap toy check.
#
# Usage:
#   scripts/run_on_fresh_tpu.sh                       # worker-0 toy_check
#   scripts/run_on_fresh_tpu.sh "<remote-command>"    # explicit command
#
# Examples:
#   scripts/run_on_fresh_tpu.sh "JAX_PLATFORMS=cpu python eval_elbo_variance.py --toy_check"
#
#   RUN_WORKERS=0 BOOTSTRAP_WORKERS=0 scripts/run_on_fresh_tpu.sh \
#     "python eval_elbo_variance.py --toy_check"
#
#   RUN_WORKERS=all scripts/run_on_fresh_tpu.sh "python eval_elbo_variance.py \
#       --config configs/training_configs/train_owt_ELF-B.yml \
#       --checkpoint_path embedded-language-flows/ELF-B-owt \
#       --distributed \
#       --num_examples 8 --steps 16,32 --probes 1,4 --repeats 4"
#
# Env overrides:
#   TPU_NAME, PROJECT, ZONE, ACCEL_TYPE, RUNTIME_VERSION,
#   GIT_REPO_HTTPS, GIT_BRANCH, CREATE_RETRY_SECONDS,
#   BOOTSTRAP_WORKERS, RUN_WORKERS, WORKER_BATCH_SIZE,
#   FORCE_DELETE_EXISTING, SKIP_BOOTSTRAP

set -euo pipefail

TPU_NAME="${TPU_NAME:-elf-elbo-v5p64-spot}"
PROJECT="${PROJECT:-pivotal-shield-449921-d4}"
ZONE="${ZONE:-us-east5-a}"
ACCEL_TYPE="${ACCEL_TYPE:-v5p-64}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5}"
GIT_REPO_HTTPS="${GIT_REPO_HTTPS:-https://github.com/justinchiu/ELF.git}"
GIT_BRANCH="${GIT_BRANCH:-elbo-uv}"
CREATE_RETRY_SECONDS="${CREATE_RETRY_SECONDS:-90}"
BOOTSTRAP_WORKERS="${BOOTSTRAP_WORKERS:-all}"
RUN_WORKERS="${RUN_WORKERS:-0}"
WORKER_BATCH_SIZE="${WORKER_BATCH_SIZE:-8}"
FORCE_DELETE_EXISTING="${FORCE_DELETE_EXISTING:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

EXPERIMENT_CMD="${1:-JAX_PLATFORMS=cpu python eval_elbo_variance.py --toy_check}"

log() { printf '[run_on_fresh_tpu] %s\n' "$*" >&2; }

gtpu() { gcloud compute tpus tpu-vm "$@" --project="$PROJECT" --zone="$ZONE"; }
gtpu_alpha() { gcloud alpha compute tpus tpu-vm "$@" --project="$PROJECT" --zone="$ZONE"; }

usage_error() {
  log "ERROR: $*"
  exit 1
}

current_state() {
  gtpu describe "$TPU_NAME" --format='value(state)' 2>/dev/null || true
}

delete_tpu() {
  log "Deleting $TPU_NAME."
  gtpu delete "$TPU_NAME" --quiet
  while [[ -n "$(current_state)" ]]; do
    log "Waiting for $TPU_NAME to disappear (current: $(current_state))..."
    sleep 10
  done
}

create_tpu() {
  local attempt=0
  until gtpu create "$TPU_NAME" \
          --accelerator-type="$ACCEL_TYPE" \
          --version="$RUNTIME_VERSION" \
          --spot --quiet; do
    attempt=$((attempt + 1))
    log "Create attempt $attempt failed (likely no spot capacity). Sleeping ${CREATE_RETRY_SECONDS}s and retrying."
    sleep "$CREATE_RETRY_SECONDS"
  done
}

wait_ready() {
  while [[ "$(current_state)" != "READY" ]]; do
    log "Waiting for $TPU_NAME to become READY (current: $(current_state))..."
    sleep 10
  done
  log "$TPU_NAME is READY."
}

ssh_tpu() {
  local workers="$1"
  local command="$2"
  if [[ "$workers" == "all" ]]; then
    gtpu_alpha ssh "$TPU_NAME" \
      --worker=all --batch-size="$WORKER_BATCH_SIZE" \
      --command="$command"
  else
    gtpu_alpha ssh "$TPU_NAME" \
      --worker="$workers" \
      --command="$command"
  fi
}

ssh_tpu_with_retry() {
  local label="$1"
  local workers="$2"
  local command="$3"
  local attempt=0
  until ssh_tpu "$workers" "$command"; do
    attempt=$((attempt + 1))
    if (( attempt >= 10 )); then
      log "$label failed after $attempt attempts."
      exit 1
    fi
    log "$label attempt $attempt failed; retrying in 15s..."
    sleep 15
  done
}

# 1. Create or reuse the TPU.
state="$(current_state)"
case "$state" in
  "")
    log "$TPU_NAME does not exist; creating spot TPU."
    create_tpu
    ;;
  PREEMPTED)
    log "Existing $TPU_NAME is PREEMPTED; deleting and recreating."
    delete_tpu
    create_tpu
    ;;
  DELETING)
    log "Existing $TPU_NAME is already DELETING; waiting for deletion to finish before recreating."
    while [[ -n "$(current_state)" ]]; do
      log "Waiting for $TPU_NAME to disappear (current: $(current_state))..."
      sleep 10
    done
    create_tpu
    ;;
  READY)
    if [[ "$FORCE_DELETE_EXISTING" == "1" ]]; then
      log "Existing $TPU_NAME is READY; FORCE_DELETE_EXISTING=1 so recreating."
      delete_tpu
      create_tpu
    else
      log "Existing $TPU_NAME is READY; reusing it. Set FORCE_DELETE_EXISTING=1 to recreate."
    fi
    ;;
  *)
    if [[ "$FORCE_DELETE_EXISTING" == "1" ]]; then
      log "Existing $TPU_NAME is state=$state; FORCE_DELETE_EXISTING=1 so recreating."
      delete_tpu
      create_tpu
    else
      usage_error "Existing $TPU_NAME is state=$state. Refusing to delete it. Re-run with FORCE_DELETE_EXISTING=1 if that is intended."
    fi
    ;;
esac

wait_ready

# 2. Bootstrap the requested workers: install uv, clone ELF, install requirements, build venv.
BOOTSTRAP=$(cat <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"
if [[ ! -d ~/ELF ]]; then
  git clone --branch "$GIT_BRANCH" "$GIT_REPO_HTTPS" ~/ELF
else
  cd ~/ELF
  git fetch origin
  git checkout "$GIT_BRANCH"
  git pull --ff-only origin "$GIT_BRANCH"
fi
cd ~/ELF
uv venv .venv --python 3.10 --clear
. .venv/bin/activate
uv pip install --prerelease=allow -r requirements.txt
EOF
)

if [[ "$SKIP_BOOTSTRAP" == "1" ]]; then
  log "SKIP_BOOTSTRAP=1; skipping bootstrap."
else
  log "Bootstrapping workers=$BOOTSTRAP_WORKERS on $TPU_NAME."
  ssh_tpu_with_retry "Bootstrap" "$BOOTSTRAP_WORKERS" "$BOOTSTRAP"
  log "Bootstrap complete on workers=$BOOTSTRAP_WORKERS."
fi

# 3. Run the experiment.
RUN=$(cat <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
. ~/ELF/.venv/bin/activate
cd ~/ELF/src
PYTHONUNBUFFERED=1 $EXPERIMENT_CMD
EOF
)

log "Running experiment on workers=$RUN_WORKERS: $EXPERIMENT_CMD"
ssh_tpu_with_retry "Experiment" "$RUN_WORKERS" "$RUN"
log "Experiment finished."
