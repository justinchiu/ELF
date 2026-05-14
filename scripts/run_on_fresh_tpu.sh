#!/usr/bin/env bash
# Create a fresh spot TPU, bootstrap it, and run an experiment on worker 0.
#
# Usage:
#   scripts/run_on_fresh_tpu.sh                       # default: toy_check
#   scripts/run_on_fresh_tpu.sh "<remote-command>"    # explicit command
#
# Examples:
#   scripts/run_on_fresh_tpu.sh "python eval_elbo_variance.py --toy_check"
#   scripts/run_on_fresh_tpu.sh "python eval_elbo_variance.py \
#       --config configs/training_configs/train_owt_ELF-B.yml \
#       --checkpoint_path embedded-language-flows/ELF-B-owt \
#       --num_examples 8 --steps 16,32 --probes 1,4 --repeats 4"
#
# Env overrides:
#   TPU_NAME, PROJECT, ZONE, ACCEL_TYPE, RUNTIME_VERSION,
#   GIT_REPO_HTTPS, GIT_BRANCH, CREATE_RETRY_SECONDS

set -euo pipefail

TPU_NAME="${TPU_NAME:-elf-elbo-v5p64-spot}"
PROJECT="${PROJECT:-pivotal-shield-449921-d4}"
ZONE="${ZONE:-us-east5-a}"
ACCEL_TYPE="${ACCEL_TYPE:-v5p-64}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5}"
GIT_REPO_HTTPS="${GIT_REPO_HTTPS:-https://github.com/justinchiu/ELF.git}"
GIT_BRANCH="${GIT_BRANCH:-elbo-uv}"
CREATE_RETRY_SECONDS="${CREATE_RETRY_SECONDS:-90}"

EXPERIMENT_CMD="${1:-python eval_elbo_variance.py --toy_check}"

log() { printf '[run_on_fresh_tpu] %s\n' "$*" >&2; }

gtpu() { gcloud compute tpus tpu-vm "$@" --project="$PROJECT" --zone="$ZONE"; }
gtpu_alpha() { gcloud alpha compute tpus tpu-vm "$@" --project="$PROJECT" --zone="$ZONE"; }

current_state() {
  gtpu describe "$TPU_NAME" --format='value(state)' 2>/dev/null || true
}

# 1. Delete prior instance if present (PREEMPTED nodes cannot be started).
state="$(current_state)"
if [[ -n "$state" ]]; then
  log "Existing $TPU_NAME in state=$state — deleting before recreate."
  gtpu delete "$TPU_NAME" --quiet
fi

# 2. Create with retry on RESOURCE_EXHAUSTED / capacity errors.
attempt=0
until gtpu create "$TPU_NAME" \
        --accelerator-type="$ACCEL_TYPE" \
        --version="$RUNTIME_VERSION" \
        --spot --quiet; do
  attempt=$((attempt + 1))
  log "Create attempt $attempt failed (likely no spot capacity). Sleeping ${CREATE_RETRY_SECONDS}s and retrying."
  sleep "$CREATE_RETRY_SECONDS"
done

# 3. Wait for READY.
until [[ "$(current_state)" == "READY" ]]; do
  log "Waiting for $TPU_NAME to become READY (current: $(current_state))..."
  sleep 10
done
log "$TPU_NAME is READY."

# 4. Bootstrap worker 0: install uv, clone ELF, install requirements, build venv.
#    Retry SSH a few times to absorb the post-create SSH-metadata propagation lag.
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
uv venv .venv --python 3.10
. .venv/bin/activate
uv pip install -r requirements.txt
EOF
)

ssh_attempt=0
until gtpu_alpha ssh "$TPU_NAME" --worker=0 --command="$BOOTSTRAP"; do
  ssh_attempt=$((ssh_attempt + 1))
  if (( ssh_attempt >= 10 )); then
    log "SSH bootstrap failed after $ssh_attempt attempts."
    exit 1
  fi
  log "SSH attempt $ssh_attempt failed; retrying in 15s..."
  sleep 15
done
log "Bootstrap complete on worker 0."

# 5. Run the experiment.
RUN=$(cat <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
. ~/ELF/.venv/bin/activate
cd ~/ELF/src
PYTHONUNBUFFERED=1 $EXPERIMENT_CMD
EOF
)

log "Running experiment on worker 0: $EXPERIMENT_CMD"
gtpu_alpha ssh "$TPU_NAME" --worker=0 --command="$RUN"
log "Experiment finished."
