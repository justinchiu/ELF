# AGENTS.md

## TPU development workflow

We run training on a Cloud TPU pod (`v5p-64` = 8 hosts × 4 chips = 32 chips). The TPU VMs are launch targets, not dev environments — edit locally, sync via git, launch via `gcloud ... --worker=all`.

### TPU details

- Name: `elf-elbo-v5p64-spot`
- Project: `pivotal-shield-449921-d4`
- Zone: `us-east5-a`
- Topology: 8 hosts, 4 chips each (v5p-64)

### Loop

1. Edit locally.
2. Commit and push to a scratch branch.
3. `git pull` on all 8 hosts.
4. Launch the script on all 8 hosts.
5. Watch logs (per-host file, or worker 0 stdout).

### Sync code (git push/pull)

```bash
# Local
git commit -am "wip" && git push

# All 8 hosts
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF && git pull"
```

Use a scratch/dev branch so the "wip" commits don't pollute `main`.

### Launch training

The same Python script runs on every host. JAX coordinates across hosts via `jax.distributed.initialize()` (auto-detected on TPU). Each process sees all 32 chips through `jax.devices()`.

```python
# train.py
import jax
jax.distributed.initialize()
print(f"process {jax.process_index()}/{jax.process_count()}, "
      f"local devices: {jax.local_device_count()}, "
      f"global devices: {jax.device_count()}")
# ... training code, sharded across jax.devices()
```

Launch:

```bash
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF && python train.py 2>&1 | tee /tmp/train.log"
```

For long runs, wrap in `tmux` per host so SSH disconnects don't kill the job:

```bash
--command="cd ~/ELF && tmux new -d -s train 'python train.py 2>&1 | tee /tmp/train.log'"
```

### Debug a single host

```bash
# Interactive shell on worker 0
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a --worker=0

# Tail the log from a specific worker
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=3 --command="tail -f /tmp/train.log"
```

Prefix logs with `jax.process_index()` so interleaved output from `--worker=all` is readable.

### Gotchas

- `--worker=all` requires `--command=...`; you cannot get an interactive shell across all 8 hosts at once.
- A Jupyter kernel on worker 0 alone sees only 4 chips, not 32. Use it for prototyping, not full-pod runs.
- `--batch-size=8` parallelizes fan-out across hosts; without it, gcloud serializes.
- Spot TPUs can be preempted — checkpoint frequently to GCS.
