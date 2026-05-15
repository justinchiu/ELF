# Progress: ELF ELBO / Loss Evaluation

Date: 2026-05-14
Branch: `elbo-uv`

## Completed

- Added `PLAN.md` with the variance-focused ELBO / likelihood evaluation plan.
- Added `src/eval_elbo_variance.py` for latent CNF likelihood diagnostics:
  - Hutchinson divergence via `jax.jvp`.
  - Backward Euler integration from `t=1` to `t=0`.
  - Masked padded positions consistently in the ODE path and prior term.
  - Distributed JAX init is opt-in via `--distributed`.
  - Optional high-budget reference estimates via `--reference_steps`,
    `--reference_probes`, and `--reference_repeats`.
  - Per-setting `bias^2`, repeat variance, MSE, and RMSE against the reference.
- Added and hardened `scripts/run_on_fresh_tpu.sh`:
  - Recreates `PREEMPTED` v5p-64 spot pods.
  - Reuses `READY` pods by default.
  - Requires `FORCE_DELETE_EXISTING=1` before deleting non-preempted pods.
  - Can bootstrap `worker=0` or `worker=all`.
  - Default toy check uses `JAX_PLATFORMS=cpu` to avoid single-worker TPU backend hangs.
- Verified the toy Hutchinson trace check on fresh worker 0:
  - Identity field: exact for Rademacher probes.
  - Zero field: exact.
  - Linear field: true trace inside Monte Carlo uncertainty.

Toy-check output:

```text
[identity | rademacher] expected=+128.0000  estimated=+128.0000 +/- 0.0000
[identity | gaussian  ] expected=+128.0000  estimated=+128.3520 +/- 1.4203
[zero     | rademacher] expected=+0.0000    estimated=+0.0000 +/- 0.0000
[zero     | gaussian  ] expected=+0.0000    estimated=+0.0000 +/- 0.0000
[linear   | rademacher] expected=+0.6338    estimated=+0.4231 +/- 1.1295
[linear   | gaussian  ] expected=+0.6338    estimated=+0.7488 +/- 1.1565
```

## ELF Training Loss

The ELF training loss is not a likelihood ELBO. The code trains a stochastic mixture of two branch losses:

```text
expected_loss = (1 - decoder_prob) * denoiser_l2 + decoder_prob * decoder_ce
```

For ELF-B-owt, `decoder_prob = 0.2`, so:

```text
expected_loss = 0.8 * denoiser_l2 + 0.2 * decoder_ce
```

The denoiser branch:

- Encodes tokens with frozen T5-small and normalizes latents.
- Samples `t` from the configured logit-normal schedule.
- Samples Gaussian noise and forms `z = t*x0 + (1-t)*noise*denoiser_noise_scale`.
- Predicts clean embeddings, converts to velocity with:

```text
v = (x_pred - z) / max(1 - t, t_eps)
```

- Computes masked mean squared error against the target velocity, including the self-conditioning target logic.

The decoder branch:

- Samples a separate per-token logit-normal noising level.
- Forms a noisy latent near `t=1`.
- Runs the decoder head and computes masked token cross-entropy.

## OWT Dataset Status

The released `embedded-language-flows/openwebtext-t5` dataset has only one split:

```text
train: 9,737,184 examples
```

There is no released OWT validation/test split in the dataset metadata, and `train_owt_ELF-B.yml` does not set `eval_data_path`.

Implication:

- We can compute ELF loss on the released training split or on a deterministic pseudo-heldout slice, for example the final `N` examples.
- We cannot honestly call that "OWT test" unless we provide a separate external OWT test/eval dataset path.

## In Progress

- Added draft `src/eval_elf_loss.py` to compute the ELF training objective from a checkpoint.
- It supports:
  - `--checkpoint_path embedded-language-flows/ELF-B-owt`.
  - `--distributed` for full-pod JAX.
  - `--streaming` to avoid downloading the full OWT Arrow repo for small smoke tests.
  - Distributed dataset partitioning by host for streaming mode.
  - The repo dataloader with `DistributedSampler` for non-streaming mode.
  - EMA parameters by default via `--param_source ema`.
- Current verification:
  - `python -m py_compile src/eval_elf_loss.py` passes locally.
  - The script has not yet been run on TPU.

## Next Steps

1. Commit `src/eval_elf_loss.py` and this progress note.
2. Recreate the TPU when needed with `scripts/run_on_fresh_tpu.sh`.
3. Run a small distributed smoke test:

```bash
RUN_WORKERS=all scripts/run_on_fresh_tpu.sh "python eval_elf_loss.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt \
  --distributed \
  --streaming \
  --num_examples 512 \
  --global_batch_size 512 \
  --repeats 1"
```

4. Run the single-example latent CNF estimator convergence check:

```bash
scripts/run_on_fresh_tpu.sh "python eval_elbo_variance.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt \
  --num_examples 1 \
  --max_length 128 \
  --steps 16,32,64,128 \
  --probes 1,2,4,8,16 \
  --repeats 16 \
  --reference_steps 512 \
  --reference_probes 256 \
  --reference_repeats 4"
```

5. If the smoke test succeeds, choose a clear evaluation set:
  - External OWT test/eval path if available.
  - Otherwise a deterministic pseudo-heldout slice, documented by `[start_index, end_index)`.

6. For a real distributed run, make sure the node is recreated and all 8 hosts are bootstrapped.

## TPU Cleanup

The active `elf-elbo-v5p64-spot` v5p-64 pod was `READY` after the toy check. Since pod TPUs do not support stop/start, deletion is the appropriate way to stop active TPU billing. A delete request was issued after this progress checkpoint.
