# Progress: ELF ELBO Estimation

Date: 2026-05-14
Branch: `elbo-uv`

## Current Conclusion

- The only metric that matters for the current question is token-level ELBO estimation quality: bias and variance at a practical Monte Carlo budget.
- `src/eval_elbo_bound.py` is the current evaluator for that question. It estimates a full token-level ELBO for `p(tokens,z)=p_flow(z)p_decoder(tokens|z)` using a Gaussian variational posterior `q(z|tokens)=N(E(tokens), sigma_q^2 I)`, reports repeat variance, and can compare against a higher-MC reference for bias/MSE/RMSE.
- `src/eval_elf_loss.py` is auxiliary only. It reproduces the training objective mixture and is useful as a checkpoint/data smoke test, but it is not a likelihood, not an ELBO, and not a success metric.
- `src/eval_elbo_variance.py` is also auxiliary. It evaluates a zero-self-conditioning CNF diagnostic, which is a valid density for an artificial deterministic vector field, not the released self-conditioned/SDE sampler and not the recommended ELF likelihood path.

## Completed

- Added `PLAN.md` with the variance-focused ELBO / likelihood evaluation plan.
- Added `src/eval_elbo_bound.py` as the current token ELBO bias-variance evaluator:
  - Samples `z ~ q(z|tokens)` from a Gaussian posterior around frozen T5 embeddings.
  - Samples `(t, eps)` directly from the latent noising process.
  - Uses the zero-self-conditioning denoiser branch with explicit latent ELBO weighting.
  - Sums squared clean-latent prediction error over latent dimensions per token, rather than using the training-loss per-dimension mean.
  - Adds the fixed latent endpoint/base-measure constant, `0.5 * d * log(2*pi*sigma^2)`.
  - Adds decoder token NLL, `-log p_decoder(tokens|z)`.
  - Adds the analytic variational posterior term, `log q(z|tokens)`.
  - Reports repeat variance and optional bias/MSE/RMSE against a higher-MC reference.
  - The bound depends on `--posterior_sigma`, so that value must be swept or fixed before comparing checkpoints.
- Added `scripts/run_elbo_estimator_grid.sh` to sweep posterior widths and full-support time proposals without overwriting outputs.
- Added `src/eval_elbo_variance.py` for latent CNF likelihood diagnostics:
  - Hutchinson divergence via `jax.jvp`.
  - Backward Euler integration from `t=1` to `t=0`.
  - Masked padded positions consistently in the ODE path and prior term.
  - Distributed JAX init is opt-in via `--distributed`.
  - Optional high-budget reference estimates via `--reference_steps`,
    `--reference_probes`, and `--reference_repeats`.
  - Per-setting `bias^2`, repeat variance, MSE, and RMSE against the reference.
  - `--parallel_repeats` mode shards estimator repeats across all TPU chips with
    `pmap` and gathers repeat results across hosts.
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

## Auxiliary: ELF Training Loss

We do not care about this as the primary metric. The ELF training loss is not a likelihood ELBO; it is a stochastic mixture of two branch losses:

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

# Codex analysis

## Self-Conditioning and Likelihood Interpretation

ELF-B-owt was trained with self-conditioning:

```text
self_cond_prob = 0.5
decoder_prob = 0.2
denoiser_noise_scale = 2.0
```

Let a token sequence `s` be encoded by frozen T5 into normalized latents:

```text
x = E(s) in R^{L x d}
```

For the denoiser branch, training samples:

```text
t ~ sigmoid(N(-1.5, 0.8^2))
eps ~ N(0, I)
z_t = t * x + (1 - t) * sigma * eps
sigma = 2.0
```

The base flow-matching target is:

```text
v*(z_t, t, x) = (x - z_t) / max(1 - t, t_eps)
```

When self-conditioning is enabled, the model input is the concatenation of the noisy latent and a self-conditioning latent:

```text
x_hat_init = stopgrad(F_theta([z_t, 0], t, w))
m ~ Bernoulli(self_cond_prob)
c = m * x_hat_init
x_hat_theta = F_theta([z_t, c], t, w)
v_hat_theta = (x_hat_theta - z_t) / max(1 - t, t_eps)
```

Ignoring the self-conditioning CFG target adjustment, the denoiser objective is:

```text
L_den = || v_hat_theta - stopgrad(v*) ||^2
```

With self-conditioning CFG tokens, training also samples a scale `w` and adjusts the target as:

```text
v_target = v* + m * (1 - 1 / w) * (v_cond - v_uncond)
```

where `v_uncond` is computed from `[z_t, 0]` and `v_cond` is computed from `[z_t, x_hat_uncond]`.

The decoder branch samples per-token noising:

```text
lambda_i = sigmoid(N(0.8, 0.8^2))
eta_i ~ N(0, decoder_noise_scale^2 I)
z_dec,i = lambda_i * x_i + (1 - lambda_i) * eta_i
```

Then it runs the decoder head at `t = 1` and optimizes token cross-entropy:

```text
logits = D_theta([z_dec, 0], t=1)
L_dec = CE(logits, s)
```

The expected training objective is therefore:

```text
E[L] = (1 - decoder_prob) * E[L_den] + decoder_prob * E[L_dec]
     = 0.8 * E[L_den] + 0.2 * E[L_dec]
```

The released sampler is not a plain continuous normalizing flow with vector field `v_theta(z, t)`.
It is recurrently self-conditioned:

```text
x_hat_prev,0 = 0
v_k, x_hat_k = F_theta([z_k, x_hat_prev,k], t_k)
z_{k+1} = z_k + (t_{k+1} - t_k) * v_k
x_hat_prev,k+1 = x_hat_k
```

So the effective sampler field is history-dependent:

```text
dz/dt = v_theta(z, t, x_hat_prev)
```

and the standard CNF likelihood formula does not directly apply to the full released self-conditioned sampler.

The current `eval_elbo_variance.py` diagnostic intentionally evaluates the zero-self-conditioning branch:

```text
v0_theta(z, t) = (F_theta([z, 0], t, w=1) - z) / max(1 - t, t_eps)
```

For this deterministic branch, the standard latent CNF likelihood diagnostic is:

```text
log p_1(x) = log p_0(z_0) - integral_0^1 div_z v0_theta(z_t, t) dt
p_0 = N(0, sigma^2 I)
sigma = 2.0
```

with Hutchinson divergence:

```text
div_z v0_theta(z, t) = tr(d v0_theta / dz)
                    ~= (1 / K) * sum_k eps_k^T (d v0_theta / dz) eps_k
```

Interpretation:

- This is a valid CNF diagnostic for the zero-self-conditioning vector field.
- The checkpoint did train on this branch because `self_cond_prob = 0.5` means roughly half of denoiser examples use zero self-conditioning.
- It should not be described as the exact likelihood of the full released self-conditioned/SDE sampler.

## OWT Dataset Status

The released `embedded-language-flows/openwebtext-t5` dataset has only one split:

```text
train: 9,737,184 examples
```

There is no released OWT validation/test split in the dataset metadata, and `train_owt_ELF-B.yml` does not set `eval_data_path`.

Implication:

- We can compute ELF loss on the released training split or on a deterministic pseudo-heldout slice, for example the final `N` examples.
- We cannot honestly call that "OWT test" unless we provide a separate external OWT test/eval dataset path.

## TPU Runs Completed

- Auxiliary only: added `src/eval_elf_loss.py` to compute the ELF training objective from a checkpoint.
- It supports:
  - `--checkpoint_path embedded-language-flows/ELF-B-owt`.
  - `--distributed` for full-pod JAX.
  - `--streaming` to avoid downloading the full OWT Arrow repo for small smoke tests.
  - Distributed dataset partitioning by host for streaming mode.
  - The repo dataloader with `DistributedSampler` for non-streaming mode.
  - EMA parameters by default via `--param_source ema`.
- Current auxiliary verification:
  - `python -m py_compile src/eval_elf_loss.py` passes locally.
  - Distributed v5p-64 smoke test passed on 2026-05-22:
    - 8 hosts, 32 global devices.
    - 512 streamed OWT examples, 490,353 tokens.
    - `denoiser_l2 = 0.6708501962`.
    - `decoder_ce = 0.2004722236`.
    - `expected_elf_loss = 0.5767746017`.
    - Runtime for the eval loop was 23.5 seconds after setup.

The single-example latent CNF convergence check also completed on the v5p-64 pod.
Artifacts were copied locally to:

```text
outputs/elbo_variance/elf_b_owt/results.jsonl
outputs/elbo_variance/elf_b_owt/summary.json
outputs/elbo_variance/elf_b_owt/per_example_estimates.npz
```

Run settings:

```text
num_examples = 1
max_length = 128
steps = [16, 32, 64, 128]
probes = [1, 2, 4, 8, 16]
repeats = 16
reference_steps = 512
reference_probes = 256
reference_repeats = 32
probe_dist = rademacher
parallel_repeats = true
```

Reference estimate:

```text
mean_nats_per_token = 15239.280273
runtime = 38.5s
```

Highest-probe rows by step:

```text
steps=16,  probes=16: mean_nats/tok= 6500.964844, repeat_std=0.164495, rmse_vs_ref=8738.315627
steps=32,  probes=16: mean_nats/tok= 9078.494141, repeat_std=0.106652, rmse_vs_ref=6160.786313
steps=64,  probes=16: mean_nats/tok=11861.451172, repeat_std=0.124584, rmse_vs_ref=3377.829629
steps=128, probes=16: mean_nats/tok=13575.762695, repeat_std=0.045579, rmse_vs_ref=1663.517959
```

Initial read:

- Hutchinson repeat variance is small at fixed step/probe settings.
- Probe count barely changes the mean relative to the step-count sweep.
- The main instability in this run is ODE discretization / estimator bias: even 128 steps remains about 1,664 nats/token RMSE from the 512-step reference.
- Conclusion: this CNF path should stay a diagnostic/regression check. It should not drive the ELF ELBO bias/variance conclusion.

## Next Steps

1. Run `scripts/run_elbo_estimator_grid.sh` on a fixed evaluation slice with a cheap MC budget and a higher-MC reference. This grids MC proposal distributions, not ELF's fixed noising schedule. Choose a time proposal whose repeat variance shrinks with MC samples, whose proposal means agree with the other unbiased proposals within MC error, and whose importance weights are not dominated by rare endpoint samples.

```bash
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF && scripts/run_elbo_estimator_grid.sh 2>&1 | tee /tmp/eval_elbo_grid.log"
```

2. For a single-estimator run, use `src/eval_elbo_bound.py` and report token NELBO, bias, variance, MSE, RMSE, and runtime.

```bash
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF/src && python eval_elbo_bound.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt \
    --distributed \
    --streaming \
    --num_examples 512 \
    --global_batch_size 512 \
    --mc_samples 16 \
    --repeats 8 \
    --reference_mc_samples 256 \
    --reference_repeats 2 \
    --posterior_sigma 0.2 \
    --time_proposal sigmoid_normal \
    --time_proposal_scale 2.0 \
    2>&1 | tee /tmp/eval_elbo_bound.log"
```

3. Sweep or justify `--posterior_sigma`; the ELBO is valid for any positive value, but tightness and variance depend on this variational family. The first-pass grid brackets `{0.05, 0.1, 0.2, 0.4, 1.0}`.
4. Sweep or justify the time proposal. Use full-support proposals (`sigmoid_normal` or `beta`) for a strict bound; `truncated_uniform` is a biased diagnostic because it omits endpoint mass.
5. Keep reporting components separately: latent flow NELBO, latent endpoint constant, decoder token NLL, posterior `log q`, and total token NELBO.
6. Check proposal diagnostics. Healthy proposals should have cross-proposal mean agreement, `max/mean` importance-weight ratios below about `100`, and effective sample size fraction above about `0.1`; otherwise treat the endpoint tail as unresolved and report the most conservative high-NELBO estimator.
7. Choose a clear evaluation set:
  - External OWT test/eval path if available.
  - Otherwise a deterministic pseudo-heldout slice, documented by `[start_index, end_index)`.
8. Keep the latent CNF script only as a diagnostic for artificial zero-self-cond CNF behavior or a future trajectory-consistent distilled model.
9. For a real distributed run, recreate the node with `scripts/run_on_fresh_tpu.sh` and bootstrap all 8 hosts.

## TPU Cleanup

The `elf-elbo-v5p64-spot` v5p-64 pod was deleted after the 2026-05-22 runs completed and artifacts were copied locally. Since pod TPUs do not support stop/start, deletion is the appropriate way to stop active TPU billing.

---

## Claude Analysis (2026-05-22): Use the flow-matching ELBO, not the CNF integral

### What the CNF convergence check showed

Initial 1-example sweep (`outputs/elbo_variance/elf_b_owt/`) with steps `[16, 32, 64, 128]` × probes `[1..16]` × 16 repeats, against a 512-step / 256-probe / 32-repeat reference:

- **Hutchinson variance is not the bottleneck.** Std across repeats scales as ~1/√P; at 128 steps × 16 probes it is ~0.046 nats/token (well under the 0.1–0.25 threshold from PLAN.md "Decision Criteria").
- **ODE discretization bias dominates by ~6 orders of magnitude.** Mean nats/token climbs monotonically `6501 → 9078 → 11861 → 13576` for steps `16 → 32 → 64 → 128`, with the 512-step reference at `15239`. Each doubling halves the gap (classic Euler `O(1/N)`). At 128 steps the bias-vs-reference is still ~1665 nats/token. Getting bias under the 0.25 nats/token threshold would need many thousands of steps even assuming the reference itself is converged.
- **The backward trajectory does not land at the prior.** `mean_z0_rms ≈ 16` at the reference; the prior is `N(0, 4 I)` with per-element RMS = 2. So `log p_0(z_0)` is in the prior tail by an enormous margin, and the bulk of the "likelihood" number is that tail penalty, not a meaningful density.

### Why the CNF approach is the wrong tool for ELF

The eval script uses self-cond input zeros and `cfg=1.0`, both of which are in the training distribution (training uses `self_cond_prob=0.5` and `self_cond_cfg ∈ [0.5, 5.0]`). So the model is being called in a trained mode. The problem is more fundamental:

**Flow-matching training is pointwise, not trajectory-wise.** The loss enforces `v_pred(z_t, t) ≈ v_target(z_t, t)` at random `(z_t, t)` pairs from the noising process. It never enforces that *integrating* `v_pred` from data back to `t=0` reproduces a noise sample. Small pointwise errors accumulate over an ODE integration, which is why the backward map does not match the inverse of the forward noising. The change-of-variables formula then computes a well-defined CNF density for the artificial zero-self-conditioning vector field, but not the density of the released ELF sampler or of a trajectory-consistent likelihood model.

This is exactly the gap the F2D2 paper (2512.02636) addresses by distilling a separate trajectory-consistent model for likelihood evaluation.

### Recommendation: flow-matching ELBO

For flow-matching / diffusion models, the **training loss with the right schedule-dependent reweighting is a variational upper bound on NLL** (Kingma et al. 2021 "Variational Diffusion Models"; Song et al. for score matching):

```text
-log p_flow(z) <= E_{t, eps} [ w_v(t) * || v_theta(z_t, t) - v_target ||^2 ]
```

This bound uses exactly the quantity the model was trained on. No ODE integration, no Hutchinson divergence.

### Concrete recipe for ELF

For a token-level bound, define the generative model and variational posterior:

```text
p_theta(s, z) = p_flow(z) * p_decoder(s | z)
q_sigma(z | s) = N(E(s), sigma_q^2 I)
```

Then:

```text
-ELBO(s) = E_{z ~ q_sigma}[
    latent_flow_NELBO(z)
  + latent_endpoint_const
  + CE_decoder(s | z)
  + log q_sigma(z | s)
]
```

For the latent term:

1. Encode the example with T5 → `x_mean`, then sample `z ~ N(x_mean, sigma_q^2 I)` on predicted token positions.
2. Sample `t ~ Uniform[0, 1]` (or any proposal with importance weights).
3. Sample `eps ~ N(0, I)`, form `z_t = t * z + (1 - t) * sigma * eps` with `sigma = denoiser_noise_scale = 2.0`.
4. Run model with self-cond input zeros, `cfg_scale = 1.0` → `x_pred`.
5. Compute masked per-token clean-latent error `|| x_pred - z ||^2`.
6. Multiply by the x-prediction VDM weight. For `alpha_t = t`, `sigma_t = (1 - t) * sigma`:
   ```text
   SNR(t) = t^2 / ((1 - t)^2 * sigma^2)
   0.5 * dSNR(t)/dt * ||x_pred - z||^2
     = t / (sigma^2 * (1 - t)^3) * ||x_pred - z||^2
   ```
7. Average over `(t, eps)` samples.
8. Add the fixed latent endpoint/base-measure constant:
   ```text
   latent_endpoint_const = 0.5 * d * log(2*pi*sigma^2)
   ```
9. Run the decoder at `t=1` on the same sampled `z` and add masked token CE.
10. Add analytic `E_q[log q_sigma(z|s)] = -0.5 * d * (1 + log(2*pi*sigma_q^2))` per predicted token.

### Why this is better

|                            | CNF (Option A)                                       | Flow-matching ELBO (Option B)                          |
| -------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Per-example cost           | ~128 ODE steps * 16 probes                           | ~64 (t, eps) samples, no ODE                           |
| Variance source            | Hutchinson + discretization                          | Just (t, eps) Monte Carlo                              |
| Bias                       | Huge and uncontrolled (no trajectory consistency)    | Bounded above by the fixed ELBO gap                    |
| Matches training?          | No (training is pointwise)                           | Yes (training optimizes the same form)                 |
| Result is                  | Density of an artificial zero-self-cond CNF, not the released ELF sampler | Token-level variational bound for the explicit `p(tokens,z)` model and chosen `q_sigma` |

### Caveats

1. It is a bound, not the density. The ELBO gap depends on the chosen `q_sigma`; sweep or justify `--posterior_sigma` before model comparisons.
2. The default latent weighting now matches the x-prediction VDM parameterization for `add_noise` and the model's clean-latent output.
3. `truncated_uniform` time sampling is only a diagnostic; use full-support `sigmoid_normal` for a strict bound.
4. This would bound the density of the model viewed as a flow-matching model. That is not identical to the density induced by the released SDE + self-conditioned sampler, but is closer to the training objective than the deterministic CNF integral.

### Implementation path

- `src/eval_elbo_bound.py` now implements the full token-level path: sample `z ~ q_sigma(z|tokens)`, estimate the latent flow NELBO, add decoder token CE, add analytic `log q`, and report total token NELBO bias/variance against a higher-MC reference when requested.
- `src/eval_elf_loss.py` should stay separate as a training-objective smoke check. It should not be used for the likelihood/ELBO conclusion.
- The existing CNF variance work stays useful as a regression check for a future trajectory-consistent (distilled) model, where Option A would be the right tool.
