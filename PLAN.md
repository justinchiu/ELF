# Plan: Evaluate ELF-B-owt ELBO Estimator Variance

## Goal

Evaluate whether ELBO / likelihood estimation for the smallest ELF model is dominated by estimator variance.

Smallest model checkpoint:

- Hugging Face: https://huggingface.co/embedded-language-flows/ELF-B-owt

Relevant papers:

- ELF: Embedded Language Flows: https://arxiv.org/pdf/2605.10938
- Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models: https://arxiv.org/pdf/2512.02636

## Motivation

The ELF paper reports generative perplexity for OpenWebText rather than validation perplexity, and notes that likelihood evaluation for flow-based models can require likelihood-specific training. The F2D2 paper gives the relevant likelihood machinery for continuous normalizing flows: likelihood evaluation requires integrating the divergence of the velocity field along a trajectory. Exact divergence is expensive, while the Hutchinson trace estimator is unbiased but can have very high variance. Their appendix reports per-step Hutchinson variance around `1.8e6` in their setting.

The core question is therefore not "what is one ELBO number?" but:

> At practical compute budgets, is the ELF-B-owt ELBO / likelihood estimator stable enough to be useful?

## Evaluation Targets

### 1. Primary diagnostic: latent CNF likelihood

Evaluate the continuous latent likelihood of frozen T5-encoded OWT examples under the ELF flow.

For an encoded latent `x_1`, backward integrate the ODE from `t = 1` to `t = 0` and estimate:

```text
log p(x_1) = log p_0(x_0) - integral_0^1 div v_theta(x_t, t) dt
```

Use the ELF repo's base noise scale, not the simplified paper notation:

```text
p_0 = N(0, denoiser_noise_scale^2 I)
denoiser_noise_scale = 2.0 for ELF-B-owt
```

This target isolates the flow likelihood estimator before adding token-decoder ELBO terms.

### 2. Secondary diagnostic: token-level ELBO

Evaluate an explicit token ELBO:

```text
E_q(z | s) [ log p_theta(s | z) + log p_flow(z) - log q(z | s) ]
```

Use a simple Gaussian posterior around the frozen T5 encoding:

```text
q(z | s) = N(encode(s), sigma_q^2 I)
```

Sweep `sigma_q` and measure how much variance comes from posterior/noising samples versus the flow likelihood estimator.

## Implementation Plan

### Step 1: Add a likelihood evaluation script

Create a script, likely `src/eval_elbo_variance.py`, that mirrors the setup in `src/eval.py`:

- Load config `src/configs/training_configs/train_owt_ELF-B.yml`.
- Load tokenizer and frozen T5-small encoder.
- Load checkpoint `embedded-language-flows/ELF-B-owt`.
- Load a held-out or deterministic OpenWebText subset.
- Encode examples into latents using the existing encoder utilities.
- Run distributed evaluation across the TPU pod.

Keep generation metrics out of this script; it should only measure likelihood / ELBO variance.

### Step 2: Implement Hutchinson divergence

For fixed `z` and `t`, estimate:

```text
div v_theta(z, t) ~= eps^T J_v(z, t) eps
```

Implementation requirements:

- Support Gaussian probes.
- Support Rademacher probes.
- Use JAX JVP/VJP based computation.
- Keep the model in deterministic mode.
- Use float32 initially.
- Report per-step divergence mean and variance before integrating.

Validate on toy flows before using ELF:

- Identity velocity with known divergence.
- Linear velocity `v(z) = A z` with known `trace(A)`.
- Zero velocity with divergence zero.

### Step 3: Define deterministic ELF velocity for likelihood

ELF predicts clean embeddings `x_pred`, and the repo converts to velocity with:

```text
v = (x_pred - z) / max(1 - t, t_eps)
```

For the first likelihood diagnostic, make the velocity deterministic:

- Use ODE mode only.
- Disable SDE noise.
- Use no classifier-free guidance.
- Freeze self-conditioning to zeros or another documented deterministic rule.

This is important because the released self-conditioned sampler is history-dependent, while a clean CNF likelihood needs a well-defined vector field `v_theta(z, t)`.

After the clean diagnostic works, separately test the released self-conditioned path as an approximate/discrete-flow diagnostic.

### Step 4: Backward ODE likelihood estimator

For each encoded latent batch:

1. Set `z = x_1`.
2. Initialize `log_det = 0`.
3. Integrate from `t = 1` to `t = 0` using a fixed grid.
4. At each step, estimate `div v_theta(z_t, t)` with Hutchinson probes.
5. Accumulate the divergence integral with Euler first; optionally add midpoint or Heun later.
6. Compute `log p_0(z_0)` under `N(0, 2.0^2 I)`.
7. Return latent log likelihood in nats and normalized nats/token.

Initial grids:

```text
steps = [32, 64, 128, 256]
```

Initial probe counts:

```text
probes_per_step = [1, 2, 4, 8, 16, 32]
```

Initial repeated estimator seeds:

```text
repeats = 8 to 32
```

## Variance Decomposition

Use fixed examples and repeated estimates to separate:

### Hutchinson probe variance

Hold examples and time grid fixed. Change only Hutchinson probe seeds.

Report:

- Mean likelihood.
- Standard deviation across probe seeds.
- Standard error.
- 95% confidence interval.
- Per-example rank stability across repeats.

### ODE discretization sensitivity

Hold examples and probe budget as fixed as possible. Sweep step counts.

Report:

- Mean shift from `32 -> 64 -> 128 -> 256` steps.
- Whether increasing steps reduces bias but increases accumulated probe noise.
- Divergence integral distribution by time region.

### Posterior/noising variance for token ELBO

Only for the secondary token ELBO target.

Hold examples fixed. Change only `q(z | s)` samples.

Report:

- Variance from posterior samples.
- Variance from flow likelihood estimation.
- Variance from decoder term `log p_theta(s | z)`.

### Data variance

Once estimator variance is understood, vary examples.

Report:

- Across-example variance.
- Confidence interval for dataset mean.
- Whether estimator noise is small enough relative to data variance.

## Decision Criteria

Call the estimator "extremely high variance" if, at practical compute budgets such as:

```text
128 ODE steps
16 Hutchinson probes per step
8-32 repeated estimates
```

any of the following hold:

- 95% confidence interval for mean nats/token is wider than roughly `0.1` to `0.25`.
- Per-example likelihood rankings are unstable across estimator seeds.
- Individual estimates contain frequent extreme outliers.
- Increasing probe count changes conclusions materially.
- Increasing ODE steps does not converge before compute becomes impractical.

Call it "usable with caveats" if:

- Confidence intervals shrink predictably with probe count.
- Rankings are stable.
- ODE step sweeps show a clear convergence trend.
- Estimator noise is much smaller than differences we care about between models or checkpoints.

## TPU Run Strategy

Use the normal TPU workflow from `AGENTS.md`.

Local:

```bash
git commit -am "add elbo variance evaluator" && git push
```

Sync all hosts:

```bash
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF && git pull"
```

Launch:

```bash
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF/src && python eval_elbo_variance.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt \
    --steps 32,64,128,256 \
    --probes 1,2,4,8,16,32 \
    --repeats 16 \
    2>&1 | tee /tmp/eval_elbo_variance.log"
```

For long runs:

```bash
gcloud compute tpus tpu-vm ssh elf-elbo-v5p64-spot \
  --project=pivotal-shield-449921-d4 --zone=us-east5-a \
  --worker=all --batch-size=8 \
  --command="cd ~/ELF/src && tmux new -d -s elbo_var 'python eval_elbo_variance.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt \
    --steps 32,64,128,256 \
    --probes 1,2,4,8,16,32 \
    --repeats 16 \
    2>&1 | tee /tmp/eval_elbo_variance.log'"
```

## Expected Outputs

Write machine-readable results:

```text
outputs/elbo_variance/elf_b_owt/results.jsonl
outputs/elbo_variance/elf_b_owt/summary.json
outputs/elbo_variance/elf_b_owt/per_example_estimates.npz
```

Each result row should include:

- Model checkpoint.
- Dataset split/subset identifier.
- Example count.
- ODE step count.
- Probe count.
- Probe distribution.
- Repeat index.
- Mean latent log likelihood.
- Mean nats/token.
- Standard deviation across examples.
- Any NaN/Inf count.
- Runtime.

Summary should include:

- Variance decomposition table.
- Confidence intervals.
- Recommended compute setting, if any.
- Final answer to whether the estimator is too high variance to trust.

## Initial Minimal Run

Before the full sweep, run a cheap smoke test:

```text
examples = 8 or 16
steps = [16, 32]
probes = [1, 4]
repeats = 4
```

Only after signs, shapes, and toy-flow checks pass should we launch the full TPU sweep.
