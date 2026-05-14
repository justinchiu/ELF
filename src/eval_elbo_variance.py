#!/usr/bin/env python
"""Latent-CNF likelihood variance evaluator for ELF flow-matching models.

Implements the primary diagnostic from PLAN.md:

    log p_1(x_1) = log p_0(z_0) - ∫_0^1 div v_θ(z_t, t) dt

with a backward Euler ODE integrator, Hutchinson divergence estimator, and a
Gaussian prior p_0 = N(0, denoiser_noise_scale^2 I). The script is set up for
the "Initial Minimal Run" smoke test before the full sweep.

Self-conditioning is disabled at eval (second-half input fixed to zeros, CFG
scale fixed to 1.0, decoder gating off) so the trained model exposes a
well-defined velocity field v_θ(z, t).
"""

import argparse
import contextlib
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer

from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.data_utils import get_pad_token_id
from utils.encoder_utils import encode_text
from utils.logging_utils import log_for_0
from utils.train_utils import TrainState


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


# ============================================================
# Hutchinson + Backward Euler primitives
# ============================================================

def sample_probes(rng, shape, dist):
    if dist == "gaussian":
        return jax.random.normal(rng, shape)
    if dist == "rademacher":
        return 2.0 * jax.random.bernoulli(rng, 0.5, shape).astype(jnp.float32) - 1.0
    raise ValueError(f"Unknown probe dist: {dist}")


def hutchinson_div_single(velocity_fn, z, t, eps):
    """eps: same shape as z. Returns div estimate per batch (B,)."""
    _, jvp_eps = jax.jvp(lambda z_: velocity_fn(z_, t), (z,), (eps,))
    reduce_axes = tuple(range(1, eps.ndim))
    return jnp.sum(eps * jvp_eps, axis=reduce_axes)


def make_velocity_fn(model_apply_fn, params, t_eps, num_self_cond_cfg_tokens):
    """Build a pure (z, t) -> v function with self-cond input zeroed, CFG=1.

    Notes:
      - During training the second half of the input is `x_pred_prev` when
        self-cond is active; with self-cond off, the second half is zeros.
        At eval we always pass zeros to match the "no self-cond" branch.
      - With num_self_cond_cfg_tokens>0 the model expects a CFG scale; passing
        scale=1.0 reduces guidance to zero (see get_sc_guided_v).
      - We never pass decoder_step_active, so the decoder branch is gated off.
    """
    def velocity_fn(z, t):
        z_input = jnp.concatenate([z, jnp.zeros_like(z)], axis=-1)
        kwargs = {"deterministic": True}
        if num_self_cond_cfg_tokens > 0:
            kwargs["self_cond_cfg_scale"] = jnp.ones_like(t)
        out = model_apply_fn({"params": params}, z_input, t, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        t_re = t.reshape(-1, 1, 1)
        return (out - z) / jnp.maximum(1.0 - t_re, t_eps)
    return velocity_fn


def integrate_backward(velocity_fn, z1, n_steps, n_probes, probe_dist, rng):
    """Backward Euler from t=1 to t=0; accumulate ∫ div(v) dt via Hutchinson.

    Returns (z0, integral_div) where integral_div ≈ ∫_0^1 div(v(z_t, t)) dt.
    """
    grid = jnp.linspace(1.0, 0.0, n_steps + 1)
    B = z1.shape[0]

    def step_body(carry, i):
        z, integral = carry
        t_k = grid[i]
        t_kp1 = grid[i + 1]
        dt = t_k - t_kp1
        t_batch = jnp.full((B,), t_k)

        probe_rng = jax.random.fold_in(rng, i)
        eps = sample_probes(probe_rng, (n_probes,) + z.shape, probe_dist)
        div_per_probe = jax.vmap(
            lambda e: hutchinson_div_single(velocity_fn, z, t_batch, e)
        )(eps)
        div_est = jnp.mean(div_per_probe, axis=0)

        v = velocity_fn(z, t_batch)
        z_next = z - v * dt
        integral_next = integral + div_est * dt
        return (z_next, integral_next), None

    (z0, integral), _ = jax.lax.scan(
        step_body, (z1, jnp.zeros((B,))), jnp.arange(n_steps),
    )
    return z0, integral


def log_p_prior(z0, scale, valid_mask=None):
    """Per-batch log N(0, scale^2 I). Only sums over valid positions if mask given.

    z0:         (B, S, C)
    valid_mask: (B, S) with 1 where the position is real, 0 where it's padding.
    Returns (log_p (B,), n_valid_tokens (B,)).
    """
    C = z0.shape[-1]
    log_norm_per_dim = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scale)
    sq = -0.5 * (z0 ** 2) / (scale ** 2)
    per_position = jnp.sum(sq, axis=-1) + C * log_norm_per_dim  # (B, S)
    if valid_mask is not None:
        per_position = per_position * valid_mask
        n_valid = valid_mask.sum(axis=-1)
    else:
        n_valid = jnp.full((z0.shape[0],), z0.shape[1], dtype=jnp.float32)
    return per_position.sum(axis=-1), n_valid


def log_likelihood(velocity_fn, x1, n_steps, n_probes, probe_dist, rng,
                   prior_scale, valid_mask):
    z0, integral = integrate_backward(
        velocity_fn, x1, n_steps, n_probes, probe_dist, rng,
    )
    log_p0, n_valid = log_p_prior(z0, prior_scale, valid_mask)
    return log_p0 - integral, n_valid, z0


# ============================================================
# Toy-flow Hutchinson validation
# ============================================================

def run_toy_checks(num_dims=128, batch=4, num_probes=128, seed=0):
    """Verify Hutchinson on velocities with known closed-form divergence."""
    rng = jax.random.PRNGKey(seed)
    A_rng, z_rng, eps_rng = jax.random.split(rng, 3)
    A = jax.random.normal(A_rng, (num_dims, num_dims)) * 0.1
    z = jax.random.normal(z_rng, (batch, 1, num_dims))
    t = jnp.zeros((batch,))

    cases = [
        ("identity", lambda z_, t_: z_,                                          float(num_dims)),
        ("zero",     lambda z_, t_: jnp.zeros_like(z_),                          0.0),
        ("linear",   lambda z_, t_: jnp.einsum("ij,bsj->bsi", A, z_),            float(jnp.trace(A))),
    ]
    log_for_0("Toy Hutchinson checks (rademacher probes):")
    for name, vf, expected in cases:
        for dist in ["rademacher", "gaussian"]:
            eps = sample_probes(eps_rng, (num_probes, batch, 1, num_dims), dist)
            per_probe = jax.vmap(lambda e: hutchinson_div_single(vf, z, t, e))(eps)
            mean = float(per_probe.mean())
            std_of_mean = float(per_probe.std() / np.sqrt(num_probes))
            log_for_0(
                f"  [{name:8s} | {dist:10s}] expected={expected:+.4f}  "
                f"estimated={mean:+.4f} ± {std_of_mean:.4f}"
            )


# ============================================================
# Data loading
# ============================================================

def load_owt_examples(num_examples, max_length, pad_token_id, split, seed):
    """Pull the first num_examples rows from the OWT-T5 dataset and pad."""
    from datasets import load_dataset
    ds = load_dataset(
        "embedded-language-flows/openwebtext-t5", split=split, streaming=True,
    )
    rows = []
    for ex in ds:
        if len(rows) >= num_examples:
            break
        rows.append(ex)

    input_ids = np.full((num_examples, max_length), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((num_examples, max_length), dtype=np.float32)
    for i, ex in enumerate(rows):
        ids = np.asarray(ex["input_ids"], dtype=np.int32)[:max_length]
        input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1.0
    return input_ids, attention_mask


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=False)
    p.add_argument("--checkpoint_path", type=str, required=False)
    p.add_argument("--num_examples", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128,
                   help="Sequence length used for the latent. Smaller is cheaper.")
    p.add_argument("--steps", type=str, default="16,32")
    p.add_argument("--probes", type=str, default="1,4")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--probe_dist", type=str, default="rademacher",
                   choices=["rademacher", "gaussian"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/elbo_variance/elf_b_owt")
    p.add_argument("--toy_check", action="store_true",
                   help="Run toy-flow Hutchinson validation only and exit.")
    p.add_argument("--data_split", type=str, default="train")
    p.add_argument("--use_cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.toy_check:
        run_toy_checks(num_dims=128, batch=4, num_probes=128, seed=args.seed)
        return

    if not args.config or not args.checkpoint_path:
        raise SystemExit("--config and --checkpoint_path are required unless --toy_check")

    config = load_config_from_yaml(args.config)
    config.max_length = args.max_length

    rng = jax.random.PRNGKey(args.seed)
    cpu_device = jax.local_devices(backend="cpu")[0] if args.use_cpu else None

    def cpu_ctx():
        return jax.default_device(cpu_device) if args.use_cpu else contextlib.nullcontext()

    log_for_0(
        f"hosts={jax.process_count()} local_devices={jax.local_device_count()} "
        f"global_devices={jax.device_count()}"
    )

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(config, "tokenizer_name", None) or config.encoder_model_name
    )
    pad_token_id = get_pad_token_id(tokenizer, getattr(config, "pad_token", "pad"))

    log_for_0(f"Loading encoder {config.encoder_model_name}...")
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)

    log_for_0(f"Building ELF model template (max_length={config.max_length})...")
    text_enc_dim = encoder_config.d_model
    input_dim = 2 * text_enc_dim if config.self_cond_prob > 0 else text_enc_dim

    with cpu_ctx():
        dummy_x = jnp.ones((1, config.max_length, input_dim))
        dummy_t = jnp.ones((1,))
        dummy_cfg = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
        model = ELF_models[config.model](
            text_encoder_dim=text_enc_dim,
            max_length=config.max_length,
            attn_drop=0.0, proj_drop=0.0,
            num_time_tokens=config.num_time_tokens,
            num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
            vocab_size=tokenizer.vocab_size,
            num_model_mode_tokens=config.num_model_mode_tokens,
            bottleneck_dim=config.bottleneck_dim,
        )
        init_rng, _ = jax.random.split(rng)
        init_kwargs = dict(x=dummy_x, t=dummy_t, deterministic=True,
                           self_cond_cfg_scale=dummy_cfg)
        params = model.init(init_rng, **init_kwargs)
        state = TrainState.create(
            apply_fn=model.apply, params=params["params"],
            tx=optax.adamw(1e-4),
            dropout_rng=init_rng,
            ema_params1=copy.deepcopy(params["params"]),
        )

    log_for_0(f"Loading checkpoint {args.checkpoint_path}...")
    state, _ = load_checkpoint(args.checkpoint_path, state)
    eval_params = state.ema_params1

    velocity_fn = make_velocity_fn(
        model.apply, eval_params, config.t_eps, config.num_self_cond_cfg_tokens,
    )

    log_for_0(f"Pulling {args.num_examples} OWT examples (max_length={config.max_length}, split={args.data_split})...")
    input_ids, attention_mask = load_owt_examples(
        args.num_examples, config.max_length, pad_token_id, args.data_split, args.seed,
    )
    input_ids = jnp.asarray(input_ids)
    attention_mask = jnp.asarray(attention_mask)

    log_for_0("Encoding to latents...")
    x0 = encode_text(
        input_ids, attention_mask, encoder_model.apply, encoder_params,
        config.latent_mean, config.latent_std,
    )
    n_tokens = int(attention_mask.sum())
    log_for_0(f"  x0 shape: {x0.shape}, valid tokens: {n_tokens}")

    steps_list = [int(s) for s in args.steps.split(",")]
    probes_list = [int(p) for p in args.probes.split(",")]
    prior_scale = float(config.denoiser_noise_scale)

    log_for_0(
        f"Sweep: steps={steps_list}  probes={probes_list}  repeats={args.repeats}  "
        f"probe_dist={args.probe_dist}  prior_scale={prior_scale}"
    )

    results = []
    per_example_estimates = {}
    for n_steps in steps_list:
        for n_probes in probes_list:
            key = f"steps{n_steps}_probes{n_probes}"
            tic = time.time()
            ll_repeats, z0_norms = [], []
            for r in range(args.repeats):
                seed_r = jax.random.fold_in(rng, r * 10000 + n_steps * 100 + n_probes)
                log_p, n_valid, z0 = log_likelihood(
                    velocity_fn, x0, n_steps, n_probes, args.probe_dist, seed_r,
                    prior_scale, attention_mask,
                )
                log_p.block_until_ready()
                ll_repeats.append((np.asarray(log_p), np.asarray(n_valid)))
                z0_norms.append(float(jnp.sqrt(jnp.mean(z0 ** 2))))
            t_elapsed = time.time() - tic

            ll = np.stack([a for a, _ in ll_repeats], axis=0)  # (R, B)
            nv = np.stack([b for _, b in ll_repeats], axis=0)  # (R, B)
            nats_per_tok = -ll / np.maximum(nv, 1.0)            # NLL/token, (R, B)

            row = {
                "n_steps": int(n_steps),
                "n_probes": int(n_probes),
                "n_repeats": int(args.repeats),
                "n_examples": int(args.num_examples),
                "probe_dist": args.probe_dist,
                "prior_scale": prior_scale,
                "mean_ll": float(ll.mean()),
                "std_ll_across_repeats_per_example": float(ll.std(axis=0).mean()),
                "std_ll_across_examples_per_repeat": float(ll.std(axis=1).mean()),
                "mean_nats_per_token": float(nats_per_tok.mean()),
                "std_nats_per_token_across_repeats": float(nats_per_tok.std(axis=0).mean()),
                "mean_z0_rms": float(np.mean(z0_norms)),
                "nan_count": int(np.isnan(ll).sum()),
                "inf_count": int(np.isinf(ll).sum()),
                "runtime_s": float(t_elapsed),
            }
            log_for_0(
                f"[{key}] mean_ll={row['mean_ll']:+.2f}  "
                f"std_ll(repeats,/ex)={row['std_ll_across_repeats_per_example']:.3f}  "
                f"nats/tok={row['mean_nats_per_token']:.3f} "
                f"± {row['std_nats_per_token_across_repeats']:.3f}  "
                f"z0_rms={row['mean_z0_rms']:.3f}  "
                f"runtime={t_elapsed:.1f}s"
            )
            results.append(row)
            per_example_estimates[key] = ll

    if jax.process_index() == 0:
        with open(os.path.join(args.output_dir, "results.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        np.savez(
            os.path.join(args.output_dir, "per_example_estimates.npz"),
            **per_example_estimates,
        )
        log_for_0(
            f"Wrote {args.output_dir}/"
            f"{{results.jsonl, summary.json, per_example_estimates.npz}}"
        )


if __name__ == "__main__":
    main()
