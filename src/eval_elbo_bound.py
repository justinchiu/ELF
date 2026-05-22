#!/usr/bin/env python
"""Estimate ELF token-level ELBO bias and variance.

This is the evaluator to use for the likelihood-style question going forward.
It estimates a full token-level variational bound for the model

    p_theta(s, z) = p_flow(z) * p_decoder(s | z)

using a Gaussian variational posterior around the frozen T5 embedding,

    q_sigma(z | s) = N(E(s), sigma_q^2 I).

The reported negative ELBO per predicted token is

    E_q [
        latent_flow_nelbo(z)
      + latent_endpoint_const
      + token_ce(s | z)
      + log q_sigma(z | s)
    ].

For the latent flow term, ELF-B-owt uses the path

    z_t = t * z + (1 - t) * sigma * eps,  eps ~ N(0, I), sigma = 2.0.

The default continuous-time weighting is the x-prediction VDM weight

    0.5 * d SNR(t) / dt * ||x_theta - z||^2
      = t / (sigma^2 * (1 - t)^3) * ||x_theta - z||^2,

where SNR(t) = t^2 / ((1 - t)^2 sigma^2).

This is still not the released self-conditioned/SDE sampler likelihood and not
the zero-self-cond CNF diagnostic. The bound depends on the chosen
posterior_sigma; sweep it before treating the number as a model comparison.
Use the default full-support sigmoid-normal time proposal for a strict bound;
truncated-uniform time sampling is a biased diagnostic.
"""

import argparse
import contextlib
import copy
import json
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

import jax


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np
import optax
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import AutoTokenizer

from configs.config import apply_config_overrides, load_config_from_yaml
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_checkpoint, load_encoder_checkpoint
from utils.data_utils import (
    get_dataloader,
    get_pad_token_id,
    load_dataset_split,
    prepare_batch,
)
from utils.encoder_utils import encode_text
from utils.logging_utils import log_for_0
from utils.sampling_utils import add_noise, restore_cond
from utils.train_utils import TrainState


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)


def maybe_initialize_distributed(enabled):
    if not enabled:
        return
    try:
        jax.distributed.initialize()
    except (RuntimeError, ValueError):
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate ELF token-level ELBO bias and variance.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Dataset path to evaluate. Defaults to config.eval_data_path, then config.data_path.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_examples", type=int, default=512)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--streaming", action="store_true",
        help="Stream examples from HF and shard them across hosts.",
    )
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--mc_samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--reference_mc_samples", type=int, default=0,
        help="If >0, compute a higher-MC reference for bias/MSE reporting.",
    )
    parser.add_argument("--reference_repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param_source", choices=["ema", "params"], default="ema")
    parser.add_argument(
        "--weight_mode",
        choices=["vdm_xpred", "none"],
        default="vdm_xpred",
        help=(
            "Latent bound weighting. vdm_xpred is the VDM x-prediction "
            "weight t/(sigma^2*(1-t)^3) on clean-latent prediction error."
        ),
    )
    parser.add_argument(
        "--posterior_sigma", type=float, default=0.2,
        help=(
            "Stddev of q(z|tokens)=N(encoder(tokens), posterior_sigma^2 I) "
            "in normalized latent units. Must be >0 for a true token ELBO."
        ),
    )
    parser.add_argument(
        "--t_min", type=float, default=1e-4,
        help=(
            "Only used with --time_proposal truncated_uniform. Clips t samples to "
            "[t_min, 1 - t_min], which intentionally truncates endpoint mass."
        ),
    )
    parser.add_argument(
        "--time_proposal",
        choices=["sigmoid_normal", "beta", "truncated_uniform"],
        default="sigmoid_normal",
        help=(
            "Proposal q(t) for the latent time integral. sigmoid_normal and beta "
            "have full support on (0,1) and preserve the ELBO; truncated_uniform "
            "is biased low but useful as a diagnostic."
        ),
    )
    parser.add_argument(
        "--time_proposal_loc", type=float, default=0.0,
        help="Mean of the normal in logit time for --time_proposal sigmoid_normal.",
    )
    parser.add_argument(
        "--time_proposal_scale", type=float, default=2.0,
        help="Stddev of the normal in logit time for --time_proposal sigmoid_normal.",
    )
    parser.add_argument(
        "--time_proposal_alpha", type=float, default=1.0,
        help="Alpha parameter for --time_proposal beta.",
    )
    parser.add_argument(
        "--time_proposal_beta", type=float, default=0.5,
        help="Beta parameter for --time_proposal beta; values <1 sample more near t=1.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/elbo_bound/elf_b_owt",
    )
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()


def make_eval_dataset(path, num_examples, start_index, streaming=False, split="train"):
    if streaming:
        if start_index < 0:
            raise ValueError("--streaming does not support negative --start_index")
        if num_examples is None or num_examples < 0:
            raise ValueError("--streaming requires finite --num_examples")
        from datasets import load_dataset as hf_load_dataset

        rank = jax.process_index()
        world = jax.process_count()
        rows = []
        stop_index = start_index + num_examples
        ds = hf_load_dataset(path, split=split, streaming=True)
        for global_idx, ex in enumerate(ds):
            if global_idx < start_index:
                continue
            if global_idx >= stop_index:
                break
            selected_offset = global_idx - start_index
            if selected_offset % world == rank:
                rows.append(ex)
        return rows, None, start_index, stop_index, True

    ds = load_dataset_split(path)
    total = len(ds)
    if start_index < 0:
        start_index = max(0, total + start_index)
    end_index = total if num_examples is None or num_examples < 0 else min(total, start_index + num_examples)
    if start_index < 0 or start_index >= total:
        raise ValueError(f"start_index={start_index} outside dataset length {total}")
    if end_index <= start_index:
        raise ValueError(f"Empty selection: start={start_index}, end={end_index}, total={total}")
    return ds.select(range(start_index, end_index)), total, start_index, end_index, False


def elbo_weight(t, mode, denoiser_noise_scale):
    if mode == "none":
        return jnp.ones_like(t)
    if mode == "vdm_xpred":
        sigma2 = jnp.asarray(denoiser_noise_scale ** 2, dtype=t.dtype)
        return t / (sigma2 * (1.0 - t) ** 3)
    raise ValueError(f"Unknown weight_mode={mode}")


def sample_time_and_integral_weight(
    rng,
    batch_size,
    dtype,
    proposal,
    proposal_loc,
    proposal_scale,
    proposal_alpha,
    proposal_beta,
    t_min,
):
    """Sample t and return 1/q(t) for estimating ∫_0^1 f(t) dt."""
    if proposal == "truncated_uniform":
        t = jax.random.uniform(
            rng,
            (batch_size,),
            minval=jnp.asarray(t_min, dtype=dtype),
            maxval=jnp.asarray(1.0 - t_min, dtype=dtype),
            dtype=dtype,
        )
        return t, jnp.full((batch_size,), 1.0 - 2.0 * t_min, dtype=dtype)

    if proposal == "sigmoid_normal":
        loc = jnp.asarray(proposal_loc, dtype=dtype)
        scale = jnp.asarray(proposal_scale, dtype=dtype)
        y = loc + jax.random.normal(rng, (batch_size,), dtype=dtype) * scale
        t = jax.nn.sigmoid(y)
        log_q_y = -0.5 * ((y - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(
            jnp.asarray(2.0 * jnp.pi, dtype=dtype)
        )
        log_q_t = log_q_y + jax.nn.softplus(-y) + jax.nn.softplus(y)
        return t, jnp.exp(-log_q_t)

    if proposal == "beta":
        alpha = jnp.asarray(proposal_alpha, dtype=dtype)
        beta = jnp.asarray(proposal_beta, dtype=dtype)
        t = jax.random.beta(rng, alpha, beta, (batch_size,), dtype=dtype)
        eps = jnp.finfo(dtype).eps
        t_safe = jnp.clip(t, eps, 1.0 - eps)
        log_q_t = (
            (alpha - 1.0) * jnp.log(t_safe)
            + (beta - 1.0) * jnp.log1p(-t_safe)
            - jsp_special.betaln(alpha, beta)
        )
        return t_safe, jnp.exp(-log_q_t)

    raise ValueError(f"Unknown time proposal: {proposal}")


def _token_loss_sums(per_token_loss, loss_mask):
    loss_mask = loss_mask.astype(per_token_loss.dtype)
    safe_loss = jnp.where(loss_mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
    return (safe_loss * loss_mask).sum(), loss_mask.sum()


def eval_token_elbo_step(
    state,
    encoder_params,
    batch,
    rng,
    encoder_apply_fn,
    config,
    weight_mode,
    posterior_sigma,
    t_min,
    time_proposal,
    time_proposal_loc,
    time_proposal_scale,
    time_proposal_alpha,
    time_proposal_beta,
):
    """Per-device token ELBO MC sample. Returns global psum'd sums."""
    rng = jax.random.fold_in(rng, jax.lax.axis_index(axis_name="batch"))
    posterior_rng, t_rng, noise_rng = jax.random.split(rng, 3)

    encoder_attention_mask = batch["encoder_attention_mask"]
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]
        cond_mask = batch["cond_seq_mask"]
        block_mask = (1 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x_mean = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=encoder_attention_mask,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
    )

    batch_size, _, latent_dim = x_mean.shape
    cond_seq_mask = batch["cond_seq_mask"][:, :, None]
    attention_mask = batch["attention_mask"]
    loss_mask = attention_mask if config.pad_token == "pad" else jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])
    pred_mask = loss_mask[:, :, None].astype(x_mean.dtype)

    posterior_noise = jax.random.normal(posterior_rng, x_mean.shape, dtype=x_mean.dtype)
    x0 = x_mean + jnp.asarray(posterior_sigma, dtype=x_mean.dtype) * posterior_noise * pred_mask

    t, integral_weight = sample_time_and_integral_weight(
        t_rng,
        batch_size,
        x0.dtype,
        time_proposal,
        time_proposal_loc,
        time_proposal_scale,
        time_proposal_alpha,
        time_proposal_beta,
        t_min,
    )
    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)

    z_t = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None]
        z_t = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(z_t), z_t)
        x0 = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(x0), x0)

    model_input = z_t
    if config.self_cond_prob > 0:
        zero_self_cond = restore_cond(jnp.zeros_like(z_t), x0, cond_seq_mask)
        model_input = jnp.concatenate([z_t, zero_self_cond], axis=-1)
    self_cond_cfg_scale = (
        jnp.ones((batch_size,), dtype=x0.dtype)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    x_pred, _ = state.apply_fn(
        {"params": state.params},
        model_input,
        t,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=jnp.array(False),
    )
    sq_error = (x_pred - x0) ** 2
    sqnorm_per_token = jnp.sum(sq_error, axis=-1)
    mse_per_dim_token = jnp.mean(sq_error, axis=-1)
    weight = (
        integral_weight
        * elbo_weight(t, weight_mode, config.denoiser_noise_scale)
    ).reshape(-1, 1)
    latent_nelbo_per_token = weight * sqnorm_per_token
    weighted_mse_per_dim_token = weight * mse_per_dim_token

    latent_nelbo_sum, token_count = _token_loss_sums(latent_nelbo_per_token, loss_mask)
    unweighted_sqnorm_sum, _ = _token_loss_sums(sqnorm_per_token, loss_mask)
    weighted_mse_per_dim_sum, _ = _token_loss_sums(weighted_mse_per_dim_token, loss_mask)
    unweighted_mse_per_dim_sum, _ = _token_loss_sums(mse_per_dim_token, loss_mask)
    weight_token_sum, _ = _token_loss_sums(jnp.broadcast_to(weight, sqnorm_per_token.shape), loss_mask)

    decoder_targets = batch["input_ids"]
    decoder_t = jnp.ones((batch_size,), dtype=x0.dtype)
    decoder_input = (
        jnp.concatenate([x0, jnp.zeros_like(x0)], axis=-1)
        if config.self_cond_prob > 0 else x0
    )
    _, decoder_logits = state.apply_fn(
        {"params": state.params},
        decoder_input,
        decoder_t,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=jnp.array(True),
    )
    log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
    ce = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
    decoder_nll_sum, ce_token_count = _token_loss_sums(ce, loss_mask)

    logq_per_token = -0.5 * latent_dim * (
        1.0 + jnp.log(2.0 * jnp.pi * jnp.asarray(posterior_sigma ** 2, dtype=x0.dtype))
    )
    posterior_logq_sum = logq_per_token * token_count
    endpoint_const_per_token = 0.5 * latent_dim * jnp.log(
        2.0 * jnp.pi * jnp.asarray(config.denoiser_noise_scale ** 2, dtype=x0.dtype)
    )
    latent_endpoint_const_sum = endpoint_const_per_token * token_count
    token_nelbo_sum = (
        latent_nelbo_sum
        + latent_endpoint_const_sum
        + decoder_nll_sum
        + posterior_logq_sum
    )

    metrics = {
        "token_nelbo_sum": token_nelbo_sum,
        "latent_nelbo_sum": latent_nelbo_sum,
        "latent_endpoint_const_sum": latent_endpoint_const_sum,
        "decoder_nll_sum": decoder_nll_sum,
        "posterior_logq_sum": posterior_logq_sum,
        "unweighted_sqnorm_sum": unweighted_sqnorm_sum,
        "weighted_mse_per_dim_sum": weighted_mse_per_dim_sum,
        "unweighted_mse_per_dim_sum": unweighted_mse_per_dim_sum,
        "weight_token_sum": weight_token_sum,
        "time_sum": t.sum(),
        "time_sq_sum": jnp.sum(t ** 2),
        "time_max": jnp.max(t),
        "integral_weight_sum": integral_weight.sum(),
        "integral_weight_sq_sum": jnp.sum(integral_weight ** 2),
        "integral_weight_max": jnp.max(integral_weight),
        "latent_weight_sum": jnp.sum(weight.reshape(-1)),
        "latent_weight_sq_sum": jnp.sum(weight.reshape(-1) ** 2),
        "latent_weight_max": jnp.max(weight),
        "sample_count": jnp.asarray(batch_size, dtype=jnp.float32),
        "token_count": token_count,
        "ce_token_count": ce_token_count,
        "example_count": jnp.asarray(batch["input_ids"].shape[0], dtype=jnp.float32),
    }
    max_keys = {"time_max", "integral_weight_max", "latent_weight_max"}
    return {
        key: (
            jax.lax.pmax(value, axis_name="batch")
            if key in max_keys
            else jax.lax.psum(value, axis_name="batch")
        )
        for key, value in metrics.items()
    }


def estimate_once(
    dataloader,
    state,
    encoder_params,
    p_eval_step,
    config,
    base_rng,
    mc_samples,
    num_local_devices,
):
    totals = {
        "token_nelbo_sum": 0.0,
        "latent_nelbo_sum": 0.0,
        "latent_endpoint_const_sum": 0.0,
        "decoder_nll_sum": 0.0,
        "posterior_logq_sum": 0.0,
        "unweighted_sqnorm_sum": 0.0,
        "weighted_mse_per_dim_sum": 0.0,
        "unweighted_mse_per_dim_sum": 0.0,
        "weight_token_sum": 0.0,
        "time_sum": 0.0,
        "time_sq_sum": 0.0,
        "time_max": 0.0,
        "integral_weight_sum": 0.0,
        "integral_weight_sq_sum": 0.0,
        "integral_weight_max": 0.0,
        "latent_weight_sum": 0.0,
        "latent_weight_sq_sum": 0.0,
        "latent_weight_max": 0.0,
        "sample_count": 0.0,
        "token_count": 0.0,
        "ce_token_count": 0.0,
        "example_count": 0.0,
    }
    host_batches = 0
    for sample_idx in range(mc_samples):
        sample_rng = jax.random.fold_in(base_rng, sample_idx)
        for batch_idx, batch in enumerate(dataloader):
            batch_rng = jax.random.fold_in(sample_rng, batch_idx)
            prepared = prepare_batch(batch, config, batch_rng)
            batch_sharded = shard(prepared)
            rng_sharded = jax.random.split(batch_rng, num_local_devices)
            metrics = p_eval_step(state, encoder_params, batch_sharded, rng_sharded)
            metrics = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), metrics)
            metrics["time_max"] = np.asarray(metrics["time_max"]).max()
            metrics["integral_weight_max"] = np.asarray(metrics["integral_weight_max"]).max()
            metrics["latent_weight_max"] = np.asarray(metrics["latent_weight_max"]).max()
            for key in totals:
                if key in ("time_max", "integral_weight_max", "latent_weight_max"):
                    totals[key] = max(totals[key], float(metrics[key]))
                else:
                    totals[key] += float(metrics[key])
            host_batches += 1
    token_count = max(totals["token_count"], 1.0)
    sample_count = max(totals["sample_count"], 1.0)
    mean_time = totals["time_sum"] / sample_count
    mean_integral_weight = totals["integral_weight_sum"] / sample_count
    mean_latent_weight = totals["latent_weight_sum"] / sample_count
    integral_weight_ess = (
        totals["integral_weight_sum"] ** 2
        / max(totals["integral_weight_sq_sum"], 1e-30)
    )
    latent_weight_ess = (
        totals["latent_weight_sum"] ** 2
        / max(totals["latent_weight_sq_sum"], 1e-30)
    )
    return {
        "token_nelbo_per_token": totals["token_nelbo_sum"] / token_count,
        "token_elbo_per_token": -totals["token_nelbo_sum"] / token_count,
        "latent_nelbo_per_token": totals["latent_nelbo_sum"] / token_count,
        "latent_endpoint_const_per_token": totals["latent_endpoint_const_sum"] / token_count,
        "decoder_nll_per_token": totals["decoder_nll_sum"] / token_count,
        "posterior_logq_per_token": totals["posterior_logq_sum"] / token_count,
        "unweighted_sqnorm_per_token": totals["unweighted_sqnorm_sum"] / token_count,
        "weighted_mse_per_dim_per_token": totals["weighted_mse_per_dim_sum"] / token_count,
        "unweighted_mse_per_dim_per_token": totals["unweighted_mse_per_dim_sum"] / token_count,
        "mean_weight": totals["weight_token_sum"] / token_count,
        "mean_time": mean_time,
        "std_time": max(totals["time_sq_sum"] / sample_count - mean_time ** 2, 0.0) ** 0.5,
        "max_time": totals["time_max"],
        "mean_integral_importance_weight": mean_integral_weight,
        "std_integral_importance_weight": max(
            totals["integral_weight_sq_sum"] / sample_count - mean_integral_weight ** 2,
            0.0,
        ) ** 0.5,
        "max_integral_importance_weight": totals["integral_weight_max"],
        "max_over_mean_integral_importance_weight": (
            totals["integral_weight_max"] / max(mean_integral_weight, 1e-30)
        ),
        "integral_importance_weight_ess": integral_weight_ess,
        "integral_importance_weight_ess_frac": integral_weight_ess / sample_count,
        "mean_latent_weight": mean_latent_weight,
        "std_latent_weight": max(
            totals["latent_weight_sq_sum"] / sample_count - mean_latent_weight ** 2,
            0.0,
        ) ** 0.5,
        "max_latent_weight": totals["latent_weight_max"],
        "max_over_mean_latent_weight": (
            totals["latent_weight_max"] / max(mean_latent_weight, 1e-30)
        ),
        "latent_weight_ess": latent_weight_ess,
        "latent_weight_ess_frac": latent_weight_ess / sample_count,
        "processed_tokens": totals["token_count"],
        "processed_examples": totals["example_count"],
        "host_batches": host_batches,
        **totals,
    }


def summarize_repeats(rows, reference_mean=None):
    estimates = np.asarray([row["token_nelbo_per_token"] for row in rows], dtype=np.float64)
    summary = {
        "mean_token_nelbo_per_token": float(estimates.mean()),
        "mean_token_elbo_per_token": float(-estimates.mean()),
        "std_token_nelbo_per_token": float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
        "stderr_token_nelbo_per_token": float(estimates.std(ddof=1) / np.sqrt(len(estimates))) if len(estimates) > 1 else 0.0,
        "min_token_nelbo_per_token": float(estimates.min()),
        "max_token_nelbo_per_token": float(estimates.max()),
        "repeat_estimates": estimates.tolist(),
    }
    for key in (
        "mean_time",
        "std_time",
        "max_time",
        "mean_integral_importance_weight",
        "std_integral_importance_weight",
        "max_integral_importance_weight",
        "max_over_mean_integral_importance_weight",
        "integral_importance_weight_ess",
        "integral_importance_weight_ess_frac",
        "mean_latent_weight",
        "std_latent_weight",
        "max_latent_weight",
        "max_over_mean_latent_weight",
        "latent_weight_ess",
        "latent_weight_ess_frac",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        reducer = np.max if key.startswith("max_") else np.mean
        summary[key] = float(reducer(values))
    if reference_mean is not None:
        errors = estimates - reference_mean
        bias = float(errors.mean())
        variance = float(estimates.var(ddof=0))
        mse = float(np.mean(errors ** 2))
        summary.update({
            "reference_token_nelbo_per_token": float(reference_mean),
            "mean_error_vs_reference": bias,
            "bias2_vs_reference": bias ** 2,
            "variance_across_repeats": variance,
            "mse_vs_reference": mse,
            "rmse_vs_reference": float(np.sqrt(mse)),
        })
    return summary


def main():
    args = parse_args()
    maybe_initialize_distributed(args.distributed)
    if args.posterior_sigma <= 0:
        raise ValueError("--posterior_sigma must be > 0 for a true token ELBO")
    if not 0.0 <= args.t_min < 0.5:
        raise ValueError("--t_min must satisfy 0 <= t_min < 0.5")
    if args.time_proposal_scale <= 0:
        raise ValueError("--time_proposal_scale must be > 0")
    if args.time_proposal_alpha <= 0 or args.time_proposal_beta <= 0:
        raise ValueError("--time_proposal_alpha and --time_proposal_beta must be > 0")

    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    if args.global_batch_size is not None:
        config.global_batch_size = args.global_batch_size
    config.use_wandb = False

    num_hosts = jax.process_count()
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    if config.global_batch_size is None:
        raise ValueError("global_batch_size must be set in config or via --global_batch_size")
    if config.global_batch_size % num_hosts != 0:
        raise ValueError(f"global_batch_size={config.global_batch_size} not divisible by hosts={num_hosts}")
    local_batch_size = config.global_batch_size // num_hosts
    if local_batch_size % num_local_devices != 0:
        raise ValueError(
            f"local_batch_size={local_batch_size} not divisible by local_devices={num_local_devices}"
        )
    config.batch_size = local_batch_size

    cpu_device = jax.local_devices(backend="cpu")[0] if args.use_cpu else None

    def cpu_ctx():
        return jax.default_device(cpu_device) if args.use_cpu else contextlib.nullcontext()

    log_for_0(
        f"hosts={num_hosts} local_devices={num_local_devices} global_devices={num_devices} "
        f"global_batch={config.global_batch_size} local_batch={local_batch_size}"
    )

    data_path = args.data_path or config.eval_data_path or config.data_path
    if data_path == config.data_path and config.eval_data_path is None:
        log_for_0(
            "No config.eval_data_path is set; evaluating a deterministic slice of config.data_path. "
            "For OWT this is not a released test split."
        )

    dataset, full_len, start_index, end_index, already_sharded = make_eval_dataset(
        data_path,
        args.num_examples,
        args.start_index,
        streaming=args.streaming,
        split=args.split,
    )
    selected_examples = end_index - start_index
    log_for_0(
        f"Dataset full_len={full_len}, selected=[{start_index}, {end_index}), "
        f"selected_examples_global={selected_examples}, streaming={args.streaming}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    log_for_0(f"Loading encoder {config.encoder_model_name}")
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    encoder_params = jax_utils.replicate(encoder_params)

    log_for_0(f"Building model {config.model}")
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    text_enc_dim = encoder_config.d_model
    input_dim = 2 * text_enc_dim if config.self_cond_prob > 0 else text_enc_dim
    with cpu_ctx():
        dummy_x = jnp.ones((1, config.max_length, input_dim))
        dummy_t = jnp.ones((1,))
        dummy_cfg = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = tokenizer.vocab_size
        model = ELF_models[config.model](
            text_encoder_dim=text_enc_dim,
            max_length=config.max_length,
            attn_drop=config.attn_dropout,
            proj_drop=config.proj_dropout,
            num_time_tokens=config.num_time_tokens,
            num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
            vocab_size=vocab_size,
            num_model_mode_tokens=config.num_model_mode_tokens,
            bottleneck_dim=config.bottleneck_dim,
        )
        params = model.init(
            init_rng,
            x=dummy_x,
            t=dummy_t,
            deterministic=True,
            self_cond_cfg_scale=dummy_cfg,
        )
        state = TrainState.create(
            apply_fn=model.apply,
            params=params["params"],
            tx=optax.adamw(1e-4),
            dropout_rng=dropout_rng,
            ema_params1=copy.deepcopy(params["params"]),
        )

    log_for_0(f"Loading checkpoint: {args.checkpoint_path}")
    state, _ = load_checkpoint(args.checkpoint_path, state)
    if args.param_source == "ema":
        state = state.replace(params=state.ema_params1)
    state = jax_utils.replicate(state)

    dataloader = get_dataloader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=not already_sharded,
    )
    p_eval_step = jax.pmap(
        partial(
            eval_token_elbo_step,
            encoder_apply_fn=encoder_model.apply,
            config=config,
            weight_mode=args.weight_mode,
            posterior_sigma=args.posterior_sigma,
            t_min=args.t_min,
            time_proposal=args.time_proposal,
            time_proposal_loc=args.time_proposal_loc,
            time_proposal_scale=args.time_proposal_scale,
            time_proposal_alpha=args.time_proposal_alpha,
            time_proposal_beta=args.time_proposal_beta,
        ),
        axis_name="batch",
    )

    reference_rows = []
    reference_mean = None
    start_time = time.time()
    if args.reference_mc_samples > 0:
        log_for_0(
            f"Reference estimate: mc_samples={args.reference_mc_samples}, "
            f"repeats={args.reference_repeats}"
        )
        for ref_idx in range(args.reference_repeats):
            ref_rng = jax.random.fold_in(rng, 1_000_000 + ref_idx)
            reference_rows.append(
                estimate_once(
                    dataloader,
                    state,
                    encoder_params,
                    p_eval_step,
                    config,
                    ref_rng,
                    args.reference_mc_samples,
                    num_local_devices,
                )
            )
        reference_mean = float(np.mean([row["token_nelbo_per_token"] for row in reference_rows]))
        log_for_0(f"Reference token_nelbo_per_token={reference_mean:.6f}")

    repeat_rows = []
    for repeat_idx in range(args.repeats):
        repeat_rng = jax.random.fold_in(rng, repeat_idx)
        row = estimate_once(
            dataloader,
            state,
            encoder_params,
            p_eval_step,
            config,
            repeat_rng,
            args.mc_samples,
            num_local_devices,
        )
        row["repeat_idx"] = repeat_idx
        repeat_rows.append(row)
        msg = (
            f"[repeat {repeat_idx + 1}/{args.repeats}] "
            f"token_nelbo/tok={row['token_nelbo_per_token']:.6f} "
            f"latent/tok={row['latent_nelbo_per_token']:.6f} "
            f"endpoint/tok={row['latent_endpoint_const_per_token']:.6f} "
            f"decoder/tok={row['decoder_nll_per_token']:.6f} "
            f"logq/tok={row['posterior_logq_per_token']:.6f} "
            f"unweighted_sqnorm/tok={row['unweighted_sqnorm_per_token']:.6f} "
            f"mean_weight={row['mean_weight']:.6f} "
            f"max_t={row['max_time']:.6f} "
            f"max/mean_iw={row['max_over_mean_integral_importance_weight']:.2f} "
            f"ess_iw={row['integral_importance_weight_ess_frac']:.3f} "
            f"max/mean_latent_w={row['max_over_mean_latent_weight']:.2f} "
            f"ess_latent_w={row['latent_weight_ess_frac']:.3f}"
        )
        if reference_mean is not None:
            msg += f" error_vs_reference={row['token_nelbo_per_token'] - reference_mean:+.6f}"
        log_for_0(msg)

    elapsed = time.time() - start_time
    summary = summarize_repeats(repeat_rows, reference_mean=reference_mean)
    result = {
        "metric": "token_level_elbo_bias_variance",
        "primary_question": "Monte Carlo bias and variance of the ELF token-level ELBO estimator",
        "calibration_status": (
            "full_token_elbo_under_gaussian_variational_posterior_and_vdm_xpred_latent_bound"
        ),
        "generative_model": "p(tokens,z)=p_flow(z)*p_decoder(tokens|z)",
        "variational_posterior": "q(z|tokens)=N(T5_encoder(tokens), posterior_sigma^2 I)",
        "latent_endpoint_const": "0.5 * latent_dim * log(2*pi*denoiser_noise_scale^2) per predicted token",
        "non_goals": [
            "released_self_conditioned_sde_sampler_likelihood",
            "zero_self_conditioned_cnf_likelihood",
            "training_objective_eval_loss",
        ],
        "checkpoint_path": args.checkpoint_path,
        "param_source": args.param_source,
        "data_path": data_path,
        "dataset_full_len": full_len,
        "selected_start_index": start_index,
        "selected_end_index": end_index,
        "selected_examples_global": selected_examples,
        "streaming": args.streaming,
        "global_batch_size": config.global_batch_size,
        "hosts": num_hosts,
        "global_devices": num_devices,
        "mc_samples": args.mc_samples,
        "repeats": args.repeats,
        "reference_mc_samples": args.reference_mc_samples,
        "reference_repeats": args.reference_repeats,
        "weight_mode": args.weight_mode,
        "posterior_sigma": args.posterior_sigma,
        "time_proposal": args.time_proposal,
        "time_proposal_loc": args.time_proposal_loc,
        "time_proposal_scale": args.time_proposal_scale,
        "time_proposal_alpha": args.time_proposal_alpha,
        "time_proposal_beta": args.time_proposal_beta,
        "t_min": args.t_min,
        "time_integral_note": (
            "sigmoid_normal samples t from a full-support proposal on (0,1) "
            "and weights by 1/q(t). truncated_uniform samples on "
            "[t_min,1-t_min] and intentionally omits endpoint mass."
        ),
        "denoiser_noise_scale": config.denoiser_noise_scale,
        "elapsed_s": elapsed,
        "summary": summary,
        "repeats_detail": repeat_rows,
        "reference_detail": reference_rows,
    }

    if jax.process_index() == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(args.output_dir) / "summary.json"
        repeats_path = Path(args.output_dir) / "repeats.jsonl"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        with open(repeats_path, "w", encoding="utf-8") as f:
            for row in repeat_rows:
                f.write(json.dumps(row) + "\n")
        log_for_0(json.dumps(result["summary"], indent=2))
        log_for_0(f"Wrote {summary_path} and {repeats_path}")


if __name__ == "__main__":
    main()
