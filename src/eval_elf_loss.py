#!/usr/bin/env python
"""Evaluate the ELF training objective on a dataset shard.

This computes the same two conditional losses used by train_step.py, without
gradients:

  expected_loss = (1 - decoder_prob) * denoiser_l2 + decoder_prob * decoder_ce

The evaluator is written for multi-host TPU runs. Dataset distribution is
handled by utils.data_utils.get_dataloader(..., distributed=True), which uses a
DistributedSampler keyed by jax.process_index()/jax.process_count().
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

import jax


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
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
from utils.sampling_utils import (
    add_noise,
    net_out_to_v_x,
    restore_cond,
    sample_cfg_scale,
    sample_timesteps,
)
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
    parser = argparse.ArgumentParser(description="Evaluate ELF training loss.")
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
    parser.add_argument(
        "--num_examples", type=int, default=512,
        help="Number of examples to evaluate after start_index. Use -1 for all examples.",
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="Global dataset start index. Negative values count from the end.",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Stream examples from HF and shard them across hosts without downloading the full dataset.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param_source", choices=["ema", "params"], default="ema")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()


def make_eval_dataset(path, num_examples, start_index, streaming=False, split="train"):
    if streaming:
        if start_index < 0:
            raise ValueError("--streaming does not support negative --start_index")
        if num_examples is None or num_examples < 0:
            raise ValueError("--streaming requires a finite --num_examples")
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
    if num_examples is None or num_examples < 0:
        end_index = total
    else:
        end_index = min(total, start_index + num_examples)
    if start_index < 0 or start_index >= total:
        raise ValueError(f"start_index={start_index} outside dataset length {total}")
    if end_index <= start_index:
        raise ValueError(f"Empty selection: start={start_index}, end={end_index}, total={total}")
    return ds.select(range(start_index, end_index)), total, start_index, end_index, False


def _token_loss_sums(per_token_loss, loss_mask):
    loss_mask = loss_mask.astype(per_token_loss.dtype)
    safe_loss = jnp.where(loss_mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
    return (safe_loss * loss_mask).sum(), loss_mask.sum()


def eval_loss_step(state, encoder_params, batch, rng, encoder_apply_fn, config):
    """Per-device loss eval. Returns global psum'd numerators/denominators."""
    t_eps = config.t_eps
    latent_mean, latent_std = config.latent_mean, config.latent_std
    decoder_prob = config.decoder_prob

    rng = jax.random.fold_in(rng, jax.lax.axis_index(axis_name="batch"))
    (
        t_rng,
        noise_rng,
        self_cond_mask_rng,
        self_cond_cfg_rng,
        model_dropout_rng,
        decoder_rng,
    ) = jax.random.split(rng, 6)
    decoder_lambda_rng, decoder_noise_rng = jax.random.split(decoder_rng)

    encoder_attention_mask = batch["encoder_attention_mask"]
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]
        cond_mask = batch["cond_seq_mask"]
        block_mask = (1 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=encoder_attention_mask,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    batch_size, seq_length = x0.shape[0], x0.shape[1]
    t = sample_timesteps(
        t_rng,
        batch_size,
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )
    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)

    cond_seq_mask = batch["cond_seq_mask"][:, :, None]
    attention_mask = batch["attention_mask"]
    loss_mask = attention_mask if config.pad_token == "pad" else jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None]
        denoiser_z = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(denoiser_z), denoiser_z)
        x0 = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(x0), x0)

    if config.self_cond_prob > 0:
        use_self_cond_mask = (
            (jax.random.uniform(self_cond_mask_rng, (batch_size,)) < config.self_cond_prob)
            .reshape(-1, 1, 1)
            .astype(x0.dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            self_cond_cfg_rng,
            batch_size,
            cfg_min=config.self_cond_cfg_min,
            cfg_max=config.self_cond_cfg_max,
        )
    else:
        self_cond_cfg_scale = None

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / jnp.maximum(1 - t_expanded, t_eps)

    def get_z_input(params, z, t_input, self_cond_cfg_input, x_tokens):
        if config.self_cond_prob == 0:
            return z
        z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_seq_mask)
        z_with_zeros = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_init = state.apply_fn(
            {"params": params},
            z_with_zeros,
            t_input,
            deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_input,
        )
        net_out_init = jax.lax.stop_gradient(net_out_init)
        _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, t_eps)
        x_pred_init = restore_cond(x_pred_init, x_tokens, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.astype(z.dtype)
        x_pred_cond = restore_cond(x_pred_cond, x_tokens, cond_seq_mask)
        return jnp.concatenate([z, x_pred_cond], axis=-1)

    def get_sc_cond_and_uncond(params, z, t_input, cond_mask, x_tokens):
        kwargs = {"self_cond_cfg_scale": self_cond_cfg_scale, "deterministic": True}
        if config.self_cond_prob == 0:
            net_out_uncond = state.apply_fn({"params": params}, z, t_input, **kwargs)
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
            return v_uncond, v_uncond

        z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_mask)
        z_input_uncond = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_uncond = state.apply_fn({"params": params}, z_input_uncond, t_input, **kwargs)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = jnp.concatenate([z, x_uncond], axis=-1)
        net_out_cond = state.apply_fn({"params": params}, z_input_cond, t_input, **kwargs)
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, t_eps)
        return v_cond, v_uncond

    def get_v_target(params, z, t_input, base_v_target, x_tokens):
        if config.num_self_cond_cfg_tokens <= 0:
            return base_v_target
        v_cond, v_uncond = get_sc_cond_and_uncond(
            params, z, t_input, cond_mask=cond_seq_mask, x_tokens=x_tokens
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = jnp.where(use_self_cond_mask, sc_guidance, jnp.zeros_like(sc_guidance))
        return jax.lax.stop_gradient(base_v_target + sc_guidance)

    params = state.params

    denoiser_input = get_z_input(
        params,
        denoiser_z,
        t,
        self_cond_cfg_input=self_cond_cfg_scale,
        x_tokens=x0,
    )
    net_out, _ = state.apply_fn(
        {"params": params},
        denoiser_input,
        t,
        deterministic=False,
        rngs={"dropout": model_dropout_rng},
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=jnp.array(False),
    )
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, t_eps)
    v_final_target = get_v_target(params, denoiser_z, t, base_v_target=v_target, x_tokens=x0)
    l2_per_token = jnp.mean((v_pred - v_final_target) ** 2, axis=-1)
    l2_sum, token_count = _token_loss_sums(l2_per_token, loss_mask)

    decoder_targets = batch["input_ids"]
    decoder_z_vals = (
        jax.random.normal(decoder_lambda_rng, (batch_size * seq_length,))
        * config.decoder_p_std
        + config.decoder_p_mean
    )
    decoder_lambda_t = jax.nn.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = (
        jax.random.normal(decoder_noise_rng, x0.shape, dtype=x0.dtype)
        * config.decoder_noise_scale
    )
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise
    decoder_t = jnp.ones_like(t)
    decoder_input = (
        jnp.concatenate([decoder_z, jnp.zeros_like(decoder_z)], axis=-1)
        if config.self_cond_prob > 0
        else decoder_z
    )
    _, decoder_logits = state.apply_fn(
        {"params": params},
        decoder_input,
        decoder_t,
        deterministic=False,
        rngs={"dropout": model_dropout_rng},
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=jnp.array(True),
    )
    log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
    ce = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
    ce_sum, ce_token_count = _token_loss_sums(ce, loss_mask)

    metrics = {
        "l2_sum": l2_sum,
        "ce_sum": ce_sum,
        "token_count": token_count,
        "ce_token_count": ce_token_count,
        "example_count": jnp.asarray(batch["input_ids"].shape[0], dtype=jnp.float32),
        "expected_loss_sum": (
            (1.0 - decoder_prob) * (l2_sum / jnp.maximum(token_count, 1.0))
            + decoder_prob * (ce_sum / jnp.maximum(ce_token_count, 1.0))
        ),
    }
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name="batch"), metrics)
    return metrics


def main():
    args = parse_args()
    maybe_initialize_distributed(args.distributed)

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

    log_for_0(f"Loading eval dataset: {data_path}")
    dataset, full_len, start_index, end_index, already_sharded = make_eval_dataset(
        data_path,
        args.num_examples,
        args.start_index,
        streaming=args.streaming,
        split=args.split,
    )
    selected_examples = end_index - start_index
    log_for_0(
        f"Dataset full_len={full_len}, selected=[{start_index}, {end_index}) "
        f"selected_examples_global={selected_examples}, "
        f"local_examples_on_rank0={len(dataset) if jax.process_index() == 0 else 'n/a'}, "
        f"streaming={args.streaming}"
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
    p_eval_loss_step = jax.pmap(
        partial(eval_loss_step, encoder_apply_fn=encoder_model.apply, config=config),
        axis_name="batch",
    )

    totals = {
        "l2_sum": 0.0,
        "ce_sum": 0.0,
        "token_count": 0.0,
        "ce_token_count": 0.0,
        "example_count": 0.0,
    }
    start_time = time.time()
    host_batches = 0
    for repeat_idx in range(args.repeats):
        repeat_rng = jax.random.fold_in(rng, repeat_idx)
        for batch_idx, batch in enumerate(dataloader):
            batch_rng = jax.random.fold_in(repeat_rng, batch_idx)
            prepared = prepare_batch(batch, config, batch_rng)
            batch_sharded = shard(prepared)
            rng_sharded = jax.random.split(batch_rng, num_local_devices)
            metrics = p_eval_loss_step(state, encoder_params, batch_sharded, rng_sharded)
            metrics = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), metrics)
            for key in totals:
                totals[key] += float(metrics[key])
            host_batches += 1

    elapsed = time.time() - start_time
    token_count = max(totals["token_count"], 1.0)
    ce_token_count = max(totals["ce_token_count"], 1.0)
    l2_loss = totals["l2_sum"] / token_count
    ce_loss = totals["ce_sum"] / ce_token_count
    expected_loss = (1.0 - config.decoder_prob) * l2_loss + config.decoder_prob * ce_loss
    result = {
        "checkpoint_path": args.checkpoint_path,
        "param_source": args.param_source,
        "data_path": data_path,
        "dataset_full_len": full_len,
        "selected_start_index": start_index,
        "selected_end_index": end_index,
        "selected_examples_global": selected_examples,
        "streaming": args.streaming,
        "processed_examples_global_times_repeats": totals["example_count"],
        "processed_tokens_global_times_repeats": totals["token_count"],
        "global_batch_size": config.global_batch_size,
        "hosts": num_hosts,
        "global_devices": num_devices,
        "repeats": args.repeats,
        "decoder_prob": config.decoder_prob,
        "denoiser_l2": l2_loss,
        "decoder_ce": ce_loss,
        "expected_elf_loss": expected_loss,
        "host_batches": host_batches,
        "elapsed_s": elapsed,
    }

    if jax.process_index() == 0:
        log_for_0(json.dumps(result, indent=2))
        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            log_for_0(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
