# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen3.5 (dense) parallelization — self-contained per-model logic.

this
file owns Qwen3.5's full TP / AC / FSDP / Prefetch pipeline. There is no
shared "default_parallelize" template — each model implements its own
``parallelize_<name>`` from torch / hyper primitives directly.
"""
import logging
import os

import torch

from hyper_parallel import fully_shard
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.compile_debug import maybe_wrap_compile_backend
from hyper_parallel.models.qwen3_5.model import Qwen3_5ForCausalLM

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str) -> bool:
    """Return whether an environment feature flag is enabled."""
    return os.environ.get(name, "0").lower() in ("1", "true", "on", "yes")


def _env_int(name: str) -> int | None:
    """Return an optional non-negative integer from the environment."""
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return None
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected an integer.", name, raw_value)
        return None
    if value < 0:
        logger.warning("Ignoring invalid %s=%r; expected a non-negative integer.", name, raw_value)
        return None
    return value


def _apply_compile(model) -> None:
    """Compile Qwen3.5 decoder blocks for temporary inductor probing."""
    if not _env_flag_enabled("HYPER_QWEN_BLOCK_COMPILE"):
        return
    if not hasattr(model, "layers"):
        logger.warning("Qwen3.5 block compile enabled but model has no .layers; skipping.")
        return

    backend = os.environ.get("HYPER_QWEN_BLOCK_COMPILE_BACKEND", "inductor")
    backend = maybe_wrap_compile_backend(backend, "qwen3_5_block")
    fullgraph = _env_flag_enabled("HYPER_QWEN_BLOCK_COMPILE_FULLGRAPH")
    layers = list(model.layers)
    max_layers = _env_int("HYPER_QWEN_BLOCK_COMPILE_LAYERS")
    if max_layers is not None:
        layers = layers[:max_layers]
    for layer in layers:
        layer.compile(backend=backend, fullgraph=fullgraph)
    logger.info_rank0(
        "torch.compile applied to %d/%d Qwen3.5 layers: backend=%s fullgraph=%s",
        len(layers), len(model.layers), backend, fullgraph,
    )


def _apply_ac(model, cfg) -> None:
    """Apply ac (internal)."""
    ac_mode = getattr(cfg.train.gradient_checkpointing, "activation_checkpoint", "off")
    if ac_mode in ("off", "none", None, False, ""):
        return
    if not hasattr(model, "layers"):
        logger.warning("AC enabled but model has no .layers; skipping.")
        return

    layers = list(model.layers)
    for i, layer in enumerate(layers):
        model.layers[i] = checkpoint_wrapper(layer)
    logger.info_rank0("AC applied to %d Qwen3.5 layers (mode=%s)", len(layers), ac_mode)


def _apply_fsdp(model, mesh, cfg) -> None:
    """Per-layer + root FSDP wrap. At dp_size==1 still wraps so the gradient
    reduction code path matches dp_size>1 (HCCL all-reduce is a no-op at
    world_size=1 but reduce-order/dtype must match for hyper-1c vs hyper-Nc
    self-consistency at fp32 ULP).
    """
    try:
        dp_mesh = mesh["fsdp"]
    except (KeyError, TypeError):
        try:
            dp_mesh = mesh["dp_shard"]
        except (KeyError, TypeError):
            dp_mesh = mesh

    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": getattr(cfg.train.accelerator, "reshard_after_forward", True),
        "comm_fusion": getattr(cfg.train.accelerator, "comm_fusion", True),
    }

    mp_cfg = getattr(cfg.train, "mixed_precision", None)
    if mp_cfg is not None and getattr(mp_cfg, "enabled", False):
        # When mixed precision is enabled at the YAML level, FSDP2's
        # ``MixedPrecisionPolicy`` is built with all three dtypes (param,
        # reduce, output) set to the same low-precision dtype.
        dtype_map = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        param_dtype = dtype_map.get(getattr(mp_cfg, "param_dtype", "bfloat16"))
        reduce_dtype = dtype_map.get(getattr(mp_cfg, "reduce_dtype", "float32"))
        output_dtype_str = getattr(mp_cfg, "output_dtype", None)
        output_dtype = dtype_map.get(output_dtype_str) if output_dtype_str else None
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
        )

    if not hasattr(model, "layers"):
        logger.warning(
            "Qwen3_5ForCausalLM has no ``.layers`` — root-only FSDP wrap. "
            "This is usually wrong for transformer-style models."
        )
        fully_shard(model, **fsdp_kwargs)
        return

    layers = list(model.layers)
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    logger.info_rank0("FSDP applied to Qwen3.5: %d layers + root", len(layers))


def parallelize_qwen3_5(model: Qwen3_5ForCausalLM, mesh, cfg) -> Qwen3_5ForCausalLM:
    """Apply AC / FSDP to a Qwen3.5 dense model (AC inside FSDP boundary)."""
    # GatedDeltaNet (3/4 layers) doesn't compose with Colwise/RowwiseParallel.
    if getattr(cfg.train.accelerator, "tp", 1) > 1:
        raise NotImplementedError(
            "Qwen3.5 TP for linear-attention layers is not yet implemented. "
            "Set parallel.tp=1 or use a fully full-attention layer_types config."
        )
    if getattr(cfg.train.accelerator, "ep", 1) > 1:
        raise NotImplementedError("Qwen3.5 dense has no experts; set parallel.ep=1.")

    _apply_ac(model, cfg)
    _apply_compile(model)
    _apply_fsdp(model, mesh, cfg)
    return model

__all__ = ["parallelize_qwen3_5"]
