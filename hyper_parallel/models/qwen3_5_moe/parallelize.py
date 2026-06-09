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
"""Qwen3.5-MoE parallelization — self-contained per-model logic.

There
is no shared "default_parallelize" template — this file owns Qwen3.5-MoE's
full AC / FSDP / Prefetch pipeline. TP and EP are explicitly rejected (the
generic ``ParallelStyle``-based path produces incorrect numerics for grouped
experts, and full EP requires per-model dispatch/combine wiring that lives
here, not in a shared helper).
"""
import logging

import torch

from hyper_parallel import fully_shard
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM

logger = logging.getLogger(__name__)


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
    logger.info_rank0("AC applied to %d Qwen3.5-MoE layers (mode=%s)", len(layers), ac_mode)


def _apply_fsdp(model, mesh, cfg) -> None:
    """Per-layer + root FSDP wrap.

    Wraps each Qwen3.5-MoE decoder layer (attention + MoE expert block) as
    its own FSDP unit so per-layer all-gather scheduling stays stable, and
    keeps the reduction code path identical at ``dp_size == 1`` and
    ``dp_size > 1``.
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
        # mp_policy: low-precision forward + caller-chosen reduce dtype.
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

    # Override default Shard(0) to Shard(1) for params whose dim-0 isn't
    # divisible by world_size but dim-1 is (e.g. ``shared_expert_gate.weight``
    # shape ``(1, hidden)``). This keeps them in the normal sharded path and
    # avoids the ``replicate_params`` route, whose ``to_sharded`` copy-back
    # introduces a no-op ``dst.copy_(src)`` that costs ±1 bf16 ULP per step
    # on NPU. Params that fit neither dim fall back to ``replicate_params``.
    try:
        world_size = dp_mesh.size() if hasattr(dp_mesh, "size") else 1
    except (AttributeError, RuntimeError):
        world_size = 1
    shard_dim_overrides: dict = {}
    replicate_params = set()
    if world_size > 1:
        for _, param in model.named_parameters():
            if param.dim() == 0:
                continue
            if param.size(0) % world_size == 0:
                continue
            shardable_dim = next(
                (d for d in range(1, param.dim()) if param.size(d) % world_size == 0),
                None,
            )
            if shardable_dim is not None:
                shard_dim_overrides[id(param)] = shardable_dim
            else:
                replicate_params.add(param)
    if shard_dim_overrides:
        overrides = shard_dim_overrides

        def _shard_placement_fn(param):
            dim = overrides.get(id(param))
            return Shard(dim) if dim is not None else None
        fsdp_kwargs["shard_placement_fn"] = _shard_placement_fn
    if replicate_params:
        fsdp_kwargs["replicate_params"] = replicate_params

    if not hasattr(model, "layers"):
        logger.warning(
            "Qwen3_5MoeForCausalLM has no ``.layers`` — root-only FSDP wrap. "
            "This is usually wrong for transformer-style models."
        )
        fully_shard(model, **fsdp_kwargs)
        return

    layers = list(model.layers)
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    logger.info_rank0(
        "FSDP applied to Qwen3.5-MoE: %d layers + root  replicate=%d",
        len(layers), len(replicate_params),
    )


def parallelize_qwen3_5_moe(
    model: Qwen3_5MoeForCausalLM, mesh, cfg
) -> Qwen3_5MoeForCausalLM:
    """Apply AC / FSDP to a Qwen3.5-MoE model.

    Order: AC before FSDP (checkpoint_wrapper sits inside the FSDP boundary).
    """
    # Generic Colwise-on-experts is numerically incorrect; full EP needs
    # per-model dispatch/combine wiring (not implemented).
    if getattr(cfg.train.accelerator, "tp", 1) > 1:
        raise NotImplementedError(
            "Qwen3.5-MoE TP is not implemented. Set parallel.tp=1."
        )
    if getattr(cfg.train.accelerator, "ep", 1) > 1:
        raise NotImplementedError(
            "Qwen3.5-MoE EP requires per-model dispatch / combine wiring "
            "that is not yet implemented. Set parallel.ep=1."
        )

    _apply_ac(model, cfg)
    _apply_fsdp(model, mesh, cfg)
    return model

__all__ = ["parallelize_qwen3_5_moe"]
