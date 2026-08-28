# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen-Image DiT parallelization — FSDP2 only.

Follows the framework convention: each model directory owns a parallelize.py
that builds proper FSDP kwargs (mesh, reshard_after_forward, comm_fusion,
mp_policy, shard placement overrides) and applies fully_shard per-block + root.
"""
from typing import TYPE_CHECKING

import torch
from torch import nn

from hyper_parallel import fully_shard
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform import get_platform
from hyper_parallel.trainer.utils.logging import get_logger

if TYPE_CHECKING:
    from hyper_parallel.trainer.config import HyperTrainerConfig

logger = get_logger(__name__)
platform = get_platform()

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}


def _world_size() -> int:
    try:
        return platform.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def _resolve_mp_policy(cfg):
    """Build FSDP mixed-precision policy from the YAML config."""
    mp_cfg = cfg.train.mixed_precision
    if not mp_cfg.enabled:
        return None
    output_dtype_str = mp_cfg.output_dtype
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP.get(mp_cfg.param_dtype),
        reduce_dtype=_DTYPE_MAP.get(mp_cfg.reduce_dtype),
        output_dtype=_DTYPE_MAP.get(output_dtype_str) if output_dtype_str else None,
    )


def _build_fsdp_kwargs(module: nn.Module, dp_mesh: DeviceMesh, cfg) -> dict:
    """Assemble FSDP kwargs over dp_mesh."""
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": cfg.train.accelerator.reshard_after_forward,
        "comm_fusion": cfg.train.accelerator.comm_fusion,
    }
    mp_policy = _resolve_mp_policy(cfg)
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    mesh_shape = getattr(dp_mesh, "mesh_shape", None)
    try:
        shard_size = int(mesh_shape[-1]) if mesh_shape else dp_mesh.size()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        shard_size = 1

    shard_dim_overrides = {}
    replicate_params = set()
    if shard_size > 1:
        for _, param in module.named_parameters():
            if param.dim() == 0 or param.size(0) % shard_size == 0:
                continue
            shardable_dim = next(
                (dim for dim in range(1, param.dim()) if param.size(dim) % shard_size == 0),
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

    return fsdp_kwargs


def _should_skip_fsdp(cfg) -> bool:
    """Return True for single-rank with no parallel axes.

    Mixed-precision runs inject a size-1 dp_shard, so we preserve the FSDP
    wrap when an output_dtype is requested — the mp_policy has no non-FSDP
    equivalent.
    """
    accelerator = cfg.train.accelerator
    single_rank = _world_size() == 1
    has_parallel_axis = any(
        int(value or 1) > 1
        for value in (
            accelerator.dp_replicate,
            accelerator.dp_shard,
            accelerator.tp,
            accelerator.cp,
            accelerator.pp,
            accelerator.ep,
        )
    )
    mp_cfg = cfg.train.mixed_precision
    needs_mp_wrap = mp_cfg.enabled and bool(mp_cfg.output_dtype)
    return single_rank and not has_parallel_axis and not needs_mp_wrap


def _apply_ac(model, cfg) -> None:
    """Apply activation checkpointing to transformer blocks."""
    ac_mode = cfg.train.gradient_checkpointing.activation_checkpoint
    if ac_mode in ("off", "none", None, False, ""):
        return
    blocks = list(model.model.transformer_blocks)
    for i, block in enumerate(blocks):
        model.model.transformer_blocks[i] = checkpoint_wrapper(block)
    logger.info_rank0(
        "AC applied to %d Qwen-Image DiT blocks (mode=%s)",
        len(blocks), ac_mode,
    )


def _resolve_fsdp_mesh(mesh):
    """Resolve the FSDP mesh (dp_shard or fsdp alias)."""
    for dim in ("fsdp", "dp_shard"):
        try:
            return mesh[dim]
        except (KeyError, TypeError):
            continue
    return None


def parallelize_qwen_image_dit(
    model: nn.Module,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> nn.Module:
    """Apply FSDP2 (+ optional AC) to a Qwen-Image DiT model.

    Order: AC first (inside FSDP boundary), then FSDP per-block + root.
    """
    if _should_skip_fsdp(cfg):
        logger.info_rank0(
            "Single-rank run has no parallel axes; skipping FSDP wrap.",
        )
        return model

    dp_mesh = _resolve_fsdp_mesh(mesh)
    if dp_mesh is None:
        logger.info_rank0("No FSDP/dp_shard mesh dim; skipping FSDP wrap.")
        return model

    _apply_ac(model, cfg)

    fsdp_kwargs = _build_fsdp_kwargs(model, dp_mesh, cfg)

    blocks = list(model.model.transformer_blocks)
    for block in blocks:
        fully_shard(block, **fsdp_kwargs)
    fully_shard(model.model, **fsdp_kwargs)

    logger.info_rank0(
        "FSDP applied to Qwen-Image DiT: %d blocks + root",
        len(blocks),
    )
    return model


__all__ = ["parallelize_qwen_image_dit"]
