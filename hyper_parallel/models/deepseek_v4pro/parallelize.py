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
"""DeepSeek-V4Pro parallelization.

The first HyperParallel port keeps the model path FSDP2-first. Tensor, context
and expert parallelism are intentionally rejected for now so the model can be
validated quickly in a small smoke run.
"""
# pylint: disable=forbidden-backend-import,missing-public-type-hints
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from hyper_parallel import fully_shard
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy

if TYPE_CHECKING:
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_mesh(mesh) -> "DeviceMesh | None":
    for dim_name in ("fsdp", "dp_shard"):
        try:
            return mesh[dim_name]
        except (KeyError, TypeError, AttributeError):
            continue
    return None


def _resolve_mp_policy(cfg):
    mp_cfg = cfg.train.mixed_precision
    if not mp_cfg.enabled:
        return None
    output_dtype_str = mp_cfg.output_dtype
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP.get(mp_cfg.param_dtype),
        reduce_dtype=_DTYPE_MAP.get(mp_cfg.reduce_dtype),
        output_dtype=_DTYPE_MAP.get(output_dtype_str) if output_dtype_str else None,
    )


def _build_fsdp_kwargs(module: nn.Module, dp_mesh, cfg) -> dict:
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": cfg.train.accelerator.reshard_after_forward,
        "comm_fusion": cfg.train.accelerator.comm_fusion,
    }
    mp_policy = _resolve_mp_policy(cfg)
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if cfg.train.accelerator.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    shard_size = dp_mesh.size() if dp_mesh is not None else 1
    shard_dim_overrides: dict[int, int] = {}
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
        def _shard_placement_fn(param):
            dim = shard_dim_overrides.get(id(param))
            return None if dim is None else Shard(dim)

        fsdp_kwargs["shard_placement_fn"] = _shard_placement_fn
    if replicate_params:
        fsdp_kwargs["replicate_params"] = replicate_params
    return fsdp_kwargs


def _apply_ac(model: nn.Module, cfg) -> None:
    ac_mode = getattr(cfg.train.gradient_checkpointing, "activation_checkpoint", "off")
    if ac_mode in ("off", "none", None, False, ""):
        return
    if not hasattr(model, "layers"):
        logger.warning("Activation checkpointing requested but model has no layers; skipping.")
        return
    for idx, layer in enumerate(list(model.layers)):
        model.layers[idx] = checkpoint_wrapper(layer)
    logger.info("Applied activation checkpointing to %d DeepSeek-V4Pro layers", len(model.layers))


def parallelize_deepseek_v4pro(model, mesh, cfg):
    """Apply FSDP2 wrapping to DeepSeek-V4Pro."""
    if getattr(cfg.train.accelerator, "tp", 1) > 1:
        raise NotImplementedError(
            "DeepSeek-V4Pro TP is not wired in the first HyperParallel port; set train.accelerator.tp=1."
        )
    if getattr(cfg.train.accelerator, "cp", 1) > 1:
        raise NotImplementedError(
            "DeepSeek-V4Pro CP is not wired in the first HyperParallel port; set train.accelerator.cp=1."
        )
    if getattr(cfg.train.accelerator, "ep", 1) > 1:
        raise NotImplementedError(
            "DeepSeek-V4Pro EP is not wired in the first HyperParallel port; set train.accelerator.ep=1."
        )

    dp_mesh = _resolve_mesh(mesh)
    if dp_mesh is None or dp_mesh.size() <= 1:
        _apply_ac(model, cfg)
        return model

    _apply_ac(model, cfg)
    fsdp_kwargs = _build_fsdp_kwargs(model, dp_mesh, cfg)
    if hasattr(model, "layers"):
        for layer in list(model.layers):
            fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    logger.info("Applied FSDP2 to DeepSeek-V4Pro over mesh %s", getattr(dp_mesh, "mesh_dim_names", None))
    return model


__all__ = ["parallelize_deepseek_v4pro"]
