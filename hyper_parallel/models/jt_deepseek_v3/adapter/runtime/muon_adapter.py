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
"""Projection layouts and model updates around the public Muon optimizer."""

import math
from functools import partial
from typing import Any, Callable

import torch
import torch_npu

from hyper_parallel.core.optimizer.muon import NSInputTransform
from hyper_parallel.components.optim.local_parameters import bind_local_parameters
from .jt_optimizer import clip_qk


def projection_transform(config: Any, name: str, tensor: torch.Tensor) -> NSInputTransform | None:
    """Optimize fused projections separately while keeping registered storage.

    Args:
        config: Model dimensions defining fused projection boundaries.
        name: Parameter name assigned by the public optimizer.
        tensor: Full logical update matrix, or a batch of local expert matrices.
    """
    if name.endswith("linear_qkv.weight"):
        parts = list(tensor.split((config.q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim)))
        restore = lambda updates, output: output.copy_(torch.cat(updates))
    elif name.endswith(("q_b_proj.weight", "kv_b_proj.weight")):
        dims = (config.qk_nope_head_dim,
                config.qk_rope_head_dim if name.endswith("q_b_proj.weight") else config.v_head_dim)
        expanded = tensor.reshape(-1, sum(dims), tensor.shape[-1])
        parts = [p.reshape(-1, tensor.shape[-1]) for p in expanded.split(dims, 1)]

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Restore separate head projections into the fused parameter layout."""
            output.copy_(torch.cat([p.reshape(-1, dim, tensor.shape[-1])
                                   for p, dim in zip(updates, dims)], 1).reshape_as(output))
    elif name.endswith("experts.gate_up_proj"):
        parts = [p.mT for p in tensor.chunk(2, dim=1)]
        restore = lambda updates, output: output.copy_(torch.cat([p.mT for p in updates], dim=1))
    elif name.endswith("experts.down_proj"):
        parts = [tensor.mT]
        restore = lambda updates, output: output.copy_(updates[0].mT)
    else:
        return None
    return NSInputTransform(parts, restore)


@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Apply coupled model updates once after all public optimizer leaves."""
    del optimizer, args, kwargs
    model.jt_optimizer_metrics = clip_qk(model, threshold)
    cfg = model.config
    if cfg.moe_router_enable_expert_bias:
        for module in model.modules():
            if getattr(module, "expert_load", None) is not None:
                direction = (1 / cfg.n_routed_experts - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(direction, alpha=cfg.moe_router_bias_update_rate)
                module.expert_load.zero_()


def _take_metrics(model: torch.nn.Module) -> dict:
    result = model.jt_optimizer_metrics
    model.jt_optimizer_metrics = {}
    return result


def _configure(model: torch.nn.Module, optimizer: Any, *, qk_clip_threshold: float) -> None:
    """Bind local parameter layouts and register model-owned update hooks."""
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    bind_local_parameters(model, optimizer, replicated_names=model.jt_replicated_names)
    optimizer.optimizers_dict["muon"].ns_transform_fn = partial(projection_transform, model.config)
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    model.jt_optimizer_metrics = {}
    optimizer.get_logging_metrics = partial(_take_metrics, model)


def optimizer_adapter(*, qk_clip_threshold: float) -> Callable:
    """Return the public Muon adapter with an explicit QK clipping threshold.

    Args:
        qk_clip_threshold: Finite positive bound for per-head attention logits.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    return partial(_configure, qk_clip_threshold=qk_clip_threshold)
