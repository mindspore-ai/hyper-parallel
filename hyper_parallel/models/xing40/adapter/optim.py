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
"""Muon for Xing4.0: AdamW decay groups and QK clipping after each step."""

# This adapter uses the Torch runtime, like the existing optimizer builders.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard

__all__ = ["build_optimizer", "clip_qk"]


def build_optimizer(*, model: nn.Module, muon_config: dict[str, Any], adamw_config: dict[str, Any],
                    qk_clip_threshold: float = 0.0) -> Muon:
    """Build the framework Muon with the MindFormers AdamW decay groups.

    Muon orthogonalizes every matrix parameter whole, as the framework does for any model;
    this adapter does not reproduce MindFormers' per-block or per-head Newton-Schulz
    splits. As in MindFormers, AdamW does not decay one-dimensional parameters (norm
    weights, mHC gates and biases) or biases.

    Args:
        model: Model whose parameters are optimized.
        muon_config: Muon hyperparameters.
        adamw_config: Hyperparameters of the AdamW fallback.
        qk_clip_threshold: Maximum attention logit enforced by QK clipping after every
            optimizer step; 0 disables clipping.

    Returns:
        The optimizer builder whose chained optimizer the Trainer steps.
    """
    # The AdamW grouping matches name fragments; full names select exactly the 1-D parameters.
    no_decay = ["bias", *(name for name, parameter in model.named_parameters() if parameter.ndim == 1)]
    builder = Muon(muon_config=dict(muon_config), adamw_config=dict(adamw_config), model=model,
                   no_decay_params=no_decay)
    if qk_clip_threshold > 0:
        # The MLA modules: the NPU Xing40MLAAttention replacement or, unreplaced, Transformers attention.
        attention = [module for module in model.modules()
                     if hasattr(module, "q_b_proj") and hasattr(module, "kv_b_proj")]
        chained = builder.get_optimizer()
        step_leaves = chained.step

        def step_and_clip(closure: Any = None) -> None:
            """Step every leaf optimizer, then clip the attention logits before any main-weight copy-back."""
            step_leaves(closure)
            clip_qk(attention, qk_clip_threshold)

        # The Trainer keeps this chained optimizer; optimizer-state swap replaces its leaves, not it.
        chained.step = step_and_clip
    return builder


@torch.no_grad()
def clip_qk(attention: Sequence[nn.Module], threshold: float) -> None:
    """Rescale the query/key projections of heads whose maximum logit exceeded ``threshold``.

    MuonClip: with ``s = threshold / max(max_logit, threshold)`` per head, the query
    no-RoPE rows and the key no-RoPE rows scale by ``sqrt(s)`` and the query RoPE rows
    by ``s``; value rows are unchanged. Each rank writes the maxima of the heads it
    computed at their global head positions, so one MAX all-reduce over every rank gives
    each head's maximum, with or without tensor parallelism. The recorded maxima are then
    cleared.

    Args:
        attention: MLA modules; the NPU fused attention records their ``max_logits`` each step.
        threshold: Maximum attention logit.

    Raises:
        RuntimeError: If a module recorded no maxima, as without the NPU fused attention.
        ValueError: If a module's maxima cover only part of its heads but its query
            projection is not sharded into contiguous head blocks along the ``tp`` mesh dimension.
    """
    maxima = [getattr(module, "max_logits", None) for module in attention]
    if any(maximum is None for maximum in maxima):
        raise RuntimeError("QK clipping requires the logit maxima recorded by the NPU fused attention")
    maxima = [_at_global_heads(module, maximum) for module, maximum in zip(attention, maxima)]
    flat = torch.cat(maxima)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flat, op=dist.ReduceOp.MAX)
    for module, maximum in zip(attention, flat.split([maximum.numel() for maximum in maxima])):
        scale = threshold / maximum.clamp_min(threshold)
        _scale_head_rows(module.q_b_proj.weight, scale, module.qk_nope_head_dim, scale, module.qk_rope_head_dim)
        _scale_head_rows(module.kv_b_proj.weight, scale, module.qk_nope_head_dim, torch.ones_like(scale),
                         module.v_head_dim)
        module.max_logits = None


def _at_global_heads(module: nn.Module, maximum: torch.Tensor) -> torch.Tensor:
    """Return ``maximum`` at its global head positions, ``-inf`` for heads another rank computed.

    Tensor parallelism shards ``q_b_proj`` rows by contiguous head blocks: the rank at
    coordinate ``r`` of the ``tp`` mesh dimension computes heads ``[r * n, (r + 1) * n)``.
    Without it every rank computes every head.

    Args:
        module: MLA module whose ``q_b_proj`` defines the global head count.
        maximum: Logit maxima of the heads this rank computed.

    Returns:
        The maxima of all the module's heads.

    Raises:
        ValueError: If ``maximum`` covers part of the heads but the query projection is not
            sharded into contiguous head blocks along the ``tp`` mesh dimension.
    """
    weight = module.q_b_proj.weight
    heads = weight.shape[0] // (module.qk_nope_head_dim + module.qk_rope_head_dim)
    if maximum.numel() == heads:
        return maximum
    names = (weight.device_mesh.mesh_dim_names or ()) if isinstance(weight, DTensor) else ()
    tp_dim = names.index("tp") if "tp" in names else None
    if (tp_dim is None or weight.placements[tp_dim] != Shard(0)
            or maximum.numel() * weight.device_mesh.size(tp_dim) != heads):
        raise ValueError("Partial QK maxima require contiguous projection-head TP shards")
    placed = maximum.new_full((heads,), float("-inf"))
    start = weight.device_mesh.get_local_rank(tp_dim) * maximum.numel()
    placed[start:start + maximum.numel()] = maximum
    return placed


def _scale_head_rows(weight: torch.Tensor, scale: torch.Tensor, nope_dim: int, tail_scale: torch.Tensor,
                     tail_dim: int) -> None:
    """Scale each head's ``nope_dim`` rows by ``sqrt(scale)`` and its next ``tail_dim`` rows by ``tail_scale``.

    As in MindFormers, only the FP32 main copy is scaled when there is one; the optimizer then
    copies it back into the parameter.
    """
    factors = torch.cat((scale.sqrt()[:, None].expand(-1, nope_dim), tail_scale[:, None].expand(-1, tail_dim)),
                        dim=-1).reshape(-1, 1)
    tensor = getattr(weight, "main_param", None)
    tensor = weight if tensor is None else tensor
    if isinstance(tensor, DTensor):
        local = distribute_tensor(factors.to(tensor.dtype), tensor.device_mesh, tensor.placements).to_local()
        tensor.to_local().mul_(local)
    else:
        tensor.mul_(factors.to(tensor.dtype))
