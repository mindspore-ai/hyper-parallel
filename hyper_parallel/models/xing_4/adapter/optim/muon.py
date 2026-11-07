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
"""Muon for Xing4: AdamW decay groups and QK clipping after each step."""

# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.models.xing_4.adapter.optim.muon_layout import Xing4MuonLayout

__all__ = ["build_optimizer", "clip_qk"]


def build_optimizer(*, model: nn.Module, muon_config: dict[str, Any], adamw_config: dict[str, Any],
                    qk_clip_threshold: float = 0.0, muon_layout: str = "native") -> Muon:
    """Build framework Muon with Xing4 decay groups and optional QK clipping.

    ``native`` uses whole matrices; ``mindformers_pynative`` uses logical
    matrix callbacks and requires fused linear_qkv attention. AdamW excludes
    one-dimensional parameters and biases from decay. A zero threshold disables
    QK clipping. With clipping enabled, both MLA up-projections must be
    trainable Muon parameters; optimizer-state swap does not support their hook.

    Args:
        model: Model after recipe module replacements.
        muon_config: Framework Muon options.
        adamw_config: Fallback AdamW options.
        qk_clip_threshold: Maximum allowed per-head attention logit.
        muon_layout: Whole-matrix or logical-matrix grouping.

    Returns:
        The framework optimizer builder.

    Raises:
        ValueError: If the layout or its projection ownership is incompatible.
    """
    options = dict(muon_config)
    attention = [module for module in model.modules()
                 if hasattr(module, "q_b_proj") and hasattr(module, "kv_b_proj")]
    if muon_layout == "mindformers_pynative":
        if any(not hasattr(module, "linear_qkv") for module in attention):
            raise ValueError("The MindFormers logical Muon layout requires fused MLA projections")
        layout = Xing4MuonLayout.from_config(model.config)
        options.pop("muon_reshape_fn", None)
        options.pop("muon_ns_transform_fn", None)
        options.update(reshape_fn=layout.reshape_fn, ns_transform_fn=layout.ns_transform_fn)
    elif muon_layout != "native":
        raise ValueError(f"Unknown Xing4 Muon layout: {muon_layout!r}")
    # The AdamW grouping matches name fragments; full names select exactly the 1-D parameters.
    no_decay = ["bias", *(name for name, parameter in model.named_parameters() if parameter.ndim == 1)]
    builder = Muon(muon_config=options, adamw_config=dict(adamw_config), model=model,
                   no_decay_params=no_decay)
    if qk_clip_threshold > 0:
        muon = builder.get_optimizer().optimizers_dict["muon"]
        owned = {id(parameter) for group in muon.param_groups for parameter in group["params"]}
        if not attention or any(
                not projection.weight.requires_grad or id(projection.weight) not in owned
                for module in attention for projection in (module.q_b_proj, module.kv_b_proj)):
            raise ValueError("QK clipping requires trainable q_b_proj and kv_b_proj weights owned by Muon")
        for module in attention:
            module.record_logit_maxima = True
        # The FP32 main-weight wrapper copies model weights back after the leaf step.
        muon.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: clip_qk(attention, qk_clip_threshold)
        )
    return builder


@torch.no_grad()
def clip_qk(attention: Sequence[nn.Module], threshold: float) -> None:
    """Reduce head maxima globally, scale projection rows, and clear recorded maxima.

    For s = threshold / max(maximum, threshold), Q no-RoPE and K rows use
    sqrt(s), Q RoPE rows use s, and V rows are unchanged. All ranks must
    provide the same modules and global-head ordering for the MAX reduction.

    Args:
        attention: MLA modules that recorded maxima during forward.
        threshold: Positive target maximum.

    Raises:
        ValueError: If a TP shard does not own contiguous complete heads.
    """
    maxima = [_at_global_heads(module, module.max_logits) for module in attention]
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
    """Place local maxima at global positions for contiguous row-sharded TP heads."""
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
        # A row factor is shared across every column shard of the projection.
        placements = tuple(Replicate() if isinstance(p, Shard) and p.dim % tensor.ndim == 1 else p
                           for p in tensor.placements)
        local = distribute_tensor(factors.to(tensor.dtype), tensor.device_mesh, placements).to_local()
        tensor.to_local().mul_(local)
    else:
        tensor.mul_(factors.to(tensor.dtype))
