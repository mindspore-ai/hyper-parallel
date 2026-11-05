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
"""Xing4 contracts for the shared MLA CP runtime and contiguous training shards."""

from __future__ import annotations

# pylint: disable=forbidden-backend-import

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.utils.communication import differentiable_all_reduce, gather_counts_via_all_gather
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLACPStrategy, apply_mla_rope
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.models.replacement import method_replacement
from hyper_parallel.models.xing_4.adapter.conversion.module_replacement import (
    Xing4MLAAttention, fusion_attention_with_logit_maxima,
)
from hyper_parallel.models.xing_4.adapter.moe import Xing4MoE


def _rotary(tensor: torch.Tensor, positions: tuple[torch.Tensor, torch.Tensor] | None,
            *, interleaved: bool = False) -> torch.Tensor:
    """Preserve official FP32 interleaved RoPE for the runtime's BSND tensors."""
    if positions is None or not interleaved:
        raise ValueError("Xing4 CP requires interleaved RoPE and local position embeddings")
    return apply_mla_rope(tensor.float(), positions, interleaved=True).to(tensor.dtype)


@inner_wrapper
def xing4_mla_cp_wrapper(target_module: nn.Module, mesh: Any, tp_mesh: Any,
                         cp_mesh: Any, ep_mesh: Any, *, strategy: MLACPStrategy = "expanded_ulysses",
                         overlap: bool = False, ulysses_degree: int | None = None,
                         kv_exchange: Literal["all_gather", "all_to_all_v"] = "all_gather") -> list:
    """Select the shared MLA CP plan through a ``when: cp`` forward rule.

    Ulysses is the default and composes with TP when the local head count is
    divisible by CP. At TP1, ``ulysses_degree=1`` selects the shared KV AllGather
    plan. The enclosing training model accepts only implicit causal sequences;
    its HF-generated local mask must not replace the CP runtime's global mask.

    Args:
        target_module: Optimized Xing4 MLA attention.
        mesh: Full framework mesh.
        tp_mesh: Tensor parallel mesh, or None.
        cp_mesh: Active context parallel mesh.
        ep_mesh: Expert parallel mesh, or None.
        strategy: Expanded QKV or expanded Q with latent KV exchange.
        overlap: Overlap input communication with independent projections.
        ulysses_degree: Head redistribution degree; None uses the CP size.
        kv_exchange: Full KV collection or selective causal-prefix distribution.

    Returns:
        Atomic forward rewrites from the shared MLA runtime.
    """
    if not isinstance(target_module, Xing4MLAAttention) or cp_mesh is None:
        raise ValueError("Xing4 CP requires an active CP mesh and the MLA module replacement")
    requests = mla_cp_wrapper(
        target_module, mesh, tp_mesh, cp_mesh, ep_mesh, strategy=strategy, overlap=overlap,
        ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
        rotary_fn=_rotary, attention_interface=fusion_attention_with_logit_maxima,
    )
    original = requests[0].forward

    @wraps(original)
    def forward(hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Use the CP runtime's global causal mask instead of the HF local-shard mask."""
        del attention_mask
        return original(hidden_states, position_embeddings=position_embeddings, attention_mask=None, **kwargs)

    requests[0] = _ForwardRewriteRequest(target_module, forward, requests[0].companion_attrs)
    return requests


@inner_wrapper
def xing4_moe_cp_wrapper(target_module: Xing4MoE, mesh: Any, tp_mesh: Any,
                         cp_mesh: Any, ep_mesh: Any) -> list:
    """Bind sequence-wise router statistics without replacing the MoE computation."""
    del mesh, tp_mesh, ep_mesh
    if not isinstance(target_module, Xing4MoE) or cp_mesh is None:
        raise ValueError("Xing4 router statistics require an MoE module and an active CP mesh")
    if target_module.gate.aux_loss_cp_group is not None:
        raise ValueError("Xing4 MoE CP group is already configured")
    target_module.gate.aux_loss_cp_group = cp_mesh.get_group()
    return []


def _cp_mesh(context: Mapping[str, Any]) -> Any:
    """Resolve the same mesh used by TextParallelBatch's contiguous CP sharder."""
    mesh = getattr(context.get("mesh_context"), "cp_mesh", None)
    if mesh is None or mesh.size() <= 1 or mesh.get_group() is None:
        raise ValueError("Xing4 sequence methods require an active CP mesh")
    return mesh


@method_replacement
def xing4_cp_shift_tokens(*, module: nn.Module, method_fqn: str,
                          original_method: Callable, context: Mapping[str, Any]) -> Callable:
    """Replace ``model.shift_tokens`` with the next-shard token exchange.

    Only IDs and labels are exchanged. Repeating this one-token shift for each
    MTP depth handles depths exceeding a shard's length without wrapping the
    global sequence tail. Communication uses the shared explicit-wait helper.
    """
    del module, method_fqn
    mesh = _cp_mesh(context)
    group, rank, size = mesh.get_group(), mesh.get_local_rank(), mesh.size()

    @wraps(original_method)
    def shift_tokens(tokens: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Shift within the global sequence, padding only its final shard."""
        first = gather_counts_via_all_gather(tokens[:, 0].contiguous(), group, size)
        tail = first[rank + 1] if rank + 1 < size else torch.full_like(first[0], pad_value)
        return torch.cat((tokens[:, 1:], tail.unsqueeze(1)), dim=1)

    return shift_tokens


@method_replacement
def xing4_cp_mean_loss(*, module: nn.Module, method_fqn: str,
                       original_method: Callable, context: Mapping[str, Any]) -> Callable:
    """Replace ``model.mean_loss`` with a global sequence objective.

    Each MTP depth has its own valid-token denominator. Trainer retains its
    ordinary foundation-token weighting across microbatches and DP replicas;
    its DP/CP normalization and FSDP gradient averaging must not be repeated here.
    """
    del module, method_fqn
    group = _cp_mesh(context).get_group()

    @wraps(original_method)
    def mean_loss(loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Average this prediction depth across its CP shards."""
        global_count = count.clone()
        dist.all_reduce(global_count, group=group)
        return differentiable_all_reduce(loss_sum, "sum", group) / global_count.clamp_min(1)

    return mean_loss
