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
"""Fixed-capacity EP transport preserving individual top-k routes.

Only copies/permutations cross ranks in forward. Expert outputs return to
source tokens before weighting and reduction; routing never determines a
Python split list. All ranks must use the same declared token capacity.
"""

from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.core.dtensor._utils import (
    differentiable_all_gather_concat,
    differentiable_all_to_all_single,
)


def ordered_combine(routes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Combine [tokens, slots, hidden] in slot order, with one final cast.

    Both multiplication and accumulation use FP32. This is shared by eager
    training and captured inference and never reduces across the rank axis.
    """
    if routes.ndim != 3 or weights.shape != routes.shape[:2]:
        raise ValueError("MoE combine requires [tokens, slots, hidden] and [tokens, slots]")
    result = torch.zeros_like(routes[:, 0], dtype=torch.float32)
    for slot in range(routes.shape[1]):
        product = routes[:, slot].float() * weights[:, slot, None].float()
        result = result + product
    return result.to(routes.dtype)


def local_routes(
    states: torch.Tensor, indices: torch.Tensor, *, rank: int, local_experts: int,
    expert_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute fixed-size route rows with zeros assigned to the last local expert.

    The original expert Module.__call__ remains the computation boundary, so
    HP parameter unshard/reshard hooks still run. Padding is masked before
    and after experts and never contributes to output or parameter gradients.
    """
    flat_indices = indices.flatten()
    valid = (flat_indices >= rank * local_experts) & (flat_indices < (rank + 1) * local_experts)
    local_indices = torch.where(valid, flat_indices % local_experts, local_experts - 1)
    expanded = states.repeat_interleave(indices.shape[1], dim=0)
    expanded = torch.where(valid[:, None], expanded, 0)
    output = expert_forward(expanded, local_indices)
    return torch.where(valid[:, None], output, 0)


def owner_routes(received: torch.Tensor, indices: torch.Tensor, local_experts: int) -> torch.Tensor:
    """Select each route's unique expert owner without a floating-point rank sum."""
    ranks = received.shape[0]
    tokens, slots = indices.shape
    hidden = received.shape[-1]
    values = received.view(ranks, tokens, slots, hidden).permute(1, 2, 0, 3)
    owners = (indices.clamp_min(0) // local_experts).view(tokens, slots, 1, 1)
    selected = values.gather(2, owners.expand(tokens, slots, 1, hidden)).squeeze(2)
    return torch.where((indices >= 0)[:, :, None], selected, 0)


def fixed_capacity_ep(
    states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, *,
    expert_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    group: Any, rank: int, world_size: int, local_experts: int, capacity: int,
) -> torch.Tensor:
    """Run exact-capacity communication with device-driven expert routing.

    Args:
        states: Local token rows [T, H], including model-level padding.
        indices: Selected global expert IDs [T, K].
        weights: Corresponding routing weights [T, K].
        expert_forward: Existing HP local-expert module call.
        group: Existing HP EP process group.
        rank: Rank in that group.
        world_size: Number of ranks in that group.
        local_experts: Number of expert weights owned by this rank.
        capacity: Static per-rank token upper bound, identical on all ranks.

    Returns:
        Local combined token rows [T, H]. No routes are dropped.
    """
    if states.ndim != 2 or indices.ndim != 2 or indices.shape != weights.shape:
        raise ValueError("EP expects states [T,H] and aligned IDs/weights [T,K]")
    if indices.shape[0] != states.shape[0] or states.shape[0] > capacity:
        raise ValueError("EP token rows exceed the shared fixed capacity or route rows disagree")
    if min(capacity, local_experts, world_size) < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid fixed-capacity EP topology")
    rows, hidden = states.shape
    slots = indices.shape[1]
    padded_states = F.pad(states, (0, 0, 0, capacity - rows))
    padded_indices = F.pad(indices, (0, 0, 0, capacity - rows), value=-1)
    if world_size == 1:
        all_states, all_indices = padded_states, padded_indices
    else:
        all_states = differentiable_all_gather_concat(padded_states, group, world_size, 0)
        all_indices = torch.empty((world_size * capacity, slots), device=indices.device, dtype=indices.dtype)
        dist.all_gather_into_tensor(all_indices, padded_indices.contiguous(), group=group)
    outgoing = local_routes(all_states, all_indices, rank=rank, local_experts=local_experts,
                            expert_forward=expert_forward)
    if world_size == 1:
        incoming = outgoing
    else:
        splits = [capacity * slots] * world_size
        incoming = differentiable_all_to_all_single(outgoing, splits, splits, group)
    selected = owner_routes(incoming.view(world_size, capacity, slots, hidden), padded_indices, local_experts)
    return ordered_combine(selected[:rows], weights)
