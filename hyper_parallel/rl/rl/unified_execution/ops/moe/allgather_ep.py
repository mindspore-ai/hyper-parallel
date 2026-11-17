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
"""AllGather inference transport with device-resident effective expert rows."""

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rl.unified_execution.ops.moe.combine import ordered_combine, owner_routes


def allgather_ep(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, *,
                 experts: Any, group: Any, rank: int, world_size: int,
                 local_experts: int, bucket_rows: int) -> torch.Tensor:
    """Gather static buffers, compute owned routes, and restore source token slots.

    Args:
        states: [T, H] local token rows; T may be zero.
        indices: [T, K] global expert IDs, preserving router top-k slot order.
        weights: [T, K] original router probabilities used by ordered_combine.
        experts: HP local experts with effective-row grouped GEMM.
        group: HP EP process group, whose rank order defines expert ownership.
        rank: This worker's EP rank.
        world_size: EP group size P.
        local_experts: Contiguous expert count E/P on each rank.
        bucket_rows: Common physical rows C for this execution, with T <= C.

    Returns:
        [T, H] output in the input dtype after selecting each route's unique
        owner and combining slots in the canonical FP32 order.

    Note:
        Gathered states have shape [P*C, H]. The return exchange carries
        unweighted [P, C, K, H] slots; it does not sum across expert owners.
    """
    import torch_npu  # pylint: disable=C0415

    rows, hidden = states.shape
    if rows > bucket_rows:
        raise ValueError("Inference EP token rows exceed the common execution bucket")
    slots = indices.shape[1]

    # bucket_rows describes this execution, independently of the MC2 limit.
    # Equal buffers allow capture even when peers have unequal or zero tokens.
    states = F.pad(states, (0, 0, 0, bucket_rows - rows))
    indices = F.pad(indices, (0, 0, 0, bucket_rows - rows), value=-1)
    all_states = torch.empty((world_size * bucket_rows, hidden), device=states.device, dtype=states.dtype)
    all_indices = torch.empty((world_size * bucket_rows, slots), device=indices.device, dtype=indices.dtype)
    dist.all_gather_into_tensor(all_states, states.contiguous(), group=group)
    dist.all_gather_into_tensor(all_indices, indices.contiguous(), group=group)
    total_experts = world_size * local_experts

    # The extra expert is a padding sentinel outside every real owner's range.
    # Per-expert device counts restrict GMM work to this rank's valid routes.
    packed, inverse, counts, _ = torch_npu.npu_moe_init_routing_v2(
        all_states, torch.where(all_indices >= 0, all_indices, total_experts).int(),
        active_num=world_size * bucket_rows * slots, expert_num=total_experts + 1,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[rank * local_experts, (rank + 1) * local_experts], quant_mode=-1,
    )
    computed = experts.forward_expert_major(packed, counts)
    owned = (all_indices.flatten() >= rank * local_experts) & (all_indices.flatten() < (rank + 1) * local_experts)

    # Non-owner inverse entries are not valid addresses. Redirect before the
    # gather, then clear their outputs without a floating-point reduction.
    safe_index = torch.where(owned, inverse, 0).long()
    outgoing = torch.where(owned[:, None], computed[safe_index], 0)
    incoming = torch.empty_like(outgoing)

    # Return unweighted slots in equal-sized buffers, then select the unique
    # owner at the source. A rank-wise sum would change the rounding order.
    dist.all_to_all_single(incoming, outgoing, group=group)
    routes = owner_routes(incoming.view(world_size, bucket_rows, slots, hidden), indices, local_experts)
    return ordered_combine(routes[:rows], weights)
