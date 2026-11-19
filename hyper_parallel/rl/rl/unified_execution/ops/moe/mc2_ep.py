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
"""Native MC2 fullmesh inference dispatch/combine around HP grouped experts."""

from typing import Any

import torch
import torch.nn.functional as F


def mc2_ep(states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, *,
           experts: Any, comm_name: str, rank: int, world_size: int,
           local_experts: int, mc2_capacity: int) -> torch.Tensor:
    """Pair native fullmesh dispatch/combine around HP EIO grouped experts.

    The physical bucket is static; a prefix mask controls useful communication
    on device. Inactive expert IDs must remain legal and unique within a row.
    Native combine receives FP32 router weights and opaque dispatch metadata.

    Args:
        states: [T, H] local token rows before EP padding.
        indices: [T, K] global expert IDs in the original top-k slot order.
        weights: [T, K] router probabilities, retained in FP32 for combine.
        experts: HP experts with local EIO weights and effective-row grouped GEMM.
        comm_name: Dedicated MC2 communicator with the HP EP rank order.
        rank: This worker's rank within that communicator.
        world_size: Number of EP ranks, P.
        local_experts: Number of contiguous expert IDs owned by this rank, E/P.
        mc2_capacity: Common physical rows C per sender; T <= C on every rank.

    Returns:
        [T, H] combined output in the input dtype, with EP padding removed.

    Note:
        Dispatch returns a bounded expert-major buffer [M, H] and device counts
        [E/P]. M is allocated capacity, not the number of useful routed rows;
        grouped GEMM consumes only the rows described by those counts.
    """
    import torch_npu  # pylint: disable=C0415

    rows = states.shape[0]
    if rows > mc2_capacity:
        raise ValueError("Inference EP token rows exceed the MC2 capacity")
    slots = indices.shape[1]

    # Equal physical rows keep graph shapes and the MC2 global_bs=0 contract
    # stable across ranks; the device mask excludes padding from useful work.
    states = F.pad(states, (0, 0, 0, mc2_capacity - rows))
    active = torch.arange(mc2_capacity, device=states.device) < rows
    padded_ids = F.pad(indices, (0, 0, 0, mc2_capacity - rows))

    # MC2 still requires valid, distinct expert IDs on inactive rows. Unlike
    # the AllGather router, this interface cannot use -1 as a padding sentinel.
    indices = torch.where(active[:, None], padded_ids, torch.arange(slots, device=states.device)).int()
    weights = F.pad(weights.float(), (0, 0, 0, mc2_capacity - rows))

    # Dispatch gathers routed tokens into local expert-major order. Type 1
    # returns per-expert counts on device, avoiding CPU all-to-all split sizes.
    packed_states, _, combine_metadata, expert_token_counts, ep_recv_counts, _, expand_scales = (
        torch_npu.npu_moe_distribute_dispatch_v2(
            x=states, expert_ids=indices, group_ep=comm_name, ep_world_size=world_size,
            ep_rank_id=rank, moe_expert_num=world_size * local_experts, x_active_mask=active,
            expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0,
            quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="fullmesh",
        )
    )

    # Counts select useful GMM rows inside the bounded receive buffer. Apply
    # router weights only in combine, retaining their original FP32 precision.
    computed = experts.forward_expert_major(packed_states, expert_token_counts)

    # The native pair owns the reverse route and metadata format. Pass its
    # opaque outputs unchanged, using the same communicator, IDs and mask.
    combined = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=computed, expert_ids=indices, assist_info_for_combine=combine_metadata,
        ep_send_counts=ep_recv_counts, expert_scales=weights, group_ep=comm_name,
        ep_world_size=world_size, ep_rank_id=rank, moe_expert_num=world_size * local_experts,
        x_active_mask=active, expand_scales=expand_scales, expert_shard_type=0,
        shared_expert_num=0, shared_expert_rank_num=0, global_bs=0,
        comm_quant_mode=0, comm_alg="fullmesh",
    )

    # Mask the unused output before slicing; native padding rows are not part
    # of the model result and need not contain initialized values.
    return torch.where(active[:, None], combined, 0)[:rows]
