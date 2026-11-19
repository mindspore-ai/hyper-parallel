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
"""Training EP using HP ragged transport and canonical FP32 slot reduction."""

from typing import Any

import torch

from hyper_parallel.distributed.expert_parallel.experts import _prepare_ep_dispatch, _run_ep_local_experts
from rl.unified_execution.ops.moe.combine import ordered_combine


def training_ep(module: Any, states: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, *,
                group: Any, rank: int, world_size: int, local_experts: int) -> torch.Tensor:
    """Restore individual routes after HP's differentiable variable-size exchanges.

    The expert Module.__call__ boundary is retained for FSDP hooks. Original
    FP32 router weights, rather than dispatch's cast weights, enter the merge.
    Training has no inference bucket capacity limit and does not drop routes.

    Args:
        module: HP MoE owner whose experts retain module/FSDP forward hooks.
        states: [T, H] flattened local tokens.
        indices: [T, K] global expert IDs in router slot order.
        weights: [T, K] original FP32 router probabilities, with autograd intact.
        group: HP EP process group for differentiable exchanges.
        rank: Local EP rank, defining the first owned expert.
        world_size: EP group size.
        local_experts: Number of contiguous experts owned by each EP rank.

    Returns:
        [T, H] output with gradients to states, weights and local expert weights.
        The restored intermediate routes have shape [T, K, H].
    """

    # Reuse HP dispatch/autograd without its BF16 weighted merge. These helpers
    # retain the expert module call needed by FSDP; HP exposes no separate public pair.
    dispatch = _prepare_ep_dispatch(
        states, indices, weights, local_expert_count=local_experts,
        global_expert_count=local_experts * world_size, ep_size=world_size, ep_group=group,
        use_fused_moe_token_permute_and_unpermute=False,
    )
    returned = _run_ep_local_experts(
        module, dispatch.states, dispatch.expert_indices, dispatch.send_counts,
        dispatch.receive_counts, group, rank * local_experts,
    )

    # The reverse exchange retains dispatch order. Restore [token, top-k slot]
    # before weighting so changing expert ownership cannot reorder the merge.
    inverse = torch.empty_like(dispatch.dispatch_order)
    inverse.scatter_(0, dispatch.dispatch_order, torch.arange(inverse.numel(), device=inverse.device))
    routes = returned[inverse].view(states.shape[0], indices.shape[1], states.shape[1])
    return ordered_combine(routes, weights)
