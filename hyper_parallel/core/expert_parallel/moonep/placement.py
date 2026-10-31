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
"""Expert placement shared by ordinary EP and the MegaMoE executor."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .allocator import allocate_experts
from .render import render_route


@dataclass(frozen=True)
class ExpertPlacement:
    """Keep the virtual routing and weight-copy ownership of one invocation.

    Attributes:
        virtual_ids: Selected virtual expert IDs, retaining original token and Top-K order.
        counts_by_source: Counts from every source for all home and borrowed expert slots.
        copies: Device plan whose rows identify the global experts borrowed by each rank.
        copies_by_rank: The same copy ownership on the host for communication split construction.
    """

    virtual_ids: torch.Tensor
    counts_by_source: torch.Tensor
    copies: torch.Tensor
    copies_by_rank: tuple[tuple[int, ...], ...]


def placement_is_profitable(
    maximum_received: int,
    target_load: int,
    hidden_size: int,
    intermediate_size: int,
    cube_cores: int,
    weight_bytes: int,
    *,
    projection_flops: int = 6,
) -> bool:
    """Compare saved rank-critical work with an A3 weight-transfer cost estimate.

    The conservative estimate uses 4 TFLOP/s per cube, 400 GB/s peer bandwidth,
    0.8 ms fixed placement overhead and five passes over a home weight shard.
    It is shared by both executors and does not change their numerical contract.

    Args:
        maximum_received: Maximum routed expert rows on an EP rank before placement.
        target_load: Average rank load, or a conservative upper bound when unavailable.
        hidden_size: Model hidden width.
        intermediate_size: Single gate/up expert projection width.
        cube_cores: Actual device cube-core count.
        weight_bytes: Home expert shard byte size in the selected transfer dtype.
        projection_flops: Multiply-add factor across expert projections: 6 for gated FFNs, 4 for ungated FFNs.

    Returns:
        Whether estimated saved compute exceeds placement and transfer cost.
    """
    excess = max(0, maximum_received - target_load)
    saved_compute = projection_flops * excess * hidden_size * intermediate_size / (cube_cores * 4e12)
    transfer_cost = 8e-4 + 5 * weight_bytes / 4e11
    return saved_compute > transfer_cost


@torch.no_grad()
def build_placement(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    counts_by_source: torch.Tensor,
    capacity: int,
    rank: int,
) -> ExpertPlacement | None:
    """Borrow experts and render their virtual IDs without changing the router.

    Each virtual rank owns its home expert slots followed by equally many optional
    borrowed slots. Original expert parameters and optimizer ownership stay unchanged.
    The caller supplies its existing count table and load target; this function
    performs no additional count collective or expert computation.

    Args:
        hidden: Local token rows on the active NPU.
        topk_ids: Original selected global expert IDs, shaped [tokens, Top-K].
        counts_by_source: Current [EP, global experts] source-count table.
        capacity: Positive target routed rows per rank for the allocator.
        rank: Rank within this ordered EP group.

    Returns:
        A per-invocation placement, or None when no expert needs to be borrowed.
    """
    # The primitive is optional and is loaded only when placement is actually used.
    import torch_npu  # pylint: disable=C0415

    if counts_by_source.ndim != 2:
        raise ValueError("MoonEP counts must be a source-by-expert matrix")
    ranks, experts = counts_by_source.shape
    if ranks < 1 or experts < 1 or experts % ranks or not 0 <= rank < ranks:
        raise ValueError("MoonEP counts must describe a valid ordered EP topology")
    if counts_by_source.dtype not in (torch.int32, torch.int64):
        raise TypeError("MoonEP counts must use INT32 or INT64")
    gathered = counts_by_source.contiguous()
    if gathered.dtype == torch.int64:
        # The legacy route wire uses INT64; the planner's arithmetic is INT32.
        total = int(gathered.sum(dtype=torch.int64).item())
        if not 0 <= total <= 2**31 - 1:
            raise ValueError("MoonEP routed row count exceeds the native INT32 planner limit")
        gathered = gathered.to(torch.int32)
    cumulative, _, _, copies = allocate_experts(gathered, max(1, capacity))
    copies_by_rank = tuple(tuple(row) for row in copies.cpu().tolist())
    if not any(expert >= 0 for row in copies_by_rank for expert in row):
        return None
    inverse = topk_ids.reshape(-1)
    if topk_ids.numel():
        _, inverse, _, _ = torch_npu.npu_moe_init_routing_v2(
            hidden, topk_ids.int(), active_num=topk_ids.numel(), expert_num=experts,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, experts], row_idx_type=0,
        )
    virtual_ids, counts = render_route(
        topk_ids, inverse, gathered, cumulative, copies, rank, experts // ranks,
    )
    return ExpertPlacement(virtual_ids, counts, copies, copies_by_rank)
