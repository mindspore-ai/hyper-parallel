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
"""Adapt MoonEP's dynamic placement to MegaMoE's external virtual-slot route."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from hyper_parallel.core.multicore._loader import load_torch_npu
from hyper_parallel.core.multicore.modules.mega_moe.function import RematerializedMegaMoeState
from hyper_parallel.core.multicore.modules.mega_moe.route import ExternalMegaMoeRoute

from .planning import ExecutionPlan


def permute_migration_input(hidden: torch.Tensor, ids: torch.Tensor,
                            plan: ExecutionPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort expanded Top-K rows by the whole-expert migration's virtual slots."""
    if plan.migration_virtual_ids is None:
        raise ValueError("whole-expert migration plan lacks virtual expert IDs")
    if hidden.numel() == 0:
        return hidden.reshape(0, hidden.shape[-1]), ids.new_empty(0, dtype=torch.int32)
    virtual_ids = plan.migration_virtual_ids[ids.reshape(-1)].reshape_as(ids)
    rows, inverse = load_torch_npu().npu_moe_token_permute(hidden.reshape(-1, hidden.shape[-1]), virtual_ids)
    return rows, inverse.reshape(-1).to(torch.int32).contiguous()


def restore_migration_output(rows: torch.Tensor, inverse: torch.Tensor,
                             scores: torch.Tensor) -> torch.Tensor:
    """Combine native routed outputs and differentiable Top-K scores."""
    if scores.numel() == 0:
        return rows[:0] + scores.sum().to(rows.dtype)
    return load_torch_npu().npu_moe_token_unpermute(
        rows, inverse, probs=scores.float().contiguous(),
    )


def virtual_route_counts(plan: ExecutionPlan, rank: int, capacity: int) -> torch.Tensor:
    """Map split MoonEP expert assignments to contiguous MegaMoE virtual slots.

    Args:
        plan: One immutable MoonEP placement and token dispatch plan.
        rank: Current EP group rank.
        capacity: Home plus replica slots available per destination rank.

    Returns:
        CPU INT32 table ``[source_rank, target_rank * capacity + slot]``.

    Example:
        With two ranks and two slots each, target rank 1's first replica is
        virtual expert 2. A source row sent to that replica increments column 2.
    """
    if plan.source_counts is None:
        raise ValueError("MoonEP plan lacks the source counts needed for external routing")
    counts = plan.source_counts.numpy()
    placement = plan.allocation.numpy()
    ranks, experts = counts.shape
    if not 0 <= rank < ranks or capacity < 1:
        raise ValueError("MoonEP external route requires a valid rank and positive slot capacity")
    source_ends = counts.cumsum(axis=0)
    source_starts = source_ends - counts
    target_ends = placement.cumsum(axis=1)
    target_starts = target_ends - placement
    overlap = np.maximum(np.minimum(source_ends[:, :, None], target_ends[None])
                         - np.maximum(source_starts[:, :, None], target_starts[None]), 0)
    virtual = np.zeros((ranks, ranks * capacity), dtype=np.int32)
    for target in range(ranks):
        active = np.flatnonzero(placement[:, target] > 0)
        if len(active) > capacity:
            raise ValueError("MoonEP placement exceeds MegaMoE's local virtual slots")
        for slot, expert in enumerate(active):
            virtual[:, target * capacity + slot] = overlap[:, expert, target]
    if int(virtual[rank].sum()) != sum(plan.token_send):
        raise RuntimeError("MoonEP source token counts disagree with MegaMoE virtual route")
    if int(virtual[:, rank * capacity:(rank + 1) * capacity].sum()) != sum(plan.token_recv):
        raise RuntimeError("MoonEP receive token counts disagree with MegaMoE virtual route")
    return torch.from_numpy(virtual)


def external_route(plan: ExecutionPlan, rank: int, capacity: int, rows_device: torch.device) -> ExternalMegaMoeRoute:
    """Build the MegaMoE route while keeping all MoonEP-specific mapping here."""
    active = len(plan.group_counts)
    if active < capacity:
        boundaries = torch.cat((plan.boundaries,
                                plan.boundaries.new_full((capacity - active,), sum(plan.group_counts))))
    else:
        boundaries = plan.boundaries
    if boundaries.device != rows_device:
        raise ValueError("MoonEP and MegaMoE expert boundaries must use the same device")
    return ExternalMegaMoeRoute(virtual_route_counts(plan, rank, capacity), boundaries)


def execute_external_route(executor: Any, rows: torch.Tensor, gate: torch.Tensor,
                           down: torch.Tensor, plan: ExecutionPlan, *,
                           on_weight_gradients: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
                           on_forward_state: Callable[[RematerializedMegaMoeState], None] | None = None,
                           ) -> torch.Tensor:
    """Execute one native token round trip with MoonEP's planned replica weights."""
    gate, down = pad_expert_weights(executor, gate, down, plan)
    capacity = executor.compute_slots
    route = external_route(plan, executor.rank, capacity, rows.device)
    return executor.mega_executor.forward_external_route(
        rows, route, expert_weights=(gate, down), on_weight_gradients=on_weight_gradients,
        on_forward_state=on_forward_state,
    )


def pad_expert_weights(executor: Any, gate: torch.Tensor, down: torch.Tensor,
                       plan: ExecutionPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend the active MoonEP expert table to MegaMoE's fixed local slot count."""
    capacity = executor.compute_slots
    active = len(plan.group_counts)
    if active < capacity:
        gate = torch.cat((gate, gate.new_zeros((capacity - active, *gate.shape[1:]))))
        down = torch.cat((down, down.new_zeros((capacity - active, *down.shape[1:]))))
    return gate, down
