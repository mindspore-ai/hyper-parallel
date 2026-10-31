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

"""MoonEP transport and per-dispatch state for HP's native routed experts.

Planning never reads expert values. Borrowed tensors are created inside the
original expert Module's forward, after its FSDP unshard hooks. HCCL autograd
returns borrowed gradients to their home parameters without an EP average.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.distributed as dist

from hyper_parallel.core.dtensor.dtensor import DTensor
from .placement import ExpertPlacement, build_placement, placement_is_profitable
from .precision import GradientTarget, borrow_parameters


@dataclass
class NativeExpertPlacement:
    """Own one forward's immutable route and differentiable borrowed parameters."""

    plan: ExpertPlacement
    ep_group: Any
    local_experts: int
    original_received_rows: int = 0

    def parameters(self, experts: nn.Module) -> tuple[
            dict[str, torch.Tensor], dict[str, GradientTarget | None] | None]:
        """Read the current unsharded parameters only from the expert forward body."""
        parameters = {}
        names, weights = [], []
        for name, parameter in experts.named_parameters(recurse=False):
            if name in ("bias1", "bias2") and not getattr(experts, "_ep_use_grouped_gemm", False):
                continue
            weight = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            if getattr(experts, "use_2d_experts", False) and name in ("gate_up_proj", "up_proj", "down_proj"):
                if name == "down_proj":
                    weight = weight.reshape(self.local_experts, -1, experts.hidden_size)
                else:
                    weight = weight.reshape(self.local_experts, experts.hidden_size, -1)
            if weight.ndim < 1 or weight.shape[0] != self.local_experts:
                raise ValueError(f"MoonEP requires an expert-major parameter: {name} shape={tuple(weight.shape)}")
            names.append(name)
            weights.append(weight)
        if not names:
            raise ValueError("MoonEP native experts expose no current expert parameters")
        return_mask = tuple(weight.requires_grad and (name not in ("bias1", "bias2")
                            or self.original_received_rows > 0) for name, weight in zip(names, weights))
        views, targets = borrow_parameters(tuple(weights), self.plan.copies_by_rank, self.ep_group, return_mask)
        parameters.update(zip(names, views))
        target_map = dict(zip(names, targets)) if any(target is not None for target in targets) else None
        return parameters, target_map


@dataclass(frozen=True)
class NativeDispatchSelection:
    """Describe virtual owners without changing HP's dispatch or combine schedule."""

    invocation: NativeExpertPlacement | None
    virtual_ids: torch.Tensor
    send_counts: list[int]
    receive_counts: list[int]


def prepare_native_placement(
    states: torch.Tensor,
    expert_indices: torch.Tensor,
    local_counts: torch.Tensor,
    *,
    ep_group: Any,
    hidden_size: int,
    intermediate_size: int,
    cube_cores: int,
    weight_bytes: int,
    projection_flops: int = 6,
) -> NativeDispatchSelection:
    """Plan only surviving slots and derive every exchange split from virtual counts.

    Native EP has no complete source-by-expert table. This opt-in path replaces
    its count exchange with one count all-gather; the disabled caller never enters here.
    Rank decisions use this same global table. Static split caches are not used
    for a placement because its virtual ownership can change on every forward.
    """
    ranks, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    experts = local_counts.numel()
    if experts % ranks:
        raise ValueError("MoonEP expert count must be divisible by the EP group")
    local_experts = experts // ranks
    table = local_counts.new_empty((ranks, experts))
    dist.all_gather_into_tensor(table.reshape(-1), local_counts.contiguous(), group=ep_group)
    rows = table.tolist()
    source_rows = [sum(row) for row in rows]
    if source_rows[rank] != states.shape[0]:
        raise ValueError("MoonEP source counts do not cover the surviving routed slots")
    global_rows = sum(source_rows)
    if global_rows > torch.iinfo(torch.int32).max or any(value < 0 for row in rows for value in row):
        raise ValueError("MoonEP allocator counts exceed its nonnegative INT32 routing extent")
    rank_counts = [[sum(row[offset:offset + local_experts])
                    for offset in range(0, experts, local_experts)] for row in rows]
    loads = [sum(column) for column in zip(*rank_counts)]
    target_load = (global_rows + ranks - 1) // ranks
    maximum_received = max(loads)
    placement = None
    if maximum_received > target_load and placement_is_profitable(
            maximum_received, target_load, hidden_size, intermediate_size, cube_cores, weight_bytes,
            projection_flops=projection_flops):
        placement = build_placement(
            states, expert_indices.reshape(-1, 1), table.to(dtype=torch.int32), max(1, target_load), rank)
    selected_ids = expert_indices.reshape(-1, 1)
    invocation = None
    if placement is not None:
        virtual_counts = placement.counts_by_source
        if tuple(virtual_counts.shape) != (ranks, 2 * experts):
            raise ValueError("MoonEP virtual count table has an invalid EP/expert extent")
        selected_rows = virtual_counts.tolist()
        if [sum(row) for row in selected_rows] != source_rows:
            raise ValueError("MoonEP selected routing changed a source's surviving slot count")
        if sum(sum(row) for row in selected_rows) != global_rows:
            raise ValueError("MoonEP selected routing changed the global routed row count")
        selected_local_experts = 2 * local_experts
        rank_counts = [[sum(row[offset:offset + selected_local_experts])
                        for offset in range(0, 2 * experts, selected_local_experts)] for row in selected_rows]
        selected_ids = placement.virtual_ids.reshape(-1, 1).to(dtype=expert_indices.dtype)
        invocation = NativeExpertPlacement(placement, ep_group, local_experts, loads[rank])
    if selected_ids.numel() != states.shape[0]:
        raise ValueError("MoonEP selected routing changed the global routed row count")
    send_counts = rank_counts[rank]
    receive_counts = [row[rank] for row in rank_counts]
    expected_receive = sum(row[rank] for row in rank_counts)
    if sum(send_counts) != states.shape[0] or sum(receive_counts) != expected_receive:
        raise ValueError("MoonEP selected split totals differ from the actual destination rows")
    return NativeDispatchSelection(invocation, selected_ids, send_counts, receive_counts)
