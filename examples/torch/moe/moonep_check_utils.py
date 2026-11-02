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
"""Build explicit 910C plans using the installed MoonEP host planner."""

import torch
import torch.distributed as dist
from moonep import MoonEPCommPlan
from npu_patch.planning_host import build_plan_host


def build_explicit_plan(route_ids: torch.Tensor, counts: torch.Tensor, group):
    """Gather counts and construct a no-dedup plan for K=1 checks."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    rows = route_ids.numel()
    experts = counts.numel()
    all_counts = [torch.empty_like(counts) for _ in range(world)]
    dist.all_gather(all_counts, counts, group=group)
    fields = build_plan_host(torch.stack(all_counts), counts, route_ids.reshape(-1),
                             rank, experts // world, rows)
    plan = MoonEPCommPlan(
        dst=fields["dst"], experts_to_copy=fields["experts_to_copy"],
        zero_fill_ranges=fields["zero_fill_ranges"], remote_stats=fields["remote_stats"],
        N=rows, R=world, E=experts, B=experts // world, NvS=rows, K=1,
        dup_groups=torch.zeros((rows, 3), dtype=torch.int32, device=counts.device),
        dup_loffs=torch.zeros(rows, dtype=torch.int32, device=counts.device),
        dup_counts=torch.zeros(2, dtype=torch.int32, device=counts.device))
    return plan, fields["cu_seqlens"]
