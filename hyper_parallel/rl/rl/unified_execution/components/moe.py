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
"""Shared graph-capable EP recipe using HP's model and expert ownership."""

from typing import Any, Callable

from rl.unified_execution.ops.moe.fixed_capacity_ep import fixed_capacity_ep
from rl.unified_execution.models import component_family

from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.distributed.recipe_spec import local_compute


def fixed_capacity_ep_forward(module: Any, hidden_states: Any, *, group: Any, rank: int,
                              world_size: int, local_experts: int, capacity: int, router_fn: Callable) -> Any:
    """Execute the original router and experts with fixed-capacity EP transport."""
    states = hidden_states.reshape(-1, hidden_states.shape[-1])
    indices, weights = router_fn(module, states)
    output = fixed_capacity_ep(
        states, indices, weights, expert_forward=module.experts,
        group=group, rank=rank, world_size=world_size, local_experts=local_experts, capacity=capacity,
    )
    return output.view(hidden_states.shape)


@local_compute
def fixed_capacity_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                                 ep_mesh: Any, capacity: int = 256) -> Callable:
    """Bind an opt-in Qwen3-MoE EP recipe without changing parameters or meshes.

    ``capacity`` bounds local token rows entering this MoE region, on both
    training and inference. Oversized batches are rejected, never truncated.
    """
    del mesh, tp_mesh, cp_mesh
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError("Graph EP capacity must be a positive integer")
    if not hasattr(module, "gate") or not hasattr(module, "experts"):
        raise ValueError("Graph EP requires Qwen3-MoE router and experts")
    family = component_family(module, "moe_block")
    expert_mesh = ep_mesh["ep"]
    size = expert_mesh.size()
    bind_local_expert_forward(module, size, use_grouped_gemm=True)
    options = {"group": expert_mesh.get_group(), "rank": expert_mesh.get_local_rank(),
               "world_size": size, "local_experts": module.experts.local_expert_count, "capacity": capacity,
               "router_fn": MOE_ROUTER_ADAPTERS[family.router_adapter]}

    def compute_fn(module: Any, hidden_states: Any) -> Any:
        """Match the HF forward signature while keeping topology in the closure."""
        return fixed_capacity_ep_forward(module, hidden_states, **options)

    return compute_fn
