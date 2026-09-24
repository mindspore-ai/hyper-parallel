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
"""Bind the model-owned router and expert computation to Hyper EP."""

# This integration uses the Torch/HF runtime.
# pylint: disable=forbidden-backend-import
from functools import partial
from typing import Any, Callable

import torch
import torch.distributed as dist

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.expert_parallel.experts import EPDispatchPolicy
from hyper_parallel.distributed.recipe_spec import local_compute


class _ModelParallelMean(torch.autograd.Function):
    """Replicate a global mean while differentiating each local contribution once."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Reduce equally weighted local objectives over an explicit group."""
        ctx.world_size = dist.get_world_size(group)
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result / ctx.world_size

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Differentiate one logical replicated result, not independent consumers."""
        return gradient / ctx.world_size, None


def _model_parallel_mean(value: torch.Tensor, group: Any = None) -> torch.Tensor:
    """Average partitioned losses with replicated-output gradient semantics.

    Each rank owns a disjoint, equally weighted contribution to one objective.
    All ranks receive the same forward mean and must supply the same upstream
    derivative. Backward returns that derivative divided by group size to each
    local contribution, with no second collective over identical loss replicas.
    This is a model-parallel boundary, not a replacement for DDP averaging or
    an all-reduce with independently consumed outputs. Unequal token partitions
    require explicit token weighting before this operation.

    Args:
        value: Local contribution to the global objective.
        group: Explicit model-parallel group; None keeps the operation local.

    Returns:
        Replicated global mean, or the original tensor for local execution.
    """
    if group is None:
        return value
    return _ModelParallelMean.apply(value, group)


@local_compute
def jt_deepseek_v3_ep_compute(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any) -> Callable:
    """Bind EP execution without installing or changing model semantics."""
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError("DeepSeek V3.2 JT requires an EP mesh")
    module.ep_group = ep_mesh.get_group("ep")
    module.ep_world = ep_mesh["ep"].size()

    def route_with_auxiliary_reduction(module: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep JT routing unchanged and reduce its local auxiliary objective once."""
        routed = type(module).route(module, hidden)
        with torch.autocast(hidden.device.type, enabled=False):
            module.auxiliary_loss = _model_parallel_mean(module.auxiliary_loss, module.ep_group)
        return routed

    executor = build_ep_compute(
        module, ep_mesh, router_fn=route_with_auxiliary_reduction, archetype_key="jt_deepseek_v3_hf",
        expected_attrs=["gate", "experts", "shared_experts", "config"],
        combine=module.combine_routed, use_grouped_gemm=True,
        dispatch_policy=EPDispatchPolicy(torch.bfloat16, torch.float32, True),
        aggregate_fn=module.aggregate_experts)
    module.ep_compute = partial(executor, module)
    return type(module).forward
