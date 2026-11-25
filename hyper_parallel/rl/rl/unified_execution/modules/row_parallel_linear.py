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
"""Row-parallel projection with shared PIK arithmetic and HP parameter ownership."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.parallelism_invariant.row_parallel_linear import PikContext, PikPlan, pik_linear, prepare_pik
from hyper_parallel.models.replacement import module_replacement


class _SumReplicatedOutputGradient(torch.autograd.Function):
    """Restore HP's non-SP row-output gradient SUM at the model boundary."""

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, group: Any) -> torch.Tensor:
        """Keep the PIK forward value and its already-reduced output layout."""
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Combine rank-local downstream contributions without changing dtype."""
        result = gradient.contiguous().clone()
        dist.all_reduce(result, group=ctx.group)
        return result, None


class PikLinear(nn.Linear):
    """Row-parallel linear with its numeric plan and TP group fixed at construction."""

    def __init__(self, module: nn.Linear, context: PikContext) -> None:
        """Reuse the original parameters; HP shards their storage after replacement."""
        # Linear.__init__ would allocate and initialize replacement Parameters.
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.in_features, self.out_features = module.in_features, module.out_features
        self.register_parameter("weight", module.weight)
        self.register_parameter("bias", module.bias)
        self.pik_context = context
        self.compute_projection = pik_linear
        self.train(module.training)

    # HP recipes bind nn.Linear's public input keyword.
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Read current HP-owned parameters, including freshly unsharded weights."""
        output = self.compute_projection(input, self.weight, self.bias, context=self.pik_context)
        if (torch.is_grad_enabled() and self.pik_context.world_size > 1
                and not self.pik_context.sequence_parallel):
            output = _SumReplicatedOutputGradient.apply(output, self.pik_context.group)
        return output


@module_replacement
def build_pik_linear(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
                     num_leaves: int = 8, leaf_layout: str = "strided",
                     fill_uninitialized_memory: bool = False) -> PikLinear:
    """Build the PIK projection from the original parameters and the current TP group."""
    if module.__class__ is not nn.Linear:  # Exact Linear avoids discarding custom computation.
        raise ValueError(f"PIK replacement requires an ordinary Linear at {module_fqn}")
    if context.get("pp"):
        raise ValueError("Qwen3 PIK PP integration is not yet validated")
    tp_group = context["mesh"].device_mesh["tp"].get_group() if context.get("tp") else None
    pik_context = prepare_pik(
        module.in_features, module.out_features, plan=PikPlan(num_leaves, leaf_layout),
        group=tp_group, sequence_parallel=tp_group is not None and bool(context.get("sequence_parallel")),
        sequence_dim=1, policy={"fill_uninitialized_memory": fill_uninitialized_memory},
    )
    return PikLinear(module, pik_context)
