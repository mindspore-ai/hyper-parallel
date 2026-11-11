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
"""PIK row modules and topology binding through the shared HP build protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn

from rl.unified_execution.ops.pik import PikPlan, pik_linear, prepare_pik
from rl.unified_execution.models import get_model_family, resolve_runtime_symbol
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.replacement import module_replacement


class _NativeRowOutputGradient(torch.autograd.Function):
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
    """Preserve the original parameters while using fixed-tree row arithmetic.

    HP owns subsequent sharding, FSDP and loading. A parallel build must bind
    its context before execution; no first-forward initialization is allowed.
    """

    def __init__(self, module: nn.Linear, plan: PikPlan, *, parallel: bool,
                 fill_uninitialized_memory: bool) -> None:
        """Reuse Parameters without allocation, initialization or weight copying."""
        # Linear.__init__ would allocate and initialize replacement Parameters.
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.in_features, self.out_features = module.in_features, module.out_features
        self.register_parameter("weight", module.weight)
        self.register_parameter("bias", module.bias)
        self.pik_plan = plan
        self.pik_policy = {"fill_uninitialized_memory": fill_uninitialized_memory}
        self.pik_context = None
        self.train(module.training)
        if not parallel:
            self.bind_pik(None, sequence_parallel=False)

    def bind_pik(self, group: Any, *, sequence_parallel: bool) -> None:
        """Prepare on the selected worker device before compilation/materialization."""
        if self.pik_context is not None:
            raise ValueError("PIK topology is already bound")
        self.pik_context = prepare_pik(
            self.in_features, self.out_features, plan=self.pik_plan, group=group,
            sequence_parallel=sequence_parallel, sequence_dim=1, policy=self.pik_policy,
        )

    # HP recipes bind nn.Linear's public input keyword.
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Read current HP-owned parameters, including freshly unsharded weights."""
        if self.pik_context is None:
            raise RuntimeError("HP must bind the PIK TP context before forward")
        output = pik_linear(input, self.weight, self.bias, context=self.pik_context)
        if (torch.is_grad_enabled() and self.pik_context.world_size > 1
                and not self.pik_context.sequence_parallel):
            output = _NativeRowOutputGradient.apply(output, self.pik_context.group)
        return output


@module_replacement
def replace_pik_linear(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
                       num_leaves: int = 8, leaf_layout: str = "strided",
                       fill_uninitialized_memory: bool = False) -> nn.Module:
    """Install shared arithmetic even when HP skips the TP1 parallel plan."""
    if module.__class__ is not nn.Linear:  # Exact Linear avoids discarding custom computation.
        raise ValueError(f"PIK replacement requires an ordinary Linear at {module_fqn}")
    if context.get("cp") or context.get("pp"):
        raise ValueError("Qwen3 PIK CP/PP integration is not yet validated")
    return PikLinear(module, PikPlan(num_leaves, leaf_layout), parallel=bool(context.get("tp")),
                     fill_uninitialized_memory=fill_uninitialized_memory)


@local_compute
def pik_row_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                       ep_mesh: Any, sequence_parallel: bool = False) -> Callable:
    """Bind the HP TP group; PIK owns the row reduction and optional SP output.

    Both child and parent output boundaries must be identity transitions:
    Replicate without SP, Shard(1) with SP. Input gather stays owned by HP.
    """
    del mesh, ep_mesh
    if not isinstance(module, PikLinear):
        raise ValueError("PIK forward binding requires the shared PikLinear replacement")
    if cp_mesh is not None and cp_mesh.size() > 1:
        raise ValueError("PIK model CP integration is not yet validated")
    group = tp_mesh.get_group() if tp_mesh is not None and tp_mesh.size() > 1 else None
    module.bind_pik(group, sequence_parallel=sequence_parallel)

    # Keep the same input keyword for HP method-boundary layout binding.
    def compute_fn(module: Any, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Call the same module implementation without recursing into HP's wrapper."""
        return PikLinear.forward(module, input)

    return compute_fn


@module_replacement
def replace_attention_with_pik(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
                               num_leaves: int = 8, leaf_layout: str = "strided",
                               fill_uninitialized_memory: bool = False) -> nn.Module:
    """Compose the existing HP attention replacement with its PIK output projection.

    HP intentionally rejects simultaneous parent/descendant replacement rules.
    Composition retains the authoritative attention factory and weight transforms.
    """
    # Keep both families on their authoritative HP factory, loaded only for a model build.
    family = get_model_family(module.config.model_type)
    factory = resolve_runtime_symbol(family.attention_factory)
    attention = factory(module=module, module_fqn=module_fqn, context=context)
    attention.o_proj = replace_pik_linear(
        module=attention.o_proj, module_fqn=module_fqn + ".o_proj", context=context,
        num_leaves=num_leaves, leaf_layout=leaf_layout, fill_uninitialized_memory=fill_uninitialized_memory,
    )
    return attention
