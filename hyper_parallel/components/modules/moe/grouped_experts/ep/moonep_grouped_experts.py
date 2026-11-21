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
"""Reserved MoonEP execution member; default expert mathematics stays shared."""

from __future__ import annotations

from torch import Tensor, nn

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import validate_ep_layout
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule,
)
from hyper_parallel.distributed.expert_parallel.moonep_context import MoonEPContext
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


class MoonEPGroupedExperts(GroupedExpertsModule):
    """Reserve dynamic execution replicas while keeping the canonical home Parameters.

    Future execution plans/prefetches/dispatches through MoonEPContext, calls
    inherited compute_experts with explicit compute weights, and combines once.
    Replica gradients must return to home before FSDP/optimizer consume them.
    """

    def __init__(
        self, *, config: GroupedExpertsConfig | None = None, module: nn.Module | None = None,
    ) -> None:
        """Construct home parameters without acquiring MoonEP runtime resources.

        Args:
            config: Canonical global algorithm configuration, exclusive with module.
            module: Canonical source whose parameters and training state are retained.
        """
        super().__init__(config=config, module=module)
        self.ep_context: MoonEPContext | None = None

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the job-owned MoonEP context with matching home ownership.

        Args:
            binding: Existing expert mesh and MoonEP context; no new process group.
        """
        context = binding.ep_context
        if not isinstance(context, MoonEPContext) or context.num_experts != self.config.num_experts:
            raise ValueError("MoonEP experts require a MoonEP context with the same global expert count")
        if self.ep_context is not None and self.ep_context is not context:
            raise ValueError("MoonEP experts already have a different communication owner")
        self.ep_context = context

    def forward(
        self, hidden_states: Tensor, topk_ids: Tensor | None = None, topk_weights: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """Reserve the full request-level MoonEP execution and autograd lifecycle.

        Args:
            hidden_states: Original source tokens [...,H], before ordinary permute.
            topk_ids: Global model-selected expert IDs [T,K].
            topk_weights: Final differentiable routing weights [T,K].
            **kwargs: Source aliases accepted by the family interface.

        Note:
            The runtime must use the same invocation plan for weight prefetch,
            dispatch, shared compute, combine, replica-gradient return and recompute.
            It must not compose ordinary slot restore around token-level combine.
        """
        raise NotImplementedError("MoonEP execution, replica autograd and invocation lifecycle are not implemented")

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare home Shard(0) ownership and a distinct request-level EP protocol."""
        return ParallelModuleSpec(
            "grouped_experts.forward.moonep", "ep", {"forward": ("hidden_states", "topk_ids", "topk_weights")},
            validate=validate_ep_layout,
            options={"num_experts": self.config.num_experts, "moonep": True},
        )
