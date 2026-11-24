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
"""MoonEP member sharing the ordinary expert algorithm and FSDP parameter lifecycle."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import validate_ep_layout
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule, match_grouped_experts_inputs,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.moonep_context import MoonEPContext
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec


class MoonEPGroupedExperts(GroupedExpertsModule):
    """Balance execution with temporary replicas while retaining home Shard(0) Parameters."""

    def __init__(
        self, *, config: GroupedExpertsConfig | None = None, module: nn.Module | None = None,
        module_fqn: str = "", context: Any = None, moonep_replica_capacity: int | None = None,
        overlap_shared_expert: bool = True,
    ) -> None:
        """Construct the shared algorithm without acquiring runtime resources.

        Args:
            config: Canonical expert algorithm configuration, exclusive with module.
            module: Packed source whose parameters and training state are retained.
            module_fqn: Source expert boundary for replacement metadata.
            context: Builder-provided model configuration and initialization facts.
            moonep_replica_capacity: Remote expert slots per rank; None means E/EP.
            overlap_shared_expert: Enable context-owned shared FC1/FC2 scheduling.
        """
        super().__init__(config=config, module=module, module_fqn=module_fqn, context=context)
        self.moonep_replica_capacity = moonep_replica_capacity
        self.overlap_shared_expert = overlap_shared_expert
        self.ep_context: MoonEPContext | None = None

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the job-owned MoonEP context with matching home ownership.

        Args:
            binding: Existing expert mesh and MoonEP runtime context.
        """
        context = binding.ep_context
        if not isinstance(context, MoonEPContext) or context.num_experts != self.config.num_experts:
            raise ValueError("MoonEP experts require a MoonEP context with the same global expert count")
        if self.ep_context is not None and self.ep_context is not context:
            raise ValueError("MoonEP experts already have a different communication owner")
        self.ep_context = context

    def forward(
        self, hidden_states: Tensor, topk_ids: Tensor | None = None, topk_weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run the inherited permute/compute/unpermute with a per-call MoonEP plan.

        Args:
            hidden_states: Source tokens [...,H].
            topk_ids: Logical expert IDs [T,K], unchanged by physical balancing.
            topk_weights: Differentiable route probabilities [T,K].
            **kwargs: Recognized source routing aliases.

        Returns:
            Routed token outputs [...,H]; routing weights are applied exactly once.
        """
        if self.ep_context is None:
            raise ValueError("MoonEP experts must be bound before forward")
        topk_ids, topk_weights = match_grouped_experts_inputs(topk_ids, topk_weights, kwargs)
        output_shape = tuple(hidden_states.shape)
        hidden_states = self.validate_and_flatten_inputs(hidden_states, topk_ids, topk_weights)
        hidden, ids, _, weights, source_inverse = self.permute(hidden_states, topk_ids, topk_weights)
        plan = self.ep_context.plan(ids)
        gate, down = self.gate_up_proj, self.down_proj
        gate = gate.to_local() if isinstance(gate, DTensor) else gate
        down = down.to_local() if isinstance(down, DTensor) else down
        gate, down = self.ep_context.expert_weights(gate, down, plan)
        hidden, counts, weights, receive_inverse = self.ep_context.dispatch(
            hidden, weights, plan=plan,
        )
        values = self.compute_experts(hidden, counts, gate, down, weights)
        values = self.ep_context.combine(values, plan, receive_inverse)
        return self.unpermute(values, source_inverse, topk_weights, output_shape)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare fixed home ownership and the collective MoonEP protocol."""
        return ParallelModuleSpec(
            "grouped_experts.forward.moonep", "ep", {"forward": ("hidden_states", "topk_ids", "topk_weights")},
            validate=validate_ep_layout,
            options={"num_experts": self.config.num_experts, "moonep": True,
                     "moonep_replica_capacity": self.moonep_replica_capacity,
                     "overlap_shared_expert": self.overlap_shared_expert},
        )
