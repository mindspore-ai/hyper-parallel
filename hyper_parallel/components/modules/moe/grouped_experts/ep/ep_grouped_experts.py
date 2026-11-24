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
"""EP stage declarations and delegation; the default member owns all expert mathematics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from hyper_parallel.components.modules.module_family import FamilyShardingContext
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


def validate_ep_layout(
    *, module: GroupedExpertsModule, sharding: ModuleShardingSpec, context: FamilyShardingContext,
) -> None:
    """Validate the final canonical expert boundary, rather than a model's source layout.

    Source adapters must convert names, shapes and mathematics before this point.
    Dense TP is allowed; each EP owner keeps complete experts (ETP=1).
    """
    if set(sharding.params) != {"gate_up_proj", "down_proj"}:
        raise ValueError("EP experts require complete canonical expert ownership")
    if any(axes != {"ep": Shard(0)} for axes in sharding.params.values()):
        raise ValueError("TP-extend-EP requires only contiguous expert Shard(0), without expert TP sharding")
    config = module.config
    expected = {"gate_up_proj": (config.num_experts, 2 * config.intermediate_size, config.hidden_size),
                "down_proj": (config.num_experts, config.hidden_size, config.intermediate_size)}
    if any(tuple(module.get_parameter(name).shape) != shape for name, shape in expected.items()):
        raise ValueError("TP-extend-EP requires complete canonical expert matrices")
    if module.config.num_experts % context.axis_sizes["ep"]:
        raise ValueError("EP size must divide the global expert count")


class EPGroupedExperts(GroupedExpertsModule):
    """Reuse the default algorithm and override only transport/binding stages."""

    def __init__(
        self, *, config: GroupedExpertsConfig | None = None, module: nn.Module | None = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
        overlap_shared_expert: bool = True,
        activation_dtype: str = "input", swiglu_limit: float | None = None,
    ) -> None:
        """Construct canonical parameters without acquiring communication resources.

        Args:
            config: Canonical global algorithm configuration, exclusive with module.
            module: Canonical member or packed expert source, exclusive with config.
            module_fqn: Source module path used in diagnostics.
            context: Job facts and source/ancestor configuration.
            overlap_shared_expert: Enable context-owned shared FC1/FC2 scheduling.
        """
        super().__init__(config=config, module=module, module_fqn=module_fqn, context=context,
                         activation_dtype=activation_dtype, swiglu_limit=swiglu_limit)
        self.ep_context: EPContext | None = None
        self.overlap_shared_expert = overlap_shared_expert

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Borrow the common binder's EP context, rejecting missing or conflicting resources.

        Args:
            binding: Job-owned EP resources with matching global expert ownership.
        """
        # MoonEP subclasses EPContext but its dispatch consumes a different plan protocol.
        if binding.ep_context.__class__ is not EPContext or binding.ep_context.num_experts != self.config.num_experts:
            raise ValueError("binding requires an EP context with the same global expert count")
        if self.ep_context is not None and self.ep_context is not binding.ep_context:
            raise ValueError("EP experts already have a different communication owner")
        self.ep_context = binding.ep_context

    def dispatch(
        self, hidden_states: Tensor, expert_ids: Tensor, tokens_per_expert: Tensor,
        routing_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, list[int], list[int], Tensor]:
        """Send locally prepared slots through the borrowed EP context.

        Args:
            hidden_states: Source expert-major activations [S,H].
            expert_ids: Global expert IDs [S], sorted in ascending order.
            tokens_per_expert: Source counts [E]; EP recomputes counts from the transported IDs.
            routing_weights: Optional differentiable before_fc2 probabilities [S].

        Returns:
            Owner-local hidden/counts/weights, this call's split lists and received-row inverse mapping.
        """
        del tokens_per_expert
        if self.ep_context is None:
            raise RuntimeError("bind_parallel must run before EP dispatch")
        return self.ep_context.dispatch(hidden_states, expert_ids, routing_weights)

    def combine(
        self, values: Tensor, send_splits: list[int] | None, recv_splits: list[int] | None, recv_inverse: Tensor | None,
    ) -> Tensor:
        """Return slots using this call's explicit inverse-transport arguments.

        Args:
            values: Owner-local expert outputs in expert-major order.
            send_splits: Per-peer source send sizes from dispatch.
            recv_splits: Per-peer source receive sizes from dispatch.
            recv_inverse: Mapping from peer-major rows to local expert-major rows [R].

        Returns:
            Source expert-major outputs [S,H], before token unpermutation.
        """
        if self.ep_context is None or send_splits is None or recv_splits is None or recv_inverse is None:
            raise RuntimeError("EP combine requires a bound context and this call's transport arguments")
        return self.ep_context.combine(values, send_splits, recv_splits, recv_inverse)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Describe transport signatures and ownership without allocating runtime resources.

        Returns:
            An EP protocol declaration. The binder validates the class methods and
            layout, constructs a context on the job's EP mesh, then calls bind_parallel.
        """
        return ParallelModuleSpec(
            protocol="grouped_experts.slots.ep",
            binding_kind="ep",
            stages={
                "dispatch": ("hidden_states", "expert_ids", "tokens_per_expert", "routing_weights"),
                "combine": ("values", "send_splits", "recv_splits", "recv_inverse"),
            },
            validate=validate_ep_layout,
            options={"num_experts": self.config.num_experts, "overlap_shared_expert": self.overlap_shared_expert},
        )
