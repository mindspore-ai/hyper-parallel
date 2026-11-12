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
"""EP member delegating only dispatch/combine to the shared context."""

from __future__ import annotations

from typing import Any

import torch

from examples.torch.module_family.grouped_experts.grouped_experts import ExpertInputs, GroupedExpertsModule
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.module_binding import ParallelModuleSpec


def _validate_layout(*, module: Any, sharding: Any, context: Any) -> None:
    if context.axis_sizes["tp"] != 1 or set(sharding.params) != {"gate_up", "down"}:
        raise ValueError("example EP experts require TP=1 and complete expert ownership")
    if any(named != {"ep": Shard(0)} for named in sharding.params.values()):
        raise ValueError("example EP experts require contiguous expert Shard(0)")
    if module.config.num_experts % context.axis_sizes["ep"]:
        raise ValueError("EP size must divide the global expert count")


class EPGroupedExperts(GroupedExpertsModule):
    """Reuse local SwiGLU math over EPContext's expert-major slots."""

    def bind_parallel(self, binding: Any) -> None:
        """Borrow the existing EP context.

        Args:
            binding: Public parallel resource bundle.
        """
        self.ep_context = binding.ep_context

    def dispatch(self, inputs: ExpertInputs) -> Any:
        """Transport the already-selected three Tensor operands.

        Args:
            inputs: Model-owned routed inputs.
        """
        return self.ep_context.dispatch(inputs.hidden_states, inputs.expert_indices, inputs.routing_weights)

    def combine(self, values: torch.Tensor, batch: Any) -> torch.Tensor:
        """Return weighted expert slots to their source tokens.

        Args:
            values: Weighted slot outputs from the inherited expert math.
            batch: Transported inputs and this call's receipt.
        """
        return self.ep_context.combine(values, batch.receipt)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare the class stages and global expert ownership contract."""
        return ParallelModuleSpec("example.swiglu_experts.ep", "ep",
                                  {"dispatch": ("inputs",), "combine": ("values", "batch")},
                                  validate=_validate_layout, options={"num_experts": self.config.num_experts})
