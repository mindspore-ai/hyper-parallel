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

"""Ordinary EP experts and MegaMoe replacement under the same routed-expert interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch  # pylint: disable=forbidden-backend-import
import torch_npu
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules import SwiGLUMLP
from hyper_parallel.components.modules.moe import GroupedExperts
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.examples.gate import Gate

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.examples.model import MoeTransformerConfig


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local value of a plain tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class NativeExperts(torch.nn.Module):
    """Expose standard EP routed experts through the MoE Transformer expert interface."""

    def __init__(
        self,
        config: MoeTransformerConfig,
        source: MegaMoeExperts,
        ep_mesh: Any,
        device: torch.device,
    ) -> None:
        """Shard common experts and copy one local MegaMoe expert shard."""
        super().__init__()
        self.module = GroupedExperts(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            num_experts=config.num_experts,
            use_grouped_mm=True,
        ).to(device=device, dtype=torch.bfloat16)
        ExpertParallel().apply(self.module, ep_mesh)
        gate_weight, up_weight = source.gate_up_weight.split(
            config.intermediate_size,
            dim=-1,
        )
        with torch.no_grad():
            _local_tensor(self.module.w1).copy_(gate_weight.transpose(1, 2))
            _local_tensor(self.module.w3).copy_(up_weight.transpose(1, 2))
            _local_tensor(self.module.w2).copy_(
                source.down_weight.transpose(1, 2)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the common routed-expert implementation.

        Args:
            hidden_states: Routed-expert input activations.
            topk_ids: Selected expert identifiers.
            topk_weights: Routing weights for selected experts.
            tokens_per_expert: Optional token counts for each expert.
        """
        if tokens_per_expert is None:
            tokens_per_expert = torch.bincount(
                topk_ids.reshape(-1).to(torch.int64), minlength=self.module.num_experts,
            )
        routed_input, mapping = torch_npu.npu_moe_token_permute(
            hidden_states.reshape(-1, hidden_states.shape[-1]), topk_ids,
        )
        expert_output = self.module(routed_input, tokens_per_expert)
        output = torch_npu.npu_moe_token_unpermute(
            expert_output, mapping, probs=topk_weights.float().contiguous(),
        )
        return output.reshape(hidden_states.shape)

    @staticmethod
    def close() -> None:
        """Match the managed expert lifecycle without owning SHMEM."""


class MoeTransformerBlock(nn.Module):
    """One sqrt-softplus router, routed-expert, and shared-expert block."""

    def __init__(
        self,
        config: MoeTransformerConfig,
        ep_group: Any | None,
        mega_kernels: tuple[str, ...],
    ) -> None:
        """Initialize the router, routed experts, and shared expert."""
        super().__init__()
        self.config = config
        self.gate = Gate(
            hidden_size=config.hidden_size, num_experts=config.num_experts, top_k=config.top_k,
            routed_scaling_factor=config.routed_scaling_factor, use_mega_gate="gate" in mega_kernels,
        )
        self.experts = MegaMoeExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            initial_capacity_factor=config.initial_capacity_factor,
            ep_group=ep_group,
            dispatch_mode=config.dispatch_mode,
            capacity_growth_factor=config.capacity_growth_factor,
        )
        shared_source = nn.Module()
        shared_source.gate_proj = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
        shared_source.up_proj = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
        shared_source.down_proj = nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False)
        self.shared_expert = SwiGLUMLP(module=shared_source)

    def use_native_experts(self, ep_mesh: Any, device: torch.device) -> None:
        """Replace MegaMoe with equal-weight ordinary EP experts before building the optimizer.

        Args:
            ep_mesh: Device mesh defining the original expert ownership.
            device: Execution device used for the parameters and workspace.
        """
        source = self.experts
        native = NativeExperts(self.config, source, ep_mesh, device)
        source.close()
        self.experts = native

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run learned routed experts and the always-active shared expert.

        Args:
            hidden_states: Input activations preserving the batch, sequence and hidden dimensions.
        """
        route = self.gate(hidden_states)
        routed_output = self.experts(hidden_states, topk_ids=route.ids, topk_weights=route.weights)
        return routed_output + self.shared_expert(hidden_states)
