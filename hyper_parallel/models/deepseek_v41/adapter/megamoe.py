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
"""Standalone DeepSeek-V4.1 MoE block backed by MegaMoe routed experts."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.activations import ACT2FN

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.activation import configure_deepseek_v41_swiglu
from hyper_parallel.models.deepseek_v41.adapter.expert_parallel import _router


class DeepseekV41MegaMoe(nn.Module):
    """Keep the source router/shared experts and execute routed experts once.

    This block owns the supplied router and shared-expert modules. Pass a deep
    copy when retaining a reference model. Construct it before the optimizer;
    it accepts complete, unsharded HF expert weights and slices them in EP
    group-local rank order. Trainer replacement and FSDP are separate contracts.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        local_num_tokens: int,
        ep_group: Any | None = None,
        dispatch_mode: str = "push",
    ) -> None:
        """Convert an unclamped, learned-routing source block to MegaMoe.

        Args:
            module: Source block with complete routed weights and limit zero.
            local_num_tokens: Fixed token count received by this rank.
            ep_group: Explicit expert-parallel process group. ``None`` is for a single-process run.
            dispatch_mode: MegaMoe transport, either ``push`` or ``pull``.

        Raises:
            ValueError: If the source semantics or parameter layouts are unsupported.
        """
        super().__init__()
        if module.is_hash:
            raise ValueError("DeepSeek-V4.1 MegaMoe requires learned routing")
        if module.experts.limit != 0 or module.shared_experts.limit != 0:
            raise ValueError("DeepSeek-V4.1 MegaMoe requires swiglu_limit=0 on routed and shared experts")
        if not isinstance(module.experts.act_fn, (nn.SiLU, type(ACT2FN["silu"]))):
            raise ValueError("DeepSeek-V4.1 MegaMoe requires the SiLU activation")
        source_up = module.experts.gate_up_proj
        source_down = module.experts.down_proj
        num_experts = module.experts.num_experts
        hidden_size = module.experts.hidden_dim
        intermediate_size = module.experts.intermediate_dim
        if (tuple(source_up.shape) != (num_experts, 2 * intermediate_size, hidden_size)
                or tuple(source_down.shape) != (num_experts, hidden_size, intermediate_size)):
            raise ValueError("DeepSeek-V4.1 MegaMoe requires complete unsharded HF expert weights")
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        ep_rank = 0 if ep_group is None else dist.get_rank(ep_group)
        if not 0 <= ep_rank < ep_size:
            raise ValueError("The current rank must belong to ep_group")
        configure_deepseek_v41_swiglu(module)
        self.gate = module.gate
        self.shared_experts = module.shared_experts
        self.is_hash = False
        self.experts = MegaMoeExperts(
            local_num_tokens=local_num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=self.gate.top_k,
            ep_size=ep_size,
            ep_group=ep_group,
            dispatch_mode=dispatch_mode,
        ).to(device=source_up.device, dtype=source_up.dtype)
        start = ep_rank * self.experts.local_experts
        end = start + self.experts.local_experts
        with torch.no_grad():
            self.experts.gate_up_weight.copy_(source_up[start:end].transpose(-1, -2))
            self.experts.down_weight.copy_(source_down[start:end].transpose(-1, -2))
        self.experts.gate_up_weight.requires_grad_(source_up.requires_grad)
        self.experts.down_weight.requires_grad_(source_down.requires_grad)
        self.train(module.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the original router, MegaMoe, and the shared expert branch.

        Args:
            hidden_states: Rank-local states ending in the configured hidden size.
            input_ids: Unused compatibility argument; hash routing is unsupported.
            image_mask: Optional text/image selection mask for the source router.

        Returns:
            Routed plus shared output with the shape of hidden_states.
        """
        del input_ids
        indices, weights = _router(self, hidden_states, image_mask)
        return self.experts(hidden_states, indices, weights) + self.shared_experts(hidden_states)

    def close(self) -> None:
        """Release the routed experts' native execution resources."""
        self.experts.close()


__all__ = ["DeepseekV41MegaMoe"]
