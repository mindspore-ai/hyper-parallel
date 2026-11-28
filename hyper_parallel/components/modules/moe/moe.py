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
"""Compose module-family members into a mixture-of-experts layer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
)
from hyper_parallel.components.modules.moe.router import GateModule
from hyper_parallel.components.modules.moe.shared_expert import SharedExpertModule


class MoE(nn.Module):
    """Combine the local router, grouped experts and an optional shared expert.

    Args:
        router_config: Configuration accepted by :class:`GateModule`.
        experts_config: Configuration for the routed experts.
        shared_expert_config: Optional configuration accepted by :class:`SharedExpertModule`.
    """

    def __init__(
        self, *, router_config: Any, experts_config: GroupedExpertsConfig,
        shared_expert_config: Any = None,
    ) -> None:
        """Construct the local members from their algorithm configuration."""
        super().__init__()
        self.gate = GateModule(config=router_config)
        self.experts = GroupedExpertsModule(config=experts_config)
        self.shared_experts = (
            SharedExpertModule(config=shared_expert_config) if shared_expert_config is not None else None
        )

    def forward(
        self, hidden_states: torch.Tensor, *, correction_bias: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None, sequence_ids: torch.Tensor | None = None,
        num_sequences: int | None = None,
    ) -> torch.Tensor:
        """Route the tokens and add the optional shared expert contribution.

        Args:
            hidden_states: Token activations of shape [..., hidden_size].
            correction_bias: Optional selection-only expert bias.
            token_mask: Boolean mask selecting tokens for balancing statistics.
            sequence_ids: Logical sample IDs for packed sequences.
            num_sequences: Number of logical samples when sequence_ids is provided.

        Returns:
            Expert output in the original token shape and order.
        """
        _, weights, indices = self.gate(
            hidden_states, correction_bias=correction_bias, token_mask=token_mask,
            sequence_ids=sequence_ids, num_sequences=num_sequences,
        )
        routed = self.experts(hidden_states, indices, weights)
        if self.shared_experts is None:
            return routed
        shared = self.shared_experts(hidden_states)
        if isinstance(shared, tuple):
            shared, bias = shared
            if bias is not None:
                shared = shared + bias
        return routed + shared
