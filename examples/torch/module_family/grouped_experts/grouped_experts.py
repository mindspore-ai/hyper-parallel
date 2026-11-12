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
"""Portable routed SwiGLU expert algorithm with identity transport stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from examples.torch.module_family.config import DecoderConfig


@dataclass(frozen=True)
class ExpertInputs:
    """Already-selected source tokens and final routing probabilities."""

    hidden_states: torch.Tensor
    expert_indices: torch.Tensor
    routing_weights: torch.Tensor


class GroupedExpertsModule(nn.Module):
    """Routed SwiGLU experts; the model owns routing and shared experts."""

    def __init__(self, *, config: DecoderConfig | None = None, module: nn.Module | None = None,
                 context: Any = None) -> None:
        """Register logical parameters without acquiring parallel resources.

        Args:
            config: Complete global algorithm dimensions for direct construction.
            module: Compatible existing member for source conversion.
            context: Job selection facts, consumed by the family entry.
        """
        super().__init__()
        del context
        self.config = config if module is None else module.config
        self.gate_up = nn.Parameter(torch.empty(self.config.num_experts, self.config.hidden_size,
                                                2 * self.config.intermediate_size))
        self.down = nn.Parameter(torch.empty(self.config.num_experts, self.config.intermediate_size,
                                             self.config.hidden_size))
        if module is None:
            self.reset_parameters()
        else:
            self.gate_up, self.down = module.gate_up, module.down
            self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize all logical expert weights on real or meta tensors."""
        nn.init.normal_(self.gate_up, mean=0.0, std=0.1)
        nn.init.normal_(self.down, mean=0.0, std=0.1)

    def dispatch(self, inputs: ExpertInputs) -> Any:
        """Preserve the ordinary local inputs without communication.

        Args:
            inputs: Already-selected routed inputs.
        """
        return inputs

    def group_experts(self, batch: Any) -> torch.Tensor:
        """Evaluate the same expert math on local or transported slots.

        Args:
            batch: Local [T,K] IDs or expert-major one-dimensional IDs.
        """
        indices = batch.expert_indices
        hidden = batch.hidden_states
        if indices.ndim == 2:
            hidden = hidden[:, None, :].expand(-1, indices.shape[1], -1).reshape(-1, hidden.shape[-1])
        ids = indices.reshape(-1)
        gate, up = torch.bmm(hidden[:, None, :], self.gate_up[ids]).squeeze(1).chunk(2, dim=-1)
        values = torch.bmm((functional.silu(gate) * up)[:, None, :], self.down[ids]).squeeze(1)
        values = values * batch.routing_weights.reshape(-1, 1)
        return values.reshape(indices.shape[0], indices.shape[1], -1).sum(1) if indices.ndim == 2 else values

    def combine(self, values: torch.Tensor, batch: Any) -> torch.Tensor:
        """Preserve local computed outputs without communication.

        Args:
            values: Weighted expert output.
            batch: Original routed inputs.
        """
        del batch
        return values

    def forward(self, hidden_states: torch.Tensor, expert_indices: torch.Tensor,
                routing_weights: torch.Tensor) -> torch.Tensor:
        """Execute dispatch, expert computation and combine in normal module calls.

        Args:
            hidden_states: Source tokens [T,H].
            expert_indices: Global already-selected expert IDs [T,K].
            routing_weights: Final probabilities [T,K].
        """
        batch = self.dispatch(ExpertInputs(hidden_states, expert_indices, routing_weights))
        return self.combine(self.group_experts(batch), batch)

    def state_contract(self) -> dict[str, str]:
        """Declare the expert representation and probability position."""
        return {"representation": "gate_up_e_h_2i_down_e_i_h", "probability": "output"}
