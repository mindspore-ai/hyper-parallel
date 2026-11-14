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
"""Local LatentMoE composition independent of activation and routing policy."""

from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import


class LatentMoE(nn.Module):
    """Route full features, combine latent experts, then normalize and project.

    Args:
        gate: Router accepting states, a validity mask and a statistics flag;
            returns expert indices and mixture weights.
        experts: Complete local expert modules for the reference execution.
        down_proj: Optional projection into the expert space.
        up_proj: Matching projection back to the decoder space.
        norm: Optional normalization applied after weighted expert combination.
        shared_experts: Optional independent full-width branch.
    """

    def __init__(self, gate: nn.Module, experts: nn.ModuleList, *,
                 down_proj: nn.Module | None = None, up_proj: nn.Module | None = None,
                 norm: nn.Module | None = None, shared_experts: nn.Module | None = None) -> None:
        """Register children directly, preserving source-compatible parameter paths."""
        super().__init__()
        if (down_proj is None) != (up_proj is None) or (norm is not None and down_proj is None):
            raise ValueError("LatentMoE requires paired down/up projections and a latent space for normalization")
        self.gate = gate
        self.experts = experts
        if shared_experts is not None:
            self.shared_experts = shared_experts
        self.use_latent_moe = down_proj is not None
        self.latent_moe_use_norm = norm is not None
        if self.use_latent_moe:
            self.routed_expert_down_proj = down_proj
            self.routed_expert_up_proj = up_proj
            if norm is not None:
                self.routed_expert_norm = norm

    def project_routed_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Flatten tokens and apply the optional latent projection.

        Args:
            hidden_states: Decoder activations with the feature axis last.

        Returns:
            Flattened expert-space token rows.
        """
        latent = hidden_states.reshape(-1, hidden_states.shape[-1])
        return self.routed_expert_down_proj(latent) if self.use_latent_moe else latent

    def project_routed_output(self, combined: torch.Tensor) -> torch.Tensor:
        """Normalize combined values before projecting back to decoder width.

        Args:
            combined: Weighted expert outputs in the expert compute dtype.

        Returns:
            Flattened decoder-space rows.
        """
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                combined = self.routed_expert_norm(combined)
            combined = self.routed_expert_up_proj(combined)
        return combined

    def compute_routed_experts(self, latent: torch.Tensor, indices: torch.Tensor,
                               weights: torch.Tensor) -> torch.Tensor:
        """Evaluate local experts and combine their outputs in FP32.

        Args:
            latent: Flattened expert-space token rows.
            indices: Selected expert IDs for each token and slot.
            weights: Corresponding differentiable mixture weights.

        Returns:
            Combined rows cast once to the latent input dtype.
        """
        combined = torch.zeros(latent.shape, dtype=torch.float32, device=latent.device)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(indices == expert_idx)
            values = expert(latent[token_idx]).float() * weights[token_idx, slot_idx, None]
            combined = combined.index_add(0, token_idx, values)
        return combined.to(latent.dtype)

    def forward(self, hidden_states: torch.Tensor, valid_token_mask: torch.Tensor | None = None,
                collect_router_stats: bool = True) -> torch.Tensor:
        """Run full-width routing and an optional shared branch.

        Args:
            hidden_states: Decoder activations with the feature axis last.
            valid_token_mask: Optional local mask used only for router statistics.
            collect_router_stats: Whether the router records this invocation.

        Returns:
            Routed plus shared values with the original input shape.
        """
        indices, weights = self.gate(hidden_states, valid_token_mask, collect_router_stats)
        latent = self.project_routed_input(hidden_states)
        combined = self.compute_routed_experts(latent, indices, weights)
        result = self.project_routed_output(combined).reshape(hidden_states.shape)
        if hasattr(self, "shared_experts"):
            result = result + self.shared_experts(hidden_states)
        return result
