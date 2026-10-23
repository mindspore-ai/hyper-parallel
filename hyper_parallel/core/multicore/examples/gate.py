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

"""Ordinary and Mega sqrt-softplus routing with identical weights and correction bias."""

from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.multicore import GateOutput, MegaGate


class Gate(MegaGate):
    """Select ordinary routing or MegaGate without changing parameter ownership."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        *,
        use_mega_gate: bool,
    ) -> None:
        """Share projection and frozen correction parameters across both paths."""
        super().__init__(
            hidden_size=hidden_size, num_experts=num_experts, top_k=top_k,
            scoring_func="sqrtsoftplus", routed_scaling_factor=routed_scaling_factor,
        )
        self.use_mega_gate = use_mega_gate
        # A production load-balancing updater owns correction-bias changes.
        self.bias.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, normalized routing weights, and global expert identifiers.

        Args:
            hidden_states: Input activations preserving the batch, sequence and hidden dimensions.
        """
        if self.use_mega_gate:
            return self.mega_forward(hidden_states)
        return self.native_forward(hidden_states)

    def native_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run projection, score transformation, biased TopK, gather, and normalization.

        Args:
            hidden_states: Input activations preserving the batch, sequence and hidden dimensions.
        """
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_flat.float(), self.weight.float())  # pylint: disable=not-callable
        scores = F.softplus(logits).sqrt()  # pylint: disable=not-callable
        _, topk_ids = torch.topk(scores + torch.detach(self.bias), self.top_k, dim=-1, sorted=False)
        topk_weights = scores.gather(-1, topk_ids)
        if self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1.0e-20)
        return GateOutput(logits, topk_weights * self.routed_scaling_factor, topk_ids)

    def mega_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace the ordinary routing sequence with the managed MegaGate call.

        Args:
            hidden_states: Input activations preserving the batch, sequence and hidden dimensions.
        """
        return super().forward(hidden_states)
