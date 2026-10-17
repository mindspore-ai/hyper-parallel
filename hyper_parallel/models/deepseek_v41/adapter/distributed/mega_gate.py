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
"""MegaGate replacement for the DeepSeek V4.1 router."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
from hyper_parallel.models.deepseek_v41.aux_loss import update_modality_bias
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _apply_router_training_state,
    _router_scores,
)
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class DeepseekV41MegaGate(MegaGate):
    """Adopt a DeepSeek V4.1 router's parameters without copying their storage."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str,
        context: Mapping[str, Any],
    ) -> None:
        """Replace one DeepSeek V4.1 router.

        Args:
            module: Original router, including CPU, NPU or meta parameters.
            module_fqn: Model path used for validation diagnostics.
            context: Builder replacement context; MegaGate does not allocate
                parallel resources.

        Raises:
            TypeError: The source is not a ``DeepseekV41TopKRouter``.
        """
        del context
        if not isinstance(module, DeepseekV41TopKRouter):
            raise TypeError(f"{module_fqn}: MegaGate replacement requires DeepseekV41TopKRouter")

        # Meta construction avoids consuming RNG or allocating throwaway parameters.
        with torch.device("meta"):
            super().__init__(
                hidden_size=module.hidden_size,
                num_experts=module.num_experts,
                top_k=module.top_k,
                scoring_func=module.scoring_func,
                routed_scaling_factor=module.routed_scaling_factor,
                vision_enabled=module.bias_vl is not None,
            )

        self.weight = module.weight
        self.bias = module.bias
        self.bias_vl = module.bias_vl
        self.aux_loss_coeff = module.aux_loss_coeff
        self.last_aux_loss = module.last_aux_loss
        self.expert_bias_update_rate = module.expert_bias_update_rate
        self.enable_expert_bias = module.enable_expert_bias
        self.expert_bias_update_groups = module.expert_bias_update_groups
        self.register_buffer("tokens_per_expert", module.tokens_per_expert, persistent=False)
        register_rebuildable_buffer(
            self,
            "tokens_per_expert",
            value=torch.zeros(2, self.num_experts, dtype=torch.long, device="cpu"),
        )
        self.train(module.training)

    def forward(
            self,
            hidden_states: torch.Tensor,
            image_mask: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
            sequence_partition_groups: tuple[Any, ...] = (),
            sequence_ids: torch.Tensor | None = None,
            num_sequences: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MegaGate while retaining the DeepSeek V4.1 training contract.

        Args:
            hidden_states: Token activations ending in the hidden dimension.
            image_mask: Optional boolean image-token mask.
            token_mask: Optional boolean mask selecting tokens for statistics.
            sequence_partition_groups: Token-partition process groups.
            sequence_ids: Optional logical sample IDs.
            num_sequences: Total logical sample count for packed inputs.

        Returns:
            Router logits, training-aware weights, and selected experts.
        """
        if image_mask is not None and (
                image_mask.shape != hidden_states.shape[:-1] or image_mask.dtype != torch.bool):
            raise ValueError("image_mask must be boolean and match the hidden-state token dimensions")
        logits, weights, indices = super().forward(hidden_states, image_mask=image_mask)
        scores = None
        if self.training and self.aux_loss_coeff:
            scores = _router_scores(logits, self.scoring_func)
        weights = _apply_router_training_state(
            self,
            hidden_states,
            scores,
            weights,
            indices,
            image_mask,
            token_mask,
            sequence_partition_groups,
            sequence_ids,
            num_sequences,
        )
        return logits, weights, indices

    @torch.no_grad()
    def update_expert_bias(self, lr: float | None = None, num_recomputations: int = 1) -> None:
        """Update text/image correction biases from accumulated routing loads.

        Args:
            lr: Optional step size, defaulting to the configured update rate.
            num_recomputations: Shared monitor compatibility argument; unused.
        """
        del num_recomputations
        if lr is None:
            lr = self.expert_bias_update_rate
        update_modality_bias(self.bias, self.tokens_per_expert[0], lr)
        if self.bias_vl is not None:
            update_modality_bias(self.bias_vl, self.tokens_per_expert[1], lr)
        self.tokens_per_expert.zero_()


__all__ = ["DeepseekV41MegaGate"]
