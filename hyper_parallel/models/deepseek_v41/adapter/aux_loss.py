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
"""V4.1 modality-specific load accounting using the common router primitives."""

from __future__ import annotations

import torch

from hyper_parallel.components.losses.moe_aux_loss import sequence_load_balancing_loss
from hyper_parallel.components.modules.moe.moe_utils import (
    accumulate_router_load as accumulate_expert_load,
    update_router_bias,
)


def accumulate_router_load(
        weights: torch.Tensor,
        selected_experts: torch.Tensor,
        image_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        accumulator: torch.Tensor,
) -> torch.Tensor:
    """Keep text/image counters independent while reusing backward load accumulation.

    Args:
        weights: Routing weights [T, K] used by expert combination.
        selected_experts: Actual dispatch IDs [T, K].
        image_mask: Boolean token metadata; true tokens are images.
        token_mask: Optional boolean metadata excluding padding.
        accumulator: Integer step counters [2, E], text then image.

    Returns:
        Numerically unchanged weights recording each modality's loads in backward.
    """
    if image_mask is not None:
        # Separate modality bins without adding another backward hook or collective.
        selected_experts = selected_experts.long() + image_mask.reshape(-1, 1).long() * accumulator.shape[-1]
    return accumulate_expert_load(weights, selected_experts, token_mask, accumulator.view(-1))


def update_modality_bias(
        bias: torch.Tensor,
        counts: torch.Tensor,
        update_rate: float,
        *,
        vision_bias: torch.Tensor | None = None,
) -> None:
    """Update V4.1 text/image biases using independent expert-load means.

    Args:
        bias: Text correction bias [E], possibly sharded.
        counts: Synchronized int64 loads [2, E], text then image.
        update_rate: Nonnegative sign-update step size.
        vision_bias: Optional image correction bias [E], possibly sharded.
    """
    update_router_bias(bias, counts[0], update_rate)
    if vision_bias is not None:
        update_router_bias(vision_bias, counts[1], update_rate)


__all__ = ["sequence_load_balancing_loss", "accumulate_router_load", "update_modality_bias"]
