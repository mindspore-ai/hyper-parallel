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
"""DeepSeek-V4.1 modality-specific expert loads."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.aux_loss import sequence_load_balancing_loss
from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch, distribute_tensor

__all__ = ["accumulate_router_load", "sequence_load_balancing_loss", "update_modality_bias"]


class _AccumulateRouterLoad(torch.autograd.Function):
    """Count a training microbatch once, even when its forward is recomputed."""

    @staticmethod
    def forward(ctx: Any, weights: torch.Tensor, counts: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
        """Keep the forward unchanged; only a consumed backward records loads.

        Args:
            ctx: Autograd context holding the accumulator reference.
            weights: Routing-weight gradient carrier.
            counts: Immutable local microbatch counts.
            accumulator: Integer step-level text/image counts.
        """
        ctx.save_for_backward(counts)
        ctx.accumulator = accumulator
        return weights

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Accumulate integer counts without retaining the model graph.

        Args:
            ctx: Context saved by forward.
            gradient: Unchanged gradient for the routing weights.
        """
        with SkipDTensorDispatch(), torch.no_grad():
            ctx.accumulator.add_(ctx.saved_tensors[0])
        return gradient, None, None


def accumulate_router_load(
        weights: torch.Tensor,
        selected_experts: torch.Tensor,
        image_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        accumulator: torch.Tensor,
) -> torch.Tensor:
    """Attach text/image load accumulation to the router's backward path.

    Args:
        weights: Routing weights used by expert combination.
        selected_experts: Actual selected indices, shaped [T, K].
        image_mask: Boolean token mask; true means image, otherwise text.
        token_mask: Boolean token mask; true tokens contribute.
        accumulator: Integer [2, E] step counter, text then image.
    """
    num_tokens, top_k = selected_experts.shape
    modalities = selected_experts.new_zeros(num_tokens) if image_mask is None else image_mask.reshape(-1).long()
    valid = selected_experts.new_ones(num_tokens) if token_mask is None else token_mask.reshape(-1).long()
    indices = modalities.unsqueeze(-1) * accumulator.shape[-1] + selected_experts
    counts = torch.zeros_like(accumulator).view(-1).scatter_add(
        0, indices.reshape(-1), valid.unsqueeze(-1).expand(-1, top_k).reshape(-1),
    ).view_as(accumulator)
    if torch.is_grad_enabled() and not weights.requires_grad:
        weights = weights.detach().requires_grad_()
    return _AccumulateRouterLoad.apply(weights, counts, accumulator)


@torch.no_grad()
def update_modality_bias(bias: torch.Tensor, counts: torch.Tensor, update_rate: float) -> None:
    """Update one modality's correction bias, including a sharded parameter.

    Counts must already include all DP and token partitions. An absent modality
    has all-zero counts and produces no update. Bias affects selection only.

    Args:
        bias: One modality's correction parameter, possibly sharded.
        counts: Global int64 expert assignment counts for that modality.
        update_rate: Nonnegative step size for the sign update.
    """
    num_experts = counts.shape[-1]
    total = counts.sum(dim=-1, keepdim=True)
    # Compare in integer arithmetic so rounding cannot erase small load
    # differences before the sign update.
    direction = (total - counts * num_experts).sign()
    delta = (update_rate * direction.float()).to(device=bias.device, dtype=bias.dtype)
    if isinstance(bias, DTensor):
        delta = distribute_tensor(delta, bias.device_mesh, bias.placements)
    bias.add_(delta)
