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
"""DeepSeek-V4.1 sequence balancing and modality-specific expert loads."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch, distribute_tensor
from hyper_parallel.core.utils.communication import differentiable_all_reduce


def sequence_load_balancing_loss(
        scores: torch.Tensor,
        selected_experts: torch.Tensor,
        sequence_ids: torch.Tensor,
        num_sequences: int,
        token_mask: torch.Tensor | None = None,
        sequence_partition_groups: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Average ``E * sum_i(fraction_i * mean_probability_i)`` over sequences.

    Args:
        scores: Nonnegative, unbiased scores for all experts, shaped ``[T, E]``.
        selected_experts: Actual bias-corrected routing decisions, shaped ``[T, K]``.
        sequence_ids: Logical sample index of each token, shaped ``[T]``.
        num_sequences: Global number of samples shared by the token-partition ranks.
        token_mask: Optional boolean ``[T]`` mask; true tokens contribute.
        sequence_partition_groups: Independent axes partitioning the same
            sequences' tokens. Never include a data-parallel/expert-dispatch
            group. Probability sums use a differentiable SUM all-reduce;
            callers must account for replicated losses in gradient reduction.

    Returns:
        Unweighted mean over nonempty sequences. Empty or fully masked inputs
        return a differentiable zero, including when a local shard is empty.
    """
    if scores.ndim != 2 or selected_experts.ndim != 2 or scores.shape[0] != selected_experts.shape[0]:
        raise ValueError("scores and selected_experts must have shapes [T, E] and [T, K]")
    num_tokens, num_experts = scores.shape
    if num_sequences <= 0 or sequence_ids.shape != (num_tokens,) or sequence_ids.dtype != torch.long:
        raise ValueError("sequence_ids must be int64 [T] and num_sequences must be positive")
    top_k = selected_experts.shape[1]
    if not 0 < top_k <= num_experts:
        raise ValueError("top_k must be positive and no greater than num_experts")
    if token_mask is not None and (token_mask.shape != (num_tokens,) or token_mask.dtype != torch.bool):
        raise ValueError("token_mask must be a boolean tensor with shape [T]")
    scores = scores.float()
    probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    valid = torch.ones(num_tokens, device=scores.device, dtype=torch.float32)
    if token_mask is not None:
        valid = token_mask.to(dtype=torch.float32)
    indices = sequence_ids.unsqueeze(-1) * num_experts + selected_experts
    counts = scores.new_zeros(num_sequences * num_experts).scatter_add(
        0, indices.reshape(-1), valid.unsqueeze(-1).expand(-1, top_k).reshape(-1),
    ).view(num_sequences, num_experts)
    probability_sum = scores.new_zeros(num_sequences, num_experts).index_add(
        0, sequence_ids, probabilities * valid.unsqueeze(-1),
    )
    for group in sequence_partition_groups:
        dist.all_reduce(counts, group=group)
        probability_sum = differentiable_all_reduce(probability_sum, "sum", group)
    assignments = counts.sum(dim=-1, keepdim=True)
    token_count = (assignments / top_k).clamp_min(1.0)
    losses = num_experts * ((counts / assignments.clamp_min(1.0)) * (probability_sum / token_count)).sum(-1)
    return losses.sum() / (assignments > 0).sum().clamp_min(1)


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
        counts: Global expert assignment counts for that modality.
        update_rate: Nonnegative step size for the sign update.
    """
    loads = counts.float()
    delta = (update_rate * (loads.mean() - loads).sign()).to(device=bias.device, dtype=bias.dtype)
    if isinstance(bias, DTensor):
        delta = distribute_tensor(delta, bias.device_mesh, bias.placements)
    bias.add_(delta)
