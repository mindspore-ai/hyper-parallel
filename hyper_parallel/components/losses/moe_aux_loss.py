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
"""Sequence load-balancing objective shared by module families and legacy adapters."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

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
        selected_experts: Actual dispatch expert IDs after selection/fix-router, shaped ``[T, K]``.
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
