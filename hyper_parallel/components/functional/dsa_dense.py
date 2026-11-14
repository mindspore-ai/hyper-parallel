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
"""Dense DSA indexer loss using CANN's fused teacher and KL operators."""

from __future__ import annotations

from typing import Any, Tuple

import torch


def _dense_indexer_lse(
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    query_lengths: torch.Tensor | None,
    key_lengths: torch.Tensor | None,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native indexer maximum and exponential sum, without gradients."""
    # Dense operators remain optional for CPU, sparse and Omni-only execution.
    from torch_npu import (  # pylint: disable=import-outside-toplevel
        npu_dense_lightning_indexer_softmax_lse,
    )

    with torch.no_grad():
        return npu_dense_lightning_indexer_softmax_lse(
            query_index.contiguous(), key_index.contiguous(), weights.contiguous(),
            actual_seq_qlen=None if query_lengths is None else query_lengths.tolist(),
            actual_seq_klen=None if key_lengths is None else key_lengths.tolist(),
            layout=layout, sparse_mode=3,
        )


def dsa_dense_indexer_lse(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    query_lengths: torch.Tensor | None = None,
    key_lengths: torch.Tensor | None = None,
    layout: str = "TND",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached FP32 indexer max/sum for fused dense KL.

    TND lengths are cumulative document ends without a leading zero; BSND
    lengths are per-batch valid lengths. TND requires both length tensors.
    The two outputs are maximum and exponential sum, rather than logsumexp.
    """
    return _dense_indexer_lse(
        index_query, index_key, head_weights, query_lengths, key_lengths, layout,
    )


class _DenseIndexerKLLoss(torch.autograd.Function):
    """Connect CANN's explicit dense KL gradients to indexer projections."""

    @staticmethod
    def forward(
        ctx: Any,
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        head_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        indexer_lse: tuple[torch.Tensor, torch.Tensor],
        scale: float,
        query_rope: torch.Tensor | None,
        key_rope: torch.Tensor | None,
        query_lengths: torch.Tensor | None,
        key_lengths: torch.Tensor | None,
        layout: str,
        loss_coefficient: float,
    ) -> torch.Tensor:
        """Compute summed KL in CANN and retain token-mean indexer gradients."""
        query_ends = None if query_lengths is None else query_lengths.tolist()
        key_ends = None if key_lengths is None else key_lengths.tolist()
        token_count = index_query.shape[0] if layout == "TND" else (
            index_query.shape[0] * index_query.shape[1] if query_ends is None else sum(query_ends)
        )
        # Load only when the dense native path is selected.
        from torch_npu import (  # pylint: disable=import-outside-toplevel
            npu_dense_lightning_indexer_grad_kl_loss,
        )

        indexer_max, indexer_sum = indexer_lse
        grad_query, grad_key, grad_weights, loss = npu_dense_lightning_indexer_grad_kl_loss(
            query.contiguous(), key.contiguous(), index_query.contiguous(), index_key.contiguous(),
            head_weights.contiguous(), softmax_max.contiguous(), softmax_sum.contiguous(),
            indexer_max.contiguous(), indexer_sum.contiguous(), scale,
            query_rope=None if query_rope is None else query_rope.expand(
                *query.shape[:-1], query_rope.shape[-1]
            ).contiguous(),
            key_rope=None if key_rope is None else key_rope.expand(
                *key.shape[:-1], key_rope.shape[-1]
            ).contiguous(),
            actual_seq_qlen=query_ends, actual_seq_klen=key_ends,
            layout=layout, sparse_mode=3,
        )
        ctx.save_for_backward(
            grad_query / token_count * loss_coefficient,
            grad_key / token_count * loss_coefficient,
            grad_weights / token_count * loss_coefficient,
        )
        return loss.squeeze() / token_count * loss_coefficient

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> Tuple:
        """Apply the upstream loss seed only to indexer gradients."""
        grad_query, grad_key, grad_weights = ctx.saved_tensors
        return (
            grad_query * grad_loss,
            grad_key * grad_loss,
            grad_weights * grad_loss,
            *((None,) * 12),
        )


def dsa_dense_indexer_kl_loss(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    head_weights: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    indexer_lse: tuple[torch.Tensor, torch.Tensor],
    *,
    scale: float,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
    key_lengths: torch.Tensor | None = None,
    layout: str = "TND",
    loss_coefficient: float = 1.0,
) -> torch.Tensor:
    """Return token-mean dense KL; propagate gradients only to indexer inputs.

    Main Attention Q/K use unabsorbed D128 states with equal head counts and
    separate RoPE. Maximum/sum use [N,T,1] or [B,N,S,1] FP32 layout.
    Pass the max/sum pair returned by dsa_dense_indexer_lse as indexer_lse.
    Teacher probabilities, explicit gradients and summed loss are computed
    inside the CANN fused operator.
    """
    return _DenseIndexerKLLoss.apply(
        index_query, index_key, head_weights, query, key, softmax_max, softmax_sum,
        indexer_lse, scale, query_rope, key_rope, query_lengths, key_lengths,
        layout, float(loss_coefficient),
    )


__all__ = [
    "dsa_dense_indexer_kl_loss",
    "dsa_dense_indexer_lse",
]
