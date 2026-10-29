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
"""Torch dense DSA indexer interfaces backed by CANN operators."""

from __future__ import annotations

from typing import Any, Tuple

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.dsa_ops import (
    _cann_dsa_dense_indexer_kl_loss_grad,
    _cann_dsa_dense_indexer_lse,
)


def dsa_dense_indexer_lse(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    query_lengths: torch.Tensor | None = None,
    key_lengths: torch.Tensor | None = None,
    layout: str = "TND",
) -> torch.Tensor:
    """Compute dense indexer log-sum-exp using the CANN DSA operator.

    ``query_lengths`` and ``key_lengths`` are cumulative document ends (without
    a leading zero) for TND, and per-batch valid lengths for BSND. For TND they
    are required. The result is detached float32 indexer LSE.
    """
    return _cann_dsa_dense_indexer_lse(
        index_query,
        index_key,
        head_weights,
        actual_seq_qlen=query_lengths,
        actual_seq_klen=key_lengths,
        layout=layout,
    )


class _DenseIndexerKLLoss(torch.autograd.Function):
    """Route CANN's explicit dense KL gradients to the indexer projections."""

    @staticmethod
    def forward(
        ctx: Any,
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        head_weights: torch.Tensor,
        teacher_probabilities: torch.Tensor,
        indexer_lse: torch.Tensor,
        query_lengths: torch.Tensor | None,
        key_lengths: torch.Tensor | None,
        layout: str,
        loss_coefficient: float,
    ) -> torch.Tensor:
        """Compute the raw dense KL and retain normalized indexer gradients."""
        if layout == "TND":
            token_count = index_query.shape[0]
        elif query_lengths is None:
            token_count = index_query.shape[0] * index_query.shape[1]
        else:
            token_count = int(query_lengths.to(device="cpu", dtype=torch.int64).sum().item())
        if token_count <= 0:
            raise ValueError("Dense DSA KL requires at least one valid query token")
        grad_query, grad_key, grad_weights, loss = _cann_dsa_dense_indexer_kl_loss_grad(
            index_query,
            index_key,
            head_weights,
            teacher_probabilities,
            indexer_lse,
            actual_seq_qlen=query_lengths,
            actual_seq_klen=key_lengths,
            layout=layout,
        )
        ctx.save_for_backward(
            grad_query / token_count * loss_coefficient,
            grad_key / token_count * loss_coefficient,
            grad_weights / token_count * loss_coefficient,
        )
        return loss.squeeze() / token_count * loss_coefficient

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> Tuple:
        """Route scaled gradients only to indexer query, key, and weights."""
        grad_query, grad_key, grad_weights = ctx.saved_tensors
        return (
            grad_query * grad_loss,
            grad_key * grad_loss,
            grad_weights * grad_loss,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dsa_dense_indexer_kl_loss(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    head_weights: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    indexer_lse: torch.Tensor,
    *,
    query_lengths: torch.Tensor | None = None,
    key_lengths: torch.Tensor | None = None,
    layout: str = "TND",
    loss_coefficient: float = 1.0,
) -> torch.Tensor:
    """Return token-mean dense KL and route gradients only to indexer inputs.

    ``teacher_probabilities`` is a detached dense causal attention target in
    ``[T, Nkv, S_k]`` (TND) or ``[B, S_q, Nkv, S_k]`` (BSND) layout. The CANN
    kernel supplies explicit gradients for ``index_query``, ``index_key``, and
    ``head_weights``; teacher tensors and metadata do not receive gradients.
    """
    return _DenseIndexerKLLoss.apply(
        index_query,
        index_key,
        head_weights,
        teacher_probabilities.detach(),
        indexer_lse,
        query_lengths,
        key_lengths,
        layout,
        float(loss_coefficient),
    )


__all__ = [
    "dsa_dense_indexer_kl_loss",
    "dsa_dense_indexer_lse",
]
