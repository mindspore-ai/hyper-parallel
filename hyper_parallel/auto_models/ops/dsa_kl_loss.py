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
"""DSA sparse-attention indexer KL loss."""

from typing import Any, Tuple

import torch  # pylint: disable=forbidden-backend-import
import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import


class _DSAKLLoss(torch.autograd.Function):
    """Autograd bridge for the DSA indexer KL-loss custom operator."""

    @staticmethod
    def forward(
        ctx: Any,
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        merge_weight: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        actual_seq_qlen: torch.Tensor,
        actual_seq_kvlen: torch.Tensor,
        scale: float,
        loss_coeff: float,
    ) -> torch.Tensor:
        """Compute the KL loss and save its precomputed indexer gradients."""
        grad_index_query, grad_index_key, grad_merge_weight, inner_loss = (
            torch.ops.custom.npu_sparse_lightning_indexer_grad_kl_loss_enhance(
                query=query,
                key=key,
                query_index=index_query,
                key_index=index_key,
                weights=merge_weight,
                sparse_indices=topk_indices,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                scale_value=scale,
                query_rope=query_rope,
                key_rope=key_rope,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_klen=actual_seq_kvlen,
                layout="TND",
                sparse_mode=3,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                sparse_block_size=1,
            )
        )
        divisor = query.size(0)
        ctx.save_for_backward(
            grad_index_query / divisor * loss_coeff,
            grad_index_key / divisor * loss_coeff,
            grad_merge_weight / divisor * loss_coeff,
        )
        return inner_loss.squeeze() / divisor * loss_coeff

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> Tuple:
        """Scale the precomputed indexer gradients by the upstream gradient."""
        grad_index_query, grad_index_key, grad_merge_weight = ctx.saved_tensors
        return (
            grad_index_query * grad_loss,
            grad_index_key * grad_loss,
            grad_merge_weight * grad_loss,
            *((None,) * 11),
        )


def dsa_kl_loss(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    merge_weight: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    """Compute the DSA indexer KL loss.

    ``index_query``, ``index_key``, and ``merge_weight`` are the flattened
    tensors returned by :func:`dsa_indexer`. ``topk_indices``, ``softmax_max``,
    and ``softmax_sum`` come from :func:`dsa_sparse_attention`. The MLA query,
    key, and rotary states use TND layout; sequence lengths are cumulative
    int32 tensors, following the custom-operator call contract.
    The result is a scalar auxiliary loss.
    """
    return _DSAKLLoss.apply(
        index_query,
        index_key,
        merge_weight,
        query,
        key,
        topk_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_kvlen,
        scale,
        loss_coeff,
    )
