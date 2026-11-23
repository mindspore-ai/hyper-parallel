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
"""CSA2 sparse Indexer KL objective, reference gradients and native dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad import (
    gather_selected_keys, sparse_lightning_indexer_kl_loss, validate_fused_kl_inputs,
)
from hyper_parallel.components.quantization.tensor.csa2_cache_tensor import CSA2CacheTensor


class _SharedCompressedIndexerKLLoss(torch.autograd.Function):
    """Sparse KL with forward-computed Indexer gradients applied during backward."""

    @staticmethod
    def forward(
            ctx: Any,
            index_query: torch.Tensor,
            index_key: torch.Tensor,
            merge_weight: torch.Tensor,
            attention_query: torch.Tensor,
            compressed_key: torch.Tensor,
            topk_indices: torch.Tensor,
            sinks: torch.Tensor,
            attention_scale: float,
            loss_coeff: float,
            query_chunk_size: int,
            reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        """Compute sparse KL and save only the three Indexer gradients."""
        batch_size, sequence_length, _, _ = index_query.shape
        denominator = batch_size * sequence_length
        total_loss = index_query.new_zeros((), dtype=torch.float32)
        grad_index_query = torch.zeros_like(index_query, dtype=torch.float32)
        grad_index_key = torch.zeros_like(index_key, dtype=torch.float32)
        grad_merge_weight = torch.zeros_like(merge_weight, dtype=torch.float32)

        with torch.no_grad(), torch.autocast(device_type=index_query.device.type, enabled=False):
            for start in range(0, sequence_length, query_chunk_size):
                end = min(start + query_chunk_size, sequence_length)
                selected = topk_indices[:, start:end]
                valid = selected >= 0
                # Graph capture forbids data-dependent control flow, so empty
                # rows contribute through masks instead of a `torch.any` skip:
                # their targets are all-zero, their gradients are all-zero, and
                # eager evaluates exactly the same masked expressions.
                safe_indices = selected.clamp_min(0).long()

                query_chunk = index_query[:, start:end].float()
                weight_chunk = merge_weight[:, start:end].float()
                selected_index_key = gather_selected_keys(index_key, safe_indices).float()
                index_dots = torch.einsum(
                    "bcid,bckd->bcik", query_chunk, selected_index_key
                )
                index_relu = index_dots.relu()
                index_scores = (index_relu * weight_chunk.unsqueeze(-1)).sum(dim=2)
                if reduce_sum is not None:
                    index_scores = reduce_sum(index_scores)
                index_scores.masked_fill_(~valid, -1.0e9)

                if isinstance(compressed_key, CSA2CacheTensor):
                    selected_attention_key = compressed_key.dequantize_selected(safe_indices).float()
                else:
                    selected_attention_key = gather_selected_keys(compressed_key, safe_indices).float()
                attention_scores = torch.einsum(
                    "bhcd,bckd->bhck",
                    attention_query[:, :, start:end].float(),
                    selected_attention_key,
                ) * attention_scale
                attention_scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
                sink_logits = sinks.float().view(1, -1, 1, 1).expand(
                    batch_size, -1, end - start, -1
                )
                target = torch.cat((attention_scores, sink_logits), dim=-1).softmax(dim=-1)
                target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0.0).sum(dim=1)
                if reduce_sum is not None:
                    target = reduce_sum(target)
                target = target / target.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(torch.float32).tiny
                )

                log_prediction = index_scores.log_softmax(dim=-1)
                target_log = target.clamp_min(torch.finfo(torch.float32).tiny).log()
                row_loss = (target * (target_log - log_prediction)).sum(dim=-1)
                # Masked reduction instead of boolean indexing: identical value
                # in eager and traceable across runtime-dependent row validity.
                row_mask = valid.any(dim=-1).to(row_loss.dtype)
                total_loss.add_((row_loss * row_mask).sum() * (loss_coeff / denominator))

                grad_scores = log_prediction.exp() * target.sum(-1, keepdim=True) - target
                grad_scores.mul_(loss_coeff / denominator)
                grad_scores.masked_fill_(~valid, 0.0)
                grad_dots = (
                    grad_scores.unsqueeze(2)
                    * weight_chunk.unsqueeze(-1)
                    * (index_dots > 0).to(index_dots.dtype)
                )
                grad_index_query[:, start:end] = torch.einsum(
                    "bcik,bckd->bcid", grad_dots, selected_index_key
                )
                grad_merge_weight[:, start:end] = (
                    grad_scores.unsqueeze(2) * index_relu
                ).sum(dim=-1)
                selected_key_grad = torch.einsum(
                    "bcik,bcid->bckd", grad_dots, query_chunk
                )
                # Preserve the einsum result if SAC cached its storage.
                selected_key_grad = selected_key_grad.masked_fill(~valid.unsqueeze(-1), 0.0)
                # Each selected key contributes a complete gradient row. Avoid
                # CANN ScatterAdd's expanded-index host preprocessing path.
                key_indices = safe_indices.reshape(batch_size, -1)
                key_grad_rows = selected_key_grad.reshape(batch_size, -1, index_key.shape[-1])
                for batch_idx in range(batch_size):
                    grad_index_key[batch_idx].index_put_(
                        (key_indices[batch_idx],), key_grad_rows[batch_idx], accumulate=True
                    )

        ctx.save_for_backward(grad_index_query, grad_index_key, grad_merge_weight)
        return total_loss

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple:
        """Return the precomputed Indexer gradients and detach the teacher."""
        grad_index_query, grad_index_key, grad_merge_weight = ctx.saved_tensors
        return (
            grad_index_query * grad_loss,
            grad_index_key * grad_loss,
            grad_merge_weight * grad_loss,
            *((None,) * 8),
        )


def shared_compressed_indexer_kl_loss(
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        merge_weight: torch.Tensor,
        attention_query: torch.Tensor,
        compressed_key: torch.Tensor,
        topk_indices: torch.Tensor,
        sinks: torch.Tensor,
        *,
        attention_scale: float,
        loss_coeff: float,
        query_chunk_size: int = 256,
        reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None,
        compress_ratio: int = 1,
        query_segments: tuple[tuple[int, int, int, int, int], ...] | None = None,
        use_fused: bool = False,
) -> torch.Tensor:
    """Compute the sparse-stage DSA KL objective on V4.1 compressed keys.

    The target first softmaxes selected main-attention scores per attention
    head, then averages and L1-normalizes across heads. Indexer inputs are
    detached by the caller, so this objective updates only Indexer parameters.

    ``use_fused`` is valid only for ordinary causal TopK selections, without
    candidate-pool holes. ``query_segments`` supplies each document-local K
    prefix; its ratio and residual encode the CP query offset for the kernel.
    ``use_fused=True`` raises for unsupported calls when ``loss_coeff != 0``;
    False explicitly selects the numerical reference. Native errors propagate.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if not loss_coeff:
        return index_query.sum(dtype=torch.float32) * 0.0
    if index_key.shape[1] == 0:
        # Keep zero gradients connected even when no document contains a complete block.
        return (index_query[..., :0].sum(dtype=torch.float32) + index_key.sum(dtype=torch.float32)
                + merge_weight[..., :0].sum(dtype=torch.float32)) * 0.0
    if topk_indices.shape[:2] != index_query.shape[:2]:
        raise ValueError("topk_indices and index_query must share batch/query dimensions")
    if use_fused:
        if reduce_sum is not None:
            raise NotImplementedError("Fused KL does not support TP score/teacher reduction")
        if query_segments is None:
            raise ValueError("Fused KL requires document-local causal query_segments")
        validate_fused_kl_inputs(index_query, index_key, merge_weight, topk_indices)
        teacher_key = compressed_key.dequantize() if isinstance(compressed_key, CSA2CacheTensor) else compressed_key
        return sparse_lightning_indexer_kl_loss(
            index_query, index_key, merge_weight, attention_query.detach(), teacher_key.detach(),
            topk_indices, sinks.detach(), attention_scale, loss_coeff, query_chunk_size,
            compress_ratio, query_segments,
        )
    return _SharedCompressedIndexerKLLoss.apply(
        index_query,
        index_key,
        merge_weight,
        attention_query.detach(),
        compressed_key.detach(),
        topk_indices,
        sinks.detach(),
        attention_scale,
        loss_coeff,
        query_chunk_size,
        reduce_sum,
    )
