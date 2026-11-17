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
"""Compressed-key teacher KL loss for the DeepSeek V4.1 DSA Indexer."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

import torch

from hyper_parallel.components.functional.dsa_dense import dsa_dense_indexer_kl_loss, dsa_dense_indexer_lse
from hyper_parallel.components.functional.dsa_kl_loss import dsa_kl_loss as fused_packed_kl
from hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad import (
    gather_selected_keys, sparse_lightning_indexer_kl_loss, validate_fused_kl_inputs,
)

from .._dependency import load_cann, require_package, require_symbol
from .._dsa_layout import DsaQueryLayout, ReduceSum, packed_segments, selected_packed_keys, selected_packed_scores
from ..registry import OpRegistry
from ..spec import OpSpec
from .dsa_indexer import indexer_native_only


class DsaIndexerKlFn(Protocol):
    """Return a scalar KL loss, differentiating only the student."""

    def __call__(
        self, x: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        attention_query: torch.Tensor, compressed_key: torch.Tensor, topk_indices: torch.Tensor,
        sinks: torch.Tensor, *, attention_scale: float, loss_coeff: float, query_chunk_size: int = 256,
        reduce_sum: ReduceSum | None = None, layout: DsaQueryLayout | None = None,
    ) -> torch.Tensor:
        """Return the coefficient-scaled loss across all local query rows."""


DSA_INDEXER_KL = OpSpec[DsaIndexerKlFn](config_key="dsa_indexer_kl", variant="compressed_teacher")


class PackedDsaKlFn(Protocol):
    """Distill the packed sparse attention distribution into the DSA indexer."""

    def __call__(
        self, x: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        query: torch.Tensor, key: torch.Tensor, topk_indices: torch.Tensor,
        softmax_max: torch.Tensor, softmax_sum: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
        query_ends: torch.Tensor, key_ends: torch.Tensor, scale: float, loss_coeff: float,
    ) -> torch.Tensor:
        """Return a token-mean KL loss with gradients on the three student inputs only."""


DSA_PACKED_KL = OpSpec[PackedDsaKlFn](config_key="dsa_indexer_kl", variant="packed_sparse_teacher")


class DenseDsaKlFn(Protocol):
    """Distill unabsorbed MLA into the indexer over each packed document."""

    def __call__(
        self, x: torch.Tensor, index_key: torch.Tensor, weights: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
        maximum: torch.Tensor, total: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
        query_ends: torch.Tensor, key_ends: torch.Tensor, *, scale: float, loss_coeff: float,
    ) -> torch.Tensor:
        """Return the token-mean KL and differentiate the indexer inputs only."""


DSA_DENSE_KL = OpSpec[DenseDsaKlFn](config_key="dsa_indexer_kl", variant="packed_dense_teacher")


def _native_dense_kl(
    x: torch.Tensor, index_key: torch.Tensor, weights: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
    maximum: torch.Tensor, total: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
    query_ends: torch.Tensor, key_ends: torch.Tensor, *, scale: float, loss_coeff: float,
) -> torch.Tensor:
    del maximum, total
    loss = x.new_zeros((), dtype=torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        for qs, qe, ks, ke in packed_segments(query_ends, key_ends):
            key_pos = torch.arange(ke - ks, device=x.device)
            for start in range(qs, qe, 128):
                end = min(start + 128, qe)
                causal = torch.arange(start, end, device=x.device) - qe + ke - ks
                valid = key_pos[None, :] <= causal[:, None]
                with torch.no_grad():
                    logits = torch.einsum("thd,khd->thk", query[start:end].float(), key[ks:ke].float())
                    rotary = torch.einsum("thd,khd->thk", query_rope[start:end].float(), key_rope[ks:ke].float())
                    teacher = ((logits + rotary) * scale).masked_fill(~valid[:, None], -1.0e9)
                    target = teacher.softmax(-1).mean(1).masked_fill(~valid, 0.0)
                student = (x[start:end].float() @ index_key[ks:ke, 0].float().T).relu()
                student = (student * weights[start:end].float().unsqueeze(-1)).sum(dim=1)
                student = student.masked_fill(~valid, -1.0e9)
                loss = loss + (target * (target.clamp_min(1.0e-30).log() - student.log_softmax(-1))).sum()
    return loss * (loss_coeff / x.shape[0])


def _cann_dense_kl(
    x: torch.Tensor, index_key: torch.Tensor, weights: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
    maximum: torch.Tensor, total: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
    query_ends: torch.Tensor, key_ends: torch.Tensor, *, scale: float, loss_coeff: float,
) -> torch.Tensor:
    indexer_stats = dsa_dense_indexer_lse(x, index_key, weights, query_lengths=query_ends, key_lengths=key_ends)
    return dsa_dense_indexer_kl_loss(
        x, index_key, weights, query, key, maximum, total, indexer_stats, scale=scale,
        query_rope=query_rope, key_rope=key_rope, query_lengths=query_ends, key_lengths=key_ends,
        loss_coefficient=loss_coeff,
    )


def _load_cann_dense_kl() -> DenseDsaKlFn:
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_dense_lightning_indexer_softmax_lse")
    require_symbol(extension, "npu_dense_lightning_indexer_grad_kl_loss")
    return _cann_dense_kl


def _native_packed_kl(
    x: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    query: torch.Tensor, key: torch.Tensor, topk_indices: torch.Tensor,
    softmax_max: torch.Tensor, softmax_sum: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
    query_ends: torch.Tensor, key_ends: torch.Tensor, scale: float, loss_coeff: float,
) -> torch.Tensor:
    del softmax_max, softmax_sum
    loss = x.new_zeros((), dtype=torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        for qs, qe, ks, ke in packed_segments(query_ends, key_ends):
            for start in range(qs, qe, 32):
                end = min(start + 32, qe)
                indices = topk_indices[start:end]
                causal = torch.arange(start, end, device=x.device) - qe + ke - ks
                with torch.no_grad():
                    selected, valid = selected_packed_keys(key[ks:ke], indices, causal_end=causal)
                    rotary_keys, _ = selected_packed_keys(key_rope[ks:ke], indices, causal_end=causal)
                    teacher = (selected_packed_scores(query[start:end], selected)
                               + selected_packed_scores(query_rope[start:end], rotary_keys)) * scale
                    teacher = teacher.masked_fill(~valid.unsqueeze(2), -1.0e9).softmax(-1)
                    target = teacher.mean(dim=2).masked_fill(~valid, 0.0)
                selected, _ = selected_packed_keys(index_key[ks:ke], indices, causal_end=causal)
                student = selected_packed_scores(x[start:end], selected).relu()
                weights = merge_weight[start:end].unflatten(-1, (selected.shape[1], -1)).float()
                student = (student * weights.unsqueeze(-1)).sum(dim=2).masked_fill(~valid, -1.0e9)
                loss = loss + (target * (target.clamp_min(1.0e-30).log() - student.log_softmax(-1))).sum()
    return loss * (loss_coeff / x.shape[0])


def _load_cann_packed_kl() -> PackedDsaKlFn:
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_sparse_lightning_indexer_grad_kl_loss")
    return fused_packed_kl


class _SharedCompressedIndexerKLLoss(torch.autograd.Function):
    """PanGu-style precomputed-gradient KL loss for compressed index keys."""

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
                scatter_indices = safe_indices.unsqueeze(-1).expand_as(selected_key_grad)
                grad_index_key.scatter_add_(
                    1,
                    scatter_indices.reshape(batch_size, -1, index_key.shape[-1]),
                    selected_key_grad.reshape(batch_size, -1, index_key.shape[-1]),
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


def native_dsa_indexer_kl(
    x: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    attention_query: torch.Tensor, compressed_key: torch.Tensor, topk_indices: torch.Tensor,
    sinks: torch.Tensor, *, attention_scale: float, loss_coeff: float, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None, layout: DsaQueryLayout | None = None,
) -> torch.Tensor:
    """Compute the reference KL loss and propagate only the student gradients."""
    index_query = x
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
    if layout is not None:
        topk_indices = _mask_compressed_keys(topk_indices, layout, index_query)
    return _SharedCompressedIndexerKLLoss.apply(
        x, index_key, merge_weight, attention_query.detach(), compressed_key.detach(),
        topk_indices, sinks.detach(), attention_scale, loss_coeff, query_chunk_size, reduce_sum,
    )


def _mask_compressed_keys(
    indices: torch.Tensor, layout: DsaQueryLayout, query: torch.Tensor,
) -> torch.Tensor:
    upper, lower = layout.compressed_bounds(query)
    outside = indices >= upper.unsqueeze(-1)
    if lower is not None:
        outside = outside | (indices < lower.unsqueeze(-1))
    return indices.masked_fill(outside, -1)


def indexer_kl_native_only(
    x: torch.Tensor, index_key: torch.Tensor, topk_indices: torch.Tensor,
    layout: DsaQueryLayout | None, reduce_sum: ReduceSum | None,
) -> str | None:
    """Explain why the CANN KL candidate cannot represent this call."""
    if layout is None or indexer_native_only(x, layout, reduce_sum) is not None:
        return "fused KL requires BF16 D=128, ordinary TopK512, document segments and no TP reduction"
    if x.shape[2] not in (8, 16, 32, 64) or not min(512, index_key.shape[1]) <= topk_indices.shape[-1] <= 512:
        return "fused KL requires supported head counts and ordinary TopK512 indices"
    return None


def _cann_kl(
    x: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    attention_query: torch.Tensor, compressed_key: torch.Tensor, topk_indices: torch.Tensor,
    sinks: torch.Tensor, *, attention_scale: float, loss_coeff: float, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None, layout: DsaQueryLayout | None = None,
) -> torch.Tensor:
    del reduce_sum
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    indices = _mask_compressed_keys(topk_indices, layout, x)
    validate_fused_kl_inputs(x, index_key, merge_weight, indices)
    return sparse_lightning_indexer_kl_loss(
        x, index_key, merge_weight, attention_query.detach(), compressed_key.detach(),
        indices, sinks.detach(), attention_scale, loss_coeff, query_chunk_size, layout.compress_ratio, layout.segments,
    )


def register_dsa_indexer_kl(registry: OpRegistry) -> None:
    """Register the compressed-teacher KL contract and its implementations."""
    registry.register_family("dsa_indexer_kl", specs=(DSA_INDEXER_KL, DSA_PACKED_KL, DSA_DENSE_KL))
    registry.register_impl(
        DSA_INDEXER_KL, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_dsa_indexer_kl,
    )
    registry.register_impl(
        DSA_PACKED_KL, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_packed_kl,
    )
    registry.register_impl(DSA_PACKED_KL, name="cann", devices=("npu",), load=_load_cann_packed_kl)
    registry.register_impl(
        DSA_DENSE_KL, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_dense_kl,
    )
    registry.register_impl(DSA_DENSE_KL, name="cann", devices=("npu",), load=_load_cann_dense_kl)
    registry.register_impl(
        DSA_INDEXER_KL, name="cann", devices=("npu",),
        load=partial(load_cann, _cann_kl, "sparse_lightning_indexer_kl_loss_grad",
                     "sparse_lightning_indexer_kl_loss_grad_metadata"),
    )
