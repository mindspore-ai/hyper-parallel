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
"""Optional CANN kernels with explicit CSA geometry and external-teacher KL."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.compressed_indexer import (
    compressed_attention_teacher,
    gather_selected_keys_fp32,
    sort_key_indices,
)


@lru_cache(maxsize=1)
def _operators() -> Any:
    """Load optional NPU operators only when a supported NPU tensor is used."""
    for name in ("cann_ops_transformer", "cann_ops_transformer_custom"):
        if find_spec(name) is not None:
            return import_module(name)
    return None


def _supported_inputs(query: torch.Tensor, key: torch.Tensor) -> bool:
    """Keep CPU, other dimensions and FP32 inputs on the numerical reference."""
    return (query.device.type == "npu" and query.dtype in (torch.bfloat16, torch.float16)
            and key.dtype == query.dtype and query.shape[-1] == 128 and key.shape[1] > 0)


@lru_cache(maxsize=16)
def _requires_metadata(device: torch.device) -> bool:
    """Use host tiling on 910B when no device sequence lengths are supplied.

    Other devices keep metadata. Dynamic seqused_q/k would require metadata
    on 910B too; neither adapter below supplies those inputs.
    """
    return "910B" not in torch.npu.get_device_name(device)


def cann_indexer_available(query: torch.Tensor, key: torch.Tensor, topk: int) -> bool:
    """Report the supported LI V2 shape and installed public interface.

    Args:
        query: Indexer query with shape [batch, queries, heads, dimension].
        key: Indexer key bank with shape [batch, keys, dimension].
        topk: Requested sparse selection width.
    """
    if not _supported_inputs(query, key) or not 1 <= topk <= 8192 or not 1 <= query.shape[2] <= 64:
        return False
    if topk > 2048 and topk % 1024:
        return False
    ops = _operators()
    return (hasattr(ops, "lightning_indexer")
            and (not _requires_metadata(query.device) or hasattr(ops, "lightning_indexer_metadata")))


@torch.no_grad()
def cann_compressed_topk(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, ratio: int, topk: int,
        segments: tuple[tuple[int, int, int, int, int], ...],
) -> torch.Tensor:
    """Select each packed segment using its visible K prefix and ratio residual.

    Args:
        query: Local Indexer queries [batch, queries, heads, 128].
        key: Global Indexer key bank [batch, compressed keys, 128].
        weights: Local per-head merge weights, including model scaling.
        ratio: Model compression ratio.
        topk: Output selection width, including invalid tail slots.
        segments: Local Q start/end, global compressed K start/end, and raw
            prefix length modulo ratio. Packed segments require batch size one.
    """
    ops = _operators()
    batch, length, heads, dim = query.shape
    result = torch.full((batch, length, topk), -1, device=query.device, dtype=torch.int32)
    for start, end, key_start, key_end, residual in segments:
        if key_end <= key_start or start == end:
            continue
        residue = torch.full((batch,), residual, device=query.device, dtype=torch.int32)
        metadata = None
        if _requires_metadata(query.device):
            metadata = ops.lightning_indexer_metadata(
                heads, 1, dim, topk, cmp_residual_k=residue,
                batch_size=batch, max_seqlen_q=end-start, max_seqlen_k=key_end-key_start,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=ratio,
            )
        with torch.autocast(device_type=query.device.type, enabled=False):
            indices, _ = ops.lightning_indexer(
                query[:, start:end].contiguous(), key[:, key_start:key_end].unsqueeze(2).contiguous(),
                weights[:, start:end].float().contiguous(), topk,
                cmp_residual_k=residue, metadata=metadata,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=ratio,
            )
        indices = indices[:, :, 0]
        indices = torch.where(indices >= 0, indices + key_start, key.shape[1])
        indices = sort_key_indices(indices, key.shape[1])
        result[:, start:end] = indices.masked_fill(indices == key.shape[1], -1)
    return result


def cann_kl_available(query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor) -> bool:
    """Use the validated shape contract; unsupported shapes retain reference KL.

    Args:
        query: Indexer query with shape [batch, queries, heads, dimension].
        key: Indexer key bank with shape [batch, keys, dimension].
        indices: Sparse selections with shape [batch, queries, selected].
    """
    if not _supported_inputs(query, key):
        return False
    if query.shape[2] not in (8, 16, 32, 64) or not 1 <= indices.shape[-1] <= min(8192, key.shape[1]):
        return False
    if key.shape[1] > 524288 or query.shape[0] > 256:
        return False
    ops = _operators()
    return (hasattr(ops, "sparse_lightning_indexer_kl_loss_grad")
            and (not _requires_metadata(query.device)
                 or hasattr(ops, "sparse_lightning_indexer_kl_loss_grad_metadata")))


def _stable_prediction_log(
        prediction: torch.Tensor, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        indices: torch.Tensor, chunk_size: int,
) -> torch.Tensor:
    """Recompute only underflowed rows with log-softmax instead of clamping loss."""
    bad = (prediction <= 0).any(dim=-1)
    log_prediction = prediction.log()
    # A single boundary check protects scalar KL from softmax underflow. It is
    # not performed per teacher chunk, and genuine kernel errors are not hidden.
    if not bool(bad.any()):
        return log_prediction
    for batch in range(query.shape[0]):
        rows = bad[batch].nonzero().flatten()
        for start in range(0, rows.numel(), chunk_size):
            chosen = rows[start:start+chunk_size]
            selected = indices[batch:batch+1].index_select(1, chosen).long()
            keys = gather_selected_keys_fp32(key[batch:batch+1], selected)
            queries = query[batch:batch+1].index_select(1, chosen).float()
            merge = weights[batch:batch+1].index_select(1, chosen).float()
            logits = torch.einsum("bqhd,bqkd->bqhk", queries, keys).relu_()
            logits = (logits * merge.unsqueeze(-1)).sum(dim=2)
            log_prediction[batch, chosen] = logits.log_softmax(-1)[0]
    return log_prediction


class _CANNCompressedKLLoss(torch.autograd.Function):
    """Precompute fused Indexer gradients while retaining Hyper's teacher/loss."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
            attention_query: torch.Tensor, main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
            scale: float, coefficient: float, chunk_size: int,
    ) -> torch.Tensor:
        """Compute the original scalar KL and save the native gradient outputs.

        Args:
            ctx: Autograd context that stores gradients and normalization.
            query: Student Indexer queries.
            key: Student Indexer key bank.
            weights: Student per-head merge weights.
            attention_query: Detached main-attention teacher queries.
            main_key: Detached compressed main-key bank.
            indices: Complete valid selections for each query.
            sinks: Main-attention scalar sink logits per head.
            scale: Main-attention score scale.
            coefficient: Auxiliary loss coefficient.
            chunk_size: Teacher query chunk size.
        """
        ops = _operators()
        batch, length, heads, dim = query.shape
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            target = query.new_empty(indices.shape, dtype=torch.float32)
            for start in range(0, length, chunk_size):
                end = min(start+chunk_size, length)
                target[:, start:end] = compressed_attention_teacher(
                    attention_query[:, :, start:end], main_key, indices[:, start:end], sinks, scale,
                )
            metadata = None
            if _requires_metadata(query.device):
                metadata = ops.sparse_lightning_indexer_kl_loss_grad_metadata(
                    heads, 1, dim, batch_size=batch, max_seqlen_q=length, max_seqlen_k=key.shape[1],
                    topk=indices.shape[-1], layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
                )
            dq, dk, dw, prediction = ops.sparse_lightning_indexer_kl_loss_grad(
                query.contiguous(), key.unsqueeze(2).contiguous(), weights.float().contiguous(),
                indices.unsqueeze(2).int().contiguous(), target.unsqueeze(2), metadata=metadata,
                layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
            )
            log_prediction = _stable_prediction_log(
                prediction[:, :, 0], query, key, weights, indices, chunk_size,
            )
            factor = coefficient / (batch * length)
            loss = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - log_prediction)).sum()
            ctx.save_for_backward(dq, dk.squeeze(2), dw)
            ctx.factor = factor
        return loss * factor

    @staticmethod
    def backward(ctx: Any, seed: torch.Tensor) -> tuple:
        """Apply the auxiliary seed and global mean scale exactly once.

        Args:
            ctx: Context holding the native gradients and loss normalization.
            seed: Autograd seed from the auxiliary loss bridge.
        """
        # Scale in FP32 before casting to the autograd input dtype. Dividing
        # precomputed BF16 gradients early would add avoidable rounding.
        gradients = tuple(value.float() * (seed * ctx.factor) for value in ctx.saved_tensors)
        return (*gradients, *((None,) * 7))


def cann_compressed_kl_loss(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        attention_query: torch.Tensor, main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
        scale: float, coefficient: float, chunk_size: int, reference: type[torch.autograd.Function],
) -> torch.Tensor:
    """Fuse full selections; retain the reference for padding and empty rows.

    NoMask KL treats negative indices as zero-logit entries rather than removing
    them from its normalizer. Only full rows may enter that kernel. Each subset
    loss is weighted by its query count to preserve the original batch mean.

    Args:
        query: Student Indexer queries [batch, queries, heads, dimension].
        key: Student key bank [batch, keys, dimension].
        weights: Student head weights [batch, queries, heads].
        attention_query: Main-attention queries [batch, heads, queries, dimension].
        main_key: Compressed main-key bank for the detached teacher.
        indices: Sparse key IDs, with negative IDs denoting invalid entries.
        sinks: Main-attention scalar sink logits per head.
        scale: Main-attention score scale.
        coefficient: Auxiliary loss coefficient.
        chunk_size: Query chunk size for teacher and reference calculations.
        reference: Autograd implementation for partial or empty selections.
    """
    full = (indices >= 0).all(dim=-1)
    denominator = query.shape[0] * query.shape[1]
    full_rows = int(full.sum())
    if full_rows == 0:
        return reference.apply(
            query, key, weights, attention_query, main_key, indices, sinks, scale, coefficient, chunk_size, None,
        )
    if query.shape[1] <= 8192 and full_rows == denominator:
        return _CANNCompressedKLLoss.apply(
            query, key, weights, attention_query, main_key, indices, sinks, scale, coefficient, chunk_size,
        )
    total = query.new_zeros((), dtype=torch.float32)
    for batch in range(query.shape[0]):
        for fused, mask in ((True, full[batch]), (False, ~full[batch])):
            rows = mask.nonzero().flatten()
            for start in range(0, rows.numel(), 8192):
                chosen = rows[start:start+8192]
                args = (
                    query[batch:batch+1].index_select(1, chosen), key[batch:batch+1],
                    weights[batch:batch+1].index_select(1, chosen),
                    attention_query[batch:batch+1].index_select(2, chosen), main_key[batch:batch+1],
                    indices[batch:batch+1].index_select(1, chosen), sinks, scale, coefficient, chunk_size,
                )
                loss = _CANNCompressedKLLoss.apply(*args) if fused else reference.apply(*args, None)
                total = total + loss * (chosen.numel() / denominator)
    return total
