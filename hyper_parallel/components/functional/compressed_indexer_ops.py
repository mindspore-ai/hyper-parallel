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
"""Fused Indexer TopK and auxiliary KL adapters for compressed sparse attention."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.compressed_attention_utils import (
    causal_attention_teacher,
    gather_selected_keys_fp32,
    sort_key_indices,
)


@lru_cache(maxsize=1)
def _load_indexer_op() -> Callable[..., Any]:
    """Import the LI V2 entry only when fused selection is requested."""
    # Keep the optional NPU extension out of CPU and eager-only imports.
    try:
        from cann_ops_transformer import lightning_indexer  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.lightning_indexer (LI V2). "
            "Install the ops-transformer PyTorch extension built for the runtime torch-npu and CANN versions."
        ) from err
    return lightning_indexer


@lru_cache(maxsize=1)
def _load_indexer_metadata_op() -> Callable[..., Any]:
    """Import device tiling only on devices that require LI metadata."""
    # The 910B host-tiling path does not use this optional entry.
    try:
        from cann_ops_transformer import lightning_indexer_metadata  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.lightning_indexer_metadata on this device. "
            "Install the ops-transformer PyTorch extension built for the runtime torch-npu and CANN versions."
        ) from err
    return lightning_indexer_metadata


@lru_cache(maxsize=1)
def _load_kl_ops() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import the fused KL gradient and its metadata entry together."""
    # Resolve concrete optional functions before allocating teacher tensors.
    try:
        from cann_ops_transformer import (  # pylint: disable=import-outside-toplevel
            sparse_lightning_indexer_kl_loss_grad,
            sparse_lightning_indexer_kl_loss_grad_metadata,
        )
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad "
            "and sparse_lightning_indexer_kl_loss_grad_metadata. Install the ops-transformer PyTorch extension "
            "built for the runtime torch-npu and CANN versions."
        ) from err
    return sparse_lightning_indexer_kl_loss_grad, sparse_lightning_indexer_kl_loss_grad_metadata


def _supported_inputs(query: torch.Tensor, key: torch.Tensor) -> bool:
    """Check device, dtype and dimensions for a native Indexer/KL call."""
    return (query.device.type == "npu" and query.dtype in (torch.bfloat16, torch.float16)
            and key.dtype == query.dtype and query.shape[-1] == 128 and key.shape[1] > 0)


@lru_cache(maxsize=16)
def _requires_metadata(device: torch.device) -> bool:
    """Use host tiling on 910B when no device sequence lengths are supplied.

    Other devices keep metadata. Dynamic seqused_q/k would require metadata
    on 910B too; neither adapter below supplies those inputs.
    """
    return "910B" not in torch.npu.get_device_name(device)


def supports_fused_indexer_inputs(query: torch.Tensor, key: torch.Tensor, topk: int) -> bool:
    """Check the LI V2 input contract independently of extension loading.

    Args:
        query: Indexer query with shape [batch, queries, heads, dimension].
        key: Indexer key bank with shape [batch, keys, dimension].
        topk: Requested sparse selection width.
    """
    if not _supported_inputs(query, key) or not 1 <= topk <= 8192 or not 1 <= query.shape[2] <= 64:
        return False
    if topk > 2048 and topk % 1024:
        return False
    return True


@torch.no_grad()
def fused_compressed_topk(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, ratio: int, topk: int,
        segments: tuple[tuple[int, int, int, int, int], ...],
) -> torch.Tensor:
    """Select each document segment using its visible K prefix and ratio residual.

    Each document/local-Q intersection is one native call. Query tiling is
    owned by the operator, independently of the external teacher chunk size.

    Args:
        query: Local Indexer queries [batch, queries, heads, 128].
        key: Global Indexer key bank [batch, compressed keys, 128].
        weights: Local per-head merge weights, including model scaling.
        ratio: Model compression ratio.
        topk: Output selection width, including invalid tail slots.
        segments: Local Q start/end, global compressed K start/end, and raw
            prefix length modulo ratio. Packed segments require batch size one.
    """
    indexer = _load_indexer_op()
    batch, length, heads, dim = query.shape
    result = torch.full((batch, length, topk), -1, device=query.device, dtype=torch.int32)
    for start, end, key_start, key_end, residual in segments:
        if key_end <= key_start or start == end:
            continue
        residue = torch.full((batch,), residual, device=query.device, dtype=torch.int32)
        metadata = None
        if _requires_metadata(query.device):
            metadata = _load_indexer_metadata_op()(
                heads, 1, dim, topk, cmp_residual_k=residue,
                batch_size=batch, max_seqlen_q=end-start, max_seqlen_k=key_end-key_start,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=ratio,
            )
        with torch.autocast(device_type=query.device.type, enabled=False):
            indices, _ = indexer(
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


def supports_fused_kl_inputs(query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor) -> bool:
    """Select the ordinary TopK512 path, including a physically short key bank.

    Args:
        query: Indexer Q [batch, queries, heads, 128].
        key: Indexer K [batch, keys, 128].
        indices: Ordinary causal selections, before optional width padding.
    """
    if not _supported_inputs(query, key) or query.shape[2] not in (8, 16, 32, 64):
        return False
    # Native length validation avoids rejecting valid long banks based only
    # on the conservative sequence range in older operator documentation.
    if indices.shape[-1] != min(512, key.shape[1]) or query.shape[0] > 256:
        return False
    return True


@lru_cache(maxsize=128)
def _kl_metadata(
        device: torch.device, batch: int, length: int, heads: int, dim: int,
        key_length: int, topk: int, ratio: int, residual: int,
) -> tuple:
    """Cache shape-only causal metadata without retaining activations."""
    residue = torch.full((batch,), residual, device=device, dtype=torch.int32) if ratio > 1 else None
    options = {"layout_q": "BSND", "layout_k": "BSND", "mask_mode": 3,
               "cmp_ratio": ratio, "cmp_residual_k": residue}
    _, metadata_op = _load_kl_ops()
    metadata = metadata_op(
        heads, 1, dim, batch_size=batch, max_seqlen_q=length, max_seqlen_k=key_length,
        topk=topk, **options,
    )
    return options, metadata


def _stable_prediction_log(
        prediction: torch.Tensor, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        indices: torch.Tensor, chunk_size: int,
) -> torch.Tensor:
    """Recover valid underflowed probabilities without treating padding as underflow."""
    valid = indices >= 0
    bad = ((prediction <= 0) & valid).any(dim=-1)
    log_prediction = prediction.masked_fill(~valid, 1).log()
    # The native interface does not return student LSE. One boundary check
    # avoids silently clamping the scalar objective when probabilities vanish.
    if not bool(bad.any()):
        return log_prediction
    for batch in range(query.shape[0]):
        rows = bad[batch].nonzero().flatten()
        for start in range(0, rows.numel(), chunk_size):
            chosen = rows[start:start+chunk_size]
            selected = indices[batch:batch+1].index_select(1, chosen).long()
            keys = gather_selected_keys_fp32(key[batch:batch+1], selected.clamp_min(0))
            queries = query[batch:batch+1].index_select(1, chosen).float()
            merge = weights[batch:batch+1].index_select(1, chosen).float()
            # SAC may cache the contraction output for backward replay.
            logits = torch.einsum("bqhd,bqkd->bqhk", queries, keys).relu()
            logits = (logits * merge.unsqueeze(-1)).sum(dim=2).masked_fill(selected < 0, -1.0e9)
            log_prediction[batch, chosen] = logits.log_softmax(-1).masked_fill(selected < 0, 0)[0]
    return log_prediction


class _CausalCompressedKLLoss(torch.autograd.Function):
    """Keep Hyper's detached teacher and scalar loss; fuse the student gradients."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
            attention_query: torch.Tensor, main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
            scale: float, coefficient: float, chunk_size: int, ratio: int, residual: int,
    ) -> torch.Tensor:
        """Save native gradients while retaining Hyper's detached teacher and loss.

        Args:
            ctx: Autograd context holding native gradients and the loss scale.
            query: Student queries [batch, queries, heads, dimension].
            key: Student key bank [batch, keys, dimension].
            weights: Per-query student head weights.
            attention_query: Detached main-attention queries for the teacher.
            main_key: Detached compressed main-attention key bank.
            indices: Selected key IDs, with invalid slots set to -1.
            sinks: Detached per-head main-attention sink logits.
            scale: Main-attention dot-product scale.
            coefficient: Indexer auxiliary loss coefficient.
            chunk_size: Maximum query count for sparse teacher temporaries.
            ratio: Key compression ratio.
            residual: Raw prefix length modulo the compression ratio.

        Returns:
            Scalar KL normalized by this document segment's batch and query count.
        """
        batch, length, heads, dim = query.shape
        kl_grad, _ = _load_kl_ops()
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            target = causal_attention_teacher(
                attention_query, main_key, indices, sinks, scale, ratio, residual, chunk_size,
            )
            options, metadata = _kl_metadata(
                query.device, batch, length, heads, dim, key.shape[1], indices.shape[-1], ratio, residual,
            )
            dq, dk, dw, prediction = kl_grad(
                query.contiguous(), key.unsqueeze(2).contiguous(), weights.float().contiguous(),
                indices.unsqueeze(2).int().contiguous(), target.unsqueeze(2), metadata=metadata, **options,
            )
            log_prediction = _stable_prediction_log(prediction[:, :, 0], query, key, weights, indices, chunk_size)
            loss = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - log_prediction)).sum()
            ctx.save_for_backward(dq, dk.squeeze(2), dw)
            ctx.factor = coefficient / (batch * length)
        return loss * ctx.factor

    @staticmethod
    def backward(ctx: Any, seed: torch.Tensor) -> tuple:
        """Apply the caller's seed in FP32, without differentiating the teacher.

        Args:
            ctx: Autograd context containing native gradients and the loss scale.
            seed: Upstream gradient of the scalar KL output.

        Returns:
            Student Q/K/weight gradients and None for teacher and sequence metadata.
        """
        gradients = tuple(value.float() * (seed * ctx.factor) for value in ctx.saved_tensors)
        return (*gradients, *((None,) * 9))


def fused_compressed_kl_loss(
        query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        attention_query: torch.Tensor, main_key: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor,
        scale: float, coefficient: float, chunk_size: int, ratio: int, segments: tuple,
) -> torch.Tensor:
    """Fuse ordinary causal selections with the original batch/query denominator.

    Submit each document segment to the KL operator once. The external
    teacher and scalar-loss recovery use ``chunk_size`` for temporaries.

    Args:
        query: Student queries [batch, queries, heads, 128].
        key: Global student key bank [batch, keys, 128].
        weights: Student per-head weights including the model's scaling.
        attention_query: Detached teacher Q [batch, heads, queries, 512].
        main_key: Detached global compressed teacher K bank.
        indices: Global sorted TopK IDs, invalid slots at the tail.
        sinks: Detached main attention sink logits.
        scale: Main attention score scale.
        coefficient: KL coefficient applied once to loss and gradient.
        chunk_size: Maximum sparse teacher queries in a Python chunk.
        ratio: Compression ratio.
        segments: Contiguous document/local-Q intersections and K prefixes.
    """
    # Zero-length views attach zero gradients even when no segment has a key.
    total = query[..., :0].float().sum() + key[..., :0].float().sum() + weights[..., :0].float().sum()
    for start, end, key_start, key_end, residual in segments:
        if key_end <= key_start:
            continue
        selected = indices[:, start:end]
        selected = torch.where(selected >= 0, selected - key_start, -1)
        if selected.shape[-1] < 512:
            selected = torch.nn.functional.pad(selected, (0, 512-selected.shape[-1]), value=-1)
        loss = _CausalCompressedKLLoss.apply(
            query[:, start:end], key[:, key_start:key_end], weights[:, start:end],
            attention_query[:, :, start:end], main_key[:, key_start:key_end], selected, sinks,
            scale, coefficient, chunk_size, ratio, residual,
        )
        total = total + loss * ((end-start) / query.shape[1])
    return total
