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
"""CANN CSA2 kernels with the existing HyperParallel Indexer objective."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Any

import torch  # pylint: disable=forbidden-backend-import


def _load_cann_ops() -> None:
    """Register optional CANN operators only when the supported NPU path runs."""
    import cann_ops_transformer  # noqa: F401  # pylint: disable=C0415,unused-import


@lru_cache(maxsize=1)
def _cann_installed() -> bool:
    """Check the optional package without loading its native extensions."""
    return find_spec("cann_ops_transformer") is not None


def supports_cann_attention(
        query: torch.Tensor,
        raw_key: torch.Tensor,
        compressed_key: torch.Tensor,
        indices: torch.Tensor,
        *,
        compress_ratio: int,
        sliding_window: int,
        query_offset: int = 0,
) -> bool:
    """Check the geometry validated for CANN 9.1 SparseFlashMla CSA."""
    if query.ndim != 4 or raw_key.ndim != 4 or compressed_key.ndim != 3 or indices.ndim != 3:
        return False
    if query.device.type != "npu":
        return False
    if not _cann_installed():
        return False
    batch, heads, length, head_dim = query.shape
    raw_length = raw_key.shape[2]
    compressed_length = compressed_key.shape[1]
    return all((
        query.dtype in (torch.float16, torch.bfloat16),
        batch == 1,
        heads in (1, 2, 4, 8, 16, 32, 64, 128),
        head_dim == 512,
        raw_key.shape == (batch, 1, raw_length, head_dim),
        compressed_key.shape == (batch, compressed_length, head_dim),
        raw_key.dtype == query.dtype == compressed_key.dtype,
        indices.shape == (batch, length, 512),
        compressed_length * compress_ratio == raw_length,
        # CANN 9.1's sparse CSA port handles ratio 4; ratio 1 is SWA-only there.
        compress_ratio == 4,
        length % 4 == 0,
        sliding_window == 128,
        query_offset >= 0,
        query_offset % 4 == 0,
        query_offset + length <= raw_length,
    ))


def supports_cann_indexer_kl(
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        merge_weight: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        compress_ratio: int | None,
        query_offset: int = 0,
) -> bool:
    """Restrict SLIKG to the NPU geometry numerically validated here."""
    if index_query.ndim != 4 or index_key.ndim != 3 or merge_weight.ndim != 3 or topk_indices.ndim != 3:
        return False
    if index_query.device.type != "npu":
        return False
    if not _cann_installed():
        return False
    if compress_ratio not in (1, 2, 4):
        return False
    batch, length, heads, head_dim = index_query.shape
    return all((
        index_query.dtype in (torch.float16, torch.bfloat16),
        batch == 1,
        heads in (32, 64),
        head_dim == 128,
        length % compress_ratio == 0,
        index_key.shape[0] == batch,
        index_key.shape[2] == head_dim,
        query_offset >= 0,
        query_offset % compress_ratio == 0,
        query_offset + length <= index_key.shape[1] * compress_ratio,
        index_key.dtype == index_query.dtype,
        merge_weight.shape == (batch, length, heads),
        topk_indices.shape == (batch, length, 512),
        length >= 512,
    ))


def supports_cann_indexer_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        sparse_count: int,
        query_offset: int,
) -> bool:
    """Select the validated full-key CSA2 geometry for LightningIndexer."""
    if query.ndim != 4 or key.ndim != 3 or merge_weight.ndim != 3:
        return False
    if query.device.type != "npu":
        return False
    if not _cann_installed():
        return False
    if compress_ratio not in (1, 2, 4):
        return False
    batch, length, heads, head_dim = query.shape
    compressed_length = key.shape[1]
    return all((
        batch == 1,
        query.dtype in (torch.float16, torch.bfloat16),
        key.dtype == query.dtype,
        heads in (32, 64),
        head_dim == 128,
        key.shape == (batch, compressed_length, head_dim),
        merge_weight.shape == (batch, length, heads),
        sparse_count == 512,
        compressed_length <= 2 ** 24,
        length >= 1024,
        length % compress_ratio == 0,
        query_offset >= 0,
        query_offset % compress_ratio == 0,
        query_offset + length <= compressed_length * compress_ratio,
    ))


def cann_indexer_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        merge_weight: torch.Tensor,
        *,
        compress_ratio: int,
        query_offset: int,
) -> torch.Tensor:
    """Run the full-key Indexer with the CSA2 causal compression mask."""
    _load_cann_ops()
    _, length, heads, head_dim = query.shape
    compressed_length = (query_offset + length) // compress_ratio
    options = {
        "cu_seqlens_q": torch.tensor([0, length], dtype=torch.int32, device=query.device),
        "cu_seqlens_k": torch.tensor([0, compressed_length], dtype=torch.int32, device=query.device),
        "cmp_residual_k": torch.zeros(1, dtype=torch.int32, device=query.device),
        "layout_q": "TND",
        "layout_k": "TND",
        "mask_mode": 3,
        "cmp_ratio": compress_ratio,
    }
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        heads, 1, head_dim, 512, batch_size=1, max_seqlen_q=length,
        max_seqlen_k=compressed_length, **options,
    )
    indices, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query.reshape(length, heads, head_dim).contiguous(),
        key[:, :compressed_length].reshape(compressed_length, 1, head_dim).contiguous(),
        merge_weight.reshape(length, heads).float().contiguous(),
        512, metadata=metadata, return_value=0, **options,
    )
    # Both kernels choose a valid Top-K set. Match the existing ascending-id
    # layout and keep invalid slots at the end for attention and KL kernels.
    indices = indices.reshape(1, length, 512)
    indices = indices.masked_fill(indices < 0, compressed_length)
    indices = indices.float().sort(dim=-1).values.to(torch.int32)
    return indices.masked_fill(indices == compressed_length, -1)


def _attention_options(
        query_length: int,
        raw_length: int,
        compressed_length: int,
        device: torch.device,
) -> dict[str, Any]:
    """Build the shared SparseFlashMla sequence-layout and mask options."""
    query_lengths = torch.tensor([0, query_length], dtype=torch.int32, device=device)
    raw_lengths = torch.tensor([0, raw_length], dtype=torch.int32, device=device)
    compressed_lengths = torch.tensor([0, compressed_length], dtype=torch.int32, device=device)
    residual = torch.zeros(1, dtype=torch.int32, device=device)
    return {
        "cu_seqlens_q": query_lengths,
        "cu_seqlens_ori_kv": raw_lengths,
        "cu_seqlens_cmp_kv": compressed_lengths,
        "cmp_residual_kv": residual,
        "cmp_ratio": 4,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "TND",
    }


class _SparseFlashMLA(torch.autograd.Function):
    """Autograd bridge for joint window, compressed KV and sink attention."""

    @staticmethod
    def forward(
            ctx: Any,
            query: torch.Tensor,
            raw_key: torch.Tensor,
            compressed_key: torch.Tensor,
            indices: torch.Tensor,
            sinks: torch.Tensor,
            scale: float,
    ) -> torch.Tensor:
        """Run joint sparse attention and keep its LSE for the paired backward."""
        _load_cann_ops()
        length, heads, head_dim = query.shape
        options = _attention_options(length, raw_key.shape[0], compressed_key.shape[0], query.device)
        metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
            heads, 1, head_dim, ori_topk=0, cmp_topk=indices.shape[-1],
            has_ori_kv=True, has_cmp_kv=True, **options,
        )
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            query, ori_kv=raw_key, cmp_kv=compressed_key,
            cmp_sparse_indices=indices, sinks=sinks, metadata=metadata,
            softmax_scale=scale, return_softmax_lse=True, **options,
        )
        ctx.save_for_backward(query, raw_key, compressed_key, indices, sinks, output, lse)
        ctx.options = options
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return attention gradients from SparseFlashMlaGrad."""
        query, raw_key, compressed_key, indices, sinks, output, lse = ctx.saved_tensors
        heads, head_dim = query.shape[1:]
        metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_grad_metadata(
            heads, 1, head_dim, ori_topk=0, cmp_topk=indices.shape[-1],
            has_ori_kv=True, has_cmp_kv=True, **ctx.options,
        )
        grad_query, grad_raw, grad_compressed, grad_sinks, _, _ = (
            torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
                query, grad_output.contiguous(), output, lse,
                ori_kv=raw_key, cmp_kv=compressed_key, cmp_sparse_indices=indices,
                sinks=sinks, metadata=metadata, softmax_scale=ctx.scale, **ctx.options,
            )
        )
        return grad_query, grad_raw, grad_compressed, None, grad_sinks, None


def cann_sparse_attention(
        query: torch.Tensor,
        raw_key: torch.Tensor,
        compressed_key: torch.Tensor,
        indices: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
        *,
        query_offset: int = 0,
) -> torch.Tensor:
    """Run the supported CSA geometry through one SparseFlashMla call."""
    _, heads, length, head_dim = query.shape
    raw_length = query_offset + length
    compressed_length = raw_length // 4
    # CANN 9.1 backward requires null seqused tensors. A prefix view lets its
    # right-down mask anchor this rank's local queries at the same global offset.
    raw_key = raw_key[:, :, :raw_length]
    compressed_key = compressed_key[:, :compressed_length]
    query_tnd = query.transpose(1, 2).reshape(length, heads, head_dim).contiguous()
    raw_tnd = raw_key.transpose(1, 2).reshape(raw_length, 1, head_dim).contiguous()
    compressed_tnd = compressed_key.reshape(compressed_length, 1, head_dim).contiguous()
    indices_tnd = indices.reshape(length, 1, indices.shape[-1]).to(torch.int32).contiguous()
    output = _SparseFlashMLA.apply(
        query_tnd, raw_tnd, compressed_tnd, indices_tnd, sinks.float(), scale,
    )
    return output.reshape(1, length, heads, head_dim)


def _indexer_target(
        attention_query: torch.Tensor,
        compressed_key: torch.Tensor,
        indices: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
        chunk_size: int,
) -> torch.Tensor:
    """Preserve the existing selected-key-and-sink teacher distribution.

    SparseFlashMlaGrad returns mass normalized over the window and sink too;
    substituting it here would change the Indexer training objective.
    """
    length = indices.shape[1]
    use_dense_scores = (
        attention_query.device.type == "npu"
        and length >= 2048
        and compressed_key.shape[1] <= 2 * indices.shape[-1]
    )
    dense_key = compressed_key.float().transpose(1, 2).unsqueeze(1) if use_dense_scores else None
    targets = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        selected = indices[:, start:end]
        valid = selected >= 0
        if use_dense_scores:
            # Gather FP32 scores after GEMM instead of materializing selected
            # keys for every query and head. Keep all keys for arbitrary indices.
            all_scores = torch.matmul(attention_query[:, :, start:end].float(), dense_key)
            score_indices = selected.clamp_min(0).long().unsqueeze(1).expand(
                -1, attention_query.shape[1], -1, -1,
            )
            scores = all_scores.gather(-1, score_indices) * scale
        else:
            selected_key = compressed_key.index_select(
                1, selected.clamp_min(0).long().reshape(-1)
            ).reshape(1, end - start, selected.shape[-1], compressed_key.shape[-1]).float()
            scores = torch.einsum(
                "bhcd,bckd->bhck", attention_query[:, :, start:end].float(), selected_key
            ) * scale
        scores.masked_fill_(~valid.unsqueeze(1), -1.0e9)
        sink_logits = sinks.float().view(1, -1, 1, 1).expand(1, -1, end - start, -1)
        target = torch.cat((scores, sink_logits), dim=-1).softmax(dim=-1)
        target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0.0).sum(dim=1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        targets.append(target)
    return torch.cat(targets, dim=1)


class _CANNIndexerKLLoss(torch.autograd.Function):
    """Fuse Indexer gradients while retaining the current KL target and scalar."""

    @staticmethod
    def forward(
            ctx: Any,
            index_query: torch.Tensor,
            index_key: torch.Tensor,
            merge_weight: torch.Tensor,
            attention_query: torch.Tensor,
            compressed_key: torch.Tensor,
            indices: torch.Tensor,
            sinks: torch.Tensor,
            attention_scale: float,
            loss_coeff: float,
            chunk_size: int,
            compress_ratio: int,
    ) -> torch.Tensor:
        """Build the current teacher and compute fused Indexer gradients."""
        _load_cann_ops()
        length, heads, head_dim = index_query.shape[1:]
        compressed_length = index_key.shape[1]
        with torch.no_grad():
            target = _indexer_target(
                attention_query, compressed_key, indices, sinks, attention_scale, chunk_size,
            )
            query_lengths = torch.tensor([0, length], dtype=torch.int32, device=index_query.device)
            key_lengths = torch.tensor([0, compressed_length], dtype=torch.int32, device=index_query.device)
            residual = torch.zeros(1, dtype=torch.int32, device=index_query.device)
            options = {
                "cu_seqlens_q": query_lengths,
                "cu_seqlens_k": key_lengths,
                "cmp_residual_k": residual,
                "layout_q": "TND",
                "layout_k": "TND",
                "mask_mode": 3,
                "cmp_ratio": compress_ratio,
            }
            metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
                heads, 1, head_dim, batch_size=1, max_seqlen_q=length,
                max_seqlen_k=compressed_length, topk=indices.shape[-1], **options,
            )
            grad_query, grad_key, grad_weight, prediction = (
                torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
                    index_query.reshape(length, heads, head_dim).contiguous(),
                    index_key.reshape(compressed_length, 1, head_dim).contiguous(),
                    merge_weight.reshape(length, heads).float().contiguous(),
                    indices.reshape(length, 1, -1).to(torch.int32).contiguous(),
                    (target * (loss_coeff / length)).reshape(length, 1, -1).contiguous(),
                    metadata=metadata, **options,
                )
            )
            tiny = torch.finfo(torch.float32).tiny
            row_loss = target * (
                target.clamp_min(tiny).log()
                - prediction.reshape(1, length, -1).clamp_min(tiny).log()
            )
            loss = row_loss.sum() * (loss_coeff / length)
        ctx.save_for_backward(
            grad_query.reshape_as(index_query),
            grad_key.reshape_as(index_key),
            grad_weight.reshape_as(merge_weight),
        )
        return loss

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple:
        """Scale the saved gradients with the auxiliary-loss gradient."""
        grad_query, grad_key, grad_weight = ctx.saved_tensors
        return (
            grad_query * grad_loss,
            grad_key * grad_loss,
            grad_weight * grad_loss,
            *((None,) * 8),
        )


def cann_indexer_kl_loss(
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        merge_weight: torch.Tensor,
        attention_query: torch.Tensor,
        compressed_key: torch.Tensor,
        indices: torch.Tensor,
        sinks: torch.Tensor,
        *,
        attention_scale: float,
        loss_coeff: float,
        query_chunk_size: int,
        compress_ratio: int,
        query_offset: int = 0,
) -> torch.Tensor:
    """Calculate the existing KL objective using CANN Indexer gradients."""
    key_prefix = (query_offset + index_query.shape[1]) // compress_ratio
    return _CANNIndexerKLLoss.apply(
        index_query, index_key[:, :key_prefix], merge_weight, attention_query.detach(),
        compressed_key.detach(), indices, sinks.detach(), attention_scale,
        loss_coeff, query_chunk_size, compress_ratio,
    )
