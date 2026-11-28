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
"""Causal attention over selected original-token blocks, with an implicit tail."""
from __future__ import annotations

import math
from typing import Any, Callable, Protocol

import torch
from torch.autograd.function import once_differentiable

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class BlockSparseAttentionFn(Protocol):
    """BSHD Q/K/V; complete block IDs [B,S,K], unique, with -1 padding.

    All heads share routing. The visible incomplete block is always included.
    Return BSHD output and detached FP32 LSE [B,S,Hq]. No dropout or cache.
    """

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 indices: torch.Tensor, *, block_size: int, scale: float,
                 query_chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend to complete blocks and the visible tail without expanding pooled main KV."""


BLOCK_SPARSE_ATTENTION = OpSpec[BlockSparseAttentionFn](config_key="block_sparse_attention")


class BlockSparseMLAFn(Protocol):
    """Absorbed Q [B,S,H,D] and one shared KV bank [B,S,1,D]."""

    def __call__(self, query: torch.Tensor, key_value: torch.Tensor, indices: torch.Tensor, *,
                 block_size: int, scale: float,
                 query_chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absorbed output and detached LSE, retaining the model's original scale."""


BLOCK_SPARSE_MLA = OpSpec[BlockSparseMLAFn](config_key="block_sparse_attention", variant="mla")


def validate_inputs(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, indices: torch.Tensor,
    block_size: int, scale: float, query_chunk_size: int,
) -> None:
    """Validate structural constraints without synchronizing device routing values."""
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("block sparse attention expects BSHD Q/K/V with matching K/V shapes")
    batch, length, heads, dim = query.shape
    if min(query.shape) < 1 or key.shape[:2] != (batch, length) or key.shape[-1] != dim:
        raise ValueError("block sparse attention requires positive, equal Q/K sequence and head dimensions")
    if key.shape[2] < 1 or heads % key.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if indices.ndim != 3 or indices.shape[:2] != (batch, length) or indices.shape[-1] < 1:
        raise ValueError("complete block indices must have shape [B,S,K] with nonempty capacity")
    if indices.dtype != torch.int32 or indices.device != query.device:
        raise ValueError("block indices must be INT32 on the query device")
    if any(t.dtype != query.dtype or t.device != query.device for t in (key, value)):
        raise ValueError("Q/K/V must have matching dtype and device")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (block_size, query_chunk_size)):
        raise ValueError("block size and query chunk size must be positive integers")
    if not math.isfinite(scale):
        raise ValueError("attention scale must be finite")


def complete_block_pattern(indices: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Put valid complete blocks first, sorted in source order for the CANN ABI."""
    length = indices.shape[1]
    complete = torch.arange(1, length + 1, device=indices.device) // block_size
    valid = (indices >= 0) & (indices < complete[None, :, None])
    sentinel = length // block_size + 1
    ordered = torch.where(valid, indices, sentinel).sort(dim=-1).values
    return torch.where(ordered < sentinel, ordered, 0).to(torch.int32), valid.sum(-1).to(torch.int32)


def block_token_mask(indices: torch.Tensor, block_size: int, start: int, end: int) -> torch.Tensor:
    """Build only a query chunk's token mask; never a persistent full S-by-S mask."""
    length = indices.shape[1]
    positions = torch.arange(start, end, device=indices.device)
    tokens = torch.arange(length, device=indices.device)
    complete = (positions + 1) // block_size
    ids = indices[:, start:end]
    valid = (ids >= 0) & (ids < complete[None, :, None])
    selected = torch.zeros(ids.shape[0], end-start, length // block_size + 1,
                           device=ids.device, dtype=torch.int32)
    selected.scatter_add_(-1, ids.clamp(0, length // block_size).long(), valid.to(torch.int32))
    selected = selected.index_select(-1, tokens // block_size) > 0
    tail = (tokens[None, :] >= complete[:, None] * block_size) & (tokens[None, :] <= positions[:, None])
    return (selected | tail[None]) & (tokens[None, None, :] <= positions[None, :, None])


def native_block_sparse_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, indices: torch.Tensor, *,
    block_size: int, scale: float, query_chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch execution; grouped heads avoid physically replicating KV."""
    batch, length, heads, dim = query.shape
    kv_heads = key.shape[2]
    outputs, normalizers = [], []
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    keys = key.to(dtype)
    # A shared latent bank must sum its K and V contributions before the dtype cast.
    values = keys if key is value else value.to(dtype)
    for start in range(0, length, query_chunk_size):
        end = min(start + query_chunk_size, length)
        q = query[:, start:end].to(dtype).reshape(batch, end-start, kv_heads, heads // kv_heads, dim)
        scores = torch.einsum("bqhgd,bkhd->bhgqk", q, keys) * scale
        valid = block_token_mask(indices, block_size, start, end)
        scores = scores.masked_fill(~valid[:, None, None], float("-inf"))
        active = valid.any(-1)
        scores = torch.where(active[:, None, None, :, None], scores, torch.zeros_like(scores))
        probs = scores.softmax(-1) * active[:, None, None, :, None]
        output = torch.einsum("bhgqk,bkhd->bqhgd", probs, values)
        outputs.append(output.reshape(batch, end-start, heads, dim).to(query.dtype))
        lse = scores.logsumexp(-1).permute(0, 3, 1, 2).reshape(batch, end-start, heads)
        normalizers.append(lse.masked_fill(~active[..., None], float("-inf")))
    return torch.cat(outputs, dim=1), torch.cat(normalizers, dim=1).float().detach()


class _CANNBlockAttention(torch.autograd.Function):
    """Pair official forward/backward kernels; LSE is a detached teacher statistic."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, indices: torch.Tensor,
        block_size: int, scale: float, kernels: tuple[Callable, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run QSAG forward and retain the state its native gradient kernel needs."""
        forward, metadata_op, backward = kernels
        batch, length, heads, dim = query.shape
        kv_heads = key.shape[2]
        ids, counts = complete_block_pattern(indices, block_size)
        fwd_ids = ids.reshape(1, batch * length, -1).expand(kv_heads, -1, -1).contiguous()
        fwd_counts = counts.reshape(1, batch * length).expand(kv_heads, -1).contiguous()
        cu = torch.arange(batch + 1, dtype=torch.int64, device=query.device) * length
        options = {"cu_seqlens_q": cu, "cu_seqlens_kv": cu, "layout_q": "TND", "layout_kv": "TND",
                   "layout_sparse_pattern": 4, "residual_block_mode": 1, "is_consistent_topk": False,
                   "mask_mode": 1, "quant_mode": 0, "softmax_precision": 0}
        metadata = metadata_op(fwd_ids, fwd_counts, heads, kv_heads, dim, [1, block_size],
                               max_seqlen_q=length, max_seqlen_kv=length, **options)
        q, k, v = (x.reshape(batch * length, x.shape[2], dim).contiguous() for x in (query, key, value))
        out, lse = forward(q, k, v, fwd_ids, fwd_counts, [1, block_size], metadata=metadata,
                           softmax_scale=scale, return_softmax_lse=True, **options)
        ctx.save_for_backward(q, k, v, out, lse, ids, counts, cu)
        ctx.backward_op, ctx.block_size, ctx.scale = backward, block_size, scale
        ctx.batch, ctx.length = batch, length
        result_lse = lse.reshape(batch, length, heads).detach()
        ctx.mark_non_differentiable(result_lse)
        return out.reshape_as(query), result_lse

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_output: torch.Tensor, _grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Propagate output gradients; LSE is a detached teacher statistic."""
        q, k, v, out, lse, ids, counts, cu = ctx.saved_tensors
        batch, length, kv_heads = ctx.batch, ctx.length, k.shape[1]
        ids = ids[:, None].expand(-1, kv_heads, -1, -1).contiguous()
        counts = counts[:, None].expand(-1, kv_heads, -1).contiguous()
        dq, dk, dv = ctx.backward_op(
            q, k, v, grad_output.reshape_as(q).contiguous(), out, lse, ids, counts, [1, ctx.block_size],
            cu_seqlens_q=cu, cu_seqlens_kv=cu, layout_q="TND", layout_kv="TND", layout_sparse_pattern=0,
            residual_block_mode=1, softmax_scale=ctx.scale, mask_mode=1, softmax_precision=0,
            is_consistent_topk=False,
        )
        return (dq.reshape(batch, length, *dq.shape[1:]), dk.reshape(batch, length, *dk.shape[1:]),
                dv.reshape(batch, length, *dv.shape[1:]), None, None, None, None)


def _native_mla(query, key_value, indices, *, block_size, scale, query_chunk_size=64):
    return native_block_sparse_attention(query, key_value, key_value, indices, block_size=block_size,
                                         scale=scale, query_chunk_size=query_chunk_size)


def _mla_token_pattern(indices, block_size):
    """Expand routing IDs, not KV, and compact each query's visible incomplete tail."""
    ids, counts = complete_block_pattern(indices, block_size)
    slots = torch.arange(ids.shape[-1], device=ids.device)
    tokens = ids[..., None] * block_size + torch.arange(block_size, device=ids.device)
    valid = slots < counts[..., None]
    length = indices.shape[1]
    tokens = torch.where(valid[..., None], tokens, length).flatten(-2)
    positions = torch.arange(1, length + 1, device=ids.device)
    tail = positions[:, None] // block_size * block_size + torch.arange(block_size - 1, device=ids.device)
    tail = torch.where(tail < positions[:, None], tail, length).expand(ids.shape[0], -1, -1)
    tokens = torch.cat((tokens, tail), dim=-1).sort(dim=-1).values
    lengths = (counts * block_size + (positions % block_size)[None]).unsqueeze(-1)
    return torch.where(tokens < length, tokens, -1).to(torch.int32).unsqueeze(2), lengths.to(torch.int32)


class _CANNBlockMLA(torch.autograd.Function):
    """Original-token SparseFlashMla mode; its backward sums shared-bank dK and dV."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key_value: torch.Tensor, indices: torch.Tensor,
        block_size: int, scale: float, kernels: tuple[Callable, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run absorbed forward with explicit per-query original-token routing."""
        forward, metadata_op, backward, backward_metadata = kernels
        query, key_value = query.contiguous(), key_value.contiguous()
        batch, length, heads, dim = query.shape
        tokens, lengths = _mla_token_pattern(indices, block_size)
        active = lengths > 0
        # Use a harmless visible key for empty rows, then mask their output/gradient.
        tokens[..., 0] = torch.where(active, tokens[..., 0], 0)
        lengths = lengths.clamp_min(1)
        options = {"ori_topk_length": lengths, "batch_size": batch, "max_seqlen_q": length,
                   "max_seqlen_ori_kv": length, "ori_topk": tokens.shape[-1], "cmp_topk": 0,
                   "cmp_ratio": 1, "ori_mask_mode": 0, "cmp_mask_mode": 0,
                   "layout_q": "BSND", "layout_kv": "BSND", "has_ori_kv": True, "has_cmp_kv": False}
        metadata = metadata_op(heads, 1, dim, **options)
        # The official training kernel requires sinks. exp(-inf)=0 preserves no-sink MLA.
        sink = torch.full((heads,), float("-inf"), device=query.device, dtype=torch.float32)
        output, lse = forward(query, sinks=sink, ori_kv=key_value, ori_sparse_indices=tokens,
                              ori_topk_length=lengths, metadata=metadata, softmax_scale=scale,
                              ori_mask_mode=0, return_softmax_lse=True)
        ctx.save_for_backward(query, key_value, tokens, lengths, output, lse, sink, active)
        options.pop("ori_topk_length")
        ctx.options, ctx.scale = options, scale
        ctx.backward_op, ctx.backward_metadata = backward, backward_metadata
        result_lse = lse.reshape(batch, length, heads).masked_fill(~active, float("-inf"))
        ctx.mark_non_differentiable(result_lse)
        return output.masked_fill(~active[..., None], 0), result_lse

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, grad_output: torch.Tensor, _grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Propagate output gradients; LSE is a detached teacher statistic."""
        query, key_value, tokens, lengths, output, lse, sink, active = ctx.saved_tensors
        metadata = ctx.backward_metadata(query.shape[2], 1, query.shape[-1],
                                         ori_topk_length=lengths, **ctx.options)
        gradients = ctx.backward_op(
            query, grad_output.masked_fill(~active[..., None], 0).contiguous(), output, lse,
            sinks=sink, ori_kv=key_value, ori_sparse_indices=tokens, ori_topk_length=lengths,
            metadata=metadata, softmax_scale=ctx.scale, cmp_ratio=1, ori_mask_mode=0,
            cmp_mask_mode=0, layout_q="BSND", layout_kv="BSND",
        )
        return gradients[0], gradients[1], None, None, None, None


def mla_cann_rejection(query: torch.Tensor) -> str | None:
    """Restrict the fused MLA path to the official shared D512 training ABI."""
    if query.dtype not in (torch.bfloat16, torch.float16) or query.shape[-1] != 512 or query.shape[2] > 128:
        return "CANN block MLA requires FP16/BF16, D=512 and at most 128 query heads"
    return None


def _load_native_mla():
    return _native_mla


def _load_cann_mla():
    package = require_package("cann_ops_transformer.ops")
    names = ("sparse_flash_mla", "sparse_flash_mla_metadata",
             "sparse_flash_mla_grad", "sparse_flash_mla_grad_metadata")
    for name in names:
        require_symbol(package, name)
    kernels = tuple(getattr(package, name) for name in names)

    def _operation(query, key_value, indices, *, block_size, scale, query_chunk_size=64):
        del query_chunk_size
        return _CANNBlockMLA.apply(query, key_value, indices, block_size, scale, kernels)

    return _operation


def cann_rejection(query: torch.Tensor, key: torch.Tensor, block_size: int) -> str | None:
    """Use the documented nonquantized A2/A3 QSAG training bounds."""
    group = query.shape[2] // key.shape[2]
    if query.dtype not in (torch.bfloat16, torch.float16) or query.shape[-1] > 256 or group > 256:
        return "CANN block attention training requires FP16/BF16, D<=256 and GQA<=256"
    if block_size > (128 if group <= 128 else 64):
        return "block size exceeds the CANN training tile limit"
    return None


def _load_native():
    return native_block_sparse_attention


def _load_cann():
    package = require_package("cann_ops_transformer.ops")
    names = ("generic_block_sparse_attention", "generic_block_sparse_attention_metadata",
             "generic_block_sparse_attention_grad")
    for name in names:
        require_symbol(package, name)
    kernels = tuple(getattr(package, name) for name in names)

    def _operation(query, key, value, indices, *, block_size, scale, query_chunk_size=64):
        del query_chunk_size
        return _CANNBlockAttention.apply(query, key, value, indices, block_size, scale, kernels)

    return _operation


def register_block_sparse_attention(registry: OpRegistry) -> None:
    """Register native and explicit official CANN block attention implementations."""
    registry.register_family("block_sparse_attention", specs=(BLOCK_SPARSE_ATTENTION, BLOCK_SPARSE_MLA))
    registry.register_impl(BLOCK_SPARSE_ATTENTION, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
    registry.register_impl(BLOCK_SPARSE_ATTENTION, name="cann", devices=("npu",), load=_load_cann)
    registry.register_impl(BLOCK_SPARSE_MLA, name="native", devices=("cpu", "cuda", "npu"), load=_load_native_mla)
    registry.register_impl(BLOCK_SPARSE_MLA, name="cann", devices=("npu",), load=_load_cann_mla)
