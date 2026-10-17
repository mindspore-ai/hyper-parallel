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
"""SparseFlashMla forward/backward adapter for CSA with scalar attention sinks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import


def pack_ranges(tensor: torch.Tensor, ranges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Pack token ranges without copying an already contiguous partition.

    Args:
        tensor: Source tensor with tokens on the first dimension.
        ranges: Nonempty, ordered half-open token intervals.

    Returns:
        Contiguous packed tokens; autograd restores gradients to the source.
    """
    if all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])):
        start, end = ranges[0][0], ranges[-1][1]
        if start == 0 and end == tensor.shape[0]:
            return tensor.contiguous()
        return tensor[start:end].contiguous()
    return torch.cat([tensor[start:end] for start, end in ranges], dim=0).contiguous()


@dataclass(frozen=True)
class CSATNDPlan:
    """Document-local lengths and canonical K offsets; no training activations."""

    segments: tuple[tuple[int, int, int, int, int], ...]
    q_ranges: tuple[tuple[int, int], ...]
    k_ranges: tuple[tuple[int, int], ...]
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    residuals: tuple[int, ...]

    def query(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack a [1,Q,...] tensor in the plan's document order."""
        return pack_ranges(tensor.squeeze(0), self.q_ranges)

    def key(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack complete visible document prefixes from a [1,K,...] bank."""
        return pack_ranges(tensor.squeeze(0), self.k_ranges)

    def lengths(self, device: torch.device) -> tuple:
        """Reuse device length tensors, including across inference and recomputation."""
        return _device_lengths(self, device)

    def indexer_options(self, device: torch.device, ratio: int) -> dict:
        """Share the document partition and causal mask between Indexer and KL."""
        cu_q, cu_k, residuals, _ = self.lengths(device)
        return {"cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k,
                "cmp_residual_k": residuals if ratio > 1 else None,
                "layout_q": "TND", "layout_k": "TND", "mask_mode": 3, "cmp_ratio": ratio}

    def local_indices(self, indices: torch.Tensor, padded_width: int = 0) -> torch.Tensor:
        """Translate bank IDs; pad only when the consumer explicitly requires a width."""
        selected = self.query(indices)
        offsets = self.lengths(indices.device)[3]
        selected = torch.where(selected >= 0, selected - offsets, -1).int()
        if selected.shape[-1] < padded_width:
            selected = torch.nn.functional.pad(selected, (0, padded_width-selected.shape[-1]), value=-1)
        return selected.unsqueeze(1).contiguous()

    def covers_queries(self, length: int) -> bool:
        """Return whether packed rows already follow the complete original Q order."""
        return (self.q_ranges[0][0] == 0 and self.q_ranges[-1][1] == length
                and all(left[1] == right[0] for left, right in zip(self.q_ranges, self.q_ranges[1:])))

    def restore(self, part: torch.Tensor, output: torch.Tensor) -> None:
        """Write packed query rows into their original positions, preserving autograd."""
        cursor = 0
        for start, end in self.q_ranges:
            output[0, start:end] = part[cursor:cursor+end-start]
            cursor += end-start


@lru_cache(maxsize=128)
def csa_tnd_plan(segments: tuple, ratio: int, has_keys: bool = True) -> CSATNDPlan | None:
    """Separate true SWA-only/empty-prefix queries without inventing valid keys."""
    active = tuple(item for item in segments
                   if item[1] > item[0] and (bool(ratio and item[3] > item[2]) == has_keys))
    if not active:
        return None
    return CSATNDPlan(
        active, tuple((s, e) for s, e, _, _, _ in active), tuple((s, e) for _, _, s, e, _ in active),
        tuple(e-s for s, e, _, _, _ in active), tuple(e-s for _, _, s, e, _ in active),
        tuple(r for _, _, _, _, r in active),
    )


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _device_lengths(plan: CSATNDPlan, device: torch.device) -> tuple:
    """Cache by the complete document partition, not merely by total Q/K sizes."""
    cu_q = torch.tensor((0, *accumulate(plan.q_lengths)), device=device, dtype=torch.int32)
    cu_k = torch.tensor((0, *accumulate(plan.k_lengths)), device=device, dtype=torch.int32)
    residuals = torch.tensor(plan.residuals, device=device, dtype=torch.int32)
    offsets = torch.tensor([s for s, _ in plan.k_ranges], device=device, dtype=torch.int32)
    lengths = torch.tensor(plan.q_lengths, device=device, dtype=torch.int64)
    offsets = torch.repeat_interleave(offsets, lengths, output_size=sum(plan.q_lengths)).unsqueeze(1)
    return cu_q, cu_k, residuals, offsets


@lru_cache(maxsize=1)
def _load_attention_ops() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Resolve attention forward, backward and metadata through the official binding."""
    # Lazy import leaves the eager baseline usable without the NPU extension.
    try:
        from cann_ops_transformer import (  # pylint: disable=import-outside-toplevel
            sparse_flash_mla as sparse_flash_mla_op,
            sparse_flash_mla_grad, sparse_flash_mla_metadata,
        )
    except ImportError as err:
        raise RuntimeError(
            "use_fused_ops=True requires cann_ops_transformer.sparse_flash_mla, sparse_flash_mla_grad "
            "and sparse_flash_mla_metadata. "
            "See hyper_parallel/components/functional/README.md for ops-transformer installation, "
            "environment setup and tested versions."
        ) from err
    return sparse_flash_mla_op, sparse_flash_mla_grad, sparse_flash_mla_metadata


def supports_fused_attention_inputs(query: torch.Tensor, ratio: int, window: int, topk: int = 512) -> bool:
    """Check the validated attention input contract without importing operators.

    Args:
        query: Main Q [batch, heads, queries, dimension].
        ratio: Model ratio, zero for a pure SWA layer.
        window: Raw attention window, including the current token.
        topk: Logical compressed selection width, before padding.
    """
    if query.device.type != "npu" or query.dtype != torch.bfloat16:
        return False
    if query.shape[-1] != 512 or window != 128 or ratio not in (0, 1, 2):
        return False
    if query.shape[1] not in (1, 2, 4, 8, 16, 32, 64, 128) or query.shape[2] == 0:
        return False
    if ratio and not 1 <= topk <= 512:
        return False
    return True


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _attention_metadata(
        device: torch.device, batch: int, length: int, heads: int, dim: int,
        raw_length: int, main_length: int, topk: int, ratio: int, residual: int, window: int,
) -> tuple:
    """Reuse shape metadata across evaluation and gradient-tracked training.

    Evaluation may populate the cache under inference_mode. The backward
    saves the cached residual, so it must be an ordinary tensor regardless
    of which mode first populated the cache.
    """
    # The training backward requires a zero residual tensor even at ratio one;
    # the forward interface instead omits that optional input at ratio one.
    residue = torch.full((batch,), residual, dtype=torch.int32, device=device) if main_length else None
    _, _, metadata_op = _load_attention_ops()
    metadata = metadata_op(
        heads, 1, dim, cmp_residual_kv=residue if ratio > 1 else None, batch_size=batch, max_seqlen_q=length,
        max_seqlen_ori_kv=raw_length, max_seqlen_cmp_kv=main_length, ori_topk=0, cmp_topk=topk,
        cmp_ratio=ratio, ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=window-1, ori_win_right=0,
        layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=bool(main_length),
    )
    return residue, metadata


class _SparseFlashMla(torch.autograd.Function):
    """Wrap SparseFlashMla and SparseFlashMlaGrad without substituting the KL teacher."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, raw: torch.Tensor, main: torch.Tensor | None,
            indices: torch.Tensor | None, sinks: torch.Tensor, scale: float,
            ratio: int, residual: int, window: int,
    ) -> torch.Tensor:
        """Save the native output and LSE required by the training backward."""
        batch, length, heads, dim = query.shape
        residue, metadata = _attention_metadata(
            query.device, batch, length, heads, dim, raw.shape[1], 0 if main is None else main.shape[1],
            0 if indices is None else indices.shape[-1], ratio, residual, window,
        )
        options = {"softmax_scale": scale, "cmp_ratio": ratio, "ori_mask_mode": 4, "cmp_mask_mode": 3,
                   "ori_win_left": window-1, "ori_win_right": 0, "layout_q": "BSND", "layout_kv": "BSND"}
        attention, _, _ = _load_attention_ops()
        output, lse = attention(
            query, ori_kv=raw, cmp_kv=main, cmp_sparse_indices=indices, sinks=sinks,
            cmp_residual_kv=residue if ratio > 1 else None, metadata=metadata, return_softmax_lse=True, **options,
        )
        ctx.save_for_backward(query, raw, main, indices, sinks, residue, output, lse)
        ctx.options = options
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return main-branch gradients without substituting the KL teacher."""
        query, raw, main, indices, sinks, residue, output, lse = ctx.saved_tensors
        _, attention_grad, _ = _load_attention_ops()
        grad_query, grad_raw, grad_main, grad_sinks, _, _ = attention_grad(
            query, grad_output.contiguous(), output, lse, ori_kv=raw, cmp_kv=main,
            cmp_sparse_indices=indices, sinks=sinks, cmp_residual_kv=residue, **ctx.options,
        )
        return grad_query, grad_raw, grad_main if main is not None else None, None, grad_sinks, None, None, None, None


def _bsnd_sparse_flash_mla(
        query: torch.Tensor, raw_key: torch.Tensor, main_key: torch.Tensor | None,
        indices: torch.Tensor | None, sinks: torch.Tensor, scale: float, ratio: int,
        window: int, query_offset: int,
        segment: tuple[int, int, int, int, int],
) -> torch.Tensor:
    """Run one document segment per batch item using the original all-gather banks.

    The operator owns Q tiling within this BSND call. Multiple packed
    documents use the TND entry with cumulative document lengths.

    Args:
        query: Local attention Q in [B, H, Q, D].
        raw_key: Global raw KV in [B, 1, K, D].
        main_key: Global compressed KV in [B, Kc, D], or None for SWA only.
        indices: Global compressed selection [B, Q, topk], invalid slots -1.
        sinks: One scalar logit per head.
        scale: Attention dot-product scale.
        ratio: Compression ratio, zero denotes the SWA-only layer.
        window: Number of raw tokens visible to each query including itself.
        query_offset: Global position of the first local query.
        segment: Local Q start/end, compressed document start/prefix end and residual,
            shared by all physical batch items.

    Returns:
        Attention output in [B, Q, H, D].
    """
    if query.dtype != torch.bfloat16 or query.shape[-1] != 512 or window != 128:
        raise ValueError("V4 attention requires BF16, head_dim=512, and window=128")
    start, end, key_start, key_end, residual = segment
    document_start = key_start * max(ratio, 1)
    raw_left = max(document_start, query_offset + start - window + 1)
    raw_right = query_offset + end
    raw = raw_key[:, :, raw_left:raw_right].transpose(1, 2).contiguous()
    main = None
    selected = None
    if ratio and key_end > key_start:
        main = main_key[:, key_start:key_end].unsqueeze(2).contiguous()
        original = indices[:, start:end]
        selected = torch.where(original >= 0, original-key_start, -1).int()
        # Backward infers the active count from this width and causality; only KL requires 512 slots.
        selected = selected.unsqueeze(2).contiguous()
    return _SparseFlashMla.apply(
        query[:, :, start:end].transpose(1, 2).contiguous(), raw, main, selected, sinks.float(),
        scale, max(ratio, 1) if main is not None else 1, residual if main is not None else 0, window,
    )


@lru_cache(maxsize=128)
@torch.inference_mode(False)
def _tnd_attention_metadata(plan, device, heads, dim, raw_ranges, ratio, window, has_keys, topk):
    """Describe independent raw and compressed coordinate systems to SMLA."""
    cu_q, cu_k, residuals, _ = plan.lengths(device)
    raw_lengths = tuple(end-start for start, end in raw_ranges)
    cu_raw = torch.tensor((0, *accumulate(raw_lengths)), device=device, dtype=torch.int32)
    _, _, metadata_op = _load_attention_ops()
    options = {"layout_q": "TND", "layout_kv": "TND", "cmp_ratio": ratio, "ori_mask_mode": 4,
               "cmp_mask_mode": 3, "ori_win_left": window-1, "ori_win_right": 0}
    metadata = metadata_op(
        heads, 1, dim, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_raw,
        cu_seqlens_cmp_kv=cu_k if has_keys else None,
        cmp_residual_kv=residuals if has_keys and ratio > 1 else None,
        batch_size=len(plan.segments), max_seqlen_q=max(plan.q_lengths),
        max_seqlen_ori_kv=max(raw_lengths), max_seqlen_cmp_kv=max(plan.k_lengths) if has_keys else 0,
        ori_topk=0, cmp_topk=topk, has_ori_kv=True, has_cmp_kv=has_keys, **options,
    )
    return cu_q, cu_raw, cu_k if has_keys else None, residuals if has_keys else None, metadata


class _SparseFlashMlaTND(torch.autograd.Function):
    """Native multi-document forward/backward; save all sequence tensors through autograd."""

    @staticmethod
    def forward(
            ctx: Any, query: torch.Tensor, raw: torch.Tensor, main: torch.Tensor | None,
            indices: torch.Tensor | None, sinks: torch.Tensor, scale: float, ratio: int, window: int,
            plan: CSATNDPlan, raw_ranges: tuple,
    ) -> torch.Tensor:
        """Save native output/LSE and independent raw/compressed cumulative lengths."""
        cu_q, cu_raw, cu_k, residuals, metadata = _tnd_attention_metadata(
            plan, query.device, query.shape[1], query.shape[2], raw_ranges, ratio, window, main is not None,
            0 if indices is None else indices.shape[-1],
        )
        options = {"layout_q": "TND", "layout_kv": "TND", "softmax_scale": scale, "cmp_ratio": ratio,
                   "ori_mask_mode": 4, "cmp_mask_mode": 3, "ori_win_left": window-1, "ori_win_right": 0}
        forward, _, _ = _load_attention_ops()
        output, lse = forward(
            query, ori_kv=raw, cmp_kv=main, cmp_sparse_indices=indices, sinks=sinks,
            cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_raw, cu_seqlens_cmp_kv=cu_k,
            cmp_residual_kv=residuals if ratio > 1 else None, metadata=metadata,
            return_softmax_lse=True, **options,
        )
        ctx.save_for_backward(query, raw, main, indices, sinks, output, lse, cu_q, cu_raw, cu_k, residuals)
        ctx.options = options
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return each physical bank's gradient to its autograd communication owner."""
        query, raw, main, indices, sinks, output, lse, cu_q, cu_raw, cu_k, residuals = ctx.saved_tensors
        _, backward, _ = _load_attention_ops()
        dq, dr, dm, ds, _, _ = backward(
            query, grad_output.contiguous(), output, lse, ori_kv=raw, cmp_kv=main,
            cmp_sparse_indices=indices, sinks=sinks, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_raw,
            cu_seqlens_cmp_kv=cu_k, cmp_residual_kv=residuals, **ctx.options,
        )
        return dq, dr, dm if main is not None else None, None, ds, None, None, None, None, None


def sparse_flash_mla(
        query: torch.Tensor, raw_key: torch.Tensor, main_key: torch.Tensor | None,
        indices: torch.Tensor | None, sinks: torch.Tensor, scale: float, ratio: int,
        window: int, query_offset: int, segments: tuple,
) -> torch.Tensor:
    """Batch packed documents in native TND while preserving the all-gather banks.

    Args:
        query: Local attention Q [batch, heads, queries, 512].
        raw_key: Global raw KV [batch, 1, keys, 512].
        main_key: Global compressed KV [batch, compressed keys, 512], or None.
        indices: Sorted global compressed key IDs, invalid slots -1.
        sinks: Per-head scalar sink logits.
        scale: Attention dot-product scale.
        ratio: Compression ratio; zero selects pure SWA.
        window: Raw sliding-window size, including the current token.
        query_offset: Global position of the first local query.
        segments: Local Q start/end, compressed document start/prefix end and residual.

    Returns:
        Attention output [batch, queries, heads, 512]. A single document or
        ordinary physical batch uses BSND. Empty compressed prefixes form a
        separate SWA-only TND group without inventing compressed keys.
    """
    if len(segments) == 1:
        return _bsnd_sparse_flash_mla(
            query, raw_key, main_key, indices, sinks, scale, ratio, window, query_offset, segments[0],
        )
    if query.shape[0] != 1:
        raise ValueError("Packed Sparse Flash MLA requires batch size one")
    if query.dtype != torch.bfloat16 or query.shape[-1] != 512 or window != 128:
        raise ValueError("V4 attention requires BF16, head_dim=512, and window=128")
    output = None
    for has_keys in (True, False):
        plan = csa_tnd_plan(segments, ratio, has_keys)
        if plan is None:
            continue
        raw_ranges = tuple(
            (max(ks * max(ratio, 1), query_offset+s-window+1), query_offset+e)
            for s, e, ks, _, _ in plan.segments
        )
        raw = pack_ranges(raw_key.squeeze(0).squeeze(0), raw_ranges).unsqueeze(1)
        main = plan.key(main_key).unsqueeze(1) if has_keys else None
        selected = plan.local_indices(indices) if has_keys else None
        part = _SparseFlashMlaTND.apply(
            plan.query(query.transpose(1, 2)), raw, main, selected, sinks.float(),
            scale, max(ratio, 1) if has_keys else 1, window, plan, raw_ranges,
        )
        # A complete Q partition is already ordered; a canvas would retain another full output.
        if plan.covers_queries(query.shape[2]):
            return part.unsqueeze(0)
        if output is None:
            output = torch.zeros_like(query.transpose(1, 2))
        plan.restore(part, output)
    return torch.zeros_like(query.transpose(1, 2)) if output is None else output
