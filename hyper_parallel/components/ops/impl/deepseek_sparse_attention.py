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
"""Attention over raw and compressed KV banks in DeepSeek V4.1."""

from __future__ import annotations

from functools import partial
from typing import Any, Protocol

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention as fused_packed_attention
from hyper_parallel.components.functional.sparse_flash_mla import sparse_flash_mla, supports_fused_attention_inputs

from .._dependency import load_cann, require_package, require_symbol
from .._dsa_layout import DsaQueryLayout, packed_segments, selected_packed_keys, selected_packed_scores
from ..registry import OpRegistry
from ..spec import OpSpec


class DsaCompressedBanksFn(Protocol):
    """Attend raw and compressed KV banks in the same document coordinates."""

    def __call__(
        self, x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
        topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
        layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
        main_key_offset: int = 0, global_key_length: int | None = None,
    ) -> torch.Tensor:
        """Return attention output in BSND layout."""


DSA_COMPRESSED_BANKS = OpSpec[DsaCompressedBanksFn](config_key="deepseek_sparse_attention", variant="compressed_banks")


class PackedDeepseekSparseAttentionFn(Protocol):
    """Attend packed, absorbed MLA states and expose per-head sparse softmax statistics."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
        topk_indices: torch.Tensor, scale: float, query_ends: torch.Tensor, key_ends: torch.Tensor,
        *, sink_key: torch.Tensor | None = None, sink_value: torch.Tensor | None = None, dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded BSND output and sparse max/sum in KV-head, token, group layout."""


DSA_ABSORBED = OpSpec[PackedDeepseekSparseAttentionFn](
    config_key="deepseek_sparse_attention", variant="absorbed_with_stats",
)
DSA_ABSORBED_SINK = OpSpec[PackedDeepseekSparseAttentionFn](
    config_key="deepseek_sparse_attention", variant="absorbed_with_parameter_sink",
)


def _native_absorbed(
    x: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
    topk_indices: torch.Tensor, scale: float, query_ends: torch.Tensor, key_ends: torch.Tensor,
    *, sink_key: torch.Tensor | None = None, sink_value: torch.Tensor | None = None, dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Accumulate shared KV gradients in FP32 before casting back to input dtype;
    # casting after gathering rounds contributions from each sparse slot first.
    """Compute packed sparse MLA outputs and detached teacher statistics."""
    query, value, q_rope, k_rope = (tensor.flatten(0, 1).float() for tensor in (x, key, query_rope, key_rope))
    outputs, maxima, totals = [], [], []
    with torch.autocast(device_type=x.device.type, enabled=False):
        for qs, qe, ks, ke in packed_segments(query_ends, key_ends):
            for start in range(qs, qe, 32):
                end = min(start + 32, qe)
                indices = topk_indices[start:end]
                causal = torch.arange(start, end, device=x.device) - qe + ke - ks
                selected, valid = selected_packed_keys(value[ks:ke], indices, causal_end=causal)
                rotary_keys, _ = selected_packed_keys(k_rope[ks:ke], indices, causal_end=causal)
                scores = (selected_packed_scores(query[start:end], selected)
                          + selected_packed_scores(q_rope[start:end], rotary_keys)) * scale
                scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
                maximum = scores.amax(dim=-1, keepdim=True)
                maximum = torch.where(maximum.isfinite(), maximum, 0.0)
                exp = (scores - maximum).exp()
                total = exp.sum(dim=-1, keepdim=True)
                if sink_key is None:
                    probabilities = exp / total.clamp_min(torch.finfo(torch.float32).tiny)
                    output = torch.einsum("tngk,tnkd->tngd", probabilities, selected.float()).flatten(1, 2)
                    output = F.pad(output, (0, q_rope.shape[-1]))
                else:
                    output = _parameter_sink(
                        query[start:end], q_rope[start:end], selected, scores, sink_key, sink_value,
                        start, end, x.shape[1], scale, dropout,
                    )
                outputs.append(output.to(x.dtype))
                maxima.append(maximum.detach().squeeze(-1))
                totals.append(total.detach().squeeze(-1))
    return (
        torch.cat(outputs).reshape(*x.shape[:3], -1),
        torch.cat(maxima).transpose(0, 1).contiguous(),
        torch.cat(totals).transpose(0, 1).contiguous(),
    )


def _parameter_sink(
    query: torch.Tensor, rope: torch.Tensor, selected: torch.Tensor, scores: torch.Tensor,
    sink_key: torch.Tensor, sink_value: torch.Tensor, start: int, end: int, seq_len: int,
    scale: float, dropout: float,
) -> torch.Tensor:
    """Combine sparse attention with learnable parameter-sink contributions."""
    batch_ids = torch.arange(start, end, device=query.device) // seq_len
    keys, values = sink_key[batch_ids].float(), sink_value[batch_ids].float()
    queries = torch.cat((query, rope), dim=-1).float()
    sink_scores = torch.einsum("thd,tshd->ths", queries, keys) * scale
    probabilities = torch.cat((scores.flatten(1, 2), sink_scores), dim=-1).softmax(-1)
    sparse_prob, sink_prob = probabilities.split((scores.shape[-1], sink_scores.shape[-1]), dim=-1)
    sparse_out = torch.einsum("tngk,tnkd->tngd", sparse_prob.reshape_as(scores),
                             selected.float()).flatten(1, 2)
    sparse_out = F.pad(sparse_out, (0, rope.shape[-1]))
    sink_out = torch.einsum("ths,tshd->thd", F.dropout(sink_prob, dropout), values)
    return sparse_out + sink_out


def _cann_absorbed(
    x: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
    topk_indices: torch.Tensor, scale: float, query_ends: torch.Tensor, key_ends: torch.Tensor,
    *, sink_key: torch.Tensor | None = None, sink_value: torch.Tensor | None = None, dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del sink_key, sink_value, dropout
    return fused_packed_attention(x, key, query_rope, key_rope, topk_indices, scale, query_ends, key_ends)


def _load_cann_absorbed() -> PackedDeepseekSparseAttentionFn:
    require_symbol(require_package("torch_npu"), "npu_sparse_flash_attention")
    require_symbol(require_package("cann_ops_transformer.ops"), "sparse_flash_attention_grad")
    return _cann_absorbed


def _load_omni_sink() -> PackedDeepseekSparseAttentionFn:
    """Load the optional Omni parameter-sink attention adapter."""
    require_package("omni_training_custom_ops")
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_fusion_attention")
    require_symbol(extension, "npu_fusion_attention_grad")
    require_symbol(torch.ops.custom, "npu_sparse_flash_attention_enhance")
    require_symbol(torch.ops.custom, "npu_sparse_flash_attention_grad_enhance")
    # Parameter-sink rescaling imports the optional Omni extension.
    from hyper_parallel.components.functional.dsa_sparse_attention_rescale import (  # pylint: disable=C0415
        dsa_sparse_attention_rescale,
    )

    def run(
        x: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
        topk_indices: torch.Tensor, scale: float, query_ends: torch.Tensor, key_ends: torch.Tensor,
        *, sink_key: torch.Tensor | None = None, sink_value: torch.Tensor | None = None, dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run parameter-sink sparse attention with the shared packed contract."""
        return dsa_sparse_attention_rescale(
            x, key, query_rope, key_rope, sink_key, sink_value, topk_indices,
            *x.shape[:3], scale, 1.0 - dropout, query_ends, key_ends,
        )

    return run


def build_sliding_window_indices(
        batch_size: int,
        sequence_length: int,
        window_size: int,
        device: torch.device,
        *,
        query_offset: int = 0,
        key_length: int | None = None,
) -> torch.Tensor:
    """Build fixed-width causal sliding-window indices."""
    if key_length is None:
        key_length = sequence_length
    width = min(key_length, window_size)
    query = torch.arange(
        query_offset,
        query_offset + sequence_length,
        device=device,
    ).unsqueeze(-1)
    offsets = torch.arange(width - 1, -1, -1, device=device)
    indices = query - offsets
    indices = indices.masked_fill(indices < 0, -1)
    return indices.unsqueeze(0).expand(batch_size, -1, -1)


def reference_sparse_attention(
        query: torch.Tensor,
        key_value: torch.Tensor,
        sparse_indices: torch.Tensor,
        sinks: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Compute sink attention with eager Torch operations on any device.

    Args:
        query: Main queries [batch, heads, local queries, head dimension].
        key_value: Shared K=V bank [batch, 1, keys, head dimension].
        sparse_indices: Visible bank IDs [batch, local queries, selections];
            negative entries are padding and repeated IDs count only once.
        sinks: Learnable scalar sink logits [heads], with zero-valued output.
        scale: Main attention logit scale.

    Returns:
        Attention values [batch, local queries, heads, head dimension].
        Softmax uses FP32; matrix multiplications retain the input precision.

    Note:
        This reference materializes a dense query/key mask and scores. It is
        intended for explicit unfused comparisons, not long-sequence efficiency.
    """
    batch_size, num_heads, sequence_length, _ = query.shape
    key_length = key_value.shape[2]
    valid = sparse_indices >= 0
    safe_indices = sparse_indices.clamp_min(0).long()
    allowed_counts = torch.zeros(
        batch_size,
        sequence_length,
        key_length,
        dtype=torch.int32,
        device=query.device,
    )
    allowed_counts.scatter_add_(2, safe_indices, valid.to(torch.int32))
    allowed = allowed_counts > 0
    logits = torch.matmul(query, key_value.transpose(-1, -2)) * scale
    logits = logits.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    sink_logits = sinks.view(1, num_heads, 1, 1).expand(batch_size, -1, sequence_length, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities[..., :-1].to(key_value.dtype), key_value).transpose(1, 2)


def native_deepseek_sparse_attention(
    x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
    topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
    layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
    main_key_offset: int = 0, global_key_length: int | None = None,
) -> torch.Tensor:
    """Assemble the reference sliding-window and compressed key indices."""
    raw_indices = build_sliding_window_indices(
        raw_key.shape[0], x.shape[2], window, raw_key.device,
        query_offset=layout.query_offset, key_length=global_sequence_length,
    )
    starts = layout.segment_starts
    if starts is None and layout.segments is not None:
        starts = torch.zeros(x.shape[2], dtype=torch.long, device=x.device)
        for (begin, end, *_), (_, origin) in zip(layout.segments, layout.raw_document_starts()):
            starts[begin:end] = origin
        starts = starts.unsqueeze(0)
    if starts is not None:
        raw_indices = raw_indices.masked_fill(raw_indices < starts.unsqueeze(-1), -1)
    raw_indices = torch.where(raw_indices >= 0, raw_indices - raw_key_offset, -1)
    if compressed_key is None:
        return reference_sparse_attention(x, raw_key, raw_indices, sinks, scale)
    indices = topk_indices
    if global_key_length is not None:
        indices = indices.masked_fill(indices >= global_key_length, -1)
    indices = torch.where(indices >= 0, indices.long() - main_key_offset + raw_key.shape[2], -1)
    return reference_sparse_attention(
        x, torch.cat((raw_key, compressed_key.unsqueeze(1)), dim=2),
        torch.cat((raw_indices, indices), dim=-1), sinks, scale,
    )


def attention_native_only(
    x: torch.Tensor, layout: DsaQueryLayout, window: int, topk_indices: torch.Tensor | None,
    global_sequence_length: int, global_key_length: int | None,
) -> str | None:
    """Explain incompatible layouts before attempting to load Sparse Flash MLA."""
    ratio = layout.compress_ratio
    topk = 0 if topk_indices is None else topk_indices.shape[-1]
    expected = global_sequence_length // max(1, ratio) if global_key_length is None else global_key_length
    if (layout.segments is None or (topk_indices is not None and topk != min(512, expected))
            or not supports_fused_attention_inputs(x, ratio, window, topk)):
        return "Sparse Flash MLA requires supported BF16 NPU inputs, document segments and ordinary TopK"
    return None


def _cann_attention(
    x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
    topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
    layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
    main_key_offset: int = 0, global_key_length: int | None = None,
) -> torch.Tensor:
    """Run compressed-bank sparse MLA through the CANN adapter."""
    del global_sequence_length, global_key_length
    output = sparse_flash_mla(
        x, raw_key, compressed_key, topk_indices, sinks, scale, layout.compress_ratio, window,
        layout.query_offset, layout.segments, raw_key_offset, main_key_offset, layout.raw_document_starts(),
    )
    if compressed_key is not None and all(end == start for _, _, start, end, _ in layout.segments):
        # Keep the shared bank's collective backward active on empty-prefix ranks.
        output = output + compressed_key[:, :0].sum().to(output.dtype)
    return output


class SparsePrefillFn(Protocol):
    """Forward-only sparse attention with optional sink and valid Top-K lengths."""

    def __call__(self, query: torch.Tensor, key_value: torch.Tensor, indices: torch.Tensor,
                 softmax_scale: float, **options: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return output, maximum logits and LSE without a backward graph."""


class SparseDecodeFn(Protocol):
    """Forward-only sparse attention over V4.1 FP8/FP4 paged KV caches."""

    def __call__(self, query: torch.Tensor, key_cache: torch.Tensor, indices: torch.Tensor,
                 scheduler_metadata: Any = None, **options: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,S,H,D] output and [B,H,S] LSE; caller owns scheduler state."""


DSA_SPARSE_PREFILL = OpSpec[SparsePrefillFn](config_key="deepseek_sparse_attention", variant="sparse_prefill")
DSA_SPARSE_DECODE = OpSpec[SparseDecodeFn](config_key="deepseek_sparse_attention", variant="sparse_decode")


def require_sparse_inference(*tensors: torch.Tensor | None) -> None:
    """Reject training before selecting a forward-only native extension."""
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise RuntimeError("sparse FlashMLA has no backward; use torch.no_grad() for inference")


def get_mla_metadata(*args: Any, **options: Any) -> Any:
    """Construct optional native scheduling metadata; callers retain its lifetime."""
    kernels = require_package("flash_mla")
    require_symbol(kernels, "get_mla_metadata")
    return kernels.get_mla_metadata(*args, **options)


def _decode_sparse_cache(cache):
    if cache.shape[-1] == 512:
        return cache
    raw = cache.view(torch.uint8)
    if raw.shape[-1] == 528:
        data = raw[..., :512].contiguous().view(torch.float8_e4m3fn).float()
        scale = torch.pow(2.0, raw[..., 512:].float() - 127)
        return (data * scale.repeat_interleave(32, -1)).to(torch.bfloat16)
    if raw.shape[-1] == 288:
        codes = torch.stack((raw[..., :256] & 15, raw[..., :256] >> 4), -1).flatten(-2)
        levels = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=cache.device)
        data = levels[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
        scale = raw[..., 256:].contiguous().view(torch.float8_e4m3fn).float()
        return (data * scale.repeat_interleave(16, -1)).to(torch.bfloat16)
    raise ValueError("V4.1 sparse KV cache rows require 528 FP8 bytes or 288 packed FP4 bytes")


def _sparse_statistics(query, selected, valid, scale, sink):
    logits = torch.einsum("thgd,thkd->thgk", query.float(), selected.float()) * scale
    logits = logits.masked_fill(~valid.unsqueeze(-2), -torch.inf)
    maximum = logits.amax(-1)
    lse = torch.logsumexp(logits, -1)
    denominator = lse if sink is None else torch.logaddexp(lse, sink.reshape(1, *lse.shape[1:]))
    denominator = torch.where(denominator == -torch.inf, torch.inf, denominator)
    probabilities = torch.exp(logits - denominator.unsqueeze(-1))
    output = torch.einsum("thgk,thkd->thgd", probabilities, selected.float())
    return output, maximum, torch.where(lse == -torch.inf, torch.inf, lse)


@torch.no_grad()
def _native_sparse_prefill(query, key_value, indices, softmax_scale, *, value_head_dim=512,
                           attention_sink=None, topk_length=None):
    if value_head_dim != 512:
        raise ValueError("sparse FlashMLA requires value_head_dim=512")
    keys = _decode_sparse_cache(key_value)
    heads = keys.shape[1]
    positions = indices.long()
    valid = (positions >= 0) & (positions < keys.shape[0])
    if topk_length is not None:
        valid &= torch.arange(indices.shape[-1], device=query.device) < topk_length[:, None, None]
    selected = keys[positions.clamp(0, keys.shape[0] - 1), torch.arange(heads, device=query.device)[None, :, None]]
    grouped = query.reshape(query.shape[0], heads, query.shape[1] // heads, query.shape[-1])
    output, maximum, lse = _sparse_statistics(grouped, selected, valid, softmax_scale, attention_sink)
    return output.flatten(1, 2).to(query.dtype), maximum.flatten(1, 2), lse.flatten(1, 2)


@torch.no_grad()
def _native_sparse_decode(query, key_cache, indices, scheduler_metadata=None, *, value_head_dim=512,
                          softmax_scale=None, attention_sink=None, topk_length=None,
                          extra_key_cache=None, extra_indices=None, extra_topk_length=None,
                          enable_batch_invariant=False):
    del scheduler_metadata, enable_batch_invariant
    if value_head_dim != 512:
        raise ValueError("sparse FlashMLA requires value_head_dim=512")
    batch, sequence, heads, width = query.shape

    def gather(cache: Any, ids: Any, lengths: Any) -> Any:
        """Gather valid cache slots while preserving masked positions."""
        keys = _decode_sparse_cache(cache).reshape(-1, width)
        selected = keys.index_select(0, ids.clamp_min(0).long().flatten()).reshape(batch * sequence, 1, -1, width)
        valid = ids != -1
        if lengths is not None:
            valid &= torch.arange(ids.shape[-1], device=query.device) < lengths[:, None, None]
        return selected, valid.reshape(batch * sequence, 1, -1)

    selected, valid = gather(key_cache, indices, topk_length)
    if extra_key_cache is not None:
        extra, extra_valid = gather(extra_key_cache, extra_indices, extra_topk_length)
        selected, valid = torch.cat((selected, extra), -2), torch.cat((valid, extra_valid), -1)
    output, _, lse = _sparse_statistics(query.reshape(batch * sequence, 1, heads, width), selected, valid,
                                       width**-0.5 if softmax_scale is None else softmax_scale, attention_sink)
    return output.reshape(query.shape).to(query.dtype), lse.reshape(batch, sequence, heads).transpose(1, 2)


def _load_flash_prefill() -> SparsePrefillFn:
    kernels = require_package("flash_mla")
    require_symbol(kernels, "flash_mla_sparse_fwd")

    def run(query: Any, key_value: Any, indices: Any, softmax_scale: Any, *, value_head_dim: Any=512,
            attention_sink: Any=None, topk_length: Any=None) -> Any:
        """Apply the operator contract using the selected implementation."""
        return kernels.flash_mla_sparse_fwd(query, key_value, indices, softmax_scale,
                                            d_v=value_head_dim, attn_sink=attention_sink, topk_length=topk_length)

    return run


def _load_flash_decode() -> SparseDecodeFn:
    kernels = require_package("flash_mla")
    require_symbol(kernels, "flash_mla_with_kvcache")
    require_symbol(kernels, "get_mla_metadata")

    def run(query: Any, key_cache: Any, indices: Any, scheduler_metadata: Any=None, *, value_head_dim: Any=512,
            softmax_scale: Any=None, attention_sink: Any=None, topk_length: Any=None, extra_key_cache: Any=None,
            extra_indices: Any=None, extra_topk_length: Any=None, enable_batch_invariant: Any=False) -> Any:
        """Apply the operator contract using the selected implementation."""
        if scheduler_metadata is None or isinstance(scheduler_metadata, dict):
            state = {} if scheduler_metadata is None else scheduler_metadata
            if "native" not in state:
                metadata = kernels.get_mla_metadata()
                state["native"] = metadata[0] if isinstance(metadata, tuple) else metadata
            scheduler_metadata = state["native"]
        return kernels.flash_mla_with_kvcache(
            query, key_cache, None, None, value_head_dim, scheduler_metadata, softmax_scale=softmax_scale,
            indices=indices, attn_sink=attention_sink, topk_length=topk_length, extra_k_cache=extra_key_cache,
            extra_indices_in_kvcache=extra_indices, extra_topk_length=extra_topk_length,
            enable_batch_invariant=enable_batch_invariant,
        )

    return run


def register_deepseek_sparse_attention(registry: OpRegistry) -> None:
    """Register the compressed-banks attention contract and its implementations."""
    registry.register_family("deepseek_sparse_attention", specs=(DSA_COMPRESSED_BANKS, DSA_ABSORBED, DSA_ABSORBED_SINK,
                                                                         DSA_SPARSE_PREFILL, DSA_SPARSE_DECODE))
    registry.register_impl(
        DSA_COMPRESSED_BANKS, name="native", devices=("cpu", "cuda", "npu"),
        load=lambda: native_deepseek_sparse_attention,
    )
    registry.register_impl(
        DSA_COMPRESSED_BANKS, name="cann", devices=("npu",),
        load=partial(load_cann, _cann_attention, "sparse_flash_mla", "sparse_flash_mla_grad",
                     "sparse_flash_mla_metadata"),
    )
    for spec in (DSA_ABSORBED, DSA_ABSORBED_SINK):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_absorbed)
    registry.register_impl(DSA_ABSORBED, name="cann", devices=("npu",), load=_load_cann_absorbed)
    registry.register_impl(DSA_ABSORBED_SINK, name="omni", devices=("npu",), load=_load_omni_sink)
    for spec, native, loader in ((DSA_SPARSE_PREFILL, _native_sparse_prefill, _load_flash_prefill),
                                 (DSA_SPARSE_DECODE, _native_sparse_decode, _load_flash_decode)):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"), load=lambda fn=native: fn)
        registry.register_impl(spec, name="flash_mla", devices=("npu",), load=loader)
