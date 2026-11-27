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
from typing import Protocol

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention as fused_packed_attention
from hyper_parallel.components.functional.sparse_flash_mla import sparse_flash_mla, supports_fused_attention_inputs

from .._dependency import load_cann, require_package, require_symbol
from .._dsa_layout import DsaQueryLayout, packed_segments, selected_packed_keys, selected_packed_scores
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec


class DsaCompressedBanksFn(Protocol):
    """Attend a causal raw-token window and optional selected compressed keys.

    Raw and compressed keys share logical document coordinates supplied by
    layout and the bank offsets. The keys also supply attention values; sinks
    contribute to the softmax normalization. No attention crosses document
    boundaries. Return BSND output while retaining gradients through attention states.
    """

    def __call__(
        self, x: torch.Tensor, raw_key: torch.Tensor, compressed_key: torch.Tensor | None,
        topk_indices: torch.Tensor | None, sinks: torch.Tensor, *, scale: float, window: int,
        layout: DsaQueryLayout, global_sequence_length: int, raw_key_offset: int = 0,
        main_key_offset: int = 0, global_key_length: int | None = None,
    ) -> torch.Tensor:
        """Return attention output in BSND layout.

        Args:
            x: Attention queries [batch, heads, query_length, head_dim] in BNSD layout.
            raw_key: Raw-token key/value bank [batch, 1, key_length, head_dim].
            compressed_key: Optional compressed key/value bank [batch, 1, compressed_length, head_dim].
            topk_indices: Optional selected compressed-key IDs; -1 denotes an unused slot.
            sinks: Per-head attention sink logits.
            scale: Query-key score multiplier.
            window: Number of raw positions in the causal sliding window.
            layout: Logical query and document geometry.
            global_sequence_length: Global raw-token sequence length.
            raw_key_offset: Global raw-token offset of the supplied raw bank.
            main_key_offset: Global compressed-key offset of the supplied main bank.
            global_key_length: Optional full compressed-bank length for visibility checks.

        Returns:
            Attention states in BSND layout.
        """


def deepseek_sparse_attention_compressed_banks(
    x: torch.Tensor,
    raw_key: torch.Tensor,
    compressed_key: torch.Tensor | None,
    topk_indices: torch.Tensor | None,
    sinks: torch.Tensor,
    *,
    scale: float,
    window: int,
    layout: DsaQueryLayout,
    global_sequence_length: int,
    raw_key_offset: int = 0,
    main_key_offset: int = 0,
    global_key_length: int | None = None,
) -> torch.Tensor:
    """Attend a sliding window and optional compressed keys in logical document coordinates.

    Args:
        x: Attention queries [batch, heads, query_length, head_dim] in BNSD layout.
        raw_key: Raw-token key/value bank [batch, 1, key_length, head_dim].
        compressed_key: Optional compressed key/value bank [batch, 1, compressed_length, head_dim].
        topk_indices: Optional selected compressed-key IDs; -1 denotes an unused slot.
        sinks: Per-head attention sink logits.
        scale: Query-key score multiplier.
        window: Number of raw positions in the causal sliding window.
        layout: Logical query and document geometry.
        global_sequence_length: Global raw-token sequence length.
        raw_key_offset: Global raw-token offset of the supplied raw bank.
        main_key_offset: Global compressed-key offset of the supplied main bank.
        global_key_length: Optional full compressed-bank length for visibility checks.

    Returns:
        Attention states in BSND layout.

    Notes:
        Attend a causal raw-token window and optional selected compressed keys.

        Raw and compressed keys share logical document coordinates supplied by
        layout and the bank offsets. The keys also supply attention values; sinks
        contribute to the softmax normalization. No attention crosses document
        boundaries. Return BSND output while retaining gradients through attention states.
    """
    implementation = _select_for_call(DSA_COMPRESSED_BANKS, device=x.device)
    return implementation(
        x,
        raw_key,
        compressed_key,
        topk_indices,
        sinks,
        scale=scale,
        window=window,
        layout=layout,
        global_sequence_length=global_sequence_length,
        raw_key_offset=raw_key_offset,
        main_key_offset=main_key_offset,
        global_key_length=global_key_length,
    )


DSA_COMPRESSED_BANKS = OpSpec[DsaCompressedBanksFn](config_key='deepseek_sparse_attention', variant='compressed_banks')


class PackedDeepseekSparseAttentionFn(Protocol):
    """Compute sparse absorbed MLA inside packed, right-aligned causal documents.

    Gather document-local TopK keys and combine latent and rotary query-key
    scores before softmax. Without parameter sinks, values come from latent
    keys and the rotary output tail is zero-padded. Optional learned sink
    keys/values participate in attention. Returned sparse statistics are detached
    teacher values; their max/sum describe the selected non-sink scores.
    """

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
        topk_indices: torch.Tensor, scale: float, query_ends: torch.Tensor, key_ends: torch.Tensor,
        *, sink_key: torch.Tensor | None = None, sink_value: torch.Tensor | None = None, dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded BSND output and sparse max/sum in KV-head, token, group layout.

        Args:
            x: Absorbed query states in BSND layout.
            key: Latent key/value states in BSND layout.
            query_rope: Rotary query components in BSND layout.
            key_rope: Rotary key components in BSND layout.
            topk_indices: Document-local key IDs [total_queries, kv_heads, top_k], with -1 padding.
            scale: Multiplier applied to the combined latent and rotary scores.
            query_ends: Cumulative packed query document ends.
            key_ends: Cumulative packed key document ends.
            sink_key: Optional learned sink keys for the parameter-sink variant.
            sink_value: Optional learned sink values paired with sink_key.
            dropout: Attention probability dropout for the parameter-sink path.

        Returns:
            A tuple of padded BSND outputs and detached sparse softmax maxima/sums in [kv_heads,
            total_queries, groups] layout.
        """


DSA_ABSORBED = OpSpec[PackedDeepseekSparseAttentionFn](
    config_key='deepseek_sparse_attention', variant='absorbed_with_stats'
)
DSA_ABSORBED_SINK = OpSpec[PackedDeepseekSparseAttentionFn](
    config_key='deepseek_sparse_attention', variant='absorbed_with_parameter_sink'
)


def deepseek_sparse_attention_absorbed(
    x: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    query_ends: torch.Tensor,
    key_ends: torch.Tensor,
    *,
    sink_key: torch.Tensor | None = None,
    sink_value: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded BSND output and sparse max/sum in KV-head, token, group layout.

    Args:
        x: Absorbed query states in BSND layout.
        key: Latent key/value states in BSND layout.
        query_rope: Rotary query components in BSND layout.
        key_rope: Rotary key components in BSND layout.
        topk_indices: Document-local key IDs [total_queries, kv_heads, top_k], with -1 padding.
        scale: Multiplier applied to the combined latent and rotary scores.
        query_ends: Cumulative packed query document ends.
        key_ends: Cumulative packed key document ends.
        sink_key: Optional learned sink keys for the parameter-sink variant.
        sink_value: Optional learned sink values paired with sink_key.
        dropout: Attention probability dropout for the parameter-sink path.

    Returns:
        A tuple of padded BSND outputs and detached sparse softmax maxima/sums in [kv_heads,
        total_queries, groups] layout.

    Notes:
        Compute sparse absorbed MLA inside packed, right-aligned causal documents.

        Gather document-local TopK keys and combine latent and rotary query-key
        scores before softmax. Without parameter sinks, values come from latent
        keys and the rotary output tail is zero-padded. Optional learned sink
        keys/values participate in attention. Returned sparse statistics are detached
        teacher values; their max/sum describe the selected non-sink scores.
    """
    operation = _select_for_call(DSA_ABSORBED, device=x.device)
    return operation(
        x,
        key,
        query_rope,
        key_rope,
        topk_indices,
        scale,
        query_ends,
        key_ends,
        sink_key=sink_key,
        sink_value=sink_value,
        dropout=dropout,
    )


def deepseek_sparse_attention_with_sink(
    x: torch.Tensor,
    key: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    query_ends: torch.Tensor,
    key_ends: torch.Tensor,
    *,
    sink_key: torch.Tensor | None = None,
    sink_value: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded BSND output and sparse max/sum in KV-head, token, group layout.

    Args:
        x: Absorbed query states in BSND layout.
        key: Latent key/value states in BSND layout.
        query_rope: Rotary query components in BSND layout.
        key_rope: Rotary key components in BSND layout.
        topk_indices: Document-local key IDs [total_queries, kv_heads, top_k], with -1 padding.
        scale: Multiplier applied to the combined latent and rotary scores.
        query_ends: Cumulative packed query document ends.
        key_ends: Cumulative packed key document ends.
        sink_key: Optional learned sink keys for the parameter-sink variant.
        sink_value: Optional learned sink values paired with sink_key.
        dropout: Attention probability dropout for the parameter-sink path.

    Returns:
        A tuple of padded BSND outputs and detached sparse softmax maxima/sums in [kv_heads,
        total_queries, groups] layout.

    Notes:
        Compute sparse absorbed MLA inside packed, right-aligned causal documents.

        Gather document-local TopK keys and combine latent and rotary query-key
        scores before softmax. Without parameter sinks, values come from latent
        keys and the rotary output tail is zero-padded. Optional learned sink
        keys/values participate in attention. Returned sparse statistics are detached
        teacher values; their max/sum describe the selected non-sink scores.
    """
    operation = _select_for_call(DSA_ABSORBED_SINK, device=x.device)
    return operation(
        x,
        key,
        query_rope,
        key_rope,
        topk_indices,
        scale,
        query_ends,
        key_ends,
        sink_key=sink_key,
        sink_value=sink_value,
        dropout=dropout,
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


def _attention_input_error(
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
    reason = _attention_input_error(x, layout, window, topk_indices, global_sequence_length, global_key_length)
    if reason is not None:
        raise ValueError(reason)
    del global_sequence_length, global_key_length
    output = sparse_flash_mla(
        x, raw_key, compressed_key, topk_indices, sinks, scale, layout.compress_ratio, window,
        layout.query_offset, layout.segments, raw_key_offset, main_key_offset, layout.raw_document_starts(),
    )
    if compressed_key is not None and all(end == start for _, _, start, end, _ in layout.segments):
        # Keep the shared bank's collective backward active on empty-prefix ranks.
        output = output + compressed_key[:, :0].sum().to(output.dtype)
    return output


def register_deepseek_sparse_attention(registry: OpRegistry) -> None:
    """Register the compressed-banks attention contract and its implementations."""
    registry.register_family("deepseek_sparse_attention", specs=(DSA_COMPRESSED_BANKS, DSA_ABSORBED, DSA_ABSORBED_SINK))
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
