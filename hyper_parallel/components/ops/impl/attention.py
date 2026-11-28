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
"""Dense, grouped-query and vision attention with canonical BNSD inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.npu_fusion_attention import (
    npu_fusion_attention_forward,
    resolve_packed_sequence_lengths,
)

from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol
from .._dsa_layout import packed_segments


class VisionAttentionFn(Protocol):
    """Compute softmax(Q @ K.T / sqrt(head_dim)) @ V over all supplied keys.

    Attention is bidirectional, with no dropout or mask. Callers must remove
    invalid keys before calling. Inputs use BNSD layout and outputs use BSND.
    The native reference accumulates FP16/BF16 inputs in FP32.
    """

    def __call__(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Return attention over all supplied valid keys.

        Args:
            x: Queries [batch, heads, query_length, head_dim].
            key: Keys [batch, heads, key_length, head_dim].
            value: Values [batch, heads, key_length, value_dim].

        Returns:
            Attention output [batch, query_length, heads, value_dim] in the query dtype.
        """


AttentionLengths = tuple[list[int], list[int]]


class DenseAttentionFn(Protocol):
    """Compute scaled dot-product attention with grouped key/value heads.

    Query heads may share key/value heads. Padding, pairwise masks, causal
    visibility and optional left windows determine valid keys. Packed documents
    are attended independently. Value dimensions may differ from query/key
    dimensions. Return BSND activations without materializing attention weights.
    """

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None,
        *, causal: bool, scale: float, dropout: float, window: int | None,
        lengths: AttentionLengths | None, query_offset: int | None,
    ) -> torch.Tensor:
        """Apply padding or pairwise masks, packed document ends and the causal window.

        Args:
            x: Queries in [batch, query_heads, query_length, head_dim] (BNSD) layout.
            key: Keys [batch, kv_heads, key_length, head_dim].
            value: Values [batch, kv_heads, key_length, value_dim].
            mask: Optional key-padding or pairwise attention mask; Boolean true denotes a visible key.
            causal: Whether to exclude keys to the right of each query.
            scale: Multiplier applied to query-key scores before softmax.
            dropout: Dropout probability applied to attention probabilities.
            window: Optional number of visible positions in the causal left window.
            lengths: Optional pair of cumulative query/key document ends for packed attention.
            query_offset: Optional global offset of the first query for causal alignment.

        Returns:
            Attention output [batch, query_length, query_heads, value_dim] in the query dtype.
        """


def _dense_mask(mask: torch.Tensor | None, key_length: int, packed: bool) -> torch.Tensor | None:
    """Normalize key padding and pairwise mask dimensions before dense attention."""
    if mask is not None and not packed:
        mask = mask[..., :key_length]
    if mask is not None and mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask


def _dense_window(
    module: torch.nn.Module, query_length: int, key_length: int, causal: bool, kwargs: dict[str, Any],
) -> int | None:
    """Resolve an explicit sliding window or the module's causal token limit."""
    window = kwargs.get("sliding_window")
    if window is None and causal and query_length == key_length:
        pre_tokens = kwargs.get("pre_tokens", getattr(module, "pre_tockens", None))
        if pre_tokens is not None and pre_tokens < query_length:
            window = pre_tokens + 1
    return window


def _dense_attention(
    spec: OpSpec[DenseAttentionFn],
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    """Normalize dense attention arguments and dispatch the selected contract."""
    if kwargs.get("indices") is not None:
        raise ValueError("Dense attention does not consume sparse attention indices")
    q_len, k_len = query.shape[2], key.shape[2]
    q_ends, k_ends = resolve_packed_sequence_lengths(kwargs, query.shape[0] * q_len, key.shape[0] * k_len)
    lengths = (q_ends, k_ends) if q_ends is not None else None
    if lengths is not None and len(q_ends) != len(k_ends):
        raise ValueError("Packed query and key lengths must contain the same number of segments")
    attention_mask = _dense_mask(attention_mask, k_len, lengths is not None)
    causal = kwargs.get("is_causal", getattr(module, "is_causal", True))
    if q_len != k_len and attention_mask is not None and attention_mask.ndim >= 3 and lengths is None:
        # CP masks already encode the query's global offset.
        causal = False
    window = _dense_window(module, q_len, k_len, causal, kwargs)
    query_offset = kwargs.get("query_offset")
    operation = _select_for_call(spec, device=query.device)
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    return (
        operation(
            query,
            key,
            value,
            attention_mask,
            causal=causal,
            scale=scale,
            dropout=dropout,
            window=window,
            lengths=lengths,
            query_offset=query_offset,
        ),
        None,
    )


def grouped_query_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run the dense grouped-query contract through a Transformers-compatible entry point.

    Args:
        module: Module providing is_causal and optional pre_tockens defaults.
        query: Queries [batch, query_heads, query_length, head_dim] in BNSD layout.
        key: Keys [batch, kv_heads, key_length, head_dim].
        value: Values [batch, kv_heads, key_length, value_dim].
        attention_mask: Optional key-padding or pairwise Boolean/additive mask; Boolean true is visible.
        dropout: Attention probability dropout.
        scaling: Score multiplier; None uses head_dim ** -0.5.
        **kwargs: Optional is_causal, sliding_window, pre_tokens, query_offset and packed document
            metadata. Sparse indices are rejected.

    Returns:
        A tuple of BSND attention states and None for unmaterialized attention weights.

    Notes:
        Compute scaled dot-product attention with grouped key/value heads.

        Query heads may share key/value heads. Padding, pairwise masks, causal
        visibility and optional left windows determine valid keys. Packed documents
        are attended independently. Value dimensions may differ from query/key
        dimensions. Return BSND activations without materializing attention weights.
    """
    return _dense_attention(
        GROUPED_QUERY_ATTENTION,
        module,
        query,
        key,
        value,
        attention_mask,
        dropout,
        scaling,
        kwargs,
    )


def multi_latent_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attend over projected MLA states, allowing the value dimension to differ from Q and K.

    Args:
        module: Module providing is_causal and optional pre_tockens defaults.
        query: Queries [batch, query_heads, query_length, head_dim] in BNSD layout.
        key: Keys [batch, kv_heads, key_length, head_dim].
        value: Values [batch, kv_heads, key_length, value_dim].
        attention_mask: Optional key-padding or pairwise Boolean/additive mask; Boolean true is visible.
        dropout: Attention probability dropout.
        scaling: Score multiplier; None uses head_dim ** -0.5.
        **kwargs: Optional is_causal, sliding_window, pre_tokens, query_offset and packed document
            metadata. Sparse indices are rejected.

    Returns:
        A tuple of BSND attention states and None for unmaterialized attention weights.

    Notes:
        Compute scaled dot-product attention with grouped key/value heads.

        Query heads may share key/value heads. Padding, pairwise masks, causal
        visibility and optional left windows determine valid keys. Packed documents
        are attended independently. Value dimensions may differ from query/key
        dimensions. Return BSND activations without materializing attention weights.
    """
    return _dense_attention(MULTI_LATENT_ATTENTION, module, query, key, value, attention_mask, dropout, scaling, kwargs)


VISION_ATTENTION = OpSpec[VisionAttentionFn](config_key='attention', variant='vision')
GROUPED_QUERY_ATTENTION = OpSpec[DenseAttentionFn](config_key='attention', variant='gqa')
MULTI_LATENT_ATTENTION = OpSpec[DenseAttentionFn](config_key='attention', variant='mla')


class MultiLatentAttentionStatsFn(Protocol):
    """Compute dropout-free, packed causal MLA and optional teacher statistics.

    Query and key document ends define independent, right-aligned causal
    segments. Softmax maxima and shifted exponential sums are detached from
    autograd, so teacher statistics do not introduce an additional gradient path.
    """

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        query_ends: torch.Tensor, key_ends: torch.Tensor, *, scale: float, need_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return BSND output and optional H,T,1 softmax maximum and exponential sum.

        Args:
            x: Queries [batch, sequence, heads, query_dim] (BSND layout).
            key: Keys [batch, sequence, heads, query_dim] before flattening packed tokens.
            value: Values in BSND layout; the last dimension may differ from query_dim.
            query_ends: Cumulative query document ends after flattening batch and sequence.
            key_ends: Cumulative key document ends for the same documents.
            scale: Query-key score multiplier.
            need_stats: Whether to return detached softmax maxima and exponential sums.

        Returns:
            A tuple of BSND output, maxima and exponential sums. Statistics have shape [heads,
            total_queries, 1], or are both None when need_stats is false.
        """


MULTI_LATENT_ATTENTION_WITH_STATS = OpSpec[MultiLatentAttentionStatsFn](
    config_key='attention', variant='mla_with_stats'
)


def vision_attention(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return attention over all supplied valid keys.

    Args:
        x: Queries [batch, heads, query_length, head_dim].
        key: Keys [batch, heads, key_length, head_dim].
        value: Values [batch, heads, key_length, value_dim].

    Returns:
        Attention output [batch, query_length, heads, value_dim] in the query dtype.

    Notes:
        Compute softmax(Q @ K.T / sqrt(head_dim)) @ V over all supplied keys.

        Attention is bidirectional, with no dropout or mask. Callers must remove
        invalid keys before calling. Inputs use BNSD layout and outputs use BSND.
        The native reference accumulates FP16/BF16 inputs in FP32.
    """
    operation = _select_for_call(VISION_ATTENTION, device=x.device)
    return operation(x, key, value)


def multi_latent_attention_with_stats(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_ends: torch.Tensor,
    key_ends: torch.Tensor,
    *,
    scale: float,
    need_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return BSND output and optional H,T,1 softmax maximum and exponential sum.

    Args:
        x: Queries [batch, sequence, heads, query_dim] (BSND layout).
        key: Keys [batch, sequence, heads, query_dim] before flattening packed tokens.
        value: Values in BSND layout; the last dimension may differ from query_dim.
        query_ends: Cumulative query document ends after flattening batch and sequence.
        key_ends: Cumulative key document ends for the same documents.
        scale: Query-key score multiplier.
        need_stats: Whether to return detached softmax maxima and exponential sums.

    Returns:
        A tuple of BSND output, maxima and exponential sums. Statistics have shape [heads,
        total_queries, 1], or are both None when need_stats is false.

    Notes:
        Compute dropout-free, packed causal MLA and optional teacher statistics.

        Query and key document ends define independent, right-aligned causal
        segments. Softmax maxima and shifted exponential sums are detached from
        autograd, so teacher statistics do not introduce an additional gradient path.
    """
    operation = _select_for_call(MULTI_LATENT_ATTENTION_WITH_STATS, device=x.device)
    return operation(x, key, value, query_ends, key_ends, scale=scale, need_stats=need_stats)


def _native(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    accumulator = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        scores = (x.to(accumulator) @ key.to(accumulator).transpose(-1, -2)) * (x.shape[-1] ** -0.5)
        return (scores.softmax(dim=-1) @ value.to(accumulator)).to(x.dtype).transpose(1, 2)


def _sdpa(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(x, key, value, is_causal=False).transpose(1, 2)  # pylint: disable=not-callable


def _npu(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output, _ = npu_fusion_attention_forward(
        None, x, key, value, attention_mask=None,
        dropout=0.0, scaling=x.shape[-1] ** -0.5, is_causal=False, sparse_mode=0,
        pre_tokens=2147483647, next_tokens=2147483647,
    )
    return output


def _load_npu() -> VisionAttentionFn:
    require_symbol(require_package("torch_npu"), "npu_fusion_attention")
    return _npu


def _causal_mask(
    query: torch.Tensor, key: torch.Tensor, window: int | None, query_offset: int | None,
) -> torch.Tensor:
    offset = max(0, key.shape[2] - query.shape[2]) if query_offset is None else query_offset
    q_pos = (torch.arange(query.shape[2], device=query.device) + offset).unsqueeze(1)
    k_pos = torch.arange(key.shape[2], device=query.device)
    allowed = k_pos <= q_pos
    return allowed if window is None else allowed & (k_pos > q_pos - window)


def _torch_dense(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None,
    scale: float, dropout: float,
) -> torch.Tensor:
    """Compute dense attention with FP32 accumulation and explicit masking."""
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key, value = (tensor.repeat_interleave(groups, dim=1) for tensor in (key, value))
    dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = (query.to(dtype) @ key.to(dtype).transpose(-1, -2)) * scale
        if mask is not None:
            valid = mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~mask & valid, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        if mask is not None:
            probabilities = probabilities.masked_fill(~mask, 0)
        return (F.dropout(probabilities, dropout) @ value.to(dtype)).to(query.dtype)


def _sdpa_dense(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None,
    scale: float, dropout: float,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, attn_mask=mask, dropout_p=dropout, scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def _make_dense_reference(kernel: Callable[..., torch.Tensor]) -> DenseAttentionFn:
    """Bind a dense attention kernel to the shared reference contract."""
    def dense(
        x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None,
        *, causal: bool, scale: float, dropout: float, window: int | None,
        lengths: AttentionLengths | None, query_offset: int | None,
    ) -> torch.Tensor:
        """Resolve packed boundaries and masks before running the dense kernel."""
        if lengths is not None:
            return _packed_reference(kernel, x, key, value, mask, lengths, causal, scale, dropout, window)
        padding = None
        if mask is not None:
            mask = mask.to(device=x.device)
            if mask.ndim == 2:
                padding = mask.to(torch.bool)
                mask = padding[:, None, None, :]
            elif mask.dtype != torch.bool:
                mask = mask == 0
        if causal:
            allowed = _causal_mask(x, key, window, query_offset)
            mask = allowed if mask is None else mask & allowed
        output = kernel(x, key, value, mask, scale, dropout).transpose(1, 2)
        if padding is not None and causal and x.shape[2] == key.shape[2]:
            output = output.masked_fill(~padding[:, :, None, None], 0)
        return output

    return dense


def _packed_reference(
    kernel: Callable[..., torch.Tensor], x: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    mask: torch.Tensor | None, lengths: AttentionLengths, causal: bool, scale: float,
    dropout: float, window: int | None,
) -> torch.Tensor:
    """Evaluate independent packed documents with their local masks."""
    q_ends, k_ends = lengths
    padding = None
    if mask is not None:
        mask = mask.to(device=x.device)
        if mask.ndim == 2:
            padding = mask.to(torch.bool).expand(key.shape[0], -1).flatten()
            if padding.numel() != key.shape[0] * key.shape[2]:
                raise ValueError("Packed attention padding must cover the physical key tokens")
            mask = None
        elif mask.dtype != torch.bool:
            mask = mask == 0
    q_flat, k_flat, v_flat = (tensor.transpose(1, 2).flatten(0, 1).transpose(0, 1)
                              for tensor in (x, key, value))
    segments = []
    for qs, qe, ks, ke in zip([0, *q_ends[:-1]], q_ends, [0, *k_ends[:-1]], k_ends):
        q_chunk, k_chunk, v_chunk = (q_flat[:, qs:qe], k_flat[:, ks:ke], v_flat[:, ks:ke])
        local_mask = None if not causal else _causal_mask(q_chunk[None], k_chunk[None], window, None)
        if padding is not None:
            valid_keys = padding[ks:ke][None, :]
            local_mask = valid_keys if local_mask is None else local_mask & valid_keys
        if mask is not None:
            packed_mask = _packed_pair_mask(mask, qs, qe, ks, ke, x, key)
            local_mask = packed_mask if local_mask is None else packed_mask & local_mask
        chunk = kernel(q_chunk[None], k_chunk[None], v_chunk[None], local_mask, scale, dropout)
        if padding is not None and causal and q_ends == k_ends:
            chunk = chunk.masked_fill(~padding[qs:qe][None, None, :, None], 0)
        segments.append(chunk.squeeze(0).transpose(0, 1))
    return torch.cat(segments).reshape(x.shape[0], x.shape[2], x.shape[1], value.shape[-1])


def _packed_pair_mask(
    mask: torch.Tensor, qs: int, qe: int, ks: int, ke: int, query: torch.Tensor, key: torch.Tensor,
) -> torch.Tensor:
    """Slice a pairwise mask using packed or batch-local token coordinates."""
    q_len, k_len = query.shape[2], key.shape[2]
    if mask.shape[-2:] == (query.shape[0] * q_len, key.shape[0] * k_len):
        return mask[..., qs:qe, ks:ke]
    q_batch, q_start = divmod(qs, q_len)
    k_batch, k_start = divmod(ks, k_len)
    q_stop, k_stop = q_start + qe - qs, k_start + ke - ks
    if q_batch != k_batch or q_stop > q_len or k_stop > k_len:
        raise ValueError("A packed pairwise mask must use flattened token indices when a segment spans batches")
    if mask.ndim == 4 and mask.shape[0] > 1:
        mask = mask[q_batch:q_batch + 1]
    return mask[..., q_start:q_stop, k_start:k_stop]


def _npu_dense(
    x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None,
    *, causal: bool, scale: float, dropout: float, window: int | None,
    lengths: AttentionLengths | None, query_offset: int | None,
) -> torch.Tensor:
    """Translate dense attention arguments for the NPU fusion kernel."""
    if lengths is not None and mask is not None:
        raise ValueError("packed NPU attention requires document ends without an external mask")
    options = {} if lengths is None else {"actual_q_len": lengths[0], "actual_kv_len": lengths[1]}
    if query_offset is not None and causal and lengths is None:
        if mask is not None and mask.ndim == 2:
            mask = mask.to(torch.bool)[:, None, None, :]
        if mask is not None and mask.dtype != torch.bool:
            mask = mask == 0
        causal_mask = _causal_mask(x, key, window, query_offset)
        mask = causal_mask if mask is None else causal_mask & mask
        causal = False
    output, _ = npu_fusion_attention_forward(
        None, x, key, value, mask, dropout=dropout, scaling=scale,
        is_causal=causal, sliding_window=window, **options,
    )
    return output


def _load_npu_dense() -> DenseAttentionFn:
    require_symbol(require_package("torch_npu"), "npu_fusion_attention")
    return _npu_dense


def _npu_mla_with_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_ends: torch.Tensor,
    key_ends: torch.Tensor,
    *,
    scale: float,
    need_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run one TND FA and return dense-indexer statistics."""
    query_ends, key_ends = query_ends.tolist(), key_ends.tolist()
    q_flat, k_flat, v_flat = (tensor.flatten(0, 1) for tensor in (query, key, value))
    num_heads = q_flat.shape[1]
    causal_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device).triu(diagonal=1)
    # The native training kernel is optional outside the Ascend backend.
    import torch_npu  # pylint: disable=import-outside-toplevel

    result = torch_npu.npu_fusion_attention(
        q_flat.contiguous(), k_flat.contiguous(), v_flat.contiguous(), num_heads, "TND",
        pse=None, padding_mask=None, atten_mask=causal_mask, scale=scale,
        pre_tockens=1048576, next_tockens=0, keep_prob=1.0, inner_precise=0,
        actual_seq_qlen=query_ends, actual_seq_kvlen=key_ends, sparse_mode=3,
    )
    output = result[0].reshape(*query.shape[:3], value.shape[-1])
    if not need_stats:
        return output, None, None

    # TND FA stores each packed segment as [head, token, 8], while exposing a
    # physical [token, head, 8] shape. Dense KL consumes [head, token, 1].
    statistics = []
    starts = [0, *query_ends[:-1]]
    for statistic in result[1:3]:
        rows = statistic.reshape(-1, statistic.shape[-1])
        segments = [
            rows[start * num_heads : end * num_heads].reshape(
                num_heads, end - start, statistic.shape[-1]
            )[..., :1]
            for start, end in zip(starts, query_ends)
        ]
        statistics.append(torch.cat(segments, dim=1).detach())
    return output, statistics[0], statistics[1]


def _make_mla_stats_reference(*, sdpa: bool = False) -> MultiLatentAttentionStatsFn:
    """Build packed MLA attention with optional detached softmax statistics."""
    def run(
        x: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        query_ends: torch.Tensor, key_ends: torch.Tensor, *, scale: float, need_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return packed MLA output and optional detached teacher statistics."""
        q_flat, k_flat, v_flat = (tensor.flatten(0, 1).transpose(0, 1) for tensor in (x, key, value))
        output, maxima, totals = [], [], []
        dtype = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            for qs, qe, ks, ke in packed_segments(query_ends, key_ends):
                q, k, v = q_flat[:, qs:qe], k_flat[:, ks:ke], v_flat[:, ks:ke]
                for start in range(0, qe - qs, 128):
                    end = min(start + 128, qe - qs)
                    local_q = q[:, start:end]
                    mask = _causal_mask(local_q[None], k[None], None, start + ke - ks - (qe - qs))
                    if sdpa:
                        output.append(_sdpa_dense(local_q[None], k[None], v[None], mask, scale, 0.0)[0])
                    if need_stats or not sdpa:
                        scores = (local_q.to(dtype) @ k.to(dtype).transpose(-1, -2)) * scale
                        scores = scores.masked_fill(~mask, -torch.inf)
                        maximum = scores.amax(dim=-1, keepdim=True)
                        maximum = torch.where(maximum.isfinite(), maximum, 0.0)
                        exp = (scores - maximum).exp()
                        total = exp.sum(dim=-1, keepdim=True)
                        if not sdpa:
                            probabilities = exp / total.clamp_min(torch.finfo(dtype).tiny)
                            output.append((probabilities @ v.to(dtype)).to(x.dtype))
                        if need_stats:
                            maxima.append(maximum.detach())
                            totals.append(total.detach())
        attention = torch.cat(output, dim=1).transpose(0, 1).reshape(*x.shape[:3], value.shape[-1])
        if not need_stats:
            return attention, None, None
        return attention, torch.cat(maxima, dim=1), torch.cat(totals, dim=1)

    return run


def _load_npu_mla_stats() -> MultiLatentAttentionStatsFn:
    require_symbol(require_package("torch_npu"), "npu_fusion_attention")
    return _npu_mla_with_stats


def register_attention(registry: OpRegistry) -> None:
    """Register reference, SDPA and Ascend vision and dense attention."""
    registry.register_family("attention", specs=(
        VISION_ATTENTION, GROUPED_QUERY_ATTENTION, MULTI_LATENT_ATTENTION, MULTI_LATENT_ATTENTION_WITH_STATS,
    ))
    registry.register_impl(VISION_ATTENTION, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(VISION_ATTENTION, name="sdpa", devices=("cpu", "cuda"), load=lambda: _sdpa)
    registry.register_impl(VISION_ATTENTION, name="torch_npu", devices=("npu",), load=_load_npu)
    for spec in (GROUPED_QUERY_ATTENTION, MULTI_LATENT_ATTENTION):
        registry.register_impl(spec, name="native", devices=("cpu", "cuda", "npu"),
                               load=lambda: _make_dense_reference(_torch_dense))
        registry.register_impl(
            spec, name="sdpa", devices=("cpu", "cuda"), load=lambda: _make_dense_reference(_sdpa_dense),
        )
        registry.register_impl(spec, name="torch_npu", devices=("npu",), load=_load_npu_dense)
    registry.register_impl(
        MULTI_LATENT_ATTENTION_WITH_STATS, name="native", devices=("cpu", "cuda", "npu"),
        load=_make_mla_stats_reference,
    )
    registry.register_impl(MULTI_LATENT_ATTENTION_WITH_STATS, name="sdpa", devices=("cpu", "cuda"),
                           load=lambda: _make_mla_stats_reference(sdpa=True))
    registry.register_impl(
        MULTI_LATENT_ATTENTION_WITH_STATS, name="torch_npu", devices=("npu",), load=_load_npu_mla_stats,
    )
