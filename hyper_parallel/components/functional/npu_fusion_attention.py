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
"""NPU fusion attention function."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import


_QUERY_LENGTH_ALIASES = (
    ("actual_seq_len", False), ("actual_q_len", False), ("actual_seq_qlen", False),
    ("cu_seq_lens", True), ("cu_seq_lens_q", True), ("cu_seqlens_q", True),
)
_KEY_LENGTH_ALIASES = (
    ("actual_seq_len", False), ("actual_kv_len", False), ("actual_seq_kvlen", False),
    ("cu_seq_lens", True), ("cu_seq_lens_k", True), ("cu_seq_lens_kv", True),
    ("cu_seqlens_k", True), ("cu_seqlens_kv", True),
)
PACKED_SEQUENCE_ARGUMENTS = frozenset(
    name for name, _ in _QUERY_LENGTH_ALIASES + _KEY_LENGTH_ALIASES
) | {"packed_seq_params"}

_COMPRESSED_CAUSAL_MASK_SIZE = 2048


@dataclass
class _FusionAttentionContext:
    """Normalized arguments required by the NPU fusion-attention operator."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    input_layout: str
    attention_mask: Optional[torch.Tensor]
    sparse_mode: int
    query_lengths: Optional[Sequence[int]]
    key_lengths: Optional[Sequence[int]]
    batch_size: int
    query_length: int
    head_dim: int
    pre_tokens: int
    next_tokens: int
    is_packed: bool
    valid_rows: Optional[torch.Tensor]
    indices: Optional[torch.Tensor] = None


def _npu_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert the Transformers attention mask to the NPU mask convention."""
    if attention_mask.dtype == torch.bool:
        return torch.logical_not(attention_mask)
    return attention_mask != 0


def _causal_attention_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    sliding_window: Optional[int],
) -> torch.Tensor:
    """Build the right-aligned causal mask used by Transformers decoding."""
    query_length = query.shape[2]
    key_length = key.shape[2]
    query_positions = torch.arange(query_length, device=query.device)
    query_positions = query_positions + max(key_length - query_length, 0)
    key_positions = torch.arange(key_length, device=query.device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if sliding_window is not None:
        allowed &= key_positions.unsqueeze(0) > (
            query_positions.unsqueeze(1) - sliding_window
        )
    return torch.logical_not(allowed)


def _get_compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Return the mask required by Ascend compressed causal sparse modes."""
    return torch.ones(
        (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
        dtype=torch.bool,
        device=device,
    ).triu(diagonal=1)


def _length_list(
    value: torch.Tensor | Sequence[int],
    *,
    includes_zero: bool,
) -> list[int]:
    """Normalize one cumulative-length representation."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError("packed cumulative sequence lengths must be one-dimensional")
        value = value.tolist()
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError("packed cumulative sequence lengths must be integers")
    lengths = [int(item) for item in value]
    if includes_zero:
        if not lengths or lengths[0] != 0:
            raise ValueError("cu_seq_lens must start with zero")
        lengths = lengths[1:]
    if not lengths or any(left >= right for left, right in zip([0] + lengths[:-1], lengths)):
        raise ValueError("packed cumulative sequence lengths must be strictly increasing")
    return lengths


def _coalesce_lengths(
    kwargs: dict[str, Any],
    aliases: tuple[tuple[str, bool], ...],
    *,
    name: str,
) -> list[int] | None:
    """Read equivalent length arguments and reject conflicting values."""
    candidates = []
    sources = (("", kwargs), ("packed_seq_params.", kwargs.get("packed_seq_params")))
    for prefix, source in sources:
        if source is None:
            continue
        for alias, includes_zero in aliases:
            value = source.get(alias) if isinstance(source, Mapping) else getattr(source, alias, None)
            if value is not None:
                candidates.append((prefix + alias, _length_list(value, includes_zero=includes_zero)))
    if not candidates:
        return None
    first_alias, first = candidates[0]
    for alias, candidate in candidates[1:]:
        if candidate != first:
            raise ValueError(
                f"conflicting {name} values were provided through "
                f"{first_alias!r} and {alias!r}"
            )
    return first


def resolve_packed_sequence_lengths(
    kwargs: dict[str, Any],
    query_tokens: int,
    key_tokens: int,
) -> tuple[list[int] | None, list[int] | None]:
    """Resolve supported packed-sequence argument aliases.

    Args:
        kwargs: Attention keyword arguments or a ``packed_seq_params`` carrier.
        query_tokens: Number of flattened query tokens.
        key_tokens: Number of flattened key/value tokens.

    Returns:
        Cumulative query and key/value sequence ends without leading zeros.
    """
    query_lengths = _coalesce_lengths(
        kwargs, _QUERY_LENGTH_ALIASES, name="query sequence lengths",
    )
    key_lengths = _coalesce_lengths(
        kwargs, _KEY_LENGTH_ALIASES, name="key/value sequence lengths",
    )
    if kwargs.get("packed_seq_params") is not None and query_lengths is None and key_lengths is None:
        raise ValueError("packed_seq_params must provide cumulative query and key/value sequence lengths")
    if (query_lengths is None) != (key_lengths is None):
        raise ValueError("packed attention requires both query and key/value lengths")
    if query_lengths is not None:
        if query_lengths[-1] != query_tokens:
            raise ValueError("the final query sequence length must equal the query token count")
        if key_lengths[-1] != key_tokens:
            raise ValueError(
                "the final key/value sequence length must equal the key/value token count"
            )
    return query_lengths, key_lengths


def _attention_options(
    module: torch.nn.Module,
    kwargs: dict[str, Any],
    query_length: int,
    key_length: int,
    *,
    is_packed: bool,
):
    """Resolve sparse-window and causal options from kwargs and the module."""
    pre_tokens = kwargs.get("pre_tokens", getattr(module, "pre_tockens", 1048576))
    sparse_mode = kwargs.get("sparse_mode", getattr(module, "sparse_mode", 0))
    sliding_window = kwargs.get("sliding_window")
    is_causal = kwargs.get("is_causal", getattr(module, "is_causal", True))
    next_tokens = kwargs.get("next_tokens", getattr(module, "next_tockens", 0 if is_causal else 2147483647))
    if sliding_window is not None:
        pre_tokens = sliding_window
    elif not is_packed and is_causal and query_length == key_length and pre_tokens < query_length:
        sliding_window = pre_tokens + 1
    return pre_tokens, next_tokens, sparse_mode, sliding_window, is_causal


def _prepare_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    is_packed: bool,
    is_causal: bool,
    sliding_window: Optional[int],
    sparse_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, Optional[torch.Tensor], int]:
    """Prepare QKV layout, mask, and sparse mode for the NPU operator."""
    if is_packed:
        query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[-1])
        key = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[-1])
        value = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[-1])
        if attention_mask is None and is_causal:
            npu_mask = _get_compressed_causal_mask(query.device)
            sparse_mode = 3
        else:
            npu_mask = None if attention_mask is None else _npu_attention_mask(attention_mask)
        return query, key, value, "TND", npu_mask, sparse_mode
    if attention_mask is None and is_causal:
        if sparse_mode in (2, 4):
            return query, key, value, "BNSD", _get_compressed_causal_mask(query.device), sparse_mode
        return query, key, value, "BNSD", _causal_attention_mask(query, key, sliding_window), 0
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=query.device, dtype=torch.bool)[:, None, None, :]
        attention_mask = attention_mask.expand(-1, 1, query.shape[2], -1)
    npu_mask = None if attention_mask is None else _npu_attention_mask(attention_mask)
    if npu_mask is not None and is_causal and sparse_mode == 0:
        npu_mask = npu_mask | _causal_attention_mask(query, key, sliding_window)
    return query, key, value, "BNSD", npu_mask, sparse_mode


def _prepare_padding_attention(
    context: _FusionAttentionContext,
    sliding_window: Optional[int],
) -> bool:
    """Gather valid QKV tokens directly into TND, retaining output indices."""
    attention_mask = context.attention_mask
    if attention_mask.shape[-1] != context.query_length:
        raise ValueError("the padding mask length must equal the query/key sequence length")
    attention_mask = attention_mask.to(device=context.query.device, dtype=torch.bool).expand(context.batch_size, -1)
    context.attention_mask = attention_mask
    if bool(attention_mask.all()):
        context.attention_mask = None
        return False
    if sliding_window is not None:
        # Removing interior holes would change the original window distances.
        starts = attention_mask[:, 0].int() + (attention_mask[:, 1:] & ~attention_mask[:, :-1]).sum(dim=-1)
        if bool((starts > 1).any()):
            return False
    context.indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    # Empty samples have no selected tokens and need no segment in the NPU call.
    context.query_lengths = tuple(lengths[lengths > 0].cumsum(0).cpu().tolist())
    context.key_lengths = context.query_lengths
    context.query, context.key, context.value = (
        tensor.transpose(1, 2).flatten(0, 1).index_select(0, context.indices)
        for tensor in (context.query, context.key, context.value)
    )
    context.input_layout = "TND"
    context.is_packed = True
    context.attention_mask = _get_compressed_causal_mask(context.query.device)
    context.sparse_mode = 3 if sliding_window is None else 4
    if sliding_window is not None:
        context.pre_tokens = sliding_window - 1
    return True


def _prepare_attention_mask(
    context: _FusionAttentionContext,
    is_causal: bool,
    sliding_window: Optional[int],
) -> None:
    """Select padding, compressed causal/window, or explicit-mask preparation."""
    equal_lengths = context.query.shape[2] == context.key.shape[2]
    compressible = not context.is_packed and equal_lengths and is_causal and context.next_tokens == 0
    compressible = compressible and (sliding_window is None or sliding_window > 0)
    if compressible and context.attention_mask is not None and context.attention_mask.ndim == 2:
        if _prepare_padding_attention(context, sliding_window):
            return
    if not context.is_packed and context.attention_mask is None and is_causal:
        context.sparse_mode = 0
        if compressible:
            if sliding_window is not None:
                context.sparse_mode = 4
                context.pre_tokens = sliding_window - 1
            elif context.query_length > _COMPRESSED_CAUSAL_MASK_SIZE:
                context.sparse_mode = 2
    prepared = _prepare_attention_inputs(
        context.query, context.key, context.value, context.attention_mask,
        is_packed=context.is_packed,
        is_causal=is_causal,
        sliding_window=sliding_window,
        sparse_mode=context.sparse_mode,
    )
    (
        context.query, context.key, context.value,
        context.input_layout, context.attention_mask, context.sparse_mode,
    ) = prepared


def _prepare_fusion_attention_context(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    kwargs: dict[str, Any],
) -> _FusionAttentionContext:
    """Normalize model inputs into one NPU fusion-attention invocation."""
    query_lengths, key_lengths = resolve_packed_sequence_lengths(
        kwargs,
        query.shape[0] * query.shape[2],
        key.shape[0] * key.shape[2],
    )
    options = _attention_options(
        module, kwargs, query.shape[2], key.shape[2], is_packed=query_lengths is not None,
    )
    context = _FusionAttentionContext(
        query, key, value, "BNSD", attention_mask, options[2],
        query_lengths,
        key_lengths,
        query.shape[0],
        query.shape[2],
        query.shape[-1],
        options[0],
        options[1],
        query_lengths is not None,
        None,
    )
    _prepare_attention_mask(context, options[4], options[3])
    if not context.is_packed and context.attention_mask is not None and context.sparse_mode == 0:
        context.valid_rows = (~context.attention_mask).any(dim=-1)
        # Give fully masked rows one finite softmax entry, then suppress their
        # output and upstream gradient. Some NPU kernels otherwise return nonzero rows.
        context.attention_mask[..., 0] &= context.valid_rows
    return context


def npu_fusion_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Compute NPU fusion attention using the Transformers AttentionInterface contract.

    Args:
        module: Owning Transformers attention module.
        query: Query tensor in BNSD layout.
        key: Key tensor in BNSD layout.
        value: Value tensor in BNSD layout.
        attention_mask: Attention mask using the Transformers mask convention.
        dropout: Dropout probability.
        scaling: Attention score scaling factor.
        **kwargs: Additional attention arguments.

    Returns:
        Attention output in BSND layout and ``None`` for attention weights.

    Raises:
        ValueError: If sparse attention indices or inconsistent sequence lengths
            are supplied.

    Note:
        QKV inputs use four-dimensional BNSD layout. Packed inputs are flattened
        to TND when cumulative sequence lengths are supplied through the project
        ``actual_*`` aliases or Transformers ``cu_seq_lens_*`` names. Float16,
        bfloat16, and float32 inputs have been verified. A boolean mask uses
        ``True`` for positions that participate in attention; an additive mask
        uses zero for those positions. A two-dimensional padding mask uses
        one/True for valid tokens. Equal-length causal padding attention uses
        variable-length attention and restores padded queries as zeros. Windowed
        attention uses this path for contiguous padding; masks with interior
        holes retain their original positional window through an explicit mask.
    """
    # Length parsing is also used by CPU CP validation; load the optional kernel only here.
    import torch_npu  # pylint: disable=C0415

    if kwargs.get("indices") is not None:
        raise ValueError(
            "npu_fusion_attention_forward does not consume sparse attention indices; "
            "select a DSA sparse-attention implementation instead."
        )
    context = _prepare_fusion_attention_context(
        module,
        query,
        key,
        value,
        attention_mask,
        kwargs,
    )
    if context.indices is not None and context.indices.numel() == 0:
        output = query.new_zeros((query.shape[0], query.shape[2], query.shape[1], query.shape[3]))
        return output + context.query.sum() + context.key.sum() + context.value.sum(), None
    output = torch_npu.npu_fusion_attention(
        context.query.contiguous(),
        context.key.contiguous(),
        context.value.contiguous(),
        context.query.shape[1],
        context.input_layout,
        pse=None,
        padding_mask=None,
        atten_mask=context.attention_mask,
        scale=context.head_dim**-0.5 if scaling is None else scaling,
        pre_tockens=context.pre_tokens,
        next_tockens=context.next_tokens,
        keep_prob=1.0 - dropout,
        inner_precise=0,
        sparse_mode=context.sparse_mode,
        actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths,
    )[0]
    if context.valid_rows is not None:
        output = output.masked_fill(~context.valid_rows.unsqueeze(-1), 0.0)
    if context.indices is not None:
        restored = output.new_zeros((context.batch_size * context.query_length, *output.shape[1:]))
        output = restored.index_copy(0, context.indices, output)
    if context.is_packed:
        output = output.reshape(
            context.batch_size,
            context.query_length,
            output.shape[1],
            output.shape[2],
        )
        return output, None
    return output.transpose(1, 2), None
