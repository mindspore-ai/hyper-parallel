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

from collections.abc import Sequence
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
import torch_npu


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


def _length_list(
    value: torch.Tensor | Sequence[int],
    *,
    includes_zero: bool,
) -> list[int]:
    """Normalize one cumulative-length representation."""
    if isinstance(value, torch.Tensor):
        lengths = [int(item) for item in value.tolist()]
    else:
        lengths = [int(item) for item in value]
    if includes_zero:
        if not lengths or lengths[0] != 0:
            raise ValueError("cu_seq_lens must start with zero")
        lengths = lengths[1:]
    if not lengths or any(left >= right for left, right in zip(lengths, lengths[1:])):
        raise ValueError("packed cumulative sequence lengths must be strictly increasing")
    return lengths


def _coalesce_lengths(
    kwargs: dict[str, Any],
    aliases: tuple[tuple[str, bool], ...],
    *,
    name: str,
) -> list[int] | None:
    """Read equivalent length arguments and reject conflicting values."""
    candidates = [
        (alias, _length_list(kwargs[alias], includes_zero=includes_zero))
        for alias, includes_zero in aliases
        if kwargs.get(alias) is not None
    ]
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


def _packed_sequence_lengths(
    kwargs: dict[str, Any],
    query_tokens: int,
    key_tokens: int,
) -> tuple[list[int] | None, list[int] | None]:
    """Resolve PR/VeOmni and Transformers packed-sequence argument names."""
    query_lengths = _coalesce_lengths(
        kwargs,
        (
            ("actual_seq_len", False),
            ("actual_q_len", False),
            ("actual_seq_qlen", False),
            ("cu_seq_lens_q", True),
        ),
        name="query sequence lengths",
    )
    key_lengths = _coalesce_lengths(
        kwargs,
        (
            ("actual_seq_len", False),
            ("actual_kv_len", False),
            ("actual_seq_kvlen", False),
            ("cu_seq_lens_k", True),
        ),
        name="key/value sequence lengths",
    )
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


def _attention_options(module: torch.nn.Module, kwargs: dict[str, Any]):
    """Resolve sparse-window and causal options from kwargs and the module."""
    pre_tokens = kwargs.get("pre_tokens", getattr(module, "pre_tockens", 1048576))
    next_tokens = kwargs.get("next_tokens", getattr(module, "next_tockens", 0))
    sparse_mode = kwargs.get("sparse_mode", getattr(module, "sparse_mode", 0))
    sliding_window = kwargs.get("sliding_window")
    is_causal = kwargs.get("is_causal", getattr(module, "is_causal", True))
    if sliding_window is not None:
        pre_tokens = sliding_window
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
            npu_mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device).triu(diagonal=1)
            sparse_mode = 3
        else:
            npu_mask = None if attention_mask is None else _npu_attention_mask(attention_mask)
        return query, key, value, "TND", npu_mask, sparse_mode
    if attention_mask is None and is_causal:
        return query, key, value, "BNSD", _causal_attention_mask(query, key, sliding_window), 0
    npu_mask = None if attention_mask is None else _npu_attention_mask(attention_mask)
    return query, key, value, "BNSD", npu_mask, sparse_mode


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
        to TND when cumulative sequence lengths are supplied through the PR/VeOmni
        ``actual_*`` names or Transformers ``cu_seq_lens_*`` names. Float16,
        bfloat16, and float32 inputs have been verified. A boolean mask uses
        ``True`` for positions that participate in attention; an additive mask
        uses zero for those positions.
    """
    if kwargs.get("indices") is not None:
        raise ValueError(
            "npu_fusion_attention_forward does not consume sparse attention indices; "
            "select a DSA sparse-attention implementation instead."
        )
    head_dim = query.shape[-1]
    batch_size = query.shape[0]
    query_length = query.shape[2]
    key_length = key.shape[2]
    query_lengths, key_lengths = _packed_sequence_lengths(
        kwargs,
        batch_size * query_length,
        key.shape[0] * key_length,
    )
    is_packed = query_lengths is not None
    pre_tokens, next_tokens, sparse_mode, sliding_window, is_causal = _attention_options(module, kwargs)
    query, key, value, input_layout, npu_attention_mask, sparse_mode = _prepare_attention_inputs(
        query,
        key,
        value,
        attention_mask,
        is_packed=is_packed,
        is_causal=is_causal,
        sliding_window=sliding_window,
        sparse_mode=sparse_mode,
    )
    output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query.shape[1],
        input_layout,
        pse=None,
        padding_mask=None,
        atten_mask=npu_attention_mask,
        scale=head_dim**-0.5 if scaling is None else scaling,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        keep_prob=1.0 - dropout,
        inner_precise=0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=query_lengths,
        actual_seq_kvlen=key_lengths,
    )[0]
    if is_packed:
        output = output.reshape(batch_size, query_length, output.shape[1], output.shape[2])
        return output, None
    return output.transpose(1, 2), None
