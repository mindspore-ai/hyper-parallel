# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Portable dense SDPA with explicit padding, causal alignment and sliding-window visibility."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Expand BS padding or BQK visibility without changing a four-dimensional mask."""
    mask = attention_mask
    if mask is not None and mask.ndim == 2:
        mask = mask.to(torch.bool)[:, None, None, :]
    elif mask is not None and mask.ndim == 3:
        mask = mask[:, None, :, :]
    return mask


def _causal_mask(query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor | None,
                 sliding_window: int | None) -> torch.Tensor:
    """Combine explicit visibility with a right-aligned causal window."""
    query_positions = torch.arange(query.shape[-2], device=query.device)
    query_positions = query_positions + (key.shape[-2] - query.shape[-2])
    key_positions = torch.arange(key.shape[-2], device=key.device)
    visible = key_positions[None, :] <= query_positions[:, None]
    if sliding_window is not None:
        visible &= key_positions[None, :] > query_positions[:, None] - sliding_window
    if mask is None:
        mask = visible
    elif mask.dtype == torch.bool:
        mask = mask & visible
    else:
        mask = mask.masked_fill(~visible, float("-inf"))
    return mask


def _eager_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     mask: torch.Tensor | None, scaling: float | None, dropout: float) -> torch.Tensor:
    """Evaluate dense scores, including zero output/VJP for empty visible rows."""
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key[:, :, None].expand(-1, -1, groups, -1, -1).reshape(key.shape[0], -1, key.shape[-2], key.shape[-1])
        value = (
            value[:, :, None]
            .expand(-1, -1, groups, -1, -1)
            .reshape(value.shape[0], -1, value.shape[-2], value.shape[-1])
        )
    weights = torch.matmul(query, key.transpose(2, 3)) * (
        scaling if scaling is not None else query.shape[-1] ** -0.5
    )
    if mask is not None:
        weights = weights.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else weights + mask
    # Softmax over an empty visible row is undefined; mask before and after
    # it so both the output and the VJP are zero, as in SDPA.
    valid_rows = ~torch.isneginf(weights).all(dim=-1, keepdim=True)
    weights = weights.masked_fill(~valid_rows, 0.0)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = weights.masked_fill(~valid_rows, 0.0)
    weights = F.dropout(weights, p=dropout, training=dropout > 0)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


def torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    is_causal: bool = True,
    scaling: float | None = None,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    implementation: str = "sdpa",
) -> torch.Tensor:
    """Compute dense attention with right-aligned causal visibility.

    Args:
        query: BNSD query heads.
        key: BNSD keys; KV heads must divide query heads.
        value: BNSD values with the same KV heads and sequence length as key.
        attention_mask: BS key padding, or broadcastable BQK/B1QK visibility/additive mask.
            Boolean True means visible; two-dimensional masks are interpreted as padding.
        is_causal: Restrict queries to their right-aligned key prefix.
        scaling: Optional score scale; defaults to the inverse square root of query width.
        dropout: Probability applied to attention weights; callers pass zero in evaluation.
        sliding_window: Optional positive causal window length.
        implementation: Eager reference math or PyTorch sdpa.

    Returns:
        Attention output in BSND. Fully masked rows have zero output and gradients.
    """
    if sliding_window is not None and (
        isinstance(sliding_window, bool) or not isinstance(sliding_window, int) or sliding_window < 1
    ):
        raise ValueError("sliding_window must be a positive integer or None")
    mask = _padding_mask(attention_mask)
    implicit_causal = (is_causal and mask is None and sliding_window is None
                       and query.shape[-2] == key.shape[-2] and implementation == "sdpa")
    if is_causal and not implicit_causal:
        mask = _causal_mask(query, key, mask, sliding_window)
    if implementation == "eager":
        return _eager_attention(query, key, value, mask, scaling, dropout)
    if implementation != "sdpa":
        raise ValueError("Portable attention implements only eager or sdpa")
    output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling,
        is_causal=implicit_causal, enable_gqa=query.shape[1] != key.shape[1]
    )
    return output.transpose(1, 2)
