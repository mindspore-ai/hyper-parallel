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
    """Consume BNSD operands and return BSND, including right-aligned unequal query/KV lengths."""
    if sliding_window is not None and (
        isinstance(sliding_window, bool) or not isinstance(sliding_window, int) or sliding_window < 1
    ):
        raise ValueError("sliding_window must be a positive integer or None")
    mask = attention_mask
    if mask is not None and mask.ndim == 2:
        mask = mask.to(torch.bool)[:, None, None, :]
    elif mask is not None and mask.ndim == 3:
        mask = mask[:, None, :, :]
    if is_causal:
        query_positions = torch.arange(query.shape[-2], device=query.device)
        query_positions = query_positions + max(key.shape[-2] - query.shape[-2], 0)
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
    if implementation == "eager":
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
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=dropout, training=dropout > 0)
        return torch.matmul(weights, value).transpose(1, 2).contiguous()
    if implementation != "sdpa":
        raise ValueError("Portable attention implements only eager or sdpa")
    output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling, enable_gqa=query.shape[1] != key.shape[1]
    )
    return output.transpose(1, 2)
