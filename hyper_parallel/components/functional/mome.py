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
"""Masked causal hidden-state mixing with a residual connection."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from hyper_parallel.components.ops import aggregate_hidden

__all__ = ["mome"]


def mome(
    x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a masked causal depthwise convolution to the original hidden states.

    Each hidden channel is filtered independently over the current and earlier
    tokens. Positions before the sequence start contribute zero. The last
    filter tap multiplies the current token. After optional bias addition, the
    mask gates the convolution update and the result is added to x. A false
    mask position retains x unchanged and may still contribute to later updates.
    Aggregation restores the input dtype before bias addition, so each operation
    retains its own rounding boundary for low-precision inputs.

    Args:
        x: Hidden states [batch, sequence, hidden].
        mask: Boolean output mask [batch, sequence].
        weight: Depthwise filter [hidden, 1, window].
        bias: Optional per-channel offset [hidden].

    Returns:
        Mixed hidden states with the input shape, dtype and device.

    Raises:
        ValueError: States, mask, filter or bias shapes violate the contract.
    """
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError("mome requires states [B, S, H] and a matching mask [B, S]")
    if weight.ndim != 3 or weight.shape[:2] != (x.shape[-1], 1) or weight.shape[-1] == 0:
        raise ValueError("mome requires a nonempty depthwise filter [H, 1, K]")
    if bias is not None and bias.shape != (x.shape[-1],):
        raise ValueError("mome bias must have shape [H]")
    padding = weight.shape[-1] - 1
    padded = F.pad(x, (0, 0, padding, 0))
    # Apply the output mask after bias addition; every real source remains visible.
    kernel_mask = F.pad(torch.ones_like(mask), (padding, 0), value=False)
    mixed = aggregate_hidden(
        padded.transpose(0, 1).contiguous(), weight.squeeze(1).transpose(0, 1), kernel_mask,
    )[padding:].transpose(0, 1).contiguous()
    if bias is not None:
        mixed = mixed + bias.to(mixed.dtype)
    return x + mixed * mask.unsqueeze(-1).to(mixed.dtype)
