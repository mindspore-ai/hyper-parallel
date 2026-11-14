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
"""Explicit block-history operations for attention residual connections."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import


class BlockHistoryBoundary(torch.autograd.Function):
    """Group later history gradients at block boundaries without copying storage."""

    @staticmethod
    def forward(ctx: Any, *blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Expose history aliases through one boundary; no tensors are saved.

        Args:
            ctx: Autograd context retaining the per-invocation state.
        """
        ctx.set_materialize_grads(False)
        return tuple(block.view_as(block) for block in blocks)

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        """Return each block's accumulated gradient to the preceding boundary.

        Args:
            ctx: Autograd context retaining the per-invocation state.
        """
        del ctx
        return gradients


def materialize_block_history(blocks: tuple[torch.Tensor, ...], prefix: torch.Tensor) -> torch.Tensor:
    """Stack one consumer's immutable history without caching an autograd graph.

    Args:
        blocks: Completed flattened block tensors, in architectural order.
        prefix: Current residual stream, used only for an empty history's geometry.

    Returns:
        History [tokens, blocks, hidden] for this particular consumer.
    """
    if blocks:
        return torch.stack(blocks, dim=1)
    return prefix.new_empty(prefix.numel() // prefix.shape[-1], 0, prefix.shape[-1])


def append_history_block(blocks: tuple[torch.Tensor, ...], block: torch.Tensor,
                         dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Append a block and preserve the boundary's backward accumulation group.

    Args:
        blocks: Previous immutable flattened history tensors.
        block: New completed block with shape [tokens, hidden].
        dtype: Promoted history dtype selected by the caller.

    Returns:
        Explicit Tensor aliases visible to checkpoint and swap input traversal.
    """
    promoted = tuple(value.to(dtype) for value in (*blocks, block))
    return BlockHistoryBoundary.apply(*promoted)
