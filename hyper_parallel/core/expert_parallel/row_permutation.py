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
"""Bijective receive-row permutations without high-dimensional scatter gradients."""
from typing import Any

import torch


def invert_permutation(indices: torch.Tensor) -> torch.Tensor:
    """Invert a trusted bijection using one-dimensional integer metadata.

    Args:
        indices: Destination-to-source row indices, each source occurring once.
            This contract is provided by EP receive reordering, not top-k routing.

    Returns:
        Source-to-destination indices on the same device.
    """
    positions = torch.arange(indices.numel(), device=indices.device, dtype=indices.dtype)
    return torch.empty_like(indices).scatter_(0, indices, positions)


class _PermuteRows(torch.autograd.Function):
    """Use the inverse bijection for dgrad instead of generic index accumulation."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, indices: torch.Tensor,
                inverse: torch.Tensor) -> torch.Tensor:
        """Gather rows and retain only the inverse map for backward.

        Args:
            ctx: Autograd context retaining the inverse map.
            values: Tensor whose first dimension contains receive rows.
            indices: Destination-to-source row map.
            inverse: Source-to-destination row map.

        Returns:
            Values gathered in destination order."""
        ctx.save_for_backward(inverse)
        return values.index_select(0, indices)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Restore row gradients by gather, including empty and strided inputs.

        Args:
            ctx: Autograd context retaining the inverse map.
            gradient: Output gradient in destination order.

        Returns:
            Input gradient and no gradients for the integer maps."""
        (inverse,) = ctx.saved_tensors
        return gradient.index_select(0, inverse), None, None


def permute_rows(values: torch.Tensor, indices: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Apply a bijective row map with a gather in both forward and backward.

    Args:
        values: Tensor with the receive row dimension first.
        indices: Destination-to-source row map.
        inverse: Cached inverse of ``indices`` with the same row count.

    Returns:
        Values in destination order, preserving empty autograd edges.
    """
    return _PermuteRows.apply(values, indices, inverse)
