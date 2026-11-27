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
"""Expert-major token permutation with a backend-independent routed-slot layout."""

from __future__ import annotations

from typing import Protocol

import torch

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec

PermutedTokens = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class MoePermuteFn(Protocol):
    """Duplicate routed tokens and stably group their slots by expert ID.

    Flatten [tokens, top_k] routes in token-major order, then sort by expert.
    order maps expert-major rows to original slot IDs; slot_positions is its
    inverse. Both maps use int64 and are shared by native and fused backends,
    allowing moe_token_unpermute to use a different backend. Only states are differentiable.
    """

    def __call__(self, x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
        """Return expert-major rows, slot order, and the position of each slot.

        Args:
            x: Token states [tokens, hidden].
            indices: Selected expert IDs [tokens, top_k].

        Returns:
            A tuple of expert-major states [tokens * top_k, hidden], original slot IDs [tokens * top_k], and
            the expert-major position of each original slot [tokens * top_k].
        """


def moe_token_permute(
    x: torch.Tensor,
    indices: torch.Tensor,
) -> PermutedTokens:
    """Group token rows by expert while retaining a backend-independent inverse map.

    Args:
        x: Token states [tokens, hidden].
        indices: Selected expert IDs [tokens, top_k].

    Returns:
        A tuple of expert-major states [tokens * top_k, hidden], original slot IDs [tokens * top_k], and
        the expert-major position of each original slot [tokens * top_k].

    Notes:
        Duplicate routed tokens and stably group their slots by expert ID.

        Flatten [tokens, top_k] routes in token-major order, then sort by expert.
        order maps expert-major rows to original slot IDs; slot_positions is its
        inverse. Both maps use int64 and are shared by native and fused backends,
        allowing moe_token_unpermute to use a different backend. Only states are differentiable.
    """
    return _select_for_call(MOE_TOKEN_PERMUTE, device=x.device)(x, indices)


MOE_TOKEN_PERMUTE = OpSpec[MoePermuteFn](config_key='moe_token_permute')


def _native_permute(x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
    order = indices.flatten().argsort(stable=True)
    positions = torch.empty_like(order)
    positions.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return x.index_select(0, order.div(indices.shape[1], rounding_mode="floor")), order, positions


def _load_npu() -> MoePermuteFn:
    """Load the stable NPU token permutation adapter and inverse mapping."""
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_moe_token_permute")

    def permute(x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
        """Return routed rows and the shared int64 slot mapping."""
        states, positions = extension.npu_moe_token_permute(x, indices.to(torch.int32))
        positions = positions.to(torch.int64)
        order = torch.empty_like(positions)
        order.scatter_(0, positions, torch.arange(positions.numel(), device=positions.device))
        return states, order, positions

    return permute


def register_moe_token_permute(registry: OpRegistry) -> None:
    """Register token permutation independently of token unpermutation."""
    registry.register_family("moe_token_permute", specs=(MOE_TOKEN_PERMUTE,))
    registry.register_impl(
        MOE_TOKEN_PERMUTE, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_permute,
    )
    registry.register_impl(MOE_TOKEN_PERMUTE, name="torch_npu", devices=("npu",), load=_load_npu)
