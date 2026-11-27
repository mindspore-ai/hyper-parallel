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
"""Route-weighted token unpermutation preserving caller-selected weight precision."""

from __future__ import annotations

from typing import Protocol

import torch

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec


class MoeUnpermuteFn(Protocol):
    """Restore token-major order and sum expert outputs weighted by route scores.

    For token t, sum x[slot_positions[t, k]] * weights[t, k] over its routes.
    The flattened slot map follows moe_token_permute across all backends.
    Preserve the supplied weight precision: native products and sums follow
    Torch type promotion, then cast to the expert-state dtype. States and route
    weights are differentiable; callers choose any weight rounding explicitly.
    """

    def __call__(self, x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Preserve route-weight precision and return restored tokens in the expert output dtype.

        Args:
            x: Expert-major outputs [tokens * top_k, hidden].
            slot_positions: Flattened token-major slot positions in x [tokens * top_k].
            weights: Route weights [tokens, top_k], retaining caller-selected precision.

        Returns:
            Restored tokens [tokens, hidden] in the x dtype.
        """


def moe_token_unpermute(
    x: torch.Tensor,
    slot_positions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Reduce expert outputs without rounding the supplied route weights.

    Args:
        x: Expert-major outputs [tokens * top_k, hidden].
        slot_positions: Flattened token-major slot positions in x [tokens * top_k].
        weights: Route weights [tokens, top_k], retaining caller-selected precision.

    Returns:
        Restored tokens [tokens, hidden] in the x dtype.

    Notes:
        Restore token-major order and sum expert outputs weighted by route scores.

        For token t, sum x[slot_positions[t, k]] * weights[t, k] over its routes.
        The flattened slot map follows moe_token_permute across all backends.
        Preserve the supplied weight precision: native products and sums follow
        Torch type promotion, then cast to the expert-state dtype. States and route
        weights are differentiable; callers choose any weight rounding explicitly.
    """
    return _select_for_call(MOE_TOKEN_UNPERMUTE, device=x.device)(x, slot_positions, weights)


MOE_TOKEN_UNPERMUTE = OpSpec[MoeUnpermuteFn](config_key='moe_token_unpermute')


def _native_unpermute(x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    order = torch.empty_like(slot_positions)
    order.scatter_(0, slot_positions, torch.arange(order.numel(), device=order.device))
    token_ids = order.div(weights.shape[1], rounding_mode="floor")
    ordered_weights = weights.flatten().index_select(0, order)
    weighted = x * ordered_weights.unsqueeze(-1)
    return weighted.new_zeros((weights.shape[0], x.shape[-1])).index_add_(
        0, token_ids, weighted,
    ).to(x.dtype)


def _load_npu() -> MoeUnpermuteFn:
    """Load NPU token restoration without changing the supplied route weights."""
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_moe_token_unpermute")

    def restore(x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Pass the original route weights to the kernel and retain the output dtype.

        Args:
            x: Expert-major outputs.
            slot_positions: Token-major slot positions in the expert-major outputs.
            weights: Caller-supplied route weights, passed to the kernel unchanged.

        Returns:
            Weighted tokens in their original order with the same dtype as ``x``.
        """
        return extension.npu_moe_token_unpermute(x, slot_positions.to(torch.int32), weights).to(x.dtype)

    return restore


def register_moe_token_unpermute(registry: OpRegistry) -> None:
    """Register weighted unpermutation independently of token permutation."""
    registry.register_family("moe_token_unpermute", specs=(MOE_TOKEN_UNPERMUTE,))
    registry.register_impl(
        MOE_TOKEN_UNPERMUTE, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_unpermute,
    )
    registry.register_impl(MOE_TOKEN_UNPERMUTE, name="torch_npu", devices=("npu",), load=_load_npu)
