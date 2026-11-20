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
"""Expert-order permutation and weighted restoration with a shared slot layout."""

from __future__ import annotations

from typing import Protocol

import torch

from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol

PermutedTokens = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class MoePermuteFn(Protocol):
    """Group token rows by expert and carry the original routed slot IDs."""

    def __call__(self, x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
        """Return expert-major rows, slot order, and the position of each slot."""


class MoeRestoreFn(Protocol):
    """Gather expert outputs in token-major order and sum weighted routes."""

    def __call__(self, x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Cast weights to the expert output dtype before weighting and restoring tokens."""


MOE_TOKEN_REORDER = OpSpec[MoePermuteFn](config_key="moe_token_reorder")
MOE_TOKEN_RESTORE = OpSpec[MoeRestoreFn](config_key="moe_token_reorder", variant="weighted_restore")


def _native_permute(x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
    order = indices.flatten().argsort(stable=True)
    positions = torch.empty_like(order)
    positions.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return x.index_select(0, order.div(indices.shape[1], rounding_mode="floor")), order, positions


def _native_restore(x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    order = torch.empty_like(slot_positions)
    order.scatter_(0, slot_positions, torch.arange(order.numel(), device=order.device))
    token_ids = order.div(weights.shape[1], rounding_mode="floor")
    ordered_weights = weights.to(x.dtype).flatten().index_select(0, order)
    return x.new_zeros((weights.shape[0], x.shape[-1])).index_add_(
        0, token_ids, x * ordered_weights.unsqueeze(-1),
    )


def _load_npu() -> MoePermuteFn:
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_moe_token_permute")
    require_symbol(extension, "npu_moe_token_unpermute")

    def permute(x: torch.Tensor, indices: torch.Tensor) -> PermutedTokens:
        states, positions = extension.npu_moe_token_permute(x, indices.to(torch.int32))
        positions = positions.to(torch.int64)
        order = torch.empty_like(positions)
        order.scatter_(0, positions, torch.arange(positions.numel(), device=positions.device))
        return states, order, positions

    return permute


def _load_npu_restore() -> MoeRestoreFn:
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_moe_token_permute")
    require_symbol(extension, "npu_moe_token_unpermute")

    def restore(x: torch.Tensor, slot_positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Round like the reference before presenting the routing dtype to the kernel.
        probabilities = weights.to(x.dtype).to(weights.dtype)
        return extension.npu_moe_token_unpermute(x, slot_positions.to(torch.int32), probabilities).to(x.dtype)

    return restore


def register_moe_token_reorder(registry: OpRegistry) -> None:
    """Register Torch and Ascend operators for the same routed-slot layout."""
    registry.register_family("moe_token_reorder", specs=(MOE_TOKEN_REORDER, MOE_TOKEN_RESTORE))
    registry.register_impl(
        MOE_TOKEN_REORDER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_permute,
    )
    registry.register_impl(
        MOE_TOKEN_RESTORE, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_restore,
    )
    registry.register_impl(MOE_TOKEN_REORDER, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(MOE_TOKEN_RESTORE, name="torch_npu", devices=("npu",), load=_load_npu_restore)
