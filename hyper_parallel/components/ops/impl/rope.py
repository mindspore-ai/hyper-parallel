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
"""Half and interleaved rotary position embedding contracts."""

from collections.abc import Callable
from typing import Protocol

import torch

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class RopeFn(Protocol):
    """Rotate a tensor with broadcast-ready cosine and sine tensors."""

    def __call__(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate the prefix of x covered by the frequencies and retain the tail."""


class InterleavedRopeFn(Protocol):
    """Rotate adjacent pairs using the Transformers interleaved RoPE layout."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        *, unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query and key with their rotated pairs laid out as two halves."""


ROPE = OpSpec[RopeFn](config_key="rope", variant="half")
ROPE_INTERLEAVED = OpSpec[InterleavedRopeFn](config_key="rope", variant="interleaved")


def _rotate_prefix(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    rotation: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Rotate the frequency-covered prefix and preserve the remaining head dimensions."""
    rotary_dim = cos.shape[-1]
    if rotary_dim != sin.shape[-1] or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("RoPE requires matching, even frequency dimensions within the input head dimension")
    if rotary_dim == x.shape[-1]:
        return rotation(x, cos, sin)
    prefix, tail = x.split((rotary_dim, x.shape[-1] - rotary_dim), dim=-1)
    return torch.cat((rotation(prefix, cos, sin), tail), dim=-1)


def _half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


def _native(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _rotate_prefix(x, cos, sin, _half) if cos.shape[-1] else x


def _rotate_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)


def _native_interleaved(
    x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent Q/K pairs and return the canonical split-half layout."""
    if unsqueeze_dim not in (1, 2):
        raise ValueError("Interleaved RoPE supports unsqueeze_dim 1 or 2")
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    rotary_dim = cos.shape[-1]
    if rotary_dim != sin.shape[-1] or rotary_dim % 2 or rotary_dim > min(x.shape[-1], key.shape[-1]):
        raise ValueError("RoPE requires matching, even frequency dimensions within the input head dimension")
    if not rotary_dim:
        return x, key
    cos = cos[..., : rotary_dim // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : rotary_dim // 2].unsqueeze(unsqueeze_dim)
    return tuple(
        torch.cat((_rotate_adjacent(tensor[..., :rotary_dim], cos, sin), tensor[..., rotary_dim:]), dim=-1)
        for tensor in (x, key)
    )


def _load_npu() -> RopeFn:
    """Load the NPU half-layout rotary embedding adapter."""
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_rotary_mul")

    def half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate a contiguous prefix with the NPU half-layout kernel."""
        return extension.npu_rotary_mul(x.contiguous(), cos, sin, rotary_mode="half")

    def run(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply the selected rotation while preserving the unrotated tail."""
        return _rotate_prefix(x, cos, sin, half) if cos.shape[-1] else x

    return run


def _load_npu_interleaved() -> InterleavedRopeFn:
    """Load the optional NPU adjacent-pair rotary embedding adapter."""
    require_symbol(require_package("torch_npu"), "npu_rotary_mul")
    # The existing pair-layout adapter imports the optional NPU extension.
    from hyper_parallel.components.functional.rotary_embedding import (  # pylint: disable=import-outside-toplevel
        apply_rotary_pos_emb_interleave,
    )

    def run(
        x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the selected rotation while preserving the unrotated tail."""
        if not cos.shape[-1]:
            return x, key
        return apply_rotary_pos_emb_interleave(x, key, cos, sin, unsqueeze_dim=unsqueeze_dim)

    return run


def register_rope(registry: OpRegistry) -> None:
    """Register half, interleaved and partial rotation on compatible devices."""
    registry.register_family("rope", specs=(ROPE, ROPE_INTERLEAVED))
    registry.register_impl(ROPE, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(
        ROPE_INTERLEAVED, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_interleaved,
    )
    registry.register_impl(ROPE, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(ROPE_INTERLEAVED, name="torch_npu", devices=("npu",), load=_load_npu_interleaved)
