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

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch.autograd.function import once_differentiable

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


class _TileKernelsInterleavedRope(torch.autograd.Function):
    """Make the upstream in-place rotation out-of-place and differentiable."""

    @staticmethod
    def forward(ctx, x, cache, kernel):
        """Keep original operands for the transpose rotation and frequency gradients."""
        ctx.save_for_backward(x, cache)
        ctx.kernel = kernel
        result = x.clone()
        kernel(result, cache, interleaved=True)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Transpose the rotation; reduce frequency gradients across attention heads."""
        x, cache = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.contiguous().clone()
            ctx.kernel(grad_input, cache, interleaved=True, conjugate=True)
        grad_cache = None
        if ctx.needs_input_grad[1]:
            even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
            grad_even, grad_odd = grad_output.float()[..., 0::2], grad_output.float()[..., 1::2]
            grad_cos = (grad_even * even + grad_odd * odd).sum(dim=1)
            grad_sin = (grad_odd * even - grad_even * odd).sum(dim=1)
            grad_cache = torch.cat((grad_cos, grad_sin), dim=-1)
        return grad_input, grad_cache, None


def _load_tile_kernels_interleaved() -> InterleavedRopeFn:
    extension = require_package("tile_kernels.transform")
    require_symbol(extension, "apply_rotary")
    kernel = extension.apply_rotary

    def run(
        x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use shared per-token frequencies and restore Hyper's split-half output layout."""
        rotary_dim = cos.shape[-1]
        if not rotary_dim:
            return x, key
        if (unsqueeze_dim not in (1, 2) or x.ndim != 4 or key.ndim != 4
                or cos.ndim not in (2, 3) or cos.ndim != sin.ndim or rotary_dim != sin.shape[-1]):
            raise ValueError("RoPE adapter requires 4-D Q/K, matching-width 2-D/3-D tables and unsqueeze_dim 1 or 2")
        batch, sequence = x.transpose(unsqueeze_dim, 2).shape[:2]
        if key.transpose(unsqueeze_dim, 2).shape[:2] != (batch, sequence):
            raise ValueError("Q/K batch and sequence dimensions must match")
        if any(t.device != x.device for t in (key, cos, sin)):
            raise ValueError("Q/K and cos/sin must be on the same device")
        frequency_dtype = torch.promote_types(cos.dtype, sin.dtype)
        cache = torch.cat((cos[..., :rotary_dim // 2].float(), sin[..., :rotary_dim // 2].float()), dim=-1)
        cache = cache.expand(batch, sequence, rotary_dim).reshape(batch * sequence, rotary_dim).contiguous()

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            prefix = tensor.transpose(unsqueeze_dim, 2)[..., :rotary_dim]
            prefix = prefix.to(torch.promote_types(tensor.dtype, frequency_dtype))
            rotated = _TileKernelsInterleavedRope.apply(prefix.flatten(0, 1).contiguous(), cache, kernel)
            rotated = rotated.reshape(prefix.shape).transpose(unsqueeze_dim, 2)
            rotated = rotated.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
            return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)

        return rotate(x), rotate(key)

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
    registry.register_impl(
        ROPE_INTERLEAVED, name="tile_kernels", devices=("npu",), load=_load_tile_kernels_interleaved,
    )
