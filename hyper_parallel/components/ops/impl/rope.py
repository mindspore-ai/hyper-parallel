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

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec


class RopeFn(Protocol):
    """Apply split-half rotary position embeddings to a prefix of each head.

    Compute prefix * cos + rotate_half(prefix) * sin, where rotate_half maps
    [first, second] to [-second, first]. The frequency width must be even and
    no larger than the head width; any remaining tail is preserved unchanged.
    A zero-width frequency tensor leaves the input unchanged.
    """

    def __call__(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate the prefix of x covered by the frequencies and retain the tail.

        Args:
            x: States with the head dimension on the final axis.
            cos: Cosines broadcastable to the rotated prefix, with even final width.
            sin: Sines with matching cosine width and broadcast layout.

        Returns:
            Rotated states with the same shape as x, including any unrotated tail.
        """


class InterleavedRopeFn(Protocol):
    """Rotate adjacent Q/K pairs and return the canonical split-half layout.

    The first half of the supplied frequencies rotates adjacent input pairs.
    All rotated even components precede all rotated odd components in each
    output prefix. Any tail beyond the frequency width is preserved. This
    layout conversion is part of the contract, rather than an in-place rotation.
    """

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        *, unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query and key with their rotated pairs laid out as two halves.

        Args:
            x: Query states with head and sequence axes arranged for unsqueeze_dim.
            key: Key states with the same rotary head prefix as the queries.
            cos: Cosines [sequence, rotary_dim] or [batch, sequence, rotary_dim].
            sin: Sines with matching cosine shape; rotary_dim must be even.
            unsqueeze_dim: Head axis inserted into frequencies: 1 for BNSD or 2 for BSND.

        Returns:
            A tuple of rotated query and key tensors, each preserving its original shape.
        """


class SingleInterleavedRopeFn(Protocol):
    """Rotate one adjacent-pair prefix into split real/imaginary half-bands.

    This is the single-input form of InterleavedRopeFn: each even component
    precedes each odd component in the output, and the nonrotary tail is unchanged.
    """

    def __call__(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        """Return one rotated tensor without computing a second Q/K branch.

        Args:
            x: States [B, N, S, D] or [B, S, N, D].
            cos: Frequencies [S, R] or [B, S, R], with even R no larger than D.
            sin: Sines matching cos; the first half of frequencies rotates adjacent pairs.
            unsqueeze_dim: Head axis: 1 for BNSD or 2 for BSND.

        Returns:
            Rotated states preserving the input shape and unrotated tail.
        """


ROPE = OpSpec[RopeFn](config_key='rope', variant='half')
ROPE_INTERLEAVED = OpSpec[InterleavedRopeFn](config_key='rope', variant='interleaved')
ROPE_INTERLEAVED_SINGLE = OpSpec[SingleInterleavedRopeFn](config_key='rope', variant='interleaved_single')


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the prefix of x covered by the frequencies and retain the tail.

    Args:
        x: States with the head dimension on the final axis.
        cos: Cosines broadcastable to the rotated prefix, with even final width.
        sin: Sines with matching cosine width and broadcast layout.

    Returns:
        Rotated states with the same shape as x, including any unrotated tail.

    Notes:
        Apply split-half rotary position embeddings to a prefix of each head.

        Compute prefix * cos + rotate_half(prefix) * sin, where rotate_half maps
        [first, second] to [-second, first]. The frequency width must be even and
        no larger than the head width; any remaining tail is preserved unchanged.
        A zero-width frequency tensor leaves the input unchanged.
    """
    operation = _select_for_call(ROPE, device=x.device)
    return operation(x, cos, sin)


def rope_interleaved(
    x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query and key with their rotated pairs laid out as two halves.

    Args:
        x: Query states with head and sequence axes arranged for unsqueeze_dim.
        key: Key states with the same rotary head prefix as the queries.
        cos: Cosines [sequence, rotary_dim] or [batch, sequence, rotary_dim].
        sin: Sines with matching cosine shape; rotary_dim must be even.
        unsqueeze_dim: Head axis inserted into frequencies: 1 for BNSD or 2 for BSND.

    Returns:
        A tuple of rotated query and key tensors, each preserving its original shape.

    Notes:
        Rotate adjacent Q/K pairs and return the canonical split-half layout.

        The first half of the supplied frequencies rotates adjacent input pairs.
        All rotated even components precede all rotated odd components in each
        output prefix. Any tail beyond the frequency width is preserved. This
        layout conversion is part of the contract, rather than an in-place rotation.
    """
    operation = _select_for_call(ROPE_INTERLEAVED, device=x.device)
    return operation(x, key, cos, sin, unsqueeze_dim=unsqueeze_dim)


def rope_interleaved_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Rotate one adjacent-pair prefix and return canonical split-half features.

    Args:
        x: States [B, N, S, D] or [B, S, N, D].
        cos: Frequencies [S, R] or [B, S, R], with even R no larger than D.
        sin: Sines matching cos.
        unsqueeze_dim: Head axis: 1 for BNSD or 2 for BSND.

    Returns:
        Rotated states preserving the input shape and nonrotary tail.

    Notes:
        The first half of the supplied frequencies rotates adjacent input pairs.
        The output prefix stores all even components followed by all odd components.
        Only one tensor is rotated; no unused key branch is evaluated.
    """
    operation = _select_for_call(ROPE_INTERLEAVED_SINGLE, device=x.device)
    return operation(x, cos, sin, unsqueeze_dim=unsqueeze_dim)


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


def _native_interleaved_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Rotate one adjacent-pair input into canonical split-half layout."""
    if unsqueeze_dim not in (1, 2):
        raise ValueError("Interleaved RoPE supports unsqueeze_dim 1 or 2")
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    rotary_dim = cos.shape[-1]
    if rotary_dim != sin.shape[-1] or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("RoPE requires matching, even frequency dimensions within the input head dimension")
    if not rotary_dim:
        return x
    cos = cos[..., : rotary_dim // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : rotary_dim // 2].unsqueeze(unsqueeze_dim)
    return torch.cat((_rotate_adjacent(x[..., :rotary_dim], cos, sin), x[..., rotary_dim:]), dim=-1)


def _native_interleaved(
    x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same single-input rotation to each Q/K branch."""
    return (
        _native_interleaved_single(x, cos, sin, unsqueeze_dim=unsqueeze_dim),
        _native_interleaved_single(key, cos, sin, unsqueeze_dim=unsqueeze_dim),
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


def _apply_interleaved_rope(
    tensor: torch.Tensor,
    pass_through: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> torch.Tensor:
    """Apply interleaved RoPE and append dimensions that bypass rotation."""
    torch_npu = require_package("torch_npu")
    sequence_first = (
        tensor.permute(2, 0, 1, 3)
        if unsqueeze_dim == 1
        else tensor.permute(1, 0, 2, 3)
    )
    seq_length, batch_size, num_heads, head_dim = sequence_first.shape
    if batch_size > 1 and seq_length > 1:
        rotated = torch_npu.npu_rotary_mul(
            sequence_first.reshape(batch_size * seq_length, 1, num_heads, head_dim),
            cos.reshape(batch_size * seq_length, 1, 1, head_dim),
            sin.reshape(batch_size * seq_length, 1, 1, head_dim),
            rotary_mode="interleave",
        ).reshape(seq_length, batch_size, num_heads, head_dim)
    else:
        rotated = torch_npu.npu_rotary_mul(
            sequence_first.clone(),
            cos.unsqueeze(-2),
            sin.unsqueeze(-2),
            rotary_mode="interleave",
        )
    rotated = rotated.permute(1, 2, 0, 3)
    # Transpose pairs instead of strided slices to avoid scattered writes in backward.
    rotated = rotated.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
    if unsqueeze_dim == 2:
        rotated = rotated.transpose(1, 2)
    return torch.cat((rotated, pass_through), dim=-1) if pass_through.shape[-1] else rotated


def _npu_interleaved_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Adapt one adjacent-pair tensor to the NPU rotation kernel."""
    if unsqueeze_dim not in (1, 2):
        raise ValueError("NPU interleaved RoPE supports unsqueeze_dim 1 or 2")
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    rotary_dim = cos.shape[-1]
    if rotary_dim != sin.shape[-1] or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("RoPE requires matching, even frequency dimensions within the input head dimension")
    if not rotary_dim:
        return x
    rotated, tail = x.split((rotary_dim, x.shape[-1] - rotary_dim), dim=-1)
    # Transformers repeats frequency half-bands; the NPU expects adjacent pairs.
    cos = cos[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    if cos.shape[0] == 1 and x.shape[0] > 1:
        cos = cos.expand(x.shape[0], -1, -1)
        sin = sin.expand(x.shape[0], -1, -1)
    return _apply_interleaved_rope(rotated, tail, cos.transpose(0, 1), sin.transpose(0, 1), unsqueeze_dim)


def _npu_interleaved(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared single-input NPU adapter to both Q/K branches."""
    return (
        _npu_interleaved_single(q, cos, sin, unsqueeze_dim=unsqueeze_dim),
        _npu_interleaved_single(k, cos, sin, unsqueeze_dim=unsqueeze_dim),
    )


def _load_npu_interleaved_single() -> SingleInterleavedRopeFn:
    """Load the single-input NPU rotary adapter on demand."""
    require_symbol(require_package("torch_npu"), "npu_rotary_mul")
    return _npu_interleaved_single


def _load_npu_interleaved() -> InterleavedRopeFn:
    """Load the optional NPU adjacent-pair rotary embedding adapter."""
    require_symbol(require_package("torch_npu"), "npu_rotary_mul")

    def run(
        x: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the selected rotation while preserving the unrotated tail."""
        if not cos.shape[-1]:
            return x, key
        return _npu_interleaved(x, key, cos, sin, unsqueeze_dim=unsqueeze_dim)

    return run


def register_rope(registry: OpRegistry) -> None:
    """Register half, interleaved and partial rotation on compatible devices."""
    registry.register_family("rope", specs=(ROPE, ROPE_INTERLEAVED, ROPE_INTERLEAVED_SINGLE))
    registry.register_impl(ROPE, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(
        ROPE_INTERLEAVED, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_interleaved,
    )
    registry.register_impl(ROPE, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(ROPE_INTERLEAVED, name="torch_npu", devices=("npu",), load=_load_npu_interleaved)
    registry.register_impl(
        ROPE_INTERLEAVED_SINGLE, name="native", devices=("cpu", "cuda", "npu"),
        load=lambda: _native_interleaved_single,
    )
    registry.register_impl(
        ROPE_INTERLEAVED_SINGLE, name="torch_npu", devices=("npu",), load=_load_npu_interleaved_single,
    )
