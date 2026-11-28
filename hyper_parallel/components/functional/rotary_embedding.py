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
"""Transformers-compatible rotary position embeddings and legacy single-tensor helpers."""

from __future__ import annotations

import numpy as np
import torch

from hyper_parallel.components.ops import rope, rope_interleaved, rope_interleaved_single
from hyper_parallel.components.ops._dependency import require_package


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply selected RoPE using the Transformers ``apply_rotary_pos_emb`` contract.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine part of the rotary embedding.
        sin: Sine part of the rotary embedding.
        unsqueeze_dim: Dimension used to make ``cos`` and ``sin`` broadcastable
            to ``q`` and ``k``.

    Returns:
        Rotated query and key tensors with the same shapes as the inputs.

    Raises:
        ValueError: If the frequency dimensions differ, are odd, or exceed
            the Q/K head dimensions.
    """
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return rope(q, cos, sin), rope(k, cos, sin)


def apply_rotary_pos_emb_interleave(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply selected RoPE using Transformers' interleaved RoPE contract.

    Args:
        q: Interleaved-pair query tensor.
        k: Interleaved-pair key tensor.
        cos: Transformers ``cat(freqs, freqs)`` cosine tensor.
        sin: Sine tensor with the same shape and dtype as ``cos``.
        position_ids: Accepted for Transformers API compatibility.
        unsqueeze_dim: Dimension used to make ``cos`` and ``sin`` broadcastable
            to ``q`` and ``k``.

    Returns:
        Rotated query and key tensors with the same shapes as the inputs.

    Raises:
        ValueError: If ``unsqueeze_dim`` or the rotary dimension is unsupported
            by the NPU interleave kernel.
    """
    del position_ids
    return rope_interleaved(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


def apply_rotary_pos_emb_bsnd(
    tensor: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Rotate one BSND tensor, preserving its tail and canonical output layout.

    Interleaved input pairs become split real/imaginary half-bands. Half-layout
    inputs retain their layout. Frequencies are rounded to the input dtype before
    rotation; a missing position pair leaves the tensor unchanged.

    Args:
        tensor: States [batch, sequence, heads, hidden].
        positions: Cosine/sine tables [sequence, rotary_dim] or [batch, sequence, rotary_dim].
        interleaved: Whether the input prefix stores adjacent real/imaginary pairs.

    Returns:
        Rotated states with the input shape and untouched nonrotary tail.
    """
    if positions is None:
        return tensor
    cos, sin = (part.to(tensor.dtype) for part in positions)
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if interleaved:
        return rope_interleaved_single(tensor, cos, sin, unsqueeze_dim=2)
    return rope(tensor, cos.unsqueeze(2), sin.unsqueeze(2))


def _rotate_half(
    x: torch.Tensor,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Rotate the last dimension of ``x`` for rotary position embedding.

    Args:
        x: Input tensor whose last dimension is the rotary dimension.
        rotary_interleaved: If ``True``, rotate adjacent even/odd pairs
            (interleaved layout). If ``False``, split the last dimension
            into two halves and rotate across them (half layout).

    Returns:
        A tensor with the same shape and dtype as ``x``.
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    dim = x.shape[-1]
    index1 = np.ones(dim)
    index1[::2] = 0
    index2 = np.zeros(dim)
    index2[::2] = -1
    rotation_matrix = np.eye(dim, k=1) * index1 + np.eye(dim, k=-1) * index2
    rotation_matrix = torch.from_numpy(rotation_matrix[None, None, :, :]).to(x.dtype).to(x.device)
    return torch.matmul(x, rotation_matrix)


def apply_rotary_pos_emb_single(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_interleaved: bool = False,
    use_fused_rotary_pos_emb: bool = False,
) -> torch.Tensor:
    """Apply interleaved RoPE and append dimensions that bypass rotation.

    Args:
        tensor: Tensor slice to rotate (the rotary part of Q or K).
        pass_through: Tensor slice that bypasses rotation, concatenated
            back after the rotated part.
        cos: Cosine tensor for the interleaved rotary mode.
        sin: Sine tensor for the interleaved rotary mode.
        unsqueeze_dim: Dimension used to make ``cos`` and ``sin``
            broadcastable to ``tensor``. Must be 1 or 2.

    Returns:
        A tensor of shape ``..., rotary_dim + pass_through_dim`` formed by
        concatenating the rotated part and ``pass_through`` along the last
        dimension.
    """
    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]
    t_pass = None
    if rot_dim != t_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    while cos.ndim < t.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    if use_fused_rotary_pos_emb and t.device.type != "cpu":
        torch_npu = require_package("torch_npu")
        rotary_mode = "interleave" if rotary_interleaved else "half"
        sequence_length, batch_size, num_heads, head_dim = t.shape
        if rotary_interleaved and batch_size > 1 and sequence_length > 1:
            t = torch_npu.npu_rotary_mul(
                t.reshape(batch_size * sequence_length, 1, num_heads, head_dim),
                cos.reshape(batch_size * sequence_length, 1, cos.shape[-2], cos.shape[-1]),
                sin.reshape(batch_size * sequence_length, 1, sin.shape[-2], sin.shape[-1]),
                rotary_mode=rotary_mode,
            ).reshape(sequence_length, batch_size, num_heads, head_dim)
        else:
            t = torch_npu.npu_rotary_mul(t.clone(), cos, sin, rotary_mode=rotary_mode)
    else:
        t = (t * cos) + (_rotate_half(t, rotary_interleaved) * sin)

    if rot_dim != t_dim:
        return torch.cat((t, t_pass), dim=-1)
    return t


def apply_rotary_pos_emb_single_freqs(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings from raw frequency values.

    Args:
        t: Input tensor to rotate.
        freqs: Frequency tensor whose last dimension is the rotary
            dimension. ``cos(freqs)`` and ``sin(freqs)`` are used
            internally.
        rotary_interleaved: If ``True``, use interleaved rotary mode;
            otherwise use half rotary mode.

    Returns:
        A tensor with the same shape as ``t``. If ``freqs.shape[-1]`` is
        smaller than ``t.shape[-1]``, the remaining dimensions of ``t``
        are passed through unchanged and concatenated back.
    """
    rot_dim = freqs.shape[-1]
    t_dim = t.shape[-1]
    t_pass = None
    if rot_dim != t_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rotary_mode = "interleave" if rotary_interleaved else "half"
    torch_npu = require_package("torch_npu")
    t = torch_npu.npu_rotary_mul(t.clone(), cos_, sin_, rotary_mode=rotary_mode)

    if t_pass is not None:
        return torch.cat((t, t_pass), dim=-1)
    return t
