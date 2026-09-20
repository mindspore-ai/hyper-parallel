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
"""NPU-accelerated Transformers-compatible rotary position embeddings."""

from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import
import torch_npu


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NPU RoPE using the Transformers ``apply_rotary_pos_emb`` contract.

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
        ValueError: If the rotary dimension does not cover the complete Q/K
            head dimension supported by the NPU operator interface.
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.shape[0] == 1 and q.shape[0] > 1:
        cos = cos.expand(q.shape[0], -1, -1)
        sin = sin.expand(q.shape[0], -1, -1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    if rotary_dim != q.shape[-1] or rotary_dim != k.shape[-1]:
        raise ValueError(
            "NPU RoPE requires the rotary dimension to equal the Q/K head dimension"
        )
    q_embed = torch_npu.npu_rotary_mul(q.clone(), cos, sin, rotary_mode="half")
    k_embed = torch_npu.npu_rotary_mul(k.clone(), cos, sin, rotary_mode="half")
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NPU RoPE using Transformers' interleaved RoPE contract.

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
    if unsqueeze_dim not in (1, 2):
        raise ValueError("NPU interleaved RoPE supports unsqueeze_dim 1 or 2")
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    rotary_dim = cos.shape[-1]
    if rotary_dim % 2:
        raise ValueError("interleaved RoPE requires an even rotary dimension")
    if rotary_dim > q.shape[-1] or rotary_dim > k.shape[-1]:
        raise ValueError("RoPE frequencies cannot exceed the Q/K head dimension")
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Transformers supplies cat(freqs, freqs). The NPU interleave mode expects
    # each frequency next to the corresponding real/imaginary input pair.
    cos = cos[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : rotary_dim // 2].repeat_interleave(2, dim=-1)
    if cos.shape[0] == 1 and q.shape[0] > 1:
        cos = cos.expand(q.shape[0], -1, -1)
        sin = sin.expand(q.shape[0], -1, -1)
    cos = cos.transpose(0, 1)
    sin = sin.transpose(0, 1)

    rotated_tensors: list[torch.Tensor] = []
    for tensor, pass_through in ((q_rot, q_pass), (k_rot, k_pass)):
        # The interleave kernel requires sequence-first input when both batch
        # and sequence dimensions are greater than one.
        sequence_first = (
            tensor.permute(2, 0, 1, 3)
            if unsqueeze_dim == 1
            else tensor.permute(1, 0, 2, 3)
        )
        seq_length, batch_size, num_heads, head_dim = sequence_first.shape
        if batch_size > 1 and seq_length > 1:
            rotated = torch_npu.npu_rotary_mul(
                sequence_first.reshape(
                    batch_size * seq_length, 1, num_heads, head_dim
                ),
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
        # Restore the output ordering returned by Transformers.
        rotated = torch.cat((rotated[..., 0::2], rotated[..., 1::2]), dim=-1)
        if unsqueeze_dim == 2:
            rotated = rotated.transpose(1, 2)
        rotated_tensors.append(torch.cat((rotated, pass_through), dim=-1))

    return rotated_tensors[0], rotated_tensors[1]


def _rotate_half(
        x: torch.Tensor,
        rotary_interleaved: bool = False,
) -> torch.Tensor:
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
    """Apply rotary embeddings using precomputed cosine and sine tensors."""
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

    if use_fused_rotary_pos_emb and HAS_NPU and t.device.type != "cpu":
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
    rot_dim = freqs.shape[-1]
    t_dim = t.shape[-1]
    t_pass = None
    if rot_dim != t_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rotary_mode = "interleave" if rotary_interleaved else "half"
    t_copy = t.clone()
    t = torch_npu.npu_rotary_mul(t_copy, cos_, sin_, rotary_mode=rotary_mode)
    t_copy.untyped_storage().resize_(0)

    if t_pass is not None:
        return torch.cat((t, t_pass), dim=-1)
    return t
