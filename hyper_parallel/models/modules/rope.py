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
"""Rotary Position Embedding — shared building block."""
import os
from typing import List, Optional

import torch
from torch import nn


def _use_npu_rotary_mul() -> bool:
    """True when ``HYPER_USE_V1_KERNELS=1`` and ``torch_npu`` is importable.

    Selects ``torch_npu.npu_rotary_mul`` for the text-side rotary apply;
    vision RoPE stays in the eager path.
    """
    if os.environ.get("HYPER_USE_V1_KERNELS", "0") != "1":
        return False
    try:
        import torch_npu  # pylint: disable=C0415,W0611
        return True
    except ImportError:
        return False


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Standard RoPE implementation matching Llama / Qwen / Mistral conventions.

    Accepts ``position_ids`` of shape ``(seq_len,)``, ``(bs, seq_len)``, or
    ``(3, bs, seq_len)`` (3D mRoPE form). For 3D input, this base class
    just collapses the T-dimension (slice 0) to behave as plain 1D RoPE —
    the ``MultiModalRotaryEmbedding`` subclass overrides ``forward`` to do
    the proper interleaved mRoPE math.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = self._build_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self) -> "torch.Tensor":
        return 1.0 / (
            self.theta ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
            )
        )

    def reset_inv_freq(self) -> None:  # pylint: disable=W0613
        """Re-derive ``inv_freq`` from ``theta`` and ``dim``.

        Call this after ``model.to_empty(device=...)`` followed by buffer
        zero-init: ``inv_freq`` is registered as a non-persistent buffer,
        so the trainer's meta-materialization step zeros its storage along
        with all other buffers, silently disabling RoPE (cos=1, sin=0
        → identity rotation). Re-running the canonical formula on the
        local device restores the correct frequency table without
        triggering an extra all-gather.
        """
        new_inv_freq = self._build_inv_freq().to(self.inv_freq.device)
        self.inv_freq.copy_(new_inv_freq)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):  # pylint: disable=W0613
        # 3D mRoPE position_ids: collapse to T (first) for plain RoPE; the
        # mRoPE subclass overrides this method for real (T, H, W) handling.
        """Forward pass."""
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
        # ``torch.outer`` requires 1-D inputs; flatten away batch if present.
        flat = position_ids.float().reshape(-1)
        freqs = torch.outer(flat, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class MultiModalRotaryEmbedding(RotaryEmbedding):
    """Interleaved multi-modal RoPE used by Qwen3-VL / Qwen2-VL.

    Splits each rotary frequency band into 3 sub-bands keyed by
    ``mrope_section = [t_dim, h_dim, w_dim]`` (e.g. ``[24, 20, 20]`` for
    head_dim=128). For text tokens the three sub-bands all see the same
    position, so it reduces to plain 1D RoPE. For image tokens the three
    sub-bands see (t, h, w) positions respectively, encoding 2-D spatial
    layout into the rotary phase.

    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        mrope_section: Optional[List[int]] = None,
    ):
        super().__init__(dim=dim, max_seq_len=max_seq_len, theta=theta)
        if mrope_section is None:
            raise ValueError(
                "MultiModalRotaryEmbedding requires non-empty mrope_section"
            )
        if sum(mrope_section) * 2 != dim:
            raise ValueError(
                f"sum(mrope_section)*2 ({sum(mrope_section) * 2}) must equal "
                f"head_dim ({dim}); got mrope_section={mrope_section}"
            )
        self.mrope_section = list(mrope_section)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):  # pylint: disable=W0613
        # ``@torch.no_grad`` keeps cos/sin out of the autograd graph.
        # Promote 1-D / 2-D position_ids to the 3-D mRoPE form by replicating
        # — text-only inputs still work, mRoPE just collapses to plain RoPE.
        if position_ids.ndim == 1:
            position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ``autocast(enabled=False)`` forces fp32 matmul under bf16 outer
        # autocast — bf16 here causes ~1% gnorm drift that compounds via Adam.
        device_type = x.device.type
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1,
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        with torch.amp.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

        # Interleaved mRoPE: ``freqs[0]`` (T) is the base; H / W overwrite
        # every-3rd slot within ``mrope_section[dim] * 3`` channels.
        out = freqs[0].clone()  # (bs, seq_len, dim/2)
        for dim_idx, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            out[..., idx] = freqs[dim_idx][..., idx]

        emb = torch.cat((out, out), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key tensors.

    Accepts ``cos`` / ``sin`` of shape ``(seq_len, head_dim)`` or
    ``(bs, seq_len, head_dim)``. Q/K are expected to be ``(bs, n_heads,
    seq_len, head_dim)``.

    When ``cos.shape[-1] < q.shape[-1]``, only the leading
    ``cos.shape[-1]`` dimensions of each head are rotated and the
    trailing dims are passed through unchanged — this is the
    ``partial_rotary_factor < 1`` case used by Qwen3.5 (only the first
    ``head_dim * partial_rotary_factor`` dims get RoPE).
    """
    if cos.ndim == 2:
        # (seq_len, head_dim) → (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    elif cos.ndim == 3:
        # (bs, seq_len, head_dim) → (bs, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1).to(q.dtype)
        sin = sin.unsqueeze(1).to(q.dtype)

    rope_dim = cos.shape[-1]
    if rope_dim == q.shape[-1]:
        if _use_npu_rotary_mul() and q.device.type == "npu":
            # pylint: disable=C0415
            import torch_npu
            q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
            k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
            return q_embed, k_embed
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # Partial rotary: rotate the leading ``rope_dim`` channels, pass-through the rest.
    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)
