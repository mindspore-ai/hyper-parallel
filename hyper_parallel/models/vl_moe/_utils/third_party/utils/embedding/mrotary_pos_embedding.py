# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from typing import Optional, List
import math

import torch
from torch import Tensor, nn


__all__ = ['MultimodalRotaryEmbedding']


def get_mrope_interleaved_id_list(a: int, b: int, c: int, force_last=False) -> List[int]:
    if force_last:
        a -= 1
    counts = {0: a, 1: b, 2: c}
    placed = {k: 0 for k in counts}   # Number of times each symbol has been placed
    rem = counts.copy()               # Remaining placement count
    seq: List[int] = []
    last = None

    total = a + b + c
    for _ in range(total):
        # Candidates: remaining > 0 and ≠ last
        cands = [k for k in rem if rem[k] > 0 and k != last]
        if not cands:
            # If only last remains, or no last in first iteration, relax the condition
            cands = [k for k in rem if rem[k] > 0]

        # Among candidates, select the most "rare" (smallest placed/total ratio), with smaller index as tiebreaker
        try:
            best = min(
                cands,
                key=lambda k: (placed[k] / counts[k], k)
            )
        except KeyError:
            best = 0

        seq.append(best)
        placed[best] += 1
        rem[best] -= 1
        last = best
    if force_last:
        seq.append(0)
    return seq


def apply_yarn_scaling(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    dim: int,
    rotary_base: int = 10000,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    original_max_position_embeddings: int = 2048,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0
) -> torch.Tensor:
    """
    Apply YaRN (Yet another RoPE extensioN method) scaling to inverse frequencies.

    Args:
        inv_freq: Original inverse frequencies
        scaling_factor: RoPE scaling factor
        dim: Dimension of rotary embedding
        rotary_base: Base period for rotary position embeddings
        beta_fast: Fast beta parameter for YaRN
        beta_slow: Slow beta parameter for YaRN
        original_max_position_embeddings: Original max sequence length
        mscale: Magnitude scaling parameter
        mscale_all_dim: All-dimension magnitude scaling parameter

    Returns:
        Scaled inverse frequencies
    """
    rotary_ratio = rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=inv_freq.device) / dim)
    freq_extra = 1.0 / rotary_ratio
    freq_inter = 1.0 / (scaling_factor * rotary_ratio)

    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        dim,
        rotary_base,
        original_max_position_embeddings,
    )

    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
        device=inv_freq.device, dtype=torch.float32
    )

    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    return inv_freq


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: int = 10000,
    max_position_embeddings: int = 2048
) -> tuple:
    """Find correction range for YaRN scaling."""
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: int = 10000,
    max_position_embeddings: int = 2048
) -> float:
    """Find correction dimension for YaRN scaling."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_linear_ramp_mask(min_: float, max_: float, dim: int) -> torch.Tensor:
    """Generate linear ramp mask for YaRN."""
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MultimodalRotaryEmbedding(nn.Module):
    """
    Multimodal Rotary Position Embedding for vision-language models.

    Supports 2D (height, width) and 3D (time, height, width) position encoding
    with optional high-low frequency interleaving for better distribution.

    Args:
        kv_channels: Projection weights dimension in multi-head attention
        rotary_percent: Percent of rotary dimension to use for rotary position embeddings
        rotary_interleaved: Whether to use interleaved rotary embedding mode
        seq_len_interpolation_factor: Scale of linearly interpolating RoPE for longer sequences
        rotary_base: Base period for rotary position embeddings (default: 10000)
        mrope_section: List specifying dimension split for each modality axis
                       [h, w] for 2D images, [t, h, w] for 3D videos
        mrope_ids_interleaved: Whether to use high-low frequency interleaving
        qk_rope_head_dim: Optional explicit head dimension for Q/K
        rope_scaling_type: Optional RoPE scaling method (e.g., "yarn")
        rope_scaling_factor: Scaling factor for YaRN (required if rope_scaling_type="yarn")
        rope_scaling_beta_fast: Fast beta for YaRN (default: 32.0)
        rope_scaling_beta_slow: Slow beta for YaRN (default: 1.0)
        rope_scaling_original_max_position_embeddings: Original max position for YaRN (default: 2048)
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        mrope_section: list = None,
        mrope_ids_interleaved: bool = False,
        qk_rope_head_dim: Optional[int] = None,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: Optional[float] = None,
        rope_scaling_beta_fast: float = 32.0,
        rope_scaling_beta_slow: float = 1.0,
        rope_scaling_original_max_position_embeddings: int = 2048,
        rope_scaling_mscale: float = 1.0,
        rope_scaling_mscale_all_dim: float = 0.0,
    ) -> None:
        super().__init__()
        """
        kv_channels=53, rotary_percent=1.0, rotary_interleaved=False, seq_len_interpolation_factor=None,
        rotary_base=6400000.0, mrope_section=[12, 10, 10], mrope_ids_interleaved=True, qk_rope_head_dim=64, rope_scaling_type=None
        """

        self.rotary_base = rotary_base
        if qk_rope_head_dim is not None:
            dim = qk_rope_head_dim
        else:
            dim = kv_channels

        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        self.mrope_ids_interleaved = mrope_ids_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor

        # Initialize inverse frequencies
        device = torch.cuda.current_device()
        self.inv_freq = 1.0 / (
            self.rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                / dim
            )
        )

        self.mrope_section = mrope_section

        # Setup high-low frequency interleaving if enabled
        if self.mrope_ids_interleaved:
            if len(self.mrope_section) == 2:
                h_num, w_num = self.mrope_section[0], self.mrope_section[1]
                mrope_dim = get_mrope_interleaved_id_list(h_num, w_num, 0)
            elif len(self.mrope_section) == 3:
                t_num, h_num, w_num = self.mrope_section[0], self.mrope_section[1], self.mrope_section[2]
                mrope_dim = get_mrope_interleaved_id_list(t_num, h_num, w_num, force_last=True)
            else:
                raise AssertionError("Cannot support the length of mrope section is not 2 or 3.")

            if not self.rotary_interleaved:
                mrope_dim = mrope_dim * 2
            else:
                mrope_dim = [item for item in mrope_dim for _ in range(2)]

            # Register buffers (non-persistent to avoid saving in state_dict)
            self.register_buffer("id_D", torch.LongTensor(mrope_dim), persistent=False)
            self.register_buffer("id_S", torch.arange(len(mrope_dim)), persistent=False)

        # Apply YaRN scaling if specified
        if rope_scaling_type == "yarn":
            if rope_scaling_factor is None:
                raise ValueError("rope_scaling_factor must be provided when rope_scaling_type='yarn'")
            self.inv_freq = apply_yarn_scaling(
                self.inv_freq,
                scaling_factor=rope_scaling_factor,
                dim=dim,
                rotary_base=rotary_base,
                beta_fast=rope_scaling_beta_fast,
                beta_slow=rope_scaling_beta_slow,
                original_max_position_embeddings=rope_scaling_original_max_position_embeddings,
            )

        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_mscale = rope_scaling_mscale
        self.rope_scaling_mscale_all_dim = rope_scaling_mscale_all_dim

    def forward(self, position_ids: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        max_seq_len = position_ids.flatten().max() + 1
        parts = len(self.mrope_section)

        # Generate base frequencies (float32)
        seq = (torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype) + offset)
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)

        # Duplicate frequencies based on interleaved mode
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, dim]
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(freqs.shape[0], -1)

        # Index by position_ids to get position-specific embeddings
        emb = emb[position_ids]  # [batch, parts, seq_len, dim]

        if self.mrope_ids_interleaved:
            emb = emb[self.id_D, :, :, self.id_S]
            emb = emb.permute(1, 2, 0)
        else:
            if not self.rotary_interleaved:
                mrope_section = self.mrope_section * 2
            else:
                mrope_section = [item for item in self.mrope_section for _ in range(2)]
            emb = torch.cat(
                [m[i % parts] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1
            )

        # Removed parallel processing code - this version is for single-device use

        mscale = 1.0
        if self.rope_scaling_type == "yarn":
            mscale = float(
                yarn_get_mscale(self.rope_scaling_factor, self.rope_scaling_mscale)
                / yarn_get_mscale(self.rope_scaling_factor, self.rope_scaling_mscale_all_dim)
            )
        return emb.cos() * mscale, emb.sin() * mscale