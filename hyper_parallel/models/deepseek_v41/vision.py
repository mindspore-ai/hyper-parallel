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
"""Training-capable DeepSeek-V4.1 native vision tower and 3x3 aligner."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import


def _vision_cos_sin(
        grid_height: int,
        grid_width: int,
        rope_dim: int,
        rope_theta: float,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DeepSeek's flattened 2D RoPE coefficients for one image grid."""
    inverse_frequency = 1.0 / (
        rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    height_positions = torch.arange(grid_height, device=device).unsqueeze(1).expand(grid_height, grid_width)
    width_positions = torch.arange(grid_width, device=device).unsqueeze(0).expand(grid_height, grid_width)
    frequencies = torch.stack((height_positions, width_positions), dim=-1).reshape(-1, 2, 1).float()
    frequencies = (frequencies * inverse_frequency).flatten(1)
    return frequencies.cos().unsqueeze(1), frequencies.sin().unsqueeze(1)


def _apply_vision_rotary(
        tensor: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
) -> torch.Tensor:
    """Apply the released split-half 2D RoPE convention to vision Q or K."""
    data_type = tensor.dtype
    first_half, second_half = tensor.float().chunk(2, dim=-1)
    return torch.cat((first_half * cosine - second_half * sine, second_half * cosine + first_half * sine), dim=-1).to(
        data_type
    )


class DeepseekV41VisionRMSNorm(nn.Module):
    """The FP32-weight RMSNorm used by the released V4.1 vision tower."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        """Create a vision RMSNorm.

        Args:
            hidden_size: Vision hidden dimension.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize a patch sequence while retaining its activation dtype."""
        data_type = hidden_states.dtype
        hidden_fp32 = hidden_states.float()
        normalized = hidden_fp32 * torch.rsqrt(hidden_fp32.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * normalized).to(data_type)


class DeepseekV41VisionPatchEmbed(nn.Module):
    """Project flattened RGB patches into the vision hidden space."""

    def __init__(self, patch_size: int, hidden_size: int) -> None:
        """Create the released linear patch projection."""
        super().__init__()
        self.proj = nn.Linear(3 * patch_size ** 2, hidden_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed ``[patches, 3, patch, patch]`` RGB patches."""
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError(f"pixel patches must have shape [N, 3, P, P], got {tuple(patches.shape)}")
        return self.proj(patches.flatten(1))


class DeepseekV41VisionAttention(nn.Module):
    """Bidirectional full-image attention with DeepSeek 2D RoPE."""

    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        """Create fused QKV and output projections.

        Args:
            hidden_size: Vision hidden dimension.
            num_attention_heads: Number of equal-size attention heads.
        """
        super().__init__()
        if hidden_size % num_attention_heads:
            raise ValueError("vision hidden_size must divide evenly into num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        # Keep the released checkpoint names (wqkv / wo) stable for future
        # full-checkpoint loading.
        self.wqkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cosine: torch.Tensor,
            sine: torch.Tensor,
    ) -> torch.Tensor:
        """Run full bidirectional attention for one image's patch sequence."""
        token_count = hidden_states.shape[0]
        query, key, value = (
            value.view(token_count, self.num_attention_heads, self.head_dim)
            for value in self.wqkv(hidden_states).chunk(3, dim=-1)
        )
        query = _apply_vision_rotary(query, cosine, sine)
        key = _apply_vision_rotary(key, cosine, sine)
        attention_output = functional.scaled_dot_product_attention(  # pylint: disable=not-callable
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            is_causal=False,
        )
        return self.wo(attention_output.transpose(0, 1).reshape(token_count, -1))


class DeepseekV41VisionMLP(nn.Module):
    """SwiGLU vision MLP matching the released parameter layout."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Create the fused gate/up and down projections."""
        super().__init__()
        # ``w1`` is a fused gate/up matrix and ``w2`` projects back to the
        # vision hidden size, matching ``inference/vision.py``.
        self.w1 = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the vision SwiGLU MLP."""
        gate, up = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(functional.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    """One pre-normalized full-attention vision Transformer block."""

    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int) -> None:
        """Create a released V4.1 vision block."""
        super().__init__()
        self.norm1 = DeepseekV41VisionRMSNorm(hidden_size)
        self.attn = DeepseekV41VisionAttention(hidden_size, num_attention_heads)
        self.norm2 = DeepseekV41VisionRMSNorm(hidden_size)
        self.mlp = DeepseekV41VisionMLP(hidden_size, intermediate_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cosine: torch.Tensor,
            sine: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention then MLP residual updates."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cosine, sine)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class DeepseekV41VisionTower(nn.Module):
    """DeepSeek V4.1 ViT over one independently processed image grid."""

    def __init__(self, config: Any) -> None:
        """Create the configured vision tower from V4.1 extension fields."""
        super().__init__()
        self.hidden_size = int(config.v41_vision_hidden_size)
        self.patch_size = int(config.v41_vision_patch_size)
        self.num_attention_heads = int(config.v41_vision_num_attention_heads)
        self.rope_theta = float(config.v41_vision_rope_theta)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("V4.1 vision hidden_size must be divisible by its attention head count")
        self.rope_dim = self.hidden_size // self.num_attention_heads // 2
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError("V4.1 vision RoPE dimension must be a positive even number")
        self.patch_embed = DeepseekV41VisionPatchEmbed(self.patch_size, self.hidden_size)
        self.blocks = nn.ModuleList([
            DeepseekV41VisionBlock(
                self.hidden_size,
                self.num_attention_heads,
                int(config.v41_vision_intermediate_size),
            )
            for _ in range(int(config.v41_vision_num_hidden_layers))
        ])
        self.norm = DeepseekV41VisionRMSNorm(self.hidden_size)

    def forward(self, patches: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        """Encode one image's raster-order ViT patches.

        Args:
            patches: Normalized RGB patches in raster order.
            grid_height: Number of ViT patch rows.
            grid_width: Number of ViT patch columns.

        Returns:
            Vision features with one row per input patch.
        """
        expected_patches = grid_height * grid_width
        if expected_patches <= 0 or patches.shape[0] != expected_patches:
            raise ValueError(
                "vision patch count must equal grid_height * grid_width; "
                f"got patches={patches.shape[0]}, grid=({grid_height}, {grid_width})"
            )
        hidden_states = self.patch_embed(patches)
        cosine, sine = _vision_cos_sin(grid_height, grid_width, self.rope_dim, self.rope_theta, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cosine, sine)
        return self.norm(hidden_states)


class DeepseekV41VisionAligner(nn.Module):
    """Pixel-unshuffle 3x3 (or configured) ViT features into LLM hidden rows."""

    def __init__(self, config: Any) -> None:
        """Create the released two-layer aligner."""
        super().__init__()
        self.downsample_ratio = int(config.v41_vision_downsample_ratio)
        vision_hidden_size = int(config.v41_vision_hidden_size)
        llm_hidden_size = int(config.hidden_size)
        if self.downsample_ratio <= 0:
            raise ValueError("V4.1 vision downsample_ratio must be positive")
        self.w1 = nn.Linear(vision_hidden_size * self.downsample_ratio ** 2, llm_hidden_size)
        self.w2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, hidden_states: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        """Downsample one image's ViT grid to the LLM image-token grid."""
        if hidden_states.ndim != 2 or hidden_states.shape[0] != grid_height * grid_width:
            raise ValueError("aligner features must be [grid_height * grid_width, vision_hidden]")
        ratio = self.downsample_ratio
        feature_grid = hidden_states.view(grid_height, grid_width, -1).permute(2, 0, 1)
        feature_grid = functional.pad(feature_grid, (0, -grid_width % ratio, 0, -grid_height % ratio))
        unshuffled = functional.unfold(feature_grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(functional.gelu(self.w1(unshuffled)))  # pylint: disable=not-callable


__all__ = [
    "DeepseekV41VisionAligner",
    "DeepseekV41VisionTower",
    "DeepseekV41VisionRMSNorm",
]
