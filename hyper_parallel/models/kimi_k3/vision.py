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
"""Trainable MoonViT mathematics with the K3 Hugging Face parameter layout.

The architecture follows the published K3 configuration. The canonical path uses
native differentiable Torch operations and does not require an inference plugin.
"""

from __future__ import annotations

import math
import importlib
from itertools import accumulate
from typing import Any

# Native Torch encoder, consistent with the DeepSeek-V4.1 vision integration.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.kimi_k3.adapter.attention import npu_attention


Grid = tuple[tuple[int, int, int], ...]


def validate_grid(grid_thws: torch.Tensor, *, max_frames: int = 4) -> Grid:
    """Validate vision metadata and materialize its small host-side grid table.

    Args:
        grid_thws: Integer [media, 3] tensor containing frame, patch-row and
            patch-column counts. Spatial counts must be positive even values.
        max_frames: Largest supported frame count per media item.

    Returns:
        Ordered (frames, height, width) tuples in flattened pixel-patch order.

    Raises:
        ValueError: Grid shape, dtype, frame count or spatial bounds are unsupported.
    """
    if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
        raise ValueError("grid_thws must have shape [number_of_media, 3]")
    if grid_thws.dtype not in (torch.int32, torch.int64):
        raise ValueError("grid_thws must use an integer dtype")
    # Grids are small batch metadata; resolve them once, not inside each block.
    grids = tuple(tuple(row) for row in grid_thws.tolist())
    if any(t < 1 or t > max_frames or h < 2 or w < 2 or h % 2 or w % 2 for t, h, w in grids):
        raise ValueError("K3 grids require 1 <= frames <= max_frames and positive even height/width")
    if any(h > 512 or w > 512 for _, h, w in grids):
        raise ValueError("K3 vision RoPE supports grid height/width up to 512")
    return grids


def temporal_positions(frames: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    """Build the parameter-independent sine/cosine temporal position table.

    Args:
        frames: Number of temporal positions.
        hidden_size: Even number of embedding channels.
        device: Device on which to construct the table.

    Returns:
        FP32 positions shaped [frames, 1, hidden_size].

    Raises:
        ValueError: hidden_size is odd.
    """
    if hidden_size % 2:
        raise ValueError("vision hidden_size must be even for temporal positions")
    frequencies = 10000.0 ** (-torch.arange(hidden_size // 2, device=device, dtype=torch.float32)
                             / (hidden_size // 2))
    angles = torch.arange(frames, device=device, dtype=torch.float32)[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1).unsqueeze(1)


class InterpolatedPositionEmbedding(nn.Module):
    """Learned spatial positions and rebuildable, nonpersistent temporal state."""

    def __init__(self, config: Any) -> None:
        """Allocate learned spatial positions and rebuildable temporal state."""
        super().__init__()
        self.num_frames = config.init_pos_emb_time
        self.hidden_size = config.vt_hidden_size
        self.interpolation_mode = config.pos_emb_interpolation_mode
        if self.interpolation_mode not in ("bilinear", "bicubic"):
            raise ValueError("K3 position interpolation requires bilinear or bicubic")
        self.weight = nn.Parameter(torch.empty(
            config.init_pos_emb_height, config.init_pos_emb_width, self.hidden_size))
        register_rebuildable_buffer(
            self, "time_weight",
            factory=lambda context: temporal_positions(self.num_frames, self.hidden_size, context.device),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the learned spatial table, including after meta allocation."""
        nn.init.normal_(self.weight)

    def forward(self, hidden_states: torch.Tensor, grids: Grid,
                patch_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Add spatial and temporal positions in flattened media/patch order.

        Args:
            hidden_states: Patch embeddings [sum(frames * height * width), hidden].
            grids: Validated media grids in the same order as the patch embeddings.
            patch_indices: Optional global patch indices for an encoder sequence shard.

        Returns:
            Position-adjusted embeddings with the input shape and dtype.
        """
        positions = []
        for frames, height, width in grids:
            if (height, width) == self.weight.shape[:2]:
                spatial = self.weight.reshape(-1, self.hidden_size)
            else:
                spatial = F.interpolate(
                    self.weight.permute(2, 0, 1).unsqueeze(0), size=(height, width),
                    mode=self.interpolation_mode, align_corners=False,
                ).squeeze(0).permute(1, 2, 0).reshape(-1, self.hidden_size)
            if frames > 1:
                spatial = (spatial.unsqueeze(0) + self.time_weight[:frames]).reshape(-1, self.hidden_size)
            positions.append(spatial)
        positions = torch.cat(positions)
        if patch_indices is not None:
            positions = positions.index_select(0, patch_indices)
        return hidden_states + positions.to(hidden_states.dtype)


class MoonVisionPatchEmbed(nn.Module):
    """Project flattened image patches, retaining official ``proj/pos_emb`` keys."""

    def __init__(self, config: Any) -> None:
        """Build patch projection and interpolated positional embeddings."""
        super().__init__()
        self.proj = nn.Conv2d(3, config.vt_hidden_size, config.patch_size, config.patch_size,
                              bias=config.patch_embed_proj_bias)
        self.pos_emb = InterpolatedPositionEmbedding(config)

    def forward(self, pixels: torch.Tensor, grids: Grid,
                patch_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Project flattened pixel patches using the active FSDP compute dtype.

        Args:
            pixels: Flattened patches [patch_count, 3, patch_size, patch_size].
            grids: Validated media grids partitioning the patch sequence.
            patch_indices: Optional global positions of the local patch rows.

        Returns:
            Position-adjusted patch embeddings [patch_count, vision_hidden_size].
        """
        hidden_states = self.proj(pixels.to(self.proj.weight.dtype)).flatten(1)
        return self.pos_emb(hidden_states, grids, patch_indices)


def spatial_rope_angles(grids: Grid, head_dim: int, device: torch.device) -> torch.Tensor:
    """Build x/y-interleaved rotation angles for adjacent channel pairs.

    Args:
        grids: Validated media grids in flattened patch order.
        head_dim: Attention head width, divisible by four.
        device: Device on which to construct the angles.

    Returns:
        FP32 angles [patch_count, head_dim // 2], repeated across each media's frames.

    Raises:
        ValueError: head_dim is not divisible by four.
    """
    if head_dim % 4:
        raise ValueError("vision attention head_dim must be divisible by four")
    frequencies = 10000.0 ** (-torch.arange(0, head_dim, 4, device=device, dtype=torch.float32) / head_dim)
    media_angles = []
    for frames, height, width in grids:
        positions = torch.arange(height * width, device=device)
        x_angles = (positions % width).float()[:, None] * frequencies
        y_angles = (positions // width).float()[:, None] * frequencies
        media_angles.append(torch.stack((x_angles, y_angles), dim=-1).flatten(1).repeat(frames, 1))
    return torch.cat(media_angles)


def apply_spatial_rope(values: torch.Tensor, angles: torch.Tensor, backend: str = "eager") -> torch.Tensor:
    """Rotate adjacent channel pairs in FP32 and restore the input dtype.

    Args:
        values: Query or key tensor [patch_count, heads, head_dim].
        angles: Spatial angles [patch_count, head_dim // 2].
        backend: eager for native operations or npu for the Ascend rotary primitive.

    Returns:
        Rotated values with unchanged shape and adjacent-pair channel order.

    Raises:
        ValueError: The backend is unsupported or npu receives non-Ascend values.
    """
    if backend == "npu":
        if values.device.type != "npu":
            raise ValueError("vision npu RoPE requires Ascend tensors")
        # The generic HF interleave adapter changes output channel order; use
        # its underlying primitive directly to preserve K3's adjacent pairs.
        operator = importlib.import_module("torch_npu").npu_rotary_mul
        cosine = angles.cos().repeat_interleave(2, dim=-1)[:, None, None, :]
        sine = angles.sin().repeat_interleave(2, dim=-1)[:, None, None, :]
        return operator(values.float().unsqueeze(1).clone(), cosine, sine,
                        rotary_mode="interleave").squeeze(1).to(values.dtype)
    if backend != "eager":
        raise ValueError("vision rope_backend must be eager or npu")
    pairs = values.float().reshape(*values.shape[:-1], -1, 2)
    cosine = angles.cos().unsqueeze(1)
    sine = angles.sin().unsqueeze(1)
    even, odd = pairs.unbind(-1)
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2).to(values.dtype)


class VisionRMSNorm(nn.RMSNorm):
    """Keep RMSNorm parameter/FSDP ownership while selecting a local kernel."""

    def __init__(self, hidden_size: int, backend: str = "eager") -> None:
        """Preserve PyTorch's accumulation-dtype default epsilon."""
        super().__init__(hidden_size)
        self.backend = backend

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize using the eager epsilon convention for the selected kernel.

        Args:
            hidden_states: Input tensor with the configured hidden width last.

        Returns:
            Normalized values with the same shape.

        Raises:
            ValueError: An NPU implementation receives non-Ascend values.
        """
        if self.backend == "eager":
            return super().forward(hidden_states)
        if hidden_states.device.type != "npu":
            raise ValueError("vision npu RMSNorm requires Ascend tensors")
        module = importlib.import_module("hyper_parallel.components.functional.rms_norm")
        accumulation_dtype = (torch.float32 if hidden_states.dtype in (torch.float16, torch.bfloat16)
                              else hidden_states.dtype)
        epsilon = self.eps if self.eps is not None else torch.finfo(accumulation_dtype).eps
        return module.rms_norm(hidden_states, self.weight, epsilon)


class VisionMLP(nn.Module):
    """The vision GELU MLP, distinct from the decoder's SiTU experts."""

    def __init__(self, config: Any) -> None:
        """Build the released two-layer approximate-GELU vision MLP."""
        super().__init__()
        self.fc0 = nn.Linear(config.vt_hidden_size, config.vt_intermediate_size, bias=config.linear_bias)
        self.fc1 = nn.Linear(config.vt_intermediate_size, config.vt_hidden_size, bias=config.linear_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the vision-specific tanh approximation to GELU.

        Args:
            hidden_states: Patch embeddings [..., vision_hidden_size].

        Returns:
            Projected embeddings with unchanged leading dimensions and hidden width.
        """
        return self.fc1(F.gelu(self.fc0(hidden_states), approximate="tanh"))  # pylint: disable=not-callable


class MoonViTEncoderLayer(nn.Module):
    """Bidirectional attention isolated per media, followed by a GELU MLP."""

    def __init__(self, config: Any) -> None:
        """Build one bidirectional per-media attention and MLP block."""
        super().__init__()
        self.num_heads = config.vt_num_attention_heads
        if config.qkv_hidden_size % self.num_heads:
            raise ValueError("vision qkv_hidden_size must be divisible by num_attention_heads")
        self.head_dim = config.qkv_hidden_size // self.num_heads
        if self.head_dim % 4:
            raise ValueError("vision head_dim must be divisible by four")
        self.attention_backend = config.attention_backend
        self.rope_backend = config.rope_backend
        self.norm0 = VisionRMSNorm(config.vt_hidden_size, config.norm_backend)
        self.norm1 = VisionRMSNorm(config.vt_hidden_size, config.norm_backend)
        self.wqkv = nn.Linear(config.vt_hidden_size, 3 * config.qkv_hidden_size, bias=config.attn_bias)
        self.wo = nn.Linear(config.qkv_hidden_size, config.vt_hidden_size, bias=config.attn_bias)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, lengths: tuple[int, ...], angles: torch.Tensor) -> torch.Tensor:
        """Apply attention independently to each media segment, followed by the MLP.

        Args:
            hidden_states: Concatenated patch embeddings [patch_count, hidden].
            lengths: Patch counts partitioning hidden_states into media segments.
            angles: Spatial RoPE angles [patch_count, head_dim // 2].

        Returns:
            Updated patch embeddings with the input shape.
        """
        projected = self.wqkv(self.norm0(hidden_states))
        query, key, value = projected.reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
        query, key = (apply_spatial_rope(part, angles, self.rope_backend) for part in (query, key))
        if self.attention_backend == "npu_fusion":
            query, key, value = (part.transpose(0, 1).unsqueeze(0) for part in (query, key, value))
            output = npu_attention(self, query, key, value, None, causal=False, scaling=self.head_dim ** -0.5,
                                   sequence_ends=tuple(accumulate(lengths)))
            output = output.reshape(hidden_states.shape[0], -1)
        else:
            outputs = []
            for query_part, key_part, value_part in zip(query.split(lengths), key.split(lengths), value.split(lengths)):
                query_part, key_part, value_part = (part.transpose(0, 1) for part in (query_part, key_part, value_part))
                scores = (query_part @ key_part.transpose(-1, -2)) / math.sqrt(self.head_dim)
                probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
                outputs.append((probabilities @ value_part).transpose(0, 1).flatten(1))
            output = torch.cat(outputs)
        hidden_states = hidden_states + self.wo(output)
        return hidden_states + self.mlp(self.norm1(hidden_states))


class MoonViTEncoder(nn.Module):
    """Native-depth vision stack with media-local RoPE and no mutable global cache."""

    def __init__(self, config: Any) -> None:
        """Build the complete vision depth and final normalization."""
        super().__init__()
        # Opt-in discovery preserves decoder-only checkpoint layer indices by default.
        if config.recompute:
            self.gradient_checkpointing = False
        self.head_dim = config.qkv_hidden_size // config.vt_num_attention_heads
        self.blocks = nn.ModuleList(MoonViTEncoderLayer(config) for _ in range(config.vt_num_hidden_layers))
        self.final_layernorm = VisionRMSNorm(config.vt_hidden_size, config.norm_backend)

    def forward(self, hidden_states: torch.Tensor, grids: Grid) -> torch.Tensor:
        """Run the full encoder with one shared position table per invocation.

        Args:
            hidden_states: Concatenated patch embeddings [patch_count, hidden].
            grids: Validated media grids partitioning hidden_states.

        Returns:
            Final normalized patch embeddings with the input shape.
        """
        angles = spatial_rope_angles(grids, self.head_dim, hidden_states.device)
        lengths = tuple(t * h * w for t, h, w in grids)
        for block in self.blocks:
            hidden_states = block(hidden_states, lengths, angles)
        return self.final_layernorm(hidden_states)


def temporal_spatial_merge(hidden_states: torch.Tensor, grids: Grid) -> list[torch.Tensor]:
    """Average frames and preserve each 2x2 patch group for projection.

    Args:
        hidden_states: Concatenated encoded patches [patch_count, hidden].
        grids: Validated grids whose spatial dimensions are divisible by two.

    Returns:
        One [height * width // 4, 4, hidden] tensor per media item, in input order.
    """
    outputs = []
    offset = 0
    hidden_size = hidden_states.shape[-1]
    for frames, height, width in grids:
        count = frames * height * width
        segment = hidden_states[offset:offset + count].reshape(frames, height // 2, 2, width // 2, 2, hidden_size)
        outputs.append(segment.permute(0, 1, 3, 2, 4, 5).mean(0).reshape(height * width // 4, 4, hidden_size))
        offset += count
    return outputs


class MoonViT3dPretrainedModel(nn.Module):
    """K3's complete vision tower; the name preserves checkpoint/tool discovery."""

    def __init__(self, config: Any) -> None:
        """Validate the image encoder configuration and build all stages."""
        super().__init__()
        if tuple(config.merge_kernel_size) != (2, 2) or config.merge_type != "sd2_tpool":
            raise ValueError("K3 vision currently requires the released 2x2 spatial/temporal-pool merge")
        if config.norm_type != "rmsnorm" or config.mlp_type != "mlp2" or config.pos_emb_type != "divided_fixed":
            raise ValueError("K3 vision requires RMSNorm, mlp2 and divided_fixed positions")
        if config.activation_func != "gelu_pytorch_tanh":
            raise ValueError("K3 vision requires gelu_pytorch_tanh activation")
        self.patch_size = config.patch_size
        self.max_frames = config.init_pos_emb_time
        self.patch_embed = MoonVisionPatchEmbed(config)
        self.encoder = MoonViTEncoder(config)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        """Encode media patches into pooled spatial groups for the projector.

        Args:
            pixel_values: Flattened patches [sum(grid products), 3, patch_size, patch_size].
            grid_thws: Integer [media, 3] tensor describing the patch sequence.

        Returns:
            One [height * width // 4, 4, vision_hidden_size] tensor per media item.

        Raises:
            ValueError: Media is empty, metadata is invalid or patch shapes disagree.
        """
        grids = validate_grid(grid_thws, max_frames=self.max_frames)
        if not grids:
            raise ValueError("vision forward requires media; the top-level model owns empty-media execution")
        expected = sum(t * h * w for t, h, w in grids)
        if pixel_values.shape != (expected, 3, self.patch_size, self.patch_size):
            raise ValueError("pixel_values shape must match sum(grid_thws.prod(-1)), 3 and patch_size")
        hidden_states = self.patch_embed(pixel_values, grids)
        hidden_states = self.encoder(hidden_states, grids)
        return temporal_spatial_merge(hidden_states, grids)


class PatchMergerMLPV2(nn.Module):
    """Project pooled spatial groups with the released bias-free FQN layout."""

    def __init__(self, config: Any) -> None:
        """Build the pooled patch projector and its output normalization."""
        super().__init__()
        if config.mm_projector_type != "patchmergerv2" or config.projector_hidden_act != "gelu":
            raise ValueError("K3 training supports the released patchmergerv2 GELU projector")
        if config.mm_hidden_size != config.vt_hidden_size:
            raise ValueError("mm_hidden_size must match vt_hidden_size")
        hidden_size = 4 * config.vt_hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False), nn.GELU(),
            nn.Linear(hidden_size, config.text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(config.text_hidden_size, eps=config.projector_ln_eps)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Project all media in one invocation, including when wrapped by FSDP.

        Args:
            features: Nonempty list of pooled groups [media_tokens, 4, vision_hidden].

        Returns:
            One [media_tokens, text_hidden_size] tensor per input media item.
        """
        lengths = tuple(item.shape[0] for item in features)
        merged = torch.cat([item.flatten(1) for item in features]).to(self.proj[0].weight.dtype)
        projected = self.post_norm(self.proj(merged))
        return projected.split(lengths)
