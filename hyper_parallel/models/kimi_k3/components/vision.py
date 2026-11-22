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
"""Complete K3 temporal-pooling vision family with separate QKV width and complex axial positions.

Vision equations and state names are adapted from the pinned Moonshot K3 source.
The family owns complete vision computation; model-specific input/media fusion
remains in the source model architecture.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import math
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import PytorchGELUTanh

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute clip-local dense attention using host sequence boundaries."""
    del k_cu_seqlens, kwargs
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[..., q_cu_seqlens[i - 1] : q_cu_seqlens[i], q_cu_seqlens[i - 1] : q_cu_seqlens[i]] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight = attn_weight.masked_fill(~attention_mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def _apply_rope_input_validation(x: torch.Tensor, freqs_cis: torch.Tensor) -> None:
    if (x.ndim != freqs_cis.ndim + 1 or x.shape[:-2] != freqs_cis.shape[:-1]
            or x.shape[-1] != 2 * freqs_cis.shape[-1] or freqs_cis.dtype != torch.complex64):
        raise ValueError("K3 complex rotary frequencies do not match the query/key dimensions")


def get_rope_shape(org: torch.Tensor, interpolation_mode: str, shape: tuple[int, int]) -> torch.Tensor:
    """Interpolate the learned spatial grid without changing its source flattening order."""
    return (
        F.interpolate(org.permute((2, 0, 1)).unsqueeze(0), size=shape, mode=interpolation_mode)
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: Complex64 frequencies shaped (..., head_dim/2), one per spatial position.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return (xq_out.type_as(xq), xk_out.type_as(xk))


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2:
        raise ValueError("Temporal sinusoidal embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim: int, t_size: int, cls_token: bool = False) -> np.ndarray:
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):

    def __init__(self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str = "bicubic") -> None:
        """Allocate learned spatial positions and nonpersistent sinusoidal temporal state."""
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames)).float().unsqueeze(1),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the learned spatial position grid."""
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """Add interpolated spatial positions and the source temporal offsets for each clip."""
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            if t > self.num_frames:
                raise ValueError("Media clip exceeds the configured temporal position grid")
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(self.weight, interpolation_mode=self.interpolation_mode, shape=(h, w))
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))
        out = x + torch.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(nn.Module):

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        patch_embed_proj_bias: bool = True,
        pos_emb_interpolation_mode: str = "bicubic",
    ) -> None:
        """Allocate Conv2D patch projection and divided spatial/temporal positions."""
        super().__init__()
        if not isinstance(patch_size, int | Sequence):
            raise ValueError("patch_size must be an integer or a height/width sequence")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError("patch_size must have exactly two spatial dimensions")
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=patch_embed_proj_bias)
        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. Each attention block applies those frequencies to query and key before attention.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim: Per-head feature width, which must be divisible by four.
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000) -> None:
        """Define fixed complex axial grid bounds and attention-head width."""
        super().__init__()
        self.dim = dim
        if self.dim % 4:
            raise ValueError("Complex axial rotary dimension must be divisible by four")
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self) -> str:
        """Describe the fixed axial rotary grid."""
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: self.dim // 4].float().to(device)
        freqs = 1.0 / self.theta_base ** (dim_range / self.dim)
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer("freqs_cis", self._precompute_freqs_cis(device), persistent=False)
            register_rebuildable_buffer(
                self, "freqs_cis", factory=lambda ctx: self._precompute_freqs_cis(torch.device("cpu"))
            )
        shapes = grid_thws.tolist()
        if not all(1 <= h <= self.max_height and 1 <= w <= self.max_width for _, h, w in shapes):
            raise ValueError("Media spatial grid exceeds the configured axial rotary bounds")
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes], dim=0
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation: Callable, bias: bool = True) -> None:
        """Build two vision projections with the source activation and initialization."""
        super().__init__()
        if len(dims) != 3:
            raise ValueError("Vision MLP requires input, intermediate and output dimensions")
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features through the intermediate activation."""
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: int | None = None,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp2",
        *,
        attn_implementation: str = "flash_attention_2",
        activation: Callable = F.gelu,
        attn_bias: bool = False,
        linear_bias: bool = True,
        use_deterministic_attn: bool = False,
    ) -> None:
        """Allocate a pre-normalized vision block with independent QKV projection width."""
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn
        if norm_type == "layernorm":
            self.norm0 = nn.LayerNorm(hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.norm0 = nn.RMSNorm(hidden_dim)
            self.norm1 = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")
        if mlp_type == "mlp2":
            self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation, bias=linear_bias)
        else:
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")
        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)
        qkv_shape = xqkv.size()[:-1] + (3, self.num_heads, self.hidden_size_per_attention_head)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        xq, xk = apply_rope(xq, xk, rope_freqs_cis)
        attn_func = {"eager": eager_attention, "sdpa": _sdpa_attention}[self.attn_implementation]
        attn_out = attn_func(
            xq,
            xk,
            xv,
            q_cu_seqlens=cu_seqlens,
            k_cu_seqlens=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            deterministic=self.use_deterministic_attn,
        )
        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply clip-local attention and MLP residual updates."""
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MoonViT3dEncoder(nn.Module):

    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict, use_deterministic_attn: bool = False) -> None:
        """Construct the shared rotary grid, residual blocks and final normalization."""
        super().__init__()
        self.use_deterministic_attn = use_deterministic_attn
        qkv_hidden_size = (
            block_cfg["hidden_dim"] if block_cfg.get("qkv_hidden_size") is None else block_cfg["qkv_hidden_size"]
        )
        self.rope_2d = Rope2DPosEmbRepeated(qkv_hidden_size // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(**block_cfg, use_deterministic_attn=self.use_deterministic_attn)
                for _ in range(num_layers)
            ]
        )
        norm_type = block_cfg.get("norm_type", "layernorm")
        if norm_type == "layernorm":
            self.final_layernorm = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.final_layernorm = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """Run vision blocks with host sequence boundaries and device rotary frequencies."""
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=hidden_states.device)
        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        max_seqlen = int(lengths.max())
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor, grid_thws: torch.Tensor, merge_kernel_size: tuple[int, int] = (2, 2)
) -> list[torch.Tensor]:
    """Average frames and regroup spatial patches in the released feature ordering."""
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = (h // kernel_height, w // kernel_width)
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w
    return outputs


class MoonViT3dPretrainedModel(nn.Module):
    config_class = None
    model_type = "moonvit3d"
    _no_split_modules = ["MoonViTEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: Any) -> None:
        """Construct the complete temporal vision graph and its training initialization policy."""
        super().__init__()
        self.config = config
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type
        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            patch_embed_proj_bias=getattr(config, "patch_embed_proj_bias", True),
            pos_emb_interpolation_mode=getattr(config, "pos_emb_interpolation_mode", "bicubic"),
        )
        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "qkv_hidden_size": getattr(config, "qkv_hidden_size", None),
                "mlp_dim": config.intermediate_size,
                "norm_type": getattr(config, "norm_type", "layernorm"),
                "mlp_type": getattr(config, "mlp_type", "mlp2"),
                "activation": PytorchGELUTanh(),
                "attn_bias": getattr(config, "attn_bias", True),
                "linear_bias": getattr(config, "linear_bias", True),
                "attn_implementation": config._attn_implementation,
            },
            use_deterministic_attn=getattr(self, "use_deterministic_attn", False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all learned vision state, including bare positions and normalization scales.

        The released inference source leaves the learned position grid outside
        its HF initializer. The training family declares its normal(0, 1)
        position policy explicitly so meta allocation never exposes empty storage.
        """
        std = float(getattr(self.config, "initializer_range", 0.02))
        for child in self.modules():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(child.weight, std=std)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, (nn.LayerNorm, nn.RMSNorm, Learnable2DInterpPosEmbDivided_fixed)):
                child.reset_parameters()

    def make_transforms(self) -> list:
        """Keep complete source vision keys and logical matrices unchanged."""
        return []

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        host_grid_thws: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws: Temporal, height and width. Keep these preprocessing
                metadata on CPU, or supply their CPU counterpart separately.
            host_grid_thws: Optional CPU geometry when the model also needs a
                device copy. No implicit device-to-host copy is performed.

        Returns:
            One grouped patch tensor per media clip.
        """
        grid_thws = grid_thws if host_grid_thws is None else host_grid_thws
        if grid_thws.device.type != "cpu":
            raise ValueError("K3 vision requires CPU grid_thws or precomputed host_grid_thws")
        if grid_thws.ndim != 2 or grid_thws.shape[1] != 3 or grid_thws.shape[0] == 0:
            raise ValueError("K3 vision geometry must have nonempty shape [clips, 3]")
        if grid_thws.dtype not in (torch.int32, torch.int64) or (grid_thws <= 0).any():
            raise ValueError("K3 vision geometry requires positive integer time/height/width")
        if int(grid_thws.prod(-1).sum()) != pixel_values.shape[0]:
            raise ValueError("K3 vision patch count differs from the declared media geometry")
        if any(int(grid_thws[:, axis].remainder(size).count_nonzero()) for axis, size in
               zip((1, 2), self.merge_kernel_size)):
            raise ValueError("K3 vision spatial grids must be divisible by the patch merge kernel")
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == "sd2_tpool":
            hidden_states = tpool_patch_merger(hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Not support {self.merge_type}")
        return hidden_states


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_cu_seqlens: torch.Tensor,
    k_cu_seqlens: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Run independent media clips through native Torch SDPA without a global quadratic mask."""
    del kwargs
    if q_cu_seqlens.device.type != "cpu" or k_cu_seqlens.device.type != "cpu":
        raise ValueError("K3 SDPA vision requires host attention boundaries from media preprocessing")
    ranges = q_cu_seqlens.tolist()
    pieces = []
    for begin, end in zip(ranges[:-1], ranges[1:]):
        q, k, v = (tensor[begin:end].transpose(0, 1).unsqueeze(0) for tensor in (query, key, value))
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # pylint: disable=not-callable
        pieces.append(output.squeeze(0).transpose(0, 1).flatten(1))
    return torch.cat(pieces)


_SOURCE_CLASSES = {
    item.__name__: item
    for item in (
        Learnable2DInterpPosEmbDivided_fixed,
        MoonVision3dPatchEmbed,
        Rope2DPosEmbRepeated,
        MLP2,
        MoonViTEncoderLayer,
        MoonViT3dEncoder,
        MoonViT3dPretrainedModel,
    )
}


def _adopt(source: nn.Module, attention_backend: str) -> nn.Module:
    """Adopt a complete supported state tree while replacing every nonprimitive vision forward."""
    if isinstance(source, nn.ModuleList):
        return nn.ModuleList([_adopt(child, attention_backend) for child in source])
    if isinstance(source, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.RMSNorm)):
        return source
    if isinstance(source, PytorchGELUTanh):
        return nn.GELU(approximate="tanh")
    implementation = _SOURCE_CLASSES.get(type(source).__name__)
    if implementation is None:
        raise ValueError(f"Unsupported K3 vision source child {type(source).__name__}")
    result = implementation.__new__(implementation)
    nn.Module.__init__(result)
    for name, value in vars(source).items():
        if not name.startswith("_") and name != "training":
            setattr(result, name, value)
    for name, parameter in source._parameters.items():
        result.register_parameter(name, parameter)
    for name, buffer in source._buffers.items():
        result.register_buffer(name, buffer, persistent=name not in source._non_persistent_buffers_set)
    for name, child in source._modules.items():
        result.add_module(name, None if child is None else _adopt(child, attention_backend))
    result.training = source.training
    if isinstance(result, MoonViTEncoderLayer):
        result.attn_implementation = attention_backend
    if isinstance(result, Learnable2DInterpPosEmbDivided_fixed):
        register_rebuildable_buffer(
            result,
            "time_weight",
            factory=lambda ctx: torch.from_numpy(get_1d_sincos_pos_embed(result.dim, result.num_frames))
            .float()
            .unsqueeze(1)
            .to(result.time_weight.dtype),
        )
    if isinstance(result, Rope2DPosEmbRepeated) and hasattr(result, "freqs_cis"):
        register_rebuildable_buffer(
            result, "freqs_cis", factory=lambda ctx: result._precompute_freqs_cis(torch.device("cpu"))
        )
    return result


def _build_vision(
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    attention_backend: str = "eager",
) -> nn.Module:
    """Build complete dimensions or retain source primitives and derived-buffer identities."""
    del module_fqn, context
    if (config is None) == (module is None):
        raise ValueError("provide exactly one of config or module")
    if attention_backend not in ("eager", "sdpa"):
        raise ValueError("K3 owned vision supports eager or sdpa attention")
    if module is None:
        config = deepcopy(config)
        config._attn_implementation = attention_backend
        result = MoonViT3dPretrainedModel(config)
        for child in result.modules():
            if isinstance(child, Learnable2DInterpPosEmbDivided_fixed):
                register_rebuildable_buffer(
                    child,
                    "time_weight",
                    factory=lambda ctx, pos=child: torch.from_numpy(get_1d_sincos_pos_embed(pos.dim, pos.num_frames))
                    .float()
                    .unsqueeze(1)
                    .to(pos.time_weight.dtype),
                )
        return result
    return _adopt(module, attention_backend)


def _vision_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own complete visual parameters independently of language projection naming templates."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(
        params={name: dict(placements) for name, _ in module.named_parameters()},
        in_src={},
        in_dst={},
        out_src=None,
        out_dst=None,
        is_boundary=True,
    )


# Full source keys are retained; primitive parameter resets and derived buffers stay family-owned.


@module_replacement
def TemporalVisionTransformer(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
    attention_backend: str = "eager",
) -> nn.Module:
    """Build the complete K3 temporal-pooling vision encoder without a source forward callback."""
    return build_family(
        TemporalVisionTransformer,
        config=config,
        module=module,
        module_fqn=module_fqn,
        context=context,
        mesh=mesh,
        attention_backend=attention_backend,
    )


register_module_family(
    ModuleFamily(
        name="kimi_k3_temporal_vision",
        baseline=TemporalVisionTransformer,
        variants={"local": "hyper_parallel.models.kimi_k3.components.vision._build_vision"},
        default_variant="local",
        sharding=_vision_sharding,
    )
)
