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
"""Qwen3-VL-MoE conditional generation model.

Mirrors the upstream ``Qwen3VLMoeForConditionalGeneration`` text + vision
architecture (vision tower, DeepStack visual injection, MoE text decoder).
"""
from __future__ import annotations

__all__ = [
    "Qwen3VLMoeTextConfig",
    "Qwen3VLMoeVisionConfig",
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeTextDecoder",
]

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.attention import GroupQueryAttention
from hyper_parallel.models.modules.feed_forward import SwiGLUMLP
from hyper_parallel.models.modules.rmsnorm import RMSNorm
from hyper_parallel.models.modules.rope import MultiModalRotaryEmbedding, apply_rotary_pos_emb


def _use_v1_kernels() -> bool:
    """True when ``HYPER_USE_V1_KERNELS=1`` and ``torch_npu`` is importable.

    When set, ``Qwen3VLMoeTextSparseMoE``
    and ``Qwen3VLMoeTextExperts`` dispatch to ``torch_npu.npu_moe_token_permute`` /
    ``npu_grouped_matmul`` / ``npu_swiglu`` / ``npu_moe_token_unpermute`` instead
    of the eager per-expert loop.
    """
    if os.environ.get("HYPER_USE_V1_KERNELS", "0") != "1":
        return False
    try:
        import torch_npu  # pylint: disable=C0415,W0611
        return True
    except ImportError:
        return False


class _GmmFunction(torch.autograd.Function):
    """Custom autograd op around ``torch_npu.npu_grouped_matmul``.

    Inputs:
      - ``x``: (T_perm, K) — token-permuted input.
      - ``weight``: (E, K, N) — per-expert weight stack.
      - ``group_list``: (E,) int64 — token count per expert (cumulative semantics
        controlled via ``group_list_type=1`` = absolute counts).

    Forward returns (T_perm, N).
    """

    # pylint: disable=W0223  # intentional — only forward/backward needed
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                group_list: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
        # pylint: disable=C0415
        """Forward pass."""
        import torch_npu
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        out = torch_npu.npu_grouped_matmul(
            [x], [weight], bias=None, group_list=group_list,
            split_item=2, group_type=0, group_list_type=1,
        )[0]
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=C0415
        """Backward pass."""
        import torch_npu
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        weight_t = torch.transpose(weight, 1, 2)
        grad_x = torch_npu.npu_grouped_matmul(
            [grad_output], [weight_t], bias=None, group_list=group_list,
            split_item=2, group_type=0, group_list_type=1,
        )[0]
        grad_w = torch_npu.npu_grouped_matmul(
            [x.T], [grad_output], bias=None, group_list=group_list,
            split_item=3, group_type=2, group_list_type=1,
        )[0]
        return grad_x, grad_w, None


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _activation(name: str):
    if name in ("silu", "swish"):
        return F.silu
    if name == "gelu_pytorch_tanh":
        return _gelu_pytorch_tanh
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class Qwen3VLMoeTextConfig:
    """Text config fields used by Qwen3-VL-MoE 30B-A3B."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 4
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    rope_theta: float = 5_000_000.0
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])

    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768

    # ``"flash_attention_2"`` routes through ``torch_npu.npu_fusion_attention``;
    # ``"eager"`` / ``"sdpa"`` use the shared SDPA path.
    _attn_implementation: str = "eager"


@dataclass
class Qwen3VLMoeVisionConfig:
    """Vision config fields used by Qwen3-VL-MoE."""

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    # ``"flash_attention_2"`` uses ``torch_npu.npu_fusion_attention``;
    # ``"eager"`` chunk-splits q/k/v by cu_seqlens and runs eager attention.
    _attn_implementation: str = "eager"


@dataclass
class Qwen3VLMoeConfig:
    """Composite config for native Qwen3-VL-MoE conditional generation."""

    text_config: Qwen3VLMoeTextConfig = field(default_factory=Qwen3VLMoeTextConfig)
    vision_config: Qwen3VLMoeVisionConfig = field(default_factory=Qwen3VLMoeVisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vl: bool = True

    def __getattr__(self, name):
        """Expose text fields for trainer parallel validation."""
        if hasattr(self.text_config, name):
            return getattr(self.text_config, name)
        raise AttributeError(name)


class Qwen3VLMoeVisionRotaryEmbedding(nn.Module):
    """2D rotary frequencies for Qwen3-VL vision attention."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Force CPU init: NPU's fp32 ``pow`` rounds 1 ULP differently from
        # CPU's libm, and the cascading norm-diff at merger output is large.
        cpu = torch.device("cpu")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=cpu) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_inv_freq(self) -> None:  # pylint: disable=W0613
        # Recompute on CPU after meta-init: ``to_empty`` wipes the
        # CPU-computed buffer, and recomputing on NPU drifts by 1 ULP
        # against the CPU path (see ``__init__``).
        """Reset inv freq."""
        cpu = torch.device("cpu")
        cpu_inv_freq = 1.0 / (
            self.theta ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=cpu)
                / self.dim
            )
        )
        self.inv_freq.copy_(cpu_inv_freq.to(self.inv_freq.device))

    def forward(self, seqlen: int) -> torch.Tensor:  # pylint: disable=W0613
        """Compute rotary position embedding frequencies for the given sequence length."""
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype,
        )
        return torch.outer(seq, self.inv_freq)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to vision Q/K (fp32 internal)."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Replicate KV heads ``n_rep`` times for grouped-query attention."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim,
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _eager_attention_forward(*args, **kwargs):
    """Eager attention forward used by the vision and text attention paths.

    Delegates to ``transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.
    eager_attention_forward``. Reusing that function (rather than a local
    copy) keeps the Python call-stack identity stable; the NPU kernel-cache
    key is sensitive to the calling function frame, and a local copy was
    observed to produce a 1-ULP bf16 divergence at the first vision
    attention block.
    """
    # pylint: disable=C0415
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        eager_attention_forward as _eager,
    )
    return _eager(*args, **kwargs)


class Qwen3VLMoeVisionAttention(nn.Module):
    """Vision self-attention for Qwen3-VL-MoE."""

    def __init__(self, config: Qwen3VLMoeVisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        # pylint: disable=W0613  # interface conformance
        # Op order, variable names, and arg signature are pinned because the
        # NPU CANN kernel cache is keyed off the Python call-site shape.
        """Forward pass."""
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin,
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # fa2 path uses full-batch query + ``cu_seq_lens_q/k``; other
        # implementations chunk-split q/k/v by ``cu_seqlens``.
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if attn_impl == "flash_attention_2":
            from transformers.modeling_flash_attention_utils import _flash_attention_forward  # pylint: disable=C0415
            # Keep ``max_seqlen`` as a tensor: ``.item()`` forces a CPU-NPU
            # sync that perturbs the CANN kernel cache.
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            # ``_flash_attention_forward`` consumes (B, S, H, D).
            q_for_fa = query_states.transpose(1, 2)
            k_for_fa = key_states.transpose(1, 2)
            v_for_fa = value_states.transpose(1, 2)
            attn_output = _flash_attention_forward(
                q_for_fa, k_for_fa, v_for_fa,
                attention_mask=None,
                query_length=q_for_fa.shape[1],
                is_causal=False,
                dropout=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                attn_implementation="flash_attention_2",
            )
        else:
            # Eager path: process each variable-length chunk separately.
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                _eager_attention_forward(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLMoeVisionMLP(nn.Module):
    """Vision MLP matching HF Qwen3VLMoeVisionMLP names."""

    def __init__(self, config: Qwen3VLMoeVisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True,
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True,
        )
        self.act_fn = _activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
        """Apply two-layer MLP with activation: fc1 -> act -> fc2."""

        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3VLMoeVisionDecoder(nn.Module):
    """One Qwen3-VL vision encoder block."""

    def __init__(self, config: Qwen3VLMoeVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLMoeVisionAttention(config)
        self.mlp = Qwen3VLMoeVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VLMoeVisionPatchEmbed(nn.Module):
    """3D patch embedding used by Qwen3-VL."""

    def __init__(self, config: Qwen3VLMoeVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [
            self.temporal_patch_size, self.patch_size, self.patch_size,
        ]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim,
            kernel_size=kernel_size, stride=kernel_size, bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
        """Forward pass."""
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))

        return hidden_states.view(-1, self.embed_dim)


class Qwen3VLMoeVisionPatchMerger(nn.Module):
    """Patch merger and DeepStack merger."""

    def __init__(
        self,
        config: Qwen3VLMoeVisionConfig,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=1e-6,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=W0613
        """Apply layer norm, two-layer MLP with GELU, and project to output hidden size."""
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)

        x = x.view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


@dataclass
class Qwen3VLMoeVisionOutput:
    """Small output container for native vision forward."""

    last_hidden_state: torch.Tensor

    pooler_output: torch.Tensor | list[torch.Tensor]
    deepstack_features: list[torch.Tensor]


class Qwen3VLMoeVisionModel(nn.Module):
    """Native Qwen3-VL-MoE vision tower."""

    def __init__(self, config: Qwen3VLMoeVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen3VLMoeVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size,
        )
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([
            Qwen3VLMoeVisionDecoder(config) for _ in range(config.depth)
        ])
        self.merger = Qwen3VLMoeVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLMoeVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        ])

    @property
    def dtype(self):
        """Return the dtype of the vision model's patch embedding weights."""
        return self.patch_embed.proj.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        # Compute ``max_hw`` / ``total_tokens`` via tensor reductions —
        # ``grid_thw.tolist()`` lands ``freq_table[pos_ids]`` on a different
        # memory layout and yields a sizable norm-diff at rotary output.
        """Rot pos emb."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size,
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size,
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Fast pos embed interpolate."""
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([
            h * w for h, w in zip(grid_hs, grid_ws)
        ])

        out = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pos_embed)
        return torch.cat(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Qwen3VLMoeVisionOutput:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_idx in self.deepstack_visual_indexes:
                merge_idx = self.deepstack_visual_indexes.index(layer_idx)
                deepstack_features.append(
                    self.deepstack_merger_list[merge_idx](hidden_states)
                )

        return Qwen3VLMoeVisionOutput(
            last_hidden_state=hidden_states,

            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )


class Qwen3VLMoeTextTopKRouter(nn.Module):
    """Router matching HF ``Qwen3VLMoeTextTopKRouter``."""

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor):  # pylint: disable=W0613
        # Routing: bf16 ``F.linear`` → fp32 softmax → ``torch.topk``. The
        # ``topk`` is deterministic under ``use_deterministic_algorithms``.
        """Forward pass."""
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        router_top_value, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1,
        )

        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(hidden_states.dtype)
        return router_logits, router_top_value, router_indices


class Qwen3VLMoeTextExperts(nn.Module):
    """Packed expert weights for Qwen3-VL-MoE text experts.

    Layout is fixed by the upstream checkpoint:
      - ``gate_up_proj`` shape (E, H, 2I), ``down_proj`` shape (E, I, H)
        — no transpose at load time; accessed as ``self.gate_up_proj[expert_idx]``
      - matmul ``current_state @ self.gate_up_proj[expert_idx]``
        (not ``F.linear``; ``@`` and ``F.linear`` can dispatch to different NPU
        kernels with different fp32 reduction order, so the ``@`` form is fixed)
      - dense routing_weights of shape (num_tokens, num_experts), indexed as
        ``routing_weights[token_idx, expert_idx, None]``
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # Stored as (E, H, 2I) / (E, I, H) — no transpose at load time.
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if _use_v1_kernels() and hidden_states.device.type == "npu":
            return self._npu_v1_forward(hidden_states, routing_weights, router_indices)

        # Eager path: ``routing_weights`` dense (num_tokens, num_experts);
        # ``router_indices`` (num_tokens, top_k).
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit[:]:
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx[0]])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            gated_output = up * F.silu(gate)
            out = gated_output @ self.down_proj[expert_idx]
            # ``out[0]`` strips the broadcast-1 leading dim from indexing.
            weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(
                0, token_idx, weighted_output.to(hidden_states.dtype),
            )
        return next_states

    def _npu_v1_forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        """NPU fused MoE expert path (``npu_moe_token_permute`` +
        ``npu_grouped_matmul`` + ``npu_swiglu`` + ``npu_moe_token_unpermute``).

        Expects sparse ``routing_weights`` (num_tokens, top_k); the sparse
        form is supplied by the sparse MoE block when
        ``HYPER_USE_V1_KERNELS=1``.
        """
        # pylint: disable=C0415
        import torch_npu
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
            hidden_states, router_indices.to(torch.int32),
        )
        tokens_per_expert = torch.histc(
            router_indices, bins=self.num_experts, min=0, max=self.num_experts,
        ).to(torch.int64)
        intermediate_hidden_states = _GmmFunction.apply(
            permuted_hidden_states, self.gate_up_proj, tokens_per_expert,
        )
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = _GmmFunction.apply(
            intermediate_activations, self.down_proj, tokens_per_expert,
        )
        next_states = torch_npu.npu_moe_token_unpermute(

            output, row_ids_map, probs=routing_weights,
        )
        return next_states


class Qwen3VLMoeTextSparseMoE(nn.Module):
    """Sparse MoE for the text decoder.

    Forward:
        router_logits = self.gate(hidden_states)              # bf16
        routing_weights = softmax(router_logits, dtype=fp32)
        routing_weights, router_indices = topk(routing_weights, k)
        routing_weights /= routing_weights.sum(-1, kd)
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = zeros_like(router_logits).scatter_(1, indices, w)
        routed_out = self.experts(hidden_states, router_weights, indices)

    Returns ``(routed_out, router_logits)`` as a tuple so the decoder
    callsite can unpack via ``isinstance(out, tuple)``.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = Qwen3VLMoeTextTopKRouter(config)
        self.experts = Qwen3VLMoeTextExperts(config)

    def forward(self, hidden_states: torch.Tensor):  # pylint: disable=W0613
        """Forward pass."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)
        router_logits = F.linear(hidden_states_2d, self.gate.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        if _use_v1_kernels() and hidden_states.device.type == "npu":
            # NPU path: pass sparse (T, top_k) routing_weights — consumed
            # directly by ``npu_moe_token_unpermute(probs=...)``.
            next_states = self.experts(
                hidden_states_2d, routing_weights, router_indices,
            )
            return next_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

        router_weights = torch.zeros_like(router_logits).scatter_(
            1, router_indices, routing_weights,
        )
        next_states = self.experts(

            hidden_states_2d, router_weights, router_indices,
        )
        return next_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


class Qwen3VLMoeTextAttention(GroupQueryAttention):
    """Qwen3-VL-MoE text attention with selectable attention backend.

    Subclasses :class:`GroupQueryAttention` (SDPA path) without modifying the
    shared module; overrides ``forward`` to dispatch to flash_attention_2 when
    ``_attn_implementation == "flash_attention_2"`` so the text path lands on
    the NPU ``torch_npu.npu_fusion_attention`` kernel. Otherwise falls back
    to the inherited SDPA path.
    """

    def __init__(self, attn_implementation: str = "eager", **kwargs):
        super().__init__(**kwargs)
        self._attn_implementation = attn_implementation
        # ``eager_attention_forward`` (delegated to via the eager path) reads
        # ``module.num_key_value_groups``; alias to the local field.
        self.num_key_value_groups = self.num_kv_groups

    def forward(self, hidden_states: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None, **kwargs):  # pylint: disable=W0613
        """Dispatch attention computation based on the configured implementation."""
        if self._attn_implementation == "sdpa":
            return super().forward(hidden_states, position_ids=position_ids, **kwargs)

        # eager / fa2: replicate projection + rotary explicitly so the
        # selected kernel (eager fp32-softmax matmul, or NPU fusion attention)
        # is exercised directly rather than the parent's SDPA path.
        bsz, seq_len, _ = hidden_states.shape
        q_raw = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim,
        )
        q = self.q_norm(q_raw).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(
                bsz, seq_len, self.num_kv_heads, self.head_dim,
            )
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim,
        ).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = kwargs.get("attention_mask")
        scaling = self.head_dim ** -0.5

        if self._attn_implementation == "flash_attention_2":
            # pylint: disable=C0415
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            # fa2 only consumes the 2D padding form + ``is_causal``.
            if attn_mask is not None and attn_mask.ndim == 4:
                attn_mask = None
            attn_output = _flash_attention_forward(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attention_mask=attn_mask,
                query_length=seq_len,
                is_causal=True,
                dropout=0.0,
                softmax_scale=scaling,
                attn_implementation="flash_attention_2",
            )
            attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
            return self.o_proj(attn_output)

        # eager: delegate so the NPU kernel-cache key stays stable.
        # pylint: disable=C0415
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            eager_attention_forward as _eager,
        )
        attn_output, _ = _eager(
            self, q, k, v,
            attention_mask=attn_mask,
            scaling=scaling,

            dropout=0.0,
        )
        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3VLMoeTextDecoder(nn.Module):
    """One Qwen3-VL-MoE text decoder layer."""

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        layer_idx: int,
        rope: MultiModalRotaryEmbedding,
    ):
        super().__init__()
        self.self_attn = Qwen3VLMoeTextAttention(
            attn_implementation=getattr(config, "_attn_implementation", "eager"),
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qkv_bias=config.attention_bias,
            out_bias=config.attention_bias,
            rope=rope,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            norm_cls=RMSNorm,
        )
        if (
            layer_idx not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3VLMoeTextSparseMoE(config)
        else:
            self.mlp = SwiGLUMLP(
                config.hidden_size, config.intermediate_size, bias=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids=position_ids, attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # ``Qwen3VLMoeTextSparseMoE`` returns either a plain tensor or a
        # ``(routed_out, router_logits)`` tuple depending on whether

        # ``output_router_logits`` is enabled — unpack the tuple form here.
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        return residual + hidden_states


class Qwen3VLMoeTextModel(nn.Module):
    """Text decoder used by the multimodal conditional generation wrapper."""

    _cp_modules = ["*.self_attn"]

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = MultiModalRotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3VLMoeTextDecoder(config, i, self.rotary_emb)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _build_causal_mask(
        attention_mask: Optional[torch.Tensor],
        bsz: int,
        seq_len: int,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Build a 4D additive causal+padding mask matching HF create_causal_mask.

        HF ``create_causal_mask`` produces a ``[B, 1, S, S]`` float mask where
        attended positions are 0.0 and blocked positions are ``-inf`` (dtype min).
        Rules:
        (1) causal: upper-triangle is blocked (future tokens cannot be attended)
        (2) padding: columns for pad tokens (``attention_mask==0``) are -inf so
            no query can attend to a pad key.

        Returns a float tensor of shape ``[B, 1, S, S]``.
        """
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_val = torch.finfo(dtype).min

        # [S, S] causal upper-triangle: -inf where blocked
        causal = torch.triu(
            torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device),
            diagonal=1,
        )
        mask_4d = causal.view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len).clone()

        # Apply padding column mask when a 1D/2D attention_mask is provided.
        if attention_mask is not None and attention_mask.ndim <= 2:
            # attention_mask: [B, S] with 1=attend, 0=pad → block pad columns
            pad_cols = attention_mask == 0  # [B, S], True where padding
            mask_4d = mask_4d.masked_fill(pad_cols.view(bsz, 1, 1, seq_len), min_val)

        return mask_4d

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Deepstack process (internal)."""
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = (
            hidden_states[visual_pos_masks, :] + visual_embeds
        )
        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=inputs_embeds.device, dtype=torch.long,
            ).view(1, -1).expand(bsz, -1)
        if position_ids.ndim == 4:
            position_ids = position_ids.squeeze(0)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        # fa2 expects the raw 2D padding mask (it combines causal + padding
        # internally); SDPA paths get a pre-built 4D causal+padding mask.
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if attn_impl == "flash_attention_2":
            layer_attention_mask = attention_mask
        else:
            layer_attention_mask = self._build_causal_mask(
                attention_mask, bsz, seq_len, inputs_embeds,
            )

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=layer_attention_mask,
            )
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):

                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
                )
        return self.norm(hidden_states)


class Qwen3VLMoeModel(nn.Module):
    """Composite Qwen3-VL-MoE model with native visual token injection."""

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLMoeVisionModel(config.vision_config)
        self.language_model = Qwen3VLMoeTextModel(config.text_config)
        self.rope_deltas = None

    @property
    def layers(self):
        """Return the language model decoder layer list."""
        return self.language_model.layers

    def get_input_embeddings(self):
        """Text token embedding accessor (restores upstream's missing method)."""
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        """Set the text token embedding."""
        self.language_model.embed_tokens = value

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get vision position ids."""
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(
            start_position, start_position + llm_grid_w, device=device,
        ).repeat(llm_grid_h * llm_grid_t)
        position_height = torch.arange(
            start_position, start_position + llm_grid_h, device=device,
        ).repeat_interleave(llm_grid_w * llm_grid_t)
        position_temporal = torch.full(
            (image_seq_length,), start_position, device=device, dtype=torch.long,
        )
        position_temporal = position_temporal * time_interval
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rope index."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        rope_deltas = []
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx in range(input_ids.shape[0]):
            token_type = mm_token_type_ids[batch_idx]
            current_input_ids = input_ids[batch_idx]
            if attention_mask is not None:
                mask = attention_mask[batch_idx].bool()
                token_type = token_type[mask]
                current_input_ids = current_input_ids[mask]
            groups = []
            prev = None
            start = 0
            for idx, val in enumerate(token_type.tolist()):
                if prev is None:
                    prev = val
                    start = idx
                elif val != prev:
                    groups.append((prev, start, idx))
                    prev = val
                    start = idx
            if prev is not None:
                groups.append((prev, start, len(token_type)))

            current_pos = 0
            pos_parts = []
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    pos_parts.append(
                        torch.arange(
                            text_len, device=input_ids.device,
                        ).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    # modality_type is 1 (image) or 2 (video) — both static keys
                    # of grid_iters, so .get() is equivalent to indexing here.
                    grid = next(grid_iters.get(modality_type))
                    pos_parts.append(
                        self.get_vision_position_ids(
                            current_pos, grid, 1, spatial_merge_size,
                            device=input_ids.device,
                        )
                    )
                    current_pos += max(grid[1], grid[2]).item() // spatial_merge_size
            llm_positions = torch.cat(pos_parts, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            rope_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        return position_ids, torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Qwen3VLMoeVisionOutput:
        """Get image features."""
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size ** 2)
        ).tolist()
        vision_output.pooler_output = torch.split(
            vision_output.pooler_output, split_sizes,
        )
        return vision_output

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get placeholder mask."""
        special_image_mask = input_ids == self.config.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={int(n_image_tokens)}, features={tuple(image_features.shape)}"
            )
        return special_image_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            deepstack_visual_embeds = image_outputs.deepstack_features

        visual_pos_masks = image_mask[..., 0] if image_mask is not None else None

        if position_ids is None and image_grid_thw is not None:
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                mm_token_type_ids[input_ids == self.config.image_token_id] = 1
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,

            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )


class Qwen3VLMoeForCausalLM(nn.Module):
    """Text-only CausalLM wrapper with HF-compatible state-dict names."""

    _cp_modules = ["*.self_attn"]

    def __init__(self, config: Qwen3VLMoeTextConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = MultiModalRotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3VLMoeTextDecoder(config, i, self.rotary_emb)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device, dtype=torch.long,
            ).view(1, -1).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),

                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


class Qwen3VLMoeForConditionalGeneration(nn.Module):
    """Native multimodal Qwen3-VL-MoE conditional generation model."""

    _cp_modules = ["*.self_attn"]

    def __init__(self, config: Qwen3VLMoeConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3VLMoeModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    @property
    def layers(self):
        """Return the language model decoder layer list."""
        return self.model.language_model.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}
