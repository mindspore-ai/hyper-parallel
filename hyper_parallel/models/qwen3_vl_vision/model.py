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
"""Qwen3-VL vision tower (ViT), shared by Qwen3-VL-MoE and Qwen3.5-VL-MoE.

The patch-embed → rotary → attention → MLP → patch-merge stack plus its config.
Both multimodal composites build ``Qwen3VLMoeVisionModel`` from here so neither
model package depends on the other for the vision encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    eager_attention_forward as _transformers_eager_attention_forward,
)


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
    # ``"sdpa"`` / ``"eager"`` chunk-split q/k/v by cu_seqlens and dispatch
    # through Transformers' attention interface.
    _attn_implementation: str = "eager"


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

    Reusing the imported eager implementation keeps the Python call-stack
    identity stable; the NPU kernel-cache key is sensitive to the calling
    function frame, and a local copy can produce small bf16 divergence in the
    first vision attention block.
    """
    return _transformers_eager_attention_forward(*args, **kwargs)

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
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_impl, _eager_attention_forward,
            )
            # Eager path: process each variable-length chunk separately.
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
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
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

@dataclass
class Qwen3VLMoeVisionOutput:
    """Small output container for native vision forward."""

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | list[torch.Tensor]
    deepstack_features: list[torch.Tensor]


# Registered as a pytree node so FSDP's backward-pre hook (which walks
# ``tree_flatten`` of a wrapped module's outputs) sees the tensors inside.
# Without this, a TRAINABLE vision tower is resharded after forward but never
# re-unsharded before backward — step-1 backward then reads freed storage.
torch.utils._pytree.register_pytree_node(  # pylint: disable=protected-access
    Qwen3VLMoeVisionOutput,
    lambda o: ((o.last_hidden_state, o.pooler_output, o.deepstack_features), None),
    lambda values, ctx: Qwen3VLMoeVisionOutput(*values),
)


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
        hidden_states = hidden_states.reshape(seq_len, -1)
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
