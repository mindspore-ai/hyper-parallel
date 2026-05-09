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
"""Grouped-Query Attention — shared building block.

Uses standard layer naming ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``
so a model-level ``_tp_plan`` with ``*.q_proj``-style patterns plugs in
without per-attention overrides.
"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.rmsnorm import RMSNorm
from hyper_parallel.models.modules.rope import RotaryEmbedding, apply_rotary_pos_emb


def _expand_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, num_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return x.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class GroupQueryAttention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA) support.

    Standard naming convention: ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``,
    matched by typical ``*.q_proj`` / ``*.o_proj`` TP plan patterns. Hosted at
    ``self_attn`` so a model-level ``_cp_modules = ["*.self_attn"]`` finds it.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` with ``is_causal=True``.

    Optional ``attn_output_gate=True`` enables Qwen3.5-style gated attention:
    ``q_proj`` emits ``2 * num_heads * head_dim`` features, the trailing half
    is split off as a per-head-channel gate, and the post-attention output
    is multiplied element-wise by ``sigmoid(gate)`` before ``o_proj``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_bias: bool = True,
        out_bias: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        attn_output_gate: bool = False,
        norm_cls: type = RMSNorm,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # ``head_dim`` may differ from ``hidden_size // num_heads`` (e.g.
        # Qwen3-VL-MoE projects to ``num_heads * head_dim != hidden_size``).
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attn_output_gate = attn_output_gate

        # When attn_output_gate is on, q_proj projects to (Q || gate) — both
        # halves are (num_heads * head_dim).
        # where ``self.q_proj`` has out_features = num_heads * head_dim * 2.
        q_out_dim = self.num_heads * self.head_dim * (2 if attn_output_gate else 1)
        self.q_proj = nn.Linear(hidden_size, q_out_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=out_bias)

        # Optional Q/K RMSNorm before RoPE; ``norm_cls`` lets the model
        # plug in a vendor-specific variant.
        if qk_norm:
            self.q_norm = norm_cls(self.head_dim, eps=rms_norm_eps)
            self.k_norm = norm_cls(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if rope is None:
            rope = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)
        self.rotary_emb = rope

    def forward(self, hidden_states: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass."""
        bsz, seq_len, _ = hidden_states.shape

        # Apply q_norm / k_norm before transposing so RMSNorm reduces over
        # contiguous (B, S, H, D); the post-transpose layout dispatches a
        # different NPU mean kernel and yields ULP-different gradients.
        if self.attn_output_gate:
            # ``q_proj`` packs (Q, gate) along ``head_dim * 2``; reshape gate
            # for the elementwise post-attention multiply.
            q_out = self.q_proj(hidden_states).view(
                bsz, seq_len, self.num_heads, self.head_dim * 2,
            )
            q_raw, gate = torch.chunk(q_out, 2, dim=-1)
            gate = gate.reshape(bsz, seq_len, -1)
            q = self.q_norm(q_raw).transpose(1, 2)  # norm on (B,S,H,D), then -> (B,H,S,D)
        else:
            q_raw = self.q_proj(hidden_states).view(
                bsz, seq_len, self.num_heads, self.head_dim,
            )
            q = self.q_norm(q_raw).transpose(1, 2)
            gate = None
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

        # Manual matmul + fp32 softmax — mirrors ``eager_attention_forward``.
        attn_mask = kwargs.get("attention_mask")
        scale = self.head_dim ** -0.5
        kv_groups = q.shape[1] // k.shape[1]
        if kv_groups > 1:
            k = _expand_kv_heads(k, kv_groups)
            v = _expand_kv_heads(v, kv_groups)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
        if attn_mask is not None and attn_mask.ndim == 4:
            if attn_mask.shape[1] == 1 and q.shape[1] > 1:
                attn_mask = attn_mask.expand(attn_mask.shape[0], q.shape[1], -1, -1)
            attn_weights = attn_weights + attn_mask
        else:
            causal = torch.triu(
                torch.full(
                    (seq_len, seq_len), float('-inf'), device=q.device, dtype=q.dtype,
                ),
                diagonal=1,
            )
            attn_weights = attn_weights + causal
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)
