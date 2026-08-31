# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Pure-PyTorch mock modules for GraphTrainer TP/SP demo.

These modules are self-contained implementations of RotaryEmbedding,
RMSNorm, GroupQueryAttention and SwiGLUMLP.

Submodule names (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``,
``gate_proj``, ``up_proj``, ``down_proj``) match the ShardingPlanner
conventions so ``ShardingPlanner.plan()`` derives correct TP placements
without any plan_overrides.

THESE ARE FOR DEMONSTRATION ONLY. NOT to be used in real trainings.
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig


class RotaryEmbedding(nn.Module):
    """Standard rotary position embedding (Transformers-compatible).

    Computes ``inv_freq = theta^(-2i/dim)`` and caches cos/sin for the
    requested sequence length.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin for max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device=None):
        """Return (cos, sin) for the first ``seq_len`` positions."""
        if seq_len > self.cos_cached.shape[0]:
            # Extend cache on the fly
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(device)
            sin = emb.sin().to(device)
            return cos, sin
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension."""
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # cos/sin: (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root mean square normalization (Transformers-compatible)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class GroupQueryAttention(nn.Module):
    """GQA with rotary embedding inside forward (SP-safe design).

    Submodule names (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``) match
    the ShardingPlanner conventions.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=getattr(config, "rope_theta", 10000.0),
        )

    def _expand_kv_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads.

        Input shape:  (batch, seq_len, num_kv_heads, head_dim)
        Output shape: (batch, seq_len, num_heads, head_dim)
        """
        batch, seq_len, num_kv, head_dim = tensor.shape
        tensor = tensor.unsqueeze(3).expand(
            batch, seq_len, num_kv, self.num_kv_groups, head_dim
        )
        return tensor.reshape(batch, seq_len, num_kv * self.num_kv_groups, head_dim)

    # pylint: disable=W0613
    def forward(self, hidden_states: torch.Tensor, position_ids=None):
        """Forward: QKV proj -> RoPE (inside forward) -> attention -> output."""
        batch, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # RoPE computed inside attention (after SP boundary all-gather)
        cos, sin = self.rotary_emb(seq_len, device=hidden_states.device)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Expand KV heads to match query heads
        key = self._expand_kv_heads(key)
        value = self._expand_kv_heads(value)

        # (batch, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with separate gate/up/down projections.

    Submodule names (``gate_proj``, ``up_proj``, ``down_proj``) match
    the ShardingPlanner conventions.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
