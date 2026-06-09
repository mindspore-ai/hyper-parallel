# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Gated DeltaNet — linear attention block with delta-rule update.

Used by Qwen3.5 / Qwen3-Next as the dominant (75 %) layer type. Pure-torch chunked
reference (no flash-linear-attention dependency).

Parameter layout (mirrors the upstream checkpoint state-dict keys):

    in_proj_qkv.weight   (key_dim*2 + value_dim, hidden_size)
    in_proj_z.weight     (value_dim, hidden_size)
    in_proj_b.weight     (num_v_heads, hidden_size)
    in_proj_a.weight     (num_v_heads, hidden_size)
    conv1d.weight        (key_dim*2 + value_dim, 1, conv_kernel_size)
    dt_bias              (num_v_heads,)
    A_log                (num_v_heads,)
    norm.weight          (head_v_dim,)            ← RMSNormGated
    out_proj.weight      (hidden_size, value_dim)

This module is for **training** (no KV-cache, no chunk recurrence reuse).
For inference with cache, use a kernel-optimised path or the recurrent
variant from ``transformers.models.qwen3_next``.
"""
# pylint: disable=C0103  # SSM/state-space convention: A_log, A

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.rmsnorm import RMSNormGated


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along ``dim``."""
    # ``(x * x).sum`` (MulBackward) instead of ``x.pow(2).sum`` (PowBackward):
    # equal math, NPU yields ULP-different gradients across the two ops.
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Chunked gated delta-rule linear attention (pure torch reference).

    fp32 internal compute, output cast back to input dtype.

    Shapes
    ------
    query / key:  ``(B, S, num_k_heads, head_k_dim)``
    value:        ``(B, S, num_v_heads, head_v_dim)`` (here num_k_heads == num_v_heads)
    g:            ``(B, S, num_v_heads)``
    beta:         ``(B, S, num_v_heads)``
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    # (B, S, H, D) → (B, H, S, D), fp32
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # (B, H, S, D) → (B, H, n_chunks, chunk_size, D)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention block (Qwen3.5 / Qwen3-Next style).

    Args:

    - ``hidden_size``: model hidden dim
    - ``num_v_heads`` / ``num_k_heads``: linear attention heads (typically
      ``num_v_heads = 2 * num_k_heads`` so Q/K are repeated)
    - ``head_k_dim`` / ``head_v_dim``: per-head dim
    - ``conv_kernel_size``: depthwise causal 1-D conv kernel (typ. 4)
    - ``rms_norm_eps``: epsilon for the gated RMSNorm
    """

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int = 32,
        num_k_heads: int = 16,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.kv_groups = num_v_heads // num_k_heads

        # Submodule order is fixed so ``model.parameters()`` ordering and
        # the resulting fp32 ``clip_grad_norm_`` reduction are bit-stable.
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )

        # discretisation parameters
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        A = torch.empty(num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # gated RMSNorm + output projection
        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # in projections (Q, K, V concatenated → conv → split)
        self.in_proj_qkv = nn.Linear(
            hidden_size, self.key_dim * 2 + self.value_dim, bias=False,
        )
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        bsz, seq_len, _ = hidden_states.shape

        # 1. Project to mixed QKV; transpose for Conv1d (B, C, S).
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)

        # 2. z, b, a paths.
        z = self.in_proj_z(hidden_states).reshape(
            bsz, seq_len, self.num_v_heads, self.head_v_dim,
        )
        b = self.in_proj_b(hidden_states)  # (B, S, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (B, S, num_v_heads)

        # 3. Conv1d + causal trim + SiLU.
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, S, conv_dim)

        # 4. Split QKV.
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        # 5. beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # 6. Repeat Q/K to match V heads (linear-attn GQA).
        if self.kv_groups > 1:
            query = query.repeat_interleave(self.kv_groups, dim=2)
            key = key.repeat_interleave(self.kv_groups, dim=2)

        # 7. Chunked gated delta rule.
        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # 8. RMSNormGated with z as the SiLU gate.
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(bsz, seq_len, self.value_dim)

        # 9. Output projection.
        return self.out_proj(core_attn_out)
