# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Minimal MoE-style decoder model for Expert Parallelism demos.

Combines a standard Llama-style attention stack with
:class:`~hyper_parallel.platform.torch.common.moe.MoE` feed-forward layers,
providing a realistic target for :class:`ExpertParallel` sharding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Shard as PlShard
from hyper_parallel.platform.torch.common.moe import (
    FeedForward,
    MoE,
)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
    """Llama-style RoPE frequencies as complex tensor ``[end, dim // 2]``.

    Args:
        dim: Head dimension.
        end: Maximum sequence length.
        theta: Base frequency for RoPE.

    Returns:
        Complex frequency tensor of shape ``[end, dim // 2]``.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast ``freqs_cis`` of shape ``[seq, head_dim/2]`` to complex ``x``.

    Args:
        freqs_cis: Frequency tensor ``[seq, head_dim // 2]``.
        x: Complex query/key tensor.

    Returns:
        Reshaped frequency tensor ready for element-wise multiply.

    Raises:
        ValueError: If ``freqs_cis`` shape is incompatible with ``x``.
    """
    if hasattr(freqs_cis, "to_local"):
        freqs_cis = freqs_cis.to_local()
    ndim = x.ndim
    if freqs_cis.shape != (x.shape[1], x.shape[-1]):
        raise ValueError(
            f"freqs_cis shape {freqs_cis.shape} incompatible with x shape {x.shape}"
        )
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    with SkipDTensorDispatch():
        return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors (Llama-style).

    Args:
        xq: Query tensor ``[batch, seq, n_heads, head_dim]``.
        xk: Key tensor ``[batch, seq, n_kv_heads, head_dim]``.
        freqs_cis: Precomputed RoPE frequencies.

    Returns:
        Tuple of (rotated query, rotated key) with original dtype.
    """
    if hasattr(freqs_cis, "to_local"):
        freqs_cis = freqs_cis.to_local()
    with SkipDTensorDispatch():
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_q = reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_k = reshape_for_broadcast(freqs_cis, xk_)
        xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv_bshd(x_bshd: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads in BSHD layout (``[B, S, n_kv, D]`` -> ``[B, S, n_heads, D]``).

    Args:
        x_bshd: KV tensor ``[batch, seq, n_kv_heads, head_dim]``.
        n_rep: Number of query-head repeats per KV head.

    Returns:
        Expanded KV tensor ``[batch, seq, n_heads, head_dim]``.
    """
    if n_rep == 1:
        return x_bshd
    b, s, n_kv, d = x_bshd.shape
    return (
        x_bshd[:, :, :, None, :]
        .expand(b, s, n_kv, n_rep, d)
        .reshape(b, s, n_kv * n_rep, d)
    )


class BshdSdpaCore(nn.Module):
    """Causal scaled dot-product attention in BSHD layout ``[B, S, H, D]``."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute causal scaled dot-product attention in BSHD layout."""
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        return out.transpose(1, 2)


class RMSNorm(nn.Module):
    """Root mean square normalization (Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to the input tensor."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


@dataclass
class MoEDemoConfig:
    """Small MoE-style hyperparameters for the EP demo."""

    dim: int = 256
    n_layers: int = 2
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 1024
    max_seq_len: int = 128
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    num_experts: int = 4
    moe_hidden_dim: int = 512
    top_k: int = 2
    shared_expert_hidden_dim: int = 0


class MoEAttention(nn.Module):
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, cfg: MoEDemoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({cfg.n_heads}) must be divisible by n_kv_heads ({cfg.n_kv_heads})."
            )
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.tp_mesh_size: int = 1
        self.tp_mesh: Optional[DeviceMesh] = None

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.sdpa_core = BshdSdpaCore()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Forward pass with GQA attention and optional TP mesh sharding."""
        tp = self.tp_mesh_size
        n_h = self.n_heads // tp
        n_kv = self.n_kv_heads // tp
        mesh = self.tp_mesh
        b, s, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        def _local_shard(t: torch.Tensor) -> torch.Tensor:
            return t.to_local() if hasattr(t, "to_local") else t

        ql = _local_shard(q).reshape(b, s, n_h, self.head_dim)
        kl = _local_shard(k).reshape(b, s, n_kv, self.head_dim)
        vl = _local_shard(v).reshape(b, s, n_kv, self.head_dim)

        ql, kl = apply_rotary_emb(ql, kl, freqs_cis)
        kl = repeat_kv_bshd(kl, self.n_rep)
        vl = repeat_kv_bshd(vl, self.n_rep)

        out_bshd = self.sdpa_core(ql, kl, vl)
        out_l = out_bshd.reshape(b, s, n_h * self.head_dim)
        if mesh is not None:
            from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415
            out_l = DTensor.from_local(out_l, mesh, [PlShard(-1)])
        return self.wo(out_l)


class MoETransformerBlock(nn.Module):
    """One decoder block: attention + MoE feed-forward."""

    def __init__(self, cfg: MoEDemoConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attention = MoEAttention(cfg)

        shared_expert = None
        if cfg.shared_expert_hidden_dim > 0:
            shared_expert = FeedForward(
                dim=cfg.dim, hidden_dim=cfg.shared_expert_hidden_dim,
            )
        self.feed_forward = MoE(
            dim=cfg.dim,
            hidden_dim=cfg.moe_hidden_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            score_before_experts=True,
            shared_expert=shared_expert,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual attention block followed by residual MoE feed-forward block."""
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


class MoEDemoModel(nn.Module):
    """Decoder-only MoE model for EP demos.

    Structure: token embedding -> N x (attention + MoE FFN) -> norm -> output linear.
    The ``feed_forward.experts`` sub-modules are the targets for
    :class:`ExpertParallel` sharding.
    """

    def __init__(self, cfg: MoEDemoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            MoETransformerBlock(cfg) for _ in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token ids forward pass.

        Args:
            token_ids: ``[batch, seq_len]`` token indices.
            freqs_cis: Optional RoPE slice; defaults to ``self.freqs_cis[:seq_len]``.

        Returns:
            Logits ``[batch, seq_len, vocab_size]``.
        """
        h = self.tok_embeddings(token_ids)
        seq_loc = h.shape[1]
        if freqs_cis is None:
            freqs = self.freqs_cis[:seq_loc]
        else:
            freqs = freqs_cis
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        return self.output(h)
