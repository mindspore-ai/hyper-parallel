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
"""Minimal Llama3-style decoder-only model (aligned with TorchTitan `models/llama3` layout).

Submodule names mirror TorchTitan's tensor-parallel plan: ``tok_embeddings``, ``layers.*``,
``attention`` (with ``wq``/``wk``/``wv``/``wo``), ``feed_forward`` (with ``w1``/``w2``/``w3``),
``attention_norm``, ``ffn_norm``, final ``norm``, ``output``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from hyper_parallel import DTensor, SkipDTensorDispatch
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Shard as PlShard


def _compute_ffn_hidden_dim(dim: int, *, multiple_of: int = 256, ffn_dim_multiplier: float = 1.0) -> int:
    hidden_dim = int(2 * ffn_dim_multiplier * dim / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
    """Llama-style RoPE frequencies as complex tensor ``[end, dim // 2]``."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast ``freqs_cis`` of shape ``[seq, head_dim/2]`` to complex ``x`` (B, S, ..., D/2)."""
    if hasattr(freqs_cis, "to_local"):
        freqs_cis = freqs_cis.to_local()
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        f"freqs_cis {freqs_cis.shape} vs x {x.shape}"
    )
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    with SkipDTensorDispatch():
        return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors using complex ``freqs_cis`` (Llama-style).

    Runs under :class:`~hyper_parallel.SkipDTensorDispatch` so reshapes avoid DTensor reshape rules.
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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


class Llama3RMSNorm(nn.Module):
    """Root mean square normalization (Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


@dataclass
class Llama3DemoConfig:
    """Small Llama3-like hyperparameters for the TP demo (`debugmodel` scale)."""

    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 2048
    max_seq_len: int = 512
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1.0


class Llama3Attention(nn.Module):
    """Multi-head attention with GQA and RoPE (separate Q/K/V projections for TP plans)."""

    def __init__(self, cfg: Llama3DemoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        # Set by ``parallelize_llama3`` to the 1-D TP mesh size (default 1 = no TP).
        self.tp_mesh_size: int = 1
        # Device mesh for wrapping local shards as DTensor before RowwiseParallel ``wo``.
        self.tp_mesh: Optional[DeviceMesh] = None

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Attention forward with optional TP local head counts (``tp_mesh_size`` set by TP plan).

        Uses local shards for Q/K/V reshape and RoPE; wraps output for ``wo`` when ``tp_mesh`` is set.
        """
        tp = self.tp_mesh_size
        n_h = self.n_heads // tp
        n_kv = self.n_kv_heads // tp
        mesh = self.tp_mesh
        b, s, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Colwise shards the last dim; reshape on plain tensors (local shards) avoids DTensor
        # ``view`` global-shape mismatch with HyperParallel reshape rules.
        def _local_shard(t: torch.Tensor) -> torch.Tensor:
            return t.to_local() if hasattr(t, "to_local") else t

        ql = _local_shard(q).reshape(b, s, n_h, self.head_dim)
        kl = _local_shard(k).reshape(b, s, n_kv, self.head_dim)
        vl = _local_shard(v).reshape(b, s, n_kv, self.head_dim)

        ql, kl = apply_rotary_emb(ql, kl, freqs_cis)

        ql = ql.transpose(1, 2)
        kl = kl.transpose(1, 2)
        vl = vl.transpose(1, 2)

        kl = repeat_kv(kl, self.n_rep)
        vl = repeat_kv(vl, self.n_rep)

        out_l = F.scaled_dot_product_attention(ql, kl, vl, is_causal=True)
        out_l = out_l.transpose(1, 2).contiguous().reshape(b, s, n_h * self.head_dim)
        if mesh is not None:
            out_l = DTensor.from_local(out_l, mesh, [PlShard(-1)])
        return self.wo(out_l)


class Llama3FeedForward(nn.Module):
    """SwiGLU feed-forward (w1 / w3 up, w2 down)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Llama3TransformerBlock(nn.Module):
    """One decoder block (`attention_norm` / `ffn_norm` names match TorchTitan)."""

    def __init__(self, cfg: Llama3DemoConfig, hidden_dim: int) -> None:
        super().__init__()
        self.attention_norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attention = Llama3Attention(cfg)
        self.feed_forward = Llama3FeedForward(cfg.dim, hidden_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


class Llama3Model(nn.Module):
    """Decoder-only Llama3-style stack for TP demos."""

    def __init__(self, cfg: Llama3DemoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = _compute_ffn_hidden_dim(
            cfg.dim, multiple_of=cfg.multiple_of, ffn_dim_multiplier=cfg.ffn_dim_multiplier
        )
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            Llama3TransformerBlock(cfg, hidden_dim) for _ in range(cfg.n_layers)
        )
        self.norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        freqs = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        h = self.tok_embeddings(token_ids)
        freqs_cis = self.freqs_cis[:seq_len]
        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self.norm(h)
        return self.output(h)
