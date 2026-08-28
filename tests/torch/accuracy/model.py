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
"""Minimal Llama3-style decoder-only model used for accuracy tests.

Self-contained copy of ``examples/torch/llama3/model.py`` so the test does not
rely on ``sys.path`` manipulation. Submodule names mirror TorchTitan's TP plan
so :func:`parallelize.parallelize_llama3` can reuse the same Colwise/Rowwise
plan for both standalone and parallel scenarios.
"""
# pylint: disable=C0413
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
    """Return the SwiGLU FFN hidden dim rounded up to a multiple of ``multiple_of``.

    Mirrors the Llama hidden-dim formula ``round_up(2/3 * ffn_dim_multiplier * dim)`` so the
    inner FFN width stays aligned for TP / FSDP sharding even when ``dim`` is not divisible by
    ``multiple_of``.
    """
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


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors using complex ``freqs_cis``."""
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
    """Tile KV heads along the head dim in **BHSD** layout (``[B, n_kv, S, D]`` -> ``[B, n_kv * n_rep, S, D]``).

    Used by GQA so each query head can attend to its grouped KV head; ``n_rep == 1`` is a no-op.
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


def repeat_kv_bshd(x_bshd: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads in **BSHD** layout (``[B, S, n_kv, D]`` -> ``[B, S, n_heads, D]``)."""
    if n_rep == 1:
        return x_bshd
    x_bh = x_bshd.transpose(1, 2)
    x_bh = repeat_kv(x_bh, n_rep)
    return x_bh.transpose(1, 2)


class Llama3BshdSdpaCore(nn.Module):
    """Causal SDPA exposed as a submodule so ``ContextParallel`` can hook ``forward(q,k,v)``."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run causal scaled dot-product attention on **BSHD** Q/K/V tensors.

        Args:
            q: Query in BSHD layout ``[B, S, n_heads, head_dim]``.
            k: Key in BSHD layout ``[B, S, n_heads, head_dim]``.
            v: Value in BSHD layout ``[B, S, n_heads, head_dim]``.

        Returns:
            Attention output in BSHD layout ``[B, S, n_heads, head_dim]``.
        """
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        return out.transpose(1, 2)


class Llama3RMSNorm(nn.Module):
    """Root mean square normalization (Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Build an RMSNorm with a learnable per-feature scale initialized to ones.

        Args:
            dim: Hidden size of the last input dimension being normalized.
            eps: Numerical stability term added inside the rsqrt.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` along its last dim and rescale by the learnable weight.

        Casts to float32 internally for stable mean-square computation and casts back to the
        original dtype before returning, so the normalizer behaves like the Llama reference even
        under autocast.
        """
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


@dataclass
class Llama3DemoConfig:
    """Small Llama3-like hyperparameters for the accuracy test."""

    dim: int = 256
    n_layers: int = 2
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 1024
    max_seq_len: int = 128
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1.0


class Llama3Attention(nn.Module):
    """Multi-head attention with GQA + RoPE (separate Q/K/V projections for TP plans)."""

    def __init__(self, cfg: Llama3DemoConfig) -> None:
        """Build the four ``Linear`` projections (``wq``/``wk``/``wv``/``wo``) and the SDPA core.

        ``tp_mesh_size`` and ``tp_mesh`` default to the single-card configuration; they are
        overwritten by :func:`tests.torch.accuracy.parallelize.parallelize_llama3` once a TP mesh
        is wired in, which lets :meth:`forward` reshape Q/K/V into local head counts and wrap
        the attention output back as a ``DTensor`` before ``wo``.
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.tp_mesh_size: int = 1
        self.tp_mesh: Optional[DeviceMesh] = None

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.sdpa_core = Llama3BshdSdpaCore()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Compute one attention block: Q/K/V projections, RoPE, GQA expand, SDPA, output proj.

        When TP is active, Q/K/V projections produce DTensor shards along the head dimension; we
        unwrap them via ``to_local()`` so RoPE and reshapes operate on plain tensors with the
        per-rank head counts (``n_heads / tp`` and ``n_kv_heads / tp``). The SDPA output is then
        rewrapped as a ``DTensor`` on ``self.tp_mesh`` so ``wo`` can apply the row-wise plan.
        """
        tp = self.tp_mesh_size
        n_h = self.n_heads // tp
        n_kv = self.n_kv_heads // tp
        mesh = self.tp_mesh
        b, s, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        def _local_shard(t: torch.Tensor) -> torch.Tensor:
            """Return ``t.to_local()`` if ``t`` is a DTensor, otherwise ``t`` itself."""
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
            out_l = DTensor.from_local(out_l, mesh, [PlShard(-1)])
        return self.wo(out_l)


class Llama3FeedForward(nn.Module):
    """SwiGLU feed-forward (w1 / w3 up, w2 down)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Build the three Linear layers used by the SwiGLU FFN.

        Args:
            dim: Hidden dim flowing into / out of the FFN.
            hidden_dim: Inner FFN width (typically computed by :func:`_compute_ffn_hidden_dim`).
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SwiGLU: ``w2(silu(w1(x)) * w3(x))``."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Llama3TransformerBlock(nn.Module):
    """One decoder block (`attention_norm` / `ffn_norm` names match TorchTitan)."""

    def __init__(self, cfg: Llama3DemoConfig, hidden_dim: int) -> None:
        """Build the pre-attention / pre-FFN RMSNorms together with the attention and FFN modules.

        Args:
            cfg: Llama3 demo hyperparameters (used for ``dim`` / ``norm_eps`` / GQA shapes).
            hidden_dim: Inner FFN width (typically computed by :func:`_compute_ffn_hidden_dim`).
        """
        super().__init__()
        self.attention_norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attention = Llama3Attention(cfg)
        self.feed_forward = Llama3FeedForward(cfg.dim, hidden_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Run a Llama-style residual block: ``x + attn(norm(x)) -> h + ffn(norm(h))``.

        Args:
            x: Hidden states ``[B, S, dim]``.
            freqs_cis: RoPE table aligned with the local sequence window seen by ``x``.
        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


class Llama3Model(nn.Module):
    """Decoder-only Llama3-style stack used by accuracy tests."""

    def __init__(self, cfg: Llama3DemoConfig) -> None:
        """Build embeddings, ``cfg.n_layers`` transformer blocks, final norm and LM head.

        Also precomputes the RoPE frequency table (``freqs_cis``) and registers it as a
        non-persistent buffer; CP scenarios slice into it with ``rope_seq_start`` so RoPE
        positions stay aligned with each rank's global sequence window.
        """
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

    def forward(
        self,
        token_ids: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        *,
        rope_seq_start: int = 0,
    ) -> torch.Tensor:
        """Run the full Llama3 forward and return logits ``[B, S, vocab_size]``.

        When ``freqs_cis`` is ``None`` (typical), the RoPE slice is taken from
        ``self.freqs_cis[rope_seq_start : rope_seq_start + S_local]``; pass
        ``rope_seq_start = cp_rank * (S / cp_size)`` under context parallel so the slice matches
        each rank's global token positions.

        Args:
            token_ids: Token id tensor ``[B, S_local]`` (the rank's local sequence slice
                under CP, or the full sequence on a single card).
            freqs_cis: Optional pre-sliced RoPE table; when supplied it bypasses
                ``rope_seq_start`` and is used directly.
            rope_seq_start: Global position of ``token_ids[0]`` along the sequence axis;
                ignored if ``freqs_cis`` is supplied explicitly.
        """
        h = self.tok_embeddings(token_ids)
        seq_loc = h.shape[1]
        if freqs_cis is None:
            freqs = self.freqs_cis[rope_seq_start : rope_seq_start + seq_loc]
        else:
            freqs = freqs_cis
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        return self.output(h)
