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
"""Minimal Llama3-style decoder-only model for MindSpore (layout mirrors ``examples/torch/llama3/model.py``).

Submodule names match the TorchTitan tensor-parallel plan: ``tok_embeddings``, ``layers.*``,
``attention`` (``wq``/``wk``/``wv``/``wo``), ``feed_forward`` (``w1``/``w2``/``w3``),
``attention_norm``, ``ffn_norm``, final ``norm``, ``output``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from hyper_parallel import DTensor, SkipDTensorDispatch
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Shard as PlShard


def _compute_ffn_hidden_dim(dim: int, *, multiple_of: int = 256, ffn_dim_multiplier: float = 1.0) -> int:
    hidden_dim = int(2 * ffn_dim_multiplier * dim / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def precompute_rope_cos_sin(head_dim: int, end: int, theta: float = 500000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Llama-style RoPE angles; returns ``cos``, ``sin`` tables ``[end, head_dim // 2]`` (float32)."""
    d_half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32)[:d_half] / head_dim))
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)


def apply_rotary_emb(xq: Tensor, xk: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply RoPE using real-valued cos/sin tables (``cos``/``sin``: ``[seq, head_dim//2]``)."""
    if hasattr(cos, "to_local"):
        cos = cos.to_local()
    if hasattr(sin, "to_local"):
        sin = sin.to_local()
    seq_len = xq.shape[1]
    cos_s = cos[:seq_len, :]
    sin_s = sin[:seq_len, :]
    c = mint.unsqueeze(mint.unsqueeze(cos_s, 0), 2)
    s_ = mint.unsqueeze(mint.unsqueeze(sin_s, 0), 2)
    with SkipDTensorDispatch():
        xq1 = xq[..., 0::2]
        xq2 = xq[..., 1::2]
        xk1 = xk[..., 0::2]
        xk2 = xk[..., 1::2]
        xq_out = ops.stack((xq1 * c - xq2 * s_, xq1 * s_ + xq2 * c), axis=-1).reshape(xq.shape)
        xk_out = ops.stack((xk1 * c - xk2 * s_, xk1 * s_ + xk2 * c), axis=-1).reshape(xk.shape)
    return xq_out, xk_out


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads to match the number of query heads for grouped-query attention (GQA)."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    x = mint.unsqueeze(x, 2)
    x = mint.broadcast_to(x, (b, n_kv, n_rep, s, d))
    return mint.reshape(x, (b, n_kv * n_rep, s, d))


def causal_scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Scaled dot-product attention with causal mask; inputs ``B, N, S, D``."""
    scores = mint.matmul(q, mint.swapaxes(k, -2, -1)) * scale
    seq_len = scores.shape[-1]
    mask = mint.triu(mint.ones((seq_len, seq_len), dtype=scores.dtype), diagonal=1)
    scores = scores + mask * Tensor(-1e9, dtype=scores.dtype)
    attn = ops.softmax(scores, axis=-1)
    return mint.matmul(attn, v)


class Llama3RMSNorm(nn.Cell):
    """Root mean square normalization (Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = ms.Parameter(mint.ones((dim,), dtype=ms.float32))

    def construct(self, x: Tensor) -> Tensor:
        """Apply RMSNorm on the last dimension (local ops under ``SkipDTensorDispatch``)."""
        with SkipDTensorDispatch():
            dtype = x.dtype
            x_fp = ops.cast(x, ms.float32)
            if hasattr(x_fp, "to_local"):
                x_fp = x_fp.to_local()
            weight = self.weight
            if hasattr(weight, "to_local"):
                weight = weight.to_local()
            norm = mint.rsqrt(mint.mean(mint.square(x_fp), dim=-1, keepdim=True) + self.eps)
            out = x_fp * norm * weight
            return ops.cast(out, dtype)


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


class Llama3Attention(nn.Cell):
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, cfg: Llama3DemoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError(f"n_heads ({cfg.n_heads}) must be divisible by n_kv_heads ({cfg.n_kv_heads})")
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.tp_mesh_size: int = 1
        self.tp_mesh: Optional[DeviceMesh] = None

        self.wq = nn.Dense(cfg.dim, cfg.n_heads * self.head_dim, has_bias=False)
        self.wk = nn.Dense(cfg.dim, cfg.n_kv_heads * self.head_dim, has_bias=False)
        self.wv = nn.Dense(cfg.dim, cfg.n_kv_heads * self.head_dim, has_bias=False)
        self.wo = nn.Dense(cfg.n_heads * self.head_dim, cfg.dim, has_bias=False)

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Forward attention: project, RoPE, causal SDPA, then output projection."""
        tp = self.tp_mesh_size
        n_h = self.n_heads // tp
        n_kv = self.n_kv_heads // tp
        mesh = self.tp_mesh
        b, s, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        def _local_shard(t: Tensor) -> Tensor:
            return t.to_local() if hasattr(t, "to_local") else t

        ql = _local_shard(q).reshape(b, s, n_h, self.head_dim)
        kl = _local_shard(k).reshape(b, s, n_kv, self.head_dim)
        vl = _local_shard(v).reshape(b, s, n_kv, self.head_dim)

        ql, kl = apply_rotary_emb(ql, kl, cos, sin)

        ql = ops.transpose(ql, (0, 2, 1, 3))
        kl = ops.transpose(kl, (0, 2, 1, 3))
        vl = ops.transpose(vl, (0, 2, 1, 3))

        kl = repeat_kv(kl, self.n_rep)
        vl = repeat_kv(vl, self.n_rep)

        scale = float(self.head_dim ** -0.5)
        out_l = causal_scaled_dot_product_attention(ql, kl, vl, scale)
        out_l = ops.transpose(out_l, (0, 2, 1, 3)).reshape(b, s, n_h * self.head_dim)
        if mesh is not None:
            out_l = DTensor.from_local(out_l, mesh, [PlShard(-1)])
        return self.wo(out_l)


class Llama3LocalEmbedding(nn.Embedding):
    """Row-parallel token embedding without DTensor ``Gather`` dispatch.

    MindSpore ``nn.Embedding`` calls ``Gather``, which HyperParallel does not yet
    register for layout inference. Subclassing keeps ``RowwiseParallel`` support
    (``is_embedding_module``); forward runs under :class:`SkipDTensorDispatch` with
    local ``ops.gather`` and the same index shift / mask rules as row-parallel
    embedding in the library.
    """

    @staticmethod
    def _as_local(tensor: Tensor) -> Tensor:
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    @staticmethod
    def _embedding_dtensor(weight_ref: Tensor) -> Tensor | None:
        if hasattr(weight_ref, "device_mesh"):
            return weight_ref
        inner = getattr(weight_ref, "data", None)
        if inner is not None and hasattr(inner, "device_mesh"):
            return inner
        return None

    def _row_parallel_lookup(self, token_ids: Tensor, weight: Tensor) -> Tensor:
        """Gather embedding rows with row-parallel vocab sharding (local ``ops.gather``)."""
        ids = self._as_local(token_ids)
        table = self._as_local(weight)
        dtensor = self._embedding_dtensor(self.embedding_table)
        if dtensor is None:
            return ops.gather(table, ids, 0)

        mesh = dtensor.device_mesh
        mesh_dim_idx = len(mesh.mesh_shape) - 1
        vocab_coord = mesh.get_local_rank(mesh_dim_idx)
        vocab_per = int(table.shape[0])
        vocab_start = int(vocab_coord * vocab_per)
        vocab_end = vocab_start + vocab_per

        mask = (ids >= vocab_start) & (ids < vocab_end)
        mask_int = ops.cast(mask, ids.dtype)
        local_ids = (ids - vocab_start) * mask_int
        out = ops.gather(table, local_ids, 0)
        mask_f = ops.cast(mask, table.dtype)
        while mask_f.ndim < out.ndim:
            mask_f = ops.expand_dims(mask_f, -1)
        return out * mask_f

    def construct(self, token_ids: Tensor) -> Tensor:
        """Embed token IDs using row-parallel vocab sharding under SkipDTensorDispatch."""
        with SkipDTensorDispatch():
            return self._row_parallel_lookup(token_ids, self.embedding_table)


class Llama3FeedForward(nn.Cell):
    """SwiGLU feed-forward (w1 / w3 up, w2 down)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w2 = nn.Dense(hidden_dim, dim, has_bias=False)
        self.w3 = nn.Dense(dim, hidden_dim, has_bias=False)

    def construct(self, x: Tensor) -> Tensor:
        """SwiGLU feed-forward: up-project with w1/w3, apply SiLU gating, then down-project with w2."""
        return self.w2(mint.nn.functional.silu(self.w1(x)) * self.w3(x))


class Llama3TransformerBlock(nn.Cell):
    """One decoder block."""

    def __init__(self, cfg: Llama3DemoConfig, hidden_dim: int) -> None:
        super().__init__()
        self.attention_norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attention = Llama3Attention(cfg)
        self.feed_forward = Llama3FeedForward(cfg.dim, hidden_dim)

    def construct(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Decoder block forward: attention with residual, then feed-forward with residual."""
        h = x + self.attention(self.attention_norm(x), cos, sin)
        return h + self.feed_forward(self.ffn_norm(h))


class Llama3Model(nn.Cell):
    """Decoder-only Llama3-style stack for TP demos."""

    def __init__(self, cfg: Llama3DemoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = _compute_ffn_hidden_dim(
            cfg.dim, multiple_of=cfg.multiple_of, ffn_dim_multiplier=cfg.ffn_dim_multiplier
        )
        self.tok_embeddings = Llama3LocalEmbedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.CellList(
            [Llama3TransformerBlock(cfg, hidden_dim) for _ in range(cfg.n_layers)]
        )
        self.norm = Llama3RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Dense(cfg.dim, cfg.vocab_size, has_bias=False)

        head_dim = cfg.dim // cfg.n_heads
        cos_np, sin_np = precompute_rope_cos_sin(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.freqs_cos = Tensor(cos_np, dtype=ms.float32)
        self.freqs_sin = Tensor(sin_np, dtype=ms.float32)

    def construct(self, token_ids: Tensor) -> Tensor:
        """Full model forward: embed tokens, run through transformer layers, and produce logits."""
        h = self.tok_embeddings(token_ids)
        cos = self.freqs_cos
        sin = self.freqs_sin
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        return self.output(h)
