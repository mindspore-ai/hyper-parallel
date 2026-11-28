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
"""Training-reference KPool and QSA indexers with separate score and routing stages.

These components select original token indices, not compressed attention values.
The reference route chunks queries; it never constructs a token-by-token mask.
Packed metadata uses cumulative token ends in flattened batch order. CP ownership
and the main sparse-attention backend are deliberately outside these components.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.ops import pooled_indexer, rms_norm_fp32_affine, rope


def document_ranges(
    batch: int, length: int, actual_seq_len: tuple[int, ...] | None,
) -> list[tuple[int, int, int]]:
    """Return (batch, start, end); packed documents cannot cross batch boundaries."""
    if batch < 1 or length < 1:
        raise ValueError("pooled indexer requires nonempty batch and sequence dimensions")
    ends = tuple((i + 1) * length for i in range(batch)) if actual_seq_len is None else actual_seq_len
    if not isinstance(ends, (tuple, list)) or not ends or ends[-1] != batch * length:
        raise ValueError("actual_seq_len must contain cumulative ends covering B*S tokens")
    start, ranges = 0, []
    for end in ends:
        if isinstance(end, bool) or not isinstance(end, int) or end <= start:
            raise ValueError("packed cumulative ends must be strictly increasing integers")
        if start // length != (end - 1) // length:
            raise ValueError("a packed document cannot cross a batch boundary")
        ranges.append((start // length, start % length, (end - 1) % length + 1))
        start = end
    return ranges


def block_scores(
    query: torch.Tensor, pooled_key: torch.Tensor, weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score [Q,H,D] against [N,D], keeping the indexer graph differentiable."""
    scores = torch.einsum("qhd,nd->qhn", query.float(), pooled_key.float())
    scores = (scores * query.shape[-1] ** -0.5).relu()
    if weights is not None:
        scores = scores * weights.float().unsqueeze(-1)
    return scores.sum(dim=1)


def select_blocks(
    query: torch.Tensor, pooled_key: torch.Tensor, weights: torch.Tensor | None,
    pool_size: int, token_budget: int, query_chunk_size: int,
) -> torch.Tensor:
    """Dispatch causal block selection; KL trains the separately projected inputs."""
    return pooled_indexer(query, pooled_key, weights, pool_size=pool_size,
                          token_budget=token_budget, query_chunk_size=query_chunk_size)


def block_indexer_kl(logits: torch.Tensor, teacher_mass: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-query KL over selectable complete blocks; teacher is stop-gradient.

    teacher_mass is the sum of MAIN-attention probabilities over original tokens
    in each block and over attention heads. Conditional normalization excludes the
    forced tail. This explicit research objective is not an official GLM/Qwen
    training recipe. The caller owns coefficient, valid-token reduction and scale.
    """
    if logits.shape != teacher_mass.shape or valid.shape != logits.shape or valid.dtype != torch.bool:
        raise ValueError("logits, teacher block mass and bool validity must have matching shapes")
    if logits.shape[-1] == 0:
        return logits.sum(-1) * 0
    target = teacher_mass.detach().float().masked_fill(~valid, 0)
    denominator = target.sum(-1, keepdim=True)
    target = target / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    values = logits.float().masked_fill(~valid, float("-inf"))
    active = valid.any(-1, keepdim=True) & (denominator > 0)
    values = torch.where(active, values, torch.zeros_like(values))
    log_probs = values.log_softmax(-1).masked_fill(~valid, 0)
    terms = target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - log_probs)
    return terms.sum(-1)


class KPoolIndexer(nn.Module):
    """GLM-5.3 indexer: LayerNorm token K, gated pooling with APE, weighted heads."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Construct an independent indexer or preserve HF parameter objects."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        config = module.config if module is not None else config
        self.pool_size, self.token_budget = config.index_kpool, config.index_topk
        self.n_heads, self.head_dim = config.index_n_heads, config.index_head_dim
        _validate_budget(self.pool_size, self.token_budget)
        if config.qk_rope_head_dim != 0 or not config.index_kpool_always_select_tail:
            raise ValueError("this GLM KPool reference requires NoPE and always-select-tail")
        if module is None:
            self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
            self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.pool_size, self.head_dim))
            self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, config.hidden_size))
        else:
            for name in ("wq_b", "wk", "k_norm", "weights_proj", "index_kpool_compress_ape",
                         "index_kpool_compress_gate"):
                setattr(self, name, getattr(module, name))
            self.train(module.training)

    def project(
        self, hidden_states: torch.Tensor, q_resid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detach backbone activations; retain gradients for all indexer parameters."""
        if hidden_states.ndim != 3 or q_resid.ndim != 3 or hidden_states.shape[:2] != q_resid.shape[:2]:
            raise ValueError("KPool hidden states and Q latent must have matching [B,S,...] dimensions")
        hidden, latent = hidden_states.detach(), q_resid.detach()
        query = self.wq_b(latent).unflatten(-1, (self.n_heads, self.head_dim))
        key = self.k_norm(self.wk(hidden))
        gates = F.linear(hidden, self.index_kpool_compress_gate)
        weights = self.weights_proj(hidden).float() * self.n_heads ** -0.5
        return query, key, gates, weights

    def pool(self, key: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Pool one document's complete blocks, preserving HF's BF16 rounding."""
        count = key.shape[0] // self.pool_size
        grouped = key[:count * self.pool_size].reshape(count, self.pool_size, self.head_dim)
        gates = gates[:count * self.pool_size].reshape_as(grouped)
        probabilities = (gates.float() + self.index_kpool_compress_ape.float()).softmax(1).to(key.dtype)
        return (probabilities * grouped).sum(1)

    @torch.no_grad()
    def forward(
        self, hidden_states: torch.Tensor, q_resid: torch.Tensor, *,
        actual_seq_len: tuple[int, ...] | None = None, query_chunk_size: int = 128,
    ) -> torch.Tensor:
        """Return document-local original-token routes [B,S,budget+p-1], no S*S mask."""
        _validate_chunk(query_chunk_size)
        query, key, gates, weights = self.project(hidden_states, q_resid)
        batch, length = hidden_states.shape[:2]
        result = torch.full((batch, length, self.token_budget + self.pool_size - 1), -1,
                            device=hidden_states.device, dtype=torch.int32)
        for b, start, end in document_ranges(batch, length, actual_seq_len):
            pooled = self.pool(key[b, start:end], gates[b, start:end])
            route = select_blocks(query[b, start:end], pooled, weights[b, start:end],
                                  self.pool_size, self.token_budget, query_chunk_size)
            result[b, start:end] = torch.where(route >= 0, route + start, route)
        return result


class _OffsetNorm(nn.Module):
    """Qwen's 1+weight scale, multiplied in FP32 before the dtype cast."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and multiply the offset scale before restoring the input dtype."""
        return rms_norm_fp32_affine(x, 1 + self.weight.float(), self.eps)


class QSAIndexer(nn.Module):
    """Qwen QSA: mean raw K, then RMSNorm and block-start partial RoPE."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Construct an independent indexer or preserve HF parameter objects."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self.n_heads, self.head_dim = module.index_n_heads, module.index_head_dim
            self.pool_size, self.token_budget = module.compress_ratio, module.token_budget
            kv_heads = module.index_kv_heads
            self.index_qk_proj = module.index_qk_proj
            self.q_layernorm, self.k_layernorm = module.q_layernorm, module.k_layernorm
            self.train(module.training)
        else:
            self.n_heads, self.head_dim = config.indexer_n_heads, config.indexer_head_dim
            self.pool_size, self.token_budget = config.indexer_compress_ratio, config.indexer_budget
            kv_heads = config.indexer_kv_heads
            self.index_qk_proj = nn.Linear(config.hidden_size, (self.n_heads + 1) * self.head_dim, bias=False)
            self.q_layernorm = _OffsetNorm(self.head_dim, config.rms_norm_eps)
            self.k_layernorm = _OffsetNorm(self.head_dim, config.rms_norm_eps)
        if kv_heads != 1:
            raise ValueError("QSA indexer requires one index KV head")
        _validate_budget(self.pool_size, self.token_budget)

    def project(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce differentiable index Q and raw K from detached backbone states."""
        if hidden_states.ndim != 3:
            raise ValueError("QSA hidden states must have shape [B,S,H]")
        qk = self.index_qk_proj(hidden_states.detach())
        query, key = qk.split((self.n_heads * self.head_dim, self.head_dim), dim=-1)
        query = self.q_layernorm(query.unflatten(-1, (self.n_heads, self.head_dim)))
        cos, sin = position_embeddings
        return rope(query, cos.unsqueeze(-2), sin.unsqueeze(-2)), key

    def pool(
        self, key: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Pool complete document blocks; rotate at the FIRST token's position."""
        count = key.shape[0] // self.pool_size
        pooled = key[:count * self.pool_size].reshape(count, self.pool_size, self.head_dim).float().mean(1)
        pooled = self.k_layernorm(pooled.to(key.dtype))
        if count == 0:
            return pooled
        cos, sin = position_embeddings
        # A block key has one head. Keep that axis for the common fused RoPE ABI.
        block_cos = cos[:count * self.pool_size:self.pool_size].unsqueeze(1)
        block_sin = sin[:count * self.pool_size:self.pool_size].unsqueeze(1)
        return rope(pooled.unsqueeze(1), block_cos, block_sin).squeeze(1)

    @torch.no_grad()
    def forward(
        self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], *,
        actual_seq_len: tuple[int, ...] | None = None, query_chunk_size: int = 128,
    ) -> torch.Tensor:
        """Select original-token routes without materializing the main attention mask."""
        _validate_chunk(query_chunk_size)
        cos, sin = position_embeddings
        batch, length = hidden_states.shape[:2]
        if cos.ndim != 3 or cos.shape[:2] != (batch, length) or sin.shape != cos.shape:
            raise ValueError("QSA position embeddings must have shape [B,S,rotary_dim]")
        query, key = self.project(hidden_states, position_embeddings)
        result = torch.full((batch, length, self.token_budget + self.pool_size - 1), -1,
                            device=hidden_states.device, dtype=torch.int32)
        for b, start, end in document_ranges(batch, length, actual_seq_len):
            pooled = self.pool(key[b, start:end], (cos[b, start:end], sin[b, start:end]))
            route = select_blocks(query[b, start:end], pooled, None,
                                  self.pool_size, self.token_budget, query_chunk_size)
            result[b, start:end] = torch.where(route >= 0, route + start, route)
        return result


def _validate_budget(pool_size, token_budget):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (pool_size, token_budget)):
        raise ValueError("pool size and token budget must be positive integers")
    if token_budget % pool_size:
        raise ValueError("token budget must be divisible by pool size")


def _validate_chunk(chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")


@torch.no_grad()
def sparse_block_teacher(
    query: torch.Tensor, key: torch.Tensor, lse: torch.Tensor, indices: torch.Tensor, *,
    pool_size: int, scale: float, query_chunk_size: int = 16,
) -> torch.Tensor:
    """Aggregate original-token sparse attention probabilities into selected blocks.

    Args:
        query: Main query [B,S,Hq,D], including its actual model Norm/RoPE.
        key: Original main key [B,S,Hkv,D], not the indexer's pooled key.
        lse: Detached main attention logsumexp [B,S,Hq], including the forced tail.
        indices: Complete block IDs [B,S,K] in the student's order; -1 is padding.
        pool_size: Original token count per complete block.
        scale: The main attention scale, unchanged by latent absorption.
        query_chunk_size: Bound on temporary selected-key materialization.

    Returns:
        Detached FP32 teacher [B,S,K], normalized over complete blocks after
        averaging heads. Rows without complete blocks are zero. The LSE must
        come from the identical routing, Q/K, scale, mask and implicit tail.
    """
    _validate_chunk(query_chunk_size)
    _validate_budget(pool_size, pool_size)
    if query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]:
        raise ValueError("teacher Q/K must be BSHD with matching batch and sequence")
    batch, length, heads, dim = query.shape
    if heads % key.shape[2] or key.shape[-1] != dim or lse.shape != (batch, length, heads):
        raise ValueError("teacher head dimensions or LSE shape do not match main attention")
    if indices.ndim != 3 or indices.shape[:2] != (batch, length):
        raise ValueError("teacher complete block indices must have shape [B,S,K]")
    result = torch.zeros_like(indices, dtype=torch.float32)
    batches = torch.arange(batch, device=query.device)[:, None, None, None]
    offsets = torch.arange(pool_size, device=query.device)
    kv_heads, groups = key.shape[2], heads // key.shape[2]
    for start in range(0, length, query_chunk_size):
        end = min(start + query_chunk_size, length)
        ids = indices[:, start:end]
        complete = torch.arange(start + 1, end + 1, device=query.device) // pool_size
        valid = (ids >= 0) & (ids < complete[None, :, None])
        token_ids = ids.clamp_min(0)[..., None] * pool_size + offsets
        selected = key[batches, token_ids.clamp_max(length-1).long()].float()
        q = query[:, start:end].float().reshape(batch, end-start, kv_heads, groups, dim)
        scores = torch.einsum("bqhgd,bqkphd->bqhgkp", q, selected) * scale
        normalizer = lse[:, start:end].float().reshape(batch, end-start, kv_heads, groups, 1, 1)
        log_prob = torch.where(valid[:, :, None, None, :, None], scores-normalizer,
                               torch.full_like(scores, float("-inf")))
        mass = log_prob.exp().sum(-1).mean((2, 3))
        result[:, start:end] = mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return result
