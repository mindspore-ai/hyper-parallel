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
"""GLM5 DSA indexer and sparse-attention boundaries."""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def _prepare_query_positions(
    query_positions: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return batch-major query positions."""
    if query_positions is None:
        query_positions = torch.arange(
            key_len - query_len,
            key_len,
            device=device,
            dtype=torch.long,
        )
    if query_positions.ndim == 1:
        query_positions = query_positions.view(1, -1).expand(batch_size, -1)
    if query_positions.ndim != 2:
        raise ValueError("GLM5 DSA positions must have shape (seq,) or (batch, seq)")
    if query_positions.shape[0] != batch_size or query_positions.shape[1] != query_len:
        raise ValueError("GLM5 DSA positions must match query shape")
    return query_positions.to(device=device, dtype=torch.long)


def _infer_key_positions(query_positions: torch.Tensor, key_len: int) -> torch.Tensor:
    """Infer key positions for append-only cached decoding."""
    query_len = query_positions.shape[1]
    past_len = key_len - query_len
    if past_len <= 0:
        return query_positions.unsqueeze(1)
    if query_len > 1:
        expected = query_positions[:, :1] + torch.arange(
            query_len,
            device=query_positions.device,
            dtype=query_positions.dtype,
        ).view(1, -1)
        if not torch.equal(query_positions, expected):
            raise ValueError(
                "GLM5 DSA cached decode requires contiguous query positions; "
                "packed or non-contiguous cached position metadata is not supported."
            )
    offsets = torch.arange(
        key_len,
        device=query_positions.device,
        dtype=query_positions.dtype,
    ).view(1, 1, key_len)
    return (query_positions[:, :1].unsqueeze(1) - past_len + offsets).clamp_min(0)


class GLM5DSAIndexerBoundary(nn.Module):
    """Select global causal key positions for each local query token."""

    def __init__(self, topk: int, query_chunk_size: int = 64) -> None:
        super().__init__()
        self.topk = topk
        self.query_chunk_size = query_chunk_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return sparse key indices for each query token."""
        query = _local_tensor(query)
        key = _local_tensor(key)
        if query_positions is not None:
            query_positions = _local_tensor(query_positions)
        batch_size, query_len, _, indexer_dim = query.shape
        key_len = key.shape[1]
        topk = min(self.topk, key_len)
        if query_len == 0:
            return query.new_empty(batch_size, query_len, topk)
        query_positions = _prepare_query_positions(
            query_positions,
            batch_size,
            query_len,
            key_len,
            query.device,
        )
        key_positions = _infer_key_positions(query_positions, key_len)
        topk_chunks = []
        for start in range(0, query_len, self.query_chunk_size):
            end = min(start + self.query_chunk_size, query_len)
            query_chunk = query[:, start:end]
            scores = torch.einsum(
                "bsnd,btnd->bst", query_chunk, key,
            ) * (indexer_dim ** -0.5)
            query_pos = query_positions[:, start:end].reshape(
                batch_size, end - start, 1,
            )
            scores = scores.masked_fill(
                ~(key_positions <= query_pos), float("-inf"),
            )
            topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)
            topk_chunks.append(
                topk_indices.masked_fill(~torch.isfinite(topk_scores), -1)
            )
        if not topk_chunks:
            return query.new_empty(batch_size, query_len, topk)
        return torch.cat(topk_chunks, dim=1).reshape(
            batch_size, query_len, topk,
        )


class GLM5DSAIndexer(nn.Module):
    """Project hidden states and invoke the CP-compatible indexer boundary."""

    def __init__(self, hidden_size: int, indexer_dim: int, topk: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, indexer_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, indexer_dim, bias=False)
        self.boundary = GLM5DSAIndexerBoundary(topk)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected positions and indexer key cache."""
        query = self.query_proj(hidden_states).unsqueeze(2)
        current_key = self.key_proj(hidden_states).unsqueeze(2)
        key = (
            torch.cat([past_key, current_key], dim=1)
            if past_key is not None
            else current_key
        )
        topk_indices = self.boundary(query, key, position_ids)
        return topk_indices, key


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the split-half RoPE dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_single_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int) -> torch.Tensor:
    """Apply official split-half GLM-MoE-DSA RoPE to one tensor."""
    return (x * cos.unsqueeze(unsqueeze_dim)) + (
        _rotate_half(x) * sin.unsqueeze(unsqueeze_dim)
    )


class GLM5OfficialDSAIndexer(nn.Module):
    """Transformers GLM-MoE-DSA compatible sparse-attention indexer."""

    def __init__(
            self,
            hidden_size: int,
            q_lora_rank: int,
            qk_rope_head_dim: int,
            index_topk: int,
            index_head_dim: int,
            index_n_heads: int) -> None:
        super().__init__()
        if index_head_dim < qk_rope_head_dim:
            raise ValueError("index_head_dim must be >= qk_rope_head_dim")
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_topk = index_topk
        self.wq_b = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, index_n_heads, bias=False)
        self.softmax_scale = index_head_dim ** -0.5
        self.register_buffer("_cached_keys", None, persistent=False)

    @torch.no_grad()
    def forward(
            self,
            hidden_states: torch.Tensor,
            q_resid: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False) -> torch.Tensor:
        """Return official GLM-MoE-DSA top-k key indices."""
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = _apply_single_rotary(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(
            k,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_pe = _apply_single_rotary(
            k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2,
        ).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if seq_len > 1:
            self._cached_keys = None
        if use_cache:
            k = (
                torch.cat([self._cached_keys, k], dim=1)
                if self._cached_keys is not None
                else k
            )
            self._cached_keys = k

        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k.float())
        scores = F.relu(scores * self.softmax_scale)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)
        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices


class GLM5SparseAttentionCore(nn.Module):
    """Sparse attention over selected global key/value positions."""

    def __init__(self, scale: float, query_chunk_size: int = 64) -> None:
        super().__init__()
        self.scale = scale
        self.query_chunk_size = query_chunk_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run sparse attention over BSHD tensors."""
        query = _local_tensor(query)
        key = _local_tensor(key)
        value = _local_tensor(value)
        topk_indices = _local_tensor(topk_indices)
        if query_positions is not None:
            query_positions = _local_tensor(query_positions)
        if attention_mask is not None:
            attention_mask = _local_tensor(attention_mask)

        batch_size, query_len, num_heads, head_dim = query.shape
        batch_indices = torch.arange(
            batch_size, device=query.device,
        ).reshape(batch_size, 1, 1)
        mask = attention_mask
        if attention_mask is not None and attention_mask.ndim == 4:
            if mask.shape[1] == 1 and num_heads > 1:
                mask = mask.expand(batch_size, num_heads, -1, -1)
            mask = mask.transpose(1, 2)
        outputs = []
        for start in range(0, query_len, self.query_chunk_size):
            end = min(start + self.query_chunk_size, query_len)
            query_chunk = query[:, start:end]
            topk_chunk = topk_indices[:, start:end]
            safe_indices = topk_chunk.clamp_min(0)
            selected_key = key[batch_indices, safe_indices].permute(
                0, 1, 3, 2, 4,
            )
            selected_value = value[batch_indices, safe_indices].permute(
                0, 1, 3, 2, 4,
            )
            scores = (
                query_chunk.unsqueeze(-2) * selected_key
            ).sum(dim=-1) * self.scale
            invalid = topk_chunk.lt(0).unsqueeze(2)
            scores = scores.masked_fill(invalid, float("-inf"))

            if mask is not None:
                mask_chunk = (
                    mask[:, start:end]
                    if mask.shape[1] == query_len
                    else mask
                )
                gather_indices = safe_indices.unsqueeze(2).expand(
                    batch_size, end - start, num_heads, safe_indices.shape[-1],
                )
                scores = scores + torch.gather(mask_chunk, -1, gather_indices)

            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            probabilities = probabilities.masked_fill(invalid, 0.0)
            outputs.append(
                (probabilities.unsqueeze(-1) * selected_value)
                .sum(dim=-2)
                .reshape(batch_size, end - start, num_heads, head_dim)
            )
        if not outputs:
            return query.new_empty(batch_size, query_len, num_heads, head_dim)
        return torch.cat(outputs, dim=1)


__all__ = [
    "GLM5DSAIndexer",
    "GLM5DSAIndexerBoundary",
    "GLM5OfficialDSAIndexer",
    "GLM5SparseAttentionCore",
]
