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
"""GLM5 attention blocks."""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.glm5.dsa import (
    GLM5OfficialDSAIndexer,
    GLM5SparseAttentionCore,
)
from hyper_parallel.models.modules.rmsnorm import RMSNorm
from hyper_parallel.models.modules.rope import RotaryEmbedding, apply_rotary_pos_emb


def _expand_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads in BHSD layout."""
    bsz, num_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return x.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


def _expand_kv_heads_bshd(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads in BSHD layout."""
    if n_rep == 1:
        return x
    return _expand_kv_heads(x.transpose(1, 2), n_rep).transpose(1, 2)


def _prepare_position_ids(
    position_ids: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return batch-major position ids for the current query tokens."""
    if position_ids is None:
        position_ids = torch.arange(
            past_len,
            past_len + seq_len,
            device=device,
            dtype=torch.long,
        )
    if position_ids.ndim == 1:
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    if position_ids.ndim != 2:
        raise ValueError("GLM5 position_ids must have shape (seq,) or (batch, seq)")
    if position_ids.shape[0] != batch_size or position_ids.shape[1] != seq_len:
        raise ValueError(
            "GLM5 position_ids shape must match the current hidden states"
        )
    return position_ids.to(device=device, dtype=torch.long)


def _infer_key_position_ids(
    query_position_ids: torch.Tensor,
    kv_seq_len: int,
) -> torch.Tensor:
    """Infer key positions for append-only cached decoding."""
    query_len = query_position_ids.shape[1]
    past_len = kv_seq_len - query_len
    if past_len <= 0:
        return query_position_ids
    if query_len > 1:
        expected = query_position_ids[:, :1] + torch.arange(
            query_len,
            device=query_position_ids.device,
            dtype=query_position_ids.dtype,
        ).view(1, -1)
        if not torch.equal(query_position_ids, expected):
            raise ValueError(
                "GLM5 cached decode requires contiguous query position_ids; "
                "packed or non-contiguous cached position metadata is not supported."
            )
    offsets = torch.arange(
        kv_seq_len,
        device=query_position_ids.device,
        dtype=query_position_ids.dtype,
    ).view(1, -1)
    return (query_position_ids[:, :1] - past_len + offsets).clamp_min(0)


def _rotary_cos_sin(
    rotary_emb: RotaryEmbedding,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RoPE tensors shaped for 1D or batch-major position ids."""
    cos, sin = rotary_emb(hidden_states, position_ids)
    if position_ids.ndim == 2 and cos.ndim == 2:
        cos = cos.view(position_ids.shape[0], position_ids.shape[1], -1)
        sin = sin.view(position_ids.shape[0], position_ids.shape[1], -1)
    return cos, sin


def _local_dim(tensor: torch.Tensor, dim: int) -> int:
    """Return the physical local dimension used by DTensor-dispatched ops."""
    if hasattr(tensor, "to_local"):
        return tensor.to_local().shape[dim]
    return tensor.shape[dim]


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the physical local tensor used inside GLM5 attention math."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _prepare_additive_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    key_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize GLM5 2D padding masks and 4D additive masks."""
    if attention_mask is None:
        return None
    attention_mask = attention_mask.to(device=device)
    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch_size, key_len):
            raise ValueError(
                "GLM5 2D attention_mask must have shape (batch, key_len)"
            )
        causal = torch.tril(
            torch.ones(query_len, key_len, dtype=torch.bool, device=device),
            diagonal=key_len - query_len,
        )
        allowed = attention_mask.to(torch.bool).view(
            batch_size, 1, 1, key_len,
        ) & causal.view(1, 1, query_len, key_len)
        return torch.zeros(
            batch_size, 1, query_len, key_len, dtype=dtype, device=device,
        ).masked_fill(~allowed, -10000.0)
    if attention_mask.ndim == 4:
        if (
            attention_mask.shape[0] not in (1, batch_size)
            or attention_mask.shape[-1] != key_len
        ):
            raise ValueError(
                "GLM5 4D attention_mask must match batch and key length"
            )
        if attention_mask.shape[-2] not in (1, query_len):
            raise ValueError(
                "GLM5 4D attention_mask query dimension must be 1 or query_len"
            )
        return attention_mask.to(dtype=dtype)
    raise NotImplementedError(
        "GLM5 attention_mask must be a 2D padding mask or a 4D additive mask"
    )


def _align_cp_attention_mask(
    attention_mask: Optional[torch.Tensor],
    query_len: int,
    key_len: int,
    cp_rank: int,
    cp_size: int,
) -> Optional[torch.Tensor]:
    """Align a 4D additive mask with ContextParallel local/global Q/K shapes."""
    if attention_mask is None or attention_mask.ndim != 4 or cp_size <= 1:
        return attention_mask

    mask_query_len = attention_mask.shape[-2]
    mask_key_len = attention_mask.shape[-1]
    if mask_key_len == key_len * cp_size:
        key_start = cp_rank * key_len
        key_end = key_start + key_len
        attention_mask = attention_mask[..., key_start:key_end]
        mask_key_len = key_len
    elif mask_key_len * cp_size == key_len:
        key_start = cp_rank * mask_key_len
        key_end = key_start + mask_key_len
        full_mask = torch.full(
            (*attention_mask.shape[:-1], key_len),
            float("-inf"),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_mask[..., key_start:key_end] = attention_mask
        attention_mask = full_mask
        mask_key_len = key_len

    if mask_query_len == query_len * cp_size:
        query_start = cp_rank * query_len
        query_end = query_start + query_len
        attention_mask = attention_mask[..., query_start:query_end, :]
    elif mask_query_len != 1 and mask_query_len * cp_size == query_len:
        query_start = cp_rank * mask_query_len
        query_end = query_start + mask_query_len
        full_mask = torch.full(
            (*attention_mask.shape[:-2], query_len, mask_key_len),
            float("-inf"),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_mask[..., query_start:query_end, :] = attention_mask
        attention_mask = full_mask
    return attention_mask


class GLM5AttentionCore(nn.Module):
    """Attention core with explicit BSHD Q/K/V inputs."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run attention over BSHD tensors."""
        q_heads = query.transpose(1, 2)
        k_heads = key.transpose(1, 2)
        v_heads = value.transpose(1, 2)
        q_len = _local_dim(q_heads, 2)
        kv_len = _local_dim(k_heads, 2)
        q_heads = _local_tensor(q_heads)
        k_heads = _local_tensor(k_heads)
        v_heads = _local_tensor(v_heads)
        attention_mask = _align_cp_attention_mask(
            attention_mask,
            query_len=q_len,
            key_len=kv_len,
            cp_rank=getattr(self, "_cp_rank", 0),
            cp_size=getattr(self, "_cp_size", 1),
        )
        attention_mask = _prepare_additive_attention_mask(
            attention_mask,
            batch_size=q_heads.shape[0],
            query_len=q_len,
            key_len=kv_len,
            dtype=q_heads.dtype,
            device=q_heads.device,
        )

        attn_weights = torch.matmul(q_heads, k_heads.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            if attention_mask.shape[1] == 1 and q_heads.shape[1] > 1:
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0],
                    q_heads.shape[1],
                    -1,
                    -1,
                )
            attn_weights = attn_weights + attention_mask
        else:
            causal = torch.triu(
                torch.full(
                    (q_len, kv_len),
                    float("-inf"),
                    device=q_heads.device,
                    dtype=q_heads.dtype,
                ),
                diagonal=1 + kv_len - q_len,
            )
            attn_weights = attn_weights + causal
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q_heads.dtype
        )
        output = torch.matmul(attn_weights, v_heads)
        return output.transpose(1, 2).contiguous()


class GLM5GQAAttention(nn.Module):
    """GLM5 grouped-query attention with a hookable attention core."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_bias: bool = False,
        out_bias: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        use_dsa: bool = False,
    ):
        del rms_norm_eps
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=out_bias)
        self.rotary_emb = rope or RotaryEmbedding(
            head_dim,
            max_position_embeddings,
            rope_theta,
        )
        self.attention_core = GLM5AttentionCore(scale=head_dim ** -0.5)
        self.sparse_attention_core = (
            GLM5SparseAttentionCore(scale=head_dim ** -0.5)
            if use_dsa
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """Forward pass."""
        bsz, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim,
        )

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
        present_key_value = (key, value)
        kv_seq_len = key.shape[1]

        position_ids = _prepare_position_ids(
            position_ids,
            bsz,
            seq_len,
            kv_seq_len - seq_len,
            hidden_states.device,
        )
        key_position_ids = _infer_key_position_ids(position_ids, kv_seq_len)
        q_cos, q_sin = _rotary_cos_sin(
            self.rotary_emb, hidden_states, position_ids,
        )
        k_cos, k_sin = _rotary_cos_sin(
            self.rotary_emb, hidden_states, key_position_ids,
        )
        # ``apply_rotary_pos_emb`` rotates a query/key pair; GLM5 applies
        # different position ids to query and key, so rotate each side alone.
        query, _ = apply_rotary_pos_emb(
            query.transpose(1, 2),
            query.transpose(1, 2),
            q_cos,
            q_sin,
        )
        key, _ = apply_rotary_pos_emb(
            key.transpose(1, 2),
            key.transpose(1, 2),
            k_cos,
            k_sin,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        kv_groups = query.shape[2] // key.shape[2]
        if kv_groups > 1:
            key = _expand_kv_heads_bshd(key, kv_groups)
            value = _expand_kv_heads_bshd(value, kv_groups)

        attention_mask = _prepare_additive_attention_mask(
            attention_mask,
            batch_size=bsz,
            query_len=seq_len,
            key_len=key.shape[1],
            dtype=query.dtype,
            device=query.device,
        )
        if topk_indices is not None:
            sparse_core = self.sparse_attention_core
            if sparse_core is None:
                raise RuntimeError("DSA topk indices require a sparse attention core.")
            attn_output = sparse_core.forward(
                query,
                key,
                value,
                topk_indices,
                position_ids,
                attention_mask,
            )
        else:
            attn_output = self.attention_core(
                query,
                key,
                value,
                attention_mask=attention_mask,
            )
        output = self.o_proj(attn_output.view(bsz, seq_len, -1))
        if use_cache:
            return output, present_key_value
        return output


class GLM5MLAAttention(nn.Module):
    """Multi-head latent attention with compressed KV states."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_dsa: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.kv_lora_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=bias)
        self.kv_lora_norm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_lora_b_proj = nn.Linear(
            kv_lora_rank,
            num_kv_heads * (head_dim + v_head_dim),
            bias=bias,
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=bias)
        self.rotary_emb = RotaryEmbedding(
            dim=qk_rope_head_dim,
            max_seq_len=max_position_embeddings,
            theta=rope_theta,
        )
        self.attention_core = GLM5AttentionCore(scale=head_dim ** -0.5)
        self.sparse_attention_core = (
            GLM5SparseAttentionCore(scale=head_dim ** -0.5)
            if use_dsa
            else None
        )

    def _project_kv(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project MLA latent states to key and value tensors."""
        bsz, seq_len, _ = latent.shape
        kv = self.kv_lora_b_proj(self.kv_lora_norm(latent))
        kv = kv.view(
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim + self.v_head_dim,
        )
        key, value = torch.split(kv, [self.head_dim, self.v_head_dim], dim=-1)
        return key.transpose(1, 2), value.transpose(1, 2)

    def _compute_attention_output(
        self,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run dense or sparse MLA attention."""
        if topk_indices is None:
            return self.attention_core(
                q,
                key,
                value,
                attention_mask=attention_mask,
            )

        sparse_core = self.sparse_attention_core
        if sparse_core is None:
            raise RuntimeError("DSA topk indices require a sparse attention core.")
        attn_output = sparse_core.forward(
            q,
            key,
            value,
            topk_indices,
            position_ids,
            attention_mask,
        )
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """Forward pass."""
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim,
        )

        latent = self.kv_lora_a_proj(hidden_states)
        if past_key_value is not None:
            latent_for_kv = torch.cat([past_key_value, latent], dim=1)
        else:
            latent_for_kv = latent
        key, value = self._project_kv(latent_for_kv)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        kv_seq_len = key.shape[1]
        position_ids = _prepare_position_ids(
            position_ids,
            bsz,
            seq_len,
            kv_seq_len - seq_len,
            hidden_states.device,
        )
        key_position_ids = _infer_key_position_ids(position_ids, kv_seq_len)
        q_cos, q_sin = _rotary_cos_sin(
            self.rotary_emb, hidden_states, position_ids,
        )
        k_cos, k_sin = _rotary_cos_sin(
            self.rotary_emb, hidden_states, key_position_ids,
        )
        # ``apply_rotary_pos_emb`` rotates a query/key pair; GLM5 applies
        # different position ids to query and key, so rotate each side alone.
        q, _ = apply_rotary_pos_emb(
            q.transpose(1, 2),
            q.transpose(1, 2),
            q_cos,
            q_sin,
        )
        key, _ = apply_rotary_pos_emb(
            key.transpose(1, 2),
            key.transpose(1, 2),
            k_cos,
            k_sin,
        )
        q = q.transpose(1, 2)
        key = key.transpose(1, 2)

        kv_groups = q.shape[2] // key.shape[2]
        if kv_groups > 1:
            key = _expand_kv_heads_bshd(key, kv_groups)
            value = _expand_kv_heads_bshd(value, kv_groups)

        if value.shape[-1] > q.shape[-1]:
            pad_size = value.shape[-1] - q.shape[-1]
            q = F.pad(q, (0, pad_size))
            key = F.pad(key, (0, pad_size))

        attention_mask = _prepare_additive_attention_mask(
            attention_mask,
            batch_size=bsz,
            query_len=seq_len,
            key_len=key.shape[1],
            dtype=q.dtype,
            device=q.device,
        )
        attn_output = self._compute_attention_output(
            q,
            key,
            value,
            topk_indices,
            position_ids,
            attention_mask,
        )
        attn_output = attn_output.view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)
        if use_cache:
            return output, latent_for_kv
        return output


class GLM5OfficialMLAAttention(nn.Module):
    """Transformers GLM-MoE-DSA compatible MLA attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        index_topk: int,
        index_head_dim: int,
        index_n_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=bias)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=bias,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=bias)
        self.indexer = GLM5OfficialDSAIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
        )
        self.rotary_emb = RotaryEmbedding(
            dim=qk_rope_head_dim,
            max_seq_len=max_position_embeddings,
            theta=rope_theta,
        )
        self.attention_core = GLM5AttentionCore(scale=self.qk_head_dim ** -0.5)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project hidden states to official MLA query, key, and value tensors."""
        bsz, seq_len, _ = hidden_states.shape
        query_shape = (bsz, seq_len, self.num_heads, self.qk_head_dim)
        key_shape = (
            bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
        )
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1,
        )
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape)
        k_pass = k_pass.transpose(1, 2)
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1,
        )
        k_rot = k_rot.view(bsz, 1, seq_len, self.qk_rope_head_dim)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k_pass = torch.cat([past_key[..., : self.qk_nope_head_dim], k_pass], dim=2)
            k_rot = torch.cat([past_key[..., self.qk_nope_head_dim :], k_rot], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        return q_resid, q_pass, q_rot, k_pass, k_rot, value_states

    def _build_sparse_attention_mask(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        q_cos_sin: tuple[torch.Tensor, torch.Tensor],
        additive_mask: torch.Tensor,
        query_states: torch.Tensor,
        kv_seq_len: int,
        use_cache: bool,
    ) -> torch.Tensor:
        """Build the official DSA additive attention mask."""
        bsz, seq_len, _ = hidden_states.shape
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            q_cos_sin,
            attention_mask=additive_mask[:, 0, :, :],
            use_cache=use_cache,
        )
        dsa_mask = torch.full(
            (bsz, seq_len, kv_seq_len),
            float("-inf"),
            device=query_states.device,
            dtype=query_states.dtype,
        )
        dsa_mask.scatter_(-1, topk_indices, 0.0)
        return dsa_mask.unsqueeze(1) + additive_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Run official GLM-MoE-DSA MLA attention."""
        del kwargs
        bsz, seq_len, _ = hidden_states.shape
        q_resid, q_pass, q_rot, k_pass, k_rot, value_states = self._project_qkv(
            hidden_states, past_key_value,
        )
        kv_seq_len = value_states.shape[2]
        position_ids = _prepare_position_ids(
            position_ids,
            bsz,
            seq_len,
            kv_seq_len - seq_len,
            hidden_states.device,
        )
        key_position_ids = _infer_key_position_ids(position_ids, kv_seq_len)
        q_cos, q_sin = _rotary_cos_sin(
            self.rotary_emb, hidden_states, position_ids,
        )
        k_cos, k_sin = _rotary_cos_sin(
            self.rotary_emb, hidden_states, key_position_ids,
        )
        q_rot, _ = apply_rotary_pos_emb(q_rot, q_rot, q_cos, q_sin)
        k_rot, _ = apply_rotary_pos_emb(k_rot, k_rot, k_cos, k_sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)
        present_key_value = (key_states, value_states)

        additive_mask = _prepare_additive_attention_mask(
            attention_mask,
            batch_size=bsz,
            query_len=seq_len,
            key_len=kv_seq_len,
            dtype=query_states.dtype,
            device=query_states.device,
        )
        if additive_mask is None:
            additive_mask = torch.triu(
                torch.full(
                    (1, 1, seq_len, kv_seq_len),
                    float("-inf"),
                    device=query_states.device,
                    dtype=query_states.dtype,
                ),
                diagonal=1 + kv_seq_len - seq_len,
            )
        attention_mask = self._build_sparse_attention_mask(
            hidden_states,
            q_resid,
            (q_cos, q_sin),
            additive_mask,
            query_states,
            kv_seq_len,
            use_cache,
        )

        attn_output = self.attention_core(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask=attention_mask,
        )
        output = self.o_proj(attn_output.reshape(bsz, seq_len, -1))
        if use_cache:
            return output, present_key_value
        return output


__all__ = [
    "GLM5AttentionCore",
    "GLM5GQAAttention",
    "GLM5MLAAttention",
    "GLM5OfficialMLAAttention",
]
