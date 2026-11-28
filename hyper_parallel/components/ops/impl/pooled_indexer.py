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
"""Whole-block causal selection; pooling and differentiable KL remain separate."""
from __future__ import annotations

from typing import Protocol

import torch

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec


class PooledIndexerFn(Protocol):
    """One document: Q [S,H,D], pooled K [S//P,D], optional weights [S,H].

    Return INT32 token IDs [S,token_budget+P-1], -1 padding. Complete selected
    blocks occupy the first token_budget slots; the visible incomplete tail is
    appended at that fixed offset. Block order is unspecified across backends.
    Routes are discrete, without autograd; projected indexer inputs train via KL.
    """

    def __call__(self, query: torch.Tensor, pooled_key: torch.Tensor, weights: torch.Tensor | None,
                 pool_size: int, token_budget: int, query_chunk_size: int) -> torch.Tensor:
        """Select complete blocks by sum_h(weights * relu(QK/sqrt(D)))."""


POOLED_INDEXER = OpSpec[PooledIndexerFn](config_key="pooled_indexer")


def validate_inputs(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor | None, pool_size: int,
    token_budget: int, query_chunk_size: int,
) -> None:
    """Check a complete, unpadded document before selecting an implementation."""
    if query.ndim != 3 or key.ndim != 2 or min(query.shape) < 1 or key.shape[-1] != query.shape[-1]:
        raise ValueError("pooled indexer requires nonempty SHD query and KD pooled key")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in (pool_size, token_budget, query_chunk_size)):
        raise ValueError("pool, token budget and query chunk sizes must be positive integers")
    if token_budget % pool_size or key.shape[0] != query.shape[0] // pool_size:
        raise ValueError("token budget must contain whole blocks and key must contain all complete pools")
    if key.device != query.device or key.dtype != query.dtype or not query.is_floating_point():
        raise ValueError("query and pooled key must share floating dtype and device")
    if weights is not None and (weights.shape != query.shape[:2] or weights.device != query.device):
        raise ValueError("indexer head weights must have shape SH on the query device")


def cann_rejection(
    query: torch.Tensor, key: torch.Tensor, pool_size: int, token_budget: int,
) -> str | None:
    """Match the unquantized PoolKeyIndexer shape and token-budget contract."""
    if query.dtype not in (torch.float16, torch.bfloat16) or query.shape[-1] != 128:
        return "CANN pooled indexer requires FP16/BF16 and D128"
    if query.shape[1] not in (4, 8, 16, 32, 64) or not 1 <= pool_size <= 128:
        return "CANN pooled indexer requires 4/8/16/32/64 heads and pool size 1..128"
    if key.shape[0] > token_budget // pool_size and not (
            token_budget <= 2048 or token_budget in (3072, 4096, 5120, 6144, 7168, 8192)):
        return "CANN sparse selection requires a supported exact token budget; padding changes TopK"
    return None


def native_pooled_indexer(
    query: torch.Tensor, pooled_key: torch.Tensor, weights: torch.Tensor | None,
    pool_size: int, token_budget: int, query_chunk_size: int,
) -> torch.Tensor:
    """Causal whole-block TopK plus the visible incomplete tail; -1 marks padding.

    Tied scores prefer the smaller block index, making reference replay stable.
    The output width is token_budget + pool_size - 1 even for short documents.
    """
    length = query.shape[0]
    result = torch.full((length, token_budget + pool_size - 1), -1, device=query.device, dtype=torch.int32)
    offsets = torch.arange(pool_size, device=query.device)
    blocks = torch.arange(pooled_key.shape[0], device=query.device)
    selected_count = min(token_budget // pool_size, pooled_key.shape[0])
    with torch.no_grad():
        for start in range(0, length, query_chunk_size):
            end = min(start + query_chunk_size, length)
            positions = torch.arange(start, end, device=query.device)
            complete = (positions + 1) // pool_size
            if selected_count:
                scores = torch.einsum("qhd,nd->qhn", query[start:end].float(), pooled_key.float())
                scores = (scores * query.shape[-1] ** -0.5).relu()
                if weights is not None:
                    scores = scores * weights[start:end].float().unsqueeze(-1)
                scores = scores.sum(dim=1)
                scores = scores.masked_fill(blocks[None, :] >= complete[:, None], float("-inf"))
                selected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :selected_count]
                selected_tokens = selected.unsqueeze(-1) * pool_size + offsets
                selected_tokens = selected_tokens.masked_fill(selected.unsqueeze(-1) >= complete[:, None, None], -1)
                result[start:end, :selected_count * pool_size] = selected_tokens.flatten(-2).to(torch.int32)
            if pool_size > 1:
                tail = complete[:, None] * pool_size + offsets[:-1]
                tail = tail.masked_fill(tail > positions[:, None], -1)
                result[start:end, token_budget:] = tail.to(torch.int32)
    return result


def _load_native():
    return native_pooled_indexer


def _load_cann():
    package = require_package("cann_ops_transformer.ops")
    require_symbol(package, "pool_key_indexer")

    @torch.no_grad()
    def _operation(query, pooled_key, weights, pool_size, token_budget, query_chunk_size):
        del query_chunk_size
        length, heads, _ = query.shape
        positions = torch.arange(length, device=query.device)
        complete = (positions + 1) // pool_size
        # No TopK is necessary while every visible complete block fits the budget.
        prefix = min(length, token_budget + pool_size - 1)
        result = torch.full((length, token_budget + pool_size - 1), -1,
                            device=query.device, dtype=torch.int32)
        tokens = torch.arange(token_budget, device=query.device, dtype=torch.int32)
        result[:prefix, :token_budget] = torch.where(tokens[None] < complete[:prefix, None] * pool_size,
                                                    tokens[None], -1)
        if prefix < length:
            tail_count = torch.full((1,), length % pool_size, dtype=torch.int64, device=query.device)
            head_weights = (query.new_ones((length-prefix, heads)) if weights is None
                            else weights[prefix:].to(query.dtype))
            # PoolKeyIndexer includes 1/sqrt(D) and expands selected blocks to
            # original token IDs. Right-aligned causality preserves this suffix's positions.
            selected, _ = package.pool_key_indexer(
                query[prefix:][None].contiguous(), pooled_key[None, :, None].contiguous(),
                head_weights[None].contiguous(), tail_count, topk=token_budget, pool_size=pool_size,
                layout_q="BSND", layout_k="BSND", mask_mode=3, sparse_indices_mode=0)
            result[prefix:, :token_budget] = selected[0, :, :token_budget]
        if pool_size > 1:
            # The kernel appends the DOCUMENT's final residual block. Training
            # also needs the visible incomplete block at EACH query position.
            offsets = torch.arange(pool_size-1, device=query.device)
            tail = complete[:, None] * pool_size + offsets
            result[:, token_budget:] = tail.masked_fill(tail > positions[:, None], -1).to(torch.int32)
        return result

    return _operation


def register_pooled_indexer(registry: OpRegistry) -> None:
    """Keep explicit CANN selection and an independent eager routing reference."""
    registry.register_family("pooled_indexer", specs=(POOLED_INDEXER,))
    registry.register_impl(POOLED_INDEXER, name="native", devices=("cpu", "cuda", "npu"), load=_load_native)
    registry.register_impl(POOLED_INDEXER, name="cann", devices=("npu",), load=_load_cann)
