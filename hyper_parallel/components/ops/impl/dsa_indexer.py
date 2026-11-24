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
"""Causal TopK selection for the DeepSeek V4.1 DSA Indexer."""

from __future__ import annotations

from functools import partial
from typing import Protocol

import torch

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer as fused_packed_indexer
from hyper_parallel.components.functional.lightning_indexer import (
    lightning_indexer, sort_key_indices, validate_indexer_tensors,
)

from .._dependency import load_cann, require_package, require_symbol
from .._dsa_layout import DsaQueryLayout, ReduceSum, packed_segments
from ..registry import OpRegistry
from ..spec import OpSpec


class DsaIndexerFn(Protocol):
    """Produce sorted causal TopK key IDs with -1 marking unused slots."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
        layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
        reduce_sum: ReduceSum | None = None,
    ) -> torch.Tensor:
        """Return int32 key IDs for each query."""


DSA_INDEXER = OpSpec[DsaIndexerFn](config_key="dsa_indexer", variant="compressed_causal_topk")


class PackedDsaIndexerFn(Protocol):
    """Select document-local key IDs and retain the flattened indexer states."""

    def __call__(
        self, x: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
        query_ends: torch.Tensor, key_ends: torch.Tensor, sparse_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return right-aligned causal TopK with -1 padding, followed by TND states and TN weights."""


DSA_PACKED_INDEXER = OpSpec[PackedDsaIndexerFn](config_key="dsa_indexer", variant="packed_causal_topk")


def _native_packed_indexer(
    x: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
    query_ends: torch.Tensor, key_ends: torch.Tensor, sparse_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select causal document-local keys and return flattened indexer inputs."""
    query, key, weights = (tensor.flatten(0, 1) for tensor in (x, key, weights))
    indices = []
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        for qs, qe, ks, ke in packed_segments(query_ends, key_ends):
            keys = key[ks:ke, 0].float()
            count = min(sparse_count, keys.shape[0])
            key_pos = torch.arange(keys.shape[0], device=x.device)
            for start in range(qs, qe, 128):
                end = min(start + 128, qe)
                q_pos = torch.arange(start, end, device=x.device) - qe + ke - ks
                scores = (query[start:end].float() @ keys.T).relu()
                scores = (scores * weights[start:end].float().unsqueeze(-1)).sum(dim=1)
                scores = scores.masked_fill(key_pos[None, :] > q_pos[:, None], -torch.inf)
                values, ids = scores.topk(count, dim=-1)
                ids = ids.masked_fill(~values.isfinite(), -1).to(torch.int32)
                indices.append(torch.nn.functional.pad(ids, (0, sparse_count - count), value=-1))
    return torch.cat(indices).unsqueeze(1), query, key, weights


def _load_packed_indexer() -> PackedDsaIndexerFn:
    kernels = require_package("cann_ops_transformer.ops")
    require_symbol(kernels, "lightning_indexer")
    require_symbol(kernels, "lightning_indexer_metadata")
    return fused_packed_indexer


def _validate_indexer_inputs(
    query: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, layout: DsaQueryLayout, chunk_size: int,
) -> None:
    """Validate compressed indexer tensor layouts and query chunk size."""
    compress_ratio = layout.compress_ratio
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {chunk_size}")
    if query.ndim != 4:
        raise ValueError(f"Indexer query must have shape [B, Q, H, D], got {tuple(query.shape)}")
    batch_size, _, _, head_dim = query.shape
    if key.ndim != 3 or key.shape[::2] != (batch_size, head_dim):
        raise ValueError(
            "compressed index key must have shape [batch, compressed_sequence, head_dim]"
        )
    if merge_weight.shape != query.shape[:3]:
        raise ValueError(
            f"merge_weight must have shape {tuple(query.shape[:3])}, got {tuple(merge_weight.shape)}"
        )
    if layout.minimum_key_indices is not None and layout.minimum_key_indices.shape != query.shape[:2]:
        raise ValueError("minimum_key_indices must have shape [batch, query]")


@torch.no_grad()
def native_dsa_indexer(
    x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
    layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None,
) -> torch.Tensor:
    """Reference ratio-aware causal TopK over a compressed key bank."""
    query = x
    _validate_indexer_inputs(query, key, merge_weight, layout, query_chunk_size)
    batch_size, sequence_length = query.shape[:2]
    compressed_length = key.shape[1]
    top_k = min(sparse_count, compressed_length)
    if top_k <= 0 or sequence_length == 0:
        return torch.empty(
            batch_size,
            sequence_length,
            max(top_k, 0),
            dtype=torch.int32,
            device=query.device,
        )

    visible_keys, minimum_key_indices = layout.compressed_bounds(query)
    key_fp32 = key.float().transpose(1, 2).unsqueeze(1)
    key_positions = torch.arange(compressed_length, device=query.device).view(1, 1, -1)
    selected_chunks = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        # SAC may retain the matmul result; do not mutate its cached tensor.
        scores = torch.matmul(query[:, start:end].float(), key_fp32).relu()
        scores = (scores * merge_weight[:, start:end].float().unsqueeze(-1)).sum(dim=2)
        if reduce_sum is not None:
            scores = reduce_sum(scores)
        # Bounds are [1, S]; slice the query axis so the mask covers this chunk.
        visible = visible_keys[:, start:end].unsqueeze(-1)
        scores.masked_fill_(key_positions >= visible, float("-inf"))
        if minimum_key_indices is not None:
            minimum = minimum_key_indices[:, start:end].unsqueeze(-1)
            scores.masked_fill_(key_positions < minimum, float("-inf"))
        top = scores.topk(top_k, dim=-1, sorted=False)
        indices = top.indices.masked_fill(~torch.isfinite(top.values), compressed_length)
        indices = sort_key_indices(indices, compressed_length)
        indices = indices.masked_fill(indices == compressed_length, -1)
        selected_chunks.append(indices.to(torch.int32))
    return torch.cat(selected_chunks, dim=1)


def indexer_native_only(x: torch.Tensor, layout: DsaQueryLayout, reduce_sum: ReduceSum | None) -> str | None:
    """Explain why the LI V2 candidate cannot represent this call."""
    if (x.dtype != torch.bfloat16 or x.shape[-1] != 128 or layout.segments is None or reduce_sum is not None):
        return "LI V2 requires BF16 D=128, document segments and no TP score reduction"
    return None


@torch.no_grad()
def _cann_indexer(
    x: torch.Tensor, key: torch.Tensor, merge_weight: torch.Tensor, *,
    layout: DsaQueryLayout, sparse_count: int, query_chunk_size: int = 256,
    reduce_sum: ReduceSum | None = None,
) -> torch.Tensor:
    del reduce_sum
    _validate_indexer_inputs(x, key, merge_weight, layout, query_chunk_size)
    validate_indexer_tensors(x, key, merge_weight)
    return lightning_indexer(x, key, merge_weight, layout.compress_ratio, min(sparse_count, key.shape[1]),
                             layout.segments)


def register_dsa_indexer(registry: OpRegistry) -> None:
    """Register the compressed causal TopK contract and its implementations."""
    registry.register_family("dsa_indexer", specs=(DSA_INDEXER, DSA_PACKED_INDEXER))
    registry.register_impl(
        DSA_INDEXER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_dsa_indexer,
    )
    registry.register_impl(
        DSA_INDEXER, name="cann", devices=("npu",), load=partial(load_cann, _cann_indexer, "lightning_indexer"),
    )
    registry.register_impl(
        DSA_PACKED_INDEXER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native_packed_indexer,
    )
    registry.register_impl(
        DSA_PACKED_INDEXER, name="cann", devices=("npu",),
        load=_load_packed_indexer,
    )
