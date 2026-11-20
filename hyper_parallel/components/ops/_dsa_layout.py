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
"""Shared logical document coordinates for the DSA operator families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


Segment = tuple[int, int, int, int, int]
ReduceSum = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DsaQueryLayout:
    """Logical document geometry in raw-query and compressed-key coordinates.

    Each segment stores local query start/end, document key start/prefix end,
    and the trailing raw tokens after the last complete block. Tensor starts
    and bounds represent the same geometry for graph packing and eager reuse.
    """

    compress_ratio: int = 1
    query_offset: int = 0
    segments: tuple[Segment, ...] | None = None
    segment_starts: torch.Tensor | None = None
    minimum_key_indices: torch.Tensor | None = None
    causal_upper: torch.Tensor | None = None

    def raw_document_starts(self) -> tuple[tuple[int, int], ...]:
        """Derive the raw token origin of each query segment."""
        ratio = max(1, self.compress_ratio)
        return tuple(
            (start, self.query_offset + end - (key_end - key_start) * ratio - residual)
            for start, end, key_start, key_end, residual in self.segments or ()
        )

    def compressed_bounds(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return each query's valid range in the compressed key bank."""
        return _causal_key_bounds(
            query, self.compress_ratio, self.query_offset, self.minimum_key_indices, self.segments, self.causal_upper,
        )


def _causal_key_bounds(
        query: torch.Tensor, ratio: int, query_offset: int,
        minimum_key_indices: torch.Tensor | None, segments: tuple | None,
        causal_upper: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Derive visibility from original query positions within each independently compressed document.

    The returned upper bound always has shape ``[1, queries]`` (shared across
    the batch), so chunked consumers slice it along the sequence dimension.
    ``causal_upper`` is the packed per-query exclusive bound shared by eager and
    graph execution; Python ``segments`` remain available for eager callers and
    the fused native kernels.
    """
    batch_size, length = query.shape[:2]
    if causal_upper is not None:
        return causal_upper, minimum_key_indices
    if segments is None:
        positions = torch.arange(query_offset, query_offset + length, device=query.device)
        return ((positions + 1) // ratio).unsqueeze(0), minimum_key_indices
    upper = torch.empty(length, device=query.device, dtype=torch.long)
    lower = torch.empty_like(upper)
    for begin, end, key_start, key_end, residual in segments:
        first = ratio * key_end + residual - (end - begin)
        positions = torch.arange(first, first + end - begin, device=query.device)
        upper[begin:end] = (positions + 1) // ratio
        lower[begin:end] = key_start
    return upper.unsqueeze(0), lower.unsqueeze(0).expand(batch_size, -1)


def packed_segments(
    query_ends: torch.Tensor, key_ends: torch.Tensor,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return query/key ranges for packed documents from cumulative ends."""
    q_ends, k_ends = query_ends.tolist(), key_ends.tolist()
    if len(q_ends) != len(k_ends):
        raise ValueError("Packed DSA query and key lengths must describe the same documents")
    return tuple(zip([0, *q_ends[:-1]], q_ends, [0, *k_ends[:-1]], k_ends))


def packed_causal_mask(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Return the right-aligned causal mask for a single DSA document."""
    q_pos = torch.arange(query.shape[0], device=query.device) + key.shape[0] - query.shape[0]
    return torch.arange(key.shape[0], device=query.device)[None, :] <= q_pos[:, None]


def selected_packed_keys(
    key: torch.Tensor, indices: torch.Tensor, *, causal_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather document-local key IDs in T,N,K,D layout and return validity."""
    valid = (indices >= 0) & (indices < key.shape[0]) & (indices <= causal_end[:, None, None])
    positions = indices.clamp(0, max(0, key.shape[0] - 1)).long()
    heads = torch.arange(key.shape[1], device=key.device)[None, :, None]
    return key[positions, heads], valid


def selected_packed_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Dot grouped query heads and selected keys in T,N,G,K layout."""
    grouped = query.unflatten(1, (keys.shape[1], -1))
    return torch.einsum("tngd,tnkd->tngk", grouped.float(), keys.float())
