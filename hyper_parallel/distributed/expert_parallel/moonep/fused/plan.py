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
"""Compile MoonEP placement into real ranges without a virtual expert dimension."""

from __future__ import annotations

import numpy as np
import torch

from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge


def _checked_tables(counts: torch.Tensor, allocation: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Validate small host tables before performing fixed-width prefix arithmetic."""
    for name, value in (("counts", counts), ("allocation", allocation)):
        if value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 2:
            raise ValueError(f"{name} must be a CPU int64 matrix")
        if bool((value < 0).any()):
            raise ValueError(f"{name} must contain nonnegative counts")
    ranks, experts = counts.shape
    if ranks < 1 or experts < 1 or allocation.shape != (experts, ranks):
        raise ValueError("counts [R,E] and allocation [E,R] must have positive matching dimensions")
    # Python integers detect overflow before NumPy cumsum can silently wrap.
    source_total = sum(sum(row) for row in counts.tolist())
    target_total = sum(sum(row) for row in allocation.tolist())
    if max(source_total, target_total) > np.iinfo(np.int64).max:
        raise ValueError("route row offsets exceed the signed int64 ABI")
    source, target = counts.numpy(), allocation.numpy()
    if not np.array_equal(source.sum(axis=0), target.sum(axis=1)):
        raise ValueError("placement must preserve every expert's selection count")
    return source, target


def _compile_rank(source: np.ndarray, target: np.ndarray, rank: int) -> RouteDescriptor:
    """Intersect only edges incident to one rank, using the global integer prefixes."""
    source_end, target_end = source.cumsum(axis=0), target.cumsum(axis=1)
    source_begin, target_begin = source_end - source, target_end - target
    source_offsets = source.cumsum(axis=1) - source
    target_offsets = target.cumsum(axis=0) - target
    send_begin = np.maximum(source_begin[rank, :, None], target_begin)
    sent = np.maximum(np.minimum(source_end[rank, :, None], target_end) - send_begin, 0)
    receive_begin = np.maximum(source_begin, target_begin[None, :, rank])
    received = np.maximum(np.minimum(source_end, target_end[None, :, rank]) - receive_begin, 0)
    outgoing, incoming = [], []
    for expert, dst in zip(*np.nonzero(sent)):
        begin = int(send_begin[expert, dst])
        outgoing.append(RouteEdge(rank, int(dst), int(expert),
                                  int(source_offsets[rank, expert]) + begin - int(source_begin[rank, expert]),
                                  int(target_offsets[expert, dst]) + begin - int(target_begin[expert, dst]),
                                  int(sent[expert, dst])))
    for src, expert in zip(*np.nonzero(received)):
        begin = int(receive_begin[src, expert])
        incoming.append(RouteEdge(int(src), rank, int(expert),
                                  int(source_offsets[src, expert]) + begin - int(source_begin[src, expert]),
                                  int(target_offsets[expert, rank]) + begin - int(target_begin[expert, rank]),
                                  int(received[src, expert])))
    active = np.flatnonzero(target[:, rank])
    return RouteDescriptor(rank, source.shape[0], int(source[rank].sum()), int(target[:, rank].sum()),
                           tuple(int(expert) for expert in active),
                           tuple(int(end) for end in target[active, rank].cumsum()),
                           tuple(outgoing), tuple(incoming))


def compile_local_route(counts: torch.Tensor, allocation: torch.Tensor, rank: int) -> RouteDescriptor:
    """Compile one rank's actual incoming and outgoing selection intervals.

    Args:
        counts: CPU int64 [source PE, global expert] histogram.
        allocation: CPU int64 [global expert, target PE] MoonEP allocation.
        rank: PE within the admitted expert group.

    Returns:
        The immutable local descriptor, with the same source ordering as the
        complete route collection. Remote-only edges are never materialized.

    Raises:
        ValueError: Invalid tables, lost selections, or a rank outside the group.
    """
    source, target = _checked_tables(counts, allocation)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < source.shape[0]:
        raise ValueError("route rank must identify a PE in the expert group")
    return _compile_rank(source, target, rank)


def compile_sparse_route(counts: torch.Tensor, allocation: torch.Tensor) -> tuple[RouteDescriptor, ...]:
    """Compile the same source-ordered interval intersections on every rank.

    Args:
        counts: CPU int64 [source PE, global expert] histogram.
        allocation: CPU int64 [global expert, target PE] MoonEP allocation.

    Returns:
        Immutable per-rank descriptors containing only nonempty ranges. Source
        offsets address stable expert-sorted selections, not expanded payloads.

    Raises:
        ValueError: Invalid dtype, shape, negative counts, overflow or lost rows.

    Example:
        Four e0 selections split two per rank produce two actual ranges, even
        if the caller's replica capacity could hold many more experts.
    """
    source, target = _checked_tables(counts, allocation)
    return tuple(_compile_rank(source, target, rank) for rank in range(source.shape[0]))
