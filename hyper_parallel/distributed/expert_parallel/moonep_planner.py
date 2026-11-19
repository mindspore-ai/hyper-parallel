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

"""MoonEP placement and collective metadata compiled from route histograms."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch


def _allocate(totals: np.ndarray, targets: np.ndarray, fill_deficit: bool) -> np.ndarray:
    """Split sender/receiver quotas across experts in descending load order."""
    ranks = targets.size
    home = totals.reshape(ranks, -1)
    balance = home.sum(axis=1) - targets
    quotas = np.zeros((ranks, ranks), dtype=np.int64)
    for _ in range(ranks - 1):
        source, target = balance.argmax(), balance.argmin()
        amount = -balance[target]
        if not fill_deficit:
            amount = min(balance[source], amount)
        quotas[source, target] += amount
        balance[source] -= amount
        balance[target] += amount

    experts = np.argsort(-home, axis=1, kind="stable")
    destinations = np.argsort(-quotas, axis=1, kind="stable")
    expert_rows = np.take_along_axis(home, experts, axis=1)
    dest_rows = np.take_along_axis(quotas, destinations, axis=1)
    expert_ends = expert_rows.cumsum(axis=1)[:, :, None]
    dest_ends = dest_rows.cumsum(axis=1)[:, None, :]
    moved = np.maximum(
        np.minimum(expert_ends, dest_ends)
        - np.maximum(expert_ends - expert_rows[:, :, None], dest_ends - dest_rows[:, None, :]),
        0,
    )
    by_expert = np.empty_like(moved)
    np.put_along_axis(by_expert, destinations[:, None, :], moved, axis=2)
    allocation = np.empty_like(moved)
    np.put_along_axis(allocation, experts[:, :, None], by_expert, axis=1)
    for rank in range(ranks):
        allocation[rank, :, rank] += home[rank] - allocation[rank].sum(axis=1)
    return allocation.reshape(totals.size, ranks)


def select_allocation(counts: np.ndarray, replica_capacity: int | None = None) -> np.ndarray:
    """Apply the moonep_with_fsdp partial-fill policy with the whole-deficit fallback.

    Args:
        counts: Global route counts in [source rank, logical expert] order.
        replica_capacity: Maximum distinct remote experts per rank; defaults to E/EP.

    Returns:
        Real row allocation in [logical expert, execution rank] order.
    """
    ranks, experts = counts.shape
    if ranks < 1 or experts < 1 or experts % ranks or np.any(counts < 0):
        raise ValueError("MoonEP requires nonnegative counts with E divisible by EP")
    local = experts // ranks
    capacity = local if replica_capacity is None else replica_capacity
    if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 0:
        raise ValueError("moonep_replica_capacity must be a nonnegative integer or None")
    totals = counts.sum(axis=0)
    total = totals.sum()
    targets = total // ranks + (np.arange(ranks) < total % ranks)
    allocation = _allocate(totals, targets, fill_deficit=False)
    remote = np.arange(experts)[:, None] // local != np.arange(ranks)
    minimum = np.maximum(totals.reshape(ranks, local).sum(axis=1) - targets, 0).sum()
    if (allocation[remote].sum() == minimum and np.all(allocation.sum(axis=0) == targets)
            and np.all(((allocation > 0) & remote).sum(axis=0) <= capacity)):
        return allocation
    allocation = _allocate(totals, targets, fill_deficit=True)
    valid = (not np.any(allocation < 0), np.array_equal(allocation.sum(axis=1), totals),
             np.array_equal(allocation.sum(axis=0), targets),
             np.all(((allocation > 0) & remote).sum(axis=0) <= capacity))
    if not all(valid):
        raise RuntimeError(f"MoonEP cannot balance routes within replica capacity {capacity}")
    return allocation


@dataclass(frozen=True)
class MoonEPPlan:
    """Routing and weight exchange metadata for one MoE forward."""

    token_send: list[int]
    token_recv: list[int]
    weight_send: list[int]
    weight_recv: list[int]
    weight_exchange: bool
    send_experts: torch.Tensor
    home_experts: torch.Tensor
    source_order: torch.Tensor
    source_inverse: torch.Tensor
    receive_order: torch.Tensor
    boundaries: torch.Tensor


def _stable_order(keys: torch.Tensor, buckets: int) -> torch.Tensor:
    size = keys.numel()
    if size * buckets < 2**24:
        positions = torch.arange(size, device=keys.device)
        return (keys * size + positions).float().argsort()
    return keys.argsort(stable=True)


def compile_plan(counts: np.ndarray, allocation: np.ndarray, ids: torch.Tensor, rank: int) -> MoonEPPlan:
    """Compile source and receiver permutations without copying token IDs to the host."""
    ranks, experts = counts.shape
    local = experts // ranks
    source_ends = counts.cumsum(axis=0)
    target_ends = allocation.cumsum(axis=1)
    overlap = np.maximum(
        np.minimum(source_ends[:, :, None], target_ends[None])
        - np.maximum((source_ends - counts)[:, :, None], (target_ends - allocation)[None]),
        0,
    )
    recv_rows = overlap[:, :, rank]
    token_send = overlap[rank].sum(axis=0).tolist()
    token_recv = recv_rows.sum(axis=1).tolist()

    expert_ids = np.arange(experts)
    owners = expert_ids // local
    remote = (allocation > 0) & (owners[:, None] != np.arange(ranks))
    weight_send = remote[owners == rank].sum(axis=0).tolist()
    weight_recv = remote[:, rank].reshape(ranks, local).sum(axis=1).tolist()
    send_experts = np.nonzero(remote[owners == rank].T)[1]
    received_experts = expert_ids[remote[:, rank]]
    home_experts = expert_ids[(owners == rank) & (allocation[:, rank] > 0)]
    active = np.concatenate((home_experts, received_experts))
    boundaries = allocation[active, rank].cumsum()

    # Sender packets are destination-major, with ascending expert IDs per peer.
    # Receiver packets are source-major; expand one offset per (expert, source) segment.
    recv_starts = (recv_rows.flatten().cumsum() - recv_rows.flatten()).reshape(recv_rows.shape)
    lengths = recv_rows[:, active].T.flatten()
    recv_offsets = recv_starts[:, active].T.flatten() - (lengths.cumsum() - lengths)
    offset = (source_ends - counts)[rank] - (counts[rank].cumsum() - counts[rank])
    rows = (offset, target_ends.flatten(), recv_offsets, lengths, send_experts, home_experts % local, boundaries)
    metadata = torch.from_numpy(np.concatenate(rows)).to(ids.device)
    offset, ends, recv_offsets, lengths, send, home, boundaries = metadata.split([row.size for row in rows])

    expert_order = _stable_order(ids, experts)
    sorted_ids = ids.index_select(0, expert_order)
    positions = torch.arange(ids.numel(), device=ids.device)
    ordinals = positions + offset[sorted_ids]
    destinations = (ordinals[:, None] >= ends.view(experts, ranks)[sorted_ids]).sum(dim=1)
    source_order = expert_order[_stable_order(destinations, ranks)]
    source_inverse = torch.empty_like(source_order, dtype=torch.int32)
    source_inverse.scatter_(0, source_order, positions.int())
    received = sum(token_recv)
    receive_order = torch.arange(received, device=ids.device) + torch.repeat_interleave(
        recv_offsets, lengths, output_size=received,
    )
    return MoonEPPlan(token_send, token_recv, weight_send, weight_recv, bool(remote.any()),
                      send, home, source_order, source_inverse, receive_order, boundaries)
