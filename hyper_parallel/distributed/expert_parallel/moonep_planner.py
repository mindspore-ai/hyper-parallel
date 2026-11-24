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

"""MoonEP collective placement compiled from route histograms.

Adapted from HyperParallel PR #1485. The whole-deficit fallback matches the
MoonEP official planning reference; the partial-fill pass minimizes moved rows
when it fits the requested replica capacity. Planning is eager CPU metadata;
no token activation or expert parameter is copied to the CPU.
"""

from __future__ import annotations

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

    allocation = np.zeros((totals.size, ranks), dtype=np.int64)
    for owner in range(ranks):
        remaining = home[owner].copy()
        allocation[owner * home.shape[1]:(owner + 1) * home.shape[1], owner] = remaining
        experts = np.argsort(-remaining, kind="stable")
        position = 0
        for destination in np.argsort(-quotas[owner], kind="stable"):
            quota = quotas[owner, destination]
            while quota > 0:
                expert = experts[position]
                amount = min(remaining[expert], quota)
                logical_id = owner * home.shape[1] + expert
                allocation[logical_id, destination] += amount
                allocation[logical_id, owner] -= amount
                remaining[expert] -= amount
                quota -= amount
                if remaining[expert] == 0:
                    position += 1
    return allocation


def select_allocation(counts: np.ndarray, replica_capacity: int | None = None) -> np.ndarray:
    """Apply the moonep_with_fsdp partial-fill policy with the whole-deficit fallback.

    Args:
        counts: Global route counts in [source rank, logical expert] order.
        replica_capacity: Maximum distinct remote experts per rank; defaults to E/EP.

    Returns:
        Real row allocation in [logical expert, execution rank] order.
    """
    ranks, experts = counts.shape
    local = experts // ranks
    capacity = local if replica_capacity is None else replica_capacity
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


def compile_plan(counts: np.ndarray, allocation: np.ndarray, ids: torch.Tensor, rank: int) -> dict:
    """Compile source and receiver permutations without copying token IDs to the host.

    Args:
        counts: Global counts [source rank, expert].
        allocation: Conserved route allocation [expert, execution rank].
        ids: This source's prepared expert-major slot IDs.
        rank: Source rank in the existing EP group.

    Returns:
        Invocation-local split lists, compute counts and permutation tensors.
    """
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
    home_experts = expert_ids[owners == rank]
    active = np.concatenate((home_experts, received_experts))
    boundaries = allocation[active, rank].cumsum()

    # Sender packets are destination-major, with ascending expert IDs per peer.
    # Receiver packets are source-major; expand one offset per (expert, source) segment.
    recv_starts = (recv_rows.flatten().cumsum() - recv_rows.flatten()).reshape(recv_rows.shape)
    lengths = recv_rows[:, active].T.flatten()
    recv_offsets = recv_starts[:, active].T.flatten() - (lengths.cumsum() - lengths)
    offset = (source_ends - counts)[rank] - (counts[rank].cumsum() - counts[rank])
    rows = (offset, target_ends.flatten(), recv_offsets, lengths, send_experts, boundaries)
    metadata = torch.from_numpy(np.concatenate(rows)).to(ids.device)
    offset, ends, recv_offsets, lengths, send, boundaries = metadata.split([row.size for row in rows])

    positions = torch.arange(ids.numel(), device=ids.device)
    ordinals = positions + offset[ids]
    destinations = (ordinals[:, None] >= ends.view(experts, ranks)[ids]).sum(dim=1)
    # Inputs are already expert-major; stable destination sorting retains that order per peer.
    source_order = destinations.float().argsort(stable=True)
    source_inverse = torch.empty_like(source_order, dtype=torch.int32)
    source_inverse.scatter_(0, source_order, positions.int())
    received = sum(token_recv)
    receive_order = torch.arange(received, device=ids.device) + torch.repeat_interleave(
        recv_offsets, lengths, output_size=received,
    )
    return {
        "token_send": token_send, "token_recv": token_recv,
        "weight_send": weight_send, "weight_recv": weight_recv, "weight_exchange": bool(remote.any()),
        "send_experts": send, "source_order": source_order, "source_inverse": source_inverse,
        "receive_order": receive_order, "counts": torch.diff(boundaries, prepend=boundaries.new_zeros(1)),
    }
