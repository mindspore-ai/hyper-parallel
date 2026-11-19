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

"""Deterministic MoonEP capacity-based assignment planning without device dependencies.

This implements load redistribution with redundant expert execution. It does
not reproduce the native MoonEP VM layout or claim its communication costs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssignmentSegment:
    """Contiguous range within one source rank's single-expert assignment bucket."""

    source: int
    expert: int
    destination: int
    offset: int
    count: int


@dataclass(frozen=True)
class CollectivePlan:
    """Immutable per-call plan; all ranks construct it from identical counts."""

    counts: tuple[tuple[int, ...], ...]
    segments: tuple[AssignmentSegment, ...]
    execution_counts: tuple[int, ...]
    experts_per_rank: int

    def experts_by_rank(self) -> tuple[tuple[int, ...], ...]:
        """Collect active expert IDs for all ranks in one segment traversal."""
        active: list[set[int]] = [set() for _ in self.counts]
        for segment in self.segments:
            active[segment.destination].add(segment.expert)
        return tuple(tuple(sorted(experts)) for experts in active)

    def experts_on(self, rank: int) -> tuple[int, ...]:
        """Return active expert IDs on an execution rank, in global order."""
        return tuple(sorted({s.expert for s in self.segments if s.destination == rank}))

    def splits(self, rank: int) -> tuple[list[int], list[int]]:
        """Return token send/receive counts in process-group rank order."""
        size = len(self.counts)
        send, receive = [0] * size, [0] * size
        for segment in self.segments:
            if segment.source == rank:
                send[segment.destination] += segment.count
            if segment.destination == rank:
                receive[segment.source] += segment.count
        return send, receive


def build_collective_plan(counts: list[list[int]]) -> CollectivePlan:
    """Balance assignments by moving the busiest owner's largest experts first.

    Args:
        counts: Nonnegative integer counts shaped [EP size, global experts].

    Returns:
        Plan with execution loads differing by at most one. Expert numbering
        uses contiguous blocks in process-group rank order. Ties use rank ID.

    Raises:
        ValueError: If counts are not rectangular, nonnegative integers or the
            number of experts is not divisible by the EP size.
    """
    size = len(counts)
    if not size or not counts[0] or len(counts[0]) % size:
        raise ValueError("counts require nonempty ranks and experts divisible by rank count")
    experts = len(counts[0])
    if any(len(row) != experts for row in counts):
        raise ValueError("counts must be rectangular")
    if any(type(value) is not int or value < 0 for row in counts for value in row):
        raise ValueError("counts must contain nonnegative integers")
    per_rank = experts // size
    expert_totals = [sum(column) for column in zip(*counts)]
    total = sum(expert_totals)
    targets = [total // size + int(rank < total % size) for rank in range(size)]
    allocation = [[0] * size for _ in range(experts)]
    loads = [0] * size
    for expert in range(experts):
        owner = expert // per_rank
        demand = expert_totals[expert]
        allocation[expert][owner] = demand
        loads[owner] += demand
    # Match the Ascend host planner when all capacities equal S*K. Unequal
    # input lengths extend that policy to floor/ceil-balanced capacities.
    while loads != targets:
        home = max(range(size), key=lambda rank: loads[rank] - targets[rank])
        destination = max(range(size), key=lambda rank: targets[rank] - loads[rank])
        move = min(loads[home] - targets[home], targets[destination] - loads[destination])
        while move:
            expert = max(
                range(home * per_rank, (home + 1) * per_rank),
                key=lambda index: allocation[index][home],
            )
            amount = min(move, allocation[expert][home])
            allocation[expert][home] -= amount
            allocation[expert][destination] += amount
            loads[home] -= amount
            loads[destination] += amount
            move -= amount
    allocations = [[(rank, amount) for rank, amount in enumerate(row) if amount] for row in allocation]
    segments = []
    for expert, allocation in enumerate(allocations):
        source, consumed = 0, 0
        for destination, demand in allocation:
            while demand:
                available = counts[source][expert] - consumed
                amount = min(demand, available)
                if amount:
                    segments.append(AssignmentSegment(source, expert, destination, consumed, amount))
                demand -= amount
                consumed += amount
                if consumed == counts[source][expert]:
                    source += 1
                    consumed = 0
    return CollectivePlan(tuple(tuple(row) for row in counts), tuple(segments), tuple(targets), per_rank)
