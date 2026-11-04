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
"""Immutable, capacity-independent metadata for routed expert kernels."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class RouteEdge:
    """A nonempty range between source-expert-major and destination-expert-major rows.

    Attributes:
        source: Source PE in the explicit communication root.
        target: Destination PE in that root.
        expert: Original global expert identity, unaffected by replication.
        source_row: First row in the stable expert-sorted source selection array.
        target_row: First row in the destination's compact expert-major storage.
        rows: Number of consecutive selections, always positive.
    """

    source: int
    target: int
    expert: int
    source_row: int
    target_row: int
    rows: int

    def blocks(self, block_rows: int) -> tuple["RouteEdge", ...]:
        """Split this edge without creating an empty tail or padding selections."""
        if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
            raise ValueError("block_rows must be a positive integer")
        return tuple(RouteEdge(self.source, self.target, self.expert,
                               self.source_row + start, self.target_row + start,
                               min(block_rows, self.rows - start))
                     for start in range(0, self.rows, block_rows))


@dataclass(frozen=True)
class RouteDescriptor:
    """One rank's sparse route; empty experts have no descriptor or compute range."""

    rank: int
    ranks: int
    source_rows: int
    received_rows: int
    active_experts: tuple[int, ...]
    boundaries: tuple[int, ...]
    outgoing: tuple[RouteEdge, ...]
    incoming: tuple[RouteEdge, ...]

    @cached_property
    def _value_hash(self) -> int:
        """Cache the immutable routing value without adding a dataclass field."""
        return hash((self.rank, self.ranks, self.source_rows, self.received_rows,
                     self.active_experts, self.boundaries, self.outgoing, self.incoming))

    def __hash__(self) -> int:
        """Reuse the structural hash for this execution's repeated metadata lookups."""
        return self._value_hash

    def __getstate__(self) -> dict:
        """Keep the derived cache out of serialized routing state."""
        return {name: value for name, value in self.__dict__.items() if name != "_value_hash"}

    def _validate_edge(self, edge: RouteEdge, incoming: bool, expert_ranges: dict[int, tuple[int, int]]) -> None:
        """Check one edge before its addressed range contributes to coverage."""
        if (not 0 <= edge.source < self.ranks or not 0 <= edge.target < self.ranks
                or edge.expert < 0 or min(edge.source_row, edge.target_row) < 0 or edge.rows < 1):
            raise ValueError("route edges require valid PEs and nonempty nonnegative ranges")
        if (edge.target if incoming else edge.source) != self.rank:
            raise ValueError("route edge does not belong to this rank")
        if incoming:
            if edge.expert not in expert_ranges:
                raise ValueError("incoming edge does not name an active expert")
            begin, end = expert_ranges[edge.expert]
            if edge.target_row < begin or edge.target_row + edge.rows > end:
                raise ValueError("incoming edge crosses its expert's compact range")

    def _validate_edge_coverage(self, incoming: bool, edges: tuple[RouteEdge, ...], extent: int,
                                expert_ranges: dict[int, tuple[int, int]]) -> None:
        """Validate one direction and require each declared row exactly once."""
        ranges = []
        for edge in edges:
            self._validate_edge(edge, incoming, expert_ranges)
            start = edge.target_row if incoming else edge.source_row
            ranges.append((start, start + edge.rows))
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError("route ranges overlap or omit selections")
            cursor = end
        if cursor != extent:
            raise ValueError("route ranges do not cover the declared row extent")

    def validate(self) -> None:
        """Reject malformed offsets and overlapping or missing local selections."""
        if not 0 <= self.rank < self.ranks or min(self.source_rows, self.received_rows) < 0:
            raise ValueError("invalid route rank or row extent")
        if len(self.active_experts) != len(self.boundaries) or any(expert < 0 for expert in self.active_experts):
            raise ValueError("active expert identities and boundaries must match")
        if tuple(sorted(set(self.active_experts))) != self.active_experts:
            raise ValueError("active expert identities must be strictly increasing")
        if any(end <= start for start, end in zip((0,) + self.boundaries, self.boundaries)):
            raise ValueError("active expert ranges must be nonempty and increasing")
        if (self.boundaries[-1] if self.boundaries else 0) != self.received_rows:
            raise ValueError("expert boundaries must cover all received rows")
        expert_ranges = {expert: (begin, end) for expert, begin, end
                         in zip(self.active_experts, (0,) + self.boundaries[:-1], self.boundaries)}
        for incoming, edges, extent in ((False, self.outgoing, self.source_rows),
                                         (True, self.incoming, self.received_rows)):
            self._validate_edge_coverage(incoming, edges, extent, expert_ranges)
