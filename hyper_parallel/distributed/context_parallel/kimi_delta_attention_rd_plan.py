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
"""Immutable reset-aware RD plans derived from packed document metadata."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache

from hyper_parallel.components.functional.linear_attention_packed import PackedDocumentInfo


@dataclass(frozen=True)
class RDScanRound:
    """One immutable exchange and its subsequent transfer-matrix consumer."""

    stage: int
    destination: int | None
    source: int | None
    save_matrix: bool


def build_scan_rounds(
    rank: int, size: int, distances: tuple[int, ...],
) -> tuple[RDScanRound, ...]:
    """Precompute dense inclusive-scan peers and transfer consumers for one rank.

    Args:
        rank: Rank in the state process group.
        size: Number of chronological ranks in the state process group.
        distances: Increasing powers of two used by the inclusive scan.
    """
    last_receive = rank.bit_length() - 1
    rounds = []
    for stage, distance in enumerate(distances):
        destination = rank + distance if rank + distance < size else None
        source = rank - distance if rank >= distance else None
        if destination is not None or source is not None:
            save_matrix = stage < last_receive
            rounds.append(RDScanRound(stage, destination, source, save_matrix))
    return tuple(rounds)


@dataclass(frozen=True)
class _RankInterval:
    """Rank span of the tail document and independent head continuation."""

    first: int
    last: int
    continues_previous: bool

    def peers(self, rank: int, distance: int) -> tuple[int | None, int | None]:
        """Return the exact live scan edge at one doubling distance.

        A document's last rank has no exit-state consumer. Earlier ranks reach
        a predecessor precisely when that predecessor belongs to this document.
        """
        destination = rank + distance if rank + distance < self.last else None
        source = rank - distance if rank < self.last and rank - distance >= self.first else None
        return destination, source

    def cache_count(self, rank: int) -> int:
        """Count powers of two that reach a preceding rank of this document."""
        return (rank - self.first).bit_length() if rank < self.last else 0


def _rank_interval(offsets: tuple[int, ...], local_length: int, rank: int) -> _RankInterval:
    """Find the tail document without treating a rank cut as a reset."""
    tail = bisect_left(offsets, (rank + 1) * local_length) - 1
    start = rank * local_length
    head = bisect_left(offsets, start)
    return _RankInterval(offsets[tail] // local_length, (offsets[tail + 1] - 1) // local_length,
                         offsets[head] != start)


def _interval_rounds(
    rank: int, distances: tuple[int, ...], interval: _RankInterval,
) -> tuple[RDScanRound, ...]:
    """Keep stage order and transfer lifetimes from the inclusive scan.

    Each effective receive caches the matrix used by its adjoint. Retain its
    composed output only when a subsequent receive consumes that output.
    """
    last_receive = interval.cache_count(rank) - 1
    rounds = []
    for stage, distance in enumerate(distances):
        destination, source = interval.peers(rank, distance)
        if destination is not None or source is not None:
            save_matrix = stage < last_receive
            rounds.append(RDScanRound(stage, destination, source, save_matrix))
    return tuple(rounds)


@dataclass(frozen=True)
class RDResetPlan:
    """An immutable local schedule derived from true document boundaries.

    The same plan is transposed by backward. Matrices are never inspected on
    device: only true document boundaries can remove an edge. Binding one rank
    retains its tail interval and rounds, without building the all-rank graph.
    """

    size: int
    distances: tuple[int, ...]
    offsets: tuple[int, ...]
    local_length: int
    local_rank: int | None
    local_rounds: tuple[RDScanRound, ...]
    _local_interval: _RankInterval | None

    def _interval(self, rank: int) -> _RankInterval:
        """Reuse local bounds or locate another rank for an audit query."""
        if rank == self.local_rank and self._local_interval is not None:
            return self._local_interval
        if not 0 <= rank < self.size:
            raise ValueError("RD execution-plan rank is outside the state process group")
        return _rank_interval(self.offsets, self.local_length, rank)

    def peers(self, stage: int, rank: int) -> tuple[int | None, int | None]:
        """Return forward destination/source, both in process-group order."""
        return self._interval(rank).peers(rank, self.distances[stage])

    def shift_peers(self, rank: int) -> tuple[int | None, int | None]:
        """Return the tail's consumer and the head's producer independently."""
        interval = self._interval(rank)
        return (rank + 1 if rank < interval.last else None,
                rank - 1 if interval.continues_previous else None)

    def is_transfer_active(self, rank: int) -> bool:
        """Whether a rank's entry can reach its exit without a document reset."""
        return rank > self._interval(rank).first

    def execution_rounds(self, rank: int) -> tuple[RDScanRound, ...]:
        """Reuse the caller's plan without materializing schedules for every peer."""
        if rank == self.local_rank:
            return self.local_rounds
        return _interval_rounds(rank, self.distances, self._interval(rank))

    def cache_count(self, rank: int) -> int:
        """Count invocation-owned pre-round transfers on one rank."""
        return self._interval(rank).cache_count(rank)


@lru_cache(maxsize=32)
def _validate_intervals(offsets: tuple[int, ...], local_length: int, size: int) -> None:
    """Share document validation across rank-specific plans with bounded storage."""
    if (size < 1 or local_length < 1 or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != size * local_length
            or any(left >= right for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("RD reset planning requires ordered documents covering equal state-rank intervals")


@lru_cache(maxsize=32)
def build_reset_plan(
    offsets: tuple[int, ...] | PackedDocumentInfo, local_length: int, size: int, rank: int | None = None,
) -> RDResetPlan:
    """Derive one rank's exact live edges from its tail document interval.

    Args:
        offsets: Global document offsets, or a batch's already validated snapshot.
            Reusing the snapshot avoids full-offset validation and hashing.
        local_length: Equal token interval length after any Ulysses exchange.
        size: Number of chronological state ranks, independent of global rank IDs.
        rank: Optional local rank whose execution schedule should be cached.

    Returns:
        An immutable CPU plan reusable across layers with identical metadata.
        Omitting rank retains only metadata for subsequent diagnostic queries.
    """
    if rank is not None and not 0 <= rank < size:
        raise ValueError("RD execution-plan rank is outside the state process group")
    if isinstance(offsets, PackedDocumentInfo):
        offsets = offsets.offsets
        if size < 1 or local_length < 1 or offsets[-1] != size * local_length:
            raise ValueError("RD reset planning requires ordered documents covering equal state-rank intervals")
    else:
        _validate_intervals(offsets, local_length, size)
    distances = tuple(1 << level for level in range((size - 1).bit_length()))
    interval = None if rank is None else _rank_interval(offsets, local_length, rank)
    rounds = () if rank is None else _interval_rounds(rank, distances, interval)
    return RDResetPlan(size, distances, offsets, local_length, rank, rounds, interval)
