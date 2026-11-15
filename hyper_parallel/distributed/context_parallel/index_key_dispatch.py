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
"""Document-prefix Index K demand, compact coordinates and differentiable dispatch."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import accumulate

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.compressed_document import (
    compressed_document_offsets, compressed_owner_offsets,
)
from hyper_parallel.distributed.context_parallel.indexer_balance import (
    indexer_task_plan, resolve_indexer_balance_block_size, validate_indexer_hybrid_threshold,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import (
    SequenceDispatchGradient, SequenceDispatchRoute,
    document_prefix_route, multi_interval_route,
)


@dataclass(frozen=True)
class IndexKeyPlan:
    """One source bank's deduplicated worker-LI and owner-KL prefix union.

    Prefix entries are (global document start, required global end, compact start).
    Owner range records the original Q owner's contiguous KL interval and its
    compact start. The gradient route excludes selection-only rows because
    worker TopK is evaluated without gradients. TopK stays globally addressed
    outside Indexer selection.
    """

    dispatch: SequenceDispatchRoute
    prefixes: tuple[tuple[int, int, int], ...]
    owner_range: tuple[int, int, int]
    global_length: int
    signature: tuple
    gradient: SequenceDispatchGradient

    @lru_cache(maxsize=128)
    def map_segments(self, segments: tuple) -> tuple:
        """Translate document prefixes while retaining each query's causal residual."""
        lookup = {start: (end, offset) for start, end, offset in self.prefixes}
        mapped = []
        for start, end, key_start, key_end, residual in segments:
            if key_end == key_start:
                offset = lookup.get(key_start, (0, 0))[1]
            else:
                if key_start not in lookup or key_end > lookup[key_start][0]:
                    raise ValueError("Index K dispatch plan does not cover this consumer's document prefix")
                offset = lookup[key_start][1]
            mapped.append((start, end, offset, offset + key_end - key_start, residual))
        return tuple(mapped)

    def global_indices(self, indices: torch.Tensor, segments: tuple) -> torch.Tensor:
        """Restore logical bank IDs before returning selection to the original owner."""
        offsets = self._global_index_offsets(segments, indices.shape[1], indices.device, indices.dtype)
        return torch.where(indices >= 0, indices + offsets, -1)

    @lru_cache(maxsize=32)
    @torch.inference_mode(False)
    def _global_index_offsets(self, segments: tuple, length: int,
                              device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Cache row offsets so fragmented tasks do not launch per-segment device operations."""
        offsets = [0] * length
        for original, mapped in zip(segments, self.map_segments(segments)):
            start, end, key_start, _, _ = original
            offsets[start:end] = [key_start - mapped[2]] * (end - start)
        return torch.tensor(offsets, device=device, dtype=dtype).view(1, length, 1)

    @lru_cache(maxsize=32)
    @torch.inference_mode(False)
    def global_positions(self, device: torch.device) -> torch.Tensor:
        """Retain reference TopK's original logical slots, including score ties."""
        return (torch.cat([torch.arange(start, end, device=device) for start, end, _ in self.prefixes])
                if self.prefixes else torch.empty(0, device=device, dtype=torch.long))

    def owner_view(self, key: torch.Tensor) -> torch.Tensor:
        """Expose the owner's KL prefix as a differentiable view of the compact bank."""
        start, end, offset = self.owner_range
        return key.narrow(1, offset, end - start)


def _validate_key_layout(lengths: tuple[int, ...], size: int, ratio: int) -> None:
    """Validate raw ownership independently of document compression and task allocation."""
    if (size < 1 or ratio < 1 or not lengths or min(lengths) < 1 or sum(lengths) % size):
        raise ValueError("Index K dispatch requires positive documents and equal contiguous raw CP shards")


def _zigzag_key_count(document: tuple, rank: int, local: int, size: int, ratio: int, capacity: int) -> int:
    """Union one worker's final visible key with its original owner's KL prefix."""
    start, end, _, chunk, remainder, pooled = document
    owner_end = min(end, (rank + 1) * local)
    owner = (owner_end - start) // ratio if max(start, rank * local) < owner_end else 0
    if chunk:
        worker = (remainder + (2 * size - rank) * chunk) // ratio
    else:
        worker_end = min(pooled + remainder, (rank + 1) * capacity)
        worker = (worker_end - pooled) // ratio if max(pooled, rank * capacity) < worker_end else 0
    return max(owner, worker)


def _zigzag_send_ranges(documents: tuple, owners: tuple, rank: int, local: int,
                         ratio: int, capacity: int) -> tuple:
    """Inspect only documents intersecting this source's physical compressed shard."""
    source_start, source_end = owners[rank:rank + 2]
    sends = []
    for peer in range(len(owners) - 1):
        ranges = []
        if peer != rank:
            for document in documents:
                key_start = document[2]
                count = _zigzag_key_count(document, peer, local, len(owners) - 1, ratio, capacity)
                first, last = max(key_start, source_start), min(key_start + count, source_end)
                if first < last or key_start < source_start == source_end < key_start + count:
                    # Preserve the generic route's zero-length intersection for empty owners.
                    ranges.append((first - source_start, last - first))
        sends.append(tuple(ranges))
    return tuple(sends)


def _zigzag_receive_counts(requested: tuple, owners: tuple, rank: int) -> tuple[int, ...]:
    """Intersect the local worker's sorted prefixes with compressed owners once."""
    receives = [0] * (len(owners) - 1)
    for start, end in requested:
        first, last = bisect_right(owners, start) - 1, bisect_left(owners, end)
        for peer in range(first, last):
            if peer != rank:
                receives[peer] += min(end, owners[peer + 1]) - max(start, owners[peer])
    return tuple(receives)


def _zigzag_key_route(lengths: tuple, boundaries: tuple, keys: tuple, owners: tuple,
                       rank: int, ratio: int, owner_route: SequenceDispatchRoute) -> tuple:
    """Construct an exact zigzag route without materializing every rank's prefixes."""
    size, local = len(owners) - 1, boundaries[-1] // (len(owners) - 1)
    capacity = sum(length % (2 * size) for length in lengths) // size
    documents, pooled = [], 0
    for start, end, key_start, length in zip(boundaries, boundaries[1:], keys, lengths):
        chunk, remainder = divmod(length, 2 * size)
        documents.append((start, end, key_start, chunk, remainder, pooled))
        pooled += remainder
    requested = []
    for document in documents:
        count = _zigzag_key_count(document, rank, local, size, ratio, capacity)
        if count:
            requested.append((document[2], document[2] + count))
    requested = tuple(requested)
    source_start, source_end = owners[rank:rank + 2]
    first, last = bisect_right(keys, source_start) - 1, bisect_left(keys, source_end)
    source_documents = tuple(documents[first:last])
    sends = _zigzag_send_ranges(source_documents, owners, rank, local, ratio, capacity)
    receives = _zigzag_receive_counts(requested, owners, rank)
    # A complete zigzag pair gives every worker a nonempty prefix (r1/r2).
    # Without any pair, the remainder pool equals the original contiguous Q shards.
    needs_exchange = owner_route.needs_exchange or (size > 1 and any(item[3] for item in documents))
    route = SequenceDispatchRoute(
        source_end - source_start, requested[0][0] if requested else source_start,
        requested[-1][1] if requested else source_start, sum(receives[:rank]),
        (0,) * size, tuple(sum(count for _, count in ranges) for ranges in sends), receives,
        needs_exchange, tuple(item for ranges in sends for item in ranges), True,
    )
    return route, requested


def _index_key_demands(lengths: tuple[int, ...], size: int, ratio: int,
                        mode: str, block_size: int | None,
                        zigzag_min_length: int | None) -> tuple:
    """Build transient owner and greedy/hybrid demands; zigzag uses its local route."""
    if mode == "zigzag":
        raise ValueError("Zigzag Index K uses rank-local route planning")
    _validate_key_layout(lengths, size, ratio)
    block_size = resolve_indexer_balance_block_size(mode, block_size)
    validate_indexer_hybrid_threshold(mode, zigzag_min_length)
    boundaries = (0, *accumulate(lengths))
    local = boundaries[-1] // size
    demands = [{} for _ in range(size)]
    for document, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        for rank in range(start // local, (end - 1) // local + 1):
            demands[rank][document] = (min(end, (rank + 1) * local) - start) // ratio
    if mode in ("greedy", "hybrid"):
        assignments = indexer_task_plan(lengths, size, ratio, mode, block_size,
                                        zigzag_min_length)
        for rank, tasks in enumerate(assignments):
            for task in tasks:
                demands[rank][task.document] = max(demands[rank].get(task.document, 0), task.end // ratio)
    keys = compressed_document_offsets(boundaries, ratio)
    return tuple(tuple((keys[document], keys[document] + count)
                       for document, count in sorted(required.items()) if count)
                 for required in demands)


@lru_cache(maxsize=128)
def index_key_plan(lengths: tuple[int, ...], size: int, rank: int, ratio: int,
                   mode: str = "none", block_size: int | None = None,
                   zigzag_min_length: int | None = None) -> IndexKeyPlan:
    """Plan one compact Index K bank shared by Full and Reindex layers.

    Args:
        lengths: Natural global packed document lengths, without token padding.
        size: Sequence-ordered CP group size.
        rank: Local CP rank.
        ratio: Source bank and selection compression ratio.
        mode: none/zigzag/greedy/hybrid query allocation, identical on all consumers.
        block_size: Greedy-pool query block option for greedy/hybrid.
        zigzag_min_length: Hybrid-only minimum original document length.

    Returns:
        A cached metadata plan. It contains no current-step bank values or TopK results.
    """
    if mode == "zigzag":
        _validate_key_layout(lengths, size, ratio)
        resolve_indexer_balance_block_size(mode, block_size)
        validate_indexer_hybrid_threshold(mode, zigzag_min_length)
        intervals = None
    else:
        intervals = _index_key_demands(lengths, size, ratio, mode, block_size, zigzag_min_length)
    local = sum(lengths) // size
    boundaries = (0, *accumulate(lengths))
    owners = compressed_owner_offsets(boundaries, ratio, local)
    keys = compressed_document_offsets(boundaries, ratio)
    gradient_route = replace(document_prefix_route(boundaries, local, size, rank, ratio), accumulate_fp32=True)
    if mode == "zigzag":
        dispatch, requested = _zigzag_key_route(lengths, boundaries, keys, owners, rank, ratio, gradient_route)
    else:
        dispatch = multi_interval_route(owners[rank+1]-owners[rank], rank, intervals,
                                        accumulate_fp32=True, owner_offsets=owners)
        requested = intervals[rank]
    offsets = (0, *accumulate(end - start for start, end in requested))
    prefixes = tuple((start, end, offset) for (start, end), offset in zip(requested, offsets))
    owner_start = keys[bisect_right(boundaries, rank * local) - 1]
    owner_end = owners[rank+1]
    owner_offset = (next(offset for start, _, offset in prefixes if start == owner_start)
                    if owner_start < owner_end else 0)
    signature = (lengths, size, rank, ratio, mode, resolve_indexer_balance_block_size(mode, block_size),
                 zigzag_min_length)
    # Worker selection has no backward. Only the original Q owner's KL prefix contributes dK.
    gradient = SequenceDispatchGradient(gradient_route, owner_offset)
    return IndexKeyPlan(dispatch, prefixes, (owner_start, owner_end, owner_offset), keys[-1], signature, gradient)
