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

from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import accumulate

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.indexer_balance import (
    indexer_task_plan, resolve_indexer_balance_block_size, validate_indexer_hybrid_thresholds,
)
from hyper_parallel.distributed.context_parallel.sequence_dispatch import (
    SequenceDispatchGradient, SequenceDispatchRoute, compressed_document_offsets, compressed_owner_offsets,
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
        result = torch.empty_like(indices)
        for original, mapped in zip(segments, self.map_segments(segments)):
            start, end, key_start, _, _ = original
            selected = indices[:, start:end]
            result[:, start:end] = torch.where(selected >= 0, selected + key_start - mapped[2], -1)
        return result

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


def _merge_zigzag_prefixes(demands: list[dict], lengths: tuple[int, ...], size: int, ratio: int) -> None:
    """Derive each worker's last visible K without expanding head/tail tasks."""
    capacity = sum(length % (2 * size) for length in lengths) // size
    pooled = 0
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        if chunk:
            # Each tail is later than the worker's head and any leading remainder.
            for rank, required in enumerate(demands):
                count = (remainder + (2 * size - rank) * chunk) // ratio
                required[document] = max(required.get(document, 0), count)
        else:
            for rank in range(pooled // capacity, (pooled + remainder - 1) // capacity + 1):
                count = min(remainder, (rank + 1) * capacity - pooled) // ratio
                demands[rank][document] = max(demands[rank].get(document, 0), count)
        pooled += remainder


def _index_key_demands(lengths: tuple[int, ...], size: int, ratio: int,
                        mode: str, block_size: int | None,
                        zigzag_min_length: int | None, zigzag_min_chunk_size: int | None) -> tuple:
    """Build transient global demands; only the current rank's resulting plan is cached."""
    _validate_key_layout(lengths, size, ratio)
    block_size = resolve_indexer_balance_block_size(mode, block_size)
    validate_indexer_hybrid_thresholds(mode, zigzag_min_length, zigzag_min_chunk_size)
    boundaries = (0, *accumulate(lengths))
    local = boundaries[-1] // size
    demands = [{} for _ in range(size)]
    for document, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        for rank in range(start // local, (end - 1) // local + 1):
            demands[rank][document] = (min(end, (rank + 1) * local) - start) // ratio
    if mode == "zigzag":
        _merge_zigzag_prefixes(demands, lengths, size, ratio)
    elif mode in ("greedy", "hybrid"):
        assignments = indexer_task_plan(lengths, size, ratio, mode, block_size,
                                        zigzag_min_length, zigzag_min_chunk_size)
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
                   zigzag_min_length: int | None = None, zigzag_min_chunk_size: int | None = None) -> IndexKeyPlan:
    """Plan one compact Index K bank shared by Full and Reindex layers.

    Args:
        lengths: Natural global packed document lengths, without token padding.
        size: Sequence-ordered CP group size.
        rank: Local CP rank.
        ratio: Source bank and selection compression ratio.
        mode: none/zigzag/greedy/hybrid query allocation, identical on all consumers.
        block_size: Greedy-pool query block option for greedy/hybrid.
        zigzag_min_length: Hybrid-only minimum original document length.
        zigzag_min_chunk_size: Hybrid-only minimum complete zigzag fragment length.

    Returns:
        A cached metadata plan. It contains no current-step bank values or TopK results.
    """
    intervals = _index_key_demands(lengths, size, ratio, mode, block_size,
                                   zigzag_min_length, zigzag_min_chunk_size)
    local = sum(lengths) // size
    boundaries = (0, *accumulate(lengths))
    owners = compressed_owner_offsets(boundaries, ratio, local)
    keys = compressed_document_offsets(boundaries, ratio)
    dispatch = multi_interval_route(owners[rank+1]-owners[rank], rank, intervals,
                                    accumulate_fp32=True, owner_offsets=owners)
    offsets = (0, *accumulate(end - start for start, end in intervals[rank]))
    prefixes = tuple((start, end, offset) for (start, end), offset in zip(intervals[rank], offsets))
    owner_start = keys[bisect_right(boundaries, rank * local) - 1]
    owner_end = owners[rank+1]
    owner_offset = (next(offset for start, _, offset in prefixes if start == owner_start)
                    if owner_start < owner_end else 0)
    signature = (lengths, size, rank, ratio, mode, resolve_indexer_balance_block_size(mode, block_size),
                 zigzag_min_length, zigzag_min_chunk_size)
    # Worker selection has no backward. Only the original Q owner's KL prefix contributes dK.
    gradient_route = replace(document_prefix_route(boundaries, local, size, rank, ratio), accumulate_fp32=True)
    gradient = SequenceDispatchGradient(gradient_route, owner_offset)
    return IndexKeyPlan(dispatch, prefixes, (owner_start, owner_end, owner_offset), keys[-1], signature, gradient)
