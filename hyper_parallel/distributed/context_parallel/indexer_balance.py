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
"""No-grad query-task redistribution for compressed Indexer selection.

Main attention, teacher and KL retain the original sequence ownership. Selection
exchanges low-rank query residuals, merge weights and original rotary embeddings;
receivers project Indexer Q and return integer TopK results. The canonical Index K
bank remains differentiable and is never repacked by this communication layer.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from heapq import heapify, heappop, heappush
from itertools import accumulate
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import


def resolve_indexer_balance_block_size(mode: str, block_size: int | None) -> int | None:
    """Resolve the greedy-pool query block option, defaulting to 2048 rows.

    None represents an omitted option so none/zigzag remain valid without it.
    Explicit values require greedy/hybrid mode and must be positive integers.
    """
    if mode not in ("none", "zigzag", "greedy", "hybrid"):
        raise ValueError("indexer_load_balance must be 'none', 'zigzag', 'greedy' or 'hybrid'")
    if block_size is not None:
        if mode not in ("greedy", "hybrid"):
            raise ValueError("indexer_balance_block_size requires indexer_load_balance='greedy' or 'hybrid'")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError("indexer_balance_block_size must be a positive integer")
    return (2048 if block_size is None else block_size) if mode in ("greedy", "hybrid") else None


def validate_indexer_hybrid_thresholds(mode: str, min_length: int | None, min_chunk_size: int | None) -> None:
    """Require explicit raw-token thresholds only when hybrid scheduling is enabled.

    These thresholds select complete zigzag fragments independently of the greedy
    block size. No hardware-dependent performance threshold is chosen implicitly.
    """
    options = (("indexer_zigzag_min_length", min_length), ("indexer_zigzag_min_chunk_size", min_chunk_size))
    for name, value in options:
        if mode != "hybrid":
            if value is not None:
                raise ValueError(f"{name} requires indexer_load_balance='hybrid'")
        elif not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"indexer_load_balance='hybrid' requires an explicit positive integer {name}")


@dataclass(frozen=True, order=True)
class IndexerTask:
    """A contiguous query interval in the original document coordinates."""

    document: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of real query rows, excluding any synthetic padding."""
        return self.end - self.start


def _merge_tasks(tasks: list[IndexerTask]) -> tuple[IndexerTask, ...]:
    merged = []
    for task in sorted(tasks):
        if merged and merged[-1].document == task.document and merged[-1].end == task.start:
            merged[-1] = IndexerTask(task.document, merged[-1].start, task.end)
        elif task.length:
            merged.append(task)
    return tuple(merged)


def _work_prefix(length: int, ratio: int) -> int:
    groups, tail = divmod(length, ratio)
    return ratio * groups * (groups - 1) // 2 + groups * (tail + 1)


def _task_work(task: IndexerTask, ratio: int) -> int:
    return _work_prefix(task.end, ratio) - _work_prefix(task.start, ratio)


def _zigzag_tasks(lengths: tuple[int, ...], size: int) -> list[list[IndexerTask]]:
    assignments = [[] for _ in range(size)]
    remainders = []
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        if remainder:
            remainders.append(IndexerTask(document, 0, remainder))
        if chunk:
            for rank in range(size):
                for slot in (rank, 2 * size - 1 - rank):
                    start = remainder + slot * chunk
                    assignments[rank].append(IndexerTask(document, start, start + chunk))
    # All ranks already own equally many paired rows. Distribute the real
    # remainder rows evenly without inserting tokens into any document.
    capacity = sum(task.length for task in remainders) // size
    rank, used = 0, 0
    for task in remainders:
        start = task.start
        while start < task.end:
            if used == capacity:
                rank += 1
                used = 0
            end = min(task.end, start + capacity - used)
            assignments[rank].append(IndexerTask(task.document, start, end))
            used += end - start
            start = end
    return assignments


def _greedy_tasks(lengths: tuple[int, ...], size: int, ratio: int, block: int) -> list[list[IndexerTask]]:
    """Keep the original cost/count/rank ordering without scanning every rank."""
    tasks = [IndexerTask(document, start, min(start + block, length))
             for document, length in enumerate(lengths) for start in range(0, length, block)]
    assignments = [[] for _ in range(size)]
    _assign_greedy_tasks(assignments, tasks, ratio, 2 * (sum(lengths) // size))
    return assignments


def _assign_greedy_tasks(assignments: list[list[IndexerTask]], tasks: list[IndexerTask],
                          ratio: int, capacity: int) -> None:
    """Add pooled tasks to the existing per-worker cost and row budgets."""
    available = [(sum(_task_work(task, ratio) for task in assigned), sum(task.length for task in assigned), rank)
                 for rank, assigned in enumerate(assignments)]
    heapify(available)
    # Reuse the cost computed for sorting when updating the selected worker.
    for negative_work, task in sorted((-_task_work(task, ratio), task) for task in tasks):
        skipped = []
        while True:
            cost, count, rank = heappop(available)
            if count + task.length <= capacity:
                break
            skipped.append((cost, count, rank))
        assignments[rank].append(task)
        heappush(available, (cost - negative_work, count + task.length, rank))
        for item in skipped:
            heappush(available, item)


def _hybrid_tasks(lengths: tuple[int, ...], size: int, ratio: int, block: int,
                   min_length: int, min_chunk_size: int) -> list[list[IndexerTask]]:
    """Keep eligible zigzag pairs intact and greedily allocate every remaining row."""
    assignments, pooled = [[] for _ in range(size)], []
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        if length >= min_length and chunk >= min_chunk_size:
            for rank in range(size):
                for slot in (rank, 2 * size - 1 - rank):
                    start = remainder + slot * chunk
                    assignments[rank].append(IndexerTask(document, start, start + chunk))
            pool_end = remainder
        else:
            pool_end = length
        pooled.extend(IndexerTask(document, start, min(start + block, pool_end))
                      for start in range(0, pool_end, block))
    _assign_greedy_tasks(assignments, pooled, ratio, 2 * (sum(lengths) // size))
    return assignments


def _validate_task_plan(lengths: tuple[int, ...], size: int, ratio: int, mode: str,
                        block_size: int | None) -> int | None:
    """Keep input validation identical for direct and all-worker planning."""
    if size < 1 or ratio < 1 or not lengths or min(lengths) < 1 or sum(lengths) % size:
        raise ValueError("Indexer balance needs positive document lengths/ratio and an equal contiguous CP partition")
    block = resolve_indexer_balance_block_size(mode, block_size)
    if mode not in ("zigzag", "greedy", "hybrid"):
        raise ValueError("Indexer task mode must be 'zigzag', 'greedy' or 'hybrid'")
    local = sum(lengths) // size
    if block is not None and block > local:
        raise ValueError(f"Indexer greedy block size {block} exceeds the local query count {local}")
    return block


@lru_cache(maxsize=128)
def indexer_task_plan(lengths: tuple[int, ...], size: int, ratio: int, mode: str,
                      block_size: int | None = None, zigzag_min_length: int | None = None,
                      zigzag_min_chunk_size: int | None = None) -> tuple:
    """Assign real queries using zigzag, greedy or a shared-budget hybrid plan.

    Args:
        lengths: Global document lengths in packed order.
        size: CP group size; total packed tokens must divide this value.
        ratio: Compression ratio used by the unchanged Indexer kernel.
        mode: 'zigzag' pairs document tails and heads after removing the leading
            remainder; 'greedy' weights blocks by the number of causal Q/K pairs;
            'hybrid' keeps long-document zigzag fragments and greedily places
            their leading remainders together with all shorter documents.
        block_size: Maximum original query rows per greedy task, before adjacent
            tasks assigned to one rank are merged. Greedy/hybrid default to 2048.
            Explicit values require greedy/hybrid and must be positive and at most local Q
            to guarantee its per-worker row bound. Omit this option for zigzag.
        zigzag_min_length: Hybrid-only minimum document length in original tokens.
            Must be specified explicitly together with zigzag_min_chunk_size.
        zigzag_min_chunk_size: Hybrid-only minimum floor(document length / (2*CP)).
            Eligible fragments remain whole, independently of block_size.

    Returns:
        A sorted, disjoint task list for each rank. Greedy/hybrid cap each worker at
        twice the original local query count; zigzag preserves equal row counts.
    """
    block = _validate_task_plan(lengths, size, ratio, mode, block_size)
    validate_indexer_hybrid_thresholds(mode, zigzag_min_length, zigzag_min_chunk_size)
    if mode == "zigzag":
        assignments = _zigzag_tasks(lengths, size)
    elif mode == "hybrid":
        assignments = _hybrid_tasks(lengths, size, ratio, block, zigzag_min_length, zigzag_min_chunk_size)
    else:
        assignments = _greedy_tasks(lengths, size, ratio, block)
    return tuple(_merge_tasks(tasks) for tasks in assignments)


def _segments(tasks: tuple, lengths: tuple[int, ...], ratio: int) -> tuple:
    key_bases = (0, *accumulate(length // ratio for length in lengths))
    result, cursor = [], 0
    for task in tasks:
        base = key_bases[task.document]
        result.append((cursor, cursor + task.length, base, base + task.end // ratio, task.end % ratio))
        cursor += task.length
    return tuple(result)


def _send_ranges(tasks: tuple, boundaries: tuple[int, ...], owner: int, local: int) -> tuple:
    """Find only tasks intersecting this owner in each sorted worker assignment."""
    first, last = owner * local, (owner + 1) * local
    start = bisect_right(tasks, first, key=lambda task: boundaries[task.document] + task.end)
    result = []
    for index in range(start, len(tasks)):
        task = tasks[index]
        begin, end = boundaries[task.document] + task.start, boundaries[task.document] + task.end
        if begin >= last:
            break
        begin, end = max(first, begin), min(last, end)
        result.append((begin - first, end - begin))
    return tuple(result)


def _receive_ranges(tasks: tuple, boundaries: tuple[int, ...], size: int, local: int) -> tuple:
    """Index a worker's tasks by owner once, retaining its original packed offsets."""
    owners, cursor = [[] for _ in range(size)], 0
    for task in tasks:
        begin, end = boundaries[task.document] + task.start, boundaries[task.document] + task.end
        for owner in range(begin // local, (end - 1) // local + 1):
            first, last = max(begin, owner * local), min(end, (owner + 1) * local)
            owners[owner].append((first - owner * local, cursor + first - begin, last - first))
        cursor += task.length
    return tuple(tuple(rows) for rows in owners)


def _append_ordered_task(tasks: list[IndexerTask], document: int, start: int, end: int) -> None:
    """Merge an already ordered interval without sorting the worker's tasks."""
    if start >= end:
        return
    if tasks and tasks[-1].document == document and tasks[-1].end == start:
        tasks[-1] = IndexerTask(document, tasks[-1].start, end)
    else:
        tasks.append(IndexerTask(document, start, end))


def _zigzag_rank_tasks(lengths: tuple[int, ...], size: int, rank: int) -> tuple[IndexerTask, ...]:
    """Construct only this worker's paired chunks and share of leading remainders."""
    capacity = sum(length % (2 * size) for length in lengths) // size
    first, last = rank * capacity, (rank + 1) * capacity
    pooled, tasks = 0, []
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        _append_ordered_task(tasks, document, max(0, first - pooled), min(remainder, last - pooled))
        pooled += remainder
        if chunk:
            for slot in (rank, 2 * size - 1 - rank):
                start = remainder + slot * chunk
                _append_ordered_task(tasks, document, start, start + chunk)
    return tuple(tasks)


def _zigzag_intersections(length: int, size: int, pooled: int, capacity: int,
                          begin: int, end: int) -> Iterator[tuple[int, int, int]]:
    """Yield each destination intersecting an owner's interval in one document."""
    chunk, remainder = divmod(length, 2 * size)
    # Positive remainders imply positive capacity after validating total % size.
    while begin < min(end, remainder):
        peer = (pooled + begin) // capacity
        stop = min(end, remainder, (peer + 1) * capacity - pooled)
        yield peer, begin, stop - begin
        begin = stop
    while begin < end:
        slot = (begin - remainder) // chunk
        peer = min(slot, 2 * size - 1 - slot)
        stop = min(end, remainder + (slot + 1) * chunk)
        yield peer, begin, stop - begin
        begin = stop


def _zigzag_send_ranges(lengths: tuple[int, ...], size: int, rank: int, boundaries: tuple) -> tuple:
    """Invert zigzag placement for local rows instead of visiting all workers."""
    local = boundaries[-1] // size
    pooled = (0, *accumulate(length % (2 * size) for length in lengths))
    capacity = pooled[-1] // size
    first, last = rank * local, (rank + 1) * local
    sends = [[] for _ in range(size)]
    document = bisect_right(boundaries, first) - 1
    while document < len(lengths) and boundaries[document] < last:
        for peer, start, count in _zigzag_intersections(
                lengths[document], size, pooled[document], capacity,
                max(0, first - boundaries[document]), min(lengths[document], last - boundaries[document])):
            sends[peer].append((boundaries[document] + start - first, count))
        document += 1
    return tuple(tuple(ranges) for ranges in sends)


@dataclass(frozen=True)
class IndexerRoute:
    """Device-independent row ordering and matching variable collective splits."""

    segments: tuple
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    send_rows: tuple[int, ...]
    receive_rows: tuple[int, ...]
    local_sources: tuple[int, ...]
    local_targets: tuple[int, ...]
    query_count: int


@lru_cache(maxsize=128)
def indexer_route(lengths: tuple[int, ...], size: int, rank: int, ratio: int, mode: str,
                  block_size: int | None = None, zigzag_min_length: int | None = None,
                  zigzag_min_chunk_size: int | None = None) -> IndexerRoute:
    """Keep self-owned tasks local and communicate the remaining rows in owner order."""
    if not 0 <= rank < size:
        raise ValueError("Indexer balance rank must be inside its CP group")
    if mode == "zigzag":
        _validate_task_plan(lengths, size, ratio, mode, block_size)
        validate_indexer_hybrid_thresholds(mode, zigzag_min_length, zigzag_min_chunk_size)
        boundaries = (0, *accumulate(lengths))
        tasks = _zigzag_rank_tasks(lengths, size, rank)
        sends = _zigzag_send_ranges(lengths, size, rank, boundaries)
        receives = _receive_ranges(tasks, boundaries, size, boundaries[-1] // size)
        return _route_from_ranges(tasks, lengths, rank, ratio, sends, receives)
    assignments = indexer_task_plan(lengths, size, ratio, mode, block_size,
                                    zigzag_min_length, zigzag_min_chunk_size)
    return _route_from_tasks(assignments, lengths, rank, ratio)


def _route_from_tasks(assignments: tuple, lengths: tuple[int, ...], rank: int, ratio: int) -> IndexerRoute:
    size = len(assignments)
    boundaries, local = (0, *accumulate(lengths)), sum(lengths) // size
    receives = _receive_ranges(assignments[rank], boundaries, size, local)
    sends = tuple(_send_ranges(tasks, boundaries, rank, local) if peer != rank else ()
                  for peer, tasks in enumerate(assignments))
    return _route_from_ranges(assignments[rank], lengths, rank, ratio, sends, receives)


def _route_from_ranges(tasks: tuple, lengths: tuple[int, ...], rank: int, ratio: int,
                       sends: tuple, receives: tuple) -> IndexerRoute:
    """Preserve peer ordering and local bypass for either task-planning path."""
    send_rows, receive_rows, inputs, outputs = [], [], [], []
    for peer, sent in enumerate(sends):
        sent = sent if peer != rank else ()
        received = receives[peer] if peer != rank else ()
        inputs.append(sum(count for _, count in sent))
        outputs.append(sum(count for _, _, count in received))
        send_rows.extend(row for start, count in sent for row in range(start, start + count))
        receive_rows.extend(row for _, start, count in received for row in range(start, start + count))
    sources = tuple(row for start, _, count in receives[rank] for row in range(start, start + count))
    targets = tuple(row for _, start, count in receives[rank] for row in range(start, start + count))
    return IndexerRoute(_segments(tasks, lengths, ratio), tuple(inputs), tuple(outputs),
                        tuple(send_rows), tuple(receive_rows), sources, targets,
                        sum(task.length for task in tasks))


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _device_rows(route: IndexerRoute, device: torch.device) -> tuple:
    return tuple(torch.tensor(rows, device=device, dtype=torch.long) for rows in
                 (route.send_rows, route.receive_rows, route.local_sources, route.local_targets))


def _exchange_rows(tensor: torch.Tensor, route: IndexerRoute, group: Any, rows: tuple) -> torch.Tensor:
    send_rows, receive_rows, local_sources, local_targets = rows
    send = tensor.index_select(0, send_rows).contiguous()
    received = tensor.new_empty((sum(route.output_splits), *tensor.shape[1:]))
    work = dist.all_to_all_single(received, send, list(route.output_splits), list(route.input_splits),
                                  group=group, async_op=True)
    output = tensor.new_empty((route.query_count, *tensor.shape[1:]))
    output.index_copy_(0, local_targets, tensor.index_select(0, local_sources))
    work.wait()
    output.index_copy_(0, receive_rows, received)
    return output


def _exchange_inputs(inputs: tuple[torch.Tensor, ...], route: IndexerRoute, group: Any, rows: tuple) -> tuple:
    """Pack fields of equal dtype together without rounding the original RoPE values."""
    by_dtype = {}
    for index, tensor in enumerate(inputs):
        by_dtype.setdefault(tensor.dtype, []).append(index)
    outputs = [None] * len(inputs)
    for indices in by_dtype.values():
        widths = [inputs[index].shape[-1] for index in indices]
        packed = torch.cat([inputs[index][0] for index in indices], dim=-1)
        received = _exchange_rows(packed, route, group, rows)
        for index, field in zip(indices, received.split(widths, dim=-1)):
            outputs[index] = field.unsqueeze(0).contiguous()
    return tuple(outputs)


@torch.no_grad()
def balanced_indexer_select(query_residual: torch.Tensor, weights: torch.Tensor,
                            position_embeddings: tuple[torch.Tensor, torch.Tensor], selector: Callable,
                            lengths: tuple[int, ...], ratio: int, mode: str, group: Any,
                            block_size: int | None = None, zigzag_min_length: int | None = None,
                            zigzag_min_chunk_size: int | None = None) -> torch.Tensor:
    """Redistribute low-rank selection inputs and return TopK to the original owners.

    Args:
        query_residual: Local low-rank query residual [1, local tokens, q_lora_rank].
        weights: Local merge weights [1, local tokens, heads].
        position_embeddings: Original per-token cos/sin, each [1, local tokens, rotary/2].
            All payload dtypes are preserved; tensors of equal dtype share a collective.
        selector: Callable (qr, weights, (cos, sin), segments) that projects Indexer Q
            with the current layer parameters and selects against the planned Index K bank.
            It runs without autograd; owner-side Q and weights retain their KL gradients.
        lengths: Global original document lengths; no per-document CP padding is added.
        ratio: Compression ratio used by this selection layer.
        mode: 'zigzag', 'greedy' or 'hybrid', identical on all CP ranks.
        group: Original sequence-ordered CP process group.
        block_size: Greedy task length in original query rows; see indexer_task_plan.
        zigzag_min_length: Hybrid-only minimum original document length.
        zigzag_min_chunk_size: Hybrid-only minimum complete zigzag fragment length.
    """
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    inputs = (query_residual, weights, *position_embeddings)
    if (len(position_embeddings) != 2 or query_residual.ndim != 3 or query_residual.shape[0] != 1
            or query_residual.shape[1] * size != sum(lengths)):
        raise ValueError("Indexer balance requires batch one and equal contiguous source query shards")
    if any(tensor.ndim != 3 or tensor.shape[:2] != query_residual.shape[:2]
           or tensor.device != query_residual.device for tensor in inputs):
        raise ValueError("Indexer qr, weights and rotary embeddings must have matching batch/token axes and device")
    route = indexer_route(lengths, size, rank, ratio, mode, block_size,
                           zigzag_min_length, zigzag_min_chunk_size)
    rows = _device_rows(route, query_residual.device)
    balanced_qr, balanced_weights, cos, sin = _exchange_inputs(inputs, route, group, rows)
    selected = selector(balanced_qr, balanced_weights, (cos, sin), route.segments)[0]
    send_rows, receive_rows, local_sources, local_targets = rows
    send = selected.index_select(0, receive_rows).contiguous()
    received = selected.new_empty((sum(route.input_splits), selected.shape[-1]))
    work = dist.all_to_all_single(received, send, list(route.input_splits), list(route.output_splits),
                                  group=group, async_op=True)
    output = selected.new_empty((query_residual.shape[1], selected.shape[-1]))
    output.index_copy_(0, local_sources, selected.index_select(0, local_targets))
    work.wait()
    output.index_copy_(0, send_rows, received)
    return output.unsqueeze(0)
