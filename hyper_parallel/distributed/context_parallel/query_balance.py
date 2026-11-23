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
"""Shared document query scheduling and reversible row routes for CP operators."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from heapq import heapify, heappop, heappush
from itertools import accumulate


@dataclass(frozen=True, order=True)
class QueryTask:
    """A contiguous query interval in the original document coordinates."""

    document: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of real query rows, excluding any synthetic padding."""
        return self.end - self.start


def _merge_tasks(tasks: list[QueryTask]) -> tuple[QueryTask, ...]:
    merged = []
    for task in sorted(tasks):
        if merged and merged[-1].document == task.document and merged[-1].end == task.start:
            merged[-1] = QueryTask(task.document, merged[-1].start, task.end)
        elif task.length:
            merged.append(task)
    return tuple(merged)


def _work_prefix(length: int, ratio: int) -> int:
    groups, tail = divmod(length, ratio)
    return ratio * groups * (groups - 1) // 2 + groups * (tail + 1)


def _task_work(task: QueryTask, ratio: int, lengths: tuple = (), causal: bool = True) -> int:
    if not causal:
        return task.length * (lengths[task.document] // ratio)
    return _work_prefix(task.end, ratio) - _work_prefix(task.start, ratio)


def _zigzag_tasks(lengths: tuple[int, ...], size: int) -> list[list[QueryTask]]:
    """Assign paired document chunks and evenly distribute their leading remainders."""
    assignments = [[] for _ in range(size)]
    remainders = []
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        if remainder:
            remainders.append(QueryTask(document, 0, remainder))
        if chunk:
            for rank in range(size):
                for slot in (rank, 2 * size - 1 - rank):
                    start = remainder + slot * chunk
                    assignments[rank].append(QueryTask(document, start, start + chunk))
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
            assignments[rank].append(QueryTask(task.document, start, end))
            used += end - start
            start = end
    return assignments


def _greedy_tasks(lengths: tuple[int, ...], size: int, ratio: int, block: int, causal: bool) -> list[list[QueryTask]]:
    """Keep the original cost/count/rank ordering without scanning every rank."""
    tasks = [QueryTask(document, start, min(start + block, length))
             for document, length in enumerate(lengths) for start in range(0, length, block)]
    assignments = [[] for _ in range(size)]
    _assign_greedy_tasks(assignments, tasks, ratio, 2 * (sum(lengths) // size), lengths, causal)
    return assignments


def _assign_greedy_tasks(assignments: list[list[QueryTask]], tasks: list[QueryTask],
                          ratio: int, capacity: int, lengths: tuple, causal: bool) -> None:
    """Add pooled tasks to the existing per-worker cost and row budgets."""
    available = [(sum(_task_work(task, ratio, lengths, causal) for task in assigned),
                  sum(task.length for task in assigned), rank)
                 for rank, assigned in enumerate(assignments)]
    heapify(available)
    # Reuse the cost computed for sorting when updating the selected worker.
    for negative_work, task in sorted((-_task_work(task, ratio, lengths, causal), task) for task in tasks):
        skipped = []
        for _ in range(len(available)):
            cost, count, rank = heappop(available)
            if count + task.length <= capacity:
                break
            skipped.append((cost, count, rank))
        else:
            raise ValueError(f"No query worker has capacity for {task.length} query rows (limit: {capacity})")
        assignments[rank].append(task)
        heappush(available, (cost - negative_work, count + task.length, rank))
        for item in skipped:
            heappush(available, item)


def _hybrid_tasks(lengths: tuple[int, ...], size: int, ratio: int, block: int,
                   min_length: int, causal: bool) -> list[list[QueryTask]]:
    """Keep eligible zigzag pairs intact and greedily allocate every remaining row.

    Documents shorter than 2*size contain no complete pair and enter the greedy pool.
    """
    assignments, pooled = [[] for _ in range(size)], []
    for document, length in enumerate(lengths):
        chunk, remainder = divmod(length, 2 * size)
        if length >= min_length and chunk:
            for rank in range(size):
                for slot in (rank, 2 * size - 1 - rank):
                    start = remainder + slot * chunk
                    assignments[rank].append(QueryTask(document, start, start + chunk))
            pool_end = remainder
        else:
            pool_end = length
        pooled.extend(QueryTask(document, start, min(start + block, pool_end))
                      for start in range(0, pool_end, block))
    _assign_greedy_tasks(assignments, pooled, ratio, 2 * (sum(lengths) // size), lengths, causal)
    return assignments


@lru_cache(maxsize=128)
def query_task_plan(
        lengths: tuple[int, ...], size: int, mode: str = "none", *, causal: bool = True,
        compression_ratio: int = 1, block_size: int | None = None,
        zigzag_min_length: int | None = None,
) -> tuple[tuple[QueryTask, ...], ...]:
    """Assign document-local queries with bounded worker capacity.

    Args:
        lengths: Positive global document lengths in packed order, shared by all ranks.
        size: Number of workers; total tokens must divide size.
        mode: Contiguous 'none', per-document 'zigzag', 'greedy', or 'hybrid'.
        causal: Estimate causal pair counts, or full document attention work.
        compression_ratio: Number of original tokens represented by each complete key.
        block_size: Greedy query block size, at most original local tokens.
        zigzag_min_length: Explicit raw-token threshold required for hybrid.

    Returns:
        Sorted disjoint query intervals per worker. Zigzag keeps equal row counts;
        greedy/hybrid cap each worker at twice the original local token count.
    """
    if (not isinstance(size, int) or isinstance(size, bool) or size < 1
            or not isinstance(compression_ratio, int) or isinstance(compression_ratio, bool)
            or compression_ratio < 1 or not lengths
            or any(not isinstance(length, int) or isinstance(length, bool) or length < 1 for length in lengths)
            or sum(lengths) % size):
        raise ValueError("query plans require positive lengths/ratio and equal contiguous source shards")
    if mode not in ("none", "zigzag", "greedy", "hybrid") or not isinstance(causal, bool):
        raise ValueError("load_balance must be none, zigzag, greedy or hybrid; causal must be bool")
    local = sum(lengths) // size
    if mode in ("greedy", "hybrid"):
        block_size = min(2048, local) if block_size is None else block_size
        if (not isinstance(block_size, int) or isinstance(block_size, bool) or not 1 <= block_size <= local):
            raise ValueError("balance_block_size must be a positive integer at most the local token count")
    elif block_size is not None:
        raise ValueError("balance_block_size requires greedy or hybrid")
    if mode == "hybrid":
        if (not isinstance(zigzag_min_length, int) or isinstance(zigzag_min_length, bool)
                or zigzag_min_length < 1):
            raise ValueError("hybrid requires an explicit positive zigzag_min_length")
    elif zigzag_min_length is not None:
        raise ValueError("zigzag_min_length requires hybrid")
    if mode == "none":
        boundaries = (0, *accumulate(lengths))
        return tuple(tuple(QueryTask(document, max(rank * local, begin) - begin,
                                      min((rank + 1) * local, end) - begin)
                           for document, (begin, end) in enumerate(zip(boundaries, boundaries[1:]))
                           if max(rank * local, begin) < min((rank + 1) * local, end))
                     for rank in range(size))
    if mode == "zigzag":
        assignments = _zigzag_tasks(lengths, size)
    elif mode == "greedy":
        assignments = _greedy_tasks(lengths, size, compression_ratio, block_size, causal)
    else:
        assignments = _hybrid_tasks(lengths, size, compression_ratio, block_size, zigzag_min_length, causal)
    return tuple(_merge_tasks(tasks) for tasks in assignments)


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


@dataclass(frozen=True)
class QueryRoute:
    """Device-independent row ordering and matching variable collective splits."""

    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    send_rows: tuple[int, ...]
    receive_rows: tuple[int, ...]
    local_sources: tuple[int, ...]
    local_targets: tuple[int, ...]
    query_count: int


@lru_cache(maxsize=128)
def query_route(assignments: tuple, lengths: tuple[int, ...], rank: int) -> QueryRoute:
    """Build a reversible owner-to-worker row route with local rows bypassing communication."""
    size = len(assignments)
    boundaries, local = (0, *accumulate(lengths)), sum(lengths) // size
    if not 0 <= rank < size:
        raise ValueError("rank must belong to the task plan")
    receives = _receive_ranges(assignments[rank], boundaries, size, local)
    sends = tuple(_send_ranges(tasks, boundaries, rank, local) if peer != rank else ()
                  for peer, tasks in enumerate(assignments))
    return query_route_from_ranges(assignments[rank], rank, sends, receives)


def query_route_from_ranges(tasks: tuple, rank: int,
                       sends: tuple, receives: tuple) -> QueryRoute:
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
    return QueryRoute(tuple(inputs), tuple(outputs),
                        tuple(send_rows), tuple(receive_rows), sources, targets,
                        sum(task.length for task in tasks))
