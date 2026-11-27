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


@lru_cache(maxsize=128)
def query_task_plan(
        lengths: tuple[int, ...], size: int, load_balance: str = "none",
) -> tuple[tuple[QueryTask, ...], ...]:
    """Assign document-local queries with bounded worker capacity.

    Args:
        lengths: Positive global document lengths in packed order, shared by all ranks.
        size: Number of workers; total tokens must divide size.
        load_balance: Contiguous 'none' or per-document 'zigzag'.

    Returns:
        Sorted disjoint query intervals per worker. Zigzag keeps equal row counts.
    """
    if (not isinstance(size, int) or isinstance(size, bool) or size < 1 or not lengths
            or any(not isinstance(length, int) or isinstance(length, bool) or length < 1 for length in lengths)
            or sum(lengths) % size):
        raise ValueError("query plans require positive lengths and equal contiguous source shards")
    if load_balance not in ("none", "zigzag"):
        raise ValueError("load_balance must be none or zigzag")
    local = sum(lengths) // size
    if load_balance == "none":
        boundaries = (0, *accumulate(lengths))
        return tuple(tuple(QueryTask(document, max(rank * local, begin) - begin,
                                      min((rank + 1) * local, end) - begin)
                           for document, (begin, end) in enumerate(zip(boundaries, boundaries[1:]))
                           if max(rank * local, begin) < min((rank + 1) * local, end))
                     for rank in range(size))
    assignments = _zigzag_tasks(lengths, size)
    return tuple(_merge_tasks(tasks) for tasks in assignments)


def _distribute_leading_remainder(assignments, document, workers, remainder):
    """Spread a document's leading remainder over only its participating ranks."""
    base, extra = divmod(remainder, len(workers))
    start = 0
    for index, worker in enumerate(workers):
        count = base + (index < extra)
        if count:
            assignments[worker].append(QueryTask(document, start, start + count))
        start += count


def _subring_zigzag(assignments, document, length, workers):
    """Pair causal head/tail chunks across the ranks intersecting one document."""
    chunk, remainder = divmod(length, 2 * len(workers))
    _distribute_leading_remainder(assignments, document, workers, remainder)
    for index, worker in enumerate(workers):
        for slot in (index, 2 * len(workers) - 1 - index):
            start = remainder + slot * chunk
            if chunk:
                assignments[worker].append(QueryTask(document, start, start + chunk))


@lru_cache(maxsize=128)
def query_task_plan_in_document_rings(
        lengths: tuple[int, ...], size: int, load_balance: str = "none",
) -> tuple[tuple[QueryTask, ...], ...]:
    """Balance each document only among CP ranks that intersect that document.

    Unlike ``query_task_plan``, this planner preserves the document's contiguous
    source-owner subring. It is used with document-pruned KV transport, so no
    query is moved to a rank outside the document's active ring.
    """
    if (not isinstance(size, int) or isinstance(size, bool) or size < 1
            or not lengths or any(not isinstance(length, int) or isinstance(length, bool) or length < 1
                                  for length in lengths)
            or sum(lengths) % size):
        raise ValueError("document subrings require positive lengths and equal contiguous source shards")
    if load_balance not in ("none", "zigzag"):
        raise ValueError("load_balance must be none or zigzag")
    local = sum(lengths) // size

    assignments = [[] for _ in range(size)]
    boundaries = (0, *accumulate(lengths))
    for document, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        first = begin // local
        last = (end - 1) // local
        workers = tuple(range(first, last + 1))
        length = end - begin
        if load_balance == "none":
            for worker in workers:
                left = max(begin, worker * local) - begin
                right = min(end, (worker + 1) * local) - begin
                assignments[worker].append(QueryTask(document, left, right))
        elif load_balance == "zigzag":
            _subring_zigzag(assignments, document, length, workers)
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
