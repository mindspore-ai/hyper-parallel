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
"""Document-aware Ring blocks and differentiable owner/worker layout conversion."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import accumulate, groupby
from functools import lru_cache
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.utils.communication import differentiable_all_to_all_single
from hyper_parallel.distributed.context_parallel.query_balance import query_route, query_task_plan


@dataclass(frozen=True)
class AttentionBlock:
    """One valid Q/K rectangle, optionally causal with bottom-right alignment."""

    query_start: int
    query_end: int
    key_start: int
    key_end: int
    causal: bool = False


@dataclass(frozen=True)
class RingTaskPlan:
    """Global ownership and this rank's executable blocks; contains no activations."""

    assignments: tuple
    route: Any
    blocks: tuple[tuple[AttentionBlock, ...], ...]
    batches: tuple[tuple[tuple[AttentionBlock, ...], ...], ...]
    capacity: int
    local_tokens: int
    fast_zigzag: bool


@dataclass
class RingAttentionTrace:
    """Optional device events per Ring round; synchronize only when reading timings.

    Events accumulate until the caller clears them after completion. Tracing does not add
    barriers or synchronize devices inside attention. Timings include each round's
    launches, block computation and waits, but exclude surrounding redistribution.
    """

    events: list = field(default_factory=list)

    def begin(self) -> Any:
        """Record the beginning of a round on the current NPU stream."""
        event = torch.npu.Event(enable_timing=True)
        event.record()
        return event

    def end(self, phase: str, step: int, begin: Any) -> None:
        """Record completion after dependencies have been placed on the stream."""
        event = torch.npu.Event(enable_timing=True)
        event.record()
        self.events.append((phase, step, begin, event))

    def timings(self) -> list[dict]:
        """Wait for recorded work and return milliseconds for each phase/round."""
        result = []
        for phase, step, begin, end in self.events:
            end.synchronize()
            result.append({"phase": phase, "step": step, "milliseconds": begin.elapsed_time(end)})
        return result


def _document_ranges(tasks):
    cursor, result = 0, {}
    for document, grouped in groupby(tasks, key=lambda task: task.document):
        parts = tuple(grouped)
        ends = tuple(accumulate(task.length for task in parts))
        result[document] = (cursor, parts, ends)
        cursor += ends[-1]
    return result


def _document_blocks(query_tasks, key_tasks, causal):
    """Coalesce each query's visible KV prefix in the compact owner buffer.

    Different owners have disjoint original token intervals. Remote causal blocks
    are therefore full rectangles; own-document blocks are causal in compact order.
    Each real Q row occurs at most once in a round, including across document gaps.
    """
    queries, keys = _document_ranges(query_tasks), _document_ranges(key_tasks)
    blocks = []
    same_owner = query_tasks == key_tasks
    for document, (query_start, parts, query_ends) in queries.items():
        if document not in keys:
            continue
        key_start, key_parts, key_ends = keys[document]
        if same_owner or not causal:
            blocks.append(AttentionBlock(query_start, query_start + query_ends[-1],
                                         key_start, key_start + key_ends[-1], causal and same_owner))
            continue
        ends = tuple(part.end for part in key_parts)
        offset = query_start
        for query in parts:
            count = bisect_right(ends, query.start)
            if count:
                key_end = key_start + key_ends[count - 1]
                if (blocks and blocks[-1].query_end == offset and blocks[-1].key_start == key_start
                        and blocks[-1].key_end == key_end and not blocks[-1].causal):
                    previous = blocks.pop()
                    blocks.append(AttentionBlock(previous.query_start, offset + query.length, key_start, key_end))
                else:
                    blocks.append(AttentionBlock(offset, offset + query.length, key_start, key_end))
            offset += query.length
    return tuple(blocks)


def _block_batches(blocks, capacity):
    """Bound repeated KV prefixes to four transport buffers per fused call."""
    batches, current, rows = [], [], 0
    for block in blocks:
        count = block.key_end - block.key_start
        if current and (rows + count > 4 * capacity or current[0].causal != block.causal):
            batches.append(tuple(current))
            current, rows = [], 0
        current.append(block)
        rows += count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


@lru_cache(maxsize=128)
def ring_task_plan(
        lengths: tuple[int, ...], size: int, rank: int, mode: str, causal: bool,
        block_size: int | None = None, zigzag_min_length: int | None = None,
) -> RingTaskPlan:
    """Build exact document-local masks and an equal-capacity transport plan.

    One divisible dense causal document uses one kernel per round for Zigzag.
    Other plans decompose valid intersections into full and suffix-causal blocks.
    Padding equalizes transport capacity only and never enters a kernel.
    """
    assignments = query_task_plan(lengths, size, mode, causal=causal, block_size=block_size,
                                  zigzag_min_length=zigzag_min_length)
    route = query_route(assignments, lengths, rank)
    local = sum(lengths) // size
    paired = mode == "zigzag" or (mode == "hybrid" and lengths[0] >= zigzag_min_length)
    fast = paired and causal and len(lengths) == 1 and lengths[0] % (2 * size) == 0
    if fast:
        half = local // 2
        blocks = tuple((AttentionBlock(0, local, 0, local, True),) if owner == rank else
                       (AttentionBlock(0, local, 0, half),) if owner < rank else
                       (AttentionBlock(half, local, 0, local),) for owner in range(size))
    else:
        blocks = tuple(_document_blocks(assignments[rank], tasks, causal) for tasks in assignments)
    capacity = max(sum(task.length for task in tasks) for tasks in assignments)
    batches = tuple(_block_batches(parts, capacity) for parts in blocks)
    return RingTaskPlan(assignments, route, blocks, batches, capacity, local, fast)


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _route_indices(route, device):
    return tuple(torch.tensor(rows, device=device, dtype=torch.long) for rows in
                 (route.send_rows, route.receive_rows, route.local_sources, route.local_targets))


def redistribute_ring_rows(tensor: torch.Tensor, plan: RingTaskPlan, group: Any,
                            sequence_dim: int, *, inverse: bool = False) -> torch.Tensor:
    """Apply a reversible all-to-all-v, keeping self-owned rows local.

    Forward packs task rows and pads to a shared Ring buffer capacity. Inverse
    restores original owner order and excludes padding. Both directions preserve
    autograd, including ranks whose remote sends or receives have zero rows.
    """
    if len(plan.assignments) == 1:
        return tensor
    route = plan.route
    send, receive, sources, targets = _route_indices(route, tensor.device)
    inputs, outputs = route.input_splits, route.output_splits
    count = plan.capacity
    if inverse:
        send, receive, sources, targets = receive, send, targets, sources
        inputs, outputs = outputs, inputs
        count = plan.local_tokens
    rows = tensor.movedim(sequence_dim, 0)
    remote = differentiable_all_to_all_single(rows.index_select(0, send).contiguous(),
                                             list(inputs), list(outputs), group)
    result = rows.new_zeros((count, *rows.shape[1:]))
    # Destinations are unique and disjoint. Fill fresh storage in-place while
    # retaining autograd edges even when remote sends or receives are empty.
    result.index_put_((targets,), rows.index_select(0, sources))
    result.index_put_((receive,), remote)
    return result.movedim(0, sequence_dim).contiguous()


def redistribute_ring_qkv(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        plan: RingTaskPlan, group: Any, sequence_dim: int, head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange Q/K/V together with one autograd-aware row collective.

    Head counts may differ for GQA. Concatenating heads preserves each token's
    row route and makes backward return all three gradients in one exchange.
    """
    if len(plan.assignments) == 1:
        return query, key, value
    heads = tuple(tensor.shape[head_dim] for tensor in (query, key, value))
    packed = torch.cat((query, key, value), dim=head_dim)
    balanced = redistribute_ring_rows(packed, plan, group, sequence_dim)
    return tuple(balanced.split(heads, dim=head_dim))
