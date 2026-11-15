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
"""Differentiable all-to-all-v dispatch of sequence bank intervals."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.compressed_document import (
    compressed_document_offsets, compressed_owner_offsets,
)
from hyper_parallel.distributed.context_parallel.sequence_pack import pack_sequence_rows


@dataclass(frozen=True)
class SequenceDispatchRoute:
    """Row splits for compact intervals assembled from contiguous owner shards.

    Global start/end bound the requested ranges, including gaps when present.
    Self rows stay local. Remote rows are ordered by owner rank, so the local
    shard fits between the received prefix and suffix without tensor indices.
    Backward reverses the exchange and SUMs overlapping consumer contributions.
    """

    local_length: int
    global_start: int
    global_end: int
    local_output_start: int
    send_offsets: tuple[int, ...]
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    needs_exchange: bool
    send_ranges: tuple[tuple[int, int], ...] = ()
    accumulate_fp32: bool = False
    _cached_hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Keep complete route equality while hashing immutable intervals only once."""
        object.__setattr__(self, "_cached_hash", hash((
            self.local_length, self.global_start, self.global_end, self.local_output_start,
            self.send_offsets, self.input_splits, self.output_splits, self.needs_exchange,
            self.send_ranges, self.accumulate_fp32,
        )))

    def __hash__(self) -> int:
        """Return a stable hash without revisiting fragmented document demands."""
        return self._cached_hash

    def owner_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return send slices in destination-rank order, including multi-range peers."""
        return self.send_ranges or tuple(zip(self.send_offsets, self.input_splits))


def _validate_interval_owners(owners: tuple[int, ...], rank: int, intervals: tuple) -> None:
    """Keep each complete owner shard in its consumer bank, including empty shards."""
    if (len(owners) != len(intervals) + 1 or not intervals or owners[0] != 0
            or not 0 <= rank < len(intervals) or any(end < start for start, end in zip(owners, owners[1:]))):
        raise ValueError("Dispatch requires ordered owner offsets and a valid rank")
    for peer, (start, end) in enumerate(intervals):
        if not 0 <= start <= owners[peer] <= owners[peer + 1] <= end <= owners[-1]:
            raise ValueError("Dispatch intervals must cover their complete owner shard within the global bank")


def interval_route(
        local_length: int, rank: int, intervals: tuple[tuple[int, int], ...],
        *, owner_offsets: tuple[int, ...] | None = None,
) -> SequenceDispatchRoute:
    """Derive owner/consumer intersections; all ranks agree whether to exchange.

    Args:
        local_length: Actual row count owned by the current rank, possibly zero.
        rank: Current rank in sequence-shard order.
        intervals: One consumer interval per rank, containing its complete owner shard.
        owner_offsets: Canonical source boundaries; omitted for equal-sized raw shards.

    Returns:
        Per-peer row splits and the unchanged owner's position in its consumer bank.
    """
    owners = (tuple(peer * local_length for peer in range(len(intervals) + 1))
              if owner_offsets is None else owner_offsets)
    _validate_interval_owners(owners, rank, intervals)
    if owners[rank + 1] - owners[rank] != local_length:
        raise ValueError("Local bank length differs from the supplied owner offsets")
    local_start = owners[rank]
    global_start, global_end = intervals[rank]
    offsets, sends, receives = [], [], []
    for peer, (start, end) in enumerate(intervals):
        first = max(local_start, start)
        count = max(0, min(owners[rank + 1], end) - first) if peer != rank else 0
        offsets.append(first - local_start if count else 0)
        sends.append(count)
        receives.append(max(0, min(owners[peer + 1], global_end) - max(owners[peer], global_start))
                        if peer != rank else 0)
    needs_exchange = any(start != owners[peer] or end != owners[peer + 1]
                         for peer, (start, end) in enumerate(intervals))
    return SequenceDispatchRoute(local_length, global_start, global_end, local_start - global_start,
                                 tuple(offsets), tuple(sends), tuple(receives), needs_exchange)


def _intersect_ranges(ranges: tuple, start: int, length: int) -> tuple:
    """Intersect sorted global ranges with one source shard, retaining row order."""
    first = bisect_right(ranges, start, key=lambda item: item[1])
    last = bisect_left(ranges, start + length, key=lambda item: item[0])
    return tuple((max(begin, start) - start, min(end, start + length) - max(begin, start))
                 for begin, end in ranges[first:last])


def _validate_dispatch_intervals(owner_offsets: tuple, rank: int, intervals: tuple) -> None:
    """Validate variable owner coverage before any rank enters a collective."""
    size = len(intervals)
    if (len(owner_offsets) != size+1 or owner_offsets[0] != 0 or not 0 <= rank < size
            or any(end < start for start, end in zip(owner_offsets, owner_offsets[1:]))):
        raise ValueError("Dispatch requires ordered owner offsets and a valid rank")
    for peer, ranges in enumerate(intervals):
        start, end = owner_offsets[peer:peer+2]
        if (any(first < 0 or last <= first or last > owner_offsets[-1] for first, last in ranges)
                or any(left[1] > right[0] for left, right in zip(ranges, ranges[1:]))
                or sum(count for _, count in _intersect_ranges(ranges, start, end-start)) != end-start):
            raise ValueError("Dispatch ranges must be sorted, disjoint, in bounds, and cover the complete local shard")


def _receive_counts(intervals: tuple, owners: tuple, rank: int) -> tuple[int, ...]:
    """Count requested rows by source, excluding the local bypass."""
    return tuple(sum(count for _, count in _intersect_ranges(intervals[rank], owners[peer],
                                                                owners[peer+1]-owners[peer]))
                     if peer != rank else 0 for peer in range(len(intervals)))


def multi_interval_route(local_length: int, rank: int, intervals: tuple[tuple, ...],
                         *, accumulate_fp32: bool = False,
                         owner_offsets: tuple[int, ...] | None = None) -> SequenceDispatchRoute:
    """Plan compact demands, retaining each owner's complete, possibly empty shard.

    Args:
        local_length: Physical K rows on this source rank.
        rank: Local rank in the sequence-ordered group.
        intervals: Sorted, disjoint global ranges required by each destination.
        accumulate_fp32: Sum half-precision consumer gradients in FP32 at owners.
        owner_offsets: Canonical source boundaries; omitted for equal-sized shards.
    """
    size = len(intervals)
    owners = tuple(peer*local_length for peer in range(size+1)) if owner_offsets is None else owner_offsets
    _validate_dispatch_intervals(owners, rank, intervals)
    if owners[rank+1]-owners[rank] != local_length:
        raise ValueError("Local K length differs from the supplied owner offsets")
    local_start = owners[rank]
    sends = tuple(_intersect_ranges(ranges, local_start, local_length) if peer != rank else ()
                  for peer, ranges in enumerate(intervals))
    receives = _receive_counts(intervals, owners, rank)
    needs_exchange = any(sum(end-start for start, end in ranges) > owners[peer+1]-owners[peer]
                         for peer, ranges in enumerate(intervals))
    requested = intervals[rank]
    return SequenceDispatchRoute(
        local_length, requested[0][0] if requested else local_start,
        requested[-1][1] if requested else local_start, sum(receives[:rank]),
        (0,) * size, tuple(sum(count for _, count in ranges) for ranges in sends), receives,
        needs_exchange, tuple(item for ranges in sends for item in ranges), accumulate_fp32,
    )


@dataclass(frozen=True)
class SequenceDispatchGradient:
    """Reverse route for a contiguous differentiable slice of a dispatched bank.

    Rows outside this slice must only be used without gradients. This permits
    Indexer selection to request more K rows than the original owner's KL needs.
    The selected route still covers the source rank's complete local K shard.
    """

    route: SequenceDispatchRoute
    start: int

    @property
    def length(self) -> int:
        """Number of differentiable rows, including the local bypass."""
        return self.route.local_length + sum(self.route.output_splits)


def _dispatch_gradient(grad: torch.Tensor, route: SequenceDispatchRoute, group: Any) -> torch.Tensor:
    """Reverse one demand route, preserving the rank-ordered FP32 owner sum."""
    first, last = route.local_output_start, route.local_output_start + route.local_length
    if not route.needs_exchange:
        return grad[first:last].clone(memory_format=torch.contiguous_format)
    send = torch.cat((grad[:first], grad[last:]), dim=0)
    received = grad.new_empty((sum(route.input_splits), *grad.shape[1:]))
    handle = dist.all_to_all_single(
        received, send, output_split_sizes=list(route.input_splits),
        input_split_sizes=list(route.output_splits), group=group, async_op=True,
    )
    accumulation_dtype = (torch.float32 if route.accumulate_fp32 and grad.dtype in (torch.float16, torch.bfloat16)
                          else grad.dtype)
    # copy=True protects upstream gradients without allocating a second FP32 copy.
    local = grad[first:last].to(dtype=accumulation_dtype, copy=True, memory_format=torch.contiguous_format)
    handle.wait()
    cursor = 0
    for offset, count in route.owner_ranges():
        if count:
            local.narrow(0, offset, count).add_(received.narrow(0, cursor, count))
            cursor += count
    return local.to(grad.dtype)


class _SequenceDispatchWait(torch.autograd.Function):
    """Attach remote bank rows to their owners without retaining send buffers."""

    @staticmethod
    def forward(ctx: Any, local: torch.Tensor, received: torch.Tensor, handle: Any,
                route: SequenceDispatchRoute, group: Any, gradient: SequenceDispatchGradient | None) -> torch.Tensor:
        """Wait before exposing received data and insert the uncommunicated local rows."""
        ctx.route = route
        ctx.gradient = gradient
        ctx.group = group
        handle.wait()
        before = route.local_output_start
        return torch.cat((received[:before], local, received[before:]), dim=0)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple:
        """Reverse dispatch, summing local contributions then remote peers in rank order."""
        route = ctx.route
        if ctx.gradient is not None:
            route = ctx.gradient.route
            grad = grad.narrow(0, ctx.gradient.start, ctx.gradient.length)
        return _dispatch_gradient(grad, route, ctx.group), None, None, None, None, None


class _SequenceDispatchOwnerWait(torch.autograd.Function):
    """Expose selection storage separately from the original owner's KL autograd edge."""

    @staticmethod
    def forward(ctx: Any, local: torch.Tensor, received: torch.Tensor | None, handle: Any,
                route: SequenceDispatchRoute, group: Any, gradient: SequenceDispatchGradient) -> tuple:
        """Share forward storage without expanding a KL gradient to worker-only rows."""
        ctx.route = gradient.route
        ctx.group = group
        if received is None:
            bank = local.detach()
        else:
            handle.wait()
            before = route.local_output_start
            bank = torch.cat((received[:before], local, received[before:]), dim=0)
        owner = bank.narrow(0, gradient.start, gradient.length)
        ctx.mark_non_differentiable(bank)
        ctx.set_materialize_grads(False)
        return bank, owner

    @staticmethod
    def backward(ctx: Any, _selection_grad: None, owner_grad: torch.Tensor) -> tuple:
        """Shared KL consumers accumulate on the same owner edge before exchange."""
        return _dispatch_gradient(owner_grad, ctx.route, ctx.group), None, None, None, None, None


@dataclass
class AsyncSequenceDispatch:
    """Single-use launch handle owning send storage until the communication completes."""

    local: torch.Tensor
    send: torch.Tensor | None
    received: torch.Tensor | None
    handle: Any
    route: SequenceDispatchRoute
    group: Any
    sequence_dim: int
    gradient: SequenceDispatchGradient | None = None
    waited: bool = False

    def wait(self) -> torch.Tensor:
        """Finish the exchange and return the compact bank in its original layout."""
        if self.waited:
            raise RuntimeError("a sequence dispatch handle cannot be waited twice")
        self.waited = True
        output = self.local
        if self.received is not None:
            output = _SequenceDispatchWait.apply(
                self.local, self.received, self.handle, self.route, self.group, self.gradient,
            )
        self.send = None
        self.received = None
        return output.movedim(0, self.sequence_dim).contiguous()

    def wait_with_gradient(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a detached selection bank and a differentiable owner prefix.

        Requires an explicit restricted gradient route. Full/Reindex layers must
        share the second output for KL; only no-grad selection uses the first.
        This avoids allocating a zero gradient for worker-only selection rows.
        """
        if self.waited:
            raise RuntimeError("a sequence dispatch handle cannot be waited twice")
        if self.gradient is None:
            raise ValueError("separate selection and KL outputs require a dispatch gradient route")
        self.waited = True
        bank, owner = _SequenceDispatchOwnerWait.apply(
            self.local, self.received, self.handle, self.route, self.group, self.gradient,
        )
        self.send = None
        self.received = None
        return (bank.movedim(0, self.sequence_dim).contiguous(),
                owner.movedim(0, self.sequence_dim).contiguous())


def async_sequence_dispatch(tensor: torch.Tensor, sequence_dim: int,
                            route: SequenceDispatchRoute, group: Any, *,
                            gradient: SequenceDispatchGradient | None = None) -> AsyncSequenceDispatch:
    """Launch variable row exchange with an autograd edge attached at wait.

    Args:
        tensor: Contiguous local owner shard of a raw, compressed or Index K bank.
        sequence_dim: Sequence dimension; feature dimensions are transported unchanged.
        route: Immutable per-batch routing agreed by every member of the CP group.
        group: Existing framework-owned CP process group, in sequence-shard rank order.
        gradient: Optional restricted reverse route; all other output rows must have zero gradient.
    """
    if not -tensor.ndim <= sequence_dim < tensor.ndim:
        raise ValueError("sequence dispatch dimension is outside the input tensor dimensions")
    sequence_dim %= tensor.ndim
    if tensor.shape[sequence_dim] != route.local_length:
        raise ValueError("sequence dispatch owner length does not match the bank's physical layout")
    if gradient is not None:
        total = route.local_length + sum(route.output_splits)
        if (gradient.route.local_length != route.local_length or gradient.start < 0
                or gradient.start + gradient.length > total
                or (route.local_length and
                    gradient.start + gradient.route.local_output_start != route.local_output_start)):
            raise ValueError("Dispatch gradient slice must retain the complete local shard in its original position")
    local = tensor.movedim(sequence_dim, 0).contiguous()
    if not route.needs_exchange:
        return AsyncSequenceDispatch(local, None, None, None, route, group, sequence_dim, gradient)
    send = pack_sequence_rows(local.detach(), route)
    received = local.new_empty((sum(route.output_splits), *local.shape[1:]))
    handle = dist.all_to_all_single(
        received, send, output_split_sizes=list(route.output_splits),
        input_split_sizes=list(route.input_splits), group=group, async_op=True,
    )
    return AsyncSequenceDispatch(local, send, received, handle, route, group, sequence_dim, gradient)


@lru_cache(maxsize=128)
def sliding_window_route(boundaries: tuple[int, ...], local_length: int, size: int,
                         rank: int, window: int) -> SequenceDispatchRoute:
    """Plan the same-document left halo; every rank agrees whether to exchange."""
    if local_length <= 0 or size <= 0 or not 0 <= rank < size or window <= 0:
        raise ValueError("SWA dispatch requires positive dimensions and a valid CP rank")
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != local_length * size
            or any(end <= start for start, end in zip(boundaries, boundaries[1:]))):
        raise ValueError("SWA document boundaries must span the full contiguous CP sequence")
    intervals = tuple(
        (max(boundaries[bisect_right(boundaries, peer * local_length) - 1], peer * local_length - window + 1),
         (peer + 1) * local_length)
        for peer in range(size)
    )
    return interval_route(local_length, rank, intervals)


@lru_cache(maxsize=128)
def document_prefix_route(boundaries: tuple[int, ...], local_length: int, size: int,
                          rank: int, ratio: int) -> SequenceDispatchRoute:
    """Route natural-document main KV prefixes using complete-block ownership."""
    if ratio not in (1, 2) or local_length <= 0 or size <= 0 or not 0 <= rank < size:
        raise ValueError("Main KV dispatch requires a supported ratio and equal positive raw CP shards")
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != local_length * size
            or any(end <= start for start, end in zip(boundaries, boundaries[1:]))):
        raise ValueError("Main KV document boundaries must span the complete contiguous CP sequence")
    keys = compressed_document_offsets(boundaries, ratio)
    owners = compressed_owner_offsets(boundaries, ratio, local_length)
    intervals = tuple(
        (keys[bisect_right(boundaries, peer * local_length) - 1], owners[peer + 1])
        for peer in range(size)
    )
    return interval_route(owners[rank + 1] - owners[rank], rank, intervals, owner_offsets=owners)
