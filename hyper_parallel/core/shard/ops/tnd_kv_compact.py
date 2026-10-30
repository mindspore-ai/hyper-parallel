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
"""Document-local causal KV intervals, independent of folding or a particular kernel."""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from functools import lru_cache
from numbers import Integral
from typing import Any, Optional, Sequence

import numpy as np

from hyper_parallel.platform import get_platform

platform = get_platform()
_INDEX_CACHE = OrderedDict()


@dataclass(frozen=True)
class TNDCompactBlock:
    """Immutable natural-coordinate query/KV interval and matching document ends."""

    query_interval: tuple
    key_interval: tuple
    query_ends: tuple
    key_ends: tuple


@lru_cache(maxsize=256)
def plan_tnd_kv(ends: tuple, full_len: int, queries: tuple) -> tuple:
    """Plan independent causal documents for any nonempty natural Q intervals.

    Args:
        ends: Global, nondecreasing document ends covering ``full_len``. Repeated
            ends represent empty documents or metadata padding.
        full_len: Complete natural-order KV length.
        queries: Immutable tuple of nonempty half-open Q intervals. No shard
            count, equal-size split, or two-block/folding assumption is needed.

    Returns:
        Tuple of TNDCompactBlock, preserving query interval order.

    Raises:
        ValueError: Metadata does not cover the sequence or an interval is invalid.

    Note:
        This API requires causal, document-isolated attention. Bidirectional or
        cross-document masks need a different interval policy.
    """
    if (not isinstance(full_len, Integral) or isinstance(full_len, bool)
            or any(not isinstance(end, Integral) or isinstance(end, bool) for end in ends)
            or not ends or full_len <= 0 or ends[-1] != full_len
            or any(a > b for a, b in zip((0,) + ends, ends))):
        raise ValueError("Compact TND KV requires monotonic document ends covering the full sequence.")
    documents = tuple(dict.fromkeys(end for end in ends if end > 0))
    result = []
    for start, stop in queries:
        if (any(not isinstance(edge, Integral) or isinstance(edge, bool) for edge in (start, stop))
                or not 0 <= start < stop <= full_len):
            raise ValueError("Compact TND query intervals must satisfy 0 <= start < stop <= full_len.")
        first, last = bisect_right(documents, start), bisect_left(documents, stop)
        origin = documents[first - 1] if first else 0
        selected = tuple(min(end, stop) for end in documents[first:last + 1])
        result.append(TNDCompactBlock((start, stop), (origin, stop),
                                      tuple(end - start for end in selected),
                                      tuple(end - origin for end in selected)))
    return tuple(result)


def _indices(values, ref):
    """Build device-local chunk indices through the platform abstraction."""
    backend = type(ref).__module__.split('.')[0]
    key = (backend, tuple(values))
    result = _INDEX_CACHE.get(key)
    if result is None or type(result).__module__.split('.')[0] != backend:
        result = platform.from_numpy(np.asarray(values, dtype=np.int32))
        _INDEX_CACHE[key] = result
        if len(_INDEX_CACHE) > 256:
            _INDEX_CACHE.popitem(last=False)
    return result.to(ref.device) if backend == 'torch' else result


def _zeros(ref, length):
    """Allocate an empty token span with reference trailing dimensions."""
    return platform.zeros((length,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)


def _select_interval(value, interval, order):
    """Select one interval, avoiding chunk copies for naturally ordered KV."""
    start, stop = interval
    if order == (0,):
        return value.narrow(0, start, stop - start)
    width = value.shape[0] // len(order)
    first, last = start // width, (stop + width - 1) // width
    chunks = value.reshape((len(order), width) + tuple(value.shape[1:]))
    selected = chunks.index_select(0, _indices(order[first:last], value))
    selected = selected.reshape(((last - first) * width,) + tuple(value.shape[1:]))
    return selected.narrow(0, start - first * width, stop - start)


def _merge_interval_slice(active, start, stop, ref):
    """Sum one nonoverlapping segment without a full-length scatter buffer."""
    slices = [grad.narrow(0, start - interval[0], stop - start)
              for interval, grad in active if interval[0] <= start and stop <= interval[1]]
    merged = slices[0] if slices else _zeros(ref, stop - start)
    for value in slices[1:]:
        merged = merged + value
    return merged


def restore_kv_gradients(gradients: Sequence[Any], intervals: tuple, order: tuple, full_len: int) -> Any:
    """Sum overlapping compact gradients and restore their saved physical layout.

    Uses slices, BF16 adds and chunk gathers instead of a token index_add that
    promotes an entire long-sequence gradient to FP32 on Ascend.
    """
    if len(gradients) != len(intervals):
        raise ValueError("Compact KV gradients must match the saved interval count.")
    active = [(interval, grad) for interval, grad in zip(intervals, gradients) if grad is not None]
    if not active:
        return None
    ref = active[0][1]
    if order == (0,) and len(active) == 1:
        (start, stop), grad = active[0]
        if start == 0 and stop == full_len:
            return grad
        return platform.cat((_zeros(ref, start), grad, _zeros(ref, full_len - stop)), dim=0)
    if order == (0,):
        edges = sorted({0, full_len, *(edge for interval, _ in active for edge in interval)})
        return platform.cat(tuple(_merge_interval_slice(active, start, stop, ref)
                                  for start, stop in zip(edges[:-1], edges[1:])), dim=0)
    width = full_len // len(order)
    first = min(interval[0] for interval, _ in active) // width
    last = (max(interval[1] for interval, _ in active) + width - 1) // width
    edges = sorted({first * width, last * width, *(edge for interval, _ in active for edge in interval)})
    parts = [_merge_interval_slice(active, start, stop, ref) for start, stop in zip(edges[:-1], edges[1:])]
    parts.append(_zeros(ref, width))
    merged = platform.cat(parts, dim=0).reshape((last - first + 1, width) + tuple(ref.shape[1:]))
    inverse = [last - first] * len(order)
    for chunk in range(first, last):
        inverse[order[chunk]] = chunk - first
    return merged.index_select(0, _indices(inverse, ref)).reshape((full_len,) + tuple(ref.shape[1:]))


class _CompactIntervals(platform.Function):
    """Select any number of intervals with an immutable per-forward adjoint."""

    @staticmethod
    def forward(ctx: Any, value: Any, intervals: tuple, order: tuple) -> tuple:
        """Select intervals in natural coordinates from the physical KV layout."""
        ctx.compact_args = (intervals, order, value.shape[0])
        return tuple(_select_interval(value, interval, order) for interval in intervals)

    @staticmethod
    def backward(ctx: Any, *gradients: Any) -> tuple:
        """Restore the saved layout even after another microbatch has run."""
        return restore_kv_gradients(gradients, *ctx.compact_args), None, None


@lru_cache(maxsize=256)
def _validate_intervals(full_len, intervals, order):
    """Validate immutable host plans once rather than sorting on every kernel call."""
    if (not order or any(not isinstance(slot, Integral) or isinstance(slot, bool) for slot in order)
            or sorted(order) != list(range(len(order))) or full_len % len(order)
            or not intervals or any(
                any(not isinstance(edge, Integral) or isinstance(edge, bool) for edge in (start, stop))
                or not 0 <= start < stop <= full_len for start, stop in intervals)):
        raise ValueError("Compact KV selection needs valid intervals and a complete physical chunk permutation.")


def select_kv_intervals(value: Any, intervals: tuple, order: Optional[tuple] = None) -> tuple:
    """Select natural KV intervals with correct overlap accumulation in backward.

    Args:
        value: Full KV tensor with token axis zero.
        intervals: Immutable half-open intervals in natural coordinates.
        order: Natural chunk ID to physical slot mapping, or None for natural KV.

    Returns:
        One tensor per interval, preserving the requested order.
    """
    order = (0,) if order is None else order
    full_len = value.shape[0]
    _validate_intervals(full_len, intervals, order)
    if order == (0,) and intervals == ((0, full_len),):
        return (value,)
    return _CompactIntervals.apply(value, intervals, order)
