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
"""Bounded immutable address plans for one-shot sequence send-buffer gathering."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
from typing import TYPE_CHECKING

import numpy as np
import torch  # pylint: disable=forbidden-backend-import

if TYPE_CHECKING:
    from hyper_parallel.distributed.context_parallel.sequence_dispatch import SequenceDispatchRoute


@dataclass(frozen=True)
class SequencePackPlan:
    """Host-only source intervals in peer order; never retain bank values."""

    source_rows: int
    ranges: tuple[tuple[int, int], ...]
    output_rows: int
    _cached_hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Hash fragmented intervals once while preserving complete equality."""
        object.__setattr__(self, "_cached_hash", hash((self.source_rows, self.ranges, self.output_rows)))

    def __hash__(self) -> int:
        """Return the immutable plan hash."""
        return self._cached_hash


@lru_cache(maxsize=128)
def prepare_sequence_pack(route: SequenceDispatchRoute) -> SequencePackPlan:
    """Prepare coalesced addresses once without changing peer splits or row order."""
    ranges = []
    for start, count in route.owner_ranges():
        if count < 0 or (count and (start < 0 or start + count > route.local_length)):
            raise ValueError("Sequence pack ranges must lie within the local source bank")
        if count:
            if ranges and ranges[-1][0] + ranges[-1][1] == start:
                previous, length = ranges[-1]
                ranges[-1] = (previous, length + count)
            else:
                ranges.append((start, count))
    output_rows = sum(count for _, count in ranges)
    if output_rows != sum(route.input_splits):
        raise ValueError("Sequence pack ranges do not match collective input splits")
    return SequencePackPlan(route.local_length, tuple(ranges), output_rows)


def _create_row_indices(plan: SequencePackPlan, device: torch.device) -> torch.Tensor:
    """Expand addresses in bulk; synchronous transfer completes before cache publication."""
    if not plan.ranges:
        return torch.empty(0, device=device, dtype=torch.long)
    intervals = np.asarray(plan.ranges, dtype=np.int64)
    starts, counts = intervals[:, 0], intervals[:, 1]
    offsets = np.cumsum(counts) - counts
    rows = np.repeat(starts - offsets, counts)
    rows += np.arange(plan.output_rows, dtype=np.int64)
    return torch.from_numpy(rows).to(device=device)


class _PackRowsCache:
    """Bound immutable row-address storage and isolate producer streams."""

    def __init__(self, maximum_bytes: int) -> None:
        """Create a byte-bounded cache of immutable address vectors."""
        self.maximum_bytes = maximum_bytes
        self.retained_bytes = 0
        self.entries = OrderedDict()
        self.lock = RLock()

    @torch.inference_mode(False)
    def get(self, plan: SequencePackPlan, device: torch.device) -> torch.Tensor:
        """Return a ready address vector; oversized vectors remain uncached."""
        stream = None if device.type == "cpu" else torch.get_device_module(device).current_stream(device)
        key = (plan, device, stream)
        with self.lock:
            cached = self.entries.get(key)
            if cached is not None:
                self.entries.move_to_end(key)
                return cached
            value = _create_row_indices(plan, device)
            size = value.untyped_storage().nbytes()
            if size <= self.maximum_bytes:
                while self.entries and (self.retained_bytes + size > self.maximum_bytes or len(self.entries) >= 32):
                    _, removed = self.entries.popitem(last=False)
                    self.retained_bytes -= removed.untyped_storage().nbytes()
                self.entries[key] = value
                self.retained_bytes += size
            return value

    def clear(self) -> None:
        """Release cached addresses without invalidating in-flight tensor references."""
        with self.lock:
            self.entries.clear()
            self.retained_bytes = 0


_PACK_ROWS = _PackRowsCache(64 * 1024 * 1024)


def pack_sequence_rows(local: torch.Tensor, route: SequenceDispatchRoute) -> torch.Tensor:
    """Gather a fresh contiguous send buffer without per-interval tensor operations.

    Only immutable addresses are cached. The dispatch caller retains the returned
    buffer until the collective finishes and attaches autograd at its wait node.
    """
    plan = prepare_sequence_pack(route)
    if local.shape[0] != plan.source_rows:
        raise ValueError("Sequence pack source length differs from its prepared route")
    if not plan.output_rows:
        return local[:0]
    return local.index_select(0, _PACK_ROWS.get(plan, local.device))
