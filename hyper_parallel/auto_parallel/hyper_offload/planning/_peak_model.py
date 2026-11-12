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
"""Storage lifetimes and safe eviction gaps for peak planning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, pairwise

from hyper_parallel.auto_parallel.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
)


@dataclass(frozen=True)
class EvictionGap:
    """One idle interval; endpoints must remain device resident."""

    storage_id: int
    size_bytes: int
    start: int
    end: int
    copy_start: int
    first: int
    last: int


@dataclass
class PeakProfile:
    """Per-operator residency, safe gaps, and terminal releases."""

    resident_bytes: list[int]
    gaps: list[EvictionGap]
    last_uses: dict[int, int]


def build_profile(trace: ActivationTrace) -> PeakProfile:
    """Account for each storage once, including retained output lifetimes.

    READ/WRITE pairs at the same operator share a residency slot. Every WRITE
    is conservatively a snapshot barrier, including view outputs recorded as
    writes by the baseline tracer.
    """
    accesses = {}
    for op_id, op in enumerate(trace.ops):
        for access in op.accesses:
            if access.op_id != op_id:
                raise ValueError("storage access op_id must match its trace position")
            if access.kind not in (AccessKind.READ, AccessKind.WRITE):
                raise ValueError("unknown storage access kind")
            accesses.setdefault(access.storage_id, {}).setdefault(op_id, set()).add(
                access.kind
            )
    delta = [0] * (len(trace.ops) + 1)
    gaps, last_uses = [], {}
    for sid, by_op in accesses.items():
        size = trace.storage_sizes.get(sid)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"storage {sid} requires a nonnegative integer byte size")
        positions = sorted(by_op)
        first, last = positions[0], positions[-1]
        resident_end = len(trace.ops) - 1 if sid in trace.retained_sids else last
        delta[first] += size
        delta[resident_end + 1] -= size
        last_uses[sid] = last
        if sid in trace.retained_sids:
            continue
        latest_write = first
        for start, end in pairwise(positions):
            if AccessKind.WRITE in by_op[start]:
                latest_write = start
            if end > start + 1 and size:
                gaps.append(
                    EvictionGap(sid, size, start, end, latest_write, first, last)
                )
    return PeakProfile(list(accumulate(delta[:-1])), gaps, last_uses)
