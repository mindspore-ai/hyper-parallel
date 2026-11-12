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
"""Overlap selected transfers without exceeding the planned resident budget."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from itertools import accumulate

from hyper_parallel.auto_parallel.hyper_offload.planning._peak_model import EvictionGap


@dataclass
class TransferPlacement:
    """D2H post-op, device release post-op, and H2D pre-op positions."""

    gap: EvictionGap
    copy_after: int
    release_after: int
    prefetch_before: int
    release_deadline: int | None = None


def mask_communications(
    selected: list[EvictionGap], resident_bytes: list[int], limit: int
) -> list[TransferPlacement]:
    """Delay device releases and advance prefetches within spare device memory.

    The input profile describes selected gaps before overlap and is updated
    in place. A snapshot cannot cross an earlier selected gap of the same
    storage: the baseline runtime must restore and retire that snapshot first.
    Runtime events remain responsible for actual transfer completion.
    """
    placements = []
    restored_at = {}
    for gap in sorted(
        selected, key=lambda item: (item.start, item.end, item.storage_id)
    ):
        copy_after = max(
            gap.copy_start, restored_at.get(gap.storage_id, gap.copy_start)
        )
        placements.append(TransferPlacement(gap, copy_after, gap.start, gap.end))
        restored_at[gap.storage_id] = gap.end

    # Preserve D2H issue order when delaying release fences on one copy stream.
    next_release = len(resident_bytes)
    for placement in sorted(
        placements, key=lambda item: (item.copy_after, item.gap.start), reverse=True
    ):
        gap = placement.gap
        upper = min(gap.end - 1, next_release)
        while placement.release_after < upper:
            slot = placement.release_after + 1
            if resident_bytes[slot] + gap.size_bytes > limit:
                break
            resident_bytes[slot] += gap.size_bytes
            placement.release_after = slot
        next_release = placement.release_after

    previous_prefetch = 0
    for placement in sorted(
        placements, key=lambda item: (item.gap.end, item.gap.start, item.gap.storage_id)
    ):
        gap = placement.gap
        lower = max(placement.release_after + 1, previous_prefetch)
        while placement.prefetch_before > lower:
            slot = placement.prefetch_before - 1
            if resident_bytes[slot] + gap.size_bytes > limit:
                break
            resident_bytes[slot] += gap.size_bytes
            placement.prefetch_before = slot
        previous_prefetch = placement.prefetch_before
    return placements


def limit_release_delay(
    placements: list[TransferPlacement], durations_ms: list[float], bandwidth_gbps: float
) -> None:
    """Estimate when to begin nonblocking source retirement checks.

    Preserve the mask's hard memory deadline separately from the estimated
    completion position. Warmup timing may overestimate replay computation,
    so an estimate must not force compute to wait for unfinished transfers.
    Missing timing keeps the memory-only mask. H2D positions stay unchanged.
    """
    if not placements:
        return
    if (
        not math.isfinite(bandwidth_gbps)
        or bandwidth_gbps <= 0
        or any(not math.isfinite(value) or value < 0 for value in durations_ms)
        or not any(durations_ms)
    ):
        return
    ends = list(accumulate(durations_ms))
    copy_complete_ms = 0.0
    for placement in sorted(
        placements, key=lambda item: (item.copy_after, item.gap.storage_id)
    ):
        transfer_ms = placement.gap.size_bytes * 1000 / (bandwidth_gbps * 1024**3)
        copy_complete_ms = max(copy_complete_ms, ends[placement.copy_after]) + transfer_ms
        ready = bisect_left(
            ends, copy_complete_ms, placement.gap.start, placement.release_after + 1
        )
        placement.release_deadline = placement.release_after
        placement.release_after = min(ready, placement.release_deadline)


def release_checkpoints(placement: TransferPlacement) -> list[int]:
    """Return sparse completion checks strictly before the memory deadline.

    Exponentially spaced operator boundaries bound event-query overhead when
    the host runs ahead of the copy stream. All checkpoints are after the
    last source use; the original release still enforces the memory budget.
    """
    deadline = placement.release_deadline
    if deadline is None:
        return []
    checkpoints = []
    position = placement.release_after
    distance = max(1, position - placement.copy_after)
    while position < deadline:
        checkpoints.append(position)
        position += distance
        distance *= 2
    return checkpoints
