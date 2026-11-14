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
from bisect import bisect_left, bisect_right
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

    # Spend spare residency on H2D overlap first; delaying D2H release is
    # useful only when it does not push a prefetch past its source retirement.
    next_release = len(resident_bytes)
    for placement in sorted(
        placements, key=lambda item: (item.copy_after, item.gap.start), reverse=True
    ):
        gap = placement.gap
        upper = min(gap.end - 1, placement.prefetch_before - 1, next_release)
        while placement.release_after < upper:
            slot = placement.release_after + 1
            if resident_bytes[slot] + gap.size_bytes > limit:
                break
            resident_bytes[slot] += gap.size_bytes
            placement.release_after = slot
        next_release = placement.release_after
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


def limit_prefetch_advance(
    placements: list[TransferPlacement], resident_bytes: list[int],
    durations_ms: list[float], bandwidth_gbps: float,
) -> None:
    """Keep H2D lead time proportional to queued transfer work.

    Memory headroom alone can move many restores hundreds of milliseconds
    before their consumers. Holding those destinations fragments the allocator
    and competes with backward workspaces. Schedule the copy queue backwards
    from consumer deadlines with a margin for faster replay than warmup.
    Moving a prefetch later only reduces the memory-only mask's residency.
    """
    if (
        not placements or not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0
        or any(not math.isfinite(value) or value < 0 for value in durations_ms)
        or not any(durations_ms)
    ):
        return
    starts = [0.0, *accumulate(durations_ms)]
    next_start = math.inf
    for placement in sorted(
        placements, key=lambda item: (item.gap.end, item.gap.storage_id), reverse=True
    ):
        transfer_ms = placement.gap.size_bytes * 1000 / (bandwidth_gbps * 1024**3)
        # Warmup dispatch and synchronization can overestimate the time available
        # in replay. Keep several copy durations of lead without filling all slack.
        start_ms = min(starts[placement.gap.end], next_start) - 4.0 * transfer_ms
        target = max(placement.prefetch_before, bisect_right(starts, start_ms) - 1)
        target = min(target, placement.gap.end)
        for slot in range(placement.prefetch_before, target):
            resident_bytes[slot] -= placement.gap.size_bytes
        placement.prefetch_before = target
        next_start = starts[target]


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
