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
"""Bounded rectangles for checking inverse-converted distributed weights."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

Box = tuple[tuple[int, ...], tuple[int, ...]]


def tile_box(offsets: tuple[int, ...], lengths: tuple[int, ...], limit: int) -> Iterator[Box]:
    """Split a rectangle into non-overlapping tiles of at most ``limit`` elements."""
    if limit < 1:
        raise ValueError("Tile limit must be positive")
    if any(size == 0 for size in lengths):
        return
    if math.prod(lengths) <= limit:
        yield offsets, lengths
        return
    axis = next((axis for axis, size in enumerate(lengths) if size > 1), 0)
    step = max(1, limit // math.prod(lengths[axis + 1:]))
    for start in range(0, lengths[axis], step):
        child_offsets = list(offsets)
        child_lengths = list(lengths)
        child_offsets[axis] += start
        child_lengths[axis] = min(step, lengths[axis] - start)
        yield from tile_box(tuple(child_offsets), tuple(child_lengths), limit)


def local_block_box(block: Any, shard_offsets: tuple[int, ...], shard_shape: tuple[int, ...]) -> Box | None:
    """Intersect an inverse conversion block with one live model shard.

    Permutations, splits and stacks map each varying model axis to at most one
    output axis. More complex affine maps fail explicitly instead of skipping data.
    """
    low = [0] * len(block.lengths)
    high = list(block.lengths)
    for size, column in zip(block.lengths, block.steps):
        if size > 1 and sum(step != 0 for step in column) != 1:
            raise ValueError("Inverse conversion has a non-rectangular shard mapping")
    for dim, origin in enumerate(block.base):
        steps = tuple(column[dim] for column in block.steps)
        moving = [(axis, step) for axis, step in enumerate(steps) if step]
        start, stop = shard_offsets[dim], shard_offsets[dim] + shard_shape[dim]
        if not moving:
            if not start <= origin < stop:
                return None
            continue
        if len(moving) != 1 or moving[0][1] != 1:
            raise ValueError("Inverse conversion has a non-rectangular shard mapping")
        axis, _ = moving[0]
        low[axis] = max(low[axis], start - origin)
        high[axis] = min(high[axis], stop - origin)
    if any(left >= right for left, right in zip(low, high)):
        return None
    return (
        tuple(origin + left for origin, left in zip(block.offsets, low)),
        tuple(right - left for left, right in zip(low, high)),
    )


def box_union_volume(boxes: Sequence[Box]) -> int:
    """Count unique elements covered by rectangular shards, including replicas."""
    unique = set((tuple(offsets), tuple(lengths)) for offsets, lengths in boxes if all(lengths))
    if not unique:
        return 0
    if not next(iter(unique))[0]:
        return 1
    endpoints = sorted({point for offsets, lengths in unique for point in (offsets[0], offsets[0] + lengths[0])})
    return sum(
        (right - left) * box_union_volume([
            (offsets[1:], lengths[1:]) for offsets, lengths in unique
            if offsets[0] <= left and offsets[0] + lengths[0] >= right
        ])
        for left, right in zip(endpoints, endpoints[1:])
    )
