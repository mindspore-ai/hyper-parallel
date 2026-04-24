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
"""Internal layout helpers for DeviceMesh bookkeeping."""
# pylint: disable=W0212
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np


def _flatten_tuple(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    result: list[int] = []
    for item in value:
        result.extend(_flatten_tuple(item))
    return tuple(result)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 0:
        return ()
    strides = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * shape[idx + 1]
    return tuple(strides)


@dataclass(frozen=True)
class _FlatLayout:
    """Canonicalized layout for a single logical DeviceMesh axis."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __init__(self, shape, stride=None) -> None:
        flat_shape = _flatten_tuple(shape)
        flat_stride = _flatten_tuple(stride if stride is not None else _contiguous_strides(flat_shape))
        if len(flat_shape) != len(flat_stride):
            raise ValueError(
                f"shape and stride must have the same length, got {len(flat_shape)} and {len(flat_stride)}"
            )

        normalized_shape: list[int] = []
        normalized_stride: list[int] = []
        for size, step in zip(flat_shape, flat_stride):
            if size < 0:
                raise ValueError(f"shape entries must be non-negative, got {flat_shape}")
            if size == 1:
                continue
            normalized_shape.append(int(size))
            normalized_stride.append(int(step))

        coalesced_shape: list[int] = []
        coalesced_stride: list[int] = []
        for size, step in zip(normalized_shape, normalized_stride):
            if coalesced_shape and coalesced_stride[-1] == step * size:
                coalesced_shape[-1] *= size
                coalesced_stride[-1] = step
            else:
                coalesced_shape.append(size)
                coalesced_stride.append(step)

        object.__setattr__(self, "shape", tuple(coalesced_shape))
        object.__setattr__(self, "stride", tuple(coalesced_stride))

    def numel(self) -> int:
        return math.prod(self.shape)

    def cosize(self) -> int:
        ranks = self.all_ranks_from_zero()
        return max(ranks) + 1 if ranks else 1

    def check_sorted(self) -> bool:
        return tuple(sorted(self.stride, reverse=True)) == self.stride

    def check_orthogonal(self) -> bool:
        if len(self.shape) < 2:
            return True
        stride, shape = zip(*sorted(zip(self.stride, self.shape), reverse=True))
        return all(
            stride[idx] % (stride[idx + 1] * shape[idx + 1]) == 0
            for idx in range(len(stride) - 1)
        )

    def _offset_for_linear_index(self, index: int) -> int:
        """Calculate the offset for a linear index."""
        if index < 0 or index >= self.numel():
            raise IndexError(f"index {index} out of range for layout with numel={self.numel()}")
        if len(self.shape) == 0:
            return 0

        coord = [0] * len(self.shape)
        remainder = index
        for dim in range(len(self.shape) - 1, -1, -1):
            size = self.shape[dim]
            coord[dim] = remainder % size
            remainder //= size
        return int(sum(pos * step for pos, step in zip(coord, self.stride)))

    def all_ranks_from_zero(self) -> list[int]:
        if len(self.shape) == 0:
            return [0]
        return [
            int(sum(coord[dim] * self.stride[dim] for dim in range(len(self.shape))))
            for coord in product(*(range(size) for size in self.shape))
        ]


@dataclass(frozen=True)
class _MeshLayout(Sequence[_FlatLayout]):
    """Logical DeviceMesh layout made of a list of flat axes."""

    axes: tuple[_FlatLayout, ...]

    def __init__(self, axes: Sequence[_FlatLayout]) -> None:
        object.__setattr__(self, "axes", tuple(axes))

    @classmethod
    def from_sizes_strides(
        cls, sizes: tuple[int, ...], strides: tuple[int, ...] | None = None
    ) -> "_MeshLayout":
        """Create a mesh layout from sizes and strides."""
        if strides is None:
            strides = _contiguous_strides(sizes)
        if len(sizes) != len(strides):
            raise ValueError(
                f"sizes and strides must have the same length, got {len(sizes)} and {len(strides)}"
            )
        return cls(tuple(_FlatLayout((size,), (stride,)) for size, stride in zip(sizes, strides)))

    def __len__(self) -> int:
        return len(self.axes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _MeshLayout(self.axes[index])
        return self.axes[index]

    def __iter__(self) -> Iterator[_FlatLayout]:
        return iter(self.axes)

    @property
    def top_level_sizes(self) -> tuple[int, ...]:
        return tuple(axis.numel() for axis in self.axes)

    def numel(self) -> int:
        return math.prod(self.top_level_sizes)

    def cosize(self) -> int:
        return self.collapse().cosize()

    def collapse(self) -> _FlatLayout:
        return _FlatLayout(tuple(axis.shape for axis in self.axes), tuple(axis.stride for axis in self.axes))

    def splice(self, start: int, end: int, layout: "_MeshLayout") -> "_MeshLayout":
        new_axes = list(self.axes)
        new_axes[start:end] = list(layout.axes)
        return _MeshLayout(new_axes)

    def remap_to_numpy(self, rank_map) -> np.ndarray:
        """Remap the layout to a NumPy array."""
        rank_map_np = np.asarray(rank_map).reshape(-1)
        if rank_map_np.size < self.cosize():
            raise AssertionError(
                f"rank_map contains {rank_map_np.size} elements, which is not enough for layout {self}"
            )

        mesh_shape = self.top_level_sizes
        if len(mesh_shape) == 0:
            return np.array(rank_map_np[0], dtype=np.int32)

        result = np.empty(mesh_shape, dtype=rank_map_np.dtype)
        for coord in product(*(range(size) for size in mesh_shape)):
            rank_index = 0
            for axis_index, axis_coord in enumerate(coord):
                rank_index += self.axes[axis_index]._offset_for_linear_index(axis_coord)
            result[coord] = rank_map_np[rank_index]
        return result
