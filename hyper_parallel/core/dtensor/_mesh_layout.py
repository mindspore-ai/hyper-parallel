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

import math
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

import numpy as np


IntTuple = Union[int, tuple["IntTuple", ...]]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_tuple(value: IntTuple) -> tuple[IntTuple, ...]:
    return value if isinstance(value, tuple) else (value,)


def _flatten_inttuple(value: IntTuple) -> tuple[int, ...]:
    if _is_int(value):
        return (value,)
    flattened: list[int] = []
    for item in value:
        flattened.extend(_flatten_inttuple(item))
    return tuple(flattened)


def _match_structure(shape: IntTuple, stride: IntTuple) -> bool:
    if _is_int(shape):
        return _is_int(stride)
    if not isinstance(shape, tuple) or not isinstance(stride, tuple) or len(shape) != len(stride):
        return False
    return all(_match_structure(sub_shape, sub_stride) for sub_shape, sub_stride in zip(shape, stride))


def _numel(value: IntTuple) -> int:
    return int(np.prod(np.array(_flatten_inttuple(value), dtype=np.int64)))


def _contiguous_strides(mesh_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(mesh_shape) == 0:
        return ()
    strides = [1] * len(mesh_shape)
    for idx in range(len(mesh_shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * mesh_shape[idx + 1]
    return tuple(strides)


def _scale_inttuple(value: IntTuple, factor: int) -> IntTuple:
    if _is_int(value):
        return int(value) * factor
    return tuple(_scale_inttuple(item, factor) for item in value)


def _enumerate_offsets(shape: IntTuple, stride: IntTuple) -> list[int]:
    if _is_int(shape):
        return [i * int(stride) for i in range(shape)]

    offsets = [0]
    for sub_shape, sub_stride in zip(_as_tuple(shape), _as_tuple(stride)):
        dim_offsets = _enumerate_offsets(sub_shape, sub_stride)
        offsets = [base + dim_offset for base in offsets for dim_offset in dim_offsets]
    return offsets


def _canonicalize_axis(shape, stride) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalize one logical axis into a flattened shape/stride pair."""
    flat_shape = _flatten_inttuple(shape)
    flat_stride = _flatten_inttuple(stride if stride is not None else _contiguous_strides(flat_shape))
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
    return tuple(coalesced_shape), tuple(coalesced_stride)


def _nested_from_flat(value: tuple[int, ...]) -> IntTuple:
    if len(value) == 1:
        return value[0]
    return tuple(value)


@dataclass(frozen=True)
class _FlatLayout:
    """Canonicalized layout for one logical DeviceMesh axis."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __init__(self, shape: IntTuple, stride: Optional[IntTuple] = None) -> None:
        """Canonicalize *shape* and *stride* into a frozen flat layout."""
        flat_shape, flat_stride = _canonicalize_axis(shape, stride)
        object.__setattr__(self, "shape", flat_shape)
        object.__setattr__(self, "stride", flat_stride)

    def numel(self) -> int:
        """Return the total number of elements in this layout."""
        return math.prod(self.shape) if len(self.shape) > 0 else 1

    def cosize(self) -> int:
        """Return the size of the smallest contiguous block containing all ranks."""
        ranks = self.all_ranks_from_zero()
        return max(ranks) + 1 if ranks else 1

    def check_sorted(self) -> bool:
        """Return True if strides are in descending order."""
        return tuple(sorted(self.stride, reverse=True)) == self.stride

    def check_orthogonal(self) -> bool:
        """Return True if each axis is independent (strides are orthogonal)."""
        if len(self.shape) < 2:
            return True
        stride, shape = zip(*sorted(zip(self.stride, self.shape), reverse=True))
        return all(
            stride[idx] % (stride[idx + 1] * shape[idx + 1]) == 0
            for idx in range(len(stride) - 1)
        )

    def all_ranks_from_zero(self) -> list[int]:
        """List every rank offset assuming the base is zero."""
        if len(self.shape) == 0:
            return [0]
        return [
            int(sum(coord[dim] * self.stride[dim] for dim in range(len(self.shape))))
            for coord in np.ndindex(self.shape)
        ]


class _MeshLayout:
    """Minimal layout helper for DeviceMesh slicing, flattening, and concatenation."""

    def __init__(
        self,
        shape_or_axes: Union[IntTuple, list[_FlatLayout], tuple[_FlatLayout, ...]],
        stride: Optional[IntTuple] = None,
    ) -> None:
        """Build a layout from a nested shape/stride pair or a list of flat axes."""
        if stride is None and isinstance(shape_or_axes, (list, tuple)) and all(
            isinstance(axis, _FlatLayout) for axis in shape_or_axes
        ):
            axes = list(shape_or_axes)
            shape = tuple(_nested_from_flat(axis.shape) for axis in axes)
            stride = tuple(_nested_from_flat(axis.stride) for axis in axes)
        else:
            shape = shape_or_axes
        if not _match_structure(shape, stride):
            raise ValueError(f"shape {shape} and stride {stride} do not match")
        self.shape: IntTuple = shape
        self.stride: IntTuple = stride

    @classmethod
    def from_sizes_strides(
            cls,
            sizes: tuple[int, ...],
            strides: Optional[tuple[int, ...]] = None,
    ) -> "_MeshLayout":
        """Create a layout from flat sizes and optional strides."""
        if strides is None:
            strides = _contiguous_strides(sizes)
        return cls(sizes, strides)

    @property
    def sizes(self) -> IntTuple:
        """Return the (possibly nested) shape tuple."""
        return self.shape

    @property
    def strides(self) -> IntTuple:
        """Return the (possibly nested) stride tuple."""
        return self.stride

    @property
    def axes(self) -> tuple[_FlatLayout, ...]:
        """Return each top-level dimension as a separate flat layout."""
        return tuple(self[idx].collapse() for idx in range(len(self)))

    def __len__(self) -> int:
        """Return the number of top-level dimensions."""
        return len(self.shape) if isinstance(self.shape, tuple) else 1

    def __iter__(self) -> Iterator["_MeshLayout"]:
        """Yield each top-level dimension as its own layout."""
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> "_MeshLayout":
        """Select a single top-level dimension by index."""
        if isinstance(self.shape, tuple):
            if idx < -len(self.shape) or idx >= len(self.shape):
                raise IndexError(
                    f"Dim {idx} is out of range for layout with {len(self.shape)} dimensions."
                )
            return _MeshLayout(self.shape[idx], self.stride[idx])
        if idx not in (0, -1):
            raise IndexError("Dim is out of range for 1D layout.")
        return _MeshLayout(self.shape, self.stride)

    def __eq__(self, other: object) -> bool:
        """Return True if *other* has the same shape and stride."""
        if not isinstance(other, _MeshLayout):
            return False
        return self.shape == other.shape and self.stride == other.stride

    def __repr__(self) -> str:
        """Return a printable representation of this layout."""
        return f"_MeshLayout(shape={self.shape}, stride={self.stride})"

    def numel(self) -> int:
        """Return the total number of elements in this layout."""
        return _numel(self.shape)

    @property
    def top_level_sizes(self) -> tuple[int, ...]:
        """Return the element count of each top-level dimension."""
        return tuple(self[idx].numel() for idx in range(len(self)))

    def all_ranks_from_zero(self) -> list[int]:
        """List every rank offset assuming the base is zero."""
        return _enumerate_offsets(self.shape, self.stride)

    def check_non_overlap(self) -> bool:
        """Return True if no rank appears more than once in this layout."""
        ranks = self.all_ranks_from_zero()
        return len(ranks) == len(set(ranks))

    def coalesce(self) -> "_MeshLayout":
        """Merge adjacent contiguous axes while preserving the represented layout."""
        if _is_int(self.shape):
            return self

        coalesced_shapes: list[IntTuple] = []
        coalesced_strides: list[IntTuple] = []
        for shape, stride in zip(self.shape, self.stride):
            child = _MeshLayout(shape, stride).coalesce()
            coalesced_shapes.append(child.shape)
            coalesced_strides.append(child.stride)

        merged_shapes: list[IntTuple] = []
        merged_strides: list[IntTuple] = []
        for shape, stride in zip(coalesced_shapes, coalesced_strides):
            can_merge_scalar_axis = (
                bool(merged_shapes)
                and _is_int(merged_shapes[-1])
                and _is_int(merged_strides[-1])
                and _is_int(shape)
                and _is_int(stride)
            )
            if can_merge_scalar_axis and merged_strides[-1] == stride * shape:
                merged_shapes[-1] *= shape
                merged_strides[-1] = stride
            else:
                merged_shapes.append(shape)
                merged_strides.append(stride)

        if len(merged_shapes) == 1:
            return _MeshLayout(merged_shapes[0], merged_strides[0])
        return _MeshLayout(tuple(merged_shapes), tuple(merged_strides))

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        """Compose *layout* on top of this axis (unflatten with scaled strides)."""
        if not _is_int(self.stride):
            raise NotImplementedError(
                "Currently, _unflatten only supports unflattening a mesh dim with scalar stride."
            )
        return _MeshLayout(layout.shape, _scale_inttuple(layout.stride, int(self.stride)))

    def nest(self) -> "_MeshLayout":
        """Wrap all dimensions into a single outer dimension."""
        if len(self) == 1:
            return self
        return _MeshLayout((self.shape,), (self.stride,))

    def splice(self, start: int, end: int, layout: "_MeshLayout") -> "_MeshLayout":
        """Replace dimensions [start, end) with the axes of *layout*."""
        sizes = list(_as_tuple(self.shape))
        strides = list(_as_tuple(self.stride))
        sizes[start:end] = list(_as_tuple(layout.shape))
        strides[start:end] = list(_as_tuple(layout.stride))
        if len(sizes) == 1:
            return _MeshLayout(sizes[0], strides[0])
        return _MeshLayout(tuple(sizes), tuple(strides))

    def collapse(self) -> _FlatLayout:
        """Flatten this layout into a single canonicalized flat layout."""
        return _FlatLayout(self.shape, self.stride)

    def remap_to_numpy(self, rank_map: Any) -> np.ndarray:
        """Materialize this layout as a dense numpy mesh over the provided rank map."""
        rank_map_np = np.asarray(rank_map).reshape(-1)
        base_offsets = self.all_ranks_from_zero()
        if len(base_offsets) == 0:
            raise ValueError("Cannot remap an empty layout.")

        groups: list[list[int]] = []
        used: set[int] = set()
        world_size = rank_map_np.shape[0]

        for anchor in range(world_size):
            if anchor in used:
                continue
            group = [anchor + offset for offset in base_offsets]
            if any(index >= world_size for index in group):
                continue
            if any(index in used for index in group):
                continue
            groups.append(group)
            used.update(group)

        if len(used) != world_size:
            raise ValueError(
                f"Layout {self} does not form a full partition over rank_map with world size {world_size}."
            )

        remapped = rank_map_np[np.array(groups, dtype=np.int64)]
        remapped = remapped.reshape((len(groups),) + self.top_level_sizes)
        return remapped
