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
"""Flat-storage geometry and distribution helpers for RaggedShard."""
from math import prod
from typing import NamedTuple, Optional, Sequence

import numpy as np

from hyper_parallel.core.dtensor._collective_utils import mesh_scatter_ragged
from hyper_parallel.core.dtensor.layout import Layout, RaggedShardInfo
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor


def _layout_has_ragged_shard(layout: object) -> bool:
    """Return whether a concrete Layout carries RaggedShard metadata."""
    return isinstance(getattr(layout, "ragged_shard", None), RaggedShardInfo)


class _RaggedSlice(NamedTuple):
    """Flat interval owned by one rank in a RaggedShard layout."""

    flat_start: int
    flat_end: int

    @property
    def local_numel(self) -> int:
        """Return the number of flat elements in the interval."""
        return self.flat_end - self.flat_start


def _normalize_global_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Normalize a concrete global tensor shape."""
    normalized = tuple(shape)
    if any(
        not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 0
        for size in normalized
    ):
        raise ValueError(
            f"DTensor global shape must contain non-negative integers, got {normalized!r}"
        )
    return tuple(int(size) for size in normalized)


def _compute_ragged_slice(
    global_shape: Sequence[int],
    layout: Layout,
    local_rank: Optional[int] = None,
) -> _RaggedSlice:
    """Compute one rank's flat RaggedShard interval.

    Phase one supports exactly one RaggedShard and Replicate placements on all
    other mesh dimensions.
    """
    info = layout.ragged_shard
    if info is None:
        raise ValueError("RaggedShard slice computation requires a ragged layout")

    for mesh_dim, placement in enumerate(layout.placements):
        if mesh_dim == info.mesh_dim:
            continue
        if not placement.is_replicate():
            raise NotImplementedError(
                "RaggedShard phase one only supports Replicate on other mesh dimensions, "
                f"got mesh_dim={mesh_dim}, placement={placement!r}"
            )

    shape = _normalize_global_shape(global_shape)
    ragged = info.placement
    prefix_ndim = len(ragged.dims)
    if prefix_ndim > len(shape):
        raise ValueError(
            f"RaggedShard dims {ragged.dims!r} exceed global shape rank {len(shape)}"
        )

    mesh_dim_size = layout.mesh.size(info.mesh_dim)
    if len(ragged.local_units) != mesh_dim_size:
        raise ValueError(
            "RaggedShard len(local_units) must equal mesh.size(mesh_dim), "
            f"got len(local_units)={len(ragged.local_units)}, mesh_dim_size={mesh_dim_size}"
        )

    prefix_cells = prod(shape[:prefix_ndim])
    total_units = sum(ragged.local_units)
    if prefix_cells % total_units != 0:
        raise ValueError(
            "RaggedShard prefix cell count must be divisible by sum(local_units), "
            f"got prefix_cells={prefix_cells}, local_units={ragged.local_units!r}"
        )

    if local_rank is None:
        local_rank = layout.mesh.get_local_rank(info.mesh_dim)
    if local_rank < 0 or local_rank >= mesh_dim_size:
        raise ValueError(
            f"RaggedShard local rank must be in [0, {mesh_dim_size}), got {local_rank}"
        )
    cells_per_unit = prefix_cells // total_units
    prefix_start = sum(ragged.local_units[:local_rank]) * cells_per_unit
    local_prefix_cells = ragged.local_units[local_rank] * cells_per_unit
    suffix_numel = prod(shape[prefix_ndim:])
    flat_start = prefix_start * suffix_numel
    flat_end = flat_start + local_prefix_cells * suffix_numel
    return _RaggedSlice(flat_start, flat_end)


def _compute_ragged_splits(
    global_shape: Sequence[int],
    layout: Layout,
) -> tuple[int, ...]:
    """Return flat element counts contributed by all ranks on the ragged axis."""
    info = layout.ragged_shard
    if info is None:
        raise ValueError("RaggedShard split computation requires a ragged layout")
    return tuple(
        _compute_ragged_slice(global_shape, layout, local_rank=rank).local_numel
        for rank in range(layout.mesh.size(info.mesh_dim))
    )


def _interval_overlap_size(first: _RaggedSlice, second: _RaggedSlice) -> int:
    """Return the size of two half-open flat intervals' intersection."""
    return max(0, min(first.flat_end, second.flat_end) - max(first.flat_start, second.flat_start))


def _compute_ragged_all_to_all_splits(
    global_shape: Sequence[int],
    from_layout: Layout,
    to_layout: Layout,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute variable all-to-all splits for a local-units-only change."""
    source_info = from_layout.ragged_shard
    target_info = to_layout.ragged_shard
    if source_info is None or target_info is None:
        raise ValueError("RaggedShard all-to-all split computation requires ragged source and target layouts")
    if source_info.mesh_dim != target_info.mesh_dim:
        raise ValueError(
            "RaggedShard all-to-all requires the same ragged mesh dimension, "
            f"got source={source_info.mesh_dim}, target={target_info.mesh_dim}"
        )
    if source_info.placement.dims != target_info.placement.dims:
        raise ValueError(
            "RaggedShard all-to-all only supports local_units changes; dims must stay unchanged, "
            f"got source={source_info.placement.dims!r}, target={target_info.placement.dims!r}"
        )

    mesh_dim = source_info.mesh_dim
    source_rank = from_layout.mesh.get_local_rank(mesh_dim)
    target_rank = to_layout.mesh.get_local_rank(mesh_dim)
    source_interval = _compute_ragged_slice(global_shape, from_layout, source_rank)
    target_interval = _compute_ragged_slice(global_shape, to_layout, target_rank)
    input_splits = tuple(
        _interval_overlap_size(
            source_interval,
            _compute_ragged_slice(global_shape, to_layout, rank),
        )
        for rank in range(to_layout.mesh.size(mesh_dim))
    )
    output_splits = tuple(
        _interval_overlap_size(
            _compute_ragged_slice(global_shape, from_layout, rank),
            target_interval,
        )
        for rank in range(from_layout.mesh.size(mesh_dim))
    )
    return input_splits, output_splits


def _slice_ragged_tensor(tensor: Tensor, layout: Layout) -> Tensor:
    """Return an independent flat local shard from a complete global tensor."""
    if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
        raise ValueError("distribute_tensor with RaggedShard requires a contiguous tensor")
    ragged_slice = _compute_ragged_slice(tuple(tensor.shape), layout)
    flat_tensor = tensor.reshape((-1,))
    return flat_tensor[ragged_slice.flat_start:ragged_slice.flat_end].clone()


def _scatter_ragged_tensor(
    tensor: Tensor,
    layout: Layout,
    src_data_rank: int,
) -> Tensor:
    """Distribute variable flat shards from one group-relative source rank."""
    if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
        raise ValueError("distribute_tensor with RaggedShard requires a contiguous tensor")
    info = layout.ragged_shard
    ragged_slice = _compute_ragged_slice(tuple(tensor.shape), layout)
    flat_tensor = tensor.reshape((-1,))
    output = platform.empty(
        (ragged_slice.local_numel,),
        dtype=tensor.dtype,
        device=getattr(tensor, "device", None),
    )

    scatter_list = None
    if layout.mesh.get_local_rank(info.mesh_dim) == src_data_rank:
        scatter_list = []
        for destination_rank in range(len(info.placement.local_units)):
            destination_slice = _compute_ragged_slice(
                tuple(tensor.shape),
                layout,
                local_rank=destination_rank,
            )
            scatter_list.append(
                flat_tensor[destination_slice.flat_start:destination_slice.flat_end]
            )

    return mesh_scatter_ragged(
        output,
        scatter_list,
        layout.mesh,
        info.mesh_dim,
        group_src=src_data_rank,
    )
