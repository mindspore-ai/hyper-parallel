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
"""
Utility functions for distributed tensor operations.

This module provides helper functions for computing local shapes, global offsets,
and other layout-related calculations in distributed settings.
"""
from typing import Sequence

from hyper_parallel.core.dtensor.layout import (
    Layout,
    infer_balanced_chunk_range,
    infer_ceil_chunk_range,
)


def compute_local_shape_and_global_offset(global_shape, device_mesh, placement):
    """
        Compute local shard shape and its global offset.

    Args:
        global_shape: Shape of the global tensor.
        device_mesh: Device mesh for distributed execution.
        placement: Sharding placements for each mesh dimension. Supports
            Placement objects or alias strings.

    Returns:
        The local shape owned by the current rank.
    """
    from hyper_parallel.core.dtensor.dtensor import _is_alias_placements  # pylint: disable=C0415
    total_layout = Layout.from_device_mesh(device_mesh)
    if _is_alias_placements(placement):
        layout = total_layout(*placement)
    else:
        layout = total_layout(placement)
        layout.placement_to_tensor_map(len(global_shape))
    local_shape = list(global_shape)
    for tensor_dim, mapped_axes in enumerate(layout.alias_tensor_map):
        if isinstance(mapped_axes, str):
            mapped_axes = (mapped_axes,)
        for mapped_axis in mapped_axes:
            if mapped_axis == "None":
                continue
            shard_count = layout.mesh.get_device_num_along_axis(mapped_axis)
            if local_shape[tensor_dim] % shard_count == 0:
                local_shape[tensor_dim] //= shard_count
                continue
            chunk_start, chunk_end = infer_balanced_chunk_range(
                local_shape[tensor_dim],
                shard_count,
                layout.mesh.get_local_rank(mapped_axis),
            )
            local_shape[tensor_dim] = chunk_end - chunk_start
    return local_shape


def compute_local_shape_and_global_offset_by_ceil_chunk(
    global_shape: Sequence[int],
    shard_dim: int,
    shard_count: int,
    shard_rank: int,
) -> tuple[list[int], list[int]]:
    """Return one FSDP local shape and offset using ceil-chunk geometry.

    Unlike balanced Shard geometry, ceil-chunk keeps a fixed maximum chunk
    size and represents ranks beyond the last chunk with an empty shard.

    Args:
        global_shape: Shape before applying the FSDP shard.
        shard_dim: Tensor dimension partitioned by FSDP.
        shard_count: Number of ranks in the FSDP shard mesh.
        shard_rank: Current rank within the FSDP shard mesh.

    Returns:
        The local shape and its global offset relative to ``global_shape``.
    """
    local_shape = list(global_shape)
    global_offset = [0] * len(local_shape)
    chunk_start, chunk_end = infer_ceil_chunk_range(
        local_shape[shard_dim],
        shard_count,
        shard_rank,
    )
    local_shape[shard_dim] = chunk_end - chunk_start
    global_offset[shard_dim] = chunk_start
    return local_shape, global_offset
