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
"""MindSpore distributed repros for StridedShard DTensor.full_tensor."""

import math
import os

import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.communication.management as D

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "mindspore")

from hyper_parallel import init_device_mesh  # pylint: disable=wrong-import-position
from hyper_parallel.core.dtensor.dtensor import distribute_tensor  # pylint: disable=wrong-import-position
from hyper_parallel.core.dtensor.placement_types import (  # pylint: disable=wrong-import-position
    Replicate,
    Shard,
    StridedShard,
)


def _rank_to_coords(rank, mesh_shape):
    coords = [0] * len(mesh_shape)
    for axis in range(len(mesh_shape) - 1, -1, -1):
        coords[axis] = rank % mesh_shape[axis]
        rank //= mesh_shape[axis]
    return coords


def _split_id_for_tensor_map(mesh_shape, tensor_map_entry, rank):
    """Return the shard index for a rank on one tensor_map entry."""
    mapping = tensor_map_entry if isinstance(tensor_map_entry, tuple) else (tensor_map_entry,)
    coords = _rank_to_coords(rank, mesh_shape)
    split_id = 0
    coef = 1
    for mesh_dim in reversed(mapping):
        if mesh_dim == -1:
            continue
        mesh_axis = len(mesh_shape) - mesh_dim - 1
        split_id += coords[mesh_axis] * coef
        coef *= mesh_shape[mesh_axis]
    return split_id


def _split_count_for_tensor_map(mesh_shape, tensor_map_entry):
    mapping = tensor_map_entry if isinstance(tensor_map_entry, tuple) else (tensor_map_entry,)
    split_count = 1
    for mesh_dim in mapping:
        if mesh_dim == -1:
            continue
        split_count *= mesh_shape[len(mesh_shape) - mesh_dim - 1]
    return split_count


def _local_slice_for_tensor_map(global_np, mesh_shape, tensor_map, rank):
    """Return the expected local shard for a rank from the full NumPy tensor."""
    slices = []
    for tensor_dim, tensor_map_entry in enumerate(tensor_map):
        split_count = _split_count_for_tensor_map(mesh_shape, tensor_map_entry)
        split_id = _split_id_for_tensor_map(mesh_shape, tensor_map_entry, rank)
        dim_size = global_np.shape[tensor_dim]
        assert dim_size % split_count == 0
        chunk_size = dim_size // split_count
        start = split_id * chunk_size
        slices.append(slice(start, start + chunk_size))
    return global_np[tuple(slices)]


def _run_roundtrip_scenario(name, mesh_shape, mesh_dim_names, placements, global_shape, expected_tensor_map):
    """Run one StridedShard local-shard and full_tensor roundtrip scenario."""
    rank = D.get_rank()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )

    numel = math.prod(global_shape)
    global_np = np.arange(numel, dtype=np.float32).reshape(global_shape)
    global_tensor = Tensor(global_np, dtype=ms.float32)
    dtensor = distribute_tensor(global_tensor, mesh, placements)

    assert tuple(dtensor.placements) == tuple(placements), name
    assert dtensor.layout.tensor_map == expected_tensor_map, name

    expected_local = _local_slice_for_tensor_map(global_np, mesh_shape, dtensor.layout.tensor_map, rank)
    np.testing.assert_array_equal(
        dtensor.to_local().asnumpy(),
        expected_local,
        err_msg=f"{name}: rank={rank}",
    )

    restored = dtensor.full_tensor()
    np.testing.assert_array_equal(
        restored.asnumpy(),
        global_np,
        err_msg=f"{name}: rank={rank}",
    )


def test_strided_shard_full_tensor_roundtrip():
    """
    Feature: StridedShard DTensor.full_tensor.
    Description: Cover same-dim StridedShard layouts on 2-D, 3-D, and 4-D 8-card meshes.
    Expectation: Local shards follow strided split order and full_tensor matches the original tensor.
    """
    D.init()
    world_size = D.get_group_size()
    assert world_size == 8, f"this repro expects world_size=8, got {world_size}"

    scenarios = [
        (
            "2d_2x4_dim0",
            (2, 4),
            ("dp", "tp"),
            (StridedShard(0, split_factor=4), Shard(0)),
            (16, 4),
            ((0, 1), -1),
        ),
        (
            "2d_4x2_dim1",
            (4, 2),
            ("dp", "tp"),
            (StridedShard(1, split_factor=2), Shard(1)),
            (4, 16),
            (-1, (0, 1)),
        ),
        (
            "3d_dim0",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (StridedShard(0, split_factor=4), StridedShard(0, split_factor=2), Shard(0)),
            (16, 4),
            ((0, 1, 2), -1),
        ),
        (
            "3d_dim1",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (StridedShard(1, split_factor=4), StridedShard(1, split_factor=2), Shard(1)),
            (4, 16),
            (-1, (0, 1, 2)),
        ),
        (
            "3d_replicated_dp_dim2",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (Replicate(), StridedShard(2, split_factor=2), Shard(2)),
            (3, 4, 8),
            (-1, -1, (0, 1)),
        ),
        (
            "3d_replicated_cp_dim0",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (StridedShard(0, split_factor=2), Replicate(), Shard(0)),
            (16, 4),
            ((0, 2), -1),
        ),
        (
            "3d_dim0_and_dim1",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (StridedShard(0, split_factor=2), Shard(1), Shard(0)),
            (16, 8),
            ((0, 2), 1),
        ),
        (
            "3d_dim0_cp_tp_dim1_dp",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (Shard(1), StridedShard(0, split_factor=2), Shard(0)),
            (8, 8),
            ((0, 1), 2),
        ),
        (
            "3d_dim2_cp_tp_dim0_dp",
            (2, 2, 2),
            ("dp", "cp", "tp"),
            (Shard(0), StridedShard(2, split_factor=2), Shard(2)),
            (8, 3, 8),
            (2, -1, (0, 1)),
        ),
        (
            "4d_dim0_with_unit_outer_axis",
            (1, 2, 2, 2),
            ("pp", "dp", "cp", "tp"),
            (
                StridedShard(0, split_factor=8),
                StridedShard(0, split_factor=4),
                StridedShard(0, split_factor=2),
                Shard(0),
            ),
            (16, 4),
            ((0, 1, 2, 3), -1),
        ),
        (
            "4d_dim1_replicated_pp",
            (1, 2, 2, 2),
            ("pp", "dp", "cp", "tp"),
            (
                Replicate(),
                StridedShard(1, split_factor=4),
                StridedShard(1, split_factor=2),
                Shard(1),
            ),
            (5, 16),
            (-1, (0, 1, 2)),
        ),
        (
            "4d_dim0_and_dim2_unit_axis",
            (2, 1, 2, 2),
            ("pp", "dp", "cp", "tp"),
            (
                Shard(2),
                StridedShard(0, split_factor=4),
                StridedShard(0, split_factor=2),
                Shard(0),
            ),
            (8, 3, 6),
            ((0, 1, 2), -1, 3),
        ),
        (
            "4d_dim0_tp_cp_dim1_dp",
            (1, 2, 2, 2),
            ("pp", "dp", "cp", "tp"),
            (
                Replicate(),
                Shard(1),
                StridedShard(0, split_factor=2),
                Shard(0),
            ),
            (8, 6),
            ((0, 1), 2),
        ),
    ]
    for scenario in scenarios:
        _run_roundtrip_scenario(*scenario)
