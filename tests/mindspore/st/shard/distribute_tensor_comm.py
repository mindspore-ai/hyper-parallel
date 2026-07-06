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
"""MindSpore ST for distribute_tensor scatter/broadcast (src_data_rank parity)."""

import numpy as np
from mindspore import Tensor
import mindspore.communication.management as D

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def _rank_coord(rank: int, mesh_shape: tuple[int, ...]) -> tuple[int, ...]:
    coord = []
    rem = rank
    for size in reversed(mesh_shape):
        coord.append(rem % size)
        rem //= size
    return tuple(reversed(coord))


def _expected_local(
    global_np: np.ndarray,
    mesh_shape: tuple[int, ...],
    placements: tuple,
    rank: int,
) -> np.ndarray:
    """CPU reference for sequential shard/broadcast semantics."""
    local = global_np
    coord = _rank_coord(rank, mesh_shape)
    for mesh_dim, placement in enumerate(placements):
        if placement.is_replicate():
            continue
        if placement.is_shard():
            dim = placement.dim
            if dim < 0:
                dim += local.ndim
            num_chunks = mesh_shape[mesh_dim]
            piece = local.shape[dim] // num_chunks
            start = piece * coord[mesh_dim]
            slices = [slice(None)] * local.ndim
            slices[dim] = slice(start, start + piece)
            local = local[tuple(slices)]
    return local


def _assert_distribute(
    case_name: str,
    tensor: Tensor,
    mesh,
    mesh_shape: tuple[int, ...],
    placements: tuple,
    src_data_rank: int,
    global_np: np.ndarray,
) -> None:
    """Run distribute_tensor and compare local/full results against CPU reference."""
    rank = D.get_rank()
    dt = distribute_tensor(tensor, mesh, placements, src_data_rank=src_data_rank)
    expected_local = _expected_local(global_np, mesh_shape, placements, rank)
    np.testing.assert_allclose(
        dt.to_local().asnumpy(),
        expected_local,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"{case_name}: rank={rank} local shard mismatch",
    )
    np.testing.assert_allclose(
        dt.full_tensor().asnumpy(),
        global_np,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"{case_name}: rank={rank} full_tensor mismatch",
    )


def _reference_tensor(shape: tuple[int, ...], seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def test_rank0_only_shard0():
    """
    Feature: distribute_tensor rank0_only with 1D Shard(0).
    Description: Only rank 0 holds the global tensor; scatter along mesh dim 0.
    Expectation: Local shards and full_tensor match the reference global tensor.
    """
    D.init()
    assert D.get_group_size() == 2, "expects 2-card run"
    rank = D.get_rank()
    mesh_shape = (2,)
    mesh = init_device_mesh(device_type="npu", mesh_shape=mesh_shape, mesh_dim_names=("dp",))
    shape = (64, 128)
    global_np = _reference_tensor(shape)
    tensor = Tensor(global_np if rank == 0 else np.zeros(shape, dtype=np.float32))
    _assert_distribute(
        "rank0_only_shard0",
        tensor,
        mesh,
        mesh_shape,
        (Shard(0),),
        src_data_rank=0,
        global_np=global_np,
    )


def test_src_only_nonzero_src():
    """
    Feature: distribute_tensor src_only with src_data_rank=1.
    Description: Only rank 1 holds the global tensor; scatter from group rank 1.
    Expectation: Local shards and full_tensor match the reference global tensor.
    """
    D.init()
    assert D.get_group_size() == 2, "expects 2-card run"
    rank = D.get_rank()
    mesh_shape = (2,)
    mesh = init_device_mesh(device_type="npu", mesh_shape=mesh_shape, mesh_dim_names=("dp",))
    shape = (64, 128)
    global_np = _reference_tensor(shape, seed=43)
    tensor = Tensor(global_np if rank == 1 else np.zeros(shape, dtype=np.float32))
    _assert_distribute(
        "src_only_nonzero_src",
        tensor,
        mesh,
        mesh_shape,
        (Shard(0),),
        src_data_rank=1,
        global_np=global_np,
    )


def test_rank0_only_replicate():
    """
    Feature: distribute_tensor rank0_only with Replicate placement.
    Description: Broadcast the logical global tensor to every rank on a 1D mesh.
    Expectation: Each rank holds a full replica; full_tensor matches the reference.
    """
    D.init()
    assert D.get_group_size() == 2, "expects 2-card run"
    rank = D.get_rank()
    mesh_shape = (2,)
    mesh = init_device_mesh(device_type="npu", mesh_shape=mesh_shape, mesh_dim_names=("dp",))
    shape = (32, 32)
    global_np = _reference_tensor(shape, seed=44)
    tensor = Tensor(global_np if rank == 0 else np.zeros(shape, dtype=np.float32))
    _assert_distribute(
        "rank0_only_replicate",
        tensor,
        mesh,
        mesh_shape,
        (Replicate(),),
        src_data_rank=0,
        global_np=global_np,
    )


def test_rank0_only_2d_shard_replicate():
    """
    Feature: distribute_tensor rank0_only on a 2D mesh with Shard + Replicate.
    Description: 2-card mesh (1, 2); shard tensor dim 1 along tp, replicate on dp.
    Expectation: Local shards and full_tensor match the reference global tensor.
    """
    D.init()
    assert D.get_group_size() == 2, "expects 2-card run"
    rank = D.get_rank()
    mesh_shape = (1, 2)
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=("dp", "tp"),
    )
    shape = (32, 64)
    global_np = _reference_tensor(shape, seed=45)
    tensor = Tensor(global_np if rank == 0 else np.zeros(shape, dtype=np.float32))
    _assert_distribute(
        "rank0_only_2d_shard_replicate",
        tensor,
        mesh,
        mesh_shape,
        (Replicate(), Shard(1)),
        src_data_rank=0,
        global_np=global_np,
    )
