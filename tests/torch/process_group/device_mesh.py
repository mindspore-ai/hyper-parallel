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
"""test_process_group.py"""

import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, DeviceMesh
from hyper_parallel import init_process_group, get_process_group_ranks


@pytest.fixture(scope="module", name="level0_rank")
def _level0_group() -> Iterator[int]:
    """Bind this rank's NPU and exercise the public initializer on a fresh WORLD."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    assert not dist.is_initialized(), "Expected a fresh default process group"
    init_process_group()
    assert dist.is_initialized(), "The public initializer did not create a default process group"
    try:
        yield dist.get_rank()
    finally:
        dist.destroy_process_group()


def _check_group_sum(group: dist.ProcessGroup, rank: int, ranks: tuple[int, ...]) -> None:
    """Require the reconstructed group to communicate on its actual NPU members."""
    value = torch.tensor(rank + 1, dtype=torch.int32, device="npu")
    dist.all_reduce(value, group=group)
    assert value.item() == sum(ranks) + len(ranks), f"Incorrect HCCL reduction on ranks {ranks}"


def test_device_mesh_from_1d_group_valid(level0_rank: int) -> None:
    """Initialize the Ascend world mesh and reconstruct it from its group."""
    rank = level0_rank
    world_size = dist.get_world_size()
    ranks = tuple(range(world_size))
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("dp",))
    assert mesh.ndim == 1
    assert mesh.mesh_shape == (world_size,)
    assert mesh.rank_list == ranks
    assert rank in mesh.rank_list
    assert mesh.get_coordinate() == (rank,)

    device_mesh = DeviceMesh.from_group(mesh.get_group(), "npu", mesh_dim_names=("tp",))
    tp_mesh = device_mesh["tp"]
    assert tp_mesh.mesh_shape == (world_size,)
    assert tp_mesh.mesh_dim_names == ("tp",)
    assert tp_mesh.rank_list == ranks
    _check_group_sum(tp_mesh.get_group(), rank, ranks)


def test_device_mesh_from_1d_subgroup(level0_rank: int) -> None:
    """Reconstruct the original two-rank mesh from a real HCCL subgroup."""
    rank = level0_rank
    first_rank = rank - rank % 2
    ranks = (first_rank, first_rank + 1)
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("replica", "dp"))["dp"]
    group = mesh.get_group()
    reconstructed = DeviceMesh.from_group(group, "npu", mesh_dim_names=("tp",))["tp"]
    assert reconstructed.rank_list == ranks, f"Unexpected subgroup ranks {reconstructed.rank_list}"
    assert reconstructed.mesh_shape == (2,), f"Unexpected subgroup shape {reconstructed.mesh_shape}"
    assert reconstructed.mesh_dim_names == ("tp",), f"Unexpected axis names {reconstructed.mesh_dim_names}"
    assert reconstructed.get_coordinate() == (rank % 2,), f"Unexpected coordinate on rank {rank}"
    assert get_process_group_ranks(reconstructed.get_group()) == list(ranks), f"Subgroup differs from {ranks}"
    _check_group_sum(reconstructed.get_group(), rank, ranks)


def test_device_mesh_from_2d_group_valid():
    """
    Feature: DeviceMesh.from_group method with 2d group.
    Description: Create 2d group via from_group method, using two group and 2d mesh.
    Expectation: Run success, values of created 2d group are in expected.
    """
    init_process_group()
    device_mesh_init = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "tp")
    )
    dp_group = device_mesh_init.get_group("dp")
    tp_group = device_mesh_init.get_group("tp")
    mesh_init = device_mesh_init.mesh
    device_mesh = DeviceMesh.from_group(
        [dp_group, tp_group],
        "npu",
        mesh=(mesh_init[0], mesh_init[1]),
        mesh_dim_names=("tp", "cp")
    )
    tp_mesh = device_mesh["tp"]
    cp_mesh = device_mesh["cp"]
    rank_id = dist.get_rank()
    world_size = dist.get_world_size()
    tp_rank_list = tuple(sorted((rank_id, rank_id + 2 if rank_id < world_size // 2 else rank_id - 2)))
    cp_rank_lists = (
        tuple(range(world_size // 2)),
        tuple(range(world_size // 2, world_size)),
    )
    cp_rank_list = cp_rank_lists[0] if rank_id in cp_rank_lists[0] else cp_rank_lists[1]
    assert tp_mesh.mesh_shape == (2,)
    assert tp_mesh.mesh_dim_names == ("tp",)
    assert tp_mesh.root_mesh == device_mesh
    assert tp_mesh.rank_list == tp_rank_list
    assert cp_mesh.mesh_shape == (2,)
    assert cp_mesh.mesh_dim_names == ("cp",)
    assert cp_mesh.root_mesh == device_mesh
    assert cp_mesh.rank_list == cp_rank_list


def test_device_mesh_from_2d_group_use_list_valid(level0_rank: int) -> None:
    """Recover both process group memberships through a list of axis names."""
    device_mesh_init = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "tp")
    )
    dp_group = device_mesh_init.get_group("dp")
    tp_group = device_mesh_init.get_group("tp")
    mesh_init = device_mesh_init.mesh
    device_mesh = DeviceMesh.from_group(
        [dp_group, tp_group],
        "npu",
        mesh=(mesh_init[0], mesh_init[1]),
        mesh_dim_names=["mp", "cp"]
    )
    mp_group = device_mesh.get_group("mp")
    cp_group = device_mesh.get_group("cp")
    mp_rank_list = get_process_group_ranks(mp_group)
    cp_rank_list = get_process_group_ranks(cp_group)
    dp_rank_list = get_process_group_ranks(dp_group)
    tp_rank_list = get_process_group_ranks(tp_group)
    assert mp_rank_list == dp_rank_list
    assert cp_rank_list == tp_rank_list
    _check_group_sum(mp_group, level0_rank, tuple(dp_rank_list))
    _check_group_sum(cp_group, level0_rank, tuple(tp_rank_list))


def test_device_mesh_from_3d_group_valid():
    """
    Feature: DeviceMesh.from_group method with 3d group.
    Description: Create 2d group via from_group method, using two group from 3d group and 2d mesh split from 3d mesh.
    Expectation: Run success, values of created 2d group are in expected.
    """
    init_process_group()
    device_mesh_init = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 1),
        mesh_dim_names=("dp", "tp", "cp")
    )
    dp_group = device_mesh_init["dp"].get_group()
    tp_group = device_mesh_init["tp"].get_group()
    mesh_init = device_mesh_init.mesh
    device_mesh = DeviceMesh.from_group(
        [dp_group, tp_group],
        "npu",
        mesh=mesh_init[:, :, device_mesh_init.get_local_rank("cp")],
        mesh_dim_names=("dp", "tp")
    )
    dp_mesh = device_mesh["dp"]
    tp_mesh = device_mesh["tp"]
    dp_init_mesh = device_mesh_init["dp"]
    tp_init_mesh = device_mesh_init["tp"]
    assert dp_mesh.mesh_shape == dp_init_mesh.mesh_shape
    assert dp_mesh.mesh_dim_names == dp_init_mesh.mesh_dim_names
    assert dp_mesh.rank_list == dp_init_mesh.rank_list
    assert tp_mesh.mesh_shape == tp_init_mesh.mesh_shape
    assert tp_mesh.mesh_dim_names == tp_init_mesh.mesh_dim_names
    assert tp_mesh.rank_list == tp_init_mesh.rank_list
