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

from hyper_parallel.core.device_mesh import init_device_mesh, DeviceMesh
from hyper_parallel import get_platform

platform = get_platform()


def test_device_mesh_from_1d_group_valid():
    """
    Feature: DeviceMesh.from_group method with 1d group.
    Description: Test create 1d group via from_group method.
    Expectation: Run success, create 1D group via from_group method valid, and values are in expected.
    """
    device_mesh_init = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("dp",)
    )
    group = device_mesh_init.get_group()
    device_mesh = DeviceMesh.from_group(group, mesh_dim_names=("tp",))
    tp_mesh = device_mesh["tp"]
    assert tp_mesh.mesh_shape == (8,)
    assert tp_mesh.mesh_dim_names == ("tp",)
    assert tp_mesh.rank_list == (0, 1, 2, 3, 4, 5, 6, 7)


def test_device_mesh_from_2d_group_valid():
    """
    Feature: DeviceMesh.from_group method with 2d group.
    Description: Create 2d group via from_group method, using two group and 2d mesh.
    Expectation: Run success, values of created 2d group are in expected.
    """
    device_mesh_init = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp")
    )
    dp_group = device_mesh_init.get_group("dp")
    tp_group = device_mesh_init.get_group("tp")
    mesh_init = device_mesh_init.mesh
    device_mesh = DeviceMesh.from_group(
        [dp_group, tp_group],
        mesh=(mesh_init[0], mesh_init[1]),
        mesh_dim_names=("tp", "cp")
    )
    tp_mesh = device_mesh["tp"]
    cp_mesh = device_mesh["cp"]
    rank_id = platform.get_rank()
    world_size = platform.get_world_size()
    tp_rank_list = tuple(sorted((rank_id, rank_id + 4 if rank_id < world_size // 2 else rank_id - 4)))
    cp_rank_lists = (
        tuple(range(world_size // 2)),
        tuple(range(world_size // 2, world_size)),
    )
    cp_rank_list = cp_rank_lists[0] if rank_id in cp_rank_lists[0] else cp_rank_lists[1]
    assert tp_mesh.mesh_shape == (2,)
    assert tp_mesh.mesh_dim_names == ("tp",)
    assert tp_mesh.root_mesh == device_mesh
    assert tp_mesh.rank_list == tp_rank_list
    assert cp_mesh.mesh_shape == (4,)
    assert cp_mesh.mesh_dim_names == ("cp",)
    assert cp_mesh.root_mesh == device_mesh
    assert cp_mesh.rank_list == cp_rank_list


def test_device_mesh_from_3d_group_valid():
    """
    Feature: DeviceMesh.from_group method with 3d group.
    Description: Create 2d group via from_group method, using two group from 3d group and 2d mesh split from 3d mesh.
    Expectation: Run success, values of created 2d group are in expected.
    """
    device_mesh_init = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "cp")
    )
    dp_group = device_mesh_init["dp"].get_group()
    tp_group = device_mesh_init["tp"].get_group()
    mesh_init = device_mesh_init.mesh
    device_mesh = DeviceMesh.from_group(
        [dp_group, tp_group],
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
