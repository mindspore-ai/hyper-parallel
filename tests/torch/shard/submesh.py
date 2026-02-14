# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test torch sub_mesh"""
import numpy as np
import torch
from torch import nn
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.platform import get_platform
from tests.torch.utils import init_dist

platform = get_platform()

class SimpleAdd(nn.Module):
    """
        y = matmul(x, w)
        out = add(y, b)
    """
    def __init__(self, weight_t, bias_t):
        super().__init__()
        self.w = weight_t
        self.b = bias_t

    def forward(self, x):
        y = torch.matmul(x, self.w)
        out = y + self.b
        return out


x_input_np = np.arange(16).reshape(4, 4).astype(np.float32)
w_input_np = np.arange(16).reshape(4, 4).astype(np.float32)
b_input_np = np.array([0, 1, 2, 3]).astype(np.float32)

def test_full_mesh_shard_forward_1():
    """
    Feature: full mesh.
    Description: Data parallel.
    Expectation: Success.
    """
    init_dist()

    x_input = torch.from_numpy(x_input_np).npu()
    w_input = torch.from_numpy(w_input_np).npu()
    b_input = torch.from_numpy(b_input_np).npu()

    # 4 devices
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2), mesh_dim_names = ("dp", "tp"))
    x_p = (Shard(0), Replicate())
    w_p = (Replicate(), Shard(1))
    b_p = (Replicate(), Shard(0))

    dist_x = DTensor.distribute_tensor(x_input, mesh, x_p)
    dist_w = DTensor.distribute_tensor(w_input, mesh, w_p)
    dist_b = DTensor.distribute_tensor(b_input, mesh, b_p)
    rank = platform.get_rank()
    if rank in (0, 1):
        arr = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (0, 2):
        arr = np.array([
            [0, 1],
            [4, 5],
            [8, 9],
            [12, 13]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
        arr_b = np.array([0, 1]).astype(np.float32)
        shard_tensor_b = torch.from_numpy(arr_b).npu()
        assert dist_b.to_local().equal(shard_tensor_b), (
            f"Expected dist_x tensor {shard_tensor_b}, but got {dist_b.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [2, 3],
            [6, 7],
            [10, 11],
            [14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
        arr_b = np.array([2, 3]).astype(np.float32)
        shard_tensor_b = torch.from_numpy(arr_b).npu()
        assert dist_b.to_local().equal(shard_tensor_b), (
            f"Expected dist_x tensor {shard_tensor_b}, but got {dist_b.to_local()}"
        )

    model = SimpleAdd(w_input, b_input)
    out_network = model(x_input)

    dist_model = SimpleAdd(dist_w, dist_b)
    dist_out_network = dist_model(dist_x)
    gather_out_network = dist_out_network.full_tensor()

    assert out_network.equal(gather_out_network), (
        f"Expected standalone output {out_network} is equal to gathered output {gather_out_network}"
    )


def test_sub_mesh_column_parallel_forward():
    """
    Feature: Submesh.
    Description: Data parallel + (weight) column parallel.
    Expectation: Success.
    """
    init_dist()

    rank = platform.get_rank()
    x_input_np_2 = None
    if rank in (0, 1): # 相当于手动做x的dp轴Shard(0)
        x_input_np_2 = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
    if rank in (2, 3):
        x_input_np_2 = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)

    x_input = torch.from_numpy(x_input_np_2).npu()
    w_input = torch.from_numpy(w_input_np).npu()
    b_input = torch.from_numpy(b_input_np).npu()

    # 4 devices
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2), mesh_dim_names = ("dp", "tp"))
    tp_mesh = mesh["tp"]
    x_p = (Replicate(),)
    w_p = (Shard(1),)
    b_p = (Shard(0),)

    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    dist_w = DTensor.distribute_tensor(w_input, tp_mesh, w_p)
    dist_b = DTensor.distribute_tensor(b_input, tp_mesh, b_p)

    if rank in (0, 1):
        arr = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (0, 2):
        arr = np.array([
            [0, 1],
            [4, 5],
            [8, 9],
            [12, 13]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
        arr_b = np.array([0, 1]).astype(np.float32)
        shard_tensor_b = torch.from_numpy(arr_b).npu()
        assert dist_b.to_local().equal(shard_tensor_b), (
            f"Expected dist_x tensor {shard_tensor_b}, but got {dist_b.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [2, 3],
            [6, 7],
            [10, 11],
            [14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
        arr_b = np.array([2, 3]).astype(np.float32)
        shard_tensor_b = torch.from_numpy(arr_b).npu()
        assert dist_b.to_local().equal(shard_tensor_b), (
            f"Expected dist_x tensor {shard_tensor_b}, but got {dist_b.to_local()}"
        )

    model = SimpleAdd(w_input, b_input)
    out_network = model(x_input)

    dist_model = SimpleAdd(dist_w, dist_b)
    dist_out_network = dist_model(dist_x)
    gather_out_network = dist_out_network.full_tensor()
    assert out_network.equal(gather_out_network), (
        f"Expected standalone output {out_network} is equal to gathered output {gather_out_network}"
    )


def test_full_mesh_shard_forward_2():
    """
    Feature: full mesh.
    Description: Data parallel.
    Expectation: Success.
    """
    init_dist()

    x_input = torch.from_numpy(x_input_np).npu()
    w_input = torch.from_numpy(w_input_np).npu()
    b_input = torch.from_numpy(b_input_np).npu()

    # 4 devices
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2), mesh_dim_names = ("dp", "tp"))
    x_p = (Shard(0), Shard(1))
    w_p = (Replicate(), Shard(0))
    b_p = (Replicate(), Replicate())

    dist_x = DTensor.distribute_tensor(x_input, mesh, x_p)
    dist_w = DTensor.distribute_tensor(w_input, mesh, w_p)
    dist_b = DTensor.distribute_tensor(b_input, mesh, b_p)
    rank = platform.get_rank()
    if rank == 0:
        arr = np.array([
            [0, 1],
            [4, 5]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 1:
        arr = np.array([
            [2, 3],
            [6, 7]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 2:
        arr = np.array([
            [8, 9],
            [12, 13]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 3:
        arr = np.array([
            [10, 11],
            [14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (0, 2):
        arr = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_w tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_w tensor {shard_tensor}, but got {dist_w.to_local()}"
        )

    model = SimpleAdd(w_input, b_input)
    out_network = model(x_input)

    dist_model = SimpleAdd(dist_w, dist_b)
    dist_out_network = dist_model(dist_x)
    dist_out_network = dist_out_network.reduce_partial()
    gather_out_network = dist_out_network.full_tensor()

    assert out_network.equal(gather_out_network), (
        f"Expected standalone output {out_network} is equal to gathered output {gather_out_network}"
    )


def test_sub_mesh_row_parallel_forward():
    """
    Feature: Submesh.
    Description: Data parallel + (weight) row parallel.
    Expectation: Success.
    """
    init_dist()

    rank = platform.get_rank()
    x_input_np_2 = None
    if rank in (0, 1): # 相当于手动做x的dp轴Shard(0)
        x_input_np_2 = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
    if rank in (2, 3):
        x_input_np_2 = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)

    x_input = torch.from_numpy(x_input_np_2).npu()
    w_input = torch.from_numpy(w_input_np).npu()
    b_input = torch.from_numpy(b_input_np).npu()

    # 4 devices
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2), mesh_dim_names = ("dp", "tp"))
    tp_mesh = mesh["tp"]
    x_p = (Shard(1),)
    w_p = (Shard(0),)
    b_p = (Replicate(),)

    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    dist_w = DTensor.distribute_tensor(w_input, tp_mesh, w_p)
    dist_b = DTensor.distribute_tensor(b_input, tp_mesh, b_p)
    if rank == 0:
        arr = np.array([
            [0, 1],
            [4, 5]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 1:
        arr = np.array([
            [2, 3],
            [6, 7]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 2:
        arr = np.array([
            [8, 9],
            [12, 13]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 3:
        arr = np.array([
            [10, 11],
            [14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (0, 2):
        arr = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_w tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_w tensor {shard_tensor}, but got {dist_w.to_local()}"
        )

    model = SimpleAdd(w_input, b_input)
    out_network = model(x_input)

    dist_model = SimpleAdd(dist_w, dist_b)
    dist_out_network = dist_model(dist_x)
    dist_out_network = dist_out_network.reduce_partial()
    gather_out_network = dist_out_network.full_tensor()

    assert out_network.equal(gather_out_network), (
        f"Expected standalone output {out_network} is equal to gathered output {gather_out_network}"
    )


def test_sub_mesh_row_parallel_redistribute_forward():
    """
    Feature: Submesh.
    Description: Data parallel + (weight) row parallel.
    Expectation: Success.
    """
    init_dist()

    rank = platform.get_rank()
    x_input_np_2 = None
    if rank in (0, 1): # 相当于手动做x的dp轴Shard(0)
        x_input_np_2 = np.array([
            [0, 1, 0, 1],
            [0, 1, 0, 1]
        ]).astype(np.float32)
    if rank in (2, 3):
        x_input_np_2 = np.array([
            [2, 3, 2, 3],
            [2, 3, 2, 3]
        ]).astype(np.float32)
    w_input_np_2 = np.array([
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [2, 3, 2, 3],
        [2, 3, 2, 3]
    ]).astype(np.float32)
    b_input_np_2 = np.array([0, 1, 2, 3]).astype(np.float32)

    x_input = torch.from_numpy(x_input_np_2).npu()
    w_input = torch.from_numpy(w_input_np_2).npu()
    b_input = torch.from_numpy(b_input_np_2).npu()

    # 4 devices
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2), mesh_dim_names = ("dp", "tp"))
    tp_mesh = mesh["tp"]
    x_p = (Shard(1),)
    w_p = (Shard(0),)
    b_p = (Replicate(),)

    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    dist_w = DTensor.distribute_tensor(w_input, tp_mesh, w_p)
    dist_b = DTensor.distribute_tensor(b_input, tp_mesh, b_p)

    if rank == 0:
        arr = np.array([
            [0, 1],
            [0, 1]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 1:
        arr = np.array([
            [0, 1],
            [0, 1],
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 2:
        arr = np.array([
            [2, 3],
            [2, 3]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank == 3:
        arr = np.array([
            [2, 3],
            [2, 3]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_tensor), (
            f"Expected dist_x tensor {shard_tensor}, but got {dist_x.to_local()}"
        )
    if rank in (0, 2):
        arr = np.array([
            [0, 1, 0, 1],
            [0, 1, 0, 1]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_w tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [2, 3, 2, 3],
            [2, 3, 2, 3]
        ]).astype(np.float32)
        shard_tensor = torch.from_numpy(arr).npu()
        assert dist_w.to_local().equal(shard_tensor), (
            f"Expected dist_w tensor {shard_tensor}, but got {dist_w.to_local()}"
        )
    # standalone
    y_input = torch.matmul(x_input, w_input)
    out_network = y_input + b_input

    # distribute
    dist_y = torch.matmul(dist_x, dist_w)
    # # redistribute y：Replicate()
    dst_placements1 = (Replicate(),)
    redist_y1 = dist_y.redistribute(tp_mesh, dst_placements1)
    redist_out_network1 = redist_y1 + dist_b
    redist_out_network1 = redist_out_network1.reduce_partial()
    redist_gather_out_network1 = redist_out_network1.full_tensor()
    assert out_network.equal(redist_gather_out_network1), (
        f"Expected standalone output {out_network} is equal to gathered output {redist_gather_out_network1}"
    )

    # redistribute y：Shard(0)
    dst_placements2 = (Shard(0),) # 先rank0,1 reduce，再tp上切0轴
    redist_y2 = dist_y.redistribute(tp_mesh, dst_placements2)
    redist_out_network2 = redist_y2 + dist_b
    redist_out_network2 = redist_out_network2.reduce_partial()
    redist_gather_out_network2 = redist_out_network2.full_tensor()
    assert out_network.equal(redist_gather_out_network2), (
        f"Expected standalone output {out_network} is equal to gathered output {redist_gather_out_network2}"
    )


def test_sub_mesh_redistribute_1():
    """
    Feature: device_mesh 4 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input = torch.from_numpy(x_input_np).npu()
    rank = platform.get_rank()

    mesh = init_device_mesh(device_type = "npu", mesh_shape = (4, 2), mesh_dim_names = ("dp", "tp"))

    # shard tp
    tp_mesh = mesh["tp"]
    # (Shard(1),) redistribute to (Shard(0),)
    x_p = (Shard(1),)
    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    if rank in (0, 2, 4, 6):
        arr = np.array([
            [0, 1],
            [4, 5],
            [8, 9],
            [12, 13]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (1, 3, 5, 7):
        arr = np.array([
            [2, 3],
            [6, 7],
            [10, 11],
            [14, 15]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(0),)
    new_dist_x = dist_x.redistribute(tp_mesh, new_x_p)
    if rank in (0, 2, 4, 6):
        arr = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 3, 5, 7):
        arr = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )

    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    # (Shard(0),) redistribute to (Replicate(),)
    x_p = (Shard(0),)
    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    if rank in (0, 2, 4, 6):
        arr = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (1, 3, 5, 7):
        arr = np.array([
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Replicate(),)
    new_dist_x = dist_x.redistribute(tp_mesh, new_x_p)
    arr = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]).astype(np.float32)
    shard_x = torch.from_numpy(arr).npu()
    assert new_dist_x.to_local().equal(shard_x), (
        f"Expected dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
    )
    assert new_dist_x.to_local().equal(x_input), (
        f"Expected dist_x tensor {x_input}, but got {new_dist_x.to_local()}"
    )


def test_sub_mesh_redistribute_2():
    """
    Feature: device_mesh 4 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input = torch.from_numpy(x_input_np).npu()
    rank = platform.get_rank()

    mesh = init_device_mesh(device_type = "npu", mesh_shape = (4, 2), mesh_dim_names = ("dp", "tp"))

    # shard dp
    dp_mesh = mesh["dp"]
    # (Shard(0),) redistribute to (Shard(1),)
    x_p = (Shard(0),)
    dist_x = DTensor.distribute_tensor(x_input, dp_mesh, x_p)
    if rank in (0, 1):
        arr = np.array([
            [0, 1, 2, 3]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (4, 5):
        arr = np.array([
            [8, 9, 10, 11]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (6, 7):
        arr = np.array([
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(1),)
    new_dist_x = dist_x.redistribute(dp_mesh, new_x_p)
    if rank in (0, 1):
        arr = np.array([
            [0],
            [4],
            [8],
            [12]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [1],
            [5],
            [9],
            [13]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 5):
        arr = np.array([
            [2],
            [6],
            [10],
            [14]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (6, 7):
        arr = np.array([
            [3],
            [7],
            [11],
            [15]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
        f"Expected dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
    )

    # (Replicate(),) redistribute to (Shard(0),)
    x_p = (Replicate(),)
    dist_x = DTensor.distribute_tensor(x_input, dp_mesh, x_p)
    new_x_p = (Shard(0),)
    new_dist_x = dist_x.redistribute(dp_mesh, new_x_p)
    if rank in (0, 1):
        arr = np.array([
            [0, 1, 2, 3]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [4, 5, 6, 7]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 5):
        arr = np.array([
            [8, 9, 10, 11]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (6, 7):
        arr = np.array([
            [12, 13, 14, 15]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
        f"Expected dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
    )


def test_sub_mesh_redistribute_3():
    """
    Feature: device_mesh 1 * 8.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(64).reshape(8, 8).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()

    mesh = init_device_mesh(device_type = "npu", mesh_shape = (1, 8), mesh_dim_names = ("dp", "tp"))

    # shard tp
    tp_mesh = mesh["tp"]
    # (Shard(1),) redistribute to (Shard(0),)
    x_p = (Shard(1),)
    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    if rank == 0:
        arr = np.array([
            [0],
            [8],
            [16],
            [24],
            [32],
            [40],
            [48],
            [56]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(0),)
    new_dist_x = dist_x.redistribute(tp_mesh, new_x_p)
    if rank == 0:
        arr = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )


def test_sub_mesh_redistribute_4():
    """
    Feature: device_mesh 1 * 8.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(64).reshape(8, 8).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()

    mesh = init_device_mesh(device_type = "npu", mesh_shape = (1, 8), mesh_dim_names = ("dp", "tp"))

    # shard tp
    tp_mesh = mesh["dp"]
    # (Shard(1),) redistribute to (Shard(0),)
    x_p = (Shard(1),)
    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    assert dist_x.to_local().equal(x_input), (
            f"Expected dist_x tensor {x_input}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(0),)
    new_dist_x = dist_x.redistribute(tp_mesh, new_x_p)
    assert new_dist_x.to_local().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.to_local()}"
        )


def test_sub_mesh_redistribute_5():
    """
    Feature: device_mesh 2 * 2 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2, 2), mesh_dim_names = ("dp", "tp", "mp"))

    # shard dp
    dp_mesh = mesh["dp"]
    # (Shard(0),) redistribute to (Shard(1),) redistribute to (Shard(2),)
    x_p = (Shard(0),)
    dist_x = DTensor.distribute_tensor(x_input, dp_mesh, x_p)
    if rank in (0, 1, 2, 3):
        arr = np.array([
            [[0, 1],
            [2, 3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (4, 5, 6, 7):
        arr = np.array([
            [[4, 5],
            [6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(1),)
    new_dist_x = dist_x.redistribute(dp_mesh, new_x_p)
    if rank in (0, 1, 2, 3):
        arr = np.array([
            [[0, 1]],
            [[4, 5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 5, 6, 7):
        arr = np.array([
            [[2, 3]],
            [[6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    new_x_p = (Shard(2),)
    new_dist_x = dist_x.redistribute(dp_mesh, new_x_p)
    if rank in (0, 1, 2, 3):
        arr = np.array([
            [[0], [2]],
            [[4], [6]],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 5, 6, 7):
        arr = np.array([
            [[1], [3]],
            [[5], [7]],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )


def test_sub_mesh_redistribute_6():
    """
    Feature: device_mesh 2 * 2 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2, 2), mesh_dim_names = ("dp", "tp", "mp"))

    # shard tp
    tp_mesh = mesh["tp"]
    # (Shard(0),) redistribute to (Shard(1),) redistribute to (Shard(2),)
    x_p = (Shard(0),)
    dist_x = DTensor.distribute_tensor(x_input, tp_mesh, x_p)
    if rank in (0, 1, 4, 5):
        arr = np.array([
            [[0, 1],
            [2, 3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (2, 3, 6, 7):
        arr = np.array([
            [[4, 5],
            [6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(1),)
    new_dist_x = dist_x.redistribute(tp_mesh, new_x_p)
    if rank in (0, 1, 4, 5):
        arr = np.array([
            [[0, 1]],
            [[4, 5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 3, 6, 7):
        arr = np.array([
            [[2, 3]],
            [[6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    new_x_p = (Shard(2),)
    new_dist_x = dist_x.redistribute(tp_mesh, new_x_p)
    if rank in (0, 1, 4, 5):
        arr = np.array([
            [[0], [2]],
            [[4], [6]],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 3, 6, 7):
        arr = np.array([
            [[1], [3]],
            [[5], [7]],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )


def test_sub_mesh_redistribute_7():
    """
    Feature: device_mesh 2 * 2 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2, 2), mesh_dim_names = ("dp", "tp", "mp"))

    # shard mp
    mp_mesh = mesh["mp"]
    # (Shard(0),) redistribute to (Shard(1),) redistribute to (Shard(2),)
    x_p = (Shard(0),)
    dist_x = DTensor.distribute_tensor(x_input, mp_mesh, x_p)
    if rank in (0, 2, 4, 6):
        arr = np.array([
            [[0, 1],
            [2, 3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (1, 3, 5, 7):
        arr = np.array([
            [[4, 5],
            [6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(1),)
    new_dist_x = dist_x.redistribute(mp_mesh, new_x_p)
    if rank in (0, 2, 4, 6):
        arr = np.array([
            [[0, 1]],
            [[4, 5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 3, 5, 7):
        arr = np.array([
            [[2, 3]],
            [[6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    new_x_p = (Shard(2),)
    new_dist_x = dist_x.redistribute(mp_mesh, new_x_p)
    if rank in (0, 2, 4, 6):
        arr = np.array([
            [[0], [2]],
            [[4], [6]],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 3, 5, 7):
        arr = np.array([
            [[1], [3]],
            [[5], [7]],
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )


def test_sub_mesh_redistribute_8():
    """
    Feature: device_mesh 2 * 2 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2, 2), mesh_dim_names = ("dp", "tp", "mp"))

    # shard dp tp
    dtp_mesh = mesh["dp", "tp"]
    # (Shard(0), Shard(1),) redistribute to (Shard(0), Shard(2),) redistribute to (Shard(1), Shard(2),)
    x_p = (Shard(0), Shard(1),)
    dist_x = DTensor.distribute_tensor(x_input, dtp_mesh, x_p)
    if rank in (0, 1):
        arr = np.array([
            [[0, 1]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [[2, 3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (4, 5):
        arr = np.array([
            [[4, 5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (6, 7):
        arr = np.array([
            [[6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(0), Shard(2),)
    new_dist_x = dist_x.redistribute(dtp_mesh, new_x_p)
    if rank in (0, 1):
        arr = np.array([
            [[0], [2]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [[1], [3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 5):
        arr = np.array([
            [[4], [6]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (6, 7):
        arr = np.array([
            [[5], [7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    new_x_p = (Shard(1), Shard(2),)
    new_dist_x = dist_x.redistribute(dtp_mesh, new_x_p)
    if rank in (0, 1):
        arr = np.array([
            [[0]], [[4]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 3):
        arr = np.array([
            [[1]], [[5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 5):
        arr = np.array([
            [[2]], [[6]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (6, 7):
        arr = np.array([
            [[3]], [[7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )


def test_sub_mesh_redistribute_9():
    """
    Feature: device_mesh 2 * 2 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2, 2), mesh_dim_names = ("dp", "tp", "mp"))

    # shard dp mp
    dtp_mesh = mesh["dp", "mp"]
    # (Shard(0), Shard(1),) redistribute to (Shard(0), Shard(2),) redistribute to (Shard(1), Shard(2),)
    x_p = (Shard(0), Shard(1),)
    dist_x = DTensor.distribute_tensor(x_input, dtp_mesh, x_p)
    if rank in (0, 2):
        arr = np.array([
            [[0, 1]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    if rank in (1, 3):
        arr = np.array([
            [[2, 3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (4, 6):
        arr = np.array([
            [[4, 5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (5, 7):
        arr = np.array([
            [[6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(0), Shard(2),)
    new_dist_x = dist_x.redistribute(dtp_mesh, new_x_p)
    if rank in (0, 2):
        arr = np.array([
            [[0], [2]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [[1], [3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 6):
        arr = np.array([
            [[4], [6]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (5, 7):
        arr = np.array([
            [[5], [7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    new_x_p = (Shard(1), Shard(2),)
    new_dist_x = dist_x.redistribute(dtp_mesh, new_x_p)
    if rank in (0, 2):
        arr = np.array([
            [[0]], [[4]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 3):
        arr = np.array([
            [[1]], [[5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (4, 6):
        arr = np.array([
            [[2]], [[6]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (5, 7):
        arr = np.array([
            [[3]], [[7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )


def test_sub_mesh_redistribute_10():
    """
    Feature: device_mesh 2 * 2 * 2.
    Description: sub_mesh + redistribute.
    Expectation: Success.
    """
    init_dist()

    x_input_np_2 = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    x_input = torch.from_numpy(x_input_np_2).npu()
    rank = platform.get_rank()
    mesh = init_device_mesh(device_type = "npu", mesh_shape = (2, 2, 2), mesh_dim_names = ("dp", "tp", "mp"))

    # shard tp mp
    dtp_mesh = mesh["tp", "mp"]
    # (Shard(0), Shard(1),) redistribute to (Shard(0), Shard(2),) redistribute to (Shard(1), Shard(2),)
    x_p = (Shard(0), Shard(1),)
    dist_x = DTensor.distribute_tensor(x_input, dtp_mesh, x_p)
    if rank in (0, 4):
        arr = np.array([
            [[0, 1]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    if rank in (1, 5):
        arr = np.array([
            [[2, 3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (2, 6):
        arr = np.array([
            [[4, 5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )
    if rank in (3, 7):
        arr = np.array([
            [[6, 7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert dist_x.to_local().equal(shard_x), (
            f"Expected dist_x tensor {shard_x}, but got {dist_x.to_local()}"
        )

    new_x_p = (Shard(0), Shard(2),)
    new_dist_x = dist_x.redistribute(dtp_mesh, new_x_p)
    if rank in (0, 4):
        arr = np.array([
            [[0], [2]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 5):
        arr = np.array([
            [[1], [3]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 6):
        arr = np.array([
            [[4], [6]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (3, 7):
        arr = np.array([
            [[5], [7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )

    new_x_p = (Shard(1), Shard(2),)
    new_dist_x = dist_x.redistribute(dtp_mesh, new_x_p)
    if rank in (0, 4):
        arr = np.array([
            [[0]], [[4]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (1, 5):
        arr = np.array([
            [[1]], [[5]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (2, 6):
        arr = np.array([
            [[2]], [[6]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    if rank in (3, 7):
        arr = np.array([
            [[3]], [[7]]
        ]).astype(np.float32)
        shard_x = torch.from_numpy(arr).npu()
        assert new_dist_x.to_local().equal(shard_x), (
            f"Expected new_dist_x tensor {shard_x}, but got {new_dist_x.to_local()}"
        )
    assert new_dist_x.full_tensor().equal(x_input), (
            f"Expected new_dist_x tensor {x_input}, but got {new_dist_x.full_tensor()}"
        )
