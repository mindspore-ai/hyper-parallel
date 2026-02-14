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
"""test one hot ext shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore import mint
from hyper_parallel import init_device_mesh, shard_module, DTensor
from hyper_parallel.core.dtensor import distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module():
    ms.set_device("Ascend")
    D.init()


class OneHotExtNet(nn.Cell):
    """OneHot network using mint.nn.functional.one_hot"""

    def __init__(self, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.relu = nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(input_plan={"input": relu_strategy})
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, tensor, num_classes):
        out = mint.nn.functional.one_hot(tensor, num_classes)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")


def test_one_hot_ext_data_parallel_1d_int64_1():
    """
    Feature: mint.nn.functional.one_hot in python shard.
    Description: Test with 1D tensor, data parallel on tensor dim 0 (indices int64).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    a, num_classes = 64, 32

    tensor = Tensor(np.random.randint(0, num_classes, size=(a,)).astype(np.int64), ms.int64)

    standalone_net = OneHotExtNet()
    standalone_output = standalone_net(tensor, num_classes)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    tensor_placements = (Shard(0), Replicate(), Replicate())
    onehot_out_placements = (Shard(0), Replicate(), Replicate())

    tensor_local = distribute_tensor(tensor, mesh, tensor_placements)

    parallel_net = OneHotExtNet(device_mesh=mesh, relu_strategy=(onehot_out_placements,))
    parallel_output = parallel_net(tensor_local, num_classes)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_one_hot_ext_data_parallel_2d_int64_2():
    """
    Feature: mint.nn.functional.one_hot in python shard.
    Description: Test with 2D tensor, data parallel on tensor dim 0 (indices int64).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    a, b, num_classes = 16, 32, 64

    tensor = Tensor(np.random.randint(0, num_classes, size=(a, b)).astype(np.int64), ms.int64)

    standalone_net = OneHotExtNet()
    standalone_output = standalone_net(tensor, num_classes)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    tensor_placements = (Shard(0), Replicate(), Replicate())
    onehot_out_placements = (Shard(0), Replicate(), Replicate())

    tensor_local = distribute_tensor(tensor, mesh, tensor_placements)

    parallel_net = OneHotExtNet(device_mesh=mesh, relu_strategy=(onehot_out_placements,))
    parallel_output = parallel_net(tensor_local, num_classes)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_one_hot_ext_replicate_all_int64_3():
    """
    Feature: mint.nn.functional.one_hot in python shard.
    Description: Test with full replication (indices int64).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    a, num_classes = 64, 32

    tensor = Tensor(np.random.randint(0, num_classes, size=(a,)).astype(np.int64), ms.int64)

    standalone_net = OneHotExtNet()
    standalone_output = standalone_net(tensor, num_classes)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    tensor_placements = (Replicate(), Replicate(), Replicate())
    onehot_out_placements = (Replicate(), Replicate(), Replicate())

    tensor_local = distribute_tensor(tensor, mesh, tensor_placements)

    parallel_net = OneHotExtNet(device_mesh=mesh, relu_strategy=(onehot_out_placements,))
    parallel_output = parallel_net(tensor_local, num_classes)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_one_hot_ext_3d_data_parallel_int64_4():
    """
    Feature: mint.nn.functional.one_hot in python shard.
    Description: Test with 3D tensor, data parallel on first dim only (indices int64).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    a, b, c, num_classes = 8, 16, 12, 32

    tensor = Tensor(np.random.randint(0, num_classes, size=(a, b, c)).astype(np.int64), ms.int64)

    standalone_net = OneHotExtNet()
    standalone_output = standalone_net(tensor, num_classes)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    tensor_placements = (Shard(0), Replicate(), Replicate())
    onehot_out_placements = (Shard(0), Replicate(), Replicate())

    tensor_local = distribute_tensor(tensor, mesh, tensor_placements)

    parallel_net = OneHotExtNet(device_mesh=mesh, relu_strategy=(onehot_out_placements,))
    parallel_output = parallel_net(tensor_local, num_classes)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_one_hot_ext_auto_depth_skewed_distribution_int64_5():
    """
    Feature: mint.nn.functional.one_hot with num_classes=-1 in python shard.
    Description: Test auto depth inference with skewed data distribution.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    rank = D.get_rank()

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    tensor_placements = (Shard(0), Replicate(), Replicate())
    onehot_out_placements = (Shard(0), Replicate(), Replicate())

    a = 64
    expected_depth = 31

    rng0 = np.random.RandomState(123)
    rng1 = np.random.RandomState(456)

    shard0 = rng0.randint(0, 8, size=(a,)).astype(np.int64)
    shard1 = rng1.randint(0, 8, size=(a,)).astype(np.int64)

    shard0[0] = 10
    shard1[0] = 30

    global_tensor = Tensor(np.concatenate([shard0, shard1], axis=0), ms.int64)

    standalone_net = OneHotExtNet()
    standalone_out = standalone_net(global_tensor, -1)
    assert standalone_out.shape[-1] == expected_depth

    dp_id = rank // 4
    local_np = shard0.copy() if dp_id == 0 else shard1.copy()
    tensor_local = DTensor.from_local(Tensor(local_np, ms.int64), mesh, tensor_placements)

    parallel_net = OneHotExtNet(device_mesh=mesh, relu_strategy=(onehot_out_placements,))
    parallel_out = parallel_net(tensor_local, -1)

    assert hasattr(parallel_out, "local_shape")
    assert parallel_out.local_shape[-1] == expected_depth

    parallel_out_global = parallel_out.full_tensor()
    if rank == 0:
        assert np.allclose(standalone_out.asnumpy(), parallel_out_global.asnumpy(), 1e-3, 1e-3)


def test_one_hot_ext_auto_depth_all_same_local_max_int64_6():
    """
    Feature: mint.nn.functional.one_hot with num_classes=-1 in python shard.
    Description: Test auto depth inference when all dp shards have the same local max.
    Expectation: Run success.
    """
    ms.set_seed(2)
    np.random.seed(2)
    rank = D.get_rank()

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)
    tensor_placements = (Shard(0), Replicate(), Replicate())
    onehot_out_placements = (Shard(0), Replicate(), Replicate())

    a = 64
    expected_depth = 21

    rng = np.random.RandomState(2024)
    shard0 = rng.randint(0, 15, size=(a,)).astype(np.int64)
    shard1 = shard0.copy()
    shard0[0] = 20
    shard1[0] = 20

    global_tensor = Tensor(np.concatenate([shard0, shard1], axis=0), ms.int64)

    standalone_net = OneHotExtNet()
    standalone_out = standalone_net(global_tensor, -1)
    assert standalone_out.shape[-1] == expected_depth

    dp_id = rank // 4
    local_np = shard0.copy() if dp_id == 0 else shard1.copy()
    tensor_local = DTensor.from_local(Tensor(local_np, ms.int64), mesh, tensor_placements)

    parallel_net = OneHotExtNet(device_mesh=mesh, relu_strategy=(onehot_out_placements,))
    parallel_out = parallel_net(tensor_local, -1)

    assert hasattr(parallel_out, "local_shape")
    assert parallel_out.local_shape[-1] == expected_depth

    parallel_out_global = parallel_out.full_tensor()
    if rank == 0:
        assert np.allclose(standalone_out.asnumpy(), parallel_out_global.asnumpy(), 1e-3, 1e-3)
