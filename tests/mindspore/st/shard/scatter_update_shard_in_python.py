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
"""test scatter update shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import init_device_mesh, shard_module, DTensor
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module():
    ms.set_device("Ascend")
    D.init()


class ScatterUpdateNet(nn.Cell):
    """ScatterUpdate composed of scatter_update and ReLU"""

    def __init__(self, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.scatter_update = ops.ScatterUpdate()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, input_x, indices, updates):
        out = self.scatter_update(input_x, indices, updates)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")


def test_scatter_update_data_parallel_1():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate data parallel on dim 1.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Define placements using Placement format
    input_placements = (Shard(1), Replicate(), Replicate())
    indices_placements = (Replicate(), Replicate(), Replicate())
    updates_placements = (Shard(1), Replicate(), Replicate())
    relu_input_placements = (Shard(1), Replicate(), Replicate())

    input_local = DTensor.distribute_tensor(input_x, mesh, input_placements)
    indices_local = DTensor.distribute_tensor(indices, mesh, indices_placements)
    updates_local = DTensor.distribute_tensor(updates, mesh, updates_placements)

    parallel_net = ScatterUpdateNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_model_parallel_2():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate model parallel on dim 2.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Define placements using Placement format
    input_placements = (Replicate(), Replicate(), Shard(2))
    indices_placements = (Replicate(), Replicate(), Replicate())
    updates_placements = (Replicate(), Replicate(), Shard(2))
    relu_input_placements = (Replicate(), Replicate(), Shard(2))

    input_local = DTensor.distribute_tensor(input_x, mesh, input_placements)
    indices_local = DTensor.distribute_tensor(indices, mesh, indices_placements)
    updates_local = DTensor.distribute_tensor(updates, mesh, updates_placements)

    parallel_net = ScatterUpdateNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_hybrid_parallel_3():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate hybrid parallel on dim 1 and 2.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Define placements using Placement format
    input_placements = (Shard(1), Replicate(), Shard(2))
    indices_placements = (Replicate(), Replicate(), Replicate())
    updates_placements = (Shard(1), Replicate(), Shard(2))
    relu_input_placements = (Shard(1), Replicate(), Shard(2))

    input_local = DTensor.distribute_tensor(input_x, mesh, input_placements)
    indices_local = DTensor.distribute_tensor(indices, mesh, indices_placements)
    updates_local = DTensor.distribute_tensor(updates, mesh, updates_placements)

    parallel_net = ScatterUpdateNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_multi_dim_indices_4():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate with 2D indices.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n, o = 4, 2

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n, o)).astype(np.int32))
    updates = Tensor(np.random.randn(n, o, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Define placements using Placement format
    input_placements = (Shard(1), Replicate(), Shard(2))
    indices_placements = (Replicate(), Replicate(), Replicate())
    updates_placements = (Shard(2), Replicate(), Shard(3))
    relu_input_placements = (Shard(1), Replicate(), Shard(2))

    input_local = DTensor.distribute_tensor(input_x, mesh, input_placements)
    indices_local = DTensor.distribute_tensor(indices, mesh, indices_placements)
    updates_local = DTensor.distribute_tensor(updates, mesh, updates_placements)

    parallel_net = ScatterUpdateNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_scatter_update_replicate_all_5():
    """
    Feature: ScatterUpdate in python shard.
    Description: Test ScatterUpdate with full replication.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b, c = 16, 256, 128
    n = 8

    input_x = Tensor(np.random.randn(a, b, c).astype(np.float32))
    indices = Tensor(np.random.randint(0, a, size=(n,)).astype(np.int32))
    updates = Tensor(np.random.randn(n, b, c).astype(np.float32))

    # Standalone
    standalone_net = ScatterUpdateNet()
    standalone_output = standalone_net(input_x, indices, updates)

    # Parallel
    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Define placements using Placement format
    input_placements = (Replicate(), Replicate(), Replicate())
    indices_placements = (Replicate(), Replicate(), Replicate())
    updates_placements = (Replicate(), Replicate(), Replicate())
    relu_input_placements = (Replicate(), Replicate(), Replicate())

    input_local = DTensor.distribute_tensor(input_x, mesh, input_placements)
    indices_local = DTensor.distribute_tensor(indices, mesh, indices_placements)
    updates_local = DTensor.distribute_tensor(updates, mesh, updates_placements)

    parallel_net = ScatterUpdateNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, indices_local, updates_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
