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
"""test max_dim shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


def setup_module():
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "tp")


class MaxDimNet(nn.Cell):
    """MaxDim network"""

    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.max_dim = ops.auto_generate.MaxDim()
        self.dim = dim
        self.keepdim = keepdim

    def construct(self, x):
        values, indices = self.max_dim(x, dim=self.dim, keepdim=self.keepdim)
        return values, indices


def test_max_dim_data_parallel_1():
    """
    Feature: MaxDim in python shard.
    Description: Test MaxDim with data parallel (batch dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate())

    d, m= 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = MaxDimNet(dim=1, keepdim=True)
    standalone_values, standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = MaxDimNet(dim=1, keepdim=True)
    parallel_values, parallel_indices = parallel_net(x_local)
    parallel_values = parallel_values.full_tensor()
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim data parallel values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim data parallel indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")


def test_max_dim_model_parallel_2():
    """
    Feature: MaxDim in python shard.
    Description: Test MaxDim with model parallel (feature dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Replicate(), Shard(1))

    d, m= 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = MaxDimNet(dim=0, keepdim=True)
    standalone_values, standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = MaxDimNet(dim=0, keepdim=True)
    parallel_values, parallel_indices = parallel_net(x_local)
    parallel_values = parallel_values.full_tensor()
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim model parallel values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim model parallel indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")


def test_max_dim_negative_dim_3():
    """
    Feature: MaxDim in python shard.
    Description: Test MaxDim with negative dimension index.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate())

    d, m= 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = MaxDimNet(dim=-1, keepdim=True)
    standalone_values, standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = MaxDimNet(dim=-1, keepdim=True)
    parallel_values, parallel_indices = parallel_net(x_local)
    parallel_values = parallel_values.full_tensor()
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim negative dim values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim negative dim indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")


def test_max_dim_keepdim_false_4():
    """
    Feature: MaxDim in python shard.
    Description: Test MaxDim with keepdim=False.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate())

    d, m= 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = MaxDimNet(dim=1, keepdim=False)
    standalone_values, standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = MaxDimNet(dim=1, keepdim=False)
    parallel_values, parallel_indices = parallel_net(x_local)
    parallel_values = parallel_values.full_tensor()
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim keepdim=False values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"MaxDim keepdim=False indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
