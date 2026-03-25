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
"""test argmax_with_value shard in python"""

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


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "tp")


class ArgMaxWithValueNet(nn.Cell):
    """ArgMaxWithValue network"""

    def __init__(self, axis=-1, keep_dims=True):
        super().__init__()
        self.argmax_with_value = ops.ArgMaxWithValue(axis=axis, keep_dims=keep_dims)

    def construct(self, x):
        indices, values = self.argmax_with_value(x)
        return indices, values


def test_argmax_with_value_data_parallel_1():
    """
    Feature: ArgMaxWithValue in python shard.
    Description: Test ArgMaxWithValue with data parallel (batch dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_net = ArgMaxWithValueNet(axis=1, keep_dims=True)
    standalone_indices, standalone_values = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgMaxWithValueNet(axis=1, keep_dims=True)
    parallel_indices, parallel_values = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    parallel_values = parallel_values.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue data parallel indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue data parallel values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")


def test_argmax_with_value_model_parallel_2():
    """
    Feature: ArgMaxWithValue in python shard.
    Description: Test ArgMaxWithValue with model parallel (feature dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Replicate(), Replicate(), Shard(2))

    d, m, k = 8, 16, 32
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_net = ArgMaxWithValueNet(axis=0, keep_dims=True)
    standalone_indices, standalone_values = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgMaxWithValueNet(axis=0, keep_dims=True)
    parallel_indices, parallel_values = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    parallel_values = parallel_values.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue model parallel indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue model parallel values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")


def test_argmax_with_value_hybrid_parallel_3():
    """
    Feature: ArgMaxWithValue in python shard.
    Description: Test ArgMaxWithValue with hybrid parallel (multiple dimensions sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate(), Shard(2))

    d, m, k = 8, 16, 32
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_net = ArgMaxWithValueNet(axis=1, keep_dims=True)
    standalone_indices, standalone_values = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgMaxWithValueNet(axis=1, keep_dims=True)
    parallel_indices, parallel_values = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    parallel_values = parallel_values.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue hybrid parallel indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue hybrid parallel values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")


def test_argmax_with_value_all_replicated_4():
    """
    Feature: ArgMaxWithValue in python shard.
    Description: Test ArgMaxWithValue with all replicated (no sharding).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Replicate(), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_net = ArgMaxWithValueNet(axis=-1, keep_dims=True)
    standalone_indices, standalone_values = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgMaxWithValueNet(axis=-1, keep_dims=True)
    parallel_indices, parallel_values = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    parallel_values = parallel_values.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue all replicated indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue all replicated values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")


def test_argmax_with_value_negative_dim_5():
    """
    Feature: ArgMaxWithValue in python shard.
    Description: Test ArgMaxWithValue with negative dimension index.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Shard(1), Replicate())

    d, m, k = 8, 16, 32
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_net = ArgMaxWithValueNet(axis=-1, keep_dims=True)
    standalone_indices, standalone_values = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgMaxWithValueNet(axis=-1, keep_dims=True)
    parallel_indices, parallel_values = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    parallel_values = parallel_values.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue negative dim indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue negative dim values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")


def test_argmax_with_value_keep_dims_false_6():
    """
    Feature: ArgMaxWithValue in python shard.
    Description: Test ArgMaxWithValue with keep_dims=False.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    x = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_net = ArgMaxWithValueNet(axis=1, keep_dims=False)
    standalone_indices, standalone_values = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgMaxWithValueNet(axis=1, keep_dims=False)
    parallel_indices, parallel_values = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    parallel_values = parallel_values.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue keep_dims=False indices test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
    assert np.allclose(standalone_values.asnumpy(), parallel_values.asnumpy(), 1e-3, 1e-3), \
        (f"ArgMaxWithValue keep_dims=False values test failed: "
         f"standalone={standalone_values.asnumpy()}, "
         f"parallel={parallel_values.asnumpy()}")
