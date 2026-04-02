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
"""test argsort shard in python"""

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


class ArgSortNet(nn.Cell):
    """ArgSort network"""

    def __init__(self, dim=-1, descending=False):
        super().__init__()
        self.argsort = ops.auto_generate.argsort_ext
        self.argsort_dim = dim
        self.argsort_descending = descending

    def construct(self, x):
        indices = self.argsort(x, dim=self.argsort_dim, descending=self.argsort_descending)
        return indices


def test_argsort_data_parallel_1():
    """
    Feature: ArgSort in python shard.
    Description: Test ArgSort with data parallel (batch dimension sharded).
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

    d, m = 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = ArgSortNet(dim=1, descending=False)
    standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgSortNet(dim=1, descending=False)
    parallel_indices = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgSort data parallel test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")


def test_argsort_model_parallel_2():
    """
    Feature: ArgSort in python shard.
    Description: Test ArgSort with model parallel (feature dimension sharded).
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

    d, m = 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = ArgSortNet(dim=0, descending=False)
    standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgSortNet(dim=0, descending=False)
    parallel_indices = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgSort model parallel test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")


def test_argsort_negative_dim_3():
    """
    Feature: ArgSort in python shard.
    Description: Test ArgSort with negative dimension index.
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
    standalone_net = ArgSortNet(dim=-1, descending=False)
    standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgSortNet(dim=-1, descending=False)
    parallel_indices = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgSort negative dim test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")


def test_argsort_descending_4():
    """
    Feature: ArgSort in python shard.
    Description: Test ArgSort with descending=True.
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

    d, m = 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = ArgSortNet(dim=1, descending=True)
    standalone_indices = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ArgSortNet(dim=1, descending=True)
    parallel_indices = parallel_net(x_local)
    parallel_indices = parallel_indices.full_tensor()
    assert np.allclose(standalone_indices.asnumpy(), parallel_indices.asnumpy(), 1e-3, 1e-3), \
        (f"ArgSort descending test failed: "
         f"standalone={standalone_indices.asnumpy()}, "
         f"parallel={parallel_indices.asnumpy()}")
