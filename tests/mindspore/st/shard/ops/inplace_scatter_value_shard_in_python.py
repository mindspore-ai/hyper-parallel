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
"""test inplace_scatter_value shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module():
    ms.set_device("Ascend")
    D.init()


class InplaceScatterValueNet(nn.Cell):
    """InplaceScatterValue composed of scatter_ and ReLU"""

    def __init__(self, dim, value, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.dim = dim
        self.value = value
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, input_x, index):
        out = input_x.scatter_(self.dim, index, self.value)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "mp")


def test_inplace_scatter_value_data_parallel_1():
    """
    Feature: InplaceScatterValue in python shard.
    Description: Test InplaceScatterValue data parallel on dim 1.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b = 16, 256
    n = 8
    dim = 1
    value = 10.0

    input_x = Tensor(np.random.randn(a, b).astype(np.float32))
    index = Tensor(np.random.randint(0, b, size=(a, n)).astype(np.int64))

    standalone_net = InplaceScatterValueNet(dim, value)
    standalone_output = standalone_net(input_x, index)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    input_placements = (Shard(0), Replicate())
    index_placements = (Shard(0), Replicate())
    relu_input_placements = (Shard(0), Replicate())

    input_local = distribute_tensor(input_x, mesh, input_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    parallel_net = InplaceScatterValueNet(dim, value, device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, index_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_inplace_scatter_value_model_parallel_2():
    """
    Feature: InplaceScatterValue in python shard.
    Description: Test InplaceScatterValue model parallel on dim 2.
    Expectation: Run success.
    """
    ms.set_seed(1)
    a, b = 16, 256
    n = 8
    dim = 0
    value = 5.0

    input_x = Tensor(np.random.randn(a, b).astype(np.float32))
    index = Tensor(np.random.randint(0, a, size=(n, b)).astype(np.int64))

    standalone_net = InplaceScatterValueNet(dim, value)
    standalone_output = standalone_net(input_x, index)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    input_placements = (Replicate(), Shard(1))
    index_placements = (Replicate(), Shard(1))
    relu_input_placements = (Replicate(), Shard(1))

    input_local = distribute_tensor(input_x, mesh, input_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    parallel_net = InplaceScatterValueNet(dim, value, device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(input_local, index_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
