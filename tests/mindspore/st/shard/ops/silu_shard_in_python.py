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
"""test SiLU shard in python"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module():
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "tp")


class SiLUNet(nn.Cell):
    """SiLU network composed of SiLU operation and ReLU"""

    def __init__(self, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.silu = ops.silu
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x):
        out = self.silu(x)
        out = self.relu(out)
        out = out + 1
        return out


def test_silu_data_parallel_1():
    """
    Feature: SiLU in python shard.
    Description: Test SiLU with data parallel (batch dimension sharded).
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
    relu_placements = (Shard(0), Replicate())

    d, m= 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = SiLUNet()
    standalone_output = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = SiLUNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local)
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3), \
        (f"SiLU data parallel test failed: "
         f"max_diff={np.max(np.abs(standalone_output.asnumpy() - parallel_output.asnumpy()))}")


def test_silu_model_parallel_2():
    """
    Feature: SiLU in python shard.
    Description: Test SiLU with model parallel (feature dimension sharded).
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
    relu_placements = (Replicate(), Shard(1))

    d, m = 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = SiLUNet()
    standalone_output = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = SiLUNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local)
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3), \
        (f"SiLU model parallel test failed: "
         f"max_diff={np.max(np.abs(standalone_output.asnumpy() - parallel_output.asnumpy()))}")
