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
"""test cumsum_ext shard in python"""

from typing import Any
import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module() -> None:
    """Initialize the distributed environment for the test module."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


class CumsumExtNet(nn.Cell):
    """CumsumExt network for testing"""

    def __init__(self, device_mesh=None, relu_strategy=None) -> None:
        """Initialize."""
        super().__init__()
        self.cumsum = ms.mint.cumsum
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x: Any, dim: Any) -> object:
        """Forward computation."""
        out = self.cumsum(x, dim)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "mp")


def test_cumsum_ext_data_parallel_1():
    """
    Feature: CumsumExt in python shard.
    Description: Test CumsumExt data parallel - cumsum on unsharded dimension.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    d, m = 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))

    standalone_net = CumsumExtNet()
    standalone_output = standalone_net(x, dim=1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate())
    relu_input_placements = (Shard(0), Replicate())

    x_local = distribute_tensor(x, mesh, x_placements)

    parallel_net = CumsumExtNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(x_local, dim=1)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3), \
        f"CumsumExt data parallel test failed, expected {standalone_output.asnumpy()}, got {parallel_output.asnumpy()}"


def test_cumsum_ext_negative_dim_2():
    """
    Feature: CumsumExt with negative dimension in python shard.
    Description: Test CumsumExt with negative dimension indexing.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)
    d, m = 8, 16
    x = Tensor(np.random.randn(d, m).astype(np.float32))

    standalone_net = CumsumExtNet()
    standalone_output = standalone_net(x, dim=-1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Replicate())
    relu_input_placements = (Shard(0), Replicate())

    x_local = distribute_tensor(x, mesh, x_placements)

    parallel_net = CumsumExtNet(device_mesh=mesh, relu_strategy=(relu_input_placements,))
    parallel_output = parallel_net(x_local, dim=-1)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3), \
        f"CumsumExt negative dim test failed, expected {standalone_output.asnumpy()}, got {parallel_output.asnumpy()}"
