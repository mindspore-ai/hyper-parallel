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
"""
test ones_like shard in python
"""

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


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")


class OnesLikeNet(nn.Cell):
    """OnesLike network composed of mint.ones_like and ReLUs"""

    def __init__(self, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.ones_like = ms.mint.ones_like
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x, dtype=None):
        # Output should have same layout as x
        out = self.ones_like(x, dtype=dtype)
        out = self.relu(out)
        return out


def _standalone_and_parallel_run(x, mesh, x_placements, relu_placements, dtype=None):
    """Run standalone and parallel graph and return outputs."""
    # Standalone
    standalone_net = OnesLikeNet()
    standalone_output = standalone_net(x, dtype=dtype)

    # Parallel
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = OnesLikeNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dtype=dtype)

    parallel_output = parallel_output.full_tensor()
    return standalone_output, parallel_output


def test_ones_like_basic_3d_1():
    """
    Feature: OnesLike in python shard.
    Description: Test OnesLike on a 3D sharded tensor.
    Expectation: Output matches standalone (full of ones).
    """
    ms.set_seed(1)
    np.random.seed(1)

    d0, d1, d2 = 16, 64, 32
    # Input data doesn't affect ones_like values, but affects layout
    x = Tensor(np.random.randn(d0, d1, d2).astype(np.float32))

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Shard input: (Shard(0), Shard(1), Shard(2))
    x_placements = (Shard(0), Shard(1), Shard(2))

    # OnesLike must preserve the layout. ReLU should expect the same.
    relu_placements = (Shard(0), Shard(1), Shard(2))

    standalone_output, parallel_output = _standalone_and_parallel_run(
        x, mesh, x_placements, relu_placements=relu_placements
    )

    assert np.allclose(
        standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3
    )
    # Validate it is actually ones
    assert np.all(parallel_output.asnumpy() == 1.0)


def test_ones_like_with_dtype_2():
    """
    Feature: OnesLike in python shard.
    Description: Test OnesLike with explicit dtype conversion on a 4D tensor.
    Expectation: Output matches standalone and has correct dtype.
    """
    ms.set_seed(2)
    np.random.seed(2)

    d0, d1, d2, d3 = 8, 16, 32, 64
    x = Tensor(np.random.randn(d0, d1, d2, d3).astype(np.float32))

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Shard input on dim 2
    x_placements = (Replicate(), Replicate(), Shard(2), Replicate())
    relu_placements = (Replicate(), Replicate(), Shard(2), Replicate())

    standalone_output, parallel_output = _standalone_and_parallel_run(
        x, mesh, x_placements, relu_placements=relu_placements, dtype=ms.float16
    )

    assert parallel_output.dtype == ms.float16
    assert np.allclose(
        standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3
    )
