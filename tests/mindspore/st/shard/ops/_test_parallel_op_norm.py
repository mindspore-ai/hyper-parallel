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
"""test norm shard in python"""

from typing import Any
import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor, ops
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module() -> None:
    """Initialize the distributed environment for the test module."""
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "tp")


class RmsNormNet(nn.Cell):
    """RmsNorm network"""

    def __init__(self, device_mesh=None, relu_strategy=None) -> None:
        """Initialize."""
        super().__init__()
        self.rmsnorm = ops.RmsNorm()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x: Any, weight: Any) -> object:
        """Forward computation."""
        out, _ = self.rmsnorm(x, weight)
        out = self.relu(out)
        out = out + 1
        return out


def test_norm_data_parallel_1():
    """
    Feature: RmsNorm in python shard.
    Description: Test RmsNorm with data parallel (batch dimension sharded).
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
    weight_placements = (Replicate(), )
    relu_placements = (Shard(0), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    normalized_shape = (k,)

    x_rmsnorm = Tensor(np.random.randn(d, m, k).astype(np.float32))
    weight = Tensor(np.ones(normalized_shape).astype(np.float32))
    standalone_rmsnorm_net = RmsNormNet()
    standalone_rmsnorm_output = standalone_rmsnorm_net(x_rmsnorm, weight)
    x_rmsnorm_local = distribute_tensor(x_rmsnorm, mesh, x_placements)
    weight_local = distribute_tensor(weight, mesh, weight_placements)
    parallel_rmsnorm_net = RmsNormNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_rmsnorm_output = parallel_rmsnorm_net(x_rmsnorm_local, weight_local)
    parallel_rmsnorm_output = parallel_rmsnorm_output.full_tensor()
    assert np.allclose(standalone_rmsnorm_output.asnumpy(), parallel_rmsnorm_output.asnumpy(), 1e-3, 1e-3), (
        f"RmsNorm data parallel test failed: "
        f"standalone={standalone_rmsnorm_output.asnumpy()}, "
        f"parallel={parallel_rmsnorm_output.asnumpy()}"
    )


def test_norm_model_parallel_2():
    """
    Feature: RmsNorm in python shard.
    Description: Test RmsNorm with model parallel (sequence dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Replicate(), Shard(1), Replicate())
    weight_placements = (Replicate(),)
    relu_placements = (Replicate(), Shard(1), Replicate())

    d, m, k = 8, 16, 32
    normalized_shape = (k,)

    x_rmsnorm = Tensor(np.random.randn(d, m, k).astype(np.float32))
    weight = Tensor(np.ones(normalized_shape).astype(np.float32))
    standalone_rmsnorm_net = RmsNormNet()
    standalone_rmsnorm_output = standalone_rmsnorm_net(x_rmsnorm, weight)
    x_rmsnorm_local = distribute_tensor(x_rmsnorm, mesh, x_placements)
    weight_local = distribute_tensor(weight, mesh, weight_placements)
    parallel_rmsnorm_net = RmsNormNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_rmsnorm_output = parallel_rmsnorm_net(x_rmsnorm_local, weight_local)
    parallel_rmsnorm_output = parallel_rmsnorm_output.full_tensor()
    assert np.allclose(standalone_rmsnorm_output.asnumpy(), parallel_rmsnorm_output.asnumpy(), 1e-3, 1e-3), (
        f"RmsNorm model parallel test failed: "
        f"standalone={standalone_rmsnorm_output.asnumpy()}, "
        f"parallel={parallel_rmsnorm_output.asnumpy()}"
    )


def test_norm_hybrid_parallel_3():
    """
    Feature: RmsNorm in python shard.
    Description: Test RmsNorm with hybrid parallel (multiple dimensions sharded).
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
    weight_placements = (Replicate(),)
    relu_placements = (Shard(0), Shard(1), Replicate())

    d, m, k = 8, 16, 32
    normalized_shape = (k,)

    x_rmsnorm = Tensor(np.random.randn(d, m, k).astype(np.float32))
    weight = Tensor(np.ones(normalized_shape).astype(np.float32))
    standalone_rmsnorm_net = RmsNormNet()
    standalone_rmsnorm_output = standalone_rmsnorm_net(x_rmsnorm, weight)
    x_rmsnorm_local = distribute_tensor(x_rmsnorm, mesh, x_placements)
    weight_local = distribute_tensor(weight, mesh, weight_placements)
    parallel_rmsnorm_net = RmsNormNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_rmsnorm_output = parallel_rmsnorm_net(x_rmsnorm_local, weight_local)
    parallel_rmsnorm_output = parallel_rmsnorm_output.full_tensor()
    assert np.allclose(standalone_rmsnorm_output.asnumpy(), parallel_rmsnorm_output.asnumpy(), 1e-3, 1e-3), (
        f"RmsNorm hybrid parallel test failed: "
        f"standalone={standalone_rmsnorm_output.asnumpy()}, "
        f"parallel={parallel_rmsnorm_output.asnumpy()}"
    )


def test_norm_all_replicated_4():
    """
    Feature: RmsNorm in python shard.
    Description: Test RmsNorm with all replicated (no sharding).
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
    weight_placements = (Replicate(),)
    relu_placements = (Replicate(), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    normalized_shape = (k,)

    x_rmsnorm = Tensor(np.random.randn(d, m, k).astype(np.float32))
    weight = Tensor(np.ones(normalized_shape).astype(np.float32))
    standalone_rmsnorm_net = RmsNormNet()
    standalone_rmsnorm_output = standalone_rmsnorm_net(x_rmsnorm, weight)
    x_rmsnorm_local = distribute_tensor(x_rmsnorm, mesh, x_placements)
    weight_local = distribute_tensor(weight, mesh, weight_placements)
    parallel_rmsnorm_net = RmsNormNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_rmsnorm_output = parallel_rmsnorm_net(x_rmsnorm_local, weight_local)
    parallel_rmsnorm_output = parallel_rmsnorm_output.full_tensor()
    assert np.allclose(standalone_rmsnorm_output.asnumpy(), parallel_rmsnorm_output.asnumpy(), 1e-3, 1e-3), (
        f"RmsNorm all replicated test failed: "
        f"standalone={standalone_rmsnorm_output.asnumpy()}, "
        f"parallel={parallel_rmsnorm_output.asnumpy()}"
    )
