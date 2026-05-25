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
"""test activation_with_axis shard in python"""

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


class SoftmaxNet(nn.Cell):
    """Softmax network"""

    def __init__(self, axis=-1, device_mesh=None, relu_strategy=None) -> None:
        """Initialize."""
        super().__init__()
        self.softmax = ops.Softmax(axis)
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x: Any) -> object:
        """Forward computation."""
        out = self.softmax(x)
        out = self.relu(out)
        out = out + 1
        return out


class SwigluNet(nn.Cell):
    """Swiglu network"""

    def __init__(self, axis=-1, device_mesh=None, relu_strategy=None) -> None:
        """Initialize."""
        super().__init__()
        self.swiglu = ops.swiglu
        self.relu = ms.nn.ReLU()
        self.axis = axis
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x: Any) -> object:
        """Forward computation."""
        out = self.swiglu(x, self.axis)
        out = self.relu(out)
        out = out + 1
        return out


def test_activation_with_axis_data_parallel_1():
    """
    Feature: ActivationWithAxis (Softmax and Swiglu) in python shard.
    Description: Test Softmax and Swiglu with data parallel (batch dimension sharded).
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
    relu_placements = (Shard(0), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_softmax_net = SoftmaxNet(axis=-1)
    standalone_softmax_output = standalone_softmax_net(x_softmax)
    x_softmax_local = distribute_tensor(x_softmax, mesh, x_placements)
    parallel_softmax_net = SoftmaxNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_softmax_output = parallel_softmax_net(x_softmax_local)
    parallel_softmax_output = parallel_softmax_output.full_tensor()
    assert np.allclose(standalone_softmax_output.asnumpy(), parallel_softmax_output.asnumpy(), 1e-3, 1e-3), \
        (f"Softmax data parallel test failed: "
         f"standalone={standalone_softmax_output.asnumpy()}, "
         f"parallel={parallel_softmax_output.asnumpy()}")

    d, m, k = 8, 16, 32
    x_swiglu = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_swiglu_net = SwigluNet(axis=-1)
    standalone_swiglu_output = standalone_swiglu_net(x_swiglu)
    x_swiglu_local = distribute_tensor(x_swiglu, mesh, x_placements)
    parallel_swiglu_net = SwigluNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_swiglu_output = parallel_swiglu_net(x_swiglu_local)
    parallel_swiglu_output = parallel_swiglu_output.full_tensor()
    assert np.allclose(standalone_swiglu_output.asnumpy(), parallel_swiglu_output.asnumpy(), 1e-3, 1e-3), \
        (f"Swiglu data parallel test failed: "
         f"standalone={standalone_swiglu_output.asnumpy()}, "
         f"parallel={parallel_swiglu_output.asnumpy()}")


def test_activation_with_axis_model_parallel_2():
    """
    Feature: ActivationWithAxis (Softmax and Swiglu) in python shard.
    Description: Test Softmax and Swiglu with model parallel (feature dimension sharded).
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
    relu_placements = (Replicate(), Replicate(), Shard(2))

    d, m, k = 8, 16, 32
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_softmax_net = SoftmaxNet(axis=0)
    standalone_softmax_output = standalone_softmax_net(x_softmax)
    x_softmax_local = distribute_tensor(x_softmax, mesh, x_placements)
    parallel_softmax_net = SoftmaxNet(axis=0, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_softmax_output = parallel_softmax_net(x_softmax_local)
    parallel_softmax_output = parallel_softmax_output.full_tensor()
    assert np.allclose(standalone_softmax_output.asnumpy(), parallel_softmax_output.asnumpy(), 1e-3, 1e-3), \
        (f"Softmax model parallel test failed: "
         f"standalone={standalone_softmax_output.asnumpy()}, "
         f"parallel={parallel_softmax_output.asnumpy()}")

    d, m, k = 8, 16, 32
    x_swiglu = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_swiglu_net = SwigluNet(axis=0)
    standalone_swiglu_output = standalone_swiglu_net(x_swiglu)
    x_swiglu_local = distribute_tensor(x_swiglu, mesh, x_placements)
    parallel_swiglu_net = SwigluNet(axis=0, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_swiglu_output = parallel_swiglu_net(x_swiglu_local)
    parallel_swiglu_output = parallel_swiglu_output.full_tensor()
    assert np.allclose(standalone_swiglu_output.asnumpy(), parallel_swiglu_output.asnumpy(), 1e-3, 1e-3), \
        (f"Swiglu model parallel test failed: "
         f"standalone={standalone_swiglu_output.asnumpy()}, "
         f"parallel={parallel_swiglu_output.asnumpy()}")


def test_activation_with_axis_hybrid_parallel_3():
    """
    Feature: ActivationWithAxis (Softmax and Swiglu) in python shard.
    Description: Test Softmax and Swiglu with hybrid parallel (multiple dimensions sharded).
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
    relu_placements = (Shard(0), Replicate(), Shard(2))

    d, m, k = 8, 16, 32
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_softmax_net = SoftmaxNet(axis=1)
    standalone_softmax_output = standalone_softmax_net(x_softmax)
    x_softmax_local = distribute_tensor(x_softmax, mesh, x_placements)
    parallel_softmax_net = SoftmaxNet(axis=1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_softmax_output = parallel_softmax_net(x_softmax_local)
    parallel_softmax_output = parallel_softmax_output.full_tensor()
    assert np.allclose(standalone_softmax_output.asnumpy(), parallel_softmax_output.asnumpy(), 1e-3, 1e-3), \
        (f"Softmax hybrid parallel test failed: "
         f"standalone={standalone_softmax_output.asnumpy()}, "
         f"parallel={parallel_softmax_output.asnumpy()}")

    d, m, k = 8, 16, 32
    x_swiglu = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_swiglu_net = SwigluNet(axis=1)
    standalone_swiglu_output = standalone_swiglu_net(x_swiglu)
    x_swiglu_local = distribute_tensor(x_swiglu, mesh, x_placements)
    parallel_swiglu_net = SwigluNet(axis=1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_swiglu_output = parallel_swiglu_net(x_swiglu_local)
    parallel_swiglu_output = parallel_swiglu_output.full_tensor()
    assert np.allclose(standalone_swiglu_output.asnumpy(), parallel_swiglu_output.asnumpy(), 1e-3, 1e-3), \
        (f"Swiglu hybrid parallel test failed: "
         f"standalone={standalone_swiglu_output.asnumpy()}, "
         f"parallel={parallel_swiglu_output.asnumpy()}")


def test_activation_with_axis_all_replicated_4():
    """
    Feature: ActivationWithAxis (Softmax and Swiglu) in python shard.
    Description: Test Softmax and Swiglu with all replicated (no sharding).
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
    relu_placements = (Replicate(), Replicate(), Replicate())

    d, m, k = 8, 16, 32
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_softmax_net = SoftmaxNet(axis=-1)
    standalone_softmax_output = standalone_softmax_net(x_softmax)
    x_softmax_local = distribute_tensor(x_softmax, mesh, x_placements)
    parallel_softmax_net = SoftmaxNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_softmax_output = parallel_softmax_net(x_softmax_local)
    parallel_softmax_output = parallel_softmax_output.full_tensor()
    assert np.allclose(standalone_softmax_output.asnumpy(), parallel_softmax_output.asnumpy(), 1e-3, 1e-3), \
        (f"Softmax all replicated test failed: "
         f"standalone={standalone_softmax_output.asnumpy()}, "
         f"parallel={parallel_softmax_output.asnumpy()}")

    d, m, k = 8, 16, 32
    x_swiglu = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_swiglu_net = SwigluNet(axis=-1)
    standalone_swiglu_output = standalone_swiglu_net(x_swiglu)
    x_swiglu_local = distribute_tensor(x_swiglu, mesh, x_placements)
    parallel_swiglu_net = SwigluNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_swiglu_output = parallel_swiglu_net(x_swiglu_local)
    parallel_swiglu_output = parallel_swiglu_output.full_tensor()
    assert np.allclose(standalone_swiglu_output.asnumpy(), parallel_swiglu_output.asnumpy(), 1e-3, 1e-3), \
        (f"Swiglu all replicated test failed: "
         f"standalone={standalone_swiglu_output.asnumpy()}, "
         f"parallel={parallel_swiglu_output.asnumpy()}")


def test_activation_with_axis_negative_dim_5():
    """
    Feature: ActivationWithAxis (Softmax and Swiglu) in python shard.
    Description: Test Softmax and Swiglu with negative dimension index.
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
    relu_placements = (Shard(0), Shard(1), Replicate())

    d, m, k = 8, 16, 32
    x_softmax = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_softmax_net = SoftmaxNet(axis=-1)
    standalone_softmax_output = standalone_softmax_net(x_softmax)
    x_softmax_local = distribute_tensor(x_softmax, mesh, x_placements)
    parallel_softmax_net = SoftmaxNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_softmax_output = parallel_softmax_net(x_softmax_local)
    parallel_softmax_output = parallel_softmax_output.full_tensor()
    assert np.allclose(standalone_softmax_output.asnumpy(), parallel_softmax_output.asnumpy(), 1e-3, 1e-3), \
        (f"Softmax negative dim test failed: "
         f"standalone={standalone_softmax_output.asnumpy()}, "
         f"parallel={parallel_softmax_output.asnumpy()}")

    d, m, k = 8, 16, 32
    x_swiglu = Tensor(np.random.randn(d, m, k).astype(np.float32))
    standalone_swiglu_net = SwigluNet(axis=-1)
    standalone_swiglu_output = standalone_swiglu_net(x_swiglu)
    x_swiglu_local = distribute_tensor(x_swiglu, mesh, x_placements)
    parallel_swiglu_net = SwigluNet(axis=-1, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_swiglu_output = parallel_swiglu_net(x_swiglu_local)
    parallel_swiglu_output = parallel_swiglu_output.full_tensor()
    assert np.allclose(standalone_swiglu_output.asnumpy(), parallel_swiglu_output.asnumpy(), 1e-3, 1e-3), \
        (f"Swiglu negative dim test failed: "
         f"standalone={standalone_swiglu_output.asnumpy()}, "
         f"parallel={parallel_swiglu_output.asnumpy()}")
