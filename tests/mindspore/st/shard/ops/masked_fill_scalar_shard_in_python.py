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
"""test masked_fill_scalar shard in python"""

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


class MaskedFillScalarNet(nn.Cell):
    """MaskedFillScalar network composed of masked_fill operation and ReLU"""

    def __init__(self, value, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.masked_fill_scalar = ops.auto_generate.masked_fill_scalar_op
        self.relu = ms.nn.ReLU()
        self.value = value
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x, mask):
        out = self.masked_fill_scalar(x, mask, self.value)
        out = self.relu(out)
        out = out + 1
        return out


def test_masked_fill_scalar_same_shape_parallel_1():
    """
    Feature: MaskedFillScalar elementwise operation in python shard.
    Description: Test MaskedFillScalar with same-shape inputs in full parallel.
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m = 16, 256
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    mask = Tensor((np.random.randn(d, m) > 0).astype(np.bool_))
    value = 0.5

    standalone_net = MaskedFillScalarNet(value)
    standalone_output = standalone_net(x, mask)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Shard(1))
    mask_placements = (Shard(0), Shard(1))
    relu_placements = (Shard(0), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    mask_local = distribute_tensor(mask, mesh, mask_placements)

    parallel_net = MaskedFillScalarNet(value, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, mask_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_masked_fill_scalar_broadcast_dim0_parallel_2():
    """
    Feature: MaskedFillScalar elementwise operation with broadcasting in python shard.
    Description: Test MaskedFillScalar with broadcasting on dimension 0 (mask: [1, 256, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m = 16, 256
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    mask = Tensor((np.random.randn(1, m) > 0).astype(np.bool_))
    value = -1.0

    standalone_net = MaskedFillScalarNet(value)
    standalone_output = standalone_net(x, mask)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Shard(1))
    mask_placements = (Replicate(), Shard(1))
    relu_placements = (Shard(0), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    mask_local = distribute_tensor(mask, mesh, mask_placements)

    parallel_net = MaskedFillScalarNet(value, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, mask_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_masked_fill_scalar_broadcast_dim1_parallel_3():
    """
    Feature: MaskedFillScalar elementwise operation with broadcasting in python shard.
    Description: Test MaskedFillScalar with broadcasting on dimension 1 (mask: [16, 1, 128]).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m= 16, 256
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    mask = Tensor((np.random.randn(d, 1) > 0).astype(np.bool_))
    value = 2.0

    standalone_net = MaskedFillScalarNet(value)
    standalone_output = standalone_net(x, mask)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Shard(1))
    mask_placements = (Shard(0), Replicate())
    relu_placements = (Shard(0), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    mask_local = distribute_tensor(mask, mesh, mask_placements)

    parallel_net = MaskedFillScalarNet(value, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, mask_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_masked_fill_scalar_partial_shard_parallel_4():
    """
    Feature: MaskedFillScalar elementwise operation in python shard.
    Description: Test MaskedFillScalar with partial sharding (only one dimension sharded).
    Expectation: Run success.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d, m= 16, 256
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    mask = Tensor((np.random.randn(d, m) > 0).astype(np.bool_))
    value = 1.5

    standalone_net = MaskedFillScalarNet(value)
    standalone_output = standalone_net(x, mask)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Replicate(), Shard(1))
    mask_placements = (Replicate(), Shard(1))
    relu_placements = (Replicate(), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    mask_local = distribute_tensor(mask, mesh, mask_placements)

    parallel_net = MaskedFillScalarNet(value, device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, mask_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
