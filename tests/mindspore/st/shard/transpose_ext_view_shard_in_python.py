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
"""
test transpose ext view shard in python
"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import init_device_mesh, shard_module, DTensor
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module():
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2, 2)
base_alias_name = ("dp", "cp", "mp")


class TransposeExtViewNet(nn.Cell):
    """TransposeExtView composed of transpose and ReLUs"""

    def __init__(self, device_mesh=None, relu_strategy=None):
        super().__init__()
        self.transpose = ms.mint.transpose
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x, dim0, dim1):
        out = self.transpose(x, dim0, dim1)
        out = self.relu(out)
        out = out + 1
        return out


def _standalone_and_parallel_run(x, mesh, x_placements, dim0, dim1, relu_placements):
    """Run standalone and parallel graph and return outputs."""
    # Standalone
    standalone_net = TransposeExtViewNet()
    standalone_output = standalone_net(x, dim0, dim1)

    # Parallel
    x_local = DTensor.distribute_tensor(x, mesh, x_placements)
    parallel_net = TransposeExtViewNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim0, dim1)

    parallel_output = parallel_output.full_tensor()
    return standalone_output, parallel_output


def test_transpose_ext_view_basic_3d_1():
    """
    Feature: TransposeExtView in python shard.
    Description: Test TransposeExtView swaps two dims on a 3D tensor.
    Expectation: Output matches standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    d0, d1, d2 = 16, 64, 32
    x = Tensor(np.random.randn(d0, d1, d2).astype(np.float32))

    mesh = init_device_mesh(
        mesh_shape=base_mesh_shape,
        alias_name=base_alias_name
    )

    # Shard input to ensure dtensor path is exercised
    # x_layout = layout("dp", "cp", "mp") -> (Shard(0), Shard(1), Shard(2))
    x_placements = (Shard(0), Shard(1), Shard(2))

    # After transpose(0, 2): shape becomes (d2, d1, d0).
    # Keep ReLU sharding simple: shard last dim by mp.
    # relu_in_layout = layout("None", "None", "mp") -> (Replicate(), Replicate(), Shard(2))
    relu_placements = (Replicate(), Replicate(), Shard(2))

    standalone_output, parallel_output = _standalone_and_parallel_run(
        x, mesh, x_placements, dim0=0, dim1=2, relu_placements=relu_placements
    )

    assert np.allclose(
        standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3
    )


def test_transpose_ext_view_negative_dims_2():
    """
    Feature: TransposeExtView in python shard.
    Description: Test TransposeExtView supports negative dims (MindSpore semantics).
    Expectation: Output matches standalone.
    """
    ms.set_seed(2)
    np.random.seed(2)

    d0, d1, d2, d3 = 8, 16, 32, 64
    x = Tensor(np.random.randn(d0, d1, d2, d3).astype(np.float32))

    mesh = init_device_mesh(
        mesh_shape=base_mesh_shape,
        alias_name=base_alias_name
    )

    # Shard input: shard dim2 by mp; others replicated
    # x_layout = layout("None", "None", "mp", "None") -> (Replicate(), Replicate(), Shard(2), Replicate())
    x_placements = (Replicate(), Replicate(), Shard(2), Replicate())

    # swap(-1, -3) => swap dim3 and dim1
    # output shape becomes (d0, d3, d2, d1)
    # relu_in_layout = layout("None", "mp", "None", "None") -> (Replicate(), Replicate(), Shard(1), Replicate())
    relu_placements = (Replicate(), Replicate(), Shard(1), Replicate())

    standalone_output, parallel_output = _standalone_and_parallel_run(
        x, mesh, x_placements, dim0=-1, dim1=-3, relu_placements=relu_placements
    )

    assert np.allclose(
        standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3
    )


def test_transpose_ext_view_same_dims_noop_3():
    """
    Feature: TransposeExtView in python shard.
    Description: Test TransposeExtView is a no-op when dim0 == dim1.
    Expectation: Output matches standalone.
    """
    ms.set_seed(3)
    np.random.seed(3)

    d0, d1, d2 = 4, 32, 128
    x = Tensor(np.random.randn(d0, d1, d2).astype(np.float32))

    mesh = init_device_mesh(
        mesh_shape=base_mesh_shape,
        alias_name=base_alias_name
    )

    # x_layout = layout("dp", "cp", "mp") -> (Shard(0), Shard(1), Shard(2))
    x_placements = (Shard(0), Shard(1), Shard(2))
    # relu_in_layout = layout("dp", "cp", "mp") -> (Shard(0), Shard(1), Shard(2))
    relu_placements = (Shard(0), Shard(1), Shard(2))

    standalone_output, parallel_output = _standalone_and_parallel_run(
        x, mesh, x_placements, dim0=1, dim1=1, relu_placements=relu_placements
    )

    assert np.allclose(
        standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3
    )
