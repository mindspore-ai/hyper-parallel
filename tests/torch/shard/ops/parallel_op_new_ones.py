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
"""test torch dtensor with distributed new_ones"""

import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
# Input tensor shape and content don't affect new_ones output values,
# but we need a tensor to call .new_ones() on.
standalone_input_np = np.random.randn(4, 4).astype(np.float32)


def test_distributed_new_ones_tuple_size():
    """
    Feature: dtensor + torch.Tensor.new_ones with tuple size
    Description:
        - Create a new tensor of ones with specific tuple shape from a distributed tensor.
        - Input: Sharded tensor.
        - Output: Should be fully Replicated (all dimensions -1) regardless of input sharding.
    Expectation: Output layout is Replicated, values match standalone execution.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    size = (3, 5)
    standalone_output = standalone_input.new_ones(size)

    # Distributed setup: Input sharded on dim 0 and 1
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    # Layout validation: Output must be fully replicated
    # size is (3, 5) -> 2D -> Expect (Replicate(), Replicate())
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Tuple size layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Tuple size output mismatch between standalone and distributed execution"


def test_distributed_new_ones_list_size():
    """
    Feature: dtensor + torch.Tensor.new_ones with list size
    Description:
        - Call new_ones with a list argument for size.
    Expectation: Success with Replicated layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    size = [2, 2, 2]
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    # Layout validation: 3D Replicated
    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"List size layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "List size output mismatch"


def test_distributed_new_ones_int_size():
    """
    Feature: dtensor + torch.Tensor.new_ones with int size
    Description:
        - Call new_ones with a single integer (1D tensor creation).
    Expectation: Success with 1D Replicated layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    size = 8
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    # Layout validation: 1D Replicated
    expected_layout = _build_layout(mesh, (Replicate(),), 1)
    assert dist_output.layout == expected_layout, \
        f"Int size layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Int size output mismatch"


def test_distributed_new_ones_scalar():
    """
    Feature: dtensor + torch.Tensor.new_ones scalar
    Description:
        - Call new_ones with empty tuple to create a scalar tensor.
    Expectation: Success with scalar Replicated layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    size = ()
    standalone_output = standalone_input.new_ones(size)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    # Layout validation: 0-dim Replicated (empty tuple placements)
    expected_layout = _build_layout(mesh, (), 0)
    assert dist_output.layout == expected_layout, \
        f"Scalar layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Scalar output mismatch"


def test_distributed_new_ones_input_sharding_ignored():
    """
    Feature: dtensor + torch.Tensor.new_ones ignores input sharding
    Description:
        - Verify that even if the input tensor is heavily sharded, the new_ones output
          is created as Replicated.
    Expectation: Output layout is strictly Replicated.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    size = (4, 4)
    standalone_output = standalone_input.new_ones(size)

    # Heavily sharded input: dim0 on dp(2), dim1 on tp(4)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.new_ones(size)

    # Expect Replicated, NOT Sharded
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    # Double check it is not sharded
    assert dist_output.layout == expected_layout, \
        f"Sharding ignore check failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "Output values mismatch with sharded input"
