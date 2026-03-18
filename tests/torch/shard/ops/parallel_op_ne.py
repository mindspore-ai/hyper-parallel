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
"""test torch dtensor with distributed ne (not equal) operator"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
# Base input arrays
input_a_2d_np = np.random.randn(8, 4).astype(np.float32)
input_b_2d_np = np.random.randn(8, 4).astype(np.float32)

# Make some elements exactly equal to test 'ne' (not equal) properly
input_b_2d_np[0:4, :] = input_a_2d_np[0:4, :]

# Broadcasting arrays
input_c_2d_np = np.random.randn(8, 1).astype(np.float32)
input_d_2d_np = np.random.randn(1, 4).astype(np.float32)


def test_distributed_ne_basic():
    """
    Feature: dtensor + torch.ne basic element-wise comparison.
    Description:
        - Compare two tensors with the exact same shape and sharding layout.
        - Input: shape (8, 4) sharded on dim=0.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_a = torch.from_numpy(input_a_2d_np).npu()
    standalone_b = torch.from_numpy(input_b_2d_np).npu()
    standalone_output = torch.ne(standalone_a, standalone_b)

    # Distributed setup: shard dim=0 ("dp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a, mesh, placements)
    dist_b = distribute_tensor(standalone_b, mesh, placements)

    dist_output = torch.ne(dist_a, dist_b)

    # Layout validation
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Ne output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Ne output mismatch between standalone and distributed execution"


def test_distributed_ne_scalar():
    """
    Feature: dtensor + torch.ne scalar comparison.
    Description:
        - Compare a sharded tensor with a scalar value.
    Expectation: Success with layout inheriting the tensor's sharding and correct boolean results.
    """
    init_dist()

    standalone_a = torch.from_numpy(input_a_2d_np).npu()
    scalar_val = input_a_2d_np[0, 0].item() # Pick a specific value from the array
    standalone_output = torch.ne(standalone_a, scalar_val)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))

    dist_a = distribute_tensor(standalone_a, mesh, placements)

    # Compare DTensor with scalar
    dist_output = torch.ne(dist_a, scalar_val)

    # Layout validation: Should inherit the layout of dist_a
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, \
        f"Ne scalar layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Ne scalar comparison output mismatch"


def test_distributed_ne_partial_error():
    """
    Feature: dtensor + torch.ne Partial input handling.
    Description:
        - Attempt to perform 'ne' operation on a tensor with Partial status.
        - Logical operators return booleans and do not support partial accumulations.
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_a = torch.from_numpy(input_a_2d_np).npu()
    standalone_b = torch.from_numpy(input_b_2d_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Assign Partial placement
    a_placements = (Partial("sum"), Replicate())
    b_placements = (Partial("sum"), Replicate())

    dist_a = distribute_tensor(standalone_a, mesh, a_placements)
    dist_b = distribute_tensor(standalone_b, mesh, b_placements)

    try:
        torch.ne(dist_a, dist_b)
        assert False, "Expected ValueError when passing Partial tensor to 'ne' operator"
    except ValueError as e:
        assert "Partial status which is not allowed" in str(e), \
            f"Unexpected error message: {str(e)}"
