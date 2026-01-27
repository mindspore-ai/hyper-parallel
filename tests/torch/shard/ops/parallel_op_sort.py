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
"""test torch dtensor with distributed sort"""

import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Set seed for reproducibility across ranks
np.random.seed(42)
torch.manual_seed(42)

# Generate random data
standalone_input_2d_np = np.random.randn(8, 16).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 8, 6).astype(np.float32)


def test_distributed_sort_basic():
    """
    Feature: dtensor + torch.sort basic execution
    Description:
        - Sort along an unsharded dimension (last dim).
        - Input: shape (8, 16) sharded on dim=0, sort dim=1.
    Expectation:
        - Output values and indices match standalone torch.sort result.
        - Output layout preserves input sharding.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    # sort returns (values, indices)
    s_values, s_indices = torch.sort(standalone_input, dim=1, descending=False)

    # Distributed setup: Shard dim 0 ("dp"), Replicate dim 1 (sort dim)
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=1, descending=False)

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 2)
    assert d_values.layout == expected_layout, \
        f"Values layout mismatch: expected {expected_layout}, got {d_values.layout}"
    assert d_indices.layout == expected_layout, \
        f"Indices layout mismatch: expected {expected_layout}, got {d_indices.layout}"

    # Numerical validation
    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "Sort values mismatch"
    assert torch.equal(s_indices, g_indices), "Sort indices mismatch"


def test_distributed_sort_descending():
    """
    Feature: dtensor + torch.sort with descending=True
    Description:
        - Sort along unsharded dimension with descending order.
        - Input: shape (8, 16) sharded on dim=0.
    Expectation: Results match standalone execution.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    s_values, s_indices = torch.sort(standalone_input, dim=1, descending=True)

    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=1, descending=True)

    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "Sort descending values mismatch"
    assert torch.equal(s_indices, g_indices), "Sort descending indices mismatch"


def test_distributed_sort_middle_dim():
    """
    Feature: dtensor + torch.sort on middle dimension
    Description:
        - Input shape (4, 8, 6).
        - Shard dim=0 and dim=2. Sort along dim=1 (unsharded).
    Expectation: Correct sorting of the middle dimension while preserving sharding on others.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    # Sort along dim 1
    s_values, s_indices = torch.sort(standalone_input, dim=1)

    # Mesh: (dp=4, tp=2). Input layout: (Shard(0), Replicate(), Shard(1))
    mesh = init_device_mesh(mesh_shape=(4, 2), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=1)

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 3)
    assert d_values.layout == expected_layout
    assert d_indices.layout == expected_layout

    # Numerical validation
    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "3D Sort values mismatch"
    assert torch.equal(s_indices, g_indices), "3D Sort indices mismatch"


def test_distributed_sort_negative_dim():
    """
    Feature: dtensor + torch.sort with negative dimension index
    Description:
        - Input shape (8, 16). Sharded on dim 0.
        - Sort using dim=-1 (which maps to unsharded dim 1).
    Expectation: Correctly infers the dimension and executes sort.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    s_values, s_indices = torch.sort(standalone_input, dim=-1)

    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    d_values, d_indices = dist_input.sort(dim=-1)

    g_values = local_to_global(d_values)
    g_indices = local_to_global(d_indices)

    assert torch.equal(s_values, g_values), "Sort negative dim values mismatch"
    assert torch.equal(s_indices, g_indices), "Sort negative dim indices mismatch"


def test_distributed_sort_sharded_dim_error():
    """
    Feature: dtensor + torch.sort error handling
    Description:
        - Attempt to sort along a dimension that is currently sharded.
    Expectation: Raises ValueError because sorting sharded data requires
    redistribution (not auto-supported in sort op yet).
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    # Shard dim 0 and Replicate dim 1
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    # Try to sort dim 0 (which is sharded)
    try:
        dist_input.sort(dim=0)
        assert False, "Expected ValueError when sorting sharded dimension"
    except ValueError as e:
        # Match the error message from DistributedSortOp
        assert "sorting along a sharded dimension" in str(e), \
            f"Unexpected error message: {str(e)}"
