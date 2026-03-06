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
"""test torch dtensor with distributed atleast_1d"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_0d_np = np.array(3.14, dtype=np.float32)
standalone_1d_np = np.random.randn(8).astype(np.float32)
standalone_2d_np = np.random.randn(8, 4).astype(np.float32)


def test_distributed_atleast_1d_0d():
    """
    Feature: dtensor + torch.atleast_1d with 0D input
    Description:
        - Convert 0-dimensional scalar tensor to 1-dimensional tensor.
        - Input: shape () fully replicated.
        - Output layout must be 1D fully replicated (size 1).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.tensor(standalone_0d_np).npu()  # shape ()
    standalone_output = torch.atleast_1d(standalone_input)   # shape (1,)

    # Distributed setup: 0D tensor has no explicit sharding axes, use Replicate()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.atleast_1d(dist_input)

    # Layout validation: Output is 1D, and the new dimension must be unsharded
    expected_layout = _build_layout(mesh, x_placements, 1)
    assert dist_output.layout == expected_layout, \
        f"0D atleast_1d output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "0D atleast_1d output mismatch between standalone and distributed execution"


def test_distributed_atleast_1d_1d():
    """
    Feature: dtensor + torch.atleast_1d with 1D input
    Description:
        - Apply atleast_1d to a 1-dimensional tensor.
        - Input: shape (8,) sharded on dim=0.
        - Output layout should strictly preserve original input sharding layout.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_1d_np).npu()  # shape (8,)
    standalone_output = torch.atleast_1d(standalone_input)       # shape (8,)

    # Shard dim=0 ("dp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.atleast_1d(dist_input)

    # Layout validation: layout must remain untouched for 1D input
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)
    assert dist_output.layout == expected_layout, \
        f"1D atleast_1d layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "1D atleast_1d output mismatch"


def test_distributed_atleast_1d_2d():
    """
    Feature: dtensor + torch.atleast_1d with 2D input
    Description:
        - Apply atleast_1d to a 2-dimensional tensor.
        - Input: shape (8, 4) sharded on both dim=0 and dim=1.
        - Output layout should strictly preserve original input mixed sharding layout.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_2d_np).npu()  # shape (8, 4)
    standalone_output = torch.atleast_1d(standalone_input)       # shape (8, 4)

    # Shard dim=0 ("dp") and dim=1 ("tp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.atleast_1d(dist_input)

    # Layout validation: layout must remain untouched for 2D input
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"2D atleast_1d layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "2D atleast_1d output mismatch"


def test_distributed_atleast_1d_multiple_tensors():
    """
    Feature: dtensor + torch.atleast_1d with multiple tensors
    Description:
        - Apply atleast_1d to a sequence of tensors with varying shapes and sharding.
        - Input: One 0D tensor (replicated), One 1D tensor (sharded).
        - Output should return a tuple of DTensors matching the shapes and expected layouts.
    Expectation: Success with correct layout conversions and equivalence per tensor.
    """
    init_dist()

    # Standalone reference
    standalone_in_0 = torch.tensor(standalone_0d_np).npu()       # shape ()
    standalone_in_1 = torch.from_numpy(standalone_1d_np).npu()   # shape (8,)
    standalone_out_0, standalone_out_1 = torch.atleast_1d(standalone_in_0, standalone_in_1)

    # Distributed setup
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    dist_in_0 = distribute_tensor(standalone_in_0, mesh, (Replicate(), Replicate()))
    dist_in_1 = distribute_tensor(standalone_in_1, mesh, (Shard(0), Replicate()))

    # Function call with multiple DTensors
    dist_out_0, dist_out_1 = torch.atleast_1d(dist_in_0, dist_in_1)

    # Layout validation
    expected_layout_0 = _build_layout(mesh, (Replicate(), Replicate()), 1)
    expected_layout_1 = _build_layout(mesh, (Shard(0), Replicate()), 1)

    assert dist_out_0.layout == expected_layout_0, "0D item layout mismatch in multiple inputs test"
    assert dist_out_1.layout == expected_layout_1, "1D item layout mismatch in multiple inputs test"

    # Numerical validation via gathering
    gathered_out_0 = local_to_global(dist_out_0)
    gathered_out_1 = local_to_global(dist_out_1)

    assert torch.equal(standalone_out_0, gathered_out_0), "Output 0 mismatch in multiple inputs test"
    assert torch.equal(standalone_out_1, gathered_out_1), "Output 1 mismatch in multiple inputs test"
