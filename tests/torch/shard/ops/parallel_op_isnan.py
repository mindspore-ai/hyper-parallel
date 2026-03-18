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
"""test torch dtensor with distributed isnan"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Generate mock data with NaN values
np.random.seed(42)

# 2D mock data
input_2d_np = np.random.randn(8, 8).astype(np.float32)
input_2d_np[1, 2] = np.nan
input_2d_np[5, 4] = np.nan
input_2d_np[7, 0] = np.nan

# 3D mock data
input_3d_np = np.random.randn(4, 8, 6).astype(np.float32)
input_3d_np[0, 1, 1] = np.nan
input_3d_np[3, 5, 2] = np.nan


def test_distributed_isnan_basic():
    """
    Feature: dtensor + torch.isnan basic element-wise check
    Description:
        - Check isnan on a 2D tensor sharded on multiple dimensions.
        - Input: shape (8, 8) sharded on dim=0 and dim=1.
        - Output layout should strictly preserve the input layout.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(input_2d_np).npu()  # shape (8, 8)
    standalone_output = torch.isnan(standalone_input)       # boolean tensor

    # Distributed setup: shard dim=0 ("dp") and dim=1 ("tp")
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.isnan(dist_input)

    # Layout validation: output layout must be identical to input layout for element-wise ops
    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"isnan output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering local shards to a global tensor
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "isnan output mismatch between standalone and distributed execution"


def test_distributed_isnan_replicate():
    """
    Feature: dtensor + torch.isnan on replicated tensor
    Description:
        - Check isnan on a fully replicated 2D tensor.
        - Output should also be fully replicated.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_2d_np).npu()
    standalone_output = torch.isnan(standalone_input)

    # Fully replicated placement
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.isnan(dist_input)

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 2)
    assert dist_output.layout == expected_layout, \
        f"isnan replicate layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "isnan replicate output mismatch"


def test_distributed_isnan_3d():
    """
    Feature: dtensor + torch.isnan on 3D tensor
    Description:
        - Check isnan on a 3D tensor with a mix of sharded and replicated dimensions.
        - Input: shape (4, 8, 6), shard dim=0, shard dim=1, replicate dim=2 implicitly.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_3d_np).npu()  # shape (4, 8, 6)
    standalone_output = torch.isnan(standalone_input)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    # Alternatively, you can test tensor method call
    dist_output = dist_input.isnan()

    # Layout validation
    expected_layout = _build_layout(mesh, x_placements, 3)
    assert dist_output.layout == expected_layout, \
        f"isnan 3D layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "isnan 3D output mismatch"
