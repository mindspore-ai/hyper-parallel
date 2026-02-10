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
"""test torch dtensor with distributed cumsum"""

import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

# Generate input data using numpy at file header
np.random.seed(42)
standalone_input_np = np.random.randn(8, 16).astype(np.float32)

def test_distributed_cumsum_layout_inference():
    """
    Feature: dtensor + torch.cumsum layout inference
    Description:
        - Test layout inference for cumsum in distributed setting.
        - Ensure that:
            a) Output layout matches input layout when operation dimension is unsharded.
            b) Results are numerically consistent between standalone and distributed execution.
            c) Non-operation dimensions may be arbitrarily sharded.
    Expectation: Success.
    """
    init_dist()
    cumsum_dim = 1  # Perform cumsum along dimension 1 (must be unsharded)

    # Single-device
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.cumsum(standalone_input, dim=cumsum_dim)

    # Distributed setup: shard on dim=0 ("dp"), keep dim=1 unsharded for cumsum
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())  # dim=0 sharded, dim=1 unsharded → VALID for cumsum(dim=1)

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.cumsum(dist_input, dim=cumsum_dim)

    assert dist_output.layout == dist_input.layout, (
        f"Cumsum: output layout {dist_output.layout} mismatch input layout {dist_input.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Cumsum output mismatch between standalone and distributed execution"


def test_distributed_cumsum_sharded_dim_error():
    """
    Feature: dtensor + torch.cumsum error on sharded operation dimension
    Description:
        - Attempt to run cumsum where the operation dimension is sharded.
        - Should raise ValueError during layout inference due to sequential dependency.
    Expectation: Raise ValueError with descriptive message.
    """
    init_dist()
    cumsum_dim = 0  # Attempt cumsum along dimension 0 (which will be sharded)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    try:
        torch.cumsum(dist_input, dim=cumsum_dim)
        assert False, (
            f"Expected ValueError when performing cumsum on sharded dimension {cumsum_dim}"
        )
    except ValueError as e:
        assert f"Cannot perform sharding on normalized dimension {cumsum_dim}" in str(e)


def test_distributed_cumsum_negative_dim_support():
    """
    Feature: dtensor + torch.cumsum with negative dimension indexing
    Description:
        - Test that negative dimension indexing (PyTorch convention) works correctly.
        - cumsum(dim=-1) on a 2D tensor should operate on last dimension (dim=1).
        - Last dimension must be unsharded for correctness.
    Expectation: Success with proper layout validation.
    """
    init_dist()
    cumsum_dim = -1  # Equivalent to dim=1 for 2D tensor

    # Standalone
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_output = torch.cumsum(standalone_input, dim=cumsum_dim)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())  # dim=0 sharded

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.cumsum(dist_input, dim=cumsum_dim)

    assert dist_output.layout == dist_input.layout, "Cumsum with negative dim: layout mismatch"

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Cumsum with negative dim output mismatch"
