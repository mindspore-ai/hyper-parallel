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
"""test torch dtensor with distributed outer"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
vec1_np = np.random.randn(8).astype(np.float32)
vec2_np = np.random.randn(16).astype(np.float32)
invalid_2d_np = np.random.randn(8, 2).astype(np.float32)


def test_distributed_outer_both_replicated():
    """
    Feature: dtensor + torch.outer basic computation
    Description:
        - Compute outer product of two fully replicated 1-D tensors.
        - Input 1: shape (8) fully replicated.
        - Input 2: shape (16) fully replicated.
        - Output layout should be fully replicated, shape (8, 16).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_vec1 = torch.from_numpy(vec1_np).npu()
    standalone_vec2 = torch.from_numpy(vec2_np).npu()
    standalone_output = torch.outer(standalone_vec1, standalone_vec2)

    # Distributed setup
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, placements)

    dist_output = torch.outer(dist_vec1, dist_vec2)

    # Layout validation
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Outer output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Outer output mismatch between standalone and distributed execution"


def test_distributed_outer_both_sharded():
    """
    Feature: dtensor + torch.outer with orthogonal sharding
    Description:
        - Compute outer product where inputs are sharded on different mesh dimensions.
        - Input 1: shape (8) sharded on dim 0 ("dp").
        - Input 2: shape (16) sharded on dim 1 ("tp").
        - Output layout should be sharded on both dimensions: (Shard(0), Shard(1)).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_vec1 = torch.from_numpy(vec1_np).npu()
    standalone_vec2 = torch.from_numpy(vec2_np).npu()
    standalone_output = torch.outer(standalone_vec1, standalone_vec2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # vec1 sharded on "dp" (mesh dim 0), replicates on "tp"
    v1_placements = (Shard(0), Replicate())
    # vec2 replicates on "dp", sharded on "tp" (mesh dim 1)
    v2_placements = (Replicate(), Shard(0))

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, v1_placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, v2_placements)

    dist_output = torch.outer(dist_vec1, dist_vec2)

    # Output should inherit "dp" for its dim 0, and "tp" for its dim 1
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_output.layout == expected_layout, \
        f"Outer orthogonal sharding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Outer orthogonal sharding output mismatch"


def test_distributed_outer_first_sharded():
    """
    Feature: dtensor + torch.outer with partial sharding
    Description:
        - Compute outer product where only the first input is sharded.
        - Input 1: shape (8) sharded on dim 0 ("dp").
        - Input 2: shape (16) fully replicated.
        - Output layout should be sharded on "dp" for dim 0, replicated for dim 1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_vec1 = torch.from_numpy(vec1_np).npu()
    standalone_vec2 = torch.from_numpy(vec2_np).npu()
    standalone_output = torch.outer(standalone_vec1, standalone_vec2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    v1_placements = (Shard(0), Replicate())
    v2_placements = (Replicate(), Replicate())

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, v1_placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, v2_placements)

    dist_output = torch.outer(dist_vec1, dist_vec2)

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Outer partial sharding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Outer partial sharding output mismatch"


def test_distributed_outer_same_dim_error():
    """
    Feature: dtensor + torch.outer constraint validation
    Description:
        - Attempt to compute outer product when both inputs are sharded on the same mesh dimension.
        - This is invalid as it violates the orthogonal placement of the resulting 2-D tensor.
    Expectation: Raise ValueError with descriptive message.
    """
    init_dist()

    standalone_vec1 = torch.from_numpy(vec1_np).npu()
    standalone_vec2 = torch.from_numpy(vec2_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Both sharded on "dp" (mesh dim 0)
    placements = (Shard(0), Replicate())

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, placements)
    dist_vec2 = distribute_tensor(standalone_vec2, mesh, placements)

    try:
        torch.outer(dist_vec1, dist_vec2)
        assert False, "Expected ValueError when inputs share a sharded mesh dimension"
    except ValueError as e:
        assert "cannot be sharded on the same device mesh dimension" in str(e), \
            f"Unexpected error message: {str(e)}"


def test_distributed_outer_not_1d_error():
    """
    Feature: dtensor + torch.outer dimensionality constraint
    Description:
        - Attempt to compute outer product with a non-1D tensor.
        - PyTorch semantics strictly require 1-D tensors for torch.outer.
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_vec1 = torch.from_numpy(vec1_np).npu()
    standalone_err = torch.from_numpy(invalid_2d_np).npu() # shape (8, 2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    dist_vec1 = distribute_tensor(standalone_vec1, mesh, (Replicate(), Replicate()))
    dist_err = distribute_tensor(standalone_err, mesh, (Replicate(), Replicate()))

    try:
        torch.outer(dist_vec1, dist_err)
        assert False, "Expected ValueError when input is not 1-D"
    except ValueError as e:
        assert "requires exactly 1-D tensors as inputs" in str(e), \
            f"Unexpected error message: {str(e)}"
