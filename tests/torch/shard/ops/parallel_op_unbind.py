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
"""test torch dtensor with distributed unbind"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
# Data shapes
# Fix: Change input_2d_shape from (3, 8) to (4, 8) to ensure dim0 is divisible
# by the mesh size (2) in test_distributed_unbind_dim1.
input_2d_shape = (4, 8)
input_3d_shape = (2, 4, 8)

input_2d_np = np.random.randn(*input_2d_shape).astype(np.float32)
input_3d_np = np.random.randn(*input_3d_shape).astype(np.float32)


def test_distributed_unbind_dim0():
    """
    Feature: dtensor + torch.unbind on dimension 0
    Description:
        - Unbind an unsharded dimension 0.
        - Input: shape (4, 8), dim0=Replicate, dim1=Shard(1) ("tp").
        - Output: Tuple of 4 tensors, shape (8,), retaining sharding on their dim0 (originally dim1).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(input_2d_np).npu()
    standalone_output = torch.unbind(standalone_input, dim=0)

    # Distributed setup: Shard dim1 ("tp"), keep dim0 unsharded (to unbind)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.unbind(dist_input, dim=0)

    # Validation: Output should be a tuple
    assert isinstance(dist_output, tuple)
    assert len(dist_output) == input_2d_shape[0]

    # Layout validation:
    # Mesh has rank 2, so placements tuple must have length 2.
    # Mesh dim 0 ("dp") is Replicate.
    # Mesh dim 1 ("tp") shards the tensor. The output tensor is 1D (shape (8,)).
    # Originally input dim 1 was sharded. After unbind(dim=0), this becomes the new dim 0.
    # So we use Shard(0).
    expected_layout = _build_layout(mesh, (Replicate(), Shard(0)), 1)

    for i, dist_tensor in enumerate(dist_output):
        assert dist_tensor.layout == expected_layout, \
            f"Unbind output[{i}] layout mismatch: expected {expected_layout}, got {dist_tensor.layout}"

        # Numerical validation
        gathered_tensor = local_to_global(dist_tensor)
        assert torch.equal(standalone_output[i], gathered_tensor), \
            f"Unbind output[{i}] mismatch between standalone and distributed execution"


def test_distributed_unbind_dim1():
    """
    Feature: dtensor + torch.unbind on dimension 1
    Description:
        - Unbind an unsharded dimension 1.
        - Input: shape (4, 8), dim0=Shard(0) ("dp"), dim1=Replicate.
        - Output: Tuple of 8 tensors, shape (4,), retaining sharding on their dim0.
    Expectation: Success with correct layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_2d_np).npu()
    standalone_output = torch.unbind(standalone_input, dim=1)

    # Distributed setup: Shard dim0 ("dp"), keep dim1 unsharded
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.unbind(dist_input, dim=1)

    assert len(dist_output) == input_2d_shape[1]

    # Layout validation:
    # Mesh dim 0 ("dp") shards input dim 0. Output tensor is 1D (shape (4,)).
    # The remaining dimension corresponds to input dim 0, so it remains sharded on dim 0.
    # Mesh dim 1 ("tp") is Replicate.
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)

    for i, dist_tensor in enumerate(dist_output):
        assert dist_tensor.layout == expected_layout, \
            f"Unbind output[{i}] layout mismatch: expected {expected_layout}, got {dist_tensor.layout}"

        gathered_tensor = local_to_global(dist_tensor)
        assert torch.equal(standalone_output[i], gathered_tensor), \
            f"Unbind output[{i}] numerical mismatch"


def test_distributed_unbind_negative_dim():
    """
    Feature: dtensor + torch.unbind with negative dimension
    Description:
        - Unbind on dim=-1 (last dimension).
        - Input: shape (2, 4, 8), dim0/dim1 sharded, dim2 (last) Replicated.
    Expectation: Correctly resolves negative index and returns correct slices.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_3d_np).npu()
    standalone_output = torch.unbind(standalone_input, dim=-1)

    # Distributed setup: Shard dim0 ("dp"), dim1 ("tp"), Replicate dim2
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "mp"))
    x_placements = (Shard(0), Shard(1), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = torch.unbind(dist_input, dim=-1)

    assert len(dist_output) == input_3d_shape[2]

    # Layout validation:
    # Mesh Rank 3.
    # Mesh dim 0 shards Out Dim 0 (Shard(0)).
    # Mesh dim 1 shards Out Dim 1 (Shard(1)).
    # Mesh dim 2 is Replicate.
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 2)

    for i, dist_tensor in enumerate(dist_output):
        assert dist_tensor.layout == expected_layout, \
            f"Unbind output[{i}] layout mismatch"

        gathered_tensor = local_to_global(dist_tensor)
        assert torch.equal(standalone_output[i], gathered_tensor), \
            f"Unbind output[{i}] numerical mismatch"


def test_distributed_unbind_sharded_dim_error():
    """
    Feature: dtensor + torch.unbind error on sharded dimension
    Description:
        - Attempt to unbind a dimension that is sharded.
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_input = torch.from_numpy(input_2d_np).npu()

    # Setup: Shard dim0 ("dp"), Replicate dim1
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    # Attempt to unbind dim0 which is sharded
    try:
        torch.unbind(dist_input, dim=0)
        assert False, "Expected ValueError when unbinding sharded dimension"
    except ValueError as e:
        assert "Unbinding a sharded dimension is not supported" in str(e), \
            f"Unexpected error message: {str(e)}"
