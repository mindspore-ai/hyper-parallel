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
"""test torch dtensor with distributed nonzero"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
# Create a 2D array with specific zero and non-zero elements
standalone_input_2d_np = np.array([
    [1.0, 0.0, 2.0],
    [0.0, 3.0, 0.0],
    [4.0, 5.0, 0.0]
], dtype=np.float32)

# Create a 3D array and randomly set some elements to zero
standalone_input_3d_np = np.random.randn(4, 2, 6).astype(np.float32)
standalone_input_3d_np[standalone_input_3d_np < 0] = 0.0


def test_distributed_nonzero_basic():
    """
    Feature: dtensor + torch.Tensor.nonzero basic execution
    Description:
        - Test nonzero() with default parameters (as_tuple=False).
        - Input MUST be fully replicated due to data-dependent dynamic shapes.
        - Output is a 2D tensor of shape (z, n) where z is the number of
          non-zero elements and n is the input dimension.
    Expectation: Success with fully replicated layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = standalone_input.nonzero()

    # Distributed setup: MUST use Replicate for all dimensions
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.nonzero()

    # Layout validation: Output is a 2D tensor, must be fully replicated
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Nonzero output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Nonzero output mismatch between standalone and distributed execution"


def test_distributed_nonzero_as_tuple():
    """
    Feature: dtensor + torch.Tensor.nonzero with as_tuple=True
    Description:
        - Test nonzero(as_tuple=True).
        - Input MUST be fully replicated.
        - Output is a tuple of 1D tensors, one for each dimension of the input.
    Expectation: Success with correct tuple layouts and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_output = standalone_input.nonzero(as_tuple=True)

    # Distributed setup: MUST use Replicate for all dimensions
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.nonzero(as_tuple=True)

    # Layout validation: Output is a tuple of 1D tensors, all fully replicated
    assert isinstance(dist_output, tuple), "Nonzero with as_tuple=True should return a tuple"
    assert len(dist_output) == standalone_input.ndim, "Returned tuple length should match input ndim"

    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
    for i, out_tensor in enumerate(dist_output):
        assert out_tensor.layout == expected_layout, \
            f"Nonzero tuple output layout mismatch at index {i}: expected {expected_layout}, got {out_tensor.layout}"

    # Numerical validation via gathering each tuple element using enumerate and zip
    for i, (expected_tensor, local_tensor) in enumerate(zip(standalone_output, dist_output)):
        gathered_output = local_to_global(local_tensor)
        assert torch.equal(
            expected_tensor, gathered_output
        ), f"Nonzero tuple output mismatch at index {i}"

def test_distributed_nonzero_sharded_error():
    """
    Feature: dtensor + torch.Tensor.nonzero error on sharded input
    Description:
        - Attempt to run nonzero() on a tensor that is sharded.
        - Because nonzero() produces data-dependent dynamic shapes, a sharded
          input will cause shape mismatches across ranks, leading to collective
          communication hangs or crashes.
    Expectation: Raise ValueError with a descriptive message.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()

    # INVALID: shard dim=0
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    try:
        # Attempt to call nonzero on a sharded tensor (should fail)
        dist_input.nonzero()
        assert False, "Expected ValueError when running nonzero on a sharded tensor"
    except ValueError as e:
        assert "fully replicated" in str(e) or "unsharded" in str(e), \
            f"Unexpected error message: {str(e)}"
