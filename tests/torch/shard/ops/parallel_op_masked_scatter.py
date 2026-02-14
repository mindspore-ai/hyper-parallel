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
"""test torch dtensor with distributed masked_scatter"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
H, W = 8, 8
standalone_input_np = np.random.randn(H, W).astype(np.float32)
# Ensure mask has random True/False values
standalone_mask_np = (np.random.rand(H, W) > 0.5).astype(bool)
# Source needs enough elements to cover the True values in mask
num_true = standalone_mask_np.sum()
standalone_source_np = np.random.randn(num_true + 10).astype(np.float32)


def test_masked_scatter_basic_replicated():
    """
    Feature: dtensor + torch.Tensor.masked_scatter basic replicated
    Description:
        - Perform masked_scatter on fully replicated tensors.
        - Input: shape (8, 8), Mask: (8, 8), Source: (N,).
        - All inputs must be fully replicated (Unsharded) on the mesh due to op restrictions.
    Expectation: Success with correct values and layout.
    """
    init_dist()

    # Standalone reference computation
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_mask = torch.from_numpy(standalone_mask_np).npu()
    standalone_source = torch.from_numpy(standalone_source_np).npu()

    # masked_scatter returns a new tensor (out-of-place for the dist wrapper usually acts like that
    # or creates new DTensor, though torch.masked_scatter_ is in-place.
    # Assuming the dist op follows functional style or handles inplace correctly on new DTensor)
    # The implementation shows it returns a new DTensor layout inference.
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    # Distributed Setup
    # Mesh shape (2, 4)
    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # All inputs must be Replicated on all mesh axes
    # placements tuple length must match mesh ndim
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    # Layout validation: Output should inherit Replicate layout from input
    # Input was 2D, so layout tensor_map should be (-1, -1) effectively
    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, \
        f"masked_scatter output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "masked_scatter output mismatch between standalone and distributed execution"


def test_masked_scatter_input_sharded_error():
    """
    Feature: dtensor + torch.Tensor.masked_scatter error on sharded input
    Description:
        - Attempt masked_scatter with sharded input tensor.
        - Should raise ValueError as implementation requires full replication.
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_mask = torch.from_numpy(standalone_mask_np).npu()
    standalone_source = torch.from_numpy(standalone_source_np).npu()

    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Shard dim 0 of input on mesh axis 0 ("dp")
    input_placements = (Shard(0), Replicate())
    repl_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, input_placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, repl_placements)
    dist_source = distribute_tensor(standalone_source, mesh, repl_placements)

    try:
        dist_input.masked_scatter(dist_mask, dist_source)
        assert False, "Expected ValueError when running masked_scatter with sharded input"
    except ValueError as e:
        assert "is sharded" in str(e), \
            f"Unexpected error message: {str(e)}"


def test_masked_scatter_mask_sharded_error():
    """
    Feature: dtensor + torch.Tensor.masked_scatter error on sharded mask
    Description:
        - Attempt masked_scatter with sharded mask tensor.
        - Should raise ValueError.
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_mask = torch.from_numpy(standalone_mask_np).npu()
    standalone_source = torch.from_numpy(standalone_source_np).npu()

    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    repl_placements = (Replicate(), Replicate())
    # Shard dim 1 of mask on mesh axis 1 ("tp")
    mask_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, repl_placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, mask_placements)
    dist_source = distribute_tensor(standalone_source, mesh, repl_placements)

    try:
        dist_input.masked_scatter(dist_mask, dist_source)
        assert False, "Expected ValueError when running masked_scatter with sharded mask"
    except ValueError as e:
        assert "is sharded" in str(e), f"Unexpected error message: {str(e)}"


def test_masked_scatter_source_sharded_error():
    """
    Feature: dtensor + torch.Tensor.masked_scatter error on sharded source
    Description:
        - Attempt masked_scatter with sharded source tensor.
        - Should raise ValueError.
    Expectation: Raise ValueError.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_mask = torch.from_numpy(standalone_mask_np).npu()
    standalone_source = torch.from_numpy(standalone_source_np).npu()

    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    repl_placements = (Replicate(), Replicate())
    # Shard source (1D tensor) on mesh axis 0 ("dp")
    # This is valid placement syntax, but invalid for this op
    source_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, repl_placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, repl_placements)
    dist_source = distribute_tensor(standalone_source, mesh, source_placements)

    try:
        dist_input.masked_scatter(dist_mask, dist_source)
        assert False, "Expected ValueError when running masked_scatter with sharded source"
    except ValueError as e:
        assert "is sharded" in str(e), f"Unexpected error message: {str(e)}"
def test_masked_scatter_1d_replicated():
    """
    Feature: dtensor + torch.Tensor.masked_scatter 1D tensor
    Description:
        - Perform masked_scatter on 1D tensors.
        - Input: shape (16,), Mask: (16,), Source: (N,).
    Expectation: Success with correct values.
    """
    init_dist()

    # Local setup
    input_np = np.random.randn(16).astype(np.float32)
    mask_np = (np.random.rand(16) > 0.5).astype(bool)
    source_np = np.random.randn(mask_np.sum() + 5).astype(np.float32)

    standalone_input = torch.from_numpy(input_np).npu()
    standalone_mask = torch.from_numpy(mask_np).npu()
    standalone_source = torch.from_numpy(source_np).npu()
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    # Distributed Setup
    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements_1d = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements_1d)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements_1d)
    dist_source = distribute_tensor(standalone_source, mesh, placements_1d)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter 1D output mismatch"


def test_masked_scatter_3d_broadcast():
    """
    Feature: dtensor + torch.Tensor.masked_scatter 3D with broadcast
    Description:
        - Input (2, 4, 4), Mask (4, 4). Mask should broadcast to Input.
        - Requires mask to be Replicated.
    Expectation: Success with broadcasted mask application.
    """
    init_dist()

    # Shape setup: Mask (4,4) broadcasts to Input (2,4,4)
    input_np = np.random.randn(2, 4, 4).astype(np.float32)
    mask_np = (np.random.rand(4, 4) > 0.5).astype(bool)
    # Note: When mask broadcasts, the number of True values multiplies by the batch size
    # PyTorch logic: mask expands to (2, 4, 4), so we need count(True) * 2 source elements.

    broadcast_num_true = mask_np.sum() * 2
    source_np = np.random.randn(broadcast_num_true + 10).astype(np.float32)

    standalone_input = torch.from_numpy(input_np).npu()
    standalone_mask = torch.from_numpy(mask_np).npu()
    standalone_source = torch.from_numpy(source_np).npu()
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter 3D broadcast output mismatch"



def test_masked_scatter_oversized_source():
    """
    Feature: dtensor + torch.Tensor.masked_scatter large source
    Description:
        - Source tensor is much larger than required number of elements.
        - Verify it correctly picks the first N elements.
    Expectation: Success.
    """
    init_dist()

    input_np = np.random.randn(4, 4).astype(np.float32)
    mask_np = (np.random.rand(4, 4) > 0.5).astype(bool)
    required = mask_np.sum()
    # Source is 10x larger than needed
    source_np = np.random.randn(required * 10 + 50).astype(np.float32)

    standalone_input = torch.from_numpy(input_np).npu()
    standalone_mask = torch.from_numpy(mask_np).npu()
    standalone_source = torch.from_numpy(source_np).npu()
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter oversized source output mismatch"


def test_masked_scatter_all_false_mask():
    """
    Feature: dtensor + torch.Tensor.masked_scatter all False mask
    Description:
        - Mask is all False. Input should remain unchanged.
    Expectation: Output equals Input.
    """
    init_dist()

    input_np = np.random.randn(8, 8).astype(np.float32)
    mask_np = np.zeros((8, 8), dtype=bool) # All False
    source_np = np.random.randn(10).astype(np.float32) # Should not be used

    standalone_input = torch.from_numpy(input_np).npu()
    standalone_mask = torch.from_numpy(mask_np).npu()
    standalone_source = torch.from_numpy(source_np).npu()

    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh("npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)

    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter all-false mask mismatch (should be identity)"
