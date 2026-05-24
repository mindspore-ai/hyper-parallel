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
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device
from tests.torch.shard.utils import local_to_global


np.random.seed(42)
H, W = 8, 8
standalone_input_np = np.random.randn(H, W).astype(np.float32)
# Ensure mask has random True/False values
standalone_mask_np = (np.random.rand(H, W) > 0.5).astype(bool)
# Source needs enough elements to cover the True values in mask
num_true = standalone_mask_np.sum()
standalone_source_np = np.random.randn(num_true + 10).astype(np.float32)


def test_masked_scatter_basic_replicated() -> None:
    """Test masked_scatter basic replicated.

    Feature: dtensor + torch.Tensor.masked_scatter basic replicated
    Description:
        - Perform masked_scatter on fully replicated tensors.
        - Input: shape (8, 8), Mask: (8, 8), Source: (N,).
        - All inputs must be fully replicated (Unsharded) on the mesh due to op restrictions.
    Expectation: Success with correct values and layout.
    """
    init_backend(_DEVICE_TYPE)

    # Standalone reference computation
    standalone_input = to_device(torch.from_numpy(standalone_input_np), _DEVICE_TYPE)
    standalone_mask = to_device(torch.from_numpy(standalone_mask_np), _DEVICE_TYPE)
    standalone_source = to_device(torch.from_numpy(standalone_source_np), _DEVICE_TYPE)

    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    # Distributed Setup
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))

    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    expected_layout = _build_layout(mesh, placements, 2)
    assert dist_output.layout == expected_layout, (
        f"masked_scatter output layout mismatch: "
        f"expected={expected_layout}, got={dist_output.layout}"
    )

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "masked_scatter output mismatch between standalone and distributed execution"


def test_masked_scatter_1d_replicated() -> None:
    """Test masked_scatter on 1D tensors.

    Feature: dtensor + torch.Tensor.masked_scatter 1D tensor
    Description:
        - Perform masked_scatter on 1D tensors.
        - Input: shape (16,), Mask: (16,), Source: (N,).
    Expectation: Success with correct values.
    """
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(16).astype(np.float32)
    mask_np = (np.random.rand(16) > 0.5).astype(bool)
    source_np = np.random.randn(mask_np.sum() + 5).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_mask = to_device(torch.from_numpy(mask_np), _DEVICE_TYPE)
    standalone_source = to_device(torch.from_numpy(source_np), _DEVICE_TYPE)
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements_1d = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements_1d)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements_1d)
    dist_source = distribute_tensor(standalone_source, mesh, placements_1d)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter 1D output mismatch"


def test_masked_scatter_3d_broadcast() -> None:
    """Test masked_scatter 3D with broadcast.

    Feature: dtensor + torch.Tensor.masked_scatter 3D with broadcast
    Description:
        - Input (2, 4, 4), Mask (4, 4). Mask should broadcast to Input.
        - Requires mask to be Replicated.
    Expectation: Success with broadcasted mask application.
    """
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(2, 4, 4).astype(np.float32)
    mask_np = (np.random.rand(4, 4) > 0.5).astype(bool)

    broadcast_num_true = mask_np.sum() * 2
    source_np = np.random.randn(broadcast_num_true + 10).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_mask = to_device(torch.from_numpy(mask_np), _DEVICE_TYPE)
    standalone_source = to_device(torch.from_numpy(source_np), _DEVICE_TYPE)
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter 3D broadcast output mismatch"


def test_masked_scatter_oversized_source() -> None:
    """Test masked_scatter large source.

    Feature: dtensor + torch.Tensor.masked_scatter large source
    Description:
        - Source tensor is much larger than required number of elements.
        - Verify it correctly picks the first N elements.
    Expectation: Success.
    """
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(4, 4).astype(np.float32)
    mask_np = (np.random.rand(4, 4) > 0.5).astype(bool)
    required = mask_np.sum()
    source_np = np.random.randn(required * 10 + 50).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_mask = to_device(torch.from_numpy(mask_np), _DEVICE_TYPE)
    standalone_source = to_device(torch.from_numpy(source_np), _DEVICE_TYPE)
    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter oversized source output mismatch"


def test_masked_scatter_all_false_mask() -> None:
    """Test masked_scatter all False mask.

    Feature: dtensor + torch.Tensor.masked_scatter all False mask
    Description:
        - Mask is all False. Input should remain unchanged.
    Expectation: Output equals Input.
    """
    init_backend(_DEVICE_TYPE)

    input_np = np.random.randn(8, 8).astype(np.float32)
    mask_np = np.zeros((8, 8), dtype=bool)
    source_np = np.random.randn(10).astype(np.float32)

    standalone_input = to_device(torch.from_numpy(input_np), _DEVICE_TYPE)
    standalone_mask = to_device(torch.from_numpy(mask_np), _DEVICE_TYPE)
    standalone_source = to_device(torch.from_numpy(source_np), _DEVICE_TYPE)

    standalone_output = standalone_input.masked_scatter(standalone_mask, standalone_source)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, placements)
    dist_mask = distribute_tensor(standalone_mask, mesh, placements)
    dist_source = distribute_tensor(standalone_source, mesh, placements)

    dist_output = dist_input.masked_scatter(dist_mask, dist_source)

    gathered_output = local_to_global(dist_output)

    assert torch.equal(standalone_output, gathered_output), \
        "masked_scatter all-false mask mismatch (should be identity)"
