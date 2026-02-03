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
"""test torch dtensor with distributed expand"""

import numpy as np
import torch
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 1).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 1, 6).astype(np.float32)
standalone_scalar_like_np = np.array([[3.14]], dtype=np.float32)


def test_distributed_expand_basic_unsharded():
    """
    Feature: dtensor + torch.Tensor.expand basic expansion
    Description:
        - Expand an unsharded singleton dimension while preserving sharding on other dimensions.
        - Input: shape (8, 1) sharded on dim=0, expand dim=1 from 1→16.
        - Output layout should preserve sharding on dim=0, unsharded on expanded dim=1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 1)
    standalone_output = standalone_input.expand(-1, 16)  # shape (8, 16)

    # Distributed setup: shard dim=0 ("dp"), keep dim=1 unsharded (singleton to expand)
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(-1, 16)

    # Layout validation: dim0 preserved (sharded), dim1 expanded (unsharded)
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Expand output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Expand output mismatch between standalone and distributed execution"


def test_distributed_expand_3d():
    """
    Feature: dtensor + torch.Tensor.expand with -1 preservation
    Description:
        - Use -1 to preserve sharded dimensions while expanding unsharded singleton dimension.
        - Input: shape (4, 1, 6) sharded on dim=0 and dim=2, expand dim=1 from 1→10.
        - Verify preserved dimensions retain original sharding.
    Expectation: Success with correct layout propagation.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()  # shape (4, 1, 6)
    standalone_output = standalone_input.expand(4, 10, 6)  # shape (4, 10, 6)

    # Shard dim=0 ("dp") and dim=2 ("tp"), keep dim=1 unsharded (to expand)
    mesh = init_device_mesh(mesh_shape = (4, 2), alias_name = ("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(-1, 10, -1)  # preserve dim0/dim2 with -1

    # Layout validation: preserved dims keep sharding, expanded dim unsharded
    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"Expand with -1 preservation failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Expand with -1 preservation output mismatch"


def test_distributed_expand_prepend_new_dimensions():
    """
    Feature: dtensor + torch.Tensor.expand prepending dimensions
    Description:
        - Prepend new dimensions to tensor (must be unsharded).
        - Input: shape (8, 1) sharded on dim=0, prepend dims (2, 3) → output shape (2, 3, 8, 16).
        - Verify new dimensions are unsharded, existing dimensions preserve sharding.
    Expectation: Success with correct layout for prepended dimensions.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 1)
    standalone_output = standalone_input.expand(2, 3, 8, 16)  # shape (2, 3, 8, 16)

    # Shard only dim=0 ("dp") in input
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Shard(0), Replicate()) # tensor_map = (1, -1)

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(2, 3, -1, 16)  # prepend (2,3), preserve dim0, expand dim1

    # Layout validation: new dims unsharded, dim2 preserved (sharded), dim3 expanded (unsharded)
    expected_layout = _build_layout(mesh, (Shard(2), Replicate()), 4)
    assert dist_output.layout == expected_layout, \
        f"Prepend expand layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Prepend expand output mismatch"


def test_distributed_expand_sharded_dim_error():
    """
    Feature: dtensor + torch.Tensor.expand error on sharded dimension
    Description:
        - Attempt to expand a sharded singleton dimension (semantically invalid).
        - Input: shape (8, 1) sharded on dim=1 (which has global size = 1 * num_shards = 4).
        - Should fail at layout inference since global size ≠ 1.
    Expectation: Raise ValueError with descriptive message.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 1)

    # INVALID: shard dim=1 (the singleton dimension we want to expand)
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Replicate(), Shard(1)) # tensor_map = (-1, 0)

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    try:
        # Attempt to expand sharded dim=1 from 1→16 (should fail)
        dist_input.expand(8, 16)
        assert False, "Expected ValueError when expanding sharded dimension"
    except ValueError as e:
        assert "Cannot expand dimension 1 which is sharded" in str(e), \
            f"Unexpected error message: {str(e)}"


def test_distributed_expand_scalar_tensor():
    """
    Feature: dtensor + torch.Tensor.expand scalar expansion
    Description:
        - Expand scalar-like tensor (1x1) to larger shape.
        - Input: shape (1, 1) fully replicated, expand to (3, 4, 5).
        - Output must be fully replicated (no sharding possible for new dimensions).
    Expectation: Success with fully replicated output layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_scalar_like_np).npu()  # shape (1, 1)
    standalone_output = standalone_input.expand(3, 4, 5)  # shape (3, 4, 5)

    # Scalar-like input must be fully replicated (cannot shard singleton dimensions meaningfully)
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.expand(3, 4, 5)

    # Layout validation: all new dimensions must be unsharded (fully replicated)
    expected_layout = _build_layout(mesh, (Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Scalar expand layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Scalar expand output mismatch"


def test_distributed_expand_as_basic():
    """
    Feature: dtensor + torch.Tensor.expand_as basic expansion
    Description:
        - Expand input tensor to match target tensor shape using expand_as.
        - Input: shape (8, 1) sharded on dim=0, target shape (8, 16).
        - Output layout should preserve sharding on dim=0, unsharded on expanded dim=1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference computation
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 1)
    target_tensor = torch.empty(8, 16, device='npu')  # Only shape matters for expand_as
    standalone_output = standalone_input.expand_as(target_tensor)  # shape (8, 16)

    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    # Target tensor MUST be fully replicated (shape metadata only; no computation on values)
    target_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    # Layout validation: dim0 preserved (sharded), dim1 expanded (unsharded)
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"expand_as output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "expand_as output mismatch between standalone and distributed execution"


def test_distributed_expand_as_3d_preservation():
    """
    Feature: dtensor + torch.Tensor.expand_as with dimension preservation
    Description:
        - Use expand_as to expand middle dimension while preserving sharding on other dimensions.
        - Input: shape (4, 1, 6) sharded on dim=0 ("dp") and dim=2 ("tp"), target shape (4, 10, 6).
        - Verify preserved dimensions retain original sharding pattern.
    Expectation: Success with correct layout propagation.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()  # shape (4, 1, 6)
    target_tensor = torch.empty(4, 10, 6, device='npu')
    standalone_output = standalone_input.expand_as(target_tensor)  # shape (4, 10, 6)

    # Shard dim=0 ("dp") and dim=2 ("tp"), keep dim=1 unsharded (to expand)
    mesh = init_device_mesh(mesh_shape = (4, 2), alias_name = ("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))
    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    # Target tensor fully replicated (shape metadata only)
    target_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    # Layout validation: preserved dims keep sharding, expanded dim unsharded
    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"expand_as with preservation failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "expand_as with dimension preservation output mismatch"


def test_distributed_expand_as_prepend_dimensions():
    """
    Feature: dtensor + torch.Tensor.expand_as prepending dimensions
    Description:
        - Use expand_as to prepend new dimensions to tensor (must be unsharded).
        - Input: shape (8, 1) sharded on dim=0, target shape (2, 3, 8, 16).
        - Verify new dimensions are unsharded, existing dimensions preserve sharding.
    Expectation: Success with correct layout for prepended dimensions.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 1)
    target_tensor = torch.empty(2, 3, 8, 16, device='npu')
    standalone_output = standalone_input.expand_as(target_tensor)  # shape (2, 3, 8, 16)

    # Shard only dim=0 ("dp") in input
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    # Target tensor fully replicated (shape metadata only)
    target_layout = _build_layout(mesh, (Shard(2), Replicate(), Shard(1)), 4)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    # Layout validation: new dims unsharded, dim2 preserved (sharded), dim3 expanded (unsharded)
    expected_layout = _build_layout(mesh, (Shard(2), Replicate(), Shard(1)), 4)
    assert dist_output.layout == expected_layout, \
        f"expand_as prepend layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "expand_as prepend dimensions output mismatch"


def test_distributed_expand_as_scalar_to_tensor():
    """
    Feature: dtensor + torch.Tensor.expand_as scalar expansion
    Description:
        - Expand scalar-like tensor (1x1) to match arbitrary target tensor shape.
        - Input: shape (1, 1) fully replicated, target shape (3, 4, 5).
        - Output must be fully replicated (no sharding possible for expanded dimensions).
    Expectation: Success with fully replicated output layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_scalar_like_np).npu()  # shape (1, 1)
    target_tensor = torch.empty(3, 4, 5, device='npu')
    standalone_output = standalone_input.expand_as(target_tensor)  # shape (3, 4, 5)

    # Scalar-like input must be fully replicated
    mesh = init_device_mesh(mesh_shape = (2, 4), alias_name = ("dp", "tp"))
    x_placements = (Replicate(), Replicate())
    dist_input = DTensor.distribute_tensor(standalone_input, mesh, x_placements)

    # Target tensor fully replicated (shape metadata only)
    target_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    dist_target = global_to_local(target_tensor, target_layout)

    dist_output = dist_input.expand_as(dist_target)

    # Layout validation: all dimensions must be unsharded (fully replicated)
    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Scalar expand_as layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Scalar expand_as output mismatch"
