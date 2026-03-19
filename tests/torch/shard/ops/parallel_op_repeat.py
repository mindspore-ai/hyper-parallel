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
"""test torch dtensor with distributed repeat"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 2).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 3, 6).astype(np.float32)
standalone_scalar_like_np = np.array([[3.14]], dtype=np.float32)
standalone_input_4d_np = np.random.randn(2, 3, 4, 5).astype(np.float32)


def test_distributed_repeat_basic_unsharded():
    """
    Feature: dtensor + torch.Tensor.repeat basic repetition
    Description:
        - Repeat an unsharded dimension while preserving sharding on other dimensions (repeat=1).
        - Input: shape (8, 2) sharded on dim=0, repeat dim=1 from 2 -> 6 (repeat times = 3).
        - Output layout should preserve sharding on dim=0, unsharded on repeated dim=1.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference computation
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 2)
    standalone_output = standalone_input.repeat(1, 3)  # shape (8, 6)

    # Distributed setup: shard dim=0 ("dp"), keep dim=1 unsharded (to repeat)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 3)

    # Layout validation: dim0 preserved (sharded), dim1 repeated (unsharded)
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat output layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation via gathering
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat output mismatch between standalone and distributed execution"


def test_distributed_repeat_3d():
    """
    Feature: dtensor + torch.Tensor.repeat with 3D tensor
    Description:
        - Use repeat=1 to preserve sharded dimensions while repeating unsharded dimensions.
        - Input: shape (4, 3, 6) sharded on dim=0 and dim=2, repeat dim=1 by 4 times.
        - Verify preserved dimensions retain original sharding.
    Expectation: Success with correct layout propagation.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()  # shape (4, 3, 6)
    standalone_output = standalone_input.repeat(1, 4, 1)  # shape (4, 12, 6)

    # Shard dim=0 ("dp") and dim=2 ("tp"), keep dim=1 unsharded (to repeat)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 4, 1)  # preserve dim0/dim2 with repeat=1

    # Layout validation: preserved dims keep sharding, repeated dim unsharded
    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"Repeat 3D layout failed: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat 3D output mismatch"



def test_distributed_repeat_scalar_tensor():
    """
    Feature: dtensor + torch.Tensor.repeat scalar expansion
    Description:
        - Repeat scalar-like tensor (1x1) to larger shape.
        - Input: shape (1, 1) fully replicated, repeat to (3, 4, 5).
        - Output must be fully replicated.
    Expectation: Success with fully replicated output layout.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_scalar_like_np).npu()  # shape (1, 1)
    standalone_output = standalone_input.repeat(3, 4, 5)  # shape (3, 4, 5)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(3, 4, 5)

    # Layout validation: all new/repeated dimensions must be unsharded
    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Scalar repeat layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Scalar repeat output mismatch"


def test_distributed_repeat_replicated_dim():
    """
    Feature: dtensor + torch.Tensor.repeat a replicated dimension
    Description:
        - Repeat a dimension that is already replicated across the device mesh.
        - Input: shape (4, 4) with dim=0 sharded, dim=1 replicated. Repeat dim=1 by 2 times.
        - Output: dim=0 remains sharded, dim=1 remains replicated.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(np.random.randn(4, 4).astype(np.float32)).npu()
    standalone_output = standalone_input.repeat(1, 2)  # shape (4, 8)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 2)

    # Layout validation: Shard(0) is preserved, Replicate() remains Replicate()
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat replicated dim layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat replicated dim output mismatch"



def test_distributed_repeat_zero_times():
    """
    Feature: dtensor + torch.Tensor.repeat with zero repetitions
    Description:
        - Repeat a dimension zero times, resulting in that dimension becoming 0.
        - Input: shape (8, 2) sharded on dim=0, replicated on dim=1.
        - Repeat dim=0 by 0 times (invalid for sharded dim), dim=1 by 0 times (valid for replicated dim).
        - Test case designed for replicated dim only for valid scenario.
    Expectation: Success with correct empty tensor shape and layout, no error on replicated dim.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 2)
    standalone_output = standalone_input.repeat(1, 0)  # shape (8, 0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 0) # Repeat replicated dim 1 by 0 times

    # Layout validation: Shard(0) for first dim, Replicate() for second dim
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat zero times layout mismatch: expected {expected_layout}, got {dist_output.layout}"
    assert dist_output.shape == (8, 0), f"Expected shape (8, 0), got {dist_output.shape}"
    assert dist_output.local_shape[1] == 0, f"Expected local shape dim 1 to be 0, got {dist_output.local_shape}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat zero times output mismatch"


def test_distributed_repeat_4d_input():
    """
    Feature: dtensor + torch.Tensor.repeat with 4D input
    Description:
        - Test repeat operation on a 4-dimensional tensor with mixed sharding.
        - Input: shape (2, 3, 4, 5) with dim=0 (dp) and dim=2 (tp) sharded.
        - Repeat dim=1 by 2 times, dim=3 by 3 times.
    Expectation: Sharded dims are preserved, replicated dims are repeated, layout is correct.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_4d_np).npu()  # shape (2, 3, 4, 5)
    standalone_output = standalone_input.repeat(1, 2, 1, 3)  # shape (2, 6, 4, 15)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Shard(1), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 2, 1, 3)

    # Layout validation: Shard(0), Replicate(), Shard(1), Replicate()
    expected_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)
    assert dist_output.layout == expected_layout, \
        f"Repeat 4D input layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat 4D input output mismatch"


def test_distributed_repeat_sharded_dim_repeat_one():
    """
    Feature: dtensor + torch.Tensor.repeat sharded dimension with repeat count of 1
    Description:
        - Attempt to repeat a sharded dimension, but with a repeat count of 1.
        - This operation means no actual repetition occurs, so it should be allowed and preserve sharding.
        - Input: shape (8, 2) sharded on dim=0. Repeat dim=0 by 1, dim=1 by 3.
    Expectation: Success with preserved sharding on dim=0, replicated on dim=1.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()  # shape (8, 2)
    standalone_output = standalone_input.repeat(1, 3)  # shape (8, 6)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(1, 3) # Dim 0 (sharded) is repeated 1 time, Dim 1 (replicated) 3 times

    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Repeat sharded dim repeat one layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat sharded dim repeat one output mismatch"



def test_distributed_repeat_all_dims_replicated():
    """
    Feature: dtensor + torch.Tensor.repeat with all dimensions replicated
    Description:
        - All dimensions of the input tensor are replicated.
        - Repeat all dimensions by varying amounts.
    Expectation: All output dimensions remain replicated, correct shape.
    """
    init_dist()

    standalone_input = torch.from_numpy(np.random.randn(2, 3, 4).astype(np.float32)).npu()
    standalone_output = standalone_input.repeat(2, 3, 1) # shape (4, 9, 4)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Replicate(), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.repeat(2, 3, 1)

    # Layout validation: All dimensions remain replicated
    expected_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"Repeat all dims replicated layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Repeat all dims replicated output mismatch"
