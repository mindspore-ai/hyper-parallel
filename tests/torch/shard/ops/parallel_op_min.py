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
"""test torch dtensor with distributed min"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 8).astype(np.float32)
standalone_input_3d_np = np.random.randn(4, 4, 4).astype(np.float32)


def test_distributed_min_element_wise():
    """
    Feature: dtensor + torch.min element-wise
    Description:
        - Compute element-wise min between two tensors.
        - Input: Two (8, 8) tensors, sharded on dim 0.
        - Output layout should preserve sharding on dim 0.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    # Standalone reference
    input_b_np = np.random.randn(8, 8).astype(np.float32)
    standalone_a = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_b = torch.from_numpy(input_b_np).npu()
    standalone_output = torch.min(standalone_a, standalone_b)

    # Distributed setup: Shard dim=0 ("dp"), Replicate dim=1
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.min(dist_a, dist_b)

    # Layout validation: Should remain Shard(0), Replicate
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    assert dist_output.layout == expected_layout, \
        f"Element-wise min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(
        standalone_output, gathered_output
    ), "Element-wise min output mismatch"


def test_distributed_min_dim_reduce_sharded():
    """
    Feature: dtensor + torch.min reduction on sharded dimension
    Description:
        - Reduce a dimension that is physically sharded.
        - Input: (8, 8) sharded on dim 0. Min reduction on dim 0.
        - Output (values) should be Partial on the reduced mesh dimension.
    Expectation: Layout becomes Partial, values match after resolution.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    # torch.min(dim) returns (values, indices)
    standalone_vals, _ = torch.min(standalone_input, dim=0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim 0 ("dp"), Replicate dim 1
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    # Reduce dim 0 -> This is the sharded dimension ("dp")
    # Output will be a tuple (values, indices). We verify 'values'.
    dist_vals, _ = torch.min(dist_input, dim=0)

    # Layout validation:
    # Dim 0 (was "dp") is reduced -> Becomes Partial(reduce_op="min")
    # Dim 1 (was Replicate) -> Remains Replicate
    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Replicate()), 1)

    assert dist_vals.layout == expected_layout, \
        f"Min reduce sharded layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    # Numerical validation
    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()

    assert torch.allclose(standalone_vals, gathered_vals), \
        f"Min reduce sharded values mismatch: expected {standalone_vals}, got {gathered_vals}"


def test_distributed_min_dim_reduce_replicated():
    """
    Feature: dtensor + torch.min reduction on replicated dimension
    Description:
        - Reduce a dimension that is physically replicated.
        - Input: (8, 8) sharded on dim 0. Min reduction on dim 1 (Replicated).
        - Output should NOT be Partial, as we reduced a local dimension.
    Expectation: Layout preserves sharding of remaining dims.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_vals, _ = torch.min(standalone_input, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim 0 ("dp"), Replicate dim 1
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    # Reduce dim 1 -> This is Replicated. No communication needed.
    dist_vals, _ = torch.min(dist_input, dim=1)

    # Layout validation:
    # Dim 0 ("dp") preserved -> Shard(0)
    # Dim 1 reduced -> Removed from map.
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 1)

    assert dist_vals.layout == expected_layout, \
        f"Min reduce replicated layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    gathered_vals = local_to_global(dist_vals)
    assert torch.equal(
        standalone_vals, gathered_vals
    ), "Min reduce replicated dimension output mismatch"


def test_distributed_min_global_reduce():
    """
    Feature: dtensor + torch.min global reduction
    Description:
        - Compute global min of a fully sharded tensor.
        - Input: (8, 8) sharded on both dims.
        - Output should be a scalar with Partial status on all sharded axes.
    Expectation: Partial layout on all axes, correct scalar value.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = torch.min(standalone_input)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard both dims
    x_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_output = torch.min(dist_input)

    # Layout validation:
    # Output is scalar (0-D).
    # Both mesh axes contributed to data -> Both become Partial(min).
    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Partial(reduce_op="min")), 0)

    assert dist_output.layout == expected_layout, \
        f"Global min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    replicated_output = dist_output.redistribute(mesh, (Replicate(), Replicate()))
    gathered_output = replicated_output.to_local()

    assert torch.allclose(standalone_output, gathered_output), \
        f"Global min value mismatch: expected {standalone_output}, got {gathered_output}"


def test_distributed_min_keepdim():
    """
    Feature: dtensor + torch.min with keepdim=True
    Description:
        - Reduce sharded dimension but keep shape.
        - Input: (8, 8) sharded on dim 0. Reduce dim 0.
        - Output should be Partial on dim 0 axis, but rank preserved.
    Expectation: Correct layout and values.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_vals, _ = torch.min(standalone_input, dim=0, keepdim=True)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=0, keepdim=True)

    # Layout validation:
    # Rank preserved (2D).
    # Mesh axis 0 ("dp") reduced -> Partial.
    # Mesh axis 1 ("tp") Replicate -> Replicate.
    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Replicate()), 2)

    assert dist_vals.layout == expected_layout, \
        f"Min keepdim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    # 1. Redistribute to Replicate (AllReduce)
    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))

    # 2. Get local tensor (now it contains the global min)
    gathered_vals = replicated_vals.to_local()

    # Value validation
    assert torch.allclose(standalone_vals, gathered_vals), \
        f"Min values mismatch: expected {standalone_vals}, got {gathered_vals}"

def test_distributed_min_3d_element_wise():
    """
    Feature: dtensor + torch.min element-wise on 3D tensor
    Description:
        - Compute element-wise min between two 3D tensors.
        - Input: Two (4, 4, 4) tensors, sharded on dim 0 and dim 1.
        - Output layout should preserve the 3D sharding.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    input_b_np = np.random.randn(4, 4, 4).astype(np.float32)
    standalone_a = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_b = torch.from_numpy(input_b_np).npu()
    standalone_output = torch.min(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim 0 ("dp"), Shard dim 1 ("tp"), Replicate dim 2 implicitly
    x_placements = (Shard(0), Shard(1))

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.min(dist_a, dist_b)

    # Layout validation
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 3)
    assert dist_output.layout == expected_layout, \
        f"3D element-wise min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical validation
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "3D element-wise min output mismatch"


def test_distributed_min_3d_reduce_negative_dim():
    """
    Feature: dtensor + torch.min reduction on negative dimension
    Description:
        - Reduce the last dimension (-1) which is physically replicated.
        - Input: (4, 4, 4) tensor sharded on dim 0 and dim 1.
    Expectation: Layout rank is reduced, sharding on dim 0 and 1 preserved.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_vals, _ = torch.min(standalone_input, dim=-1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=-1)

    # Layout validation: Output rank becomes 2. Dim 0 and 1 preserve their Shards.
    expected_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)
    assert dist_vals.layout == expected_layout, \
        f"Min reduce negative dim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    gathered_vals = local_to_global(dist_vals)
    assert torch.allclose(standalone_vals, gathered_vals), "Min reduce negative dim values mismatch"


def test_distributed_min_3d_reduce_sharded_dim():
    """
    Feature: dtensor + torch.min reduction on 3D sharded dimension
    Description:
        - Reduce dimension 1 which is physically sharded by "tp".
        - Output values should be Partial on the "tp" mesh dimension.
    Expectation: Correct layout and values.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_vals, _ = torch.min(standalone_input, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=1)

    # Layout validation: Rank becomes 2. "dp" shards dim 0. "tp" becomes Partial.
    expected_layout = _build_layout(mesh, (Shard(0), Partial(reduce_op="min")), 2)
    assert dist_vals.layout == expected_layout, \
        f"3D Min reduce sharded layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()
    assert torch.allclose(standalone_vals, gathered_vals), "3D Min reduce sharded values mismatch"


def test_distributed_min_1d_mesh_global_reduce():
    """
    Feature: dtensor + torch.min global reduction on 1D mesh
    Description:
        - Compute global min on a 1D DeviceMesh.
    Expectation: Partial layout on the single axis, correct scalar value.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = torch.min(standalone_input)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    x_placements = (Shard(0),)
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_output = torch.min(dist_input)

    # Layout validation: 0D output. Single mesh dim becomes Partial.
    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"),), 0)
    assert dist_output.layout == expected_layout, \
        f"1D mesh global min layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    replicated_output = dist_output.redistribute(mesh, (Replicate(),))
    gathered_output = replicated_output.to_local()
    assert torch.allclose(standalone_output, gathered_output), "1D mesh global min value mismatch"


def test_distributed_min_1d_mesh_element_wise():
    """
    Feature: dtensor + torch.min element-wise on 1D mesh
    Description:
        - Compute element-wise min between two tensors on a 1D DeviceMesh.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    input_b_np = np.random.randn(8, 8).astype(np.float32)
    standalone_a = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_b = torch.from_numpy(input_b_np).npu()
    standalone_output = torch.min(standalone_a, standalone_b)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    x_placements = (Shard(0),)

    dist_a = distribute_tensor(standalone_a, mesh, x_placements)
    dist_b = distribute_tensor(standalone_b, mesh, x_placements)

    dist_output = torch.min(dist_a, dist_b)

    expected_layout = _build_layout(mesh, (Shard(0),), 2)
    assert dist_output.layout == expected_layout, \
        f"1D mesh element-wise layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output), "1D mesh element-wise mismatch"


def test_distributed_min_keepdim_negative_dim():
    """
    Feature: dtensor + torch.min with keepdim=True and negative dim
    Description:
        - Reduce a negative dimension (-2) and keep shape.
    Expectation: Correct Partial layout on dim -2 axis, rank preserved.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_vals, _ = torch.min(standalone_input, dim=-2, keepdim=True)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=-2, keepdim=True)

    # Layout validation: Rank 2 preserved. Dim 0 ("dp") reduced -> Partial.
    expected_layout = _build_layout(mesh, (Partial(reduce_op="min"), Replicate()), 2)
    assert dist_vals.layout == expected_layout, \
        f"Min keepdim negative dim layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()
    assert torch.allclose(standalone_vals, gathered_vals), "Min keepdim negative dim values mismatch"


def test_distributed_min_4_cards():
    """
    Feature: dtensor + torch.min using 4 cards
    Description:
        - Reduce on a 2x2 mesh explicitly forcing a 4-card topology.
    Expectation: Successful dimension reduction and global value match.
    """
    init_dist()

    input_np = np.random.randn(4, 4).astype(np.float32)
    standalone_input = torch.from_numpy(input_np).npu()
    standalone_vals, _ = torch.min(standalone_input, dim=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)

    dist_vals, _ = torch.min(dist_input, dim=1)

    # Layout validation: dim 1 reduced. tp becomes Partial. dp stays Shard(0).
    expected_layout = _build_layout(mesh, (Shard(0), Partial(reduce_op="min")), 1)
    assert dist_vals.layout == expected_layout, \
        f"4-cards Min reduce layout mismatch: expected {expected_layout}, got {dist_vals.layout}"

    replicated_vals = dist_vals.redistribute(mesh, (Replicate(), Replicate()))
    gathered_vals = replicated_vals.to_local()
    assert torch.allclose(standalone_vals, gathered_vals), "4-cards Min reduce values mismatch"
