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
"""test torch dtensor with distributed flatten"""

import numpy as np
import torch
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
standalone_input_2d_np = np.random.randn(8, 4).astype(np.float32)
standalone_input_3d_np = np.random.randn(8, 4, 6).astype(np.float32)
standalone_input_4d_np = np.random.randn(4, 2, 4, 6).astype(np.float32)
standalone_scalar_np = np.array(3.14, dtype=np.float32)


def test_distributed_flatten_all_dims():
    """
    Feature: dtensor + torch.Tensor.flatten all dimensions
    Description:
        - Flatten a 3D tensor across all dimensions.
        - Input: shape (8, 4, 6) sharded ONLY on dim=0 ("dp").
        - Output layout should map the new 1D tensor to "dp".
    Expectation:
        - The distributed flatten operation completes successfully.
        - The output layout correctly reflects the new 1D tensor sharding ("dp").
        - The gathered distributed output matches the standalone flatten output numerically.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_output = standalone_input.flatten(0, -1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # FIX: Only shard dim=0. If we sharded multiple dimensions being flattened,
    # it would hit the framework limitation.
    x_placements = (Shard(0), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(0, -1)

    # Layout validation: The new single dimension should inherit "dp"
    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map, \
        f"Flatten output mismatch: expected {expected_layout.tensor_map}, got {dist_output.layout.tensor_map}"

    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)



def test_distributed_flatten_middle_dims():
    """
    Feature: dtensor + torch.Tensor.flatten on middle dimensions with partial sharding
    Description:
        - Flatten dimensions 1 and 2 of a 4D distributed tensor.
        - Input tensor has shape (4, 2, 4, 6), sharded on dim0 ("dp") and dim1 ("tp").
        - Only one of the flattened dimensions (dim1) is sharded.
    Expectation:
        - The distributed flatten operation completes successfully.
        - The output layout correctly reflects the new tensor shape and sharding (e.g., "dp", "tp", "None").
        - The gathered distributed output matches the standalone flatten output numerically.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_4d_np).npu()
    standalone_output = standalone_input.flatten(1, 2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # Sharding dim0 and dim1. Flattening dim1 and dim2.
    # Only dim1 is sharded among the flattened dimensions, so it's valid!
    x_placements = (Shard(0), Shard(1), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(1, 2)

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp", "tp", "None")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_distributed_flatten_unsharded():
    """
    Feature: dtensor + torch.Tensor.flatten on unsharded middle dimensions
    Description:
        - Flatten dimensions 1 and 2 of a 3D distributed tensor.
        - Input tensor has shape (8, 4, 6), sharded only on dim0 ("dp").
        - The dimensions being flattened (dim1, dim2) are both replicated.
    Expectation:
        - The distributed flatten operation completes successfully.
        - The output layout correctly reflects the new tensor shape and existing sharding (e.g., "dp", "None").
        - The gathered distributed output matches the standalone flatten output numerically.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_output = standalone_input.flatten(1, 2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(1, 2)

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp", "None")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_distributed_flatten_negative_dims():
    """
    Feature: dtensor + torch.Tensor.flatten with negative dimension indices
    Description:
        - Flatten dimensions using negative indices (-2, -1) on a 3D distributed tensor.
        - Input tensor has shape (8, 4, 6), sharded on dim0 ("dp") and dim1 ("tp").
        - The flattened dimensions correspond to dim1 and dim2, where dim1 is sharded.
    Expectation:
        - The distributed flatten operation handles negative indices correctly.
        - The output layout correctly reflects the new tensor shape and sharding (e.g., "dp", "tp").
        - The gathered distributed output matches the standalone flatten output numerically.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_output = standalone_input.flatten(-2, -1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    # Shard dim0 and dim1. Flatten dim1 and dim2. Only dim1 is sharded in that range.
    x_placements = (Shard(0), Shard(1), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(-2, -1)

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp", "tp")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_distributed_flatten_scalar():
    """
    Feature: dtensor + torch.Tensor.flatten on a distributed scalar
    Description:
        - Apply flatten(0, -1) to a distributed scalar tensor.
        - A scalar has no dimensions, so flatten should conceptually have no effect on its shape.
    Expectation:
        - The distributed flatten operation completes successfully for a scalar.
        - The output layout indicates no sharding (e.g., "None").
        - The gathered distributed output matches the standalone flatten output, remaining a scalar.
    """
    init_dist()
    standalone_input = torch.tensor(standalone_scalar_np, device='npu')
    standalone_output = standalone_input.flatten(0, -1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = ()

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(0, -1)

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("None")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)
def test_distributed_flatten_default_args():
    """
    Feature: dtensor + torch.Tensor.flatten with default arguments
    Description:
        - Apply flatten() without explicit start_dim and end_dim.
        - The default behavior is flattening all dimensions (start_dim=0, end_dim=-1).
        - Input tensor has shape (8, 4, 6), sharded only on dim0 ("dp").
    Expectation:
        - The distributed flatten operation completes successfully.
        - The output layout correctly reflects the new 1D tensor sharding ("dp").
        - The gathered distributed output matches the standalone flatten output.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_3d_np).npu()
    standalone_output = standalone_input.flatten()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten()

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)


def test_distributed_flatten_single_dim():
    """
    Feature: dtensor + torch.Tensor.flatten with start_dim == end_dim
    Description:
        - Flatten a single dimension (e.g., start_dim=1, end_dim=1).
        - Conceptually, this operation should not change the shape or layout.
        - Input tensor has shape (4, 2, 4, 6), sharded on dim0 ("dp") and dim1 ("tp").
    Expectation:
        - The operation runs successfully without raising errors.
        - The output layout is exactly the same as the input layout ("dp", "tp", "None", "None").
        - The gathered distributed output matches the standalone output.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_4d_np).npu()
    standalone_output = standalone_input.flatten(1, 1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Shard(1), Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(1, 1)

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp", "tp", "None", "None")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)



def test_distributed_flatten_2d_to_1d():
    """
    Feature: dtensor + torch.Tensor.flatten from 2D to 1D
    Description:
        - Flatten a 2D tensor to 1D (start_dim=0, end_dim=1).
        - Input tensor has shape (8, 4), sharded on dim0 ("dp").
    Expectation:
        - The output is a 1D tensor inheriting the "dp" sharding.
        - The gathered distributed output matches the standalone flatten output.
    """
    init_dist()
    standalone_input = torch.from_numpy(standalone_input_2d_np).npu()
    standalone_output = standalone_input.flatten(0, 1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_output = dist_input.flatten(0, 1)

    expected_layout = Layout(mesh_shape=mesh.mesh_shape, alias_name=mesh.mesh_dim_names)
    expected_layout = expected_layout("dp")

    assert dist_output.layout.tensor_map == expected_layout.tensor_map
    gathered_output = local_to_global(dist_output)
    assert torch.equal(standalone_output, gathered_output)
