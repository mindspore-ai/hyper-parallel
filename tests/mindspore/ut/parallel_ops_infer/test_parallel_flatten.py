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
"""parallel_flatten test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_flatten import FlattenDistributedOp

op = FlattenDistributedOp("flatten")


def test_flatten_layout_inference_all_dims():
    """
    Feature: Flatten all dimensions (default behavior)
    Description: Flatten a 3D tensor across all dimensions (start_dim=0, end_dim=-1)
    Expectation: Output layout correctly merges dimensions into a 1D layout,
                 preserving the sharding of the leading sharded dimension.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input shape: (2, 4, 8)
    # Placements: sharded on dim0 ("dp"), unsharded on dim1, unsharded on dim2
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    # Flatten default: start_dim=0, end_dim=-1
    # extra_args contains the list of input shapes as the last element (appended by WithShape)
    output_layout = op.infer_layout((x_layout,), extra_args=([(2, 4, 8)],))

    # Expect the 3 dimensions to collapse into 1.
    # Since dim0 is sharded and dim1/dim2 are replicated, the flattened 1D tensor remains sharded on "dp".
    expected_map = (1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Flatten all dims failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_flatten_layout_inference_specific_start_dim():
    """
    Feature: Flatten with specific start_dim
    Description: Flatten a 3D tensor starting from dim 1 (start_dim=1, end_dim=-1)
    Expectation: Leading dimension retains its original layout, and trailing dimensions
                 are merged correctly into the second dimension.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input shape: (2, 4, 8)
    # Placements: sharded on dim0 ("dp"->1), sharded on dim1 ("mp"->0), unsharded on dim2
    x_placements = (Shard(0), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    # Flatten starting from dim 1: start_dim=1
    output_layout = op.infer_layout((x_layout,), extra_args=(1, [(2, 4, 8)]))

    # Output shape will be (2, 32)
    # Dim 0 is preserved (sharded on dp). Dim 1 and 2 merge (sharded on mp).
    expected_map = (1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Flatten from start_dim failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_flatten_layout_scalar():
    """
    Feature: Flatten a 0-D scalar tensor
    Description: PyTorch flatten converts a 0-D tensor into a 1-D tensor of shape (1,).
    Expectation: Output layout becomes a 1-D unsharded layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input shape: ()
    x_placements = ()
    x_layout = _build_layout(mesh, x_placements, 0)

    # Flatten default
    output_layout = op.infer_layout((x_layout,), extra_args=([()],))

    # Should become 1D replicated
    expected_map = (-1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Flatten scalar failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_flatten_layout_no_op():
    """
    Feature: Flatten with start_dim > end_dim
    Description: When start_dim > end_dim, PyTorch semantics dictate that the tensor is returned unchanged.
    Expectation: Output layout exactly matches the input layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input shape: (2, 4)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Flatten invalid range: start_dim=1, end_dim=0
    output_layout = op.infer_layout((x_layout,), extra_args=(1, 0, [(2, 4)]))

    # Should remain unchanged (1, -1)
    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Flatten no-op failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_flatten_invalid_dimension():
    """
    Feature: Flatten with out-of-bounds dimension
    Description: Test input validation when start_dim or end_dim exceed tensor bounds.
    Expectation: ValueError is raised with clear dimension tracking.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input shape: (2, 4)
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    with pytest.raises(ValueError, match="Dimension out of range for flatten"):
        # start_dim = 2 (out of bounds for a 2D tensor)
        op.infer_layout((x_layout,), extra_args=(2, [(2, 4)]))
