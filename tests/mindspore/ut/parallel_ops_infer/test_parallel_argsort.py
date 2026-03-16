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
"""parallel_argsort test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_argsort import ArgsortDistributedOp

op = ArgsortDistributedOp("argsort")


def test_argsort_layout_inference_basic():
    """
    Feature: Argsort on an unsharded dimension
    Description: Perform argsort along the last dimension (dim=-1), which is fully replicated.
    Expectation: Output layout is identical to the input layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Argsort on dim1 (passed as -1)
    output_layout = op.infer_layout((x_layout,), extra_args=(-1,))

    expected_map = (1, -1)  # Layout should remain completely unchanged
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Basic argsort failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_argsort_layout_inference_specific_dim():
    """
    Feature: Argsort on a specific unsharded dimension with extra kwargs
    Description: Perform argsort on dim=0, with descending=True. dim=0 is Replicate.
    Expectation: Output layout is identical to the input layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: unsharded on dim0, sharded on dim1 ("mp")
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    # Argsort on dim0, passing mock values for descending and stable
    output_layout = op.infer_layout((x_layout,), extra_args=(0, True, False))

    expected_map = (-1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Specific dim argsort failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_argsort_layout_inference_negative_dim():
    """
    Feature: Argsort handling of negative dimensions
    Description: Perform argsort on a 3D tensor using dim=-2, which maps to the middle unsharded dimension.
    Expectation: Resolves the negative dimension correctly and returns the identical layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0, unsharded on dim1, sharded on dim2
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    # Argsort on dim=-2 (which is actual_dim=1)
    output_layout = op.infer_layout((x_layout,), extra_args=(-2,))

    expected_map = (2, -1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Negative dim argsort failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_argsort_layout_invalid_sharded_dim():
    """
    Feature: Argsort on a sharded dimension
    Description: Attempt to perform argsort along a dimension that is currently sharded.
    Expectation: ValueError is raised preventing the mathematically incorrect operation.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp" -> device dim 1), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to argsort on dim=0, which is sharded
    with pytest.raises(ValueError, match="Cannot perform argsort along dimension 0 because it is currently sharded"):
        op.infer_layout((x_layout,), extra_args=(0,))


def test_argsort_layout_invalid_out_of_bounds_dim():
    """
    Feature: Argsort with out-of-bounds dimension
    Description: Attempt to perform argsort using a dimension index larger than tensor rank.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to argsort on dim=2 for a 2D tensor
    with pytest.raises(ValueError, match="is out of bounds for tensor of dimension 2"):
        op.infer_layout((x_layout,), extra_args=(2,))
