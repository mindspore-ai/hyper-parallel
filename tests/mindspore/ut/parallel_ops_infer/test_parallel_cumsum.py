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
"""parallel_cumsum test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_cumsum import CumsumDistributedOp

# 创建 cumsum 算子实例
op = CumsumDistributedOp("cumsum")


def test_cumsum_layout_data_parallel():
    """
    Feature: Cumsum data parallel
    Description: Data parallel on non-cumsum dimension (dim=-1 unsharded)
    Expectation: Output layout identical to input layout
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    # Input: sharded on dim0 ("dp"), unsharded on dim1 (cumsum dimension)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2) # tensor_map = (1, -1)

    # Perform cumsum on last dimension (dim=-1, normalized to 1)
    output_layout = op.infer_layout((x_layout,), extra_args=(-1,))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Data parallel cumsum failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_cumsum_layout_tensor_parallel():
    """
    Feature: Cumsum tensor parallel
    Description: Tensor parallel on non-cumsum dimension (dim=0 unsharded)
    Expectation: Output layout identical to input layout
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    # Input: unsharded on dim0 (cumsum dimension), sharded on dim1 ("mp")
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (-1, 0)

    # Perform cumsum on first dimension (dim=0)
    output_layout = op.infer_layout((x_layout,), extra_args=(0,))

    expected_map = (-1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tensor parallel cumsum failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_cumsum_layout_mixed_parallel():
    """
    Feature: Cumsum mixed parallel
    Description: Mixed parallel with cumsum on unsharded middle dimension
    Expectation: Output layout identical to input layout
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 3, 4),
        alias_name = ("dp", "tp", "mp")
    )

    # Input: sharded on dim0 ("dp"), unsharded on dim1 (cumsum dimension), sharded on dim2 ("mp")
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3) # tensor_map = (2, -1, 0)

    # Perform cumsum on middle dimension (dim=1)
    output_layout = op.infer_layout((x_layout,), extra_args=(1,))

    expected_map = (2, -1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Mixed parallel cumsum failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_cumsum_layout_negative_dim():
    """
    Feature: Cumsum with negative dimension
    Description: Test negative dimension indexing (dim=-2) on 3D tensor
    Expectation: Correctly normalized and validated
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    # Input: unsharded on dim0 (cumsum dimension), sharded on dim1
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (-1, 0)

    # dim=-2 normalized to 0 (first dimension)
    output_layout = op.infer_layout((x_layout,), extra_args=(-2,))

    expected_map = (-1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Negative dimension cumsum failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_cumsum_layout_invalid_sharding_on_cumsum_dim():
    """
    Feature: Cumsum on sharded dimension
    Description: Attempt cumsum on a sharded dimension should fail
    Expectation: ValueError raised with clear message
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    # Input: sharded on BOTH dimensions including cumsum dimension
    x_placements = (Shard(0), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)   # tensor_map = (1, 0)

    # Attempt cumsum on last dimension (dim=-1) which is sharded (index 0 in mesh)
    with pytest.raises(ValueError, match="Cannot perform sharding on normalized dimension 1"):
        op.infer_layout((x_layout,), extra_args=(-1,))


def test_cumsum_layout_dim_out_of_range_positive():
    """
    Feature: Cumsum with invalid positive dimension
    Description: Dimension index exceeds tensor rank
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (-1, -1)

    # 2D tensor, dim=2 is out of range [0, 1]
    with pytest.raises(ValueError, match="Dimension 2 out of range for 2-dimensional input tensor"):
        op.infer_layout((x_layout,), extra_args=(2,))


def test_cumsum_layout_missing_dim_parameter():
    """
    Feature: Cumsum without dim parameter
    Description: extra_args missing required 'dim' parameter
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Missing extra_args entirely
    with pytest.raises(ValueError, match="cumsum requires 'dim' parameter in extra_args"):
        op.infer_layout((x_layout,), extra_args=None)

    # extra_args present but dim is None
    with pytest.raises(ValueError, match="cumsum requires 'dim' parameter in extra_args"):
        op.infer_layout((x_layout,), extra_args=(None,))


def test_cumsum_layout_invalid_dim_type():
    """
    Feature: Cumsum with non-integer dim
    Description: dim parameter must be integer
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(
        mesh_shape = (2, 4),
        alias_name = ("dp", "mp")
    )

    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    with pytest.raises(ValueError, match="'dim' must be an integer, got <class 'str'>"):
        op.infer_layout((x_layout,), extra_args=("invalid",))
