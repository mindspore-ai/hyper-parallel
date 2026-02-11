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
"""parallel_sort test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_sort import SortDistributedOp

# Initialize the operator
# Note: Sort returns two outputs (values, indices)
op = SortDistributedOp("sort")

def test_sort_layout_inference_basic():
    """
    Feature: Sort along an unsharded dimension
    Description: Input is sharded on dim0, sorting is performed on dim1 (unsharded).
    Expectation: Returns a tuple of two layouts (values, indices), both identical to input layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: Sharded on dim0 ("dp"->1), Replicate on dim1
    # tensor_map: (1, -1)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Sort arguments: dim=1, descending=False, stable=False
    # extra_args contains non-tensor arguments in order
    output_layouts = op.infer_layout((x_layout,), extra_args=(1, False, False))

    assert isinstance(output_layouts, tuple) and len(output_layouts) == 2, \
        "Sort must return a tuple of two layouts (values, indices)"

    values_layout, indices_layout = output_layouts

    expected_map = (1, -1)

    assert values_layout.to_dict()["tensor_map"] == expected_map, \
        f"Values layout incorrect. Expected {expected_map}, got {values_layout.to_dict()['tensor_map']}"
    assert indices_layout.to_dict()["tensor_map"] == expected_map, \
        f"Indices layout incorrect. Expected {expected_map}, got {indices_layout.to_dict()['tensor_map']}"


def test_sort_layout_inference_sharded_dim_error():
    """
    Feature: Sort along a sharded dimension
    Description: Input is sharded on dim0, attempt to sort on dim0.
    Expectation: Should raise ValueError because sorting requires global data along the sort axis.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: Sharded on dim0
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to sort on dim0 (which is sharded)
    # Updated to match the actual error message format from the implementation
    # Actual message: "For 'sort', sorting along a sharded dimension (dim 0 mapped to 1) is not supported..."
    with pytest.raises(ValueError, match="sorting along a sharded dimension .* is not supported"):
        op.infer_layout((x_layout,), extra_args=(0, True, False))


def test_sort_layout_inference_negative_dim():
    """
    Feature: Sort with negative dimension index
    Description: Input (2D) sharded on dim0, sort on dim=-1 (last dim, which is unsharded).
    Expectation: Successfully infers layout, converting -1 to correct dimension index.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Input: Sharded on dim0, Replicate on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Sort on dim=-1 (which maps to dim1, Replicate)
    output_layouts = op.infer_layout((x_layout,), extra_args=(-1, False, True))

    values_layout, indices_layout = output_layouts
    expected_map = (1, -1)  # 1 corresponds to 'dp' in (dp, tp) mesh (idx 0 -> 2-1-0=1)

    assert values_layout.to_dict()["tensor_map"] == expected_map
    assert indices_layout.to_dict()["tensor_map"] == expected_map


def test_sort_layout_inference_preserve_other_dims():
    """
    Feature: Sort preserves sharding on other dimensions
    Description: 3D input sharded on dim0 and dim2. Sort on dim1 (unsharded).
    Expectation: Output layouts preserve sharding on dim0 and dim2.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: Shard(0), Replicate, Shard(2)
    # tensor_map: (2, -1, 0)  -> (dp, None, mp)
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    # Sort on dim 1
    output_layouts = op.infer_layout((x_layout,), extra_args=(1, False, False))

    values_layout, indices_layout = output_layouts
    expected_map = (2, -1, 0)

    assert values_layout.to_dict()["tensor_map"] == expected_map
    assert indices_layout.to_dict()["tensor_map"] == expected_map


def test_sort_layout_inference_all_replicate():
    """
    Feature: Sort on fully replicated tensor
    Description: Input is fully replicated. Sort on any dimension.
    Expectation: Output is fully replicated.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Sort on dim 0
    output_layouts = op.infer_layout((x_layout,), extra_args=(0, False, False))

    values_layout, _ = output_layouts
    expected_map = (-1, -1)

    assert values_layout.to_dict()["tensor_map"] == expected_map
