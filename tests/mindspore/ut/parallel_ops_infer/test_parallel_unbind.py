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
"""parallel_unbind test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_unbind import UnbindDistributedOp

op = UnbindDistributedOp("unbind")


def test_unbind_layout_inference_dim0():
    """
    Feature: Unbind on unsharded dimension 0
    Description: Unbind input (3, 8) on dim 0. Dim 0 is unsharded, Dim 1 is sharded.
    Expectation: Return tuple of layouts, size equals dim 0 size. Remaining dimension preserves sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: (3, 8). dim0=Replicate (-1), dim1=Shard(1) ("mp" -> tensor_map 0)
    # tensor_map: (-1, 0)
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)
    input_shape = (3, 8)

    # Unbind on dim 0
    # extra_args format for unbind: (dim, [input_shape])
    output_layouts = op.infer_layout((x_layout,), extra_args=(0, [input_shape]))

    # Expect 3 outputs corresponding to input_shape[0]
    assert isinstance(output_layouts, tuple)
    assert len(output_layouts) == 3

    # Resulting layout should be 1D, preserving the original dim1 sharding (0)
    expected_map = (0,)
    for layout in output_layouts:
        assert layout.to_dict()["tensor_map"] == expected_map, \
            f"Unbind dim0 failed. Expected {expected_map}, got {layout.to_dict()['tensor_map']}"


def test_unbind_layout_inference_dim1():
    """
    Feature: Unbind on unsharded dimension 1
    Description: Unbind input (4, 4) on dim 1. Dim 0 is sharded, Dim 1 is unsharded.
    Expectation: Return tuple of layouts. Dim 0 preserves sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: (4, 4). dim0=Shard(0) ("dp" -> tensor_map 1), dim1=Replicate (-1)
    # tensor_map: (1, -1)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    input_shape = (4, 4)

    # Unbind on dim 1
    output_layouts = op.infer_layout((x_layout,), extra_args=(1, [input_shape]))

    assert len(output_layouts) == 4

    # Resulting layout should be 1D, preserving the original dim0 sharding (1)
    expected_map = (1,)
    for layout in output_layouts:
        assert layout.to_dict()["tensor_map"] == expected_map, \
            f"Unbind dim1 failed. Expected {expected_map}, got {layout.to_dict()['tensor_map']}"


def test_unbind_layout_inference_negative_dim():
    """
    Feature: Unbind with negative dimension
    Description: Unbind input (2, 4, 8) on dim -1 (last dim). Only last dim is unsharded.
    Expectation: Correctly resolves negative index and infers layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: (2, 4, 8). dim0=Shard(0) ("dp"->2), dim1=Shard(1) ("tp"->1), dim2=Replicate
    # tensor_map: (2, 1, -1)
    x_placements = (Shard(0), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)
    input_shape = (2, 4, 8)

    # Unbind on dim -1 (equivalent to dim 2)
    output_layouts = op.infer_layout((x_layout,), extra_args=(-1, [input_shape]))

    assert len(output_layouts) == 8

    # Remaining tensor_map should contain dim0 and dim1: (2, 1)
    expected_map = (2, 1)
    for layout in output_layouts:
        assert layout.to_dict()["tensor_map"] == expected_map, \
            f"Unbind negative dim failed. Expected {expected_map}, got {layout.to_dict()['tensor_map']}"


def test_unbind_layout_sharded_dim_error():
    """
    Feature: Unbind sharded dimension error
    Description: Attempt to unbind a dimension that is currently sharded.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: dim0=Shard(0), dim1=Replicate
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    input_shape = (4, 4)

    # Attempt to unbind dim 0 which is sharded
    with pytest.raises(ValueError, match="Unbinding a sharded dimension is not supported"):
        op.infer_layout((x_layout,), extra_args=(0, [input_shape]))


def test_unbind_layout_dim_out_of_range():
    """
    Feature: Unbind dimension out of range
    Description: Attempt to unbind a dimension index larger than rank.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    input_shape = (4, 4)

    # Input rank is 2, index 2 is out of range
    with pytest.raises(ValueError, match="Dimension out of range"):
        op.infer_layout((x_layout,), extra_args=(2, [input_shape]))
