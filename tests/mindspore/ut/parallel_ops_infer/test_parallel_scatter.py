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
"""parallel_scatter test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_scatter import ScatterDistributedOp

op = ScatterDistributedOp("scatter")


def _create_mesh(shape=(2, 4), alias=("dp", "mp")):
    """Helper to create mesh and suppress pylint false positives."""
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter,missing-kwoa
    return init_device_mesh(
        device_type="npu",
        mesh_shape=shape,
        mesh_dim_names=alias,
        init_backend=False
    )


def test_scatter_infer_layout_success():
    """
    Feature: Scatter on valid dimension
    Description: Scatter along a replicated dimension while another dimension is sharded.
                 Index and Src layouts match Input layout.
    Expectation: Output layout is identical to input layout.
    """
    mesh = _create_mesh()
    # Input: sharded on dim0 ("dp"), replicated on dim1
    # Placements: dim0 -> Shard(0), dim1 -> Replicate
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # Index and Src must match Input layout
    index_layout = input_layout
    src_layout = input_layout

    # Scatter along dim 1 (Replicated dimension)
    # layouts order: input, dim(None), index, src
    layouts = (input_layout, None, index_layout, src_layout)
    # extra_args order: dim (int)
    extra_args = (1,)

    output_layout = op.infer_layout(layouts, extra_args=extra_args)

    # Validate output tensor map and basic properties
    # Fixed: Check mesh_shape instead of device_mesh attribute
    assert output_layout.tensor_map == input_layout.tensor_map
    assert output_layout.mesh_shape == input_layout.mesh_shape


def test_scatter_fail_sharded_dim():
    """
    Feature: Scatter on sharded dimension restriction
    Description: Attempt to scatter along a dimension that is sharded.
    Expectation: ValueError raised indicating scatter on sharded dimension is not supported.
    """
    mesh = _create_mesh()
    # Input: sharded on dim0
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    layouts = (input_layout, None, input_layout, input_layout)
    # Attempt scatter along dim 0 (Sharded)
    extra_args = (0,)

    with pytest.raises(ValueError, match="Scatter along sharded dimension 0 is not supported"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_fail_index_mismatch():
    """
    Feature: Index layout validation
    Description: Index tensor layout does not match Input tensor layout.
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    # Input: sharded on dim0
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # Index: fully replicated (mismatch)
    index_placements = (Replicate(), Replicate())
    index_layout = _build_layout(mesh, index_placements, 2)

    layouts = (input_layout, None, index_layout, input_layout)
    # Scatter along dim 1 (valid dim, but index mismatch)
    extra_args = (1,)

    with pytest.raises(ValueError, match="Index tensor layout .* must match input tensor layout"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_fail_src_mismatch():
    """
    Feature: Source layout validation
    Description: Src tensor layout does not match Input tensor layout.
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    # Input: sharded on dim0
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # Src: fully replicated (mismatch)
    src_placements = (Replicate(), Replicate())
    src_layout = _build_layout(mesh, src_placements, 2)

    layouts = (input_layout, None, input_layout, src_layout)
    extra_args = (1,)

    with pytest.raises(ValueError, match="Src tensor layout .* must match input tensor layout"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_negative_dim_normalization():
    """
    Feature: Dimension normalization
    Description: Use negative index for dim.
    Expectation: Correctly identifies the dimension and passes if valid.
    """
    mesh = _create_mesh()
    # Input: sharded on dim0, replicated on dim1
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    layouts = (input_layout, None, input_layout, input_layout)
    # Scatter along dim -1 (which is dim 1, Replicated -> Valid)
    extra_args = (-1,)

    output_layout = op.infer_layout(layouts, extra_args=extra_args)
    assert output_layout.tensor_map == input_layout.tensor_map


def test_scatter_fail_negative_dim_sharded():
    """
    Feature: Dimension normalization with sharded check
    Description: Use negative index for dim that maps to a sharded dimension.
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    # Input: replicated on dim0, sharded on dim1
    # Fixed: Use Shard(1) (mapped to 'mp') instead of Shard(0) to ensure predictable sharding on dim1
    input_placements = (Replicate(), Shard(1))
    input_layout = _build_layout(mesh, input_placements, 2)

    # Ensure setup is correct: dim 1 must be sharded for this test to be valid
    # In tensor_map, a sharded dim is represented by an integer (>=0), Replicated is -1.
    assert input_layout.tensor_map[1] != -1, "Test setup error: Dimension 1 is not sharded"

    layouts = (input_layout, None, input_layout, input_layout)
    # Scatter along dim -1 (which is dim 1, Sharded -> Invalid)
    extra_args = (-1,)

    with pytest.raises(ValueError, match="Scatter along sharded dimension 1 is not supported"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_scalar_src():
    """
    Feature: Scalar source support
    Description: Src argument is not a tensor (None in layouts), representing a scalar.
    Expectation: Success, skipping src layout check.
    """
    mesh = _create_mesh()
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # layouts[3] is None (scalar src)
    layouts = (input_layout, None, input_layout, None)
    extra_args = (1,)

    output_layout = op.infer_layout(layouts, extra_args=extra_args)
    assert output_layout.tensor_map == input_layout.tensor_map


def test_scatter_fail_partial_input():
    """
    Feature: Partial input validation
    Description: Input tensor is in Partial state (e.g. from a reduction).
    Expectation: ValueError raised as scatter cannot operate on partial tensors.
    """
    mesh = _create_mesh()
    # Input has Partial status (e.g. result of a matmul/reduce before AllReduce)
    input_placements = (Partial(), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    layouts = (input_layout, None, input_layout, input_layout)
    extra_args = (1,)

    with pytest.raises(ValueError, match="has Partial status which is not allowed"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_3d_complex_sharding():
    """
    Feature: 3D Tensor Scatter
    Description: 3D tensor sharded on dim0 and dim2, scatter on replicated dim1.
    Expectation: Success, output layout matches input.
    """
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter,missing-kwoa
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "mp", "sp"),
        init_backend=False
    )
    # Input: (Shard(0), Replicate(), Shard(2))
    # tensor_map: (dp, -1, sp) -> (2, -1, 0) approx
    input_placements = (Shard(0), Replicate(), Shard(2))
    input_layout = _build_layout(mesh, input_placements, 3)

    # All layouts align
    layouts = (input_layout, None, input_layout, input_layout)
    # Scatter on dim 1 (Replicated)
    extra_args = (1,)

    output_layout = op.infer_layout(layouts, extra_args=extra_args)
    assert output_layout.tensor_map == input_layout.tensor_map


def test_scatter_all_replicated():
    """
    Feature: Fully replicated input
    Description: Input tensor is fully replicated on all devices.
    Expectation: Scatter allowed on any dimension (since all are replicated).
    """
    mesh = _create_mesh()
    input_placements = (Replicate(), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    layouts = (input_layout, None, input_layout, input_layout)

    # Try scattering on dim 0 (valid because it is Replicated)
    extra_args = (0,)
    output_layout = op.infer_layout(layouts, extra_args=extra_args)
    assert output_layout.tensor_map == input_layout.tensor_map


def test_scatter_fail_invalid_dim_type():
    """
    Feature: Dim type validation
    Description: Pass a float instead of int for dim.
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    layouts = (input_layout, None, input_layout, input_layout)
    extra_args = (1.5,) # Invalid type

    with pytest.raises(ValueError, match="'dim' must be an integer"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_fail_dim_out_of_bounds_high():
    """
    Feature: Dim bounds check (Upper)
    Description: Dim index larger than ndim.
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2) # ndim=2

    layouts = (input_layout, None, input_layout, input_layout)
    extra_args = (2,) # Index 2 is out of bounds for size 2

    with pytest.raises(ValueError, match="is out of bounds"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_fail_dim_out_of_bounds_low():
    """
    Feature: Dim bounds check (Lower)
    Description: Negative dim index smaller than -ndim.
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2) # ndim=2

    layouts = (input_layout, None, input_layout, input_layout)
    extra_args = (-3,) # -3 is out of bounds for size 2

    with pytest.raises(ValueError, match="is out of bounds"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_fail_missing_dim_arg():
    """
    Feature: Extra args validation
    Description: extra_args tuple is empty (missing dim).
    Expectation: ValueError raised.
    """
    mesh = _create_mesh()
    input_layout = _build_layout(mesh, (Replicate(),), 1)

    layouts = (input_layout, None, input_layout, input_layout)
    extra_args = () # Empty

    with pytest.raises(ValueError, match="requires 'dim' parameter"):
        op.infer_layout(layouts, extra_args=extra_args)


def test_scatter_fail_no_input_layout():
    """
    Feature: Layouts validation
    Description: layouts tuple is None or empty.
    Expectation: ValueError raised for empty tuple; TypeError for None.
    """
    # extra_args provided but layouts missing
    extra_args = (0,)

    with pytest.raises(ValueError, match="requires a valid input tensor layout"):
        op.infer_layout((), extra_args=extra_args)

    with pytest.raises(TypeError):
        op.infer_layout(None, extra_args=extra_args)


def test_scatter_null_index_layout():
    """
    Feature: Null index layout tolerance
    Description: Index layout passed as None (simulating optional/scalar index scenario if supported by caller).
                 Code should skip index layout validation.
    Expectation: Success (if dim check passes).
    """
    mesh = _create_mesh()
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # Index layout is None
    layouts = (input_layout, None, None, input_layout)
    extra_args = (1,)

    output_layout = op.infer_layout(layouts, extra_args=extra_args)
    assert output_layout.tensor_map == input_layout.tensor_map


def test_scatter_fail_sharded_src_mismatch_complex():
    """
    Feature: Src layout mismatch with replication
    Description: Input is fully replicated, but Src is sharded.
    Expectation: ValueError raised (Src must match Input).
    """
    mesh = _create_mesh()
    # Input: Fully Replicated
    input_placements = (Replicate(), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # Src: Sharded (Mismatch)
    src_placements = (Shard(0), Replicate())
    src_layout = _build_layout(mesh, src_placements, 2)

    layouts = (input_layout, None, input_layout, src_layout)
    extra_args = (0,)

    with pytest.raises(ValueError, match="Src tensor layout .* must match input tensor layout"):
        op.infer_layout(layouts, extra_args=extra_args)
