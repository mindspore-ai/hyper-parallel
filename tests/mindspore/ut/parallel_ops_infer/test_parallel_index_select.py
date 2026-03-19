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
"""parallel_gather test for index_select"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_gather import IndexSelectDistributedOp
# Initialize the operator
op = IndexSelectDistributedOp("index_select")


def test_index_select_valid_axis_0():
    """
    Feature: Valid index_select on axis 0
    Description: Param is unsharded on axis 0 and sharded on axis 1. Index is 1D and sharded.
    Expectation: Output layout correctly splices the index alias map and the param alias map.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D: Unsharded on dim0, sharded on dim1 ("tp") -> Alias: ("None", "tp")
    p_placements = (Replicate(), Shard(1))
    p_layout = _build_layout(mesh, p_placements, 2)

    # Index 1D: Sharded on dim0 ("dp") -> Alias: ("dp",)
    i_placements = (Shard(0), Replicate())
    i_layout = _build_layout(mesh, i_placements, 1)

    # layouts format: (p_layout, dim_layout(None), i_layout)
    layouts = (p_layout, None, i_layout)
    extra_args = (0,)  # axis = 0

    output_layout = op.infer_layout(layouts, extra_args)

    # Expected alias: ("dp", "tp") -> tensor_map: (1, 0)
    expected_map = (1, 0)
    assert output_layout.tensor_map == expected_map, \
        f"Axis 0 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"


def test_index_select_valid_axis_1():
    """
    Feature: Valid index_select on axis 1
    Description: Param is sharded on axis 0 and unsharded on axis 1. Index is 1D and sharded.
    Expectation: Output layout combines the correct sharding strategies.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D: Sharded on dim0 ("dp"), unsharded on dim1 -> Alias: ("dp", "None")
    p_placements = (Shard(0), Replicate())
    p_layout = _build_layout(mesh, p_placements, 2)

    # Index 1D: Sharded on dim0 ("tp") -> Alias: ("tp",)
    i_placements = (Replicate(), Shard(0))
    i_layout = _build_layout(mesh, i_placements, 1)

    layouts = (p_layout, None, i_layout)
    extra_args = (1,)  # axis = 1

    output_layout = op.infer_layout(layouts, extra_args)

    # Expected alias: ("dp", "tp") -> tensor_map: (1, 0)
    expected_map = (1, 0)
    assert output_layout.tensor_map == expected_map, \
        f"Axis 1 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"


def test_index_select_valid_negative_axis():
    """
    Feature: Valid index_select with negative axis
    Description: Pass a negative axis (-1) for a 2D parameter tensor.
    Expectation: Operation calculates the correct positive axis and proceeds without errors.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D: Sharded on dim0 ("dp"), unsharded on dim1 -> Alias: ("dp", "None")
    p_placements = (Shard(0), Replicate())
    p_layout = _build_layout(mesh, p_placements, 2)

    # Index 1D: Sharded on dim0 ("tp") -> Alias: ("tp",)
    i_placements = (Replicate(), Shard(0))
    i_layout = _build_layout(mesh, i_placements, 1)

    layouts = (p_layout, None, i_layout)
    extra_args = (-1,)  # negative axis

    output_layout = op.infer_layout(layouts, extra_args)

    # Expected alias: ("dp", "tp") -> tensor_map: (1, 0)
    expected_map = (1, 0)
    assert output_layout.tensor_map == expected_map, \
        f"Negative axis inference failed. Expected {expected_map}, got {output_layout.tensor_map}"


def test_index_select_invalid_index_ndim():
    """
    Feature: Invalid multi-dimensional index tensor
    Description: Provide an index tensor that is 2D instead of the required 1D.
    Expectation: Raises ValueError regarding index dimension.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D: Unsharded -> Alias: ("None", "None")
    p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    # Index 2D: Invalid for this gather logic implementation
    i_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

    layouts = (p_layout, None, i_layout)
    extra_args = (0,)

    with pytest.raises(ValueError, match="index is not a one-dimensional Tensor"):
        op.infer_layout(layouts, extra_args)





def test_index_select_invalid_axis_positive():
    """
    Feature: Invalid positive axis
    Description: Pass a positive axis value that exceeds the dimensions of the parameter tensor.
    Expectation: Raises ValueError for index out of bounds.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D tensor
    p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    # Index 1D
    i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

    layouts = (p_layout, None, i_layout)
    extra_args = (2,)  # Valid range is [-2, 1]

    with pytest.raises(ValueError, match="is out of valid range"):
        op.infer_layout(layouts, extra_args)


def test_index_select_invalid_axis_negative():
    """
    Feature: Invalid negative axis
    Description: Pass a negative axis value that exceeds the negative dimension range.
    Expectation: Raises ValueError for index out of bounds.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D tensor
    p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    # Index 1D
    i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

    layouts = (p_layout, None, i_layout)
    extra_args = (-3,)  # Valid range is [-2, 1]

    with pytest.raises(ValueError, match="is out of valid range"):
        op.infer_layout(layouts, extra_args)


def test_index_select_invalid_layouts_length():
    """
    Feature: Invalid layouts length
    Description: Pass an incomplete layouts tuple to infer_layout.
    Expectation: Raises ValueError about missing required layouts length.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )
    # Param 2D tensor
    p_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    # Index 1D tensor
    i_layout = _build_layout(mesh, (Replicate(), Replicate()), 1)

    # Missing the intermediate dummy layout (e.g., for dim)
    layouts = (p_layout, i_layout)
    extra_args = (0,)

    with pytest.raises(ValueError, match="Gather ops requires 3 layouts"):
        op.infer_layout(layouts, extra_args)

    # Also test invalid extra_args length
    layouts_valid = (p_layout, None, i_layout)
    extra_args_invalid = ()

    with pytest.raises(ValueError, match="Gather ops requires 1 extra args"):
        op.infer_layout(layouts_valid, extra_args_invalid)
def test_index_select_unsharded_axis_unsharded_index():
    """
    Feature: Index select layout inference
    Description: Unsharded axis with an unsharded index tensor.
    Expectation: Output layout preserves sharding on non-axis dims, axis dim remains unsharded.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    # Input tensor: sharded on dim0 ("dp" -> 1), unsharded on dim1
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)  # tensor_map: (1, -1)

    # Index tensor: 1D, fully replicated
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)  # tensor_map: (-1,)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Unsharded axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_sharded_axis_unsharded_index():
    """
    Feature: Index select layout inference
    Description: Sharded axis with an unsharded index tensor.
    Expectation: Output layout drops sharding on the axis, replacing it with the unsharded index layout.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    # Input tensor: sharded on dim0 ("dp" -> 1), sharded on dim1 ("mp" -> 0)
    p_placements = [Shard(0), Shard(1)]
    p_layout = _build_layout(mesh, p_placements, 2)  # tensor_map: (1, 0)

    # Index tensor: 1D, fully replicated
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)  # tensor_map: (-1,)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Sharded axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"



def test_index_select_negative_axis():
    """
    Feature: Index select layout inference
    Description: Using a negative axis value.
    Expectation: Output layout processes the negative axis correctly as a positive index internally.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)
    p_placements = [Shard(0), Shard(1)]
    p_layout = _build_layout(mesh, p_placements, 2)  # tensor_map: (1, 0)

    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)  # tensor_map: (-1,)

    # axis = -1, translates to 1 for a 2D tensor
    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(-1,))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Negative axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_axis_out_of_bounds_positive():
    """
    Feature: Index select layout inference
    Description: Axis value exceeds the input tensor dimensions limits.
    Expectation: ValueError is raised with clear range context.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    with pytest.raises(ValueError, match="dim value 2 is out of valid range"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(2,))


def test_index_select_axis_out_of_bounds_negative():
    """
    Feature: Index select layout inference
    Description: Negative axis value is smaller than -ndim.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    with pytest.raises(ValueError, match="dim value -3 is out of valid range"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(-3,))


def test_index_select_invalid_index_ndim_2():
    """
    Feature: Index select layout inference
    Description: Index tensor is provided as a 2D tensor instead of 1D.
    Expectation: ValueError is raised enforcing 1D index requirement.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)

    # Create a 2D index layout (invalid for index_select)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 2)

    with pytest.raises(ValueError, match="index is not a one-dimensional Tensor"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0,))


def test_index_select_invalid_layouts_length_2():
    """
    Feature: Index select layout inference
    Description: An invalid number of layouts are passed to the operator.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    with pytest.raises(ValueError, match="Gather ops requires 3 layouts"):
        op.infer_layout((p_layout, i_layout), extra_args=(0,))


def test_index_select_invalid_extra_args_length():
    """
    Feature: Index select layout inference
    Description: An invalid number of extra arguments are passed.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    with pytest.raises(ValueError, match="Gather ops requires 1 extra args"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0, 1))


def test_index_select_invalid_partial_input():
    """
    Feature: Index select layout inference
    Description: Input layout has a Partial status, which is not supported for this op.
    Expectation: ValueError is raised via _check_partial_inputs blocking.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    # Create an input with Partial placement configuration
    p_placements = [Partial("sum"), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    with pytest.raises(ValueError, match="Partial status which is not allowed"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0,))
def test_index_select_3d_input_axis_0():
    """
    Feature: Index select layout inference
    Description: 3D input tensor, selecting on axis 0, with an unsharded index.
    Expectation: The output layout correctly replaces the first dimension's sharding with the index's sharding.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    # Input tensor (3D): sharded on dim0 ("dp"), sharded on dim1 ("mp"), unsharded on dim2
    p_placements = [Shard(0), Shard(1), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 3)  # tensor_map: (1, 0, -1)

    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)  # tensor_map: (-1,)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    expected_map = (-1, 0, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"3D input axis 0 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_1d_input_axis_0():
    """
    Feature: Index select layout inference
    Description: 1D input tensor and 1D unsharded index tensor.
    Expectation: Output layout is 1D and reflects the index tensor's unsharded layout.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 1)  # tensor_map: (1,)

    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)  # tensor_map: (-1,)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    expected_map = (-1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D input axis 0 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"



def test_index_select_scalar_index_invalid():
    """
    Feature: Index select layout inference
    Description: The index tensor is a 0D scalar (invalid for PyTorch index_select).
    Expectation: ValueError is raised ensuring the index is strictly 1-dimensional.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)

    # Scalar index layout (0D)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 0)  # tensor_map: ()

    with pytest.raises(ValueError, match="index is not a one-dimensional Tensor"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0,))


def test_index_select_get_expand_impl_unsharded():
    """
    Feature: Index select expand implementation
    Description: The selected axis is not sharded ("None").
    Expectation: The implementation falls back and returns the original standard function.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    # Input tensor: unsharded on dim0, sharded on dim1
    p_placements = [Replicate(), Shard(1)]
    p_layout = _build_layout(mesh, p_placements, 2)  # alias_tensor_map: ("None", "mp")

    # Define a dummy original function
    def dummy_func():
        pass

    # Request implementation for axis 0 (unsharded)
    impl = op.get_expand_impl(dummy_func, None, [p_layout], [0])

    # Should fallback to the original function
    assert impl is dummy_func, "Should return original function when axis is unsharded"


def test_index_select_get_expand_impl_sharded():
    """
    Feature: Index select expand implementation
    Description: The selected axis is sharded across a device mesh dimension.
    Expectation: The implementation returns a custom wrapper function for distributed execution.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    # Input tensor: sharded on dim0 ("dp"), unsharded on dim1
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)  # alias_tensor_map: ("dp", "None")

    def dummy_func():
        pass

    # Request implementation for axis 0 (sharded on "dp")
    impl = op.get_expand_impl(dummy_func, None, [p_layout], [0])

    # Should return the custom 'expand_impl' closure, not the original function
    assert impl is not dummy_func, "Should return custom wrapper when axis is sharded"
    assert impl.__name__ == "expand_impl", "Wrapper function should be named 'expand_impl'"


def test_index_select_negative_axis_on_3d_tensor():
    """
    Feature: Index select layout inference
    Description: Use a negative axis (-2) on a 3D tensor to verify robust boundary and translation mapping.
    Expectation: Axis -2 correctly translates to positive axis 1, yielding correct layout mapping.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "mp"), init_backend=False)

    p_placements = [Shard(0), Shard(1), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 3)  # tensor_map: (1, 0, -1)

    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)  # tensor_map: (-1,)

    # Axis -2 on 3D tensor means axis 1
    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(-2,))

    expected_map = (1, -1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Negative axis on 3D tensor failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
def test_index_select_4d_param_axis_2():
    """
    Feature: Index select on 4D tensor
    Description: 4D parameter tensor sharded on dim 0 and dim 3. Index select on unsharded dim 2.
    Expectation: The output layout correctly drops the index mapping for dim 2 and inserts the index tensor's map.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)

    # Param 4D: "dp" shards dim 0, "tp" shards dim 3 -> tensor_map: (1, -1, -1, 0)
    p_placements = [Shard(0), Shard(3)]
    p_layout = _build_layout(mesh, p_placements, 4)

    # Index 1D: Fully replicated -> tensor_map: (-1,)
    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(2,))

    expected_map = (1, -1, -1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"4D tensor axis 2 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_3d_mesh_fully_sharded_axis_1():
    """
    Feature: Index select with a 3D DeviceMesh
    Description: Use a 3D device mesh ("dp", "cp", "tp"). 3D Param is sharded across all 3 mesh dimensions.
    Expectation: Selecting axis 1 replaces its sharding with the index tensor's replicated layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "tp"),
        init_backend=False
    )

    # Param 3D: "dp"->dim 0, "cp"->dim 1, "tp"->dim 2 -> tensor_map: (2, 1, 0)
    p_placements = [Shard(0), Shard(1), Shard(2)]
    p_layout = _build_layout(mesh, p_placements, 3)

    # Index 1D: Fully replicated -> tensor_map: (-1,)
    i_placements = [Replicate(), Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

    expected_map = (2, -1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"3D mesh fully sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_1d_mesh_param_sharded():
    """
    Feature: Index select with 1D DeviceMesh
    Description: 1D device mesh ("dp"). 2D Param is sharded on dim 0. Index is replicated.
    Expectation: Output layout processes 1D mesh correctly without crashing.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",), init_backend=False)

    # Param 2D: "dp" shards dim 0 -> tensor_map: (0, -1)
    p_placements = [Shard(0)]
    p_layout = _build_layout(mesh, p_placements, 2)

    # Index 1D: Replicated -> tensor_map: (-1,)
    i_placements = [Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    expected_map = (-1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D mesh param sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_1d_mesh_index_sharded():
    """
    Feature: Index select with 1D DeviceMesh and sharded index
    Description: 1D device mesh ("dp"). 2D Param is replicated. Index is sharded on dim 0.
    Expectation: The output inherits the index's sharding on the selected axis.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",), init_backend=False)

    # Param 2D: Replicated -> tensor_map: (-1, -1)
    p_placements = [Replicate()]
    p_layout = _build_layout(mesh, p_placements, 2)

    # Index 1D: "dp" shards dim 0 -> tensor_map: (0,)
    i_placements = [Shard(0)]
    i_layout = _build_layout(mesh, i_placements, 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(1,))

    expected_map = (-1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D mesh index sharded failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_1d_param_negative_axis():
    """
    Feature: Index select on 1D tensor with negative axis
    Description: 1D Param tensor, axis is -1.
    Expectation: Correctly resolves -1 to 0 and computes layout correctly.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)

    # Param 1D: "dp" shards dim 0 -> tensor_map: (1,)
    p_placements = [Shard(0), Replicate()]
    p_layout = _build_layout(mesh, p_placements, 1)

    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(-1,))

    expected_map = (-1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D param negative axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_5d_param_last_axis():
    """
    Feature: Index select on 5D tensor
    Description: 5D parameter tensor, selecting the last axis (axis=4).
    Expectation: The layout maps the first 4 dimensions strictly and replaces the 5th.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)

    # Param 5D: "dp"->dim 0, "tp"->dim 4 -> tensor_map: (1, -1, -1, -1, 0)
    p_placements = [Shard(0), Shard(4)]
    p_layout = _build_layout(mesh, p_placements, 5)

    i_placements = [Replicate(), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(4,))

    expected_map = (1, -1, -1, -1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"5D param last axis failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_index_select_output_mesh_shape_preservation():
    """
    Feature: Output layout properties
    Description: Verify the output layout correctly preserves the input mesh shape.
    Expectation: Output layout mesh shape is identical to input layout mesh shape.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
    i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    assert output_layout.mesh_shape == p_layout.mesh_shape, \
        "Output mesh shape does not match input mesh shape."


def test_index_select_output_alias_name_preservation():
    """
    Feature: Output layout properties
    Description: Verify the output layout correctly preserves the input alias names.
    Expectation: Output layout alias name tuple is identical to input alias name tuple.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
    i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    assert output_layout.alias_name == p_layout.alias_name, \
        "Output alias name does not match input alias name."


def test_index_select_output_rank_list_preservation():
    """
    Feature: Output layout properties
    Description: Verify the output layout correctly preserves the process rank list.
    Expectation: Output layout rank list is identical to input layout rank list.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)
    i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

    output_layout = op.infer_layout((p_layout, None, i_layout), extra_args=(0,))

    assert output_layout.rank_list == p_layout.rank_list, \
        "Output rank list does not match input rank list."


def test_index_select_expand_impl_negative_sharded_axis():
    """
    Feature: get_expand_impl with negative axis
    Description: Check if get_expand_impl correctly identifies a sharded axis when axis is negative.
    Expectation: Returns a custom `expand_impl` closure rather than the original function.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    # Param 2D: "tp" shards dim 1 -> map: (-1, 0)
    p_layout = _build_layout(mesh, [Replicate(), Shard(1)], 2)

    def dummy_func():
        pass

    # Axis is -1 (which targets dim 1, the sharded dim)
    impl = op.get_expand_impl(dummy_func, None, [p_layout], [-1])

    assert impl is not dummy_func, "Should return custom wrapper for negative sharded axis."
    assert impl.__name__ == "expand_impl", "Wrapper function should be named 'expand_impl'."


def test_index_select_expand_impl_other_dims_sharded_only():
    """
    Feature: get_expand_impl with unsharded axis but other sharded dims
    Description: The target axis is unsharded, but a different dimension of the tensor is sharded.
    Expectation: Returns the original function because the selected axis itself requires no cross-device sync.
    """
    # Break long initialization into multiple lines
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    # Param 2D: "dp" shards dim 0, dim 1 is Replicate -> map: (1, -1)
    p_layout = _build_layout(mesh, [Shard(0), Replicate()], 2)

    def dummy_func():
        pass

    # Target axis is 1 (which is Replicate/"None")
    impl = op.get_expand_impl(dummy_func, None, [p_layout], [1])

    # Use parentheses to safely wrap the long assertion message across multiple lines
    assert impl is dummy_func, (
        "Should return original function when target axis is unsharded, "
        "regardless of other dims."
    )


def test_index_select_partial_index_layout_invalid():
    """
    Feature: Partial layout rejection
    Description: The index layout contains a Partial placement.
    Expectation: Raises ValueError blocking Partial inputs.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    p_layout = _build_layout(mesh, [Replicate(), Replicate()], 2)

    # Index 1D uses Partial
    i_placements = [Partial("sum"), Replicate()]
    i_layout = _build_layout(mesh, i_placements, 1)

    with pytest.raises(ValueError, match="Partial status which is not allowed"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0,))


def test_index_select_empty_layouts_invalid():
    """
    Feature: Layout validation
    Description: Pass an empty tuple for layouts.
    Expectation: Raises ValueError regarding the required layout length.
    """
    with pytest.raises(ValueError, match="Gather ops requires 3 layouts"):
        op.infer_layout((), extra_args=(0,))


def test_index_select_0d_param_invalid():
    """
    Feature: 0D scalar parameter layout
    Description: The parameter tensor is a 0D scalar (which has no dimensions to index).
    Expectation: Raises ValueError regarding out-of-bounds axis.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)

    # 0D Param
    p_layout = _build_layout(mesh, [Replicate(), Replicate()], 0)
    i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

    with pytest.raises(ValueError, match="dim value 0 is out of valid range"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0,))


def test_index_select_two_extra_args_invalid():
    """
    Feature: Extra arguments validation
    Description: Pass an extra argument tuple with length greater than 1.
    Expectation: Raises ValueError regarding required extra_args length.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    p_layout = _build_layout(mesh, [Replicate(), Replicate()], 2)
    i_layout = _build_layout(mesh, [Replicate(), Replicate()], 1)

    with pytest.raises(ValueError, match="Gather ops requires 1 extra args"):
        op.infer_layout((p_layout, None, i_layout), extra_args=(0, 1))
