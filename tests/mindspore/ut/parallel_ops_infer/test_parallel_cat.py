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
"""parallel_concat test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_concat import ConcatDistributedOp

op = ConcatDistributedOp("cat")


def test_cat_layout_inference_mismatch():
    """
    Feature: Cat layout inference with mismatched input layouts
    Description: Attempt to concatenate tensors that have different sharding strategies
    Expectation: ValueError is raised indicating that all input tensors must have the same layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Tensor 1: Sharded on dim0
    x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    # Tensor 2: Sharded on dim1
    y_layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

    extra_args = [0]
    with pytest.raises(ValueError, match="All input tensors must have the same layout"):
        op.infer_layout((x_layout, y_layout), extra_args=extra_args)



def test_cat_get_expand_impl_unsharded_dim():
    """
    Feature: Get execution implementation for unsharded dimension
    Description: Try to get expand implementation when concatenating along a dimension that is NOT sharded
    Expectation: get_expand_impl returns None, delegating the operation to original PyTorch cat
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Tensor sharded on dim0, replicated on dim1
    placements = (Shard(0), Replicate())
    layout = _build_layout(mesh, placements, 2)

    # We are concatenating on dim=1, which is Replicate (unsharded, mapping is -1)
    extra_args = [1]

    # get_expand_impl expects: func, output_layout, input_layouts, extra_args
    # Directly assert the function call without assigning it to a variable
    assert op.get_expand_impl(None, layout, (layout, layout), extra_args) is None, \
        "Concatenating on an unsharded dimension should return None to use the fallback PyTorch cat."

def test_cat_layout_inference_multiple_inputs():
    """
    Feature: Cat layout inference with more than 2 inputs
    Description: Concatenate 3 tensors with identical layouts
    Expectation: Output layout is identical to the base input layout without errors
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, placements, 2)

    extra_args = [1]  # dim = 1
    # Passing 3 identical layouts
    output_layout = op.infer_layout((x_layout, x_layout, x_layout), extra_args=extra_args)

    assert output_layout == x_layout, "Output layout should be identical to the input layout."
    assert extra_args[0] == 1, "Dimension should remain 1."



def test_cat_layout_inference_negative_dim_3d():
    """
    Feature: Negative dimension normalization for 3D tensors
    Description: Concatenate 3D tensors using dim=-2
    Expectation: Output layout matches, and extra_args normalizes dim=-2 to dim=1
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, placements, 3)

    extra_args = [-2]  # Should normalize to 3 + (-2) = 1
    op.infer_layout((x_layout, x_layout), extra_args=extra_args)

    assert extra_args[0] == 1, f"Negative dimension -2 should be normalized to 1, got {extra_args[0]}."


def test_cat_layout_inference_dim_minus_ndim():
    """
    Feature: Negative dimension normalization boundary case (dim = -ndim)
    Description: Concatenate 2D tensors using dim=-2
    Expectation: extra_args normalizes dim=-2 to dim=0
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    extra_args = [-2]  # ndim = 2. Should normalize to 2 + (-2) = 0
    op.infer_layout((x_layout, x_layout), extra_args=extra_args)

    assert extra_args[0] == 0, f"Dimension -2 on a 2D tensor should normalize to 0, got {extra_args[0]}."




def test_cat_get_expand_impl_unsharded_dim_3d():
    """
    Feature: Execution implementation for unsharded dimension in 3D
    Description: Concatenating 3D tensors along an unsharded middle dimension
    Expectation: get_expand_impl returns None
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Middle dimension (dim=1) is unsharded (Replicate)
    x_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(2)), 3)

    extra_args = [1]

    assert op.get_expand_impl(None, x_layout, (x_layout, x_layout), extra_args) is None, \
        "Concatenating on an unsharded dimension should return None."


def test_cat_layout_inference_mismatch_multiple():
    """
    Feature: Mismatched layouts across multiple inputs
    Description: Provide 3 input tensors where the 3rd one has a different layout
    Expectation: ValueError is raised indicating inconsistency
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout_1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
    layout_2 = _build_layout(mesh, (Replicate(), Shard(1)), 2)

    extra_args = [0]
    with pytest.raises(ValueError, match="All input tensors must have the same layout"):
        op.infer_layout((layout_1, layout_1, layout_2), extra_args=extra_args)



def test_cat_get_expand_impl_all_replicated():
    """
    Feature: Execution implementation when fully replicated
    Description: Tensors are fully replicated on all dimensions
    Expectation: get_expand_impl returns None for any concatenation dimension
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Fully replicated tensor
    x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    # Try dim 0
    extra_args_0 = [0]

    # Try dim 1
    extra_args_1 = [1]

    assert op.get_expand_impl(None, x_layout, (x_layout, x_layout), extra_args_0) is None, \
        "Fully replicated tensor should return None for dim=0."
    assert op.get_expand_impl(None, x_layout, (x_layout, x_layout), extra_args_1) is None, \
        "Fully replicated tensor should return None for dim=1."
def test_cat_layout_inference_sharded_dim_raises():
    """
    Feature: Cat layout inference on a sharded dimension
    Description: Attempt to concatenate along dim=0 which is sharded
    Expectation: ValueError is raised because concatenation on a sharded dim is not supported
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Tensor sharded on dim0
    layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    extra_args = [0]
    with pytest.raises(ValueError, match="Concatenation along a sharded dimension"):
        op.infer_layout((layout, layout), extra_args=extra_args)


def test_cat_layout_inference_negative_sharded_dim_raises():
    """
    Feature: Cat layout inference on a negative sharded dimension
    Description: Attempt to concatenate along dim=-1 where the last dimension is sharded
    Expectation: ValueError is raised with the properly normalized dimension
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Tensor sharded on dim1 (the last dimension)
    layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

    extra_args = [-1]
    with pytest.raises(ValueError, match=r"normalized_dim=1\) is not supported"):
        op.infer_layout((layout, layout), extra_args=extra_args)


def test_cat_layout_inference_empty_inputs():
    """
    Feature: Cat layout inference with empty inputs
    Description: Pass an empty tuple to layouts
    Expectation: ValueError is raised requiring at least one input DTensor
    """
    extra_args = [0]
    with pytest.raises(ValueError, match="cat requires at least one input DTensor"):
        op.infer_layout((), extra_args=extra_args)


def test_cat_layout_inference_scalar_args_ignored():
    """
    Feature: Cat layout inference with scalar inputs mixed in
    Description: Pass a tuple of (layout, None, layout) simulating a scalar argument interleave
    Expectation: The None layout is ignored and the output layout is correctly inferred
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

    extra_args = [0]
    # dim 0 is Replicate, so it should succeed
    output_layout = op.infer_layout((layout, None, layout), extra_args=extra_args)
    assert output_layout == layout, "Output layout should match the valid base layout."


def test_cat_layout_inference_default_dim():
    """
    Feature: Cat layout inference with default dimension
    Description: Pass an empty extra_args list, which defaults dim to 0 internally
    Expectation: Defaults to dim=0, normalizes it, and appends it to extra_args
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Ensure dim 0 is Replicate so it passes
    layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

    extra_args = []
    output_layout = op.infer_layout((layout, layout), extra_args=extra_args)

    assert output_layout == layout
    assert len(extra_args) == 1
    assert extra_args[0] == 0, "Default dimension 0 should be appended to extra_args."


def test_cat_layout_inference_default_dim_sharded():
    """
    Feature: Cat layout inference with default dimension (sharded)
    Description: Default dim is 0, but dim 0 is sharded
    Expectation: ValueError is raised for sharded dimension 0
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    extra_args = []
    with pytest.raises(ValueError, match="Concatenation along a sharded dimension"):
        op.infer_layout((layout, layout), extra_args=extra_args)


def test_cat_layout_inference_single_input():
    """
    Feature: Cat layout inference with a single input tensor
    Description: Concatenate a single tensor (trivial case)
    Expectation: Returns the same layout without errors
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)

    extra_args = [0]
    output_layout = op.infer_layout((layout,), extra_args=extra_args)
    assert output_layout == layout


def test_cat_layout_inference_sharded_dim_multiple_inputs():
    """
    Feature: Cat layout inference on a sharded dim with >2 inputs
    Description: Concatenate 3 tensors along a sharded middle dimension
    Expectation: ValueError is raised
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Shard on dim=1
    layout = _build_layout(mesh, (Replicate(), Shard(1), Replicate()), 3)

    extra_args = [1]
    with pytest.raises(ValueError, match="Concatenation along a sharded dimension"):
        op.infer_layout((layout, layout, layout), extra_args=extra_args)




def test_cat_layout_inference_all_none_layouts():
    """
    Feature: Cat layout inference when all layouts are None
    Description: Extreme edge case where no DTensor layout is provided (e.g. all inputs are scalar/normal tensors)
    Expectation: ValueError is raised indicating at least one DTensor is required
    """
    extra_args = [0]
    with pytest.raises(ValueError, match="cat requires at least one input DTensor"):
        op.infer_layout((None, None), extra_args=extra_args)
def test_cat_layout_inference_1d_mesh():
    """
    Feature: Cat layout inference on 1D DeviceMesh
    Description: Concatenate tensors on a 1D device mesh without sharding
    Expectation: Output layout is properly inferred
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4,),
        mesh_dim_names=("dp",),
        init_backend=False
    )
    layout = _build_layout(mesh, (Replicate(),), 1)

    extra_args = [0]
    output_layout = op.infer_layout((layout, layout), extra_args=extra_args)

    assert output_layout == layout, "1D mesh concatenation on Replicate dim should succeed."
    assert extra_args[0] == 0


def test_cat_layout_inference_4d_tensor_dim2():
    """
    Feature: Cat layout inference on 4D tensors
    Description: Concatenate 4D tensors along dim=2, where dim 0 and 1 are sharded
    Expectation: Output layout is identical to the input layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # 4D tensor: Sharded on dim0 and dim1, Replicate on dim2 and dim3
    layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate(), Replicate()), 4)

    extra_args = [2]
    output_layout = op.infer_layout((layout, layout), extra_args=extra_args)

    assert output_layout == layout, "Concatenating 4D tensor on unsharded dim 2 should succeed."



def test_cat_layout_inference_last_dim():
    """
    Feature: Cat layout inference on the last dimension
    Description: Concatenate along the highest positive dimension index (ndim - 1)
    Expectation: Success when the last dimension is Replicated
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    extra_args = [1]  # The last dimension for a 2D tensor
    output_layout = op.infer_layout((layout, layout), extra_args=extra_args)

    assert output_layout == layout
    assert extra_args[0] == 1



def test_cat_get_expand_impl_empty_extra_args():
    """
    Feature: get_expand_impl with empty extra_args
    Description: Call get_expand_impl with extra_args=[] to ensure it handles boundary conditions safely
    Expectation: Returns None, delegating to native cat
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

    extra_args = []


    assert op.get_expand_impl(None, layout, (layout, layout), extra_args) is None, \
        "Empty extra_args should be safely handled by get_expand_impl."


def test_cat_layout_inference_interleaved_parallel():
    """
    Feature: Cat layout inference with interleaved parallel (virtual sharding)
    Description: Use an interleaved_parallel mesh dim name and concatenate on an unsharded dimension
    Expectation: Output layout is correctly inferred
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "interleaved_parallel"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    extra_args = [1]
    output_layout = op.infer_layout((layout, layout), extra_args=extra_args)

    assert output_layout == layout


def test_cat_layout_inference_negative_dim_replicate():
    """
    Feature: Negative dimension normalization on a replicated dimension
    Description: Use dim=-1 on a 2D tensor where the last dimension is Replicate
    Expectation: Normalizes to dim=1 and succeeds
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Tensor sharded on dim0, replicated on dim1
    layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    extra_args = [-1]
    output_layout = op.infer_layout((layout, layout), extra_args=extra_args)

    assert output_layout == layout
    assert extra_args[0] == 1, "Negative dimension -1 should be successfully normalized to 1."



def test_cat_layout_inference_interspersed_nones():
    """
    Feature: Cat layout inference with interspersed None layouts
    Description: Pass a tuple with multiple None values simulating multiple scalar/non-tensor arguments
    Expectation: The valid layout is correctly identified and returned
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    extra_args = [1]
    output_layout = op.infer_layout((None, layout, None, layout, None), extra_args=extra_args)

    assert output_layout == layout, "Should correctly ignore all None layouts and return the valid base layout."


def test_cat_layout_inference_multi_axis_sharded_dim_raises():
    """
    Feature: Cat layout inference on a multi-axis mesh
    Description: Attempt to concatenate along a dimension that is sharded across the mesh
    Expectation: ValueError is raised
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Shard(0), Shard(1), Replicate()), 3)

    extra_args = [0]  # dim 0 is sharded
    with pytest.raises(ValueError, match="Concatenation along a sharded dimension"):
        op.infer_layout((layout, layout), extra_args=extra_args)


def test_cat_layout_inference_out_of_bounds_dim_positive():
    """
    Feature: Cat layout inference with out-of-bounds dimension
    Description: Pass a dimension index that exceeds the tensor's dimensionality
    Expectation: IndexError is naturally raised when trying to access tensor_map
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    extra_args = [5]  # A 2D tensor only has dim 0 and 1
    with pytest.raises(IndexError):
        op.infer_layout((layout, layout), extra_args=extra_args)



def test_cat_get_expand_impl_multi_layout_tuple():
    """
    Feature: get_expand_impl with multiple inputs
    Description: Request expand implementation for 3 concatenated tensors
    Expectation: Returns None, delegating entirely to PyTorch backend
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout = _build_layout(mesh, (Replicate(), Shard(0)), 2)

    extra_args = [0]

    assert op.get_expand_impl(None, layout, (layout, layout, layout), extra_args) is None, \
    "Should return None to use native cat implementation."


def test_cat_layout_inference_mismatch_shard_vs_replicate():
    """
    Feature: Layout mismatch detection (Shard vs Replicate)
    Description: Concatenate two tensors where one is sharded and the other is replicated
    Expectation: ValueError is raised due to mismatched layouts
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
    layout2 = _build_layout(mesh, (Replicate(), Replicate()), 2)

    extra_args = [1]
    with pytest.raises(ValueError, match="All input tensors must have the same layout"):
        op.infer_layout((layout1, layout2), extra_args=extra_args)


def test_cat_layout_inference_mismatch_shard0_vs_shard1():
    """
    Feature: Layout mismatch detection (Different Shard Mesh Dimensions)
    Description: Concatenate two tensors sharded on different mesh dimensions
    Expectation: ValueError is raised due to mismatched layouts
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
    layout2 = _build_layout(mesh, (Shard(1), Replicate()), 2)

    extra_args = [1]
    with pytest.raises(ValueError, match="All input tensors must have the same layout"):
        op.infer_layout((layout1, layout2), extra_args=extra_args)


def test_cat_layout_inference_negative_dim_normalization_3d():
    """
    Feature: Negative dimension normalization for 3D tensors (dim=-1)
    Description: Use dim=-1 on a 3D tensor where the last dimension is Replicate
    Expectation: Normalizes to dim=2 and successfully infers layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # 3D tensor: Sharded on dim0, Replicated on dim1 and dim2
    layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 3)

    extra_args = [-1]
    op.infer_layout((layout, layout), extra_args=extra_args)

    assert extra_args[0] == 2, "Negative dim -1 for a 3D tensor should correctly normalize to 2."
