# Copyright 2025 Huawei Technologies Co., Ltd
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
"""parallel_argmax_with_value_ops test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_argmax_with_value_ops import ArgMaxWithValueDistributedOp


@pytest.fixture(params=["ArgMaxWithValue"], name="op")
def fixture_op(request):
    """Fixture to test ArgMaxWithValueDistributedOp operations."""
    return ArgMaxWithValueDistributedOp(request.param)


def test_argmax_with_value_data_parallel_success(op):
    """
    Feature: ArgMaxWithValue data parallel
    Description: Data parallel scenario with argmax on unsharded axis
    Expectation: Success, output layout correctly reduced
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (1, True))

    expected_map = (1, -1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Data Parallel test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_model_parallel_success(op):
    """
    Feature: ArgMaxWithValue model parallel
    Description: Model parallel scenario with argmax on unsharded batch dimension
    Expectation: Success, output layout correctly reduced
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (0, True))

    expected_map = (-1, 0, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Model Parallel test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_hybrid_parallel_success(op):
    """
    Feature: ArgMaxWithValue hybrid parallel
    Description: Hybrid parallel scenario with argmax on unsharded middle dimension
    Expectation: Success, output layout correctly reduced
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (1, True))

    expected_map = (2, -1, 0)
    assert output_layout.tensor_map == expected_map, (
        f"Hybrid Parallel test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_all_replicated(op):
    """
    Feature: ArgMaxWithValue all replicated
    Description: All dimensions replicated scenario
    Expectation: Success, output layout correctly reduced
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (0, True))

    expected_map = (-1, -1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"All Replicated test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_negative_dim(op):
    """
    Feature: ArgMaxWithValue negative dimension index
    Description: Test negative dimension index (dim=-1)
    Expectation: Success, output layout correctly reduced
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (-1, True))

    expected_map = (1, -1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Negative dim test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_keep_dims_false(op):
    """
    Feature: ArgMaxWithValue with keep_dims=False
    Description: Test with keep_dims=False, reduced dimension removed
    Expectation: Success, output layout has reduced dimension removed
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (1, False))

    expected_map = (1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Keep dims False test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_sharded_dim_failure(op):
    """
    Feature: ArgMaxWithValue sharded dimension check
    Description: Attempting to compute argmax on a sharded dimension
    Expectation: Raise ValueError
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    with pytest.raises(ValueError, match="cannot perform sharding on axis dim"):
        op.infer_layout((x_layout, None, None), (0, True))


def test_argmax_with_value_model_parallel_on_mp_axis_failure(op):
    """
    Feature: ArgMaxWithValue model parallel check
    Description: Model Parallel scenario where the feature dimension is sharded
    Expectation: Raise ValueError when computing argmax on the MP axis
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    with pytest.raises(ValueError, match="cannot perform sharding on axis dim"):
        op.infer_layout((x_layout, None, None), (1, True))


def test_argmax_with_value_3d_tensor(op):
    """
    Feature: ArgMaxWithValue on 3D tensor
    Description: Test argmax on 3D tensor with mixed placements
    Expectation: Success
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (2, True))

    expected_map = (2, 1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"3D tensor test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_argmax_with_value_partial_input(op):
    """
    Feature: ArgMaxWithValue with partial input
    Description: Input with partial state
    Expectation: Raise ValueError since _allow_partial_inputs is False
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)
    x_layout.set_partial_by_dev_axis("dp", "sum")

    with pytest.raises(ValueError, match="has Partial status which is not allowed"):
        op.infer_layout((x_layout, None, None), (1, True))


def test_argmax_with_value_invalid_layouts_count(op):
    """
    Feature: ArgMaxWithValue invalid layouts count
    Description: Pass wrong number of layouts
    Expectation: Raise ValueError
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    with pytest.raises(ValueError, match="ArgMaxWithValue requires 3 layouts"):
        op.infer_layout((x_layout,), (1, True))


def test_argmax_with_value_invalid_extra_args_count(op):
    """
    Feature: ArgMaxWithValue invalid extra args count
    Description: Pass wrong number of extra args
    Expectation: Raise ValueError
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    with pytest.raises(ValueError, match="ArgMaxWithValue requires 2 extra args"):
        op.infer_layout((x_layout, None, None), (1,))


def test_argmax_with_value_get_expand_impl(op):
    """
    Feature: ArgMaxWithValue get_expand_impl
    Description: Verify get_expand_impl returns None
    Expectation: Returns None
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout, _ = op.infer_layout((x_layout, None, None), (1, True))

    impl = op.get_expand_impl(None, output_layout, (x_layout, None, None), (1, True))
    assert impl is None, (
        f"get_expand_impl test failed. Expected None, "
        f"got {impl}"
    )
