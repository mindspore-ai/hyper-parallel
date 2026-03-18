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
"""parallel_activation_with_axis test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_activation_with_axis import ActivationWithAxisDistributedOp

@pytest.fixture(params=["Softmax", "Swiglu", "softmax"], name="op")
def fixture_op(request):
    """
    Fixture to test'ActivationWithAxisDistributedOp' operations.
    """
    return ActivationWithAxisDistributedOp(request.param)


def test_activation_with_axis_data_parallel_success(op):
    """
    Feature: ActivationWithAxis data parallel
    Description: Data parallel scenario with softmax on unsharded axis
    Expectation: Success, output layout equals input layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), (1,))

    expected_map = (1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Data Parallel test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_model_parallel_success(op):
    """
    Feature: ActivationWithAxis model parallel
    Description: Model parallel scenario with softmax on unsharded batch dimension
    Expectation: Success, output layout equals input layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), (0,))

    expected_map = (-1, 0)
    assert output_layout.tensor_map == expected_map, (
        f"Model Parallel test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_hybrid_parallel_success(op):
    """
    Feature: ActivationWithAxis hybrid parallel
    Description: Hybrid parallel scenario with softmax on unsharded middle dimension
    Expectation: Success, output layout equals input layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout = op.infer_layout((x_layout,), (1,))

    expected_map = (2, -1, 0)
    assert output_layout.tensor_map == expected_map, (
        f"Hybrid Parallel test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_all_replicated(op):
    """
    Feature: ActivationWithAxis all replicated
    Description: All dimensions replicated scenario
    Expectation: Success, output layout equals input layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), (0,))

    expected_map = (-1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"All Replicated test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_negative_dim(op):
    """
    Feature: ActivationWithAxis negative dimension index
    Description: Test negative dimension index (dim=-1)
    Expectation: Success, output layout equals input layout
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), (-1,))

    expected_map = (1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Negative dim test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_sharded_dim_failure(op):
    """
    Feature: ActivationWithAxis sharded dimension check
    Description: Attempting to compute activation on a sharded dimension
    Expectation: Raise ValueError
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    with pytest.raises(ValueError, match="requires the reduction axis to be un-sharded"):
        op.infer_layout((x_layout,), (0,))


def test_activation_with_axis_model_parallel_on_mp_axis_failure(op):
    """
    Feature: ActivationWithAxis model parallel check
    Description: Model Parallel scenario where the feature dimension is sharded
    Expectation: Raise ValueError when computing activation on the MP axis
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    with pytest.raises(ValueError, match="requires the reduction axis to be un-sharded"):
        op.infer_layout((x_layout,), (-1,))


def test_activation_with_axis_multi_axis_tuple(op):
    """
    Feature: ActivationWithAxis with multiple axes
    Description: Test activation with tuple of axes
    Expectation: Success if all axes are unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    output_layout = op.infer_layout((x_layout,), (0, 2))

    expected_map = (-1, -1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Multi axis tuple test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_multi_axis_sharded_failure(op):
    """
    Feature: ActivationWithAxis with multiple axes, one sharded
    Description: Test activation with tuple of axes where one is sharded
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

    with pytest.raises(ValueError, match="requires the reduction axis to be un-sharded"):
        op.infer_layout((x_layout,), (0, 2))


def test_activation_with_axis_input_consistency_failure(op):
    """
    Feature: ActivationWithAxis layout consistency check
    Description: Inputs have different layouts
    Expectation: Raise ValueError
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
    layout2 = _build_layout(mesh, (Replicate(), Replicate()), 2)

    with pytest.raises(ValueError, match="requires all tensor inputs to have the same layout"):
        op.infer_layout((layout1, layout2), (1,))


def test_activation_with_axis_multi_input_same_layout(op):
    """
    Feature: ActivationWithAxis with multiple inputs (same layout)
    Description: Multiple inputs with consistent layout
    Expectation: Success
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    layout1 = _build_layout(mesh, (Shard(0), Replicate()), 2)
    layout2 = _build_layout(mesh, (Shard(0), Replicate()), 2)

    output_layout = op.infer_layout((layout1, layout2), (1,))

    expected_map = (1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"Multi input same layout test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_3d_tensor(op):
    """
    Feature: ActivationWithAxis on 3D tensor
    Description: Test activation on 3D tensor with mixed placements
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

    output_layout = op.infer_layout((x_layout,), (2,))

    expected_map = (2, 1, -1)
    assert output_layout.tensor_map == expected_map, (
        f"3D tensor test failed. Expected {expected_map}, "
        f"got {output_layout.tensor_map}"
    )


def test_activation_with_axis_invalid_extra_args_type(op):
    """
    Feature: ActivationWithAxis invalid extra args type
    Description: Pass invalid type as extra args
    Expectation: Raise ValueError
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    with pytest.raises(ValueError, match="should be int or tuple"):
        op.infer_layout((x_layout,), ("invalid",))


def test_activation_with_axis_partial_input(op):
    """
    Feature: ActivationWithAxis with partial input
    Description: Input with partial state
    Expectation: Raise ValueError since _allow_partial_inputs is False
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)
    x_layout.set_partial_by_dev_axis("dp", "sum")

    with pytest.raises(ValueError, match="has Partial status which is not allowed"):
        op.infer_layout((x_layout,), (1,))


def test_activation_with_axis_get_expand_impl(op):
    """
    Feature: ActivationWithAxis get_expand_impl
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
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), (1,))

    impl = op.get_expand_impl(None, output_layout, (x_layout,), (1,))
    assert impl is None, (
        f"get_expand_impl test failed. Expected None, "
        f"got {impl}"
    )
