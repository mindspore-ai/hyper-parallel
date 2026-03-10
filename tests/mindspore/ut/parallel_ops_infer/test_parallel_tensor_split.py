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
"""parallel_tensor_split test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_split import TensorSplitDistributedOp

# Initialize the operator
op = TensorSplitDistributedOp("tensor_split")


class MockTensor:
    """Mock class to simulate a 1D Tensor for testing indices_or_sections."""
    def __init__(self, shape):
        self.shape = shape


def test_tensor_split_integer_default_dim():
    """
    Feature: Basic tensor_split with integer sections
    Description: Split tensor into 3 sections with default dim=0.
    Expectation: Output contains 3 identical layouts, dim 0 must be unsharded.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: unsharded on dim0, sharded on dim1
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    # extra_args contains only indices_or_sections=3, dim defaults to 0
    output_layouts = op.infer_layout((x_layout,), extra_args=(3,))

    assert len(output_layouts) == 3, f"Expected 3 output layouts, got {len(output_layouts)}"
    for out_layout in output_layouts:
        assert out_layout.to_dict()["tensor_map"] == x_layout.to_dict()["tensor_map"], \
            "Output layout should be identical to the input layout."


def test_tensor_split_tuple_specific_dim():
    """
    Feature: tensor_split with tuple/list indices and specific dim
    Description: Split tensor using list indices [1, 3] on dim=1.
    Expectation: Output contains len(indices) + 1 = 3 identical layouts.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0, unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # extra_args contains indices_or_sections=[1, 3] and dim=1
    output_layouts = op.infer_layout((x_layout,), extra_args=([1, 3], 1))

    assert len(output_layouts) == 3, f"Expected 3 output layouts, got {len(output_layouts)}"


def test_tensor_split_negative_dim():
    """
    Feature: tensor_split with negative dimension
    Description: Split tensor using negative dim (-1) which resolves to the last dimension.
    Expectation: Operator accurately calculates the true dimension and infers correctly.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0, unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # extra_args contains indices_or_sections=2 and dim=-1
    output_layouts = op.infer_layout((x_layout,), extra_args=(2, -1))

    assert len(output_layouts) == 2, f"Expected 2 output layouts, got {len(output_layouts)}"


def test_tensor_split_1d_tensor():
    """
    Feature: tensor_split with 1D tensor as indices
    Description: Split tensor using a 1D tensor (mocked) for indices_or_sections.
    Expectation: Output contains shape[0] + 1 identical layouts.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Mock a 1D tensor with shape (4,)
    mock_tensor = MockTensor(shape=(4,))
    output_layouts = op.infer_layout((x_layout,), extra_args=(mock_tensor, 0))

    # Expected output sections = 4 + 1 = 5
    assert len(output_layouts) == 5, f"Expected 5 output layouts, got {len(output_layouts)}"


def test_tensor_split_sharded_dim_error():
    """
    Feature: Error handling for splitting on a sharded dimension
    Description: Attempt to split along an axis that is actively sharded across the device mesh.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"->1)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to split the sharded dim0
    with pytest.raises(ValueError, match=r"Cannot perform tensor_split on sharded axis\[0\]"):
        op.infer_layout((x_layout,), extra_args=(2, 0))


def test_tensor_split_invalid_type_error():
    """
    Feature: Error handling for invalid indices_or_sections type
    Description: Pass a string type as indices_or_sections.
    Expectation: TypeError is raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)


    error_msg = "tensor_split: indices_or_sections must be an integer, list, tuple, or 1D tensor."
    with pytest.raises(TypeError, match=error_msg):
        op.infer_layout((x_layout,), extra_args=("invalid_string", 0))


def test_tensor_split_out_of_bounds_dim():
    """
    Feature: Error handling for out of bounds dimension
    Description: Pass a dimension index that exceeds the tensor's rank.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Tensor is 2D (dims 0, 1), attempt to split on dim 2
    with pytest.raises(ValueError, match=r"Dimension out of range \(expected \[0, 2\), got 2\)."):
        op.infer_layout((x_layout,), extra_args=(2, 2))
