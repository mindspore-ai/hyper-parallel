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
"""parallel_outer test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_outer import OuterDistributedOp

op = OuterDistributedOp("outer")


def test_outer_layout_inference_both_replicated():
    """
    Feature: Outer product with fully replicated inputs
    Description: Compute outer product of two 1-D tensors that are fully replicated.
    Expectation: Output layout is a 2-D tensor that is fully replicated.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    # Inputs: both fully replicated
    x1_placements = (Replicate(), Replicate())
    x2_placements = (Replicate(), Replicate())

    x1_layout = _build_layout(mesh, x1_placements, 1)
    x2_layout = _build_layout(mesh, x2_placements, 1)

    output_layout = op.infer_layout((x1_layout, x2_layout))

    expected_map = (-1, -1)  # 2D output, both dimensions unsharded
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Replicated outer failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_outer_layout_inference_orthogonal_sharding():
    """
    Feature: Outer product with orthogonal sharding
    Description: Compute outer product of two 1-D tensors sharded on different mesh dimensions.
    Expectation: Output layout inherits dim0 sharding from input1 and dim1 sharding from input2.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    # Input 1: sharded on dim0 ("dp"), replicated on dim1 ("tp")
    x1_placements = (Shard(0), Replicate())
    x1_layout = _build_layout(mesh, x1_placements, 1)  # tensor_map = (1,)

    # Input 2: replicated on dim0 ("dp"), sharded on dim1 ("tp")
    x2_placements = (Replicate(), Shard(0))
    x2_layout = _build_layout(mesh, x2_placements, 1)  # tensor_map = (0,)

    output_layout = op.infer_layout((x1_layout, x2_layout))

    # Output dim0 should map to mesh dim 1 ("dp"), output dim1 should map to mesh dim 0 ("tp")
    expected_map = (1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Orthogonal sharding failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_outer_layout_inference_partial_sharding():
    """
    Feature: Outer product with partial sharding
    Description: Compute outer product where only the first input is sharded.
    Expectation: Output inherits sharding for dim0, and is unsharded for dim1.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    # Input 1: sharded on dim0 ("dp")
    x1_placements = (Shard(0), Replicate())
    x1_layout = _build_layout(mesh, x1_placements, 1)  # tensor_map = (1,)

    # Input 2: fully replicated
    x2_placements = (Replicate(), Replicate())
    x2_layout = _build_layout(mesh, x2_placements, 1)  # tensor_map = (-1,)

    output_layout = op.infer_layout((x1_layout, x2_layout))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Partial sharding failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_outer_layout_invalid_overlapping_sharding():
    """
    Feature: Validate overlapping sharding
    Description: Attempt to compute outer product when both inputs are sharded on the same mesh dimension.
    Expectation: ValueError is raised because it violates orthogonal placement rules.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    # Both inputs sharded on dim0 ("dp")
    placements = (Shard(0), Replicate())
    layout = _build_layout(mesh, placements, 1)

    with pytest.raises(ValueError, match="the two inputs cannot be sharded on the same device mesh dimension"):
        op.infer_layout((layout, layout))


def test_outer_layout_invalid_not_1d_tensor():
    """
    Feature: Validate input dimensionality
    Description: Attempt to compute outer product using inputs that are not exactly 1-D.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    # Valid 1-D layout
    x1_placements = (Replicate(), Replicate())
    x1_layout = _build_layout(mesh, x1_placements, 1)

    # Invalid 2-D layout
    x2_placements = (Replicate(), Replicate())
    x2_layout = _build_layout(mesh, x2_placements, 2)

    with pytest.raises(ValueError, match="requires exactly 1-D tensors as inputs"):
        op.infer_layout((x1_layout, x2_layout))


def test_outer_layout_invalid_num_inputs():
    """
    Feature: Validate number of inputs
    Description: Attempt to compute outer product without exactly two layouts.
    Expectation: ValueError is raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
        init_backend=False
    )

    x1_placements = (Replicate(), Replicate())
    x1_layout = _build_layout(mesh, x1_placements, 1)

    # Only passing one layout
    with pytest.raises(ValueError, match="requires exactly 2 input layouts"):
        op.infer_layout((x1_layout,))

    # Passing three layouts
    with pytest.raises(ValueError, match="requires exactly 2 input layouts"):
        op.infer_layout((x1_layout, x1_layout, x1_layout))
