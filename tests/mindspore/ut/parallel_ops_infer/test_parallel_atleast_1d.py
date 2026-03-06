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
"""parallel_atleast_1d test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate, Partial
from hyper_parallel.core.shard.ops.parallel_atleast_1d import Atleast1DDistributedOp

op = Atleast1DDistributedOp("atleast_1d")


def test_atleast_1d_layout_inference_0d():
    """
    Feature: Convert 0D scalar to 1D
    Description: Input is a 0-dimensional scalar tensor.
    Expectation: Output layout is 1-dimensional and fully unsharded (Replicate).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Scalar input layout (0 dimensions)
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 0)

    # Convert 0D to 1D
    output_layout = op.infer_layout((x_layout,))

    expected_map = (-1,)  # The new single dimension must be unsharded
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"0D to 1D conversion failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_atleast_1d_layout_inference_1d():
    """
    Feature: Preserve 1D tensor layout
    Description: Input is a 1-dimensional tensor sharded across 'dp'.
    Expectation: Output layout is identical to the input layout (preserves sharding).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 1D tensor, sharded on dim0 ("dp"->0)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 1)

    # 1D remains 1D
    output_layout = op.infer_layout((x_layout,))

    expected_map = (1,)  # dim0 preserved (sharded on dp)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D preservation failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_atleast_1d_layout_inference_nd():
    """
    Feature: Preserve N-dimensional tensor layout (N > 1)
    Description: Input is a 2-dimensional tensor with mixed sharding.
    Expectation: Output layout is identical to the input layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 2D tensor, sharded on dim0 ("dp"), sharded on dim1 ("mp")
    x_placements = (Shard(0), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    # 2D remains 2D
    output_layout = op.infer_layout((x_layout,))

    expected_map = (1, 0)  # Both dimensions retain original sharding pattern
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"ND preservation failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_atleast_1d_invalid_partial():
    """
    Feature: Reject Partial status
    Description: Input layout contains a Partial reduction state.
    Expectation: ValueError raised with clear message (atleast_1d does not support partials).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 1D tensor, pending Partial sum on "dp"
    x_placements = (Partial("sum"), Replicate())
    x_layout = _build_layout(mesh, x_placements, 1)

    # Attempt to infer layout with Partial status
    with pytest.raises(ValueError, match="Partial status which is not allowed"):
        op.infer_layout((x_layout,))
