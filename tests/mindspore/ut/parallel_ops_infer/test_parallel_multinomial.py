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
"""parallel_multinomial test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_multinomial import MultinomialDistributedOp

# Initialize the operator
op = MultinomialDistributedOp("multinomial")

def test_multinomial_infer_layout_1d():
    """
    Feature: Multinomial layout inference for 1D input
    Description: Input is a 1D probability vector (replicated).
    Expectation: Output layout should be 1D and Replicated ("None").
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 1D Tensor, Replicated on all mesh dims
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 1) # Tensor Dim = 1

    # Call infer_layout with dummy extra_args (num_samples=10, replacement=True, generator=None)
    output_layout = op.infer_layout((x_layout,), extra_args=(10, True, None))

    # Expectation: Output is 1D, Replicated ("None")
    # tensor_map: (-1,)
    expected_map = (-1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_multinomial_infer_layout_2d_batch_sharded():
    """
    Feature: Multinomial layout inference for 2D input (Batch Sharded)
    Description: Input (N, C) is sharded on Batch dim (dim 0), C dim (dim 1) is replicated.
    Expectation: Output (N, num_samples) preserves sharding on dim 0, dim 1 is Replicated.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 2D Tensor
    # Mesh dim 0 ("dp") shards Tensor dim 0 (Batch)
    # Mesh dim 1 ("mp") is Replicate
    # Resulting Alias Map: ("dp", "None")
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), extra_args=(10, True, None))

    # Expectation:
    # Dim 0 (Batch): Preserves "dp" -> index 1 (in mesh definition order reverse mapping)
    # Dim 1 (Samples): New dimension is "None" -> index -1
    # tensor_map: (1, -1)
    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"2D Batch sharded inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_multinomial_infer_layout_2d_fully_replicated():
    """
    Feature: Multinomial layout inference for 2D input (Fully Replicated)
    Description: Input (N, C) is fully replicated.
    Expectation: Output (N, num_samples) is fully replicated.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 2D Tensor, Replicated
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), extra_args=(10, True, None))

    # Expectation: tensor_map (-1, -1)
    expected_map = (-1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"2D Replicated inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_multinomial_error_sharded_prob_dim():
    """
    Feature: Error handling for sharded probability dimension
    Description: Attempt to run multinomial when the last dimension (probability classes) is sharded.
    Expectation: ValueError raised requiring redistribution.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 2D Tensor
    # Mesh dim 0 ("dp") is Replicate
    # Mesh dim 1 ("mp") shards Tensor dim 1 (Probability Dim) -> THIS IS ILLEGAL for multinomial
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    # Check alias tensor map is indeed ("None", "mp")
    assert x_layout.alias_tensor_map == ("None", "mp")

    with pytest.raises(ValueError, match="must not be sharded"):
        op.infer_layout((x_layout,), extra_args=(10, True, None))


def test_multinomial_error_invalid_ndim_3d():
    """
    Feature: Error handling for invalid input dimensions
    Description: Attempt to run multinomial on a 3D tensor.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: 3D Tensor
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)

    with pytest.raises(ValueError, match="input dimension must be 1 or 2"):
        op.infer_layout((x_layout,), extra_args=(10, True, None))
