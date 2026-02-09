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
"""parallel_new_ones test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_new_ones import NewOnesDistributedOp

op = NewOnesDistributedOp("new_ones")

def test_new_ones_infer_layout_tuple_size():
    """
    Feature: Infer layout for new_ones with tuple size
    Description: Create a new tensor of ones with a specific tuple shape from a sharded input tensor.
    Expectation: Output layout should be fully replicated (all dimensions -1) on the same device mesh.
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 4),
    alias_name=("dp", "mp")
    )
    # Input: sharded on dim0 ("dp"), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Call new_ones with size (3, 4)
    # extra_args[0] is size
    output_layout = op.infer_layout((x_layout,), extra_args=((3, 4),))

    # Expected: Tensor map should be (-1, -1) because new_ones defaults to Replicated
    expected_map = (-1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Tuple size inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
    assert output_layout.mesh_shape == mesh.mesh_shape

def test_new_ones_infer_layout_list_size():
    """
    Feature: Infer layout for new_ones with list size
    Description: Create a new tensor of ones with a specific list shape.
    Expectation: Output layout should be fully replicated.
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 2),
    alias_name=("dp", "tp")
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    # Call new_ones with size [5, 2, 2]
    output_layout = op.infer_layout((x_layout,), extra_args=([5, 2, 2],))

    expected_map = (-1, -1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"List size inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"

def test_new_ones_infer_layout_int_size():
    """
    Feature: Infer layout for new_ones with int size
    Description: Create a 1D new tensor of ones with an integer size.
    Expectation: Output layout should be a 1D Replicated layout.
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 4),
    alias_name=("dp", "mp")
    )
    x_placements = (Shard(1), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    # Call new_ones with size 10
    output_layout = op.infer_layout((x_layout,), extra_args=(10,))

    expected_map = (-1,)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Int size inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"

def test_new_ones_ignores_input_sharding():
    """
    Feature: Ignore input sharding
    Description: Even if input is fully sharded, the new tensor should be Replicated.
    Expectation: Output layout is strictly Replicated (-1, ...).
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 2),
    alias_name=("dp", "tp")
    )
    # Input is sharded on both dimensions
    x_placements = (Shard(0), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)
    # new_ones((4, 4))
    output_layout = op.infer_layout((x_layout,), extra_args=((4, 4),))

    expected_map = (-1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Should ignore input sharding. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"

def test_new_ones_scalar_shape():
    """
    Feature: Infer layout for scalar new_ones
    Description: Create a scalar tensor (empty tuple size).
    Expectation: Output layout tensor_map should be empty tuple.
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 2),
    alias_name=("dp", "tp")
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)
    # new_ones(()) -> scalar
    output_layout = op.infer_layout((x_layout,), extra_args=((),))

    expected_map = ()
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Scalar inference failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"

def test_new_ones_missing_args():
    """
    Feature: Error handling for missing args
    Description: Call infer_layout without size argument.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 4),
    alias_name=("dp", "mp")
    )
    x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    with pytest.raises(ValueError, match="expected 'size' in extra_args"):
        op.infer_layout((x_layout,), extra_args=())

def test_new_ones_invalid_size_type():
    """
    Feature: Error handling for invalid size type
    Description: Call infer_layout with a string as size.
    Expectation: TypeError raised.
    """
    mesh = init_device_mesh(
    mesh_shape=(2, 4),
    alias_name=("dp", "mp")
    )
    x_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    with pytest.raises(TypeError, match="must be int, tuple or list"):
        op.infer_layout((x_layout,), extra_args=("invalid_size",))
