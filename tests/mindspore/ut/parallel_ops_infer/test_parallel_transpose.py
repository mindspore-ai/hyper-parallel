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
"""test ut for parallel transpose infer layout"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_transpose import TransposeDistributedOp


op_ms = TransposeDistributedOp("Transpose")
op_torch_permute = TransposeDistributedOp("permute")
op_torch_transpose = TransposeDistributedOp("transpose")
op_view = TransposeDistributedOp("TransposeView")


def test_basic_transpose_operation():
    """
    Feature: Transpose distributed operator basic functionality
    Description: Test transpose operation with valid 2D layout and axis permutation
    Expectation: Output layout tensor map correctly permuted according to axis
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: Shard(1) -> "mp" (idx 0), Shard(0) -> "dp" (idx 1). TensorMap: (0, 1)
    x_placements = (Shard(1), Shard(0))
    x_layout = _build_layout(mesh, x_placements, 2)

    # MindSpore style: Transpose(input, (1, 0))
    output_layout = op_ms.infer_layout((x_layout,), extra_args=((1, 0),))

    expected_map = (1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_3d_transpose_operation():
    """
    Feature: Transpose distributed operator with 3D tensor
    Description: Test transpose operation with 3D layout and complex axis permutation
    Expectation: Output layout tensor map correctly follows the given permutation
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "cp", "mp"),
        init_backend=False
    )
    # x_placements: (Shard(2), Shard(1), Shard(0)) -> TensorMap: (0, 1, 2)
    x_placements = (Shard(2), Shard(1), Shard(0))
    x_layout = _build_layout(mesh, x_placements, 3)

    # Permutation (2, 0, 1)
    output_layout = op_ms.infer_layout((x_layout,), extra_args=((2, 0, 1),))

    expected_map = (2, 0, 1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_dimension_mismatch_error():
    """
    Feature: Transpose distributed operator error handling
    Description: Test transpose operation with mismatched tensor map and axis dimensions
    Expectation: Raise ValueError with appropriate message
    """
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"), init_backend=False)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

    with pytest.raises(ValueError, match="Input tensor shape and permutation must have the same size"):
        op_ms.infer_layout((x_layout,), extra_args=((1, 0, 2),))


def test_negative_index_error():
    """
    Feature: Transpose distributed operator validation
    Description: Test MS-style transpose with negative index (not allowed in MS style logic)
    Expectation: Raise ValueError indicating invalid permutation
    """
    mesh = init_device_mesh("npu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"), init_backend=False)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

    with pytest.raises(ValueError, match="Invalid permutation"):
        op_ms.infer_layout((x_layout,), extra_args=((-1, 0, 1),))


def test_duplicate_indices_error():
    """
    Feature: Transpose distributed operator uniqueness validation
    Description: Test transpose operation with duplicate indices in permutation
    Expectation: Raise ValueError indicating invalid permutation
    """
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"), init_backend=False)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

    with pytest.raises(ValueError, match="Invalid permutation"):
        op_ms.infer_layout((x_layout,), extra_args=((1, 1),))


def test_permute_3d_basic():
    """
    Feature: Permute distributed operator basic functionality (3D)
    Description: PyTorch style permute moves dimensions on a 3D sharded tensor
    Expectation: Tensor map updated correctly following PyTorch semantics
    """
    mesh = init_device_mesh("npu", (2, 2, 2), mesh_dim_names=("tp", "cp", "dp"), init_backend=False)
    # placements: (Shard(0), Shard(1), Shard(2)) -> Map: (2, 1, 0)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

    # permute(2, 0, 1) -> out[0]=in[2]=0, out[1]=in[0]=2, out[2]=in[1]=1
    output_layout = op_torch_permute.infer_layout((x_layout,), extra_args=((2, 0, 1),))

    expected_map = (0, 2, 1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_permute_4d_complex():
    """
    Feature: Permute distributed operator with 4D tensor
    Description: Test permute operation with 4D layout to verify higher dimension support
    Expectation: Output layout tensor map correctly follows the given 4D permutation
    """
    mesh = init_device_mesh("npu", (2, 2, 2, 2), mesh_dim_names=("dp", "mp", "cp", "tp"), init_backend=False)
    # placements: (Shard(0), Shard(1), Shard(2), Shard(3)) -> Map: (3, 2, 1, 0)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2), Shard(3)), 4)

    # permute(0, 3, 1, 2) -> out[0]=in[0]=3, out[1]=in[3]=0, out[2]=in[1]=2, out[3]=in[2]=1
    output_layout = op_torch_permute.infer_layout((x_layout,), extra_args=((0, 3, 1, 2),))

    expected_map = (3, 0, 2, 1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_torch_transpose_basic():
    """
    Feature: Torch Transpose distributed operator basic functionality
    Description: Swap two dimensions in a 2D sharded tensor
    Expectation: Tensor map dimensions swapped successfully
    """
    mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"), init_backend=False)
    # placements: (Shard(0), Shard(1)) -> Map: (1, 0)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1)), 2)

    # transpose(0, 1) -> Map becomes (0, 1)
    output_layout = op_torch_transpose.infer_layout((x_layout,), extra_args=(0, 1))

    expected_map = (0, 1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_torch_transpose_negative_indices():
    """
    Feature: Torch Transpose negative indices support
    Description: Test transpose with negative dimension indices on a 3D sharded tensor
    Expectation: Negative indices are correctly interpreted and dimensions are swapped
    """
    mesh = init_device_mesh("npu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"), init_backend=False)
    # placements: (Shard(0), Shard(1), Shard(2)) -> Map: (2, 1, 0)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

    # transpose(0, -1) -> swap dim 0 and 2. Map becomes (0, 1, 2)
    output_layout = op_torch_transpose.infer_layout((x_layout,), extra_args=(0, -1))

    expected_map = (0, 1, 2)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_torch_transpose_out_of_bounds():
    """
    Feature: Torch Transpose bounds checking
    Description: Attempt to transpose dimensions using indices that exceed tensor rank
    Expectation: Raise ValueError indicating dimensions are out of bounds
    """
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "tp"), init_backend=False)
    x_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)

    with pytest.raises(ValueError, match="Transpose dimensions out of bounds"):
        op_torch_transpose.infer_layout((x_layout,), extra_args=(0, 2))


def test_transpose_view_3d_basic():
    """
    Feature: TransposeView functionality
    Description: Verify TransposeView operation with standard 3D sharded layout
    Expectation: Output layout tensor map updated correctly according to the provided axis
    """
    mesh = init_device_mesh("npu", (2, 2, 2), mesh_dim_names=("tp", "cp", "dp"), init_backend=False)
    x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(2)), 3)

    output_layout = op_view.infer_layout((x_layout,), extra_args=((2, 0, 1),))

    expected_map = (0, 2, 1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_identity_transpose():
    """
    Feature: Identity operation
    Description: Transpose with (0, 1, 2) on a 3D tensor to verify no change
    Expectation: Output layout remains identical to the input layout
    """
    mesh = init_device_mesh("npu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"), init_backend=False)

    # Mesh Dim 0 (dp) -> Tensor Dim 0 (Map: 2), Dim 1 (cp) -> Dim 1 (Map: 1), Dim 2 (tp) -> Dim 2 (Map: 0)
    x_placements = (Shard(0), Shard(1), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3) # Map: (2, 1, 0)

    output_layout = op_ms.infer_layout((x_layout,), extra_args=((0, 1, 2),))

    expected_map = (2, 1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map
