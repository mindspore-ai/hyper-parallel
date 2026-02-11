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
"""parallel_isin test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_expand import ExpandDistributedOp
from hyper_parallel.core.shard.ops.parallel_expand import ExpandAsDistributedOp

op = ExpandDistributedOp("expand")
op2 = ExpandAsDistributedOp("expand_as")


def test_expand_layout_inference():
    """
    Feature: Expand unsharded singleton dimension
    Description: Expand last dimension (unsharded) while preserving sharded first dimension
    Expectation: Output layout preserves sharding on preserved dimension, expanded dimension unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"), unsharded on dim1 (singleton to expand)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Expand dim1 from 1 to 5, preserve dim0 with -1
    output_layout = op.infer_layout((x_layout,), extra_args=(-1, 5))

    expected_map = (1, -1)  # dim0 preserved (sharded), dim1 expanded (unsharded)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Basic expand failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_layout_inference_3d():
    """
    Feature: Expand with -1 preservation
    Description: Multiple dimensions with -1 preservation and one expansion on unsharded dim
    Expectation: Preserved dimensions keep original sharding, expanded dimension becomes unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"→2), unsharded on dim1 (to expand), sharded on dim2 ("mp"→0)
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)  # tensor_map = (2, -1, 0)

    # Expand dim1 from 1 to 10, preserve others with -1
    output_layout = op.infer_layout((x_layout,), extra_args=(-1, 10, -1))

    expected_map = (2, -1, 0)  # All dimensions retain original sharding pattern
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Preserve with -1 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_layout_prepend_new_dimensions():
    """
    Feature: Expand prepending multiple new dimensions
    Description: Prepend two new dimensions to 2D tensor with mixed sharding
    Expectation: Both new dimensions unsharded, existing dimensions preserve sharding
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: unsharded on dim0, sharded on dim1 ("mp"→0)
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    # Prepend two new dimensions (sizes 2,3), preserve existing dimensions
    output_layout = op.infer_layout((x_layout,), extra_args=(2, 3, -1, -1))

    expected_map = (-1, -1, -1, 0)  # New dims unsharded, dim2 preserved (unsharded), dim3 preserved (sharded)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Prepend multiple new dimensions failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_layout_scalar_expansion():
    """
    Feature: Expand scalar tensor
    Description: Expand 0-D scalar tensor to 2D shape (3,4)
    Expectation: Output layout fully unsharded (both dimensions unsharded)
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Scalar input layout (0 dimensions)
    x_placements = ()
    x_layout = _build_layout(mesh, x_placements, 0)

    # Expand scalar to (3,4)
    output_layout = op.infer_layout((x_layout,), extra_args=(3, 4))

    expected_map = (-1, -1)  # Both new dimensions must be unsharded
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Scalar expansion failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_layout_invalid_expand_sharded_dim():
    """
    Feature: Expand sharded dimension
    Description: Attempt to expand a sharded dimension (should fail)
    Expectation: ValueError raised with clear message
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"→1), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to expand sharded dim0 from 1 to 5
    with pytest.raises(ValueError, match="Cannot expand dimension 0 which is sharded"):
        op.infer_layout((x_layout,), extra_args=(5, -1))


def test_expand_layout_invalid_minus_one_for_new_dim():
    """
    Feature: Expand with -1 for new dimension
    Description: Attempt to use -1 for prepended dimension (invalid per PyTorch semantics)
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to prepend dimension with -1 (invalid)
    with pytest.raises(ValueError, match="Cannot use -1 for new dimension at position 0"):
        op.infer_layout((x_layout,), extra_args=(-1, 3, 4))


def test_expand_layout_invalid_dimension_reduction():
    """
    Feature: Expand reducing dimensions
    Description: Attempt to reduce dimensions with expand (output_ndim < input_ndim)
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to reduce 2D tensor to 1D (invalid for expand)
    with pytest.raises(ValueError, match="Cannot reduce dimensions with expand"):
        op.infer_layout((x_layout,), extra_args=(5,))


def test_expand_as_layout_basic_expansion():
    """
    Feature: Basic expand_as with unsharded singleton dimension
    Description: Input (8,1) sharded on dim0, target (8,16) → expand dim1 (unsharded)
    Expectation: Output layout preserves sharding on dim0, dim1 unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"), unsharded on dim1 (singleton to expand)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Global shapes: input (8,1) → target (8,16)
    input_global_shape = (8, 1)
    target_shape = (8, 16)

    output_layout = op2.infer_layout(
        (x_layout,),
        extra_args=((input_global_shape, target_shape),)
    )

    expected_map = (1, -1)  # dim0 preserved (sharded), dim1 expanded (unsharded)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Basic expand_as failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_as_layout_3d_preservation():
    """
    Feature: 3D expand_as with middle dimension expansion
    Description: Input (4,1,6) sharded on dim0/dim2, target (4,10,6) → expand dim1
    Expectation: Preserved dims keep sharding, expanded dim unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"→1), unsharded on dim1 (to expand), sharded on dim2 ("mp"→0)
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)  # tensor_map = (2, -1, 0)

    # Global shapes: input (4,1,6) → target (4,10,6)
    input_global_shape = (4, 1, 6)
    target_shape = (4, 10, 6)

    output_layout = op2.infer_layout(
        (x_layout,),
        extra_args=((input_global_shape, target_shape),)
    )

    expected_map = (2, -1, 0)  # All dimensions retain correct sharding pattern
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"3D expand_as preservation failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_as_layout_prepend_dimensions():
    """
    Feature: Expand_as with prepended dimensions (rank promotion)
    Description: Input (8,1) sharded on dim0, target (2,3,8,16) → prepend 2 dims + expand dim1
    Expectation: New dims unsharded, dim2 preserved (sharded), dim3 expanded (unsharded)
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"→1), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (1, -1)

    # Global shapes: input (8,1) → target (2,3,8,16)
    input_global_shape = (8, 1)
    target_shape = (2, 3, 8, 16)

    output_layout = op2.infer_layout(
        (x_layout,),
        extra_args=((input_global_shape, target_shape),)
    )

    expected_map = (-1, -1, 1, -1)  # New dims unsharded, dim2 preserved, dim3 expanded
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Prepend dimensions failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_as_layout_scalar_expansion():
    """
    Feature: Scalar tensor expand_as
    Description: Input () scalar → target (3,4,5)
    Expectation: Output layout fully unsharded (all dimensions unsharded)
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Scalar input layout (0 dimensions)
    x_placements = ()
    x_layout = _build_layout(mesh, x_placements, 0)  # tensor_map = (1, -1)

    # Global shapes: input () → target (3,4,5)
    input_global_shape = ()
    target_shape = (3, 4, 5)

    output_layout = op2.infer_layout(
        (x_layout,),
        extra_args=((input_global_shape, target_shape),)
    )

    expected_map = (-1, -1, -1)  # All new dimensions must be unsharded
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Scalar expand_as failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_expand_as_layout_invalid_sharded_singleton():
    """
    Feature: Expand_as on sharded singleton dimension
    Description: Input (8,1) with dim1 sharded → target (8,16) (invalid: cannot shard singleton)
    Expectation: ValueError raised with clear message about sharded dimension
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # INVALID: shard dim1 (singleton dimension)
    x_placements = (Replicate(), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (-1, 0) - sharded singleton

    # Global shapes: input (8,1) → target (8,16)
    input_global_shape = (8, 1)
    target_shape = (8, 16)

    with pytest.raises(ValueError, match="Cannot expand sharded dimension 1"):
        op2.infer_layout(
            (x_layout,),
            extra_args=((input_global_shape, target_shape),)
        )


def test_expand_as_layout_invalid_non_singleton_mismatch():
    """
    Feature: Non-singleton dimension size mismatch
    Description: Input (8,3) → target (8,5) (invalid: 3≠5 and 3≠1)
    Expectation: ValueError raised for incompatible dimension sizes
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (1, -1)

    # Global shapes: input (8,3) → target (8,5) - invalid non-singleton mismatch
    input_global_shape = (8, 3)
    target_shape = (8, 5)

    with pytest.raises(ValueError, match="Cannot expand dimension 1 from size 3 to 5"):
        op2.infer_layout(
            (x_layout,),
            extra_args=((input_global_shape, target_shape),)
        )


def test_expand_as_layout_invalid_rank_reduction():
    """
    Feature: Target rank smaller than input rank
    Description: Input (4,5,6) → target (4,5) (invalid: cannot reduce dimensions with expand)
    Expectation: ValueError raised for rank reduction
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(), Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 3)  # tensor_map = (-1, -1, -1)

    # Global shapes: input (4,5,6) → target (4,5) - invalid rank reduction
    input_global_shape = (4, 5, 6)
    target_shape = (4, 5)

    with pytest.raises(ValueError, match="target shape.*cannot be smaller than input shape"):
        op2.infer_layout(
            (x_layout,),
            extra_args=((input_global_shape, target_shape),)
        )


def test_expand_as_layout_right_aligned_broadcast():
    """
    Feature: Right-aligned dimension matching (PyTorch broadcast semantics)
    Description: Input (1,4) → target (3,1,4) - implicit leading singleton added
    Expectation: Leading dimension unsharded (new), middle dimension unsharded (expanded), last preserved
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: unsharded on dim1 (singleton), sharded on dim0 ("dp"→1)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)  # tensor_map = (1, -1)

    # Global shapes: input (4,1) → target (3,4,1)
    # PyTorch aligns from right: input treated as (1,4,1) before expansion
    input_global_shape = (4, 1)
    target_shape = (3, 4, 1)

    output_layout = op2.infer_layout(
        (x_layout,),
        extra_args=((input_global_shape, target_shape),)
    )

    # Expected mapping after right-alignment:
    #   dim0 (new): unsharded (-1)
    #   dim1 (preserved 4→4): sharded (1)
    #   dim2 (expanded from implicit 1→1): unsharded (-1) [was implicit leading singleton]
    expected_map = (-1, 1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Right-aligned broadcast failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
