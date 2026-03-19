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
"""parallel_repeat test"""

import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_repeat import RepeatDistributedOp

op = RepeatDistributedOp("repeat")


def test_repeat_layout_inference():
    """
    Feature: Repeat unsharded dimension
    Description: Repeat last dimension (unsharded) while preserving sharded first dimension (repeat=1)
    Expectation: Output layout preserves sharding on preserved dimension, repeated dimension unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"), unsharded on dim1 (dimension to repeat)
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Repeat dim1 5 times, preserve dim0 with repeat=1
    output_layout = op.infer_layout((x_layout,), extra_args=(1, 5))

    expected_map = (1, -1)  # dim0 preserved (sharded), dim1 repeated (unsharded)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Basic repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_inference_3d():
    """
    Feature: Repeat with preservation (repeat=1)
    Description: Multiple dimensions with repeat=1 preservation and one repetition on unsharded dim
    Expectation: Preserved dimensions keep original sharding, repeated dimension becomes unsharded
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"→2), unsharded on dim1 (to repeat), sharded on dim2 ("mp"→0)
    x_placements = (Shard(0), Replicate(), Shard(2))
    x_layout = _build_layout(mesh, x_placements, 3)  # tensor_map = (2, -1, 0)

    # Repeat dim1 10 times, preserve others with repeat=1
    output_layout = op.infer_layout((x_layout,), extra_args=(1, 10, 1))

    expected_map = (2, -1, 0)  # All dimensions retain original sharding pattern
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Preserve with repeat=1 failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_prepend_new_dimensions():
    """
    Feature: Repeat prepending multiple new dimensions
    Description: Prepend two new dimensions to 2D tensor with mixed sharding
    Expectation: Both new dimensions unsharded, existing non-repeated dimensions preserve sharding
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

    # Prepend two new dimensions (repeat sizes 2,3), preserve existing dimensions (repeat 1,1)
    output_layout = op.infer_layout((x_layout,), extra_args=(2, 3, 1, 1))

    expected_map = (-1, -1, -1, 0)  # New dims unsharded, dim2 preserved (unsharded), dim3 preserved (sharded)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Prepend multiple new dimensions failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_scalar_expansion():
    """
    Feature: Repeat scalar tensor
    Description: Repeat 0-D scalar tensor to 2D shape (3,4)
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

    # Repeat scalar to (3,4)
    output_layout = op.infer_layout((x_layout,), extra_args=(3, 4))

    expected_map = (-1, -1)  # Both new dimensions must be unsharded
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Scalar repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_invalid_repeat_sharded_dim():
    """
    Feature: Repeat sharded dimension
    Description: Attempt to repeat a sharded dimension > 1 times (should fail)
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

    # Attempt to repeat sharded dim0 5 times
    with pytest.raises(ValueError, match="Cannot repeat dimension 0 which is sharded"):
        op.infer_layout((x_layout,), extra_args=(5, 1))



def test_repeat_layout_packed_tuple_args():
    """
    Feature: Repeat with tuple args
    Description: Attempt to pass repeat args as a single packed tuple, ensuring unpack logic works
    Expectation: Output layout correctly inferred without crashing
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Pass args packed inside a single tuple: e.g. .repeat((1, 5))
    output_layout = op.infer_layout((x_layout,), extra_args=((1, 5),))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Packed tuple repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
def test_repeat_layout_all_ones():
    """
    Feature: Repeat with all ones (shape-preserving)
    Description: Repeat tensor with 1s across all existing dimensions.
    Expectation: All dimensions preserve their original sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"->1), sharded on dim1 ("mp"->0)
    x_placements = (Shard(0), Shard(1))
    x_layout = _build_layout(mesh, x_placements, 2)

    # Repeat with 1s (preserves shape and sharding)
    output_layout = op.infer_layout((x_layout,), extra_args=(1, 1))

    expected_map = (1, 0)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"All ones repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_zero_repeat():
    """
    Feature: Repeat with 0 (creating a zero-size tensor)
    Description: Repeat an unsharded dimension 0 times.
    Expectation: Dimension becomes 0-size but remains unsharded (-1).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"->1), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Repeat dim1 0 times (valid in torch to create empty tensors)
    output_layout = op.infer_layout((x_layout,), extra_args=(1, 0))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Zero repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_invalid_zero_repeat_sharded():
    """
    Feature: Repeat sharded dimension with 0
    Description: Attempt to repeat a sharded dimension 0 times.
    Expectation: ValueError raised because repeat_times != 1 on a sharded dimension.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"->1), unsharded on dim1
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Attempt to repeat sharded dim0 0 times
    with pytest.raises(ValueError, match="Cannot repeat dimension 0 which is sharded"):
        op.infer_layout((x_layout,), extra_args=(0, 1))


def test_repeat_layout_1d_to_4d_prepend():
    """
    Feature: Prepend multiple dimensions to 1D tensor
    Description: Prepend 3 new dimensions to a 1D sharded tensor.
    Expectation: 3 prepended dims are unsharded, the original 1D sharding is preserved.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: sharded on dim0 ("dp"->1)
    x_placements = (Shard(0),)
    x_layout = _build_layout(mesh, x_placements, 1)

    # Prepend 3 dims (sizes 2,3,4) and preserve the original dim (size 1)
    output_layout = op.infer_layout((x_layout,), extra_args=(2, 3, 4, 1))

    expected_map = (-1, -1, -1, 1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"1D to 4D prepend failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_fully_replicated():
    """
    Feature: Repeat fully replicated tensor
    Description: Tensor is unsharded everywhere, repeated across all dimensions.
    Expectation: Output remains completely unsharded (-1) across all dimensions.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # Input: fully replicated
    x_placements = (Replicate(), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    output_layout = op.infer_layout((x_layout,), extra_args=(5, 6))

    expected_map = (-1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Fully replicated repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_missing_layout():
    """
    Feature: Validate empty layouts
    Description: Call infer_layout with an empty layout or None layout.
    Expectation: ValueError is raised complaining about invalid input layout.
    """
    with pytest.raises(ValueError, match="requires a valid input tensor layout"):
        op.infer_layout((None,), extra_args=(1, 2))

    with pytest.raises(ValueError, match="requires a valid input tensor layout"):
        op.infer_layout(tuple(), extra_args=(1, 2))


def test_repeat_layout_missing_extra_args():
    """
    Feature: Validate empty extra_args
    Description: Call infer_layout without passing repeat sizes in extra_args.
    Expectation: ValueError is raised complaining about missing sizes.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Replicate(),)
    x_layout = _build_layout(mesh, x_placements, 1)

    with pytest.raises(ValueError, match="requires repeat sizes in extra_args"):
        op.infer_layout((x_layout,), extra_args=tuple())


def test_repeat_layout_list_packed_args():
    """
    Feature: Robust parsing of extra_args as list
    Description: User passes repeat sizes as a list inside the extra_args tuple (e.g., `x.repeat([1, 5])`).
    Expectation: Layout is correctly inferred by unpacking the list.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Passed as list inside extra_args
    output_layout = op.infer_layout((x_layout,), extra_args=([1, 5],))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"List packed args failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_float_args_cast():
    """
    Feature: Convert float arguments to integer
    Description: User passes float values as repeat sizes (e.g., `x.repeat(1.0, 5.0)`).
    Expectation: Op implementation safely casts floats to ints and infers layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    x_placements = (Shard(0), Replicate())
    x_layout = _build_layout(mesh, x_placements, 2)

    # Float values
    output_layout = op.infer_layout((x_layout,), extra_args=(1.0, 5.0))

    expected_map = (1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Float args cast failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"


def test_repeat_layout_complex_3d_mesh():
    """
    Feature: Repeat over 3D DeviceMesh
    Description: 4D tensor mapped to a 3D mesh. Prepend 2 new dimensions, preserve existing 4, repeat the last.
    Expectation: Prepended dimensions are unsharded, preserved keep layout, repeated becomes/remains unsharded.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "mp"),
        init_backend=False
    )
    # Input: 4D tensor. tensor_map = (2, 1, 0, -1)
    x_placements = (Shard(0), Shard(1), Shard(2), Replicate())
    x_layout = _build_layout(mesh, x_placements, 4)

    # Prepend 2 dims (sizes 2,2), preserve 4 dims (1,1,1,1 -> wait, we want to repeat the last one to test)
    # Let's repeat the last unsharded dim 5 times. extra_args=(2, 2, 1, 1, 1, 5)
    output_layout = op.infer_layout((x_layout,), extra_args=(2, 2, 1, 1, 1, 5))

    expected_map = (-1, -1, 2, 1, 0, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Complex 3D mesh repeat failed. Expected {expected_map}, got {output_layout.to_dict()['tensor_map']}"
