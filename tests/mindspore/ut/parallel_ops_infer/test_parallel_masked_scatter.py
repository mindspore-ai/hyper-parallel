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
"""parallel_masked_scatter test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_masked_scatter import MaskedScatterDistributedOp

# Initialize the operator
op = MaskedScatterDistributedOp("masked_scatter")

def test_masked_scatter_success():
    """
    Feature: MaskedScatter with fully replicated inputs
    Description: All inputs (input, mask, source) are Replicated
    Expectation: Output layout is the same as input layout (Replicated)
    """
    mesh = init_device_mesh(
        mesh_shape=(2, 4),
        alias_name=("dp", "mp")
    )

    # 1. Prepare fully replicated layouts
    # Input: (Replicate, Replicate) -> Unsharded
    input_placements = (Replicate(), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    # Mask: (Replicate, Replicate) -> Unsharded
    mask_placements = (Replicate(), Replicate())
    mask_layout = _build_layout(mesh, mask_placements, 2)

    # Source: (Replicate) -> Unsharded
    source_placements = (Replicate(),)
    source_layout = _build_layout(mesh, source_placements, 1)

    # 2. Infer layout
    output_layout = op.infer_layout((input_layout, mask_layout, source_layout))

    # 3. Validation
    # Expect tensor_map to contain only -1 (Replicated)
    expected_map = (-1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map, \
        f"Expected fully replicated output (-1, -1), but got {output_layout.to_dict()['tensor_map']}"


def test_masked_scatter_fail_sharded_input():
    """
    Feature: MaskedScatter with sharded input
    Description: The 'input' tensor is sharded
    Expectation: ValueError raised (masked_scatter does not support sharded inputs)
    """
    mesh = init_device_mesh(
        mesh_shape=(2, 4),
        alias_name=("dp", "mp")
    )

    # Input is sharded on dim 0
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)

    mask_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    source_layout = _build_layout(mesh, (Replicate(),), 1)

    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((input_layout, mask_layout, source_layout))


def test_masked_scatter_fail_sharded_mask():
    """
    Feature: MaskedScatter with sharded mask
    Description: The 'mask' tensor is sharded
    Expectation: ValueError raised (masked_scatter does not support sharded mask)
    """
    mesh = init_device_mesh(
        mesh_shape=(2, 4),
        alias_name=("dp", "mp")
    )

    input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    # Mask is sharded on dim 1
    mask_placements = (Replicate(), Shard(1))
    mask_layout = _build_layout(mesh, mask_placements, 2)

    source_layout = _build_layout(mesh, (Replicate(),), 1)

    with pytest.raises(ValueError, match=r"(?s)Input 1 .* is sharded"):
        op.infer_layout((input_layout, mask_layout, source_layout))


def test_masked_scatter_fail_sharded_source():
    """
    Feature: MaskedScatter with sharded source
    Description: The 'source' tensor is sharded
    Expectation: ValueError raised (masked_scatter does not support sharded source)
    """
    mesh = init_device_mesh(
        mesh_shape=(2, 4),
        alias_name=("dp", "mp")
    )

    input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    mask_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    # Source is sharded
    source_placements = (Shard(0),)
    source_layout = _build_layout(mesh, source_placements, 1)

    with pytest.raises(ValueError, match=r"(?s)Input 2 .* is sharded"):
        op.infer_layout((input_layout, mask_layout, source_layout))


def test_masked_scatter_success_scalar():
    """
    Feature: MaskedScatter with scalar inputs
    Description: Inputs are 0-D scalars (fully replicated implicitly)
    Expectation: Success, output is scalar layout
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # 0-D tensor layout (empty placements)
    scalar_layout = _build_layout(mesh, (), 0)

    # input(scalar), mask(scalar), source(scalar)
    output_layout = op.infer_layout((scalar_layout, scalar_layout, scalar_layout))

    assert output_layout.to_dict()["tensor_map"] == (), \
        "Expected scalar output (empty tensor_map)"


def test_masked_scatter_success_high_dim2():
    """
    Feature: MaskedScatter with high-dimensional tensors
    Description: 4D tensors (e.g. NCHW image batch) fully replicated
    Expectation: Success
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # 4D Replicated
    placements = (Replicate(), Replicate(), Replicate(), Replicate())
    layout_4d = _build_layout(mesh, placements, 4)

    # Source often is 1D in practice, but logically must be Replicated
    layout_1d = _build_layout(mesh, (Replicate(),), 1)

    output_layout = op.infer_layout((layout_4d, layout_4d, layout_1d))

    expected_map = (-1, -1, -1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_masked_scatter_success_mixed_ranks():
    """
    Feature: MaskedScatter with mixed rank inputs
    Description: Input/Mask are 3D, Source is 1D (typical PyTorch usage)
    Expectation: Success, output follows input layout
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # Input: (B, S, H)
    layout_3d = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    # Source: Flattened 1D
    layout_1d = _build_layout(mesh, (Replicate(),), 1)

    output_layout = op.infer_layout((layout_3d, layout_3d, layout_1d))

    expected_map = (-1, -1, -1)
    assert output_layout.to_dict()["tensor_map"] == expected_map


def test_masked_scatter_fail_shard_last_dim():
    """
    Feature: MaskedScatter fail on last dim sharding
    Description: Sharding the last dimension (common in Tensor Parallelism)
    Expectation: ValueError raised
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # Input sharded on last dim (dim 1)
    input_placements = (Replicate(), Shard(1))
    input_layout = _build_layout(mesh, input_placements, 2)

    mask_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    source_layout = _build_layout(mesh, (Replicate(),), 1)

    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((input_layout, mask_layout, source_layout))


def test_masked_scatter_success_1d_mesh():
    """
    Feature: MaskedScatter on 1D Mesh
    Description: Verify logic works on a simple 1D device mesh
    Expectation: Success if replicated
    """
    mesh = init_device_mesh(mesh_shape=(8,), alias_name=("dp",))

    layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    source = _build_layout(mesh, (Replicate(),), 1)

    output_layout = op.infer_layout((layout, layout, source))
    assert output_layout.to_dict()["tensor_map"] == (-1, -1)


def test_masked_scatter_success_3d_mesh():
    """
    Feature: MaskedScatter on 3D Mesh
    Description: Verify logic works on complex 3D device mesh
    Expectation: Success if replicated
    """
    mesh = init_device_mesh(mesh_shape=(2, 2, 2), alias_name=("dp", "tp", "pp"))

    layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    source = _build_layout(mesh, (Replicate(),), 1)

    output_layout = op.infer_layout((layout, layout, source))
    assert output_layout.to_dict()["tensor_map"] == (-1, -1)


def test_masked_scatter_fail_multiple_sharded():
    """
    Feature: MaskedScatter with multiple sharded inputs
    Description: Both Input and Mask are sharded
    Expectation: ValueError raised (should fail fast on the first one)
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # Both sharded
    input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    mask_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    source_layout = _build_layout(mesh, (Replicate(),), 1)

    # Should catch Input 0 first
    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((input_layout, mask_layout, source_layout))


def test_masked_scatter_ignore_extra_args():
    """
    Feature: MaskedScatter ignoring extra args
    Description: Verify that passing extra arguments doesn't break inference
    Expectation: Success
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    source = _build_layout(mesh, (Replicate(),), 1)

    # Pass arbitrary extra args
    output_layout = op.infer_layout(
        (layout, layout, source),
        extra_args={"some_arg": 123}
    )

    assert output_layout.to_dict()["tensor_map"] == (-1, -1)

def test_masked_scatter_success_scalar_inputs():
    """
    Feature: MaskedScatter with scalar inputs
    Description: All inputs are 0-D scalars.
    Expectation: Success, output is 0-D scalar layout (empty tensor_map).
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # 0-D tensor layout
    scalar_layout = _build_layout(mesh, (), 0)

    # input(scalar), mask(scalar), source(scalar)
    output_layout = op.infer_layout((scalar_layout, scalar_layout, scalar_layout))

    # Expect empty tuple for scalar tensor_map
    assert output_layout.to_dict()["tensor_map"] == ()


def test_masked_scatter_success_1d_tensors():
    """
    Feature: MaskedScatter with 1D tensors
    Description: All inputs are 1D vectors, fully replicated.
    Expectation: Success.
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # 1D Replicated layout
    layout_1d = _build_layout(mesh, (Replicate(),), 1)

    output_layout = op.infer_layout((layout_1d, layout_1d, layout_1d))

    assert output_layout.to_dict()["tensor_map"] == (-1,)


def test_masked_scatter_success_source_broadcast():
    """
    Feature: MaskedScatter with source broadcast
    Description: Input and Mask are 3D, but Source is 1D (Common PyTorch usage).
    Expectation: Success, output layout matches Input layout.
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # Input/Mask: (B, S, H) Replicated
    layout_3d = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 3)
    # Source: Flattened 1D Replicated
    layout_source = _build_layout(mesh, (Replicate(),), 1)

    output_layout = op.infer_layout((layout_3d, layout_3d, layout_source))

    # Output should match input (3D)
    assert output_layout.to_dict()["tensor_map"] == (-1, -1, -1)


def test_masked_scatter_success_high_dim():
    """
    Feature: MaskedScatter with high-dimensional tensors
    Description: 5D tensors (e.g. video batch), fully replicated.
    Expectation: Success.
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # 5D Replicated
    placements = (Replicate(),) * 5
    layout_5d = _build_layout(mesh, placements, 5)

    output_layout = op.infer_layout((layout_5d, layout_5d, layout_5d))

    assert output_layout.to_dict()["tensor_map"] == (-1, -1, -1, -1, -1)


def test_masked_scatter_fail_1d_mesh_sharded():
    """
    Feature: MaskedScatter failure on 1D Mesh
    Description: Sharding on a simple 1D mesh.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(mesh_shape=(8,), alias_name=("dp",))

    # Input sharded
    input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    replicated_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((input_layout, replicated_layout, replicated_layout))


def test_masked_scatter_fail_3d_mesh_sharded_inner():
    """
    Feature: MaskedScatter failure on 3D Mesh inner dimension
    Description: Sharding on the 'tp' dimension of a (dp, tp, pp) mesh.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(mesh_shape=(2, 2, 2), alias_name=("dp", "tp", "pp"))

    # Shard on 'tp' (which is dimension 1 of the mesh, mapped to tensor dim 0)
    # Note: _build_layout map logic depends on implementation, but Shard(0) means tensor dim 0 is sharded.
    input_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    replicated_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)

    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((input_layout, replicated_layout, replicated_layout))


def test_masked_scatter_fail_last_dim_sharding():
    """
    Feature: MaskedScatter failure on last dimension sharding
    Description: Input is sharded on the last dimension (common in Tensor Parallelism).
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # Input: (Replicate, Shard(1)) -> Sharded on last dim
    input_placements = (Replicate(), Shard(1))
    input_layout = _build_layout(mesh, input_placements, 2)

    layout_ok = _build_layout(mesh, (Replicate(), Replicate()), 2)

    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((input_layout, layout_ok, layout_ok))


def test_masked_scatter_fail_sharded_mask_only():
    """
    Feature: MaskedScatter failure when only Mask is sharded
    Description: Input is Replicated, but Mask is sharded.
    Expectation: ValueError raised identifying Input 1 (Mask).
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    # Mask sharded
    mask_layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
    source_layout = _build_layout(mesh, (Replicate(),), 1)

    with pytest.raises(ValueError, match=r"(?s)Input 1 .* is sharded"):
        op.infer_layout((input_layout, mask_layout, source_layout))


def test_masked_scatter_fail_multiple_bad_inputs():
    """
    Feature: MaskedScatter failure order
    Description: Input and Source are both sharded.
    Expectation: ValueError raised for Input 0 (Fail Fast).
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    # Both sharded
    bad_layout_2d = _build_layout(mesh, (Shard(0), Replicate()), 2)
    bad_layout_1d = _build_layout(mesh, (Shard(0),), 1)

    # Should report Input 0 error
    with pytest.raises(ValueError, match=r"(?s)Input 0 .* is sharded"):
        op.infer_layout((bad_layout_2d, bad_layout_2d, bad_layout_1d))


def test_masked_scatter_ignore_extra_args2():
    """
    Feature: MaskedScatter robustness with extra_args
    Description: Verify that passing unknown extra arguments does not cause failure.
    Expectation: Success.
    """
    mesh = init_device_mesh(mesh_shape=(2, 4), alias_name=("dp", "mp"))

    layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    source = _build_layout(mesh, (Replicate(),), 1)

    # Passing dummy extra_args
    output_layout = op.infer_layout(
        (layout, layout, source),
        extra_args={"some_flag": True, "value": 100}
    )

    assert output_layout.to_dict()["tensor_map"] == (-1, -1)
