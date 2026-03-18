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
"""parallel_gatherd test"""
import pytest
from hyper_parallel.core.dtensor.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_gather import GatherDDistributedOp


@pytest.fixture(params=["GatherD"], name="op")
def fixture_op(request):
    """
    Fixture to test'GatherD' operations.
    """
    return GatherDDistributedOp(request.param)


def test_gatherd_data_parallel_dim0(op):
    """
    Feature: Data Parallel for GatherD
    Description: Input is sharded on the gather dimension, while index is replicated.
    Expectation: Output layout becomes fully replicated and enters partial sum state.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: [16, 32] -> shard dim0 with dp
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [16, 32] -> fully replicated
    index_placements = (Replicate(), Replicate())
    index_layout = _build_layout(mesh, index_placements, 2)
    dim = 0
    output_layout = op.infer_layout((input_layout, None, index_layout), [dim])
    # Expected output: fully replicated tensor_map with partial sum state
    expected_map = (-1, -1)
    assert output_layout.tensor_map == expected_map, \
        f"Data parallel dim0 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
    assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
    assert output_layout.partial == ['sum', None], \
        f"Expected partial ['sum', None], got {output_layout.partial}"
    # Expand implementation should be provided when the gather dimension is sharded
    impl = op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
    assert impl is not None, "Data parallel dim0 should have expand implementation"
    assert callable(impl), "Returned impl should be a callable function"

def test_gatherd_data_parallel_dim1(op):
    """
    Feature: GatherD dim-axis sharding inference
    Description: Input is sharded on the gather dimension, while index is replicated.
    Expectation: Output layout becomes fully replicated and enters partial sum state.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: [batch, seq] -> shard dim1 with mp
    input_placements = (Replicate(), Shard(1))
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [batch, seq] -> fully replicated
    index_placements = (Replicate(), Replicate())
    index_layout = _build_layout(mesh, index_placements, 2)
    # Gather on batch dim while index stays replicated
    dim = 1
    output_layout = op.infer_layout((input_layout, None, index_layout), [dim])
    expected_map = (-1, -1)
    assert output_layout.tensor_map == expected_map, \
        f"Data parallel dim1 inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
    assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
    assert output_layout.partial == [None, 'sum'], \
        f"Expected partial [None, 'sum'], got {output_layout.partial}"
    impl = op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
    assert impl is not None, "Data parallel dim1 should have expand implementation"
    assert callable(impl), "Returned impl should be a callable function"

def test_gatherd_input_both_shard_replicate_index(op):
    """
    Feature: GatherD multi-dim sharding validation
    Description: Input is sharded on both batch and sequence dimensions, while index is replicated.
    Expectation: infer_layout should reject mismatched non-dim sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )

    # input: [batch, seq] -> shard batch with dp and seq with mp
    input_placements = (Shard(0), Shard(1))
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [batch, seq] -> fully replicated
    index_placements = (Replicate(), Replicate())
    index_layout = _build_layout(mesh, index_placements, 2)
    # Gather on batch dim while input is sharded on both tensor dimensions
    dim = 0
    with pytest.raises(ValueError, match="same sharding on non-dim axis 1"):
        op.infer_layout((input_layout, None, index_layout), [dim])


def test_gatherd_column_parallel(op):
    """
    Feature: GatherD cross-axis sharding validation
    Description: Input is sharded on the gather dimension, and index is sharded on another dimension.
    Expectation: infer_layout should reject mismatched non-dim sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: [16, 32] -> shard dim0 with dp
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [16, 32] -> shard dim1 with mp
    index_placements = (Replicate(), Shard(1))
    index_layout = _build_layout(mesh, index_placements, 2)
    dim = 0
    with pytest.raises(ValueError, match="same sharding on non-dim axis 1"):
        op.infer_layout((input_layout, None, index_layout), [dim])

def test_gatherd_row_parallel(op):
    """
    Feature: Enhanced Model Parallel for GatherD
    Description: Both input and index sharded identically on dim axis.
    Expectation: Output follows the sharded batch mapping and has Partial Sum state.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: [16, 32] -> shard dim0 with dp
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [16, 32] -> shard dim0 with mp
    index_placements = (Replicate(), Shard(0))
    index_layout = _build_layout(mesh, index_placements, 2)
    dim = 0
    output_layout = op.infer_layout((input_layout, None, index_layout), [dim])
    # Expected output: sharded on dim0 and replicated on dim1, with Partial Sum
    expected_map = (0, -1)
    assert output_layout.tensor_map == expected_map, \
        f"Row parallel expand failed. Expected {expected_map}, got {output_layout.tensor_map}"
    assert output_layout.is_partial(), "Row parallel should generate partial state"
    assert output_layout.partial == ['sum', None], \
        f"Expected partial ['sum', None], got {output_layout.partial}"
    # Expand implementation should NOT be None for enhanced MP
    impl = op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
    assert impl is not None, "Row parallel should have expand implementation"
    assert callable(impl), "Returned impl should be a callable function"

def test_gatherd_input_shard_dp_index_both_shard_conflict(op):
    """
    Feature: GatherD conflicting shard and partial inference
    Description: Input is sharded on batch with dp, while index is sharded on both batch and
                 sequence. This makes the output require dp for both sharding and partial reduction.
    Expectation: infer_layout should raise ValueError: "Partial dim must be replicate."
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: [16, 32] -> shard dim0 with dp
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [16, 32] -> shard dim0 with dp and shard dim1 with mp
    index_placements = (Shard(0), Replicate())
    index_layout = _build_layout(mesh, index_placements, 2)
    # Gather on batch dim, so index dim0 sharding tries to introduce partial on dp,
    # which already shards the output batch dimension.
    dim = 0
    with pytest.raises(ValueError, match="Partial dim must be replicate."):
        op.infer_layout((input_layout, None, index_layout), [dim])

def test_gatherd_input_multi_shard_with_matched_index_non_dim_shard(op):
    """
    Feature: GatherD multi-shard inference with matched non-dim axis
    Description: Input is sharded on both batch and sequence, while index matches the input sharding
                 on the non-gather axis and stays replicated on the gather axis.
    Expectation: Output follows the index layout and enters partial sum state.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: [16, 32] -> shard dim0 with dp and dim1 with mp
    input_placements = (Shard(0), Shard(1))
    input_layout = _build_layout(mesh, input_placements, 2)
    # index: [16, 32] -> replicate dim0 and shard dim1 with mp
    index_placements = (Replicate(), Shard(1))
    index_layout = _build_layout(mesh, index_placements, 2)
    # Gather on batch dim. Non-dim axis sharding matches between input and index.
    dim = 0
    output_layout = op.infer_layout((input_layout, None, index_layout), [dim])
    expected_map = (-1, 0)
    assert output_layout.tensor_map == expected_map, \
        f"Matched non-dim shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
    assert output_layout.is_partial(), "Sharded gather dimension should generate partial state"
    assert output_layout.partial == ['sum', None], \
        f"Expected partial ['sum', None], got {output_layout.partial}"
    impl = op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
    assert impl is not None, "Matched non-dim shard case should have expand implementation"
    assert callable(impl), "Returned impl should be a callable function"

def test_gatherd_3d_mesh_input_multi_shard(op):
    """
    Feature: GatherD 3D mesh sharding inference
    Description: Mesh shape (2,2,2) with dims ("dp","tp","cp").
                 Input is sharded on the gather axis by tp, and index is sharded on the same
                 gather axis by cp, while non-dim axes remain replicated.
    Expectation: Output follows the index layout and enters partial sum state.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "cp"),
        init_backend=False
    )
    # input: [16, 32, 64] -> shard dim1 with tp, replicate other dims
    input_placements = (Replicate(), Shard(1), Replicate())
    input_layout = _build_layout(mesh, input_placements, 3)  # tensor_map: (-1, 1, -1)
    # index: [16, 32, 64] -> shard dim1 on cp, replicate other dims
    index_placements = (Replicate(), Replicate(), Shard(1))
    index_layout = _build_layout(mesh, index_placements, 3)  # tensor_map: (-1, 0, -1)
    dim = 1
    output_layout = op.infer_layout((input_layout, None, index_layout), [dim])
    # Expected output map: inherit index sharding on dim1
    expected_map = (-1, 0, -1)
    assert output_layout.tensor_map == expected_map, \
        f"3D mesh multi-shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
    assert output_layout.is_partial(), "3D mesh with sharded index should generate partial state"
    assert output_layout.partial == [None, 'sum', None], \
        f"Expected partial [None, 'sum', None], got {output_layout.partial}"
    # Expand implementation should be provided when the gather dimension is sharded
    impl = op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
    assert impl is not None, "3D mesh with sharded index should have expand implementation"
    assert callable(impl), "Returned impl should be a callable function"

def test_gatherd_3d_mesh_matched_non_dim_shard(op):
    """
    Feature: GatherD 3D mesh sharding inference
    Description: Mesh shape (2,2,2) with dims ("dp","tp","cp").
                 Input and index match on non-gather axes, while the gather axis is sharded on
                 different mesh axes.
    Expectation: Output follows the index layout and enters partial sum state.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "cp"),
        init_backend=False
    )
    # input: [16, 32, 64] -> shard dim0 with dp, dim1 with tp, replicate dim2
    input_placements = (Shard(0), Shard(1), Replicate())
    input_layout = _build_layout(mesh, input_placements, 3)  # tensor_map: (2, 1, -1)
    # index: [16, 32, 64] -> shard dim0 with dp, shard dim1 with cp, replicate dim2
    index_placements = (Shard(0), Replicate(), Shard(1))
    index_layout = _build_layout(mesh, index_placements, 3)  # tensor_map: (2, 0, -1)
    dim = 1
    output_layout = op.infer_layout((input_layout, None, index_layout), [dim])
    # Expected output map: inherit index sharding on dim0 and dim1
    expected_map = (2, 0, -1)
    assert output_layout.tensor_map == expected_map, \
        f"3D mesh matched non-dim shard inference failed. Expected {expected_map}, got {output_layout.tensor_map}"
    assert output_layout.is_partial(), "3D mesh with sharded gather axis should generate partial state"
    assert output_layout.partial == [None, 'sum', None], \
        f"Expected partial [None, 'sum', None], got {output_layout.partial}"
    # Expand implementation should be provided when the gather dimension is sharded
    impl = op.get_expand_impl(None, output_layout, (input_layout, None, index_layout), [dim])
    assert impl is not None, "3D mesh matched non-dim shard case should have expand implementation"
    assert callable(impl), "Returned impl should be a callable function"

def test_gatherd_rank_mismatch_error(op):
    """
    Feature: GatherD layout inference error handling
    Description: Input and index have different ranks.
    Expectation: Raise ValueError.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    # input: 3D
    input_placements = (Shard(0), Replicate(), Replicate())
    input_layout = _build_layout(mesh, input_placements, 3)
    # index: 2D (mismatch!)
    index_placements = (Shard(0), Replicate())
    index_layout = _build_layout(mesh, index_placements, 2)
    dim = 1
    with pytest.raises(ValueError, match="same number of dimensions"):
        op.infer_layout((input_layout, None, index_layout), [dim])

def test_gatherd_invalid_layouts_error(op):
    """
    Feature: GatherD input layout validation
    Description: Fewer than the required three layouts are provided.
    Expectation: Raise ValueError.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    dim = 0
    with pytest.raises(ValueError, match="requires 3 layouts"):
        op.infer_layout((input_layout, None, ), [dim])


def test_gatherd_missing_dim_arg_error(op):
    """
    Feature: GatherD extra argument validation
    Description: The required dim argument is not provided.
    Expectation: Raise ValueError.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    input_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    index_layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
    with pytest.raises(ValueError, match="requires 1 extra arg"):
        op.infer_layout((input_layout, None, index_layout), [])


def test_gatherd_invalid_dim_error(op):
    """
    Feature: GatherD layout inference error handling
    Description: Dim value is out of valid range.
    Expectation: Raise ValueError.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    input_placements = (Shard(0), Replicate())
    input_layout = _build_layout(mesh, input_placements, 2)
    index_layout = _build_layout(mesh, input_placements, 2)
    dim = 5  # Invalid for 2D tensor
    with pytest.raises(ValueError, match="out of valid range"):
        op.infer_layout((input_layout, None, index_layout), [dim])
