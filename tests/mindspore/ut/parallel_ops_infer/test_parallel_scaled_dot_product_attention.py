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
"""parallel_scaled_dot_product_attention unit test"""

import pytest
from hyper_parallel import Layout, init_device_mesh
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_scaled_dot_product_attention import (
    ScaledDotProductAttentionDistributedOp,
)

op = ScaledDotProductAttentionDistributedOp("scaled_dot_product_attention")


def run_scenario(scenario_name, q_layout, k_layout, v_layout, expected_out_map):
    """Infer layout and verify attention output tensor_map."""
    output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])
    assert output_layout.to_dict()["tensor_map"] == expected_out_map, \
        f"Scenario '{scenario_name}' failed. " \
        f"Expected {expected_out_map}, got {output_layout.to_dict()['tensor_map']}"


def test_sdpa_no_parallel_1():
    """
    Feature: Layout inference with no parallelism.
    Description: All dimensions replicated on a 3D mesh, BNSD 4D input.
    Expectation: Output tensor_map is all -1.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
    k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
    v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)

    run_scenario("No Parallelism", q_layout, k_layout, v_layout, (-1, -1, -1, -1))


def test_sdpa_data_parallel_2():
    """
    Feature: Layout inference with data parallelism.
    Description: BNSD batch dimension sharded on dp axis.
    Expectation: Output batch dimension remains sharded on dp.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    placements = (Shard(0), Replicate(), Replicate())
    q_layout = _build_layout(mesh, placements, 4)
    k_layout = _build_layout(mesh, placements, 4)
    v_layout = _build_layout(mesh, placements, 4)

    run_scenario("Data Parallel (DP)", q_layout, k_layout, v_layout, (2, -1, -1, -1))


def test_sdpa_head_parallel_3():
    """
    Feature: Layout inference with head parallelism.
    Description: BNSD head dimension sharded on mp axis.
    Expectation: Output head dimension remains sharded on mp.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    placements = (Replicate(), Replicate(), Shard(1))
    q_layout = _build_layout(mesh, placements, 4)
    k_layout = _build_layout(mesh, placements, 4)
    v_layout = _build_layout(mesh, placements, 4)

    run_scenario("Head Parallel (MP)", q_layout, k_layout, v_layout, (-1, 0, -1, -1))


def test_sdpa_sequence_parallel_4():
    """
    Feature: Layout inference with sequence parallelism.
    Description: BNSD query sequence sharded on sp axis, KV replicated.
    Expectation: Output sequence dimension sharded on sp.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Replicate(), Shard(2), Replicate()), 4)
    k_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)
    v_layout = _build_layout(mesh, (Replicate(), Replicate(), Replicate()), 4)

    run_scenario("Sequence Parallel (SP)", q_layout, k_layout, v_layout, (-1, -1, 1, -1))


def test_sdpa_hybrid_dp_mp_5():
    """
    Feature: Layout inference with hybrid DP + MP.
    Description: BNSD batch on dp, head on mp.
    Expectation: Both dimensions remain sharded.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    placements = (Shard(0), Replicate(), Shard(1))
    q_layout = _build_layout(mesh, placements, 4)
    k_layout = _build_layout(mesh, placements, 4)
    v_layout = _build_layout(mesh, placements, 4)

    run_scenario("Hybrid DP + MP", q_layout, k_layout, v_layout, (2, 0, -1, -1))


def test_sdpa_hybrid_dp_sp_mp_6():
    """
    Feature: Layout inference with full hybrid DP + SP + MP.
    Description: BNSD with all three parallel strategies, KV only dp+mp.
    Expectation: Output preserves dp+sp+mp sharding from query.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
    k_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)
    v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)

    run_scenario("Hybrid DP + SP + MP", q_layout, k_layout, v_layout, (2, 0, 1, -1))


def test_sdpa_kv_different_layout_7():
    """
    Feature: Layout inference with different KV sharding.
    Description: Key has sp sharding but Value does not. Layout inference does not
        validate KV consistency; that is deferred to get_expand_impl at runtime.
    Expectation: Layout inference succeeds based on query layout.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
    k_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
    v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1)), 4)

    run_scenario("KV Different Layout", q_layout, k_layout, v_layout, (2, 0, 1, -1))


def test_sdpa_dp_mp_2d_mesh_8():
    """
    Feature: Layout inference with DP + MP on a 2D mesh.
    Description: BNSD batch on dp, head on mp, 2D mesh (4, 2).
    Expectation: Output preserves sharding on both dimensions.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"), init_backend=False,
    )
    placements = (Shard(0), Shard(1))
    q_layout = _build_layout(mesh, placements, 4)
    k_layout = _build_layout(mesh, placements, 4)
    v_layout = _build_layout(mesh, placements, 4)

    run_scenario("DP + MP on 2D Mesh", q_layout, k_layout, v_layout, (1, 0, -1, -1))


def test_sdpa_dp_sp_2d_mesh_9():
    """
    Feature: Layout inference with DP + SP on a 2D mesh.
    Description: BNSD batch on dp, sequence on sp, head replicated.
    Expectation: Output preserves batch and sequence sharding.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "sp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0), Shard(2)), 4)
    k_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)
    v_layout = _build_layout(mesh, (Shard(0), Replicate()), 4)

    run_scenario("DP + SP on 2D Mesh", q_layout, k_layout, v_layout, (1, -1, 0, -1))


def test_sdpa_single_device_10():
    """
    Feature: Layout inference with single device mesh.
    Description: Mesh size is 1, all dimensions replicated.
    Expectation: Output tensor_map is all -1.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(1,),
        mesh_dim_names=("dp",), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Replicate(),), 4)
    k_layout = _build_layout(mesh, (Replicate(),), 4)
    v_layout = _build_layout(mesh, (Replicate(),), 4)

    run_scenario("Single Device", q_layout, k_layout, v_layout, (-1, -1, -1, -1))


def test_sdpa_large_world_size_11():
    """
    Feature: Layout inference with large world size.
    Description: 32 devices on a 4D mesh (2,2,2,4) with dp+sp+mp+pp axes.
    Expectation: Output scales correctly with 4D mesh.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2, 4),
        mesh_dim_names=("dp", "sp", "mp", "pp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1), Replicate()), 4)
    k_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)
    v_layout = _build_layout(mesh, (Shard(0), Replicate(), Shard(1), Replicate()), 4)

    run_scenario(
        "Large World Size (32 devices)", q_layout, k_layout, v_layout, (3, 1, 2, -1)
    )


def test_sdpa_single_output_layout_12():
    """
    Feature: Layout inference returns a single output layout.
    Description: Unlike FA which returns 4 outputs (attention_out, softmax_max,
        softmax_sum, softmax_out), SDPA returns only attention_out.
    Expectation: Output is a single Layout, not a tuple.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"), init_backend=False,
    )
    placements = (Shard(0), Shard(1))
    q_layout = _build_layout(mesh, placements, 4)
    k_layout = _build_layout(mesh, placements, 4)
    v_layout = _build_layout(mesh, placements, 4)

    output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])

    assert isinstance(output_layout, Layout), "Should return a single Layout"
    assert output_layout.to_dict()["tensor_map"] == (1, 0, -1, -1)


def test_sdpa_head_dim_always_replicated_13():
    """
    Feature: Layout inference never shards head_dim.
    Description: Verify the last dimension (D) is always -1 in the output,
        since sharding the head dimension size would break attention computation.
    Expectation: Output head_dim (dim 3) is -1.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)
    k_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)
    v_layout = _build_layout(mesh, (Shard(0), Shard(1)), 4)

    output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])

    tensor_map = output_layout.to_dict()["tensor_map"]
    assert tensor_map[3] == -1, "Head dim (dim 3) should not be sharded"


def test_sdpa_3d_input_head_parallel_14():
    """
    Feature: Layout inference for 3D input [N, S, D].
    Description: 3D tensor without batch dimension, head on mp.
    Expectation: Output preserves 3D sharding structure.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(8,),
        mesh_dim_names=("mp",), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0),), 3)
    k_layout = _build_layout(mesh, (Shard(0),), 3)
    v_layout = _build_layout(mesh, (Shard(0),), 3)

    run_scenario("3D Input Head Parallel", q_layout, k_layout, v_layout, (0, -1, -1))


def test_sdpa_3d_input_sp_mp_15():
    """
    Feature: Layout inference for 3D input with SP + MP.
    Description: 3D tensor [N, S, D] with head on mp and seq on sp.
    Expectation: Output preserves head and sequence sharding.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(1), Shard(0)), 3)
    k_layout = _build_layout(mesh, (Replicate(), Shard(0)), 3)
    v_layout = _build_layout(mesh, (Replicate(), Shard(0)), 3)

    run_scenario("3D Input SP + MP", q_layout, k_layout, v_layout, (0, 1, -1))


def test_sdpa_output_matches_query_layout_16():
    """
    Feature: Output layout always matches query layout.
    Description: Verify that the output tensor_map is identical to query tensor_map
        regardless of KV sharding patterns.
    Expectation: Output tensor_map equals query tensor_map.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(0), Shard(2), Shard(1)), 4)
    k_layout = _build_layout(mesh, (Shard(0), Replicate(), Replicate()), 4)
    v_layout = _build_layout(mesh, (Replicate(), Replicate(), Shard(1)), 4)

    output_layout = op.infer_layout((q_layout, k_layout, v_layout), [])

    assert output_layout.to_dict()["tensor_map"] == q_layout.to_dict()["tensor_map"], \
        "Output tensor_map should match query tensor_map"


def test_sdpa_sp_only_on_query_17():
    """
    Feature: Layout inference with SP only applied to query.
    Description: BNSD with query seq on sp, KV fully replicated. This is the
        standard SP pattern where each device computes attention on its local
        Q chunk against the full KV.
    Expectation: Output preserves query sequence sharding.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(8,),
        mesh_dim_names=("sp",), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(2),), 4)
    k_layout = _build_layout(mesh, (Replicate(),), 4)
    v_layout = _build_layout(mesh, (Replicate(),), 4)

    run_scenario("SP Only on Query", q_layout, k_layout, v_layout, (-1, -1, 0, -1))


def test_sdpa_sp_mp_2d_mesh_18():
    """
    Feature: Layout inference with SP + MP on a 2D mesh.
    Description: BNSD with query seq on sp, head on mp, KV only mp.
    Expectation: Output preserves both sp and mp sharding.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("sp", "mp"), init_backend=False,
    )
    q_layout = _build_layout(mesh, (Shard(2), Shard(1)), 4)
    k_layout = _build_layout(mesh, (Replicate(), Shard(1)), 4)
    v_layout = _build_layout(mesh, (Replicate(), Shard(1)), 4)

    run_scenario("SP + MP on 2D Mesh", q_layout, k_layout, v_layout, (-1, 0, 1, -1))


def test_sdpa_error_query_layout_none_19():
    """
    Feature: Error handling when query layout is None.
    Description: Pass None as query layout.
    Expectation: ValueError raised.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(8,),
        mesh_dim_names=("dp",), init_backend=False,
    )
    k_layout = _build_layout(mesh, (Replicate(),), 4)
    v_layout = _build_layout(mesh, (Replicate(),), 4)

    with pytest.raises(ValueError, match="Query layout cannot be None"):
        op.infer_layout((None, k_layout, v_layout), [])
