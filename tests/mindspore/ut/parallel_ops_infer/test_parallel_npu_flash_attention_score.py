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
"""parallel_npu_fusion_attention unit test"""

import pytest
from hyper_parallel.core.dtensor import _build_layout
from hyper_parallel import init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_npu_flash_attention_score import FlashAttentionScoreDistributedOp

# Initialize the operator
op = FlashAttentionScoreDistributedOp("npu_fusion_attention")


def run_scenario(scenario_name, mesh, q_placements, k_placements, v_placements,
                 ndim, expected_out_map, extra_args):
    """Infer layout and verify attention output tensor_map."""
    q_layout = _build_layout(mesh, q_placements, ndim)
    k_layout = _build_layout(mesh, k_placements, ndim)
    v_layout = _build_layout(mesh, v_placements, ndim)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), extra_args)
    attention_out_layout = output_layouts[0]
    assert attention_out_layout.to_dict()["tensor_map"] == expected_out_map, \
        f"Scenario '{scenario_name}' failed. " \
        f"Expected {expected_out_map}, got {attention_out_layout.to_dict()['tensor_map']}"


def test_flash_attention_no_parallel_1():
    """
    Feature: Layout inference with no parallelism.
    Description: All dimensions replicated on a 3D mesh.
    Expectation: Output tensor_map is all -1.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Replicate(), Replicate(), Replicate())

    run_scenario(
        "No Parallelism",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(-1, -1, -1),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_data_parallel_2():
    """
    Feature: Layout inference with data parallelism.
    Description: BSH batch dimension sharded on dp axis.
    Expectation: Output batch dimension remains sharded on dp.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate(), Replicate())

    run_scenario(
        "Data Parallel (DP)",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(2, -1, -1),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_head_parallel_3():
    """
    Feature: Layout inference with head parallelism.
    Description: BSH hidden dimension sharded on mp axis.
    Expectation: Output hidden dimension remains sharded on mp.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Replicate(), Replicate(), Shard(2))

    run_scenario(
        "Head Parallel (MP)",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(-1, -1, 0),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_sequence_parallel_4():
    """
    Feature: Layout inference with sequence parallelism.
    Description: BSH query sequence sharded on sp axis, KV not sharded.
    Expectation: Output sequence dimension sharded on sp.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    q_placements = (Replicate(), Shard(1), Replicate())
    kv_placements = (Replicate(), Replicate(), Replicate())

    run_scenario(
        "Sequence Parallel (SP)",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=3,
        expected_out_map=(-1, 1, -1),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_hybrid_dp_mp_5():
    """
    Feature: Layout inference with hybrid DP + MP.
    Description: BSH batch on dp, hidden on mp.
    Expectation: Both dimensions remain sharded.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate(), Shard(2))

    run_scenario(
        "Hybrid DP + MP",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(2, -1, 0),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_hybrid_dp_sp_mp_6():
    """
    Feature: Layout inference with full hybrid DP + SP + MP.
    Description: BSH with all three parallel strategies, KV only dp+mp.
    Expectation: Output preserves dp+sp+mp sharding from query.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1), Shard(2))
    kv_placements = (Shard(0), Replicate(), Shard(2))

    run_scenario(
        "Hybrid DP + SP + MP",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=3,
        expected_out_map=(2, 1, 0),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_kv_different_layout_7():
    """
    Feature: Layout inference with different KV sharding.
    Description: Key has sp sharding but Value does not. Layout inference does not
        validate KV consistency; that is deferred to get_expand_impl at runtime.
    Expectation: Layout inference succeeds based on query layout.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1), Shard(2))
    k_placements = (Shard(0), Shard(1), Shard(2))
    v_placements = (Shard(0), Replicate(), Shard(2))

    run_scenario(
        "KV Different Layout",
        mesh, q_placements, k_placements, v_placements,
        ndim=3,
        expected_out_map=(2, 1, 0),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_bnsd_layout_8():
    """
    Feature: Layout inference for BNSD input layout.
    Description: 4D tensor with dp on batch, mp on head.
    Expectation: Output preserves 4D sharding structure.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(1))

    run_scenario(
        "BNSD Layout",
        mesh, placements, placements, placements,
        ndim=4,
        expected_out_map=(1, 0, -1, -1),
        extra_args=[16, "BNSD"]
    )


def test_flash_attention_sbh_layout_9():
    """
    Feature: Layout inference for SBH input layout.
    Description: Sequence-first 3D layout with head parallelism on mp.
    Expectation: Output preserves SBH dimension order with mp sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("mp",),
        init_backend=False
    )
    placements = (Shard(2),)

    run_scenario(
        "SBH Layout",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(-1, -1, 0),
        extra_args=[16, "SBH"]
    )


def test_flash_attention_bsnd_layout_10():
    """
    Feature: Layout inference for BSND input layout.
    Description: 4D tensor with dp on batch, mp on head (dim 2).
    Expectation: Output preserves 4D sharding structure.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    run_scenario(
        "BSND Layout",
        mesh, placements, placements, placements,
        ndim=4,
        expected_out_map=(1, -1, 0, -1),
        extra_args=[16, "BSND"]
    )


def test_flash_attention_sparse_mode_0_14():
    """
    Feature: Layout inference with sparse_mode=0 in extra_args.
    Description: defaultMask mode passed via extra_args.
    Expectation: Layout inference succeeds normally.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate(), Replicate())

    run_scenario(
        "Sparse Mode 0",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(2, -1, -1),
        extra_args=[16, "BSH", 0]
    )


def test_flash_attention_sparse_mode_2_15():
    """
    Feature: Layout inference with sparse_mode=2 and sequence parallelism.
    Description: leftUpCausal mode with query seq sharded on sp.
    Expectation: Layout inference succeeds with seq sharding preserved.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("sp",),
        init_backend=False
    )
    q_placements = (Shard(1),)
    kv_placements = (Replicate(),)

    run_scenario(
        "Sparse Mode 2 (Causal + SP)",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=3,
        expected_out_map=(-1, 0, -1),
        extra_args=[16, "BSH", 2]
    )


def test_flash_attention_multiple_inputs_same_layout_16():
    """
    Feature: Layout inference returns 4 output layouts.
    Description: Q/K/V all have the same dp+mp sharding.
    Expectation: Returns exactly 4 layouts; attention_out matches query.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate(), Shard(2))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "BSH"])

    assert len(output_layouts) == 4, "Should return 4 output layouts"
    assert output_layouts[0].to_dict()["tensor_map"] == (2, -1, 0)


def test_flash_attention_output_layouts_bsh_17():
    """
    Feature: All 4 output layouts for BSH with dp+mp.
    Description: Verify attention_out, softmax_max, softmax_sum, and softmax_out
        layouts for BSH input with dp on batch and mp on hidden.
    Expectation: attention_out matches query; softmax_max/sum are 4D (B,N,S,8);
        softmax_out is a fully replicated scalar placeholder with empty tensor_map.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "BSH"])

    attention_out, softmax_max, softmax_sum, softmax_out = output_layouts

    assert attention_out.to_dict()["tensor_map"] == q_layout.to_dict()["tensor_map"]
    assert attention_out.to_dict()["tensor_map"] == (1, -1, 0)

    assert len(softmax_max.to_dict()["tensor_map"]) == 4
    assert len(softmax_sum.to_dict()["tensor_map"]) == 4
    assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
    assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)

    assert softmax_out.to_dict()["tensor_map"] == ()


def test_flash_attention_multi_dimensional_mesh_18():
    """
    Feature: Layout inference with 3D device mesh.
    Description: BSH with dp+sp+mp on a (2,2,2) mesh, all inputs fully sharded.
    Expectation: Output preserves all three sharding axes.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(1), Shard(2))

    run_scenario(
        "Multi-dimensional Mesh",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(2, 1, 0),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_single_device_19():
    """
    Feature: Layout inference with single device mesh.
    Description: Mesh size is 1, all dimensions replicated.
    Expectation: Output tensor_map is all -1.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(1,),
        mesh_dim_names=("dp",),
        init_backend=False
    )
    placements = (Replicate(),)

    run_scenario(
        "Single Device",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(-1, -1, -1),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_large_world_size_20():
    """
    Feature: Layout inference with large world size.
    Description: 32 devices on a 4D mesh (2,2,2,4) with dp+sp+mp+pp axes.
    Expectation: Output scales correctly with 4D mesh.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2, 4),
        mesh_dim_names=("dp", "sp", "mp", "pp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1), Shard(2), Replicate())
    kv_placements = (Shard(0), Replicate(), Shard(2), Replicate())

    run_scenario(
        "Large World Size (32 devices)",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=4,
        expected_out_map=(3, 2, 1, -1),
        extra_args=[32, "BSH"]
    )


def test_flash_attention_output_layouts_bnsd_21():
    """
    Feature: Softmax output layouts for BNSD input.
    Description: BNSD with dp on batch, mp on head. Verify softmax_max/sum are 4D
        and map BNSD dimensions correctly.
    Expectation: softmax_max/sum tensor_map is (dp, mp, -1, -1).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(1))

    q_layout = _build_layout(mesh, placements, 4)
    k_layout = _build_layout(mesh, placements, 4)
    v_layout = _build_layout(mesh, placements, 4)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "BNSD"])

    attention_out, softmax_max, softmax_sum, _ = output_layouts

    assert attention_out.to_dict()["tensor_map"] == (1, 0, -1, -1)
    assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
    assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)


def test_flash_attention_output_layouts_sbh_22():
    """
    Feature: Softmax output layouts for SBH input.
    Description: SBH with dp on batch (dim 1), mp on hidden (dim 2). Verify softmax
        correctly reorders SBH dimensions to (B, N, S, 8).
    Expectation: softmax_max/sum tensor_map is (dp, mp, -1, -1).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(1), Shard(2))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "SBH"])

    attention_out, softmax_max, softmax_sum, _ = output_layouts

    assert attention_out.to_dict()["tensor_map"] == (-1, 1, 0)
    assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
    assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)


def test_flash_attention_output_layouts_tnd_23():
    """
    Feature: Softmax output layouts for TND input.
    Description: TND with dp on token dim (dim 0), mp on head dim (dim 1). TND branch
        maps softmax[0]=query_tm[0] and softmax[1]=query_tm[1], rest are -1.
    Expectation: softmax_max/sum tensor_map is (dp, mp, -1, -1).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(1))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "TND"])

    attention_out, softmax_max, softmax_sum, _ = output_layouts

    assert attention_out.to_dict()["tensor_map"] == (1, 0, -1)
    assert softmax_max.to_dict()["tensor_map"] == (1, 0, -1, -1)
    assert softmax_sum.to_dict()["tensor_map"] == (1, 0, -1, -1)


def test_flash_attention_output_layouts_mixed_parallel_24():
    """
    Feature: Softmax output layouts with full DP + SP + MP parallelism.
    Description: BSH with dp on batch, sp on seq, mp on hidden. Softmax 4D maps
        B->dp, N->mp (via hidden), S->sp.
    Expectation: softmax_max/sum tensor_map is (dp, mp, sp, -1).
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1), Shard(2))
    kv_placements = (Shard(0), Replicate(), Shard(2))

    q_layout = _build_layout(mesh, q_placements, 3)
    k_layout = _build_layout(mesh, kv_placements, 3)
    v_layout = _build_layout(mesh, kv_placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "BSH"])

    attention_out, softmax_max, softmax_sum, _ = output_layouts

    assert attention_out.to_dict()["tensor_map"] == (2, 1, 0)
    assert softmax_max.to_dict()["tensor_map"] == (2, 0, 1, -1)
    assert softmax_sum.to_dict()["tensor_map"] == (2, 0, 1, -1)


def test_flash_attention_softmax_no_sharding_25():
    """
    Feature: Softmax output layouts with no sharding.
    Description: All dimensions replicated on a single-axis mesh.
    Expectation: softmax_max/sum tensor_map is all -1.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("dp",),
        init_backend=False
    )
    placements = (Replicate(),)

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "BSH"])

    _, softmax_max, softmax_sum, _ = output_layouts

    assert softmax_max.to_dict()["tensor_map"] == (-1, -1, -1, -1)
    assert softmax_sum.to_dict()["tensor_map"] == (-1, -1, -1, -1)


def test_flash_attention_tnd_sp_27():
    """
    Feature: Layout inference for TND with sequence parallelism.
    Description: TND layout with T (token) dim sharded on sp, N (head) dim sharded
        on mp. This exercises the most complex actual_seq_len adjustment path in
        expanded_impl at runtime.
    Expectation: Output preserves T-dim sharding on sp and N-dim sharding on mp.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1))
    kv_placements = (Replicate(), Shard(1))

    run_scenario(
        "TND + SP",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=3,
        expected_out_map=(1, 0, -1),
        extra_args=[16, "TND"]
    )


def test_flash_attention_softmax_layout_param_29():
    """
    Feature: Layout inference with softmax_layout specified in extra_args.
    Description: extra_args includes sparse_mode=0 and softmax_layout string.
        infer_layout should handle the additional parameter without error.
    Expectation: Layout inference succeeds; attention_out matches query.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    run_scenario(
        "With softmax_layout in extra_args",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(1, -1, 0),
        extra_args=[16, "BSH", 0, "NZ"]
    )


def test_flash_attention_softmax_layout_empty_30():
    """
    Feature: Layout inference with empty softmax_layout string.
    Description: extra_args includes sparse_mode=0 and empty softmax_layout.
    Expectation: Equivalent to not passing softmax_layout at all.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    # With empty softmax_layout
    output_with = op.infer_layout(
        (q_layout, k_layout, v_layout), [16, "BSH", 0, ""]
    )
    # Without softmax_layout
    output_without = op.infer_layout(
        (q_layout, k_layout, v_layout), [16, "BSH", 0]
    )

    assert output_with[0].to_dict()["tensor_map"] == \
        output_without[0].to_dict()["tensor_map"], \
        "Empty softmax_layout should be equivalent to omitting it"


def test_flash_attention_bsnd_sp_31():
    """
    Feature: Layout inference for BSND with sequence parallelism.
    Description: 4D BSND tensor with dp on batch (dim 0), sp on seq (dim 1),
        mp on head (dim 2).
    Expectation: Output preserves all three sharding axes on 4D tensor.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1), Shard(2))
    kv_placements = (Shard(0), Replicate(), Shard(2))

    run_scenario(
        "BSND + SP",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=4,
        expected_out_map=(2, 1, 0, -1),
        extra_args=[16, "BSND"]
    )


def test_flash_attention_bnsd_sp_32():
    """
    Feature: Layout inference for BNSD with sequence parallelism.
    Description: 4D BNSD tensor with dp on batch (dim 0), mp on head (dim 1),
        sp on seq (dim 2).
    Expectation: Output preserves dp+mp+sp sharding on 4D tensor.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(2), Shard(1))
    kv_placements = (Shard(0), Replicate(), Shard(1))

    run_scenario(
        "BNSD + SP",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=4,
        expected_out_map=(2, 0, 1, -1),
        extra_args=[16, "BNSD"]
    )


def test_flash_attention_ring_attention_33():
    """
    Feature: Layout inference with both Q and KV sequence-sharded.
    Description: BSH with sp sharding on seq dim for Q, K, and V (ring attention).
    Expectation: Layout inference succeeds; output seq dim sharded on sp.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "sp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(1))

    run_scenario(
        "Ring Attention (Q+KV seq sharded)",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(1, 0, -1),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_sparse_mode_3_sp_34():
    """
    Feature: Layout inference with sparse_mode=3 (rightDownToLeftUp) and SP.
    Description: Causal variant with reversed direction, query seq sharded on sp.
    Expectation: Layout inference succeeds with seq sharding preserved.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4,),
        mesh_dim_names=("sp",),
        init_backend=False
    )
    q_placements = (Shard(1),)
    kv_placements = (Replicate(),)

    run_scenario(
        "Sparse Mode 3 (Reverse Causal + SP)",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=3,
        expected_out_map=(-1, 0, -1),
        extra_args=[16, "BSH", 3]
    )


def test_flash_attention_sparse_mode_4_band_35():
    """
    Feature: Layout inference with sparse_mode=4 (band) and DP.
    Description: Band mask mode with data parallelism only.
    Expectation: Layout inference succeeds normally.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("dp",),
        init_backend=False
    )
    placements = (Shard(0),)

    run_scenario(
        "Sparse Mode 4 (Band + DP)",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(0, -1, -1),
        extra_args=[16, "BSH", 4]
    )


def test_flash_attention_invalid_input_layout_36():
    """
    Feature: Error handling for invalid input_layout string.
    Description: extra_args specifies an unsupported input_layout.
    Expectation: Raises an appropriate error.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    with pytest.raises((ValueError, KeyError, RuntimeError)):
        op.infer_layout((q_layout, k_layout, v_layout), [16, "INVALID"])


def test_flash_attention_missing_extra_args_37():
    """
    Feature: Error handling for missing extra_args.
    Description: extra_args is empty, missing required head_num and input_layout.
    Expectation: Raises an appropriate error.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Replicate())

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    with pytest.raises((IndexError, TypeError, ValueError, RuntimeError)):
        op.infer_layout((q_layout, k_layout, v_layout), [])


def test_flash_attention_ndim_mismatch_layout_38():
    """
    Feature: Graceful handling for tensor ndim vs input_layout mismatch.
    Description: 3D tensor with BNSD (4D) input_layout in extra_args.
        The code handles this gracefully without raising an error.
    Expectation: Layout inference completes without error; output has 3D tensor_map.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    q_layout = _build_layout(mesh, placements, 3)
    k_layout = _build_layout(mesh, placements, 3)
    v_layout = _build_layout(mesh, placements, 3)

    # Should not raise — code handles mismatch gracefully
    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "BNSD"])
    attention_out_layout = output_layouts[0]
    # Output tensor_map should preserve 3D structure from query layout
    assert len(attention_out_layout.to_dict()["tensor_map"]) == 3, \
        "Output should preserve 3D tensor_map from query layout"


def test_flash_attention_head_num_1_39():
    """
    Feature: Layout inference with head_num=1.
    Description: Single-head attention with mp sharding. head_num=1 cannot be
        further divided by TP, but layout inference only records sharding strategy
        without validating divisibility.
    Expectation: Layout inference succeeds; sharding recorded as specified.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("dp", "mp"),
        init_backend=False
    )
    placements = (Shard(0), Shard(2))

    run_scenario(
        "head_num=1",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(1, -1, 0),
        extra_args=[1, "BSH"]
    )


def test_flash_attention_dp_only_no_mp_no_sp_40():
    """
    Feature: Layout inference with pure DP on a single-axis mesh.
    Description: Only data parallelism, no model or sequence parallelism.
    Expectation: Only batch dimension sharded.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("dp",),
        init_backend=False
    )
    placements = (Shard(0),)

    run_scenario(
        "Pure DP (single-axis mesh)",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(0, -1, -1),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_mp_only_no_dp_no_sp_41():
    """
    Feature: Layout inference with pure MP on a single-axis mesh.
    Description: Only model parallelism on hidden dim, no DP or SP.
    Expectation: Only hidden dimension sharded.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(8,),
        mesh_dim_names=("mp",),
        init_backend=False
    )
    placements = (Shard(2),)

    run_scenario(
        "Pure MP (single-axis mesh)",
        mesh, placements, placements, placements,
        ndim=3,
        expected_out_map=(-1, -1, 0),
        extra_args=[16, "BSH"]
    )


def test_flash_attention_sbh_sp_42():
    """
    Feature: Layout inference for SBH with sequence parallelism.
    Description: SBH layout with sp on seq (dim 0) and mp on hidden (dim 2).
    Expectation: Output preserves SBH ordering with sp and mp sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(2))
    kv_placements = (Replicate(), Shard(2))

    run_scenario(
        "SBH + SP",
        mesh, q_placements, kv_placements, kv_placements,
        ndim=3,
        expected_out_map=(1, -1, 0),
        extra_args=[16, "SBH"]
    )


def test_flash_attention_tnd_softmax_output_sp_43():
    """
    Feature: Softmax output layouts for TND with sequence parallelism.
    Description: TND with sp on T (dim 0), mp on N (dim 1). Verify softmax layouts
        correctly map the sharding from TND to 4D softmax output (B,N,S,8).
    Expectation: softmax_max/sum preserve sp and mp sharding.
    """
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(4, 2),
        mesh_dim_names=("sp", "mp"),
        init_backend=False
    )
    q_placements = (Shard(0), Shard(1))
    kv_placements = (Replicate(), Shard(1))

    q_layout = _build_layout(mesh, q_placements, 3)
    k_layout = _build_layout(mesh, kv_placements, 3)
    v_layout = _build_layout(mesh, kv_placements, 3)

    output_layouts = op.infer_layout((q_layout, k_layout, v_layout), [16, "TND"])

    attention_out, softmax_max, softmax_sum, _ = output_layouts

    assert attention_out.to_dict()["tensor_map"] == (1, 0, -1)
    assert len(softmax_max.to_dict()["tensor_map"]) == 4
    assert len(softmax_sum.to_dict()["tensor_map"]) == 4
