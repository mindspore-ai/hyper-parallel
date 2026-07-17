# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.12: ShardingPlanner.plan() 主入口 golden diff（tiny_llama SP on/off、
tiny_hf_llama、tiny_moe）。"""

from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    TP,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard


def _assert_placement(named, mesh_dim_names, *want):
    got = tuple(resolve_placements(named, mesh_dim_names))
    assert got == want


def test_tiny_llama_golden_sp_on(tiny_llama, make_mesh):
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2,
                                  sequence_parallel=True)
    dims = plan.mesh_dim_names
    assert dims == ("tp",)

    attn = plan.modules["model.layers.0.self_attn"]
    assert attn.params["q_proj.weight"][TP] == Shard(0)
    assert attn.params["k_proj.weight"][TP] == Shard(0)
    assert attn.params["v_proj.weight"][TP] == Shard(0)
    assert attn.params["o_proj.weight"][TP] == Shard(1)
    _assert_placement(attn.in_src["hidden_states"], dims, Shard(1))
    _assert_placement(attn.in_dst["hidden_states"], dims, Replicate())
    _assert_placement(attn.out_src["output"], dims, Partial())
    _assert_placement(attn.out_dst["output"], dims, Shard(1))
    assert attn._needs_cp_attn is True
    assert attn.use_local_map is False

    mlp = plan.modules["model.layers.0.mlp"]
    assert mlp.params["gate_proj.weight"][TP] == Shard(0)
    assert mlp.params["up_proj.weight"][TP] == Shard(0)
    assert mlp.params["down_proj.weight"][TP] == Shard(1)
    _assert_placement(mlp.in_dst["hidden_states"], dims, Replicate())

    norm = plan.modules["model.layers.0.input_layernorm"]
    assert norm.params["weight"][TP] == Replicate()
    _assert_placement(norm.in_src["hidden_states"], dims, Shard(1))
    _assert_placement(norm.out_dst["output"], dims, Shard(1))

    embed = plan.modules["model.embed_tokens"]
    assert embed.params["weight"][TP] == Shard(0)
    _assert_placement(embed.out_src["output"], dims, Partial())
    _assert_placement(embed.out_dst["output"], dims, Shard(1))

    lm = plan.modules["lm_head"]
    assert lm.params["weight"][TP] == Shard(0)
    _assert_placement(lm.out_src["output"], dims, Shard(-1))
    _assert_placement(lm.out_dst["output"], dims, Replicate())  # loss_parallel=False
    assert lm._is_terminal is True


def test_tiny_llama_golden_sp_off(tiny_llama, make_mesh):
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2,
                                  sequence_parallel=False)
    dims = plan.mesh_dim_names
    attn = plan.modules["model.layers.0.self_attn"]
    _assert_placement(attn.in_src["hidden_states"], dims, Replicate())
    _assert_placement(attn.in_dst["hidden_states"], dims, Replicate())
    _assert_placement(attn.out_src["output"], dims, Partial())
    _assert_placement(attn.out_dst["output"], dims, Replicate())

    norm = plan.modules["model.norm"]
    _assert_placement(norm.in_src["hidden_states"], dims, Replicate())
    _assert_placement(norm.out_dst["output"], dims, Replicate())


def test_tiny_hf_llama_golden(tiny_hf_llama, make_mesh):
    """真实 HF FQN（mock config）下推导结果与 tiny_llama 一致。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_llama, mesh, tp_size=2)
    assert "model.layers.0.self_attn" in plan.modules
    attn = plan.modules["model.layers.0.self_attn"]
    assert attn.params["o_proj.weight"][TP] == Shard(1)


def test_tiny_moe_golden(tiny_moe, make_mesh):
    mesh = make_mesh((1, 1), ("tp", "ep"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    assert plan.mesh_dim_names == ("tp", "ep")
    moe = plan.modules["model.layers.0.mlp"]
    assert moe.use_local_map is True
    # gate 全复制
    assert moe.params["gate.weight"][TP] == Replicate()
    assert moe.params["gate.weight"][EP] == Replicate()
    # experts: EP Shard(0) + TP colwise/rowwise（D-08：3D [E,out,in] 权重
    # 的 TP 维平移——colwise→Shard(1)、rowwise→Shard(2)）
    assert moe.params["experts.w1"][EP] == Shard(0)
    assert moe.params["experts.w1"][TP] == Shard(1)
    assert moe.params["experts.w2"][EP] == Shard(0)
    assert moe.params["experts.w2"][TP] == Shard(2)
    assert moe.params["experts.w3"][EP] == Shard(0)
    assert moe.params["experts.w3"][TP] == Shard(1)
    # I/O 契约
    dims = plan.mesh_dim_names
    _assert_placement(moe.in_src["x_BLD"], dims, Shard(1), Replicate())
    _assert_placement(moe.in_dst["x_BLD"], dims, Replicate(), Replicate())
    _assert_placement(moe.out_src["output"], dims, Partial(), Replicate())
    _assert_placement(moe.out_dst["output"], dims, Shard(1), Replicate())


def test_plan_global_flags(tiny_llama, make_mesh):
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2,
                                  sequence_parallel=False, loss_parallel=True)
    assert plan.sequence_parallel is False
    assert plan.loss_parallel is True
    lm = plan.modules["lm_head"]
    assert lm.out_dst["output"][TP] == Shard(-1)  # loss_parallel=True
