# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.6: _build_spec_from_template 13 角色 → placement 映射。"""

import pytest

from hyper_models.components.distributed.param_role import ParamRole
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    TP,
    TEMPLATES,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

P = ShardingPlanner()
T = TEMPLATES["attention"]


@pytest.mark.parametrize("role,path,tp_want", [
    (ParamRole.COLWISE, "q_proj.weight", Shard(0)),
    (ParamRole.EMBED, "weight", Shard(0)),
    (ParamRole.LM_HEAD, "weight", Shard(0)),
    (ParamRole.FUSED_QKV, "fused_qkv.weight", Shard(0)),
    (ParamRole.FUSED_GATE_UP, "gate_up_proj.weight", Shard(0)),
    (ParamRole.ROWWISE, "o_proj.weight", Shard(1)),
    (ParamRole.NORM, "weight", Replicate()),
    (ParamRole.MOE_GATE, "gate.weight", Replicate()),
    (ParamRole.BIAS, "q_proj.bias", Replicate()),
])
def test_role_to_tp_placement(role, path, tp_want):
    out = P._placement_for_role(path, role, T, has_tp=True, has_ep=False)
    assert out[TP] == tp_want
    # CP 维参数恒 Replicate；EP 维非 MoE 参数 Replicate
    assert out[CP] == Replicate()
    assert out[EP] == Replicate()


def test_moe_expert_ep_shard_tp_by_name():
    """D-08：per-expert 2D 布局 → 标准 Shard(0)/Shard(1)。"""
    moe_t = TEMPLATES["moe_mlp"]
    w1 = P._placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=2)
    assert w1[EP] == Shard(0) and w1[TP] == Shard(0)
    w2 = P._placement_for_role("experts.w2", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=2)
    assert w2[EP] == Shard(0) and w2[TP] == Shard(1)


def test_moe_expert_3d_batched_tp_dims_shifted():
    """D-08：3D batched [E, H_out, H_in] → colwise=Shard(1)、rowwise=Shard(2)。"""
    moe_t = TEMPLATES["moe_mlp"]
    w1 = P._placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=3)
    assert w1[EP] == Shard(0) and w1[TP] == Shard(1)
    w2 = P._placement_for_role("experts.w2", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=3)
    assert w2[EP] == Shard(0) and w2[TP] == Shard(2)


def test_moe_expert_no_tp_explicit_replicate():
    """05 §3.5 NOTE：has_tp=False 时 MOE_EXPERT 仍显式 TP:Replicate。"""
    moe_t = TEMPLATES["moe_mlp"]
    out = P._placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                                has_tp=False, has_ep=True)
    assert out[TP] == Replicate()
    assert out[EP] == Shard(0)


def test_shared_expert_ep_replicate():
    moe_t = TEMPLATES["moe_mlp"]
    w1 = P._placement_for_role("shared_experts.w1", ParamRole.SHARED_EXPERT,
                               moe_t, True, True)
    assert w1[EP] == Replicate() and w1[TP] == Shard(0)
    w2 = P._placement_for_role("shared_experts.w2", ParamRole.SHARED_EXPERT,
                               moe_t, True, True)
    assert w2[EP] == Replicate() and w2[TP] == Shard(1)


def test_special_and_skip_return_none():
    assert P._placement_for_role("a_log", ParamRole.SPECIAL, T, True, False) is None
    assert P._placement_for_role("inv_freq", ParamRole.SKIP, T, True, False) is None


def test_has_tp_false_drops_tp_key_for_dense():
    out = P._placement_for_role("q_proj.weight", ParamRole.COLWISE, T,
                                has_tp=False, has_ep=False)
    assert TP not in out
    assert out[CP] == Replicate()


def test_has_ep_false_drops_ep_key_for_expert():
    moe_t = TEMPLATES["moe_mlp"]
    out = P._placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                                has_tp=True, has_ep=False)
    assert EP not in out
