# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.4: Phase 3 _infer_boundary_type 表驱动用例。"""

import pytest

from hyper_models.components.distributed.param_role import ParamRole
from hyper_models.components.distributed.sharding_planner import ShardingPlanner

P = ShardingPlanner()
C, R, N = ParamRole.COLWISE, ParamRole.ROWWISE, ParamRole.NORM


@pytest.mark.parametrize("fqn,group,want", [
    # 显式模式
    ("model.embed_tokens", [("x.weight", ParamRole.EMBED)], "embed"),
    ("model.wte", [("x.weight", ParamRole.EMBED)], "embed"),
    ("lm_head", [("x.weight", ParamRole.LM_HEAD)], "lm_head"),
    ("model.embed_out", [("x.weight", ParamRole.LM_HEAD)], "lm_head"),
    ("model.layers.0.input_layernorm", [("x.weight", N)], "norm"),
    ("model.norm", [("x.weight", N)], "norm"),
    ("model.layers.0.mlp.router", [("x.weight", ParamRole.MOE_GATE)], "moe_gate"),
    # 角色组合
    ("model.layers.0.self_attn", [("a", C), ("b", C), ("c", C), ("d", R)], "attention"),
    ("model.layers.0.mlp", [("a", C), ("b", C), ("d", R)], "mlp"),
    # colwise+rowwise 组合默认归 attention
    ("model.layers.0.block", [("a", C), ("d", R)], "attention"),
    # 仅 colwise → mlp（需 fqn 命中 mlp 模式）
    ("model.layers.0.mlp", [("a", C), ("b", C)], "mlp"),
    ("model.layers.0.self_attn.q_proj", [("a", C)], "unknown"),  # 叶守卫
    # MoE
    ("model.layers.0.mlp", [("a", ParamRole.MOE_GATE), ("b", ParamRole.MOE_EXPERT)],
     "moe_mlp"),
    ("model.layers.0.mlp.experts", [("b", ParamRole.MOE_EXPERT)], "unknown"),  # 叶守卫
    # 均无 → unknown
    ("model.layers.0", [("a", ParamRole.SKIP)], "unknown"),
])
def test_infer_boundary_type(fqn, group, want):
    assert P._infer_boundary_type(fqn, group) == want
