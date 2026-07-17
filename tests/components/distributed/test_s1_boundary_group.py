# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.3: Phase 2 _group_by_boundary。"""

from hyper_models.components.distributed.param_role import ParamRole
from hyper_models.components.distributed.sharding_planner import ShardingPlanner


def _planner():
    return ShardingPlanner()


class TestBoundaryGrouping:
    def test_direct_hit(self):
        """去 leaf 后直接命中边界（norm 叶模块自身即边界）。"""
        p = _planner()
        groups = p._group_by_boundary(
            {"model.norm.weight": ParamRole.NORM})
        assert set(groups) == {"model.norm"}

    def test_backtrack_multi_level(self):
        """回溯多级命中：q_proj → self_attn（叶守卫 → 父级 attention）。"""
        p = _planner()
        roles = {
            "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
            "model.layers.0.self_attn.k_proj.weight": ParamRole.COLWISE,
            "model.layers.0.self_attn.v_proj.weight": ParamRole.COLWISE,
            "model.layers.0.self_attn.o_proj.weight": ParamRole.ROWWISE,
        }
        groups = p._group_by_boundary(roles)
        assert set(groups) == {"model.layers.0.self_attn"}
        assert len(groups["model.layers.0.self_attn"]) == 4

    def test_backtrack_to_root_unknown(self):
        """回溯到根仍 unknown → 归入参数所在模块。"""
        p = _planner()
        groups = p._group_by_boundary({"zzz.qqq.weight": ParamRole.SKIP})
        assert set(groups) == {"zzz.qqq"}

    def test_skip_params_fold_into_boundary(self):
        """SKIP 参数向上归入所在边界，不单独成组。"""
        p = _planner()
        roles = {
            "model.layers.0.mlp.gate_proj.weight": ParamRole.COLWISE,
            "model.layers.0.mlp.up_proj.weight": ParamRole.COLWISE,
            "model.layers.0.mlp.down_proj.weight": ParamRole.ROWWISE,
            "model.layers.0.mlp.some_scale": ParamRole.SKIP,
        }
        groups = p._group_by_boundary(roles)
        assert set(groups) == {"model.layers.0.mlp"}
        assert len(groups["model.layers.0.mlp"]) == 4

    def test_moe_params_fold_into_mlp(self):
        """gate + experts 共享同一个 moe mlp 边界。"""
        p = _planner()
        roles = {
            "model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE,
            "model.layers.0.mlp.experts.w1": ParamRole.MOE_EXPERT,
            "model.layers.0.mlp.experts.w2": ParamRole.MOE_EXPERT,
            "model.layers.0.mlp.shared_experts.w1": ParamRole.SHARED_EXPERT,
        }
        groups = p._group_by_boundary(roles)
        assert set(groups) == {"model.layers.0.mlp"}
        assert len(groups["model.layers.0.mlp"]) == 4

    def test_tiny_llama_boundaries(self, tiny_llama):
        """tiny_llama 完整边界集合 == 期望。"""
        p = _planner()
        roles = p._classify_all_params(tiny_llama, "tiny_llama")
        groups = p._group_by_boundary(roles)
        expected = {
            "model.embed_tokens", "model.norm", "lm_head",
            "model.layers.0.input_layernorm", "model.layers.0.self_attn",
            "model.layers.0.post_attention_layernorm", "model.layers.0.mlp",
            "model.layers.1.input_layernorm", "model.layers.1.self_attn",
            "model.layers.1.post_attention_layernorm", "model.layers.1.mlp",
        }
        assert set(groups) == expected
