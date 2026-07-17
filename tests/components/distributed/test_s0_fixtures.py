# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S0.5: fixtures 自检（FQN 清单 + golden plan 内部自洽）。"""

from hyper_models.components.distributed.sharding_planner import ShardingPlanner


EXPECTED_TINY_LLAMA_FQNS = {
    "model.embed_tokens",
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp",
    "model.layers.1.input_layernorm",
    "model.layers.1.self_attn",
    "model.layers.1.post_attention_layernorm",
    "model.layers.1.mlp",
    "model.norm",
    "lm_head",
}


def test_tiny_llama_fqn_inventory(tiny_llama, make_mesh):
    """tiny_llama 的边界集合 == 期望 FQN 清单。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    assert set(plan.modules) == EXPECTED_TINY_LLAMA_FQNS


def test_tiny_llama_golden_plan_self_consistent(tiny_llama, make_mesh):
    """plan 内部自洽：相邻模块 out_dst == in_src（链式契约）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    ordered = [fqn for fqn, _ in tiny_llama.named_modules() if fqn in plan.modules]
    from hyper_models.components.distributed.sharding_config import (
        resolve_placements,
    )
    for a, b in zip(ordered[:-1], ordered[1:]):
        sa, sb = plan.modules[a], plan.modules[b]
        if sa.out_dst is None or not sb.in_src:
            continue
        out_vals = {tuple(resolve_placements(v, plan.mesh_dim_names))
                    for v in sa.out_dst.values()}
        in_vals = {tuple(resolve_placements(v, plan.mesh_dim_names))
                   for v in sb.in_src.values()}
        assert out_vals == in_vals, f"{a} → {b}"


def test_tiny_moe_boundaries(tiny_moe, make_mesh):
    """tiny_moe 的 mlp 归 moe_mlp 边界且 use_local_map。"""
    mesh = make_mesh((1, 1), ("tp", "ep"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    for layer in ("0", "1"):
        spec = plan.modules[f"model.layers.{layer}.mlp"]
        assert spec.use_local_map is True
        assert any(p.startswith("experts.") for p in spec.params)
        assert "gate.weight" in spec.params


def test_tiny_hf_llama_arch(tiny_hf_llama):
    planner = ShardingPlanner()
    assert planner._get_architecture(tiny_hf_llama) == "llama"
