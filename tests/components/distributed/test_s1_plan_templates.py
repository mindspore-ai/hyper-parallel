# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s1_plan_templates.py: 核心套件合并文件。

来源: test_s1_templates.py, test_s1_plan_golden.py, test_s1_mesh_dims.py, test_s1_boundary_group.py
"""

from hyper_models.components.distributed.param_role import ParamRole
import pytest
import torch.nn as nn
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    TEMPLATES,
    TP,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


# ==========================================================================
# 来源: test_s1_templates.py
# S1.5: ShardingTemplate + TEMPLATES 7 模板字段完整性。
# ==========================================================================

EXPECTED_TEMPLATES = {
    "attention", "mlp", "norm", "embed", "lm_head", "moe_gate", "moe_mlp",
}


def test_seven_templates_enumerated():
    assert set(TEMPLATES) == EXPECTED_TEMPLATES


def test_field_completeness_sp_and_nosp():
    """每模板 SP/non-SP I/O 字段全填。"""
    for name, t in TEMPLATES.items():
        for prefix in ("sp", "nosp"):
            assert getattr(t, f"{prefix}_in_src"), f"{name}.{prefix}_in_src"
            assert getattr(t, f"{prefix}_in_dst"), f"{name}.{prefix}_in_dst"
            assert getattr(t, f"{prefix}_out_src") is not None, f"{name}.{prefix}_out_src"
            assert getattr(t, f"{prefix}_out_dst") is not None, f"{name}.{prefix}_out_dst"


def test_cp_dim_present_in_io_contracts(self=None):
    """I/O 契约声明 CP 维（激活），参数侧 CP 恒 Replicate 由 _multi_dim 保证。"""
    for name, t in TEMPLATES.items():
        for key, named in t.sp_in_dst.items():
            assert CP in named, f"{name}.sp_in_dst[{key}] 缺 CP 维"


def test_attention_needs_cp_attn_and_keeps_cp_shard():
    t = TEMPLATES["attention"]
    assert t.needs_cp_attn is True
    # §6.3.2 非对称职责：attention sp_in_dst 的 CP 维保持 Shard(1)（boundary 不 gather）
    assert t.sp_in_dst["hidden_states"][CP] == Shard(1)
    assert t.sp_out_dst[CP] == Shard(1)


def test_moe_mlp_region_dispatch():
    assert TEMPLATES["moe_mlp"].region_dispatch is False
    assert TEMPLATES["moe_mlp"].moe_expert_placement == Shard(0)


def test_moe_gate_out_dst_ep_shard():
    t = TEMPLATES["moe_gate"]
    from hyper_models.components.distributed.sharding_config import EP
    assert t.sp_out_dst[EP] == Shard(0)
    assert t.nosp_out_dst[EP] == Shard(0)


def test_lm_head_out_src_shard_last_dim():
    # 标量简写：{TP: Shard(-1), ...}
    from hyper_models.components.distributed.sharding_config import TP
    assert TEMPLATES["lm_head"].sp_out_src[TP] == Shard(-1)


def test_norm_template_all_replicate_params():
    assert TEMPLATES["norm"].norm_placement == Replicate()


# ==========================================================================
# 来源: test_s1_plan_golden.py
# S1.12: ShardingPlanner.plan() 主入口 golden diff（tiny_llama SP on/off、
# ==========================================================================

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
    assert attn.region_dispatch is None

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
    assert moe.region_dispatch is False
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


# ==========================================================================
# 来源: test_s1_mesh_dims.py
# S1.8: _build_mesh_dim_names。
# ==========================================================================

P = ShardingPlanner()


class _FakeMesh:
    def __init__(self, names):
        self.mesh_dim_names = names


class TestBuildMeshDimNames:
    def test_authority_order_from_mesh(self):
        """以 mesh.mesh_dim_names 为权威顺序过滤 tp/cp/ep。"""
        mesh = _FakeMesh(("dp", "ep", "cp", "tp"))
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=2, ep_size=2)
        assert out == ("ep", "cp", "tp")

    def test_fallback_order(self):
        """未声明 mesh_dim_names 时按 (tp, cp, ep) 回退。"""
        mesh = _FakeMesh(None)
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=1, ep_size=4)
        assert out == ("tp", "ep")

    def test_size_one_axis_dropped(self):
        mesh = _FakeMesh(("tp", "cp", "ep"))
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=1, ep_size=1)
        assert out == ("tp",)

    def test_dp_axis_never_included(self):
        mesh = _FakeMesh(("dp_shard", "tp"))
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=1, ep_size=1)
        assert out == ("tp",)


# ==========================================================================
# 来源: test_s1_boundary_group.py
# S1.3: Phase 2 _group_by_boundary。
# ==========================================================================

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
        """gate + experts 共享同一个 moe mlp 边界；shared_experts 按 F3
        成为独立的嵌套 mlp 边界（其边界出口持有 RowWise Partial 归约，
        accuracy_fix_plan.md §2）。"""
        p = _planner()
        roles = {
            "model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE,
            "model.layers.0.mlp.experts.w1": ParamRole.MOE_EXPERT,
            "model.layers.0.mlp.experts.w2": ParamRole.MOE_EXPERT,
            "model.layers.0.mlp.shared_experts.w1": ParamRole.SHARED_EXPERT,
        }
        groups = p._group_by_boundary(roles)
        assert set(groups) == {"model.layers.0.mlp",
                               "model.layers.0.mlp.shared_experts"}
        assert len(groups["model.layers.0.mlp"]) == 3
        assert groups["model.layers.0.mlp.shared_experts"] == [
            ("model.layers.0.mlp.shared_experts.w1", ParamRole.SHARED_EXPERT)]

    def test_moe_gate_only_group_does_not_anchor(self):
        """F3 结构 lint：MOE_GATE-only 组不锚定 MoE 边界（如标量 gate
        Linear 误判路由），向上合并。"""
        p = _planner()
        roles = {
            "model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE,
        }
        groups = p._group_by_boundary(roles)
        assert "model.layers.0.mlp" not in groups

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


# ==========================================================================
# F4 plan 期 lint（accuracy_fix_plan.md §2）：分片整除校验 + 可训练参数覆盖。
# ==========================================================================

class _OddMlpModel(nn.Module):
    """colwise 参数行数不整除 tp_size 的玩具模型（mlp 边界）。"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 3, bias=False)  # (3, 8)：Shard(0) 不整除 tp=2
        self.mlp.up_proj = nn.Linear(8, 4, bias=False)
        self.mlp.down_proj = nn.Linear(4, 8, bias=False)


class TestPlanTimeLints:
    def test_shard_divisibility_fails_at_plan_time(self, make_mesh):
        """F4a：(3, 8) Shard(0) over tp=2 → plan 期教学化报错（不再等 apply
        期空分片，accuracy_problem.md 10.1 同类）。"""
        mesh = make_mesh((1,), ("tp",))
        with pytest.raises(ValueError, match="not divisible by tp size 2") as exc:
            ShardingPlanner().plan(_OddMlpModel(), mesh, tp_size=2)
        assert "classification" in str(exc.value)   # 指向最可能的分类错误根因

    def test_uncovered_trainable_param_fails(self, tiny_llama, make_mesh):
        """F4b：可训练参数不被任何 spec.params/special_handlers 覆盖 →
        plan 期硬报错（梯度同步语义不允许被消费侧默认静默决定）。"""
        tiny_llama.model.layers[0].extra = nn.Linear(4, 4)   # 无边界声明
        mesh = make_mesh((1,), ("tp",))
        with pytest.raises(ValueError, match="coverage check failed") as exc:
            ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
        msg = str(exc.value)
        assert "model.layers.0.extra.weight" in msg
        assert "allow_uncovered_params" in msg       # 逃生舱可见

    def test_allow_uncovered_params_downgrades_to_warning(self, tiny_llama,
                                                          make_mesh, caplog):
        """F4b 逃生舱：allow_uncovered_params=True → 降级为 WARNING（仅限
        探索性调试）。"""
        import logging
        tiny_llama.model.layers[0].extra = nn.Linear(4, 4)
        mesh = make_mesh((1,), ("tp",))
        with caplog.at_level(logging.WARNING):
            plan = ShardingPlanner(allow_uncovered_params=True).plan(
                tiny_llama, mesh, tp_size=2)
        assert "coverage check failed" in caplog.text
        assert "model.layers.0.extra.weight" in caplog.text
        assert plan.modules  # plan 正常产出

    def test_frozen_param_needs_no_coverage(self, tiny_llama, make_mesh):
        """requires_grad=False 的参数不参与 F4b（冻结即显式语义）。"""
        extra = nn.Linear(4, 4)
        extra.weight.requires_grad_(False)
        extra.bias.requires_grad_(False)
        tiny_llama.model.layers[0].extra = extra
        mesh = make_mesh((1,), ("tp",))
        ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)   # 不报错
