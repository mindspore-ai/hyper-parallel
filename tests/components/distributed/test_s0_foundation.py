# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s0_foundation.py: 核心套件合并文件。

来源: test_s0_error.py, test_s0_fixtures.py, test_s0_param_role.py, test_s0_placement_utils.py, test_s0_spec_fields.py
"""

import pytest
import torch.nn as nn
from hyper_models.components.distributed.param_role import (
    ParamRole,
    ParameterClassifier,
    SEGMENT_EXACT,
    SEGMENT_SUBSTRING,
    _build_default_rules,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    ModuleShardingSpec,
    PlacementMismatchError,
    ShardingPlan,
    TP,
    _multi_dim,
    _normalize_out_fields,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


# ==========================================================================
# 来源: test_s0_error.py
# S0.4: PlacementMismatchError message 内容。
# ==========================================================================

def test_message_contains_all_fields():
    err = PlacementMismatchError(
        "model.layers.0.self_attn", (Shard(0),), (Replicate(),), "out_src"
    )
    msg = str(err)
    assert "model.layers.0.self_attn" in msg
    assert "out_src" in msg
    assert "Shard" in msg and "Replicate" in msg
    assert err.module_name == "model.layers.0.self_attn"
    assert err.stage == "out_src"
    assert err.expected == (Shard(0),)
    assert err.actual == (Replicate(),)


def test_is_value_error():
    with pytest.raises(ValueError):
        raise PlacementMismatchError("m", 1, 2, "chain")


# ==========================================================================
# 来源: test_s0_fixtures.py
# S0.5: fixtures 自检（FQN 清单 + golden plan 内部自洽）。
# ==========================================================================

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
    """tiny_moe 的 mlp 归 moe_mlp 边界且 region_dispatch。"""
    mesh = make_mesh((1, 1), ("tp", "ep"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    for layer in ("0", "1"):
        spec = plan.modules[f"model.layers.{layer}.mlp"]
        assert spec.region_dispatch is False
        assert any(p.startswith("experts.") for p in spec.params)
        assert "gate.weight" in spec.params


def test_tiny_hf_llama_arch(tiny_hf_llama):
    planner = ShardingPlanner()
    assert planner._get_architecture(tiny_hf_llama) == "llama"


# ==========================================================================
# 来源: test_s0_param_role.py
# S0.1 + S1.1: ParamRole 枚举完备性 + ParameterClassifier 默认规则。
# ==========================================================================

class TestParamRoleEnum:
    def test_fourteen_roles(self):
        assert len(ParamRole) == 14

    def test_expected_role_names(self):
        expected = {
            "COLWISE", "ROWWISE", "NORM", "EMBED", "LM_HEAD", "MOE_GATE",
            "MOE_EXPERT", "SHARED_EXPERT", "FUSED_QKV", "FUSED_GATE_UP",
            "BIAS", "REPLICATED", "SPECIAL", "SKIP",
        }
        assert {r.name for r in ParamRole} == expected


class _NamedModel(nn.Module):
    """用 ordered dict 注册参数以精确控制 FQN。"""
    def __init__(self, names):
        super().__init__()
        for n in names:
            # 把点分名注册到嵌套模块
            *path, leaf = n.split(".")
            obj = self
            for p in path:
                if not hasattr(obj, p):
                    setattr(obj, p, nn.Module())
                obj = getattr(obj, p)
            obj.register_parameter(leaf, nn.Parameter(__import__("torch").zeros(2)))


class TestParameterClassifier:
    def setup_method(self):
        self.clf = ParameterClassifier()

    def _role(self, name):
        model = _NamedModel([name])
        return self.clf.classify(model)[name]

    def test_each_role_hit(self):
        """默认规则可产生的 13 个角色每个至少 1 个命中用例（SPECIAL/SKIP 在内；
        REPLICATED 仅经 ARCH_OVERRIDES 指派，见 test_s1_mla_deepseek.py）。"""
        cases = {
            "model.embed_tokens.weight": ParamRole.EMBED,
            "lm_head.weight": ParamRole.LM_HEAD,
            "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
            "model.layers.0.self_attn.o_proj.weight": ParamRole.ROWWISE,
            "model.layers.0.input_layernorm.weight": ParamRole.NORM,
            "model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE,
            "model.layers.0.mlp.experts.w1": ParamRole.MOE_EXPERT,
            "model.layers.0.mlp.shared_experts.w2": ParamRole.SHARED_EXPERT,
            "model.layers.0.attn.fused_qkv.weight": ParamRole.FUSED_QKV,
            "model.layers.0.mlp.gate_up_proj.weight": ParamRole.FUSED_GATE_UP,
            "model.layers.0.self_attn.q_proj.bias": ParamRole.COLWISE,
            "model.layers.0.gated_delta.a_log": ParamRole.SPECIAL,
            "model.rotary_emb.inv_freq": ParamRole.SKIP,
        }
        for name, want in cases.items():
            assert self._role(name) == want, name

    def test_ln_rule_no_false_positive(self):
        """ln_ / norm 规则不误伤 linear / kernel。"""
        assert self._role("model.linear.weight") == ParamRole.SKIP
        assert self._role("model.kernel.weight") == ParamRole.SKIP
        assert self._role("model.layers.0.ln_1.weight") == ParamRole.NORM

    def test_shared_experts_not_moe_expert(self):
        """shared_experts 必须先于 experts 命中 SHARED_EXPERT。"""
        assert self._role("m.mlp.shared_experts.w1") == ParamRole.SHARED_EXPERT
        assert self._role("m.mlp.experts.w1") == ParamRole.MOE_EXPERT

    def test_gate_proj_not_moe_gate(self):
        """dense gate_proj 不得误判为 MOE_GATE。"""
        assert self._role("m.mlp.gate_proj.weight") == ParamRole.COLWISE

    def test_expert_gate_proj_is_moe_expert(self):
        """per-expert 的 gate_proj（experts.N.gate_proj）归 MOE_EXPERT。"""
        assert self._role("m.mlp.experts.0.gate_proj.weight") == ParamRole.MOE_EXPERT

    def test_unmatched_returns_skip(self):
        model = _NamedModel(["a.b.c"])
        assert self.clf.classify(model)["a.b.c"] == ParamRole.SKIP

    def test_default_rules_structure(self):
        rules = _build_default_rules()
        for rule in rules:
            # F1: 规则为 (patterns, role, mode) 三元组；mode ∈ 段感知常量
            pats, r, mode = rule
            assert isinstance(pats, list) and isinstance(r, ParamRole)
            assert mode in (SEGMENT_EXACT, SEGMENT_SUBSTRING)

    def test_shared_expert_gate_not_shared_expert(self):
        """F1 段精确匹配：shared_expert_gate 段 ≠ shared_expert 段
        （accuracy_problem.md 10.1 误判来源）；默认识别为 COLWISE（
        qwen2moe 由 ARCH_OVERRIDES 显式覆盖为 REPLICATED）。"""
        assert self._role("m.mlp.shared_expert_gate.weight") != ParamRole.SHARED_EXPERT
        assert self._role("m.mlp.shared_expert.weight") == ParamRole.SHARED_EXPERT
        assert self._role("m.mlp.shared_experts.weight") == ParamRole.SHARED_EXPERT

    def test_dotted_pattern_keeps_path_substring(self):
        """F1 兼容性：带点 pattern 保持旧的全路径子串语义。"""
        assert self._role("model.layers.0.self_attn.q_proj.weight") == ParamRole.COLWISE

    def test_segment_substring_within_one_segment(self):
        """F1 段子串：fragment 命中单段内部（不跨段）。"""
        assert self._role("m.mlp.experts.gate_up_proj.weight") == ParamRole.MOE_EXPERT


# ==========================================================================
# 来源: test_s0_placement_utils.py
# S0.3: resolve_placements / _multi_dim / _normalize_out_fields。
# ==========================================================================

class TestResolvePlacements:
    def test_axis_order_follows_mesh_dim_names(self):
        named = {TP: Shard(0), CP: Replicate(), EP: Shard(0)}
        # mesh 轴序为 (ep, cp, tp) → 输出按该顺序重排
        out = resolve_placements(named, ("ep", "cp", "tp"))
        assert out == [Shard(0), Replicate(), Shard(0)]

    def test_missing_axis_fills_replicate(self):
        named = {TP: Shard(1)}
        out = resolve_placements(named, ("tp", "cp", "ep"))
        assert out == [Shard(1), Replicate(), Replicate()]

    def test_extra_keys_dropped(self):
        named = {TP: Shard(1), CP: Shard(1), EP: Replicate()}
        out = resolve_placements(named, ("tp",))
        assert out == [Shard(1)]

    def test_str_enum_key_interop(self):
        """plain string key 与 MeshAxisName key 互通。"""
        named = {"tp": Shard(0)}
        assert resolve_placements(named, ("tp",)) == [Shard(0)]


class TestMultiDim:
    def test_none_dims_filtered(self):
        out = _multi_dim(tp=Shard(0), cp=Replicate(), ep=None)
        assert EP not in out and out[TP] == Shard(0) and out[CP] == Replicate()

    def test_all_none_empty(self):
        assert _multi_dim() == {}


class TestNormalizeOutFields:
    def test_scalar_shorthand_wrapped(self):
        spec = ModuleShardingSpec(out_src={TP: Partial(), CP: Replicate()})
        _normalize_out_fields(spec)
        assert spec.out_src == {"output": {TP: Partial(), CP: Replicate()}}

    def test_dict_contract_untouched(self):
        spec = ModuleShardingSpec(
            out_src={"hidden_states": {TP: Shard(1)}},
            out_dst={"output": {TP: Replicate()}},
        )
        _normalize_out_fields(spec)
        assert spec.out_src == {"hidden_states": {TP: Shard(1)}}
        assert spec.out_dst == {"output": {TP: Replicate()}}

    def test_none_untouched(self):
        spec = ModuleShardingSpec(out_src=None, out_dst=None)
        _normalize_out_fields(spec)
        assert spec.out_src is None and spec.out_dst is None

    def test_idempotent(self):
        spec = ModuleShardingSpec(out_src={TP: Partial()})
        _normalize_out_fields(spec)
        _normalize_out_fields(spec)
        assert spec.out_src == {"output": {TP: Partial()}}


# ==========================================================================
# 来源: test_s0_spec_fields.py
# S0.2: ShardingPlan / ModuleShardingSpec 字段与 05 §3.1-3.2 对齐。
# ==========================================================================

def test_spec_defaults():
    spec = ModuleShardingSpec()
    # 2026-08-05「不写继承，写了照办」：契约字段缺省 None（未声明），
    # 与显式空 {}（不切分/无契约）语义分离；plan 输出经
    # _normalize_contract_fields 落成具体 dict
    assert spec.params is None
    assert spec.in_src is None
    assert spec.in_dst is None
    assert spec.out_src is None
    assert spec.out_dst is None
    assert spec.out_names is None
    assert spec.is_boundary is True
    # 内部标记存在且默认 False
    assert spec._is_terminal is False
    assert spec.region_dispatch is None
    assert spec._needs_cp_attn is False


def test_plan_defaults():
    plan = ShardingPlan()
    assert plan.modules == {}
    assert plan.sequence_parallel is True
    assert plan.loss_parallel is False
    assert plan.special_handlers == {}
    assert plan.mesh_dim_names == ()
    assert plan.tied_pairs == []


def test_spec_mutable_fields_independent():
    """显式构造的可变字段互不影响（None 缺省下由构造方持有具体 dict）。"""
    a, b = ModuleShardingSpec(params={}), ModuleShardingSpec(params={})
    a.params["w"] = {}
    assert b.params == {}
