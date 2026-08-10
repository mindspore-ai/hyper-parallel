# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s1_plan_arch.py: 核心套件合并文件。

来源: test_s1_arch_override.py, test_s1_head_count.py, test_s1_role_mapping.py, test_s1_semantic_infer.py, test_s1_mla_deepseek.py, test_s1_special_handlers.py, test_s1_compat.py, test_s1_sp_loss_matrix.py
"""

import logging
import pytest
import torch.nn as nn
from hyper_models.components.distributed.head_count import (
    _is_head_sharded,
    update_module_head_counts,
)
from hyper_models.components.distributed.param_role import (
    ParamRole,
    ParameterClassifier,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    ModuleShardingSpec,
    TEMPLATES,
    TP,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import (
    ARCH_OVERRIDES,
    SPECIAL_HANDLERS,
    ShardingPlanner,
    validate_model_compatibility,
)
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
)


# ==========================================================================
# 来源: test_s1_arch_override.py
# S1.2: ARCH_OVERRIDES 覆盖优先级 + _get_architecture。
# ==========================================================================

class _Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(8, 4)
        self.output_head = nn.Linear(4, 8, bias=False)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestArchOverridePriority:
    def test_override_beats_default(self):
        """override 命中 → 覆盖默认规则（embed_tokens 默认 EMBED，强制为 SKIP）。"""
        overrides = {"myarch": [("embed_tokens.weight", ParamRole.SKIP)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        model = _Model()
        roles = clf.classify(model, "myarch")
        assert roles["token_embed.weight"] == ParamRole.SKIP

    def test_override_list_of_patterns(self):
        """list-of-patterns 写法：任一子模式命中即覆盖。"""
        overrides = {"myarch": [(["token_embed", "word_embed"], ParamRole.EMBED)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), "myarch")
        assert roles["token_embed.weight"] == ParamRole.EMBED

    def test_default_when_no_override_match(self):
        """override 未命中 → 默认规则（output_head 非标准名 → SKIP）。"""
        overrides = {"myarch": [("token_embed", ParamRole.EMBED)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), "myarch")
        assert roles["output_head.weight"] == ParamRole.SKIP

    def test_unknown_arch_falls_back_to_default(self):
        overrides = {"other": [("token_embed", ParamRole.EMBED)]}
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), "myarch")
        assert roles["token_embed.weight"] == ParamRole.SKIP


class TestGetArchitecture:
    def setup_method(self):
        self.planner = ShardingPlanner()

    def test_architectures_first(self):
        m = _Model(config=_Cfg(architectures=["Qwen2ForCausalLM"], model_type="qwen2"))
        assert self.planner._get_architecture(m) == "qwen2"

    def test_model_type_fallback(self):
        m = _Model(config=_Cfg(architectures=None, model_type="mixtral"))
        assert self.planner._get_architecture(m) == "mixtral"

    def test_classname_fallback(self):
        class LlamaForCausalLM(nn.Module):
            config = None
        assert self.planner._get_architecture(LlamaForCausalLM()) == "llama"

    def test_suffix_stripping(self):
        for cls_name, want in [
            ("LlamaForCausalLM", "llama"),
            ("Blip2ForConditionalGeneration", "blip2"),
            ("BertForSequenceClassification", "bert"),
            ("PaliGemmaForImageTextToText", "paligemma"),
        ]:
            cls = type(cls_name, (nn.Module,), {"config": None})
            assert self.planner._get_architecture(cls()) == want

    def test_no_config_attribute(self):
        class Tiny(nn.Module):
            pass
        assert self.planner._get_architecture(Tiny()) == "tiny"


# ==========================================================================
# 来源: test_s1_head_count.py
# S1.14: head_count — TP 本地头数改写（D-17）。
# ==========================================================================

def _spec(**params):
    return ModuleShardingSpec(params=params)


class TestIsHeadSharded:
    def test_qkv_colwise_detected(self):
        spec = _spec(**{
            "q_proj.weight": {TP: Shard(0)},
            "k_proj.weight": {TP: Shard(0)},
            "v_proj.weight": {TP: Shard(0)},
            "o_proj.weight": {TP: Shard(1)},
        })
        assert _is_head_sharded(spec, ("tp",)) is True

    def test_mla_q_b_proj_detected(self):
        """D-14 MLA：q_b_proj 上投影按头维 colwise → 命中。"""
        spec = _spec(**{
            "q_a_proj.weight": {TP: Replicate()},
            "q_b_proj.weight": {TP: Shard(0)},
            "kv_b_proj.weight": {TP: Shard(0)},
        })
        assert _is_head_sharded(spec, ("tp",)) is True

    def test_mlp_not_detected(self):
        spec = _spec(**{
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        })
        assert _is_head_sharded(spec, ("tp",)) is False

    def test_no_tp_axis_not_detected(self):
        spec = _spec(**{"q_proj.weight": {TP: Shard(0), CP: Replicate()}})
        assert _is_head_sharded(spec, ("cp",)) is False

    def test_rowwise_only_not_detected(self):
        """q/k/v 均为 Replicate（如 fused QKV rowwise 方案）→ 头数未切。"""
        spec = _spec(**{
            "q_proj.weight": {TP: Replicate()},
            "o_proj.weight": {TP: Shard(1)},
        })
        assert _is_head_sharded(spec, ("tp",)) is False


class _NamedAttrAttention(nn.Module):
    """transformers 命名变体覆盖：n_heads / num_kv_heads（falcon 风格）。"""

    def __init__(self):
        super().__init__()
        self.n_heads = 8
        self.num_kv_heads = 2
        self.num_key_value_groups = 4
        self.head_dim = 16
        self.config = TinyConfig(num_attention_heads=8)


class TestUpdateModuleHeadCounts:
    def test_divide_and_preserve_invariants(self):
        attn = TinyLlamaAttention(TinyConfig())   # num_heads=4, head_dim=4
        attn.num_key_value_heads = 4
        n = update_module_head_counts(attn, 2, "self_attn")
        assert n == 2
        assert attn.num_heads == 2
        assert attn.num_key_value_heads == 2
        assert attn.head_dim == 4                     # 头维不切
        assert attn.config.num_attention_heads == 4   # config 不改写
        assert attn._hp_full_head_counts == {
            "num_heads": 4, "num_key_value_heads": 4}

    def test_name_variants(self):
        attn = _NamedAttrAttention()
        n = update_module_head_counts(attn, 4, "self_attn")
        assert n == 1                     # num_kv_heads=2 对 tp=4 不整除 → 仅告警
        assert attn.n_heads == 2
        assert attn.num_kv_heads == 2     # 不整除 → 保持原值

    def test_num_key_value_groups_untouched(self):
        attn = _NamedAttrAttention()
        update_module_head_counts(attn, 2, "self_attn")
        assert attn.n_heads == 4
        assert attn.num_kv_heads == 1
        assert attn.num_key_value_groups == 4   # 比值不变量，绝不动

    def test_idempotent(self):
        attn = TinyLlamaAttention(TinyConfig())
        assert update_module_head_counts(attn, 2) == 1
        assert update_module_head_counts(attn, 2) == 0   # 不二次除法
        assert attn.num_heads == 2

    def test_non_divisible_warns_and_keeps(self, caplog):
        attn = TinyLlamaAttention(TinyConfig())   # num_heads=4
        with caplog.at_level(logging.WARNING):
            n = update_module_head_counts(attn, 3, "self_attn")
        assert n == 0
        assert attn.num_heads == 4
        assert "not divisible" in caplog.text
        # 重复调用不重复告警
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            update_module_head_counts(attn, 3, "self_attn")
        assert "not divisible" not in caplog.text

    def test_tp1_noop(self):
        attn = TinyLlamaAttention(TinyConfig())
        assert update_module_head_counts(attn, 1) == 0
        assert attn.num_heads == 4
        assert not hasattr(attn, "_hp_full_head_counts")


# ==========================================================================
# 来源: test_s1_role_mapping.py
# S1.6: _build_spec_from_template 13 角色 → placement 映射。
# ==========================================================================

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


# ==========================================================================
# 来源: test_s1_semantic_infer.py
# S1.4: Phase 3 _infer_boundary_type 表驱动用例。
# ==========================================================================

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


# ==========================================================================
# 来源: test_s1_mla_deepseek.py
# S1.14: DeepSeek MLA 架构覆盖（ARCH_OVERRIDES + ParamRole.REPLICATED）。
# ==========================================================================

class _TinyMlaAttention(nn.Module):
    """DeepSeek MLA 结构（FQN 与 HF DeepseekV2/V3 Attention 一致）。"""

    def __init__(self, hidden=8, rank=4, q_out=8, kv_out=8):
        super().__init__()
        self.q_a_proj = nn.Linear(hidden, rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(rank)
        self.q_b_proj = nn.Linear(rank, q_out, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(rank)
        self.kv_b_proj = nn.Linear(rank, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden, bias=False)


class _TinyMlp(nn.Module):
    def __init__(self, hidden=8, inter=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class _TinyDeepseek(nn.Module):
    """2 层 MLA 小模型：FQN 仿 HF DeepseekV3（model.layers.N.self_attn.*）。"""

    def __init__(self, architectures=("DeepseekV3ForCausalLM",)):
        super().__init__()
        self.config = _Cfg(architectures=list(architectures),
                           model_type="deepseek_v3")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = _TinyMlaAttention()
            layer.mlp = _TinyMlp()
            layer.input_layernorm = nn.RMSNorm(8)
            layer.post_attention_layernorm = nn.RMSNorm(8)
            self.model.layers.append(layer)
        self.model.norm = nn.RMSNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)


class TestMlaArchOverride:
    def test_arch_overrides_registered_both_spellings(self):
        """architectures 拼写（deepseekv3）与 model_type 拼写（deepseek_v3）
        均注册同一份 MLA 覆盖；v2/v3 同构。"""
        for key in ("deepseekv2", "deepseekv3", "deepseek_v2", "deepseek_v3"):
            assert key in ARCH_OVERRIDES
            roles = [r for _, r in ARCH_OVERRIDES[key]]
            assert ParamRole.REPLICATED in roles
            assert ParamRole.COLWISE in roles

    def test_classifier_mla_roles(self):
        """q_a/kv_a → REPLICATED；q_b/kv_b → COLWISE；o_proj → ROWWISE（默认规则）。"""
        model = _TinyDeepseek()
        clf = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        roles = clf.classify(model, "deepseekv3")
        p = "model.layers.0.self_attn."
        assert roles[p + "q_a_proj.weight"] == ParamRole.REPLICATED
        assert roles[p + "kv_a_proj_with_mqa.weight"] == ParamRole.REPLICATED
        assert roles[p + "q_b_proj.weight"] == ParamRole.COLWISE
        assert roles[p + "kv_b_proj.weight"] == ParamRole.COLWISE
        assert roles[p + "o_proj.weight"] == ParamRole.ROWWISE
        assert roles[p + "q_a_layernorm.weight"] == ParamRole.NORM
        assert roles[p + "kv_a_layernorm.weight"] == ParamRole.NORM

    def test_classifier_model_type_spelling(self):
        """model_type 拼写（deepseek_v3）同样命中覆盖。"""
        model = _TinyDeepseek()
        clf = ParameterClassifier(arch_overrides=ARCH_OVERRIDES)
        roles = clf.classify(model, "deepseek_v3")
        assert roles["model.layers.0.self_attn.q_a_proj.weight"] == ParamRole.REPLICATED

    def test_without_override_mla_params_skip(self):
        """回归保护：无覆盖时 MLA 投影落 SKIP（即修复前的静默缺口）。"""
        model = _TinyDeepseek()
        clf = ParameterClassifier()   # 无 arch_overrides
        roles = clf.classify(model, "deepseekv3")
        p = "model.layers.0.self_attn."
        assert roles[p + "q_a_proj.weight"] == ParamRole.SKIP
        assert roles[p + "kv_b_proj.weight"] == ParamRole.SKIP


class TestMlaPlan:
    def test_attention_boundary_and_placements(self, make_mesh):
        """端到端：architectures 检测 → 覆盖生效 → attention 边界生成，
        REPLICATED 全复制 / q_b,kv_b colwise / o_proj rowwise / cp_attn 置位。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(
            _TinyDeepseek(), mesh, tp_size=2, sequence_parallel=True)

        spec = plan.modules["model.layers.0.self_attn"]
        names = ("tp",)
        assert tuple(resolve_placements(
            spec.params["q_a_proj.weight"], names)) == (Replicate(),)
        assert tuple(resolve_placements(
            spec.params["kv_a_proj_with_mqa.weight"], names)) == (Replicate(),)
        assert tuple(resolve_placements(
            spec.params["q_b_proj.weight"], names)) == (Shard(0),)
        assert tuple(resolve_placements(
            spec.params["kv_b_proj.weight"], names)) == (Shard(0),)
        assert tuple(resolve_placements(
            spec.params["o_proj.weight"], names)) == (Shard(1),)
        # attention 模板标记：CP 激活时注入 inner attention wrapper
        assert spec._needs_cp_attn is True
        # 其余边界不受覆盖影响
        assert "model.layers.0.mlp" in plan.modules
        assert "model.embed_tokens" in plan.modules
        assert plan.modules["lm_head"]._is_terminal is True

    def test_model_type_fallback_also_hits(self, make_mesh):
        """config.architectures 缺失时回退 model_type='deepseek_v3' 同样命中。"""
        mesh = make_mesh((1,), ("tp",))
        model = _TinyDeepseek(architectures=())
        model.config.architectures = None
        plan = ShardingPlanner().plan(model, mesh, tp_size=2)
        spec = plan.modules["model.layers.0.self_attn"]
        assert tuple(resolve_placements(
            spec.params["q_b_proj.weight"], ("tp",))) == (Shard(0),)

    def test_both_layers_sharded(self, make_mesh):
        """两层 MLA 均生成 attention spec（无遗漏）。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(_TinyDeepseek(), mesh, tp_size=2)
        for i in range(2):
            assert f"model.layers.{i}.self_attn" in plan.modules
            spec = plan.modules[f"model.layers.{i}.self_attn"]
            assert len(spec.params) == 5   # q_a/q_b/kv_a/kv_b/o（layernorm 独立边界）


# ==========================================================================
# 来源: test_s1_special_handlers.py
# S1.10: Phase 6 _collect_special_handlers + SPECIAL_HANDLERS 注册表。
# ==========================================================================

def test_special_role_mapped_to_handler():
    roles = {
        "model.layers.0.gated_delta.a_log": ParamRole.SPECIAL,
        "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
    }
    out = P._collect_special_handlers(roles)
    assert out == {"model.layers.0.gated_delta.a_log": "gated_delta_tp_shard"}


def test_unregistered_pattern_defaults():
    class _P(ShardingPlanner):
        def __init__(self):
            super().__init__()
            self._special_handler_patterns = {}

    p = _P()
    out = p._collect_special_handlers({"m.x.special_w": ParamRole.SPECIAL})
    assert out == {"m.x.special_w": "default"}


def test_non_special_roles_ignored():
    out = P._collect_special_handlers({
        "a.b.weight": ParamRole.COLWISE,
        "a.c.weight": ParamRole.SKIP,
    })
    assert out == {}


def test_special_handlers_registry():
    assert "gated_delta_tp_shard" in SPECIAL_HANDLERS
    assert callable(SPECIAL_HANDLERS["gated_delta_tp_shard"])


# ==========================================================================
# 来源: test_s1_compat.py
# S1.11: validate_model_compatibility。
# ==========================================================================

def _model(**kw):
    return TinyLlamaForCausalLM(TinyConfig(**kw))


class TestCompat:
    def test_heads_not_divisible(self):
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_model_compatibility(
                _model(num_attention_heads=3), tp_size=2)

    def test_kv_heads_not_divisible(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            validate_model_compatibility(
                _model(num_attention_heads=4, num_key_value_heads=3), tp_size=2)

    def test_seq_len_not_divisible_2cp(self):
        with pytest.raises(ValueError, match=r"2\*cp"):
            validate_model_compatibility(_model(), cp_size=2, seq_len=10)

    def test_seq_len_ok(self):
        validate_model_compatibility(_model(), cp_size=2, seq_len=8)

    def test_num_experts_not_divisible(self):
        with pytest.raises(ValueError, match="num_experts"):
            validate_model_compatibility(_model(num_experts=3), ep_size=2)

    def test_ep_requires_moe(self):
        with pytest.raises(ValueError, match="MoE"):
            validate_model_compatibility(_model(num_experts=0), ep_size=2)

    def test_moe_inter_dim_not_divisible_tp(self):
        with pytest.raises(ValueError, match="moe_intermediate_size"):
            validate_model_compatibility(
                _model(num_experts=4, moe_intermediate_size=7), tp_size=2, ep_size=2)

    def test_all_pass(self):
        validate_model_compatibility(
            _model(num_experts=4, moe_intermediate_size=8),
            tp_size=2, cp_size=2, ep_size=2, seq_len=16)


# ==========================================================================
# 来源: test_s1_sp_loss_matrix.py
# S1.7: SP on/off × loss_parallel on/off 四组合 I/O 契约。
# ==========================================================================

def _plan(tiny_llama, make_mesh, sp, lp):
    mesh = make_mesh((1,), ("tp",))
    return ShardingPlanner().plan(
        tiny_llama, mesh, tp_size=2, sequence_parallel=sp, loss_parallel=lp)


@pytest.mark.parametrize("sp,lp", [
    (True, False), (True, True), (False, False), (False, True),
])
def test_embed_contract(tiny_llama, make_mesh, sp, lp):
    spec = _plan(tiny_llama, make_mesh, sp, lp).modules["model.embed_tokens"]
    assert spec.in_src["input"][TP] == Replicate()
    assert spec.out_src["output"][TP] == Partial()
    want_out = Shard(1) if sp else Replicate()
    assert spec.out_dst["output"][TP] == want_out


@pytest.mark.parametrize("sp,lp", [
    (True, False), (True, True), (False, False), (False, True),
])
def test_attention_contract(tiny_llama, make_mesh, sp, lp):
    spec = _plan(tiny_llama, make_mesh, sp, lp).modules["model.layers.0.self_attn"]
    want_in = Shard(1) if sp else Replicate()
    assert spec.in_src["hidden_states"][TP] == want_in
    assert spec.in_dst["hidden_states"][TP] == Replicate()
    assert spec.out_src["output"][TP] == Partial()
    assert spec.out_dst["output"][TP] == want_in


@pytest.mark.parametrize("sp,lp,want_out_dst", [
    (True, False, Replicate()), (True, True, Shard(-1)),
    (False, False, Replicate()), (False, True, Shard(-1)),
])
def test_lm_head_out_dst_loss_parallel(tiny_llama, make_mesh, sp, lp, want_out_dst):
    spec = _plan(tiny_llama, make_mesh, sp, lp).modules["lm_head"]
    assert spec.out_src["output"][TP] == Shard(-1)
    assert spec.out_dst["output"][TP] == want_out_dst


def test_sp_cp_dim(tiny_llama, make_mesh):
    """SP 开启时 embed out_dst / norm in_src 的 CP 维为 Shard(1)。"""
    spec = _plan(tiny_llama, make_mesh, True, False).modules["model.norm"]
    assert spec.in_src["hidden_states"][CP] == Shard(1)
