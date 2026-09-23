# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

# White-box tests deliberately exercise private planner/spec APIs.
# pylint: disable=protected-access

"""test_s1_plan_arch.py: 核心套件合并文件。

来源: test_s1_arch_override.py, test_s1_head_count.py, test_s1_role_mapping.py,
test_s1_semantic_infer.py, test_s1_mla_deepseek.py, test_s1_special_handlers.py,
test_s1_compat.py, test_s1_sp_loss_matrix.py, test_s1_deferred_bias.py,
test_dsa_template.py, test_mhc_template.py, test_mtp_template.py,
test_shared_expert_template.py
"""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder import parameter_sharding
from hyper_parallel.distributed._builder.model_structure import (
    MHC_PARAMS,
    MTP_PREV_PROJ,
    SHARED_EXPERT_LINEAR,
    detect_model_structure,
)
from hyper_parallel.distributed.tensor_parallel.head_count import (
    _is_head_sharded,
    _update_user_tp_attrs,
    update_module_head_counts,
)
from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParamRole,
    ParameterClassifier,
)
from hyper_parallel.distributed.recipe_spec import (
    CP,
    EP,
    ModuleShardingSpec,
    TP,
    resolve_placements,
)
from hyper_parallel.distributed._builder.default_templates import (
    TEMPLATES,
    _placement_for_role,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _add_bias_to_primary_output,
)
from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
    validate_model_compatibility,
)
from hyper_parallel.distributed._builder.special_handlers import (
    SPECIAL_HANDLERS,
    _collect_special_handlers,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from tests.ut.dual_mode_dtensor.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
)


def test_deferred_bias_preserves_attention_tuple_output():
    """Deferred RowWise bias applies to hidden states without changing metadata."""
    hidden_states = torch.zeros(2, 3, 4)
    attention_weights = object()
    bias = torch.arange(4, dtype=hidden_states.dtype)

    output = _add_bias_to_primary_output(
        (hidden_states, attention_weights), bias, "TinyAttention"
    )

    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], hidden_states + bias)
    assert output[1] is attention_weights


def test_deferred_bias_rejects_non_tensor_primary_output():
    """Malformed structured outputs fail before bias is applied to a wrong field."""
    with pytest.raises(TypeError, match="output index 0 to be a Tensor"):
        _add_bias_to_primary_output((None, torch.ones(4)), torch.ones(4), "TinyAttention")


# ==========================================================================
# 来源: test_s1_arch_override.py
# S1.2: arch_overrides 覆盖优先级 + _get_architecture。
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
    """arch_overrides take precedence over the default role mapping."""

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
    """ShardingPlanner._get_architecture resolution priority and fallbacks."""

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
    """Head-sharding detection from q/k/v weight placements."""

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
    """TP-local cached head-count rewriting (D-17)."""

    def test_divide_and_preserve_invariants(self):
        """Dividing head counts preserves head_dim and the config object."""
        attn = TinyLlamaAttention(TinyConfig())   # num_heads=4, head_dim=4
        attn.num_key_value_heads = 4
        attn.num_index_heads = 8
        n = update_module_head_counts(attn, 2, "self_attn")
        assert n == 3
        assert attn.num_heads == 2
        assert attn.num_key_value_heads == 2
        assert attn.num_index_heads == 4
        assert attn.head_dim == 4                     # 头维不切
        assert attn.config.num_attention_heads == 4   # config 不改写
        assert attn._hp_full_head_counts == {
            "num_heads": 4, "num_index_heads": 8,
            "num_key_value_heads": 4}

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
        """A non-divisible head count warns once and stays unchanged."""
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

    def test_user_tp_attr_divide_is_idempotent(self):
        """User-declared TP-local attributes divide once and are then stable."""
        attn = TinyLlamaAttention(TinyConfig())
        attn.hidden_size = 16
        assert _update_user_tp_attrs(
            attn, ("hidden_size",), 2, "self_attn") == 1
        assert attn.hidden_size == 8
        assert _update_user_tp_attrs(
            attn, ("hidden_size",), 2, "self_attn") == 0
        assert attn.hidden_size == 8
        with pytest.raises(ValueError, match="incompatible with tp_size=4"):
            _update_user_tp_attrs(
                attn, ("hidden_size",), 4, "self_attn")


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
    (ParamRole.COLWISE, "q_proj.bias", Shard(0)),
    (ParamRole.BIAS, "o_proj.bias", Replicate()),
    (ParamRole.BIAS, "unmatched.bias", Replicate()),
])
def test_role_to_tp_placement(role, path, tp_want):
    """Each role maps to the expected TP placement for a dense boundary."""
    out = _placement_for_role(path, role, T, has_tp=True, has_ep=False)
    assert out[TP] == tp_want
    # CP 维参数恒 Replicate；EP 维非 MoE 参数 Replicate
    assert out[CP] == Replicate()
    assert out[EP] == Replicate()


def test_moe_expert_ep_shard_tp_by_name():
    """D-08：per-expert 2D 布局 → 标准 Shard(0)/Shard(1)。"""
    moe_t = TEMPLATES["moe_mlp"]
    w1 = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=2)
    assert w1[EP] == Shard(0) and w1[TP] == Shard(0)
    w2 = _placement_for_role("experts.w2", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=2)
    assert w2[EP] == Shard(0) and w2[TP] == Shard(1)


def test_moe_expert_3d_batched_tp_dims_shifted():
    """D-08：3D batched [E, H_out, H_in] → colwise=Shard(1)、rowwise=Shard(2)。"""
    moe_t = TEMPLATES["moe_mlp"]
    w1 = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=3)
    assert w1[EP] == Shard(0) and w1[TP] == Shard(1)
    w2 = _placement_for_role("experts.w2", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=3)
    assert w2[EP] == Shard(0) and w2[TP] == Shard(2)


def test_moe_expert_no_tp_explicit_replicate():
    """05 §3.5 NOTE：has_tp=False 时 MOE_EXPERT 仍显式 TP:Replicate。"""
    moe_t = TEMPLATES["moe_mlp"]
    out = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                                has_tp=False, has_ep=True)
    assert out[TP] == Replicate()
    assert out[EP] == Shard(0)


def test_shared_expert_ep_replicate():
    moe_t = TEMPLATES["moe_mlp"]
    w1 = _placement_for_role("shared_experts.w1", ParamRole.SHARED_EXPERT,
                               moe_t, True, True)
    assert w1[EP] == Replicate() and w1[TP] == Shard(0)
    w2 = _placement_for_role("shared_experts.w2", ParamRole.SHARED_EXPERT,
                               moe_t, True, True)
    assert w2[EP] == Replicate() and w2[TP] == Shard(1)


def test_special_and_skip_return_none():
    assert _placement_for_role("a_log", ParamRole.SPECIAL, T, True, False) is None
    assert _placement_for_role("inv_freq", ParamRole.SKIP, T, True, False) is None


def test_has_tp_false_drops_tp_key_for_dense():
    out = _placement_for_role("q_proj.weight", ParamRole.COLWISE, T,
                                has_tp=False, has_ep=False)
    assert TP not in out
    assert out[CP] == Replicate()


def test_has_ep_false_drops_ep_key_for_expert():
    moe_t = TEMPLATES["moe_mlp"]
    out = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
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
    """Phase 3 boundary-type inference is driven by the role table."""
    assert P._infer_boundary_type(fqn, group) == want


# ==========================================================================
# 来源: test_s1_mla_deepseek.py
# S1.14: DeepSeek MLA family sharding rules（ModelAdapterSpec.sharding_rules + ParamRole.REPLICATED）。
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
    """DeepSeek MLA family sharding rules apply to both architecture spellings."""

    def test_arch_overrides_registered_both_spellings(self):
        """architectures 拼写（deepseekv3）与 model_type 拼写（deepseek_v3）
        均解析到同一份 MLA 规则（family 注册）；v2/v3 同构。"""
        for key in ("deepseekv2", "deepseekv3", "deepseek_v2", "deepseek_v3"):
            spec = get_model_adapter(key)
            assert spec is not None and spec.sharding_rules is not None
            roles = [r for _, r in spec.sharding_rules()]
            assert ParamRole.REPLICATED in roles
            assert ParamRole.COLWISE in roles

    def test_classifier_mla_roles(self):
        """q_a/kv_a → REPLICATED；q_b/kv_b → COLWISE；o_proj → ROWWISE（默认规则）。"""
        model = _TinyDeepseek()
        roles = ShardingPlanner()._classify_all_params(model, "deepseekv3")
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
        roles = ShardingPlanner()._classify_all_params(model, "deepseek_v3")
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
    """End-to-end MLA planning produces the expected attention boundaries."""

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
    out = _collect_special_handlers(roles, P._special_handler_patterns)
    assert out == {"model.layers.0.gated_delta.a_log": "gated_delta_tp_shard"}


def test_unregistered_pattern_defaults():
    out = _collect_special_handlers({"m.x.special_w": ParamRole.SPECIAL}, {})
    assert out == {"m.x.special_w": "default"}


def test_non_special_roles_ignored():
    out = _collect_special_handlers({
        "a.b.weight": ParamRole.COLWISE,
        "a.c.weight": ParamRole.SKIP,
    }, P._special_handler_patterns)
    assert not out


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
    """validate_model_compatibility fails fast on non-divisible topologies."""

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


# ==========================================================================
# 来源: test_s1_deferred_bias.py
# S1.9: D-22 rowwise bias 后置——_deferred_bias_params 标记检测 / fail-fast /
# WARNING（检测锚定最终 spec 声明 + 模型结构，与 ParamRole 无关）。
# ==========================================================================

class _TinyBiasAttention(nn.Module):
    """q/k/v/o_proj 全部带 bias 的 toy attention（OPT/GPT-NeoX 风格）。"""

    def __init__(self, h=8):
        super().__init__()
        self.q_proj = nn.Linear(h, h, bias=True)
        self.k_proj = nn.Linear(h, h, bias=True)
        self.v_proj = nn.Linear(h, h, bias=True)
        self.o_proj = nn.Linear(h, h, bias=True)

    def forward(self, hidden_states):
        return self.o_proj(
            self.q_proj(hidden_states) + self.k_proj(hidden_states)
            + self.v_proj(hidden_states))


class _TinyBiasAttnModel(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.config = _Cfg(architectures=["TinyBiasForCausalLM"],
                           model_type="tiny_bias")
        self.self_attn = _TinyBiasAttention(h)

    def forward(self, hidden_states):
        return self.self_attn(hidden_states)


class _CustomLinear(nn.Module):
    """非 nn.Linear 的带 bias 线性层（自研模块；D-22 WARNING + 跳过路径）。"""

    def __init__(self, h=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(h, h))
        self.bias = nn.Parameter(torch.randn(h))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class _WoBlock(nn.Module):
    """非标准命名的 rowwise Linear 容器（wo 命中不了任何角色规则）。"""

    def __init__(self, h=8):
        super().__init__()
        self.wo = nn.Linear(h, h, bias=True)

    def forward(self, x):
        return self.wo(x)


class _CustomLinearModel(nn.Module):
    def __init__(self, h=8, block=None):
        super().__init__()
        self.config = _Cfg(architectures=["TinyBiasForCausalLM"],
                           model_type="tiny_bias")
        self.block = block if block is not None else _CustomLinear(h)

    def forward(self, x):
        return self.block(x)


def _rowwise_spec(weight_path, extra_params=None, out_src=None):
    """自声明 rowwise 契约（insert/derive=False 形态）。"""
    params = {weight_path: {TP: Shard(1)}}
    params.update(extra_params or {})
    return ModuleShardingSpec(
        params=params,
        in_src={"x": {TP: Replicate()}},
        in_dst={"x": {TP: Replicate()}},
        out_src=out_src or {TP: Partial()},
        out_dst={TP: Replicate()},
    )


class TestDeferredBias:
    """D-22 标记检测（模板推导路径）。"""

    def test_rowwise_bias_deferred(self, make_mesh):
        """o_proj.bias 被标记后置；q/k/v bias 随权重 COLWISE 不后置。"""
        mesh = make_mesh((1,), ("tp",))
        model = _TinyBiasAttnModel()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2)
        spec = plan.modules["self_attn"]
        assert spec._deferred_bias_params == ("o_proj.bias",)
        # D-19：colwise bias 随权重沿输出通道切分（区域内本地加，不经归约）
        for name in ("q_proj.bias", "k_proj.bias", "v_proj.bias"):
            assert spec.params[name][TP] == Shard(0)
        # rowwise bias 保持 Replicate（归约后整体加一次）
        assert spec.params["o_proj.bias"][TP] == Replicate()

    def test_explain_shows_deferred_bias(self, make_mesh):
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(_TinyBiasAttnModel(), mesh, tp_size=2)
        assert "deferred bias" in plan.explain()
        assert "o_proj.bias" in plan.explain()

    def test_no_tp_no_defer(self, make_mesh):
        """无 tp 轴（tp_size=1）→ 无 Partial 归约 → 不后置。"""
        mesh = make_mesh((1,), ("tp",))
        plan = ShardingPlanner().plan(_TinyBiasAttnModel(), mesh, tp_size=1)
        assert plan.modules["self_attn"]._deferred_bias_params == ()

    def test_out_src_non_partial_no_defer(self, make_mesh):
        """merge 覆盖 out_src 为非 Partial（无边界归约）→ bias 本就只加一次，不后置。"""
        mesh = make_mesh((1,), ("tp",))
        overrides = {"self_attn": ModuleShardingSpec(out_src={TP: Replicate()})}
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            _TinyBiasAttnModel(), mesh, tp_size=2)
        assert plan.modules["self_attn"]._deferred_bias_params == ()

    def test_user_insert_spec_rowwise_bias(self, make_mesh):
        """用户自声明 spec（insert，derive=False）：非标准命名 wo 也按声明判定，
        且 bias 无需在 params 中声明（物理存在即被检测）。"""
        mesh = make_mesh((1,), ("tp",))
        overrides = {"block": _rowwise_spec("wo.weight")}
        model = _CustomLinearModel(block=_WoBlock())
        plan = ShardingPlanner(plan_overrides=overrides, derive=False).plan(
            model, mesh, tp_size=2)
        assert plan.modules["block"]._deferred_bias_params == ("wo.bias",)

    def test_bias_declared_non_replicate_fails(self, make_mesh):
        """rowwise 兄弟 + bias 显式声明非 Replicate → fail-fast。"""
        mesh = make_mesh((1,), ("tp",))
        overrides = {"block": _rowwise_spec(
            "wo.weight", extra_params={"wo.bias": {TP: Shard(0)}})}
        with pytest.raises(ValueError, match="Replicate"):
            ShardingPlanner(plan_overrides=overrides, derive=False).plan(
                _CustomLinearModel(block=_WoBlock()), mesh, tp_size=2)

    def test_lm_head_bias_template_mismatch(self, make_mesh):
        """lm_head 带 bias（权重沿输出维 Shard(0) 而 bias 被 BIAS 兜底为
        Replicate）→ plan 期 fail-fast 模板不匹配，而不是运行期形状崩溃。"""
        class _LmHeadBiasModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Cfg(architectures=["TinyBiasForCausalLM"],
                                   model_type="tiny_bias")
                self.lm_head = nn.Linear(8, 8, bias=True)

            def forward(self, x):
                return self.lm_head(x)

        mesh = make_mesh((1,), ("tp",))
        with pytest.raises(ValueError, match="template mismatch"):
            ShardingPlanner().plan(_LmHeadBiasModel(), mesh, tp_size=2)

    def test_non_linear_owner_warns_and_skips(self, make_mesh, caplog):
        """rowwise 契约 + Partial out_src，但 owner 非 nn.Linear → WARNING + 跳过。"""
        mesh = make_mesh((1,), ("tp",))
        overrides = {"block": _rowwise_spec("weight",
                                            extra_params={"bias": {TP: Replicate()}})}
        with caplog.at_level(logging.WARNING):
            plan = ShardingPlanner(plan_overrides=overrides, derive=False).plan(
                _CustomLinearModel(), mesh, tp_size=2)
        assert plan.modules["block"]._deferred_bias_params == ()
        assert any("not nn.Linear" in r.message for r in caplog.records)

    def test_multi_output_partial_fails(self, make_mesh):
        """多输出 + Partial + rowwise bias：无法归因到唯一输出 → fail-fast。"""
        mesh = make_mesh((1,), ("tp",))
        spec = _rowwise_spec(
            "weight",
            out_src={"a": {TP: Partial()}, "b": {TP: Partial()}})
        with pytest.raises(ValueError, match="single-output"):
            ShardingPlanner(plan_overrides={"block": spec}, derive=False).plan(
                _CustomLinearModel(), mesh, tp_size=2)


# ==========================================================================
# F2（accuracy_fix_plan.md §2）：Qwen2-MoE 架构覆盖 —— shared_expert_gate
# 是逐 token 标量门 Linear(H, 1)，"必须复制" ≠ "路由语义"，显式 REPLICATED。
# ==========================================================================

class _TinyQwen2MoeMlp(nn.Module):
    """Qwen2-MoE 结构：gate + experts + shared_expert + shared_expert_gate。"""

    def __init__(self, hidden=8, inter=16):
        super().__init__()
        self.gate = nn.Linear(hidden, 4, bias=False)
        self.experts = nn.Module()          # 容器即可（角色由命名规则判定）
        self.experts.w1 = nn.Parameter(torch.randn(4, inter, hidden))
        self.shared_expert = nn.Linear(hidden, inter, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)


class _TinyQwen2Moe(nn.Module):
    """A minimal Qwen2-MoE-shaped model for architecture-override tests."""

    def __init__(self, architectures=("Qwen2MoeForCausalLM",),
                 model_type="qwen2_moe"):
        super().__init__()
        self.config = _Cfg(architectures=list(architectures),
                           model_type=model_type)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        layer = nn.Module()
        layer.mlp = _TinyQwen2MoeMlp()
        self.model.layers.append(layer)


class TestQwen2MoeArchOverride:
    """Qwen2-MoE shared-expert-gate override registration and classification."""

    def test_arch_overrides_registered_both_spellings(self):
        for key in ("qwen2moe", "qwen2_moe"):
            spec = get_model_adapter(key)
            assert spec is not None and spec.sharding_rules is not None
            assert (["shared_expert_gate"], ParamRole.REPLICATED) in list(
                spec.sharding_rules())

    def test_shared_expert_gate_replicated(self):
        """shared_expert_gate.weight → REPLICATED（不是 MOE_GATE —— 不会
        锚定虚假路由边界；不是 SHARED_EXPERT —— 不会把单行权重 Shard(0)
        成空分片，accuracy_problem.md 10.1）。"""
        model = _TinyQwen2Moe()
        roles = ShardingPlanner()._classify_all_params(model, "qwen2moe")
        p = "model.layers.0.mlp."
        assert roles[p + "shared_expert_gate.weight"] == ParamRole.REPLICATED
        assert roles[p + "shared_expert.weight"] == ParamRole.SHARED_EXPERT
        assert roles[p + "gate.weight"] == ParamRole.MOE_GATE
        assert roles[p + "experts.w1"] == ParamRole.MOE_EXPERT

    def test_model_type_spelling(self):
        """model_type 拼写（qwen2_moe）同样命中覆盖。"""
        model = _TinyQwen2Moe(architectures=())
        roles = ShardingPlanner()._classify_all_params(model, "qwen2_moe")
        assert roles["model.layers.0.mlp.shared_expert_gate.weight"] == (
            ParamRole.REPLICATED)


# ==========================================================================
# 来源: test_dsa_template.py
# S1.15: DSA 注意力结构模板（model_structure.py 的 dsa 家族）——结构发现 →
# Phase 1 角色 / Phase 2 无参数边界补种 / Phase 3 边界类型 / Phase 4 模板。
# ==========================================================================

class _LayoutAttention(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.linear_proj = nn.Linear(8, 8, bias=False)


class _LayoutLayer(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.self_attention = _LayoutAttention(attention_type)


class _LayoutMtpBlock(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.self_attention = _LayoutAttention(attention_type)


class _LayoutMtpLayer(nn.Module):
    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.mtp_block = _LayoutMtpBlock(attention_type)


class _MixedAttentionModel(nn.Module):
    """DSA / MLA / GQA 混排，用来证明 reduce-scatter 轴由父模块决定。"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _LayoutLayer("dsa"),
            _LayoutLayer("mla"),
            _LayoutLayer("gqa"),
            _LayoutMtpLayer("mla"),
        ])


class _TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param_sink_k_pe = nn.Parameter(torch.ones(8, 4))
        self.param_sink_compressed_kv = nn.Parameter(torch.ones(8, 4))


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = _TinyAttention()


class _TinyVlModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(architectures=["UnrelatedArchitecture"])
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_TinyLayer()])


class _PlainLmHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(8, 8, bias=False)


class _HeadCountAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 16
        self.num_index_heads = 24
        self.num_key_value_heads = 1
        self.linear_qb = nn.Linear(8, 8, bias=False)


class _HeadCountModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].self_attention = _HeadCountAttention()


class _AuxAttention(nn.Module):
    """DSA 注意力根 + 它那些无参数的辅助叶子。"""

    def __init__(self) -> None:
        super().__init__()
        self.attention_type = "dsa"
        self.param_sink_k_pe = nn.Parameter(torch.ones(8, 4))
        self.q_layernorm = nn.LayerNorm(8)
        self.linear_qkv = nn.Linear(8, 8, bias=False)
        self.linear_qb = nn.Linear(8, 8, bias=False)
        self.linear_proj = nn.Linear(8, 8, bias=False)
        self.rotary_emb = nn.Module()          # 无参数
        self.rotary_emb.register_buffer("inv_freq", torch.ones(4))
        self.gather_rotary_emb = nn.Module()   # 无参数
        self.sparse_lightning_indexer_kllloss = nn.Module()   # 无参数


class _AuxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].self_attention = _AuxAttention()


class _TpMesh:
    def __contains__(self, name):
        return name == "tp"

    def __getitem__(self, name):
        assert name == "tp"
        return SimpleNamespace(size=lambda: 2)


def _tiny_mesh():
    """不带进程组构造 TP mesh（显式 ``rank_list``）。"""
    return init_device_mesh(
        "cpu", (2,), mesh_dim_names=("tp",), rank_list=(0, 1),
        init_backend=False,
    )


def test_linear_proj_sequence_axis_follows_attention_runtime_layout():
    """混排的注意力层按各自运行时的序列轴做 reduce-scatter。"""
    plan = ShardingPlanner().plan(_MixedAttentionModel(), _tiny_mesh(), tp_size=2)

    def out_dst(prefix):
        return plan.modules[prefix].out_dst["output"]["tp"]

    assert out_dst("layers.0.self_attention.linear_proj") == Shard(1)
    assert out_dst("layers.1.self_attention.linear_proj") == Shard(0)
    assert out_dst("layers.2.self_attention.linear_proj") == Shard(1)
    assert out_dst("layers.3.mtp_block.self_attention.linear_proj") == Shard(0)


def test_dsa_rules_own_only_sink_parameters():
    """DSA 规则只复制 sink 参数，不越界接管 MHC 之类。"""
    model = _TinyVlModel()

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)

    fqn = "model.language_model.layers.0.self_attention"
    assert set(plan.modules) == {fqn}
    spec = plan.modules[fqn]
    assert spec.is_boundary is False
    assert set(spec.params) == {
        "param_sink_k_pe",
        "param_sink_compressed_kv",
    }
    assert all(
        isinstance(placements["tp"], Replicate)
        for placements in spec.params.values()
    )


def test_dsa_rules_do_not_claim_plain_lm_head_without_dsa_structure():
    """结构规则不得覆盖普通模型的常规边界。"""
    model = _PlainLmHeadModel()

    assert detect_model_structure(model).dsa is None

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)
    # 常规 lm_head 模板保留它序列分片的输入。
    assert plan.modules["lm_head"].in_src["hidden_states"]["tp"] == Shard(1)


def test_dsa_roles_follow_the_boundary_type():
    """Phase 1 按发现到的边界类型给参数角色。"""
    model = _HeadCountModel()
    planner = ShardingPlanner()
    planner._structure = detect_model_structure(model)

    roles = planner._classify_all_params(model, "unrelatedarchitecture")

    assert roles["layers.0.self_attention.linear_qb.weight"] == ParamRole.COLWISE


def test_parameter_sharding_updates_dsa_parent_head_count_owner(monkeypatch):
    """DSA 分片叶子会把父模块缓存的 head count 改成 TP 本地值（D-17）。"""
    model = _HeadCountModel()
    leaf_fqn = "layers.0.self_attention.linear_qb"
    owner_fqn = "layers.0.self_attention"
    spec = SimpleNamespace(
        params={},
        _ep_stack={},
        _ep_size=0,
        _head_count_owner=owner_fqn,
    )
    plan = SimpleNamespace(
        modules={leaf_fqn: spec},
        mesh_dim_names=("tp",),
    )
    monkeypatch.setattr(parameter_sharding, "_shard_module_params", lambda *args: None)

    # 直接跑内部 applier，证明 owner 已接通。
    parameter_sharding._shard_planned_parameters(
        [model], plan, _TpMesh(), expert_mesh=None, validate_mode=False)

    attention = model.layers[0].self_attention
    assert attention.num_heads == 8
    assert attention.num_index_heads == 12
    # MQA 的单个共享 KV head 复制而不是被 TP 除。
    assert attention.num_key_value_heads == 1
    assert attention._hp_full_head_counts == {
        "num_heads": 16,
        "num_key_value_heads": 1,
        "num_index_heads": 24,
    }


def test_dsa_linear_qb_tags_the_parent_as_head_count_owner():
    """head-count owner 由持有该投影的叶子自己声明。"""
    plan = ShardingPlanner().plan(_HeadCountModel(), _tiny_mesh(), tp_size=2)

    spec = plan.modules["layers.0.self_attention.linear_qb"]
    assert spec._head_count_owner == "layers.0.self_attention"
    assert spec.params["weight"]["tp"] == Shard(0)
    assert spec.in_src["input"]["tp"] == Shard(1)
    assert spec.in_dst["input"]["tp"] == Replicate()
    assert spec.out_dst["output"]["tp"] == Shard(-1)


def test_dsa_key_sharded_and_identity_leaves_get_their_contracts():
    """head 分片 / 直接消费 / 恒等 三类叶子各自持不同合同。"""
    plan = ShardingPlanner().plan(_AuxModel(), _tiny_mesh(), tp_size=2)

    head = plan.modules["layers.0.self_attention.linear_qb"]
    assert head.params["weight"]["tp"] == Shard(0)
    identity = plan.modules["layers.0.self_attention.linear_qkv"]
    assert identity.params["weight"]["tp"] == Replicate()
    assert identity.in_dst["input"]["tp"] == Shard(1)
    assert identity.out_dst["output"]["tp"] == Shard(1)
    layernorm = plan.modules["layers.0.self_attention.q_layernorm"]
    assert layernorm.params["weight"]["tp"] == Replicate()
    assert layernorm.in_dst["hidden_states"]["tp"] == Shard(1)


def test_parameterless_dsa_boundaries_are_seeded():
    """无参数的边界只能来自 Phase 2 的补种。"""
    model = _AuxModel()

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)

    for leaf in ("rotary_emb", "gather_rotary_emb",
                 "sparse_lightning_indexer_kllloss"):
        spec = plan.modules[f"layers.0.self_attention.{leaf}"]
        assert spec.params == {}
        assert spec.is_boundary is True
    assert plan.modules[
        "layers.0.self_attention.rotary_emb"].in_src["t"]["tp"] == Replicate()
    indexer = plan.modules["layers.0.self_attention.sparse_lightning_indexer_kllloss"]
    assert indexer.in_src["query"]["tp"] == Shard(1)
    assert indexer.in_dst["query"]["tp"] == Replicate()
    # ...同时每个可训练参数仍被覆盖。
    declared = {
        f"{fqn}.{param_name}"
        for fqn, spec in plan.modules.items()
        for param_name in (spec.params or {})
    }
    assert set(dict(model.named_parameters())) == declared


def test_dsa_contracts_follow_the_sequence_parallel_switch():
    """关 SP 后没有序列分片可保：恒等合同退化为 Replicate。"""
    plan = ShardingPlanner().plan(
        _AuxModel(), _tiny_mesh(), tp_size=2, sequence_parallel=False)

    identity = plan.modules["layers.0.self_attention.linear_qkv"]
    assert isinstance(identity.in_src["input"]["tp"], Replicate)
    assert isinstance(identity.out_dst["output"]["tp"], Replicate)
    proj = plan.modules["layers.0.self_attention.linear_proj"]
    assert isinstance(proj.out_dst["output"]["tp"], Replicate)
    head = plan.modules["layers.0.self_attention.linear_qb"]
    assert isinstance(head.in_src["input"]["tp"], Replicate)


# ==========================================================================
# 来源: test_mhc_template.py
# S1.15: MHC 结构模板——子树内每个物理 owner 各自登记为 param-only spec。
# ==========================================================================

class _TinyMhc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch_alpha = nn.Parameter(torch.ones(4))
        self.branch_beta = nn.Parameter(torch.ones(4))
        self.norm_gamma = nn.Parameter(torch.ones(16))
        self.phi = nn.Linear(16, 4, bias=False)


class _TinySinkAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param_sink_k_pe = nn.Parameter(torch.ones(8, 4))
        self.param_sink_compressed_kv = nn.Parameter(torch.ones(8, 4))


class _TinyMhcLayer(nn.Module):
    def __init__(self, *, merge: bool) -> None:
        super().__init__()
        self.self_attention = _TinySinkAttention()
        self.attn_mhc_pre_module = _TinyMhc()
        self.mlp_mhc_pre_module = _TinyMhc()
        if merge:
            self.merge_mhc_module = _TinyMhc()


class _TinyMhcVlModel(nn.Module):
    """最小 VL 形状模型：结构与 config.architectures 无关。"""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["UnrelatedArchitecture"],
        )
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([
            _TinyMhcLayer(merge=False),
            _TinyMhcLayer(merge=True),
        ])


class _LookalikeModel(nn.Module):
    """名字里含 MHC 字样、但没有 MHC 结构的模块。"""

    def __init__(self) -> None:
        super().__init__()
        self.attn_mhc_pre_module_like = nn.Linear(4, 4, bias=False)
        self.attn_mhc_pre_module_empty = nn.Module()


def test_mhc_discovery_owns_only_mhc_parameter_modules():
    """发现按 MHC 段名 + 直接持有参数判定。"""
    structure = detect_model_structure(_TinyMhcVlModel())

    facts = structure.mhc
    assert facts
    assert all("mhc_" in fqn for fqn in facts)
    # 嵌套投影自己是物理 owner。
    assert "model.language_model.layers.0.attn_mhc_pre_module.phi" in facts

    # 形似段名不会被接管："attn_mhc_pre_module_like" 不是整段匹配，而没有
    # 参数的 MHC 同名模块不会让家族成立。
    assert detect_model_structure(_LookalikeModel()).mhc is None


def test_mhc_params_are_replicated_role_and_boundary_type():
    """Phase 1 给 REPLICATED，Phase 3 给 mhc_params 类型。"""
    model = _TinyMhcVlModel()
    structure = detect_model_structure(model)
    planner = ShardingPlanner()
    planner._structure = structure

    roles = planner._classify_all_params(model, "unrelatedarchitecture")
    mhc_roles = {name: role for name, role in roles.items() if "mhc" in name}
    assert mhc_roles
    assert set(mhc_roles.values()) == {ParamRole.REPLICATED}

    fqn = "model.language_model.layers.0.attn_mhc_pre_module"
    group = [(f"{fqn}.branch_alpha", ParamRole.REPLICATED)]
    assert planner._infer_boundary_type(fqn, group) == MHC_PARAMS


def test_planner_composes_dsa_and_mhc_rules_by_structure():
    """两个独立的结构模板共同覆盖各自的参数。"""
    model = _TinyMhcVlModel()

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)

    declared = {}
    for fqn, spec in plan.modules.items():
        for param_name, placements in (spec.params or {}).items():
            declared[f"{fqn}.{param_name}"] = placements
            assert spec.is_boundary is False
            assert isinstance(placements["tp"], Replicate)

    assert set(declared) == set(dict(model.named_parameters()))


def test_mhc_boundaries_are_param_only_specs():
    """MHC 模块只声明参数，不带 I/O 合同。"""
    model = _TinyMhcVlModel()

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)

    fqn = "model.language_model.layers.1.merge_mhc_module"
    spec = plan.modules[fqn]
    assert spec.is_boundary is False
    assert set(spec.params) == {"branch_alpha", "branch_beta", "norm_gamma"}
    # param-only 边界不落任何合同。
    assert spec.in_src == {}
    assert spec.in_dst == {}
    assert spec.out_src is None
    assert spec.out_dst is None


# ==========================================================================
# 来源: test_mtp_template.py
# S1.15: MTP 结构模板——prev_proj 复制权重、保住序列分片；同名无结构不接管。
# ==========================================================================

class _TinyMtpLayer(nn.Module):
    def __init__(self) -> None:
        """构造一个 MTP 层完整的结构契约。"""
        super().__init__()
        self.mtp_block = nn.Identity()
        self.prev_norm = nn.LayerNorm(8)
        self.emb_norm = nn.LayerNorm(8)
        self.prev_proj = nn.Linear(16, 8, bias=False)


class _LookalikeLayer(nn.Module):
    def __init__(self) -> None:
        """同名但没有 MTP 结构的投影。"""
        super().__init__()
        self.prev_proj = nn.Linear(16, 8, bias=False)


class _TinyMtpModel(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        """把候选层放在规范 decoder layer 路径上。"""
        super().__init__()
        self.config = SimpleNamespace(architectures=["UnrelatedArchitecture"])
        self.layers = nn.ModuleList([layer])


def test_mtp_prev_proj_is_replicated_with_sequence_shards():
    """MTP prev_proj 的权重在 TP 上保持复制。"""
    model = _TinyMtpModel(_TinyMtpLayer())

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)

    spec = plan.modules["layers.0.prev_proj"]
    assert isinstance(spec.params["weight"]["tp"], Replicate)
    assert spec.in_src["input"]["tp"] == Shard(1)
    assert spec.in_dst["input"]["tp"] == Shard(1)
    assert spec.out_src["output"]["tp"] == Shard(1)
    assert spec.out_dst["output"]["tp"] == Shard(1)


def test_mtp_rules_reject_a_prev_proj_name_without_mtp_structure():
    """匹配是结构性的：光有一个 prev_proj 名字不会被接管。"""
    model = _TinyMtpModel(_LookalikeLayer())

    assert detect_model_structure(model).mtp is None


def test_mtp_role_rule_replicates_prev_proj():
    """Phase 1 把发现到的 prev_proj 参数判成复制。"""
    model = _TinyMtpModel(_TinyMtpLayer())
    planner = ShardingPlanner()
    planner._structure = detect_model_structure(model)

    roles = planner._classify_all_params(model, "unrelatedarchitecture")

    assert roles["layers.0.prev_proj.weight"] == ParamRole.REPLICATED
    # MTP 层的 norm 保持它原来的角色。
    assert roles["layers.0.prev_norm.weight"] == ParamRole.NORM

    group = [("layers.0.prev_proj.weight", ParamRole.REPLICATED)]
    assert planner._infer_boundary_type(
        "layers.0.prev_proj", group) == MTP_PREV_PROJ


def test_mtp_contracts_follow_the_sequence_parallel_switch():
    """关 SP 后没有序列分片可保：恒等退化为 Replicate。"""
    model = _TinyMtpModel(_TinyMtpLayer())

    plan = ShardingPlanner().plan(
        model, _tiny_mesh(), tp_size=2, sequence_parallel=False)

    spec = plan.modules["layers.0.prev_proj"]
    assert isinstance(spec.params["weight"]["tp"], Replicate)
    assert isinstance(spec.in_src["input"]["tp"], Replicate)
    assert isinstance(spec.in_dst["input"]["tp"], Replicate)
    assert isinstance(spec.out_src["output"]["tp"], Replicate)
    assert isinstance(spec.out_dst["output"]["tp"], Replicate)


def test_planner_covers_all_mtp_specific_trainable_parameters():
    """plan 声明了每个 MTP 专有可训练参数的切分。"""
    model = _TinyMtpModel(_TinyMtpLayer())

    plan = ShardingPlanner().plan(model, _tiny_mesh(), tp_size=2)

    declared = {
        f"{fqn}.{param_name}"
        for fqn, spec in plan.modules.items()
        for param_name in (spec.params or {})
    }
    assert set(dict(model.named_parameters())) == declared
    assert isinstance(plan.modules["layers.0.prev_proj"].params["weight"]["tp"], Replicate)


# ==========================================================================
# 来源: test_shared_expert_template.py
# S1.15: dispatcher 持有的共享专家——TP 参与 EP 分发域，隐藏维不得再切。
# ==========================================================================

class _SharedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(8, 16)
        self.linear_fc2 = nn.Linear(16, 8)


class _DispatcherMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8)])
        self.token_dispatcher = object()
        self.shared_expert = _SharedExpert()


class _DispatcherMoEModel(nn.Module):
    def __init__(self, moe: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([moe])


def _plan_dispatcher_moe(model, **kwargs):
    """规划 *model*，容忍 fixture 里没被覆盖的 routed expert。"""
    # ``experts.0`` 是数字容器下的裸 Linear，自己不成边界（这个 fixture 只
    # 用来验证共享专家规则）。
    return ShardingPlanner(allow_uncovered_params=True).plan(
        model, _tiny_mesh(), tp_size=2, **kwargs)


def test_shared_expert_linears_are_replicated_and_preserve_sequence_shards():
    """共享专家参数保持复制，同时保住它们的序列分片。"""
    model = _DispatcherMoEModel(_DispatcherMoE())

    plan = _plan_dispatcher_moe(model)

    expected = {
        "layers.0.shared_expert.linear_fc1",
        "layers.0.shared_expert.linear_fc2",
    }
    assert expected <= set(plan.modules)
    for fqn in expected:
        spec = plan.modules[fqn]
        assert all(isinstance(value["tp"], Replicate) for value in spec.params.values())
        assert spec.in_src["hidden_states"]["tp"] == Shard(1)
        assert spec.in_dst["hidden_states"]["tp"] == Shard(1)
        assert spec.out_src["output"]["tp"] == Shard(1)
        assert spec.out_dst["output"]["tp"] == Shard(1)


def test_shared_expert_rules_reject_non_dispatcher_mlp():
    """没有 dispatcher 的 MLP 不被接管。"""
    model = _DispatcherMoEModel(_SharedExpert())

    assert detect_model_structure(model).shared_expert is None


def test_shared_expert_roles_are_replicated_not_shared_expert():
    """Phase 1 覆盖默认的 SHARED_EXPERT 角色（仅限 dispatcher 持有的）。"""
    model = _DispatcherMoEModel(_DispatcherMoE())
    planner = ShardingPlanner()
    planner._structure = detect_model_structure(model)

    roles = planner._classify_all_params(model, "")

    assert roles["layers.0.shared_expert.linear_fc1.weight"] == ParamRole.REPLICATED
    assert roles["layers.0.shared_expert.linear_fc2.bias"] == ParamRole.REPLICATED

    group = [("layers.0.shared_expert.linear_fc1.weight", ParamRole.REPLICATED)]
    assert planner._infer_boundary_type(
        "layers.0.shared_expert.linear_fc1", group) == SHARED_EXPERT_LINEAR


def test_shared_expert_contracts_follow_the_sequence_parallel_switch():
    """关 SP 后恒等合同退化为 Replicate。"""
    model = _DispatcherMoEModel(_DispatcherMoE())

    plan = _plan_dispatcher_moe(model, sequence_parallel=False)

    spec = plan.modules["layers.0.shared_expert.linear_fc1"]
    assert isinstance(spec.in_src["hidden_states"]["tp"], Replicate)
    assert isinstance(spec.out_dst["output"]["tp"], Replicate)
