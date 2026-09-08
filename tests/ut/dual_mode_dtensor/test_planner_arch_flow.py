# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s1_plan_arch.py: merged core suite file (feature-combined, 15 cases).

Sources: test_s1_arch_override.py, test_s1_head_count.py, test_s1_role_mapping.py, test_s1_semantic_infer.py, test_s1_mla_deepseek.py, test_s1_special_handlers.py, test_s1_compat.py, test_s1_sp_loss_matrix.py
"""

import logging
import pytest
import torch
from torch import nn
from hyper_parallel.distributed.recipe_spec import (
    CP,
    EP,
    ModuleShardingSpec,
    TP,
    resolve_placements,
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
from hyper_parallel.distributed._builder.default_templates import TEMPLATES
from hyper_parallel.distributed._builder.forward_rewriter import (
    _add_bias_to_primary_output,
)
from hyper_parallel.distributed._builder.planner import (
    ShardingPlanner,
    validate_model_compatibility,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.distributed._builder.default_templates import (
    _placement_for_role,
)
from hyper_parallel.distributed._builder.special_handlers import (
    SPECIAL_HANDLERS,
    _collect_special_handlers,
)
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


# ==========================================================================
# _add_bias_to_primary_output: tuple passthrough + fail-fast on non-Tensor
# primary output.
# ==========================================================================

def test_deferred_bias_output_helper():
    """_add_bias_to_primary_output: tuple passthrough and fail-fast on non-Tensor primary output."""
    # case: preserves_attention_tuple_output -- bias only added to hidden states, metadata untouched
    hidden_states = torch.zeros(2, 3, 4)
    attention_weights = object()
    bias = torch.arange(4, dtype=hidden_states.dtype)

    output = _add_bias_to_primary_output(
        (hidden_states, attention_weights), bias, "TinyAttention")

    assert isinstance(output, tuple), "case: preserves_attention_tuple_output"
    torch.testing.assert_close(output[0], hidden_states + bias)
    assert output[1] is attention_weights, "case: preserves_attention_tuple_output"

    # case: rejects_non_tensor_primary_output -- malformed structured output fails before touching the wrong field
    with pytest.raises(TypeError, match="output index 0 to be a Tensor"):
        _add_bias_to_primary_output((None, torch.ones(4)), torch.ones(4), "TinyAttention")


# ==========================================================================
# Source: test_s1_arch_override.py
# S1.2: arch-override priority (classifier arch_overrides) + _get_architecture.
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


def test_arch_override_priority():
    """ParameterClassifier: four paths of override hit / pattern list / miss / unknown arch."""
    cases = [
        # case: override_beats_default -- embed_tokens defaults to EMBED, forced to SKIP
        ("override_beats_default",
         {"myarch": [("embed_tokens.weight", ParamRole.SKIP)]}, "myarch",
         "token_embed.weight", ParamRole.SKIP),
        # case: override_list_of_patterns -- any sub-pattern hit applies the override
        ("override_list_of_patterns",
         {"myarch": [(["token_embed", "word_embed"], ParamRole.EMBED)]}, "myarch",
         "token_embed.weight", ParamRole.EMBED),
        # case: default_when_no_override_match -- miss falls back to default (non-standard name -> SKIP)
        ("default_when_no_override_match",
         {"myarch": [("token_embed", ParamRole.EMBED)]}, "myarch",
         "output_head.weight", ParamRole.SKIP),
        # case: unknown_arch_falls_back_to_default
        ("unknown_arch_falls_back_to_default",
         {"other": [("token_embed", ParamRole.EMBED)]}, "myarch",
         "token_embed.weight", ParamRole.SKIP),
    ]
    for name, overrides, arch, param, want in cases:
        clf = ParameterClassifier(arch_overrides=overrides)
        roles = clf.classify(_Model(), arch)
        assert roles[param] == want, f"case: {name}"


def test_get_architecture():
    """ShardingPlanner._get_architecture: architectures/model_type/class-name fallback and suffix stripping."""
    planner = ShardingPlanner()

    # case: architectures_first
    m = _Model(config=_Cfg(architectures=["Qwen2ForCausalLM"], model_type="qwen2"))
    assert planner._get_architecture(m) == "qwen2", "case: architectures_first"

    # case: model_type_fallback
    m = _Model(config=_Cfg(architectures=None, model_type="mixtral"))
    assert planner._get_architecture(m) == "mixtral", "case: model_type_fallback"

    # case: classname_fallback
    class LlamaForCausalLM(nn.Module):
        config = None
    assert planner._get_architecture(LlamaForCausalLM()) == "llama", \
        "case: classname_fallback"

    # case: suffix_stripping
    for cls_name, want in [
        ("LlamaForCausalLM", "llama"),
        ("Blip2ForConditionalGeneration", "blip2"),
        ("BertForSequenceClassification", "bert"),
        ("PaliGemmaForImageTextToText", "paligemma"),
    ]:
        cls = type(cls_name, (nn.Module,), {"config": None})
        assert planner._get_architecture(cls()) == want, \
            f"case: suffix_stripping[{cls_name}]"

    # case: no_config_attribute
    class Tiny(nn.Module):
        pass
    assert planner._get_architecture(Tiny()) == "tiny", "case: no_config_attribute"


# ==========================================================================
# Source: test_s1_head_count.py
# S1.14: head_count -- TP local head-count rewrite (D-17).
# ==========================================================================

def _spec(**params):
    return ModuleShardingSpec(params=params)


def test_is_head_sharded():
    """Five verdicts: qkv colwise / MLA q_b up-projection / mlp / no tp axis / rowwise-only."""
    cases = [
        # case: qkv_colwise_detected
        ("qkv_colwise_detected", {
            "q_proj.weight": {TP: Shard(0)},
            "k_proj.weight": {TP: Shard(0)},
            "v_proj.weight": {TP: Shard(0)},
            "o_proj.weight": {TP: Shard(1)},
        }, ("tp",), True),
        # case: mla_q_b_proj_detected -- D-14 MLA: q_b_proj colwise on head dim -> hit
        ("mla_q_b_proj_detected", {
            "q_a_proj.weight": {TP: Replicate()},
            "q_b_proj.weight": {TP: Shard(0)},
            "kv_b_proj.weight": {TP: Shard(0)},
        }, ("tp",), True),
        # case: mlp_not_detected
        ("mlp_not_detected", {
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        }, ("tp",), False),
        # case: no_tp_axis_not_detected
        ("no_tp_axis_not_detected", {
            "q_proj.weight": {TP: Shard(0), CP: Replicate()},
        }, ("cp",), False),
        # case: rowwise_only_not_detected -- q/k/v all Replicate (fused QKV rowwise)
        ("rowwise_only_not_detected", {
            "q_proj.weight": {TP: Replicate()},
            "o_proj.weight": {TP: Shard(1)},
        }, ("tp",), False),
    ]
    for name, params, axes, want in cases:
        assert _is_head_sharded(_spec(**params), axes) is want, f"case: {name}"


class _NamedAttrAttention(nn.Module):
    """transformers naming-variant coverage: n_heads / num_kv_heads (falcon style)."""

    def __init__(self):
        super().__init__()
        self.n_heads = 8
        self.num_kv_heads = 2
        self.num_key_value_groups = 4
        self.head_dim = 16
        self.config = TinyConfig(num_attention_heads=8)


def test_update_module_head_counts(caplog):
    """update_module_head_counts / _update_user_tp_attrs: all branches."""

    # case: divide_and_preserve_invariants -- head counts divided by tp; head_dim/config untouched
    attn = TinyLlamaAttention(TinyConfig())   # num_heads=4, head_dim=4
    attn.num_key_value_heads = 4
    n = update_module_head_counts(attn, 2, "self_attn")
    assert n == 2, "case: divide_and_preserve_invariants"
    assert attn.num_heads == 2, "case: divide_and_preserve_invariants"
    assert attn.num_key_value_heads == 2, "case: divide_and_preserve_invariants"
    assert attn.head_dim == 4, "case: divide_and_preserve_invariants"           # head dim not sharded
    assert attn.config.num_attention_heads == 4, \
        "case: divide_and_preserve_invariants"                                  # config not rewritten
    assert attn._hp_full_head_counts == {
        "num_heads": 4, "num_key_value_heads": 4}, \
        "case: divide_and_preserve_invariants"

    # case: name_variants -- num_kv_heads=2 not divisible by tp=4 -> warn only, keep original
    attn = _NamedAttrAttention()
    n = update_module_head_counts(attn, 4, "self_attn")
    assert n == 1, "case: name_variants"
    assert attn.n_heads == 2, "case: name_variants"
    assert attn.num_kv_heads == 2, "case: name_variants"

    # case: num_key_value_groups_untouched -- ratio invariant, never touched
    attn = _NamedAttrAttention()
    update_module_head_counts(attn, 2, "self_attn")
    assert attn.n_heads == 4, "case: num_key_value_groups_untouched"
    assert attn.num_kv_heads == 1, "case: num_key_value_groups_untouched"
    assert attn.num_key_value_groups == 4, "case: num_key_value_groups_untouched"

    # case: idempotent -- no double division
    attn = TinyLlamaAttention(TinyConfig())
    assert update_module_head_counts(attn, 2) == 1, "case: idempotent"
    assert update_module_head_counts(attn, 2) == 0, "case: idempotent"
    assert attn.num_heads == 2, "case: idempotent"

    # case: non_divisible_warns_and_keeps -- warns once, does not warn again
    attn = TinyLlamaAttention(TinyConfig())   # num_heads=4
    with caplog.at_level(logging.WARNING):
        n = update_module_head_counts(attn, 3, "self_attn")
    assert n == 0, "case: non_divisible_warns_and_keeps"
    assert attn.num_heads == 4, "case: non_divisible_warns_and_keeps"
    assert "not divisible" in caplog.text, "case: non_divisible_warns_and_keeps"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        update_module_head_counts(attn, 3, "self_attn")
    assert "not divisible" not in caplog.text, "case: non_divisible_warns_and_keeps"

    # case: tp1_noop
    attn = TinyLlamaAttention(TinyConfig())
    assert update_module_head_counts(attn, 1) == 0, "case: tp1_noop"
    assert attn.num_heads == 4, "case: tp1_noop"
    assert not hasattr(attn, "_hp_full_head_counts"), "case: tp1_noop"

    # case: user_tp_attr_divide_is_idempotent
    attn = TinyLlamaAttention(TinyConfig())
    attn.hidden_size = 16
    assert _update_user_tp_attrs(
        attn, ("hidden_size",), 2, "self_attn") == 1, \
        "case: user_tp_attr_divide_is_idempotent"
    assert attn.hidden_size == 8, "case: user_tp_attr_divide_is_idempotent"
    assert _update_user_tp_attrs(
        attn, ("hidden_size",), 2, "self_attn") == 0, \
        "case: user_tp_attr_divide_is_idempotent"
    assert attn.hidden_size == 8, "case: user_tp_attr_divide_is_idempotent"
    with pytest.raises(ValueError, match="incompatible with tp_size=4"):
        _update_user_tp_attrs(attn, ("hidden_size",), 4, "self_attn")


# ==========================================================================
# Source: test_s1_role_mapping.py
# S1.6: _build_spec_from_template 13 roles -> placement mapping.
# ==========================================================================

P = ShardingPlanner()


T = TEMPLATES["attention"]


def test_role_to_placement():
    """_placement_for_role: role->TP matrix + MoE expert dim shift + has_tp/has_ep variants."""

    # role -> TP placement matrix (CP always Replicate; non-MoE params EP Replicate)
    tp_matrix = [
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
    ]
    for role, path, tp_want in tp_matrix:
        out = _placement_for_role(path, role, T, has_tp=True, has_ep=False)
        assert out[TP] == tp_want, f"case: role_to_tp_placement[{role},{path}]"
        assert out[CP] == Replicate(), f"case: role_to_tp_placement[{role},{path}]"
        assert out[EP] == Replicate(), f"case: role_to_tp_placement[{role},{path}]"

    moe_t = TEMPLATES["moe_mlp"]

    # case: moe_expert_ep_shard_tp_by_name -- D-08 per-expert 2D layout
    w1 = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=2)
    assert w1[EP] == Shard(0) and w1[TP] == Shard(0), \
        "case: moe_expert_ep_shard_tp_by_name"
    w2 = _placement_for_role("experts.w2", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=2)
    assert w2[EP] == Shard(0) and w2[TP] == Shard(1), \
        "case: moe_expert_ep_shard_tp_by_name"

    # case: moe_expert_3d_batched_tp_dims_shifted -- 3D [E, H_out, H_in] dim shift
    w1 = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=3)
    assert w1[EP] == Shard(0) and w1[TP] == Shard(1), \
        "case: moe_expert_3d_batched_tp_dims_shifted"
    w2 = _placement_for_role("experts.w2", ParamRole.MOE_EXPERT, moe_t,
                               True, True, ndim=3)
    assert w2[EP] == Shard(0) and w2[TP] == Shard(2), \
        "case: moe_expert_3d_batched_tp_dims_shifted"

    # case: moe_expert_no_tp_explicit_replicate -- has_tp=False still yields explicit TP:Replicate
    out = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                                has_tp=False, has_ep=True)
    assert out[TP] == Replicate(), "case: moe_expert_no_tp_explicit_replicate"
    assert out[EP] == Shard(0), "case: moe_expert_no_tp_explicit_replicate"

    # case: shared_expert_ep_replicate
    w1 = _placement_for_role("shared_experts.w1", ParamRole.SHARED_EXPERT,
                               moe_t, True, True)
    assert w1[EP] == Replicate() and w1[TP] == Shard(0), \
        "case: shared_expert_ep_replicate"
    w2 = _placement_for_role("shared_experts.w2", ParamRole.SHARED_EXPERT,
                               moe_t, True, True)
    assert w2[EP] == Replicate() and w2[TP] == Shard(1), \
        "case: shared_expert_ep_replicate"

    # case: special_and_skip_return_none
    assert _placement_for_role("a_log", ParamRole.SPECIAL, T, True, False) is None, \
        "case: special_and_skip_return_none"
    assert _placement_for_role("inv_freq", ParamRole.SKIP, T, True, False) is None, \
        "case: special_and_skip_return_none"

    # case: has_tp_false_drops_tp_key_for_dense
    out = _placement_for_role("q_proj.weight", ParamRole.COLWISE, T,
                                has_tp=False, has_ep=False)
    assert TP not in out, "case: has_tp_false_drops_tp_key_for_dense"
    assert out[CP] == Replicate(), "case: has_tp_false_drops_tp_key_for_dense"

    # case: has_ep_false_drops_ep_key_for_expert
    out = _placement_for_role("experts.w1", ParamRole.MOE_EXPERT, moe_t,
                                has_tp=True, has_ep=False)
    assert EP not in out, "case: has_ep_false_drops_ep_key_for_expert"


# ==========================================================================
# Source: test_s1_semantic_infer.py
# S1.4: Phase 3 _infer_boundary_type table-driven cases.
# ==========================================================================

C, R, N = ParamRole.COLWISE, ParamRole.ROWWISE, ParamRole.NORM


def test_infer_boundary_type():
    """Full table: explicit patterns / role combos / leaf guard / MoE / unknown."""
    cases = [
        # explicit patterns
        ("model.embed_tokens", [("x.weight", ParamRole.EMBED)], "embed"),
        ("model.wte", [("x.weight", ParamRole.EMBED)], "embed"),
        ("lm_head", [("x.weight", ParamRole.LM_HEAD)], "lm_head"),
        ("model.embed_out", [("x.weight", ParamRole.LM_HEAD)], "lm_head"),
        ("model.layers.0.input_layernorm", [("x.weight", N)], "norm"),
        ("model.norm", [("x.weight", N)], "norm"),
        ("model.layers.0.mlp.router", [("x.weight", ParamRole.MOE_GATE)], "moe_gate"),
        # role combinations
        ("model.layers.0.self_attn", [("a", C), ("b", C), ("c", C), ("d", R)], "attention"),
        ("model.layers.0.mlp", [("a", C), ("b", C), ("d", R)], "mlp"),
        # colwise+rowwise combo defaults to attention
        ("model.layers.0.block", [("a", C), ("d", R)], "attention"),
        # colwise only -> mlp (requires fqn to hit the mlp pattern)
        ("model.layers.0.mlp", [("a", C), ("b", C)], "mlp"),
        ("model.layers.0.self_attn.q_proj", [("a", C)], "unknown"),  # leaf guard
        # MoE
        ("model.layers.0.mlp", [("a", ParamRole.MOE_GATE), ("b", ParamRole.MOE_EXPERT)],
         "moe_mlp"),
        ("model.layers.0.mlp.experts", [("b", ParamRole.MOE_EXPERT)], "unknown"),  # leaf guard
        # none of the above -> unknown
        ("model.layers.0", [("a", ParamRole.SKIP)], "unknown"),
    ]
    for fqn, group, want in cases:
        assert P._infer_boundary_type(fqn, group) == want, \
            f"case: infer_boundary_type[{fqn}]"


# ==========================================================================
# Source: test_s1_mla_deepseek.py
# S1.14: DeepSeek MLA family sharding rules (ModelAdapterSpec.sharding_rules + ParamRole.REPLICATED).
# ==========================================================================

class _TinyMlaAttention(nn.Module):
    """DeepSeek MLA structure (FQN matches HF DeepseekV2/V3 Attention)."""

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
    """2-layer MLA mini model: FQN mimics HF DeepseekV3 (model.layers.N.self_attn.*)."""

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


def test_mla_arch_override():
    """MLA rules live in the deepseek_v3 family registration (all spellings resolve) + classification hits + no-override regression guard."""

    # case: family_rules_registered_all_spellings -- v2/v3 are isomorphic; every
    # spelling (arch / model_type / canonical) resolves to the family's rules
    for key in ("deepseekv2", "deepseekv3", "deepseek_v2", "deepseek_v3"):
        spec = get_model_adapter(key)
        assert spec is not None and spec.sharding_rules is not None, \
            f"case: family_rules_registered_all_spellings[{key}]"
        roles = [r for _, r in spec.sharding_rules()]
        assert ParamRole.REPLICATED in roles, \
            f"case: family_rules_registered_all_spellings[{key}]"
        assert ParamRole.COLWISE in roles, \
            f"case: family_rules_registered_all_spellings[{key}]"

    # case: classifier_mla_roles -- q_a/kv_a->REPLICATED; q_b/kv_b->COLWISE; o->ROWWISE
    model = _TinyDeepseek()
    roles = ShardingPlanner()._classify_all_params(model, "deepseekv3")
    p = "model.layers.0.self_attn."
    assert roles[p + "q_a_proj.weight"] == ParamRole.REPLICATED, \
        "case: classifier_mla_roles"
    assert roles[p + "kv_a_proj_with_mqa.weight"] == ParamRole.REPLICATED, \
        "case: classifier_mla_roles"
    assert roles[p + "q_b_proj.weight"] == ParamRole.COLWISE, "case: classifier_mla_roles"
    assert roles[p + "kv_b_proj.weight"] == ParamRole.COLWISE, "case: classifier_mla_roles"
    assert roles[p + "o_proj.weight"] == ParamRole.ROWWISE, "case: classifier_mla_roles"
    assert roles[p + "q_a_layernorm.weight"] == ParamRole.NORM, "case: classifier_mla_roles"
    assert roles[p + "kv_a_layernorm.weight"] == ParamRole.NORM, "case: classifier_mla_roles"

    # case: classifier_model_type_spelling -- deepseek_v3 hits as well
    roles = ShardingPlanner()._classify_all_params(_TinyDeepseek(), "deepseek_v3")
    assert roles["model.layers.0.self_attn.q_a_proj.weight"] == ParamRole.REPLICATED, \
        "case: classifier_model_type_spelling"

    # case: without_override_mla_params_skip -- regression guard for the pre-fix silent gap
    clf = ParameterClassifier()   # no arch_overrides
    roles = clf.classify(_TinyDeepseek(), "deepseekv3")
    assert roles[p + "q_a_proj.weight"] == ParamRole.SKIP, \
        "case: without_override_mla_params_skip"
    assert roles[p + "kv_b_proj.weight"] == ParamRole.SKIP, \
        "case: without_override_mla_params_skip"


def test_mla_plan_attention_boundary_and_placements(make_mesh):
    """End-to-end: architectures detection -> override applied -> attention boundary generated;
    REPLICATED fully replicated / q_b,kv_b colwise / o_proj rowwise / cp_attn flag set."""
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
    # attention template flag: inject inner attention wrapper when CP is active
    assert spec._needs_cp_attn is True
    # other boundaries unaffected by the override
    assert "model.layers.0.mlp" in plan.modules
    assert "model.embed_tokens" in plan.modules
    assert plan.modules["lm_head"]._is_terminal is True


def test_mla_plan_fallback_and_layer_coverage(make_mesh):
    """plan variants: fallback to model_type when architectures missing; both MLA layers covered."""

    # case: model_type_fallback_also_hits -- fallback to model_type='deepseek_v3' hits as well
    mesh = make_mesh((1,), ("tp",))
    model = _TinyDeepseek(architectures=())
    model.config.architectures = None
    plan = ShardingPlanner().plan(model, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.self_attn"]
    assert tuple(resolve_placements(
        spec.params["q_b_proj.weight"], ("tp",))) == (Shard(0),), \
        "case: model_type_fallback_also_hits"

    # case: both_layers_sharded -- both MLA layers get an attention spec
    plan = ShardingPlanner().plan(_TinyDeepseek(), mesh, tp_size=2)
    for i in range(2):
        assert f"model.layers.{i}.self_attn" in plan.modules, \
            "case: both_layers_sharded"
        spec = plan.modules[f"model.layers.{i}.self_attn"]
        assert len(spec.params) == 5, "case: both_layers_sharded"   # q_a/q_b/kv_a/kv_b/o


# ==========================================================================
# Source: test_s1_special_handlers.py
# S1.10: Phase 6 _collect_special_handlers + SPECIAL_HANDLERS registry.
# ==========================================================================

def test_special_handlers():
    """SPECIAL role -> handler mapping / unregistered default / non-SPECIAL ignored / registry."""

    # case: special_role_mapped_to_handler
    roles = {
        "model.layers.0.gated_delta.a_log": ParamRole.SPECIAL,
        "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
    }
    out = _collect_special_handlers(roles, P._special_handler_patterns)
    assert out == {"model.layers.0.gated_delta.a_log": "gated_delta_tp_shard"}, \
        "case: special_role_mapped_to_handler"

    # case: unregistered_pattern_defaults
    out = _collect_special_handlers({"m.x.special_w": ParamRole.SPECIAL}, {})
    assert out == {"m.x.special_w": "default"}, "case: unregistered_pattern_defaults"

    # case: non_special_roles_ignored
    out = _collect_special_handlers({
        "a.b.weight": ParamRole.COLWISE,
        "a.c.weight": ParamRole.SKIP,
    }, P._special_handler_patterns)
    assert not out, "case: non_special_roles_ignored"

    # case: special_handlers_registry
    assert "gated_delta_tp_shard" in SPECIAL_HANDLERS, "case: special_handlers_registry"
    assert callable(SPECIAL_HANDLERS["gated_delta_tp_shard"]), \
        "case: special_handlers_registry"


# ==========================================================================
# Source: test_s1_compat.py
# S1.11: validate_model_compatibility.
# ==========================================================================

def _model(**kw):
    return TinyLlamaForCausalLM(TinyConfig(**kw))


def test_validate_model_compatibility():
    """All error paths (table-driven) + the passing path for a valid config."""

    # error paths: each config must raise a ValueError with a matching message
    error_cases = [
        # case: heads_not_divisible
        ("heads_not_divisible",
         {"num_attention_heads": 3}, {"tp_size": 2}, "num_attention_heads"),
        # case: kv_heads_not_divisible
        ("kv_heads_not_divisible",
         {"num_attention_heads": 4, "num_key_value_heads": 3},
         {"tp_size": 2}, "num_key_value_heads"),
        # case: seq_len_not_divisible_2cp
        ("seq_len_not_divisible_2cp",
         {}, {"cp_size": 2, "seq_len": 10}, r"2\*cp"),
        # case: num_experts_not_divisible
        ("num_experts_not_divisible",
         {"num_experts": 3}, {"ep_size": 2}, "num_experts"),
        # case: ep_requires_moe
        ("ep_requires_moe",
         {"num_experts": 0}, {"ep_size": 2}, "MoE"),
        # case: moe_inter_dim_not_divisible_tp
        ("moe_inter_dim_not_divisible_tp",
         {"num_experts": 4, "moe_intermediate_size": 7},
         {"tp_size": 2, "ep_size": 2}, "moe_intermediate_size"),
    ]
    for name, model_kw, plan_kw, match in error_cases:
        try:
            with pytest.raises(ValueError, match=match):
                validate_model_compatibility(_model(**model_kw), **plan_kw)
        except Exception as exc:
            raise AssertionError(f"case: {name} failed: {exc}") from exc

    # case: seq_len_ok
    validate_model_compatibility(_model(), cp_size=2, seq_len=8)

    # case: all_pass
    validate_model_compatibility(
        _model(num_experts=4, moe_intermediate_size=8),
        tp_size=2, cp_size=2, ep_size=2, seq_len=16)


# ==========================================================================
# Source: test_s1_sp_loss_matrix.py
# S1.7: SP on/off x loss_parallel on/off -- four-combo I/O contract.
# ==========================================================================

def _plan(tiny_llama, make_mesh, sp, lp):
    mesh = make_mesh((1,), ("tp",))
    return ShardingPlanner().plan(
        tiny_llama, mesh, tp_size=2, sequence_parallel=sp, loss_parallel=lp)


def test_sp_loss_contract_matrix(tiny_llama, make_mesh):
    """embed / attention / lm_head contracts under the four SP x LP combos + SP CP dim."""
    for sp, lp in [(True, False), (True, True), (False, False), (False, True)]:
        combo = f"sp={sp},lp={lp}"
        plan = _plan(tiny_llama, make_mesh, sp, lp)

        # case: embed_contract -- in Replicate / out_src Partial / out_dst follows SP
        spec = plan.modules["model.embed_tokens"]
        assert spec.in_src["input"][TP] == Replicate(), f"case: embed_contract[{combo}]"
        assert spec.out_src["output"][TP] == Partial(), f"case: embed_contract[{combo}]"
        want_out = Shard(1) if sp else Replicate()
        assert spec.out_dst["output"][TP] == want_out, f"case: embed_contract[{combo}]"

        # case: attention_contract -- hidden_states resharded in/out
        spec = plan.modules["model.layers.0.self_attn"]
        want_in = Shard(1) if sp else Replicate()
        assert spec.in_src["hidden_states"][TP] == want_in, \
            f"case: attention_contract[{combo}]"
        assert spec.in_dst["hidden_states"][TP] == Replicate(), \
            f"case: attention_contract[{combo}]"
        assert spec.out_src["output"][TP] == Partial(), \
            f"case: attention_contract[{combo}]"
        assert spec.out_dst["output"][TP] == want_in, \
            f"case: attention_contract[{combo}]"

        # case: lm_head_out_dst_loss_parallel -- out_src always Shard(-1),
        # out_dst follows loss_parallel
        spec = plan.modules["lm_head"]
        want_out_dst = Shard(-1) if lp else Replicate()
        assert spec.out_src["output"][TP] == Shard(-1), \
            f"case: lm_head_out_dst_loss_parallel[{combo}]"
        assert spec.out_dst["output"][TP] == want_out_dst, \
            f"case: lm_head_out_dst_loss_parallel[{combo}]"

    # case: sp_cp_dim -- with SP on, the CP dim of norm in_src is Shard(1)
    spec = _plan(tiny_llama, make_mesh, True, False).modules["model.norm"]
    assert spec.in_src["hidden_states"][CP] == Shard(1), "case: sp_cp_dim"


# ==========================================================================
# Source: test_s1_deferred_bias.py
# S1.9: D-22 rowwise bias deferral -- _deferred_bias_params marker detection /
# fail-fast / WARNING (detection anchors on the final spec declaration + model
# structure, independent of ParamRole).
# ==========================================================================

class _TinyBiasAttention(nn.Module):
    """toy attention with bias on all of q/k/v/o_proj (OPT/GPT-NeoX style)."""

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
    """Linear layer with bias that is not nn.Linear (custom module; D-22 WARNING + skip path)."""

    def __init__(self, h=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(h, h))
        self.bias = nn.Parameter(torch.randn(h))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class _WoBlock(nn.Module):
    """Container of a rowwise Linear with a non-standard name (wo matches no role rule)."""

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
    """Self-declared rowwise contract (insert/derive=False form)."""
    params = {weight_path: {TP: Shard(1)}}
    params.update(extra_params or {})
    return ModuleShardingSpec(
        params=params,
        in_src={"x": {TP: Replicate()}},
        in_dst={"x": {TP: Replicate()}},
        out_src=out_src or {TP: Partial()},
        out_dst={TP: Replicate()},
    )


def test_deferred_bias_detection(make_mesh, caplog):
    """D-22 marker detection (template-derived + user-inserted): all branches with fail-fast/WARNING."""

    # case: rowwise_bias_deferred -- o_proj.bias deferred; q/k/v bias COLWISE with weights
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(_TinyBiasAttnModel(), mesh, tp_size=2)
    spec = plan.modules["self_attn"]
    assert spec._deferred_bias_params == ("o_proj.bias",), "case: rowwise_bias_deferred"
    # D-19: colwise bias sharded with the weight along output channels (added locally in-region, no reduction)
    for name in ("q_proj.bias", "k_proj.bias", "v_proj.bias"):
        assert spec.params[name][TP] == Shard(0), "case: rowwise_bias_deferred"
    # rowwise bias stays Replicate (added once as a whole after the reduction)
    assert spec.params["o_proj.bias"][TP] == Replicate(), "case: rowwise_bias_deferred"

    # case: explain_shows_deferred_bias
    plan = ShardingPlanner().plan(_TinyBiasAttnModel(), mesh, tp_size=2)
    assert "deferred bias" in plan.explain(), "case: explain_shows_deferred_bias"
    assert "o_proj.bias" in plan.explain(), "case: explain_shows_deferred_bias"

    # case: no_tp_no_defer -- tp_size=1 has no Partial reduction -> no deferral
    plan = ShardingPlanner().plan(_TinyBiasAttnModel(), mesh, tp_size=1)
    assert plan.modules["self_attn"]._deferred_bias_params == (), "case: no_tp_no_defer"

    # case: out_src_non_partial_no_defer -- merged override makes out_src non-Partial -> no deferral
    overrides = {"self_attn": ModuleShardingSpec(out_src={TP: Replicate()})}
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        _TinyBiasAttnModel(), mesh, tp_size=2)
    assert plan.modules["self_attn"]._deferred_bias_params == (), \
        "case: out_src_non_partial_no_defer"

    # case: user_insert_spec_rowwise_bias -- non-standard wo judged by declaration;
    # bias need not be declared in params (detected as long as it physically exists)
    overrides = {"block": _rowwise_spec("wo.weight")}
    plan = ShardingPlanner(plan_overrides=overrides, derive=False).plan(
        _CustomLinearModel(block=_WoBlock()), mesh, tp_size=2)
    assert plan.modules["block"]._deferred_bias_params == ("wo.bias",), \
        "case: user_insert_spec_rowwise_bias"

    # case: bias_declared_non_replicate_fails -- explicitly declaring non-Replicate -> fail-fast
    overrides = {"block": _rowwise_spec(
        "wo.weight", extra_params={"wo.bias": {TP: Shard(0)}})}
    with pytest.raises(ValueError, match="Replicate"):
        ShardingPlanner(plan_overrides=overrides, derive=False).plan(
            _CustomLinearModel(block=_WoBlock()), mesh, tp_size=2)

    # case: lm_head_bias_template_mismatch -- weight Shard(0) while bias falls back to
    # Replicate -> fail-fast at plan time instead of a runtime shape crash
    class _LmHeadBiasModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg(architectures=["TinyBiasForCausalLM"],
                               model_type="tiny_bias")
            self.lm_head = nn.Linear(8, 8, bias=True)

        def forward(self, x):
            return self.lm_head(x)

    with pytest.raises(ValueError, match="template mismatch"):
        ShardingPlanner().plan(_LmHeadBiasModel(), mesh, tp_size=2)

    # case: non_linear_owner_warns_and_skips -- owner not nn.Linear -> WARNING + skip
    overrides = {"block": _rowwise_spec("weight",
                                        extra_params={"bias": {TP: Replicate()}})}
    with caplog.at_level(logging.WARNING):
        plan = ShardingPlanner(plan_overrides=overrides, derive=False).plan(
            _CustomLinearModel(), mesh, tp_size=2)
    assert plan.modules["block"]._deferred_bias_params == (), \
        "case: non_linear_owner_warns_and_skips"
    assert any("not nn.Linear" in r.message for r in caplog.records), \
        "case: non_linear_owner_warns_and_skips"
    caplog.clear()

    # case: multi_output_partial_fails -- multiple outputs + Partial: cannot attribute a unique output
    spec = _rowwise_spec(
        "weight",
        out_src={"a": {TP: Partial()}, "b": {TP: Partial()}})
    with pytest.raises(ValueError, match="single-output"):
        ShardingPlanner(plan_overrides={"block": spec}, derive=False).plan(
            _CustomLinearModel(), mesh, tp_size=2)


# ==========================================================================
# F2 (accuracy_fix_plan.md section 2): Qwen2-MoE architecture override --
# shared_expert_gate is a per-token scalar-gate Linear(H, 1); "must replicate"
# is not "routing semantics", so it is explicitly REPLICATED.
# ==========================================================================

class _TinyQwen2MoeMlp(nn.Module):
    """Qwen2-MoE structure: gate + experts + shared_expert + shared_expert_gate."""

    def __init__(self, hidden=8, inter=16):
        super().__init__()
        self.gate = nn.Linear(hidden, 4, bias=False)
        self.experts = nn.Module()          # a plain container suffices (role decided by naming rules)
        self.experts.w1 = nn.Parameter(torch.randn(4, inter, hidden))
        self.shared_expert = nn.Linear(hidden, inter, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)


class _TinyQwen2Moe(nn.Module):
    """Single-layer Qwen2-MoE mini model (FQN mimics HF Qwen2Moe: model.layers.N.mlp.*)."""

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


def test_qwen2moe_arch_override():
    """Qwen2-MoE rules live in the qwen2_moe family registration (all spellings resolve) + shared_expert_gate REPLICATED."""

    # case: family_rules_registered_all_spellings
    for key in ("qwen2moe", "qwen2_moe"):
        spec = get_model_adapter(key)
        assert spec is not None and spec.sharding_rules is not None, \
            f"case: family_rules_registered_all_spellings[{key}]"
        assert (["shared_expert_gate"], ParamRole.REPLICATED) in list(
            spec.sharding_rules()), \
            f"case: family_rules_registered_all_spellings[{key}]"

    # case: shared_expert_gate_replicated -- do not anchor a fake routing boundary (!= MOE_GATE),
    # and do not Shard(0) the single-row weight into empty shards (!= SHARED_EXPERT,
    # accuracy_problem.md 10.1)
    model = _TinyQwen2Moe()
    roles = ShardingPlanner()._classify_all_params(model, "qwen2moe")
    p = "model.layers.0.mlp."
    assert roles[p + "shared_expert_gate.weight"] == ParamRole.REPLICATED, \
        "case: shared_expert_gate_replicated"
    assert roles[p + "shared_expert.weight"] == ParamRole.SHARED_EXPERT, \
        "case: shared_expert_gate_replicated"
    assert roles[p + "gate.weight"] == ParamRole.MOE_GATE, \
        "case: shared_expert_gate_replicated"
    assert roles[p + "experts.w1"] == ParamRole.MOE_EXPERT, \
        "case: shared_expert_gate_replicated"

    # case: model_type_spelling -- qwen2_moe hits the override as well
    model = _TinyQwen2Moe(architectures=())
    roles = ShardingPlanner()._classify_all_params(model, "qwen2_moe")
    assert roles["model.layers.0.mlp.shared_expert_gate.weight"] == (
        ParamRole.REPLICATED), "case: model_type_spelling"
