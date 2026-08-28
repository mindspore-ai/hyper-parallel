# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s0_foundation.py: merged core suite file.

Sources: test_s0_error.py, test_s0_fixtures.py, test_s0_param_role.py, test_s0_placement_utils.py, test_s0_spec_fields.py
8 merged feature-group cases (originally 32 atomic cases).
"""

import ast
from pathlib import Path

import pytest
from torch import nn

import hyper_parallel.auto_models.components.distributed as dist_pkg
from hyper_parallel.auto_models.components.distributed.param_role import (
    ParamRole,
    ParameterClassifier,
    SEGMENT_EXACT,
    SEGMENT_SUBSTRING,
    _build_default_rules,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
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
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


# ==========================================================================
# Source: test_s0_error.py
# S0.4: PlacementMismatchError message content and exception type.
# ==========================================================================

def test_placement_mismatch_error():
    """PlacementMismatchError: message contains all fields; subclass of ValueError."""
    # ---- case: message_contains_all_fields ----
    err = PlacementMismatchError(
        "model.layers.0.self_attn", (Shard(0),), (Replicate(),), "out_src"
    )
    msg = str(err)
    assert "model.layers.0.self_attn" in msg, "case: message_contains_all_fields"
    assert "out_src" in msg, "case: message_contains_all_fields"
    assert "Shard" in msg and "Replicate" in msg, "case: message_contains_all_fields"
    assert err.module_name == "model.layers.0.self_attn", "case: message_contains_all_fields"
    assert err.stage == "out_src", "case: message_contains_all_fields"
    assert err.expected == (Shard(0),), "case: message_contains_all_fields"
    assert err.actual == (Replicate(),), "case: message_contains_all_fields"

    # ---- case: is_value_error ----
    with pytest.raises(ValueError):
        raise PlacementMismatchError("m", 1, 2, "chain")


# ==========================================================================
# Source: test_s0_fixtures.py
# S0.5: fixture self-checks (FQN inventory + golden plan internal consistency
# + moe boundaries + architecture detection).
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


def test_fixture_self_checks(tiny_llama, tiny_moe, tiny_hf_llama, make_mesh):
    """Fixture self-checks: FQN inventory, golden plan chain contract, moe boundaries, architecture detection."""
    # ---- case: tiny_llama_fqn_inventory ----
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    assert set(plan.modules) == EXPECTED_TINY_LLAMA_FQNS, "case: tiny_llama_fqn_inventory"

    # ---- case: tiny_llama_golden_plan_self_consistent ----
    # Plan internal consistency: adjacent modules satisfy out_dst == in_src
    # (chain contract).
    ordered = [fqn for fqn, _ in tiny_llama.named_modules() if fqn in plan.modules]
    for a, b in zip(ordered[:-1], ordered[1:]):
        sa, sb = plan.modules[a], plan.modules[b]
        if sa.out_dst is None or not sb.in_src:
            continue
        out_vals = {tuple(resolve_placements(v, plan.mesh_dim_names))
                    for v in sa.out_dst.values()}
        in_vals = {tuple(resolve_placements(v, plan.mesh_dim_names))
                   for v in sb.in_src.values()}
        assert out_vals == in_vals, f"case: tiny_llama_golden_plan_self_consistent {a} → {b}"

    # ---- case: tiny_moe_boundaries ----
    # tiny_moe's mlp falls under the moe_mlp boundary with region_dispatch.
    mesh = make_mesh((1, 1), ("tp", "ep"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    for layer in ("0", "1"):
        spec = plan.modules[f"model.layers.{layer}.mlp"]
        assert spec.region_dispatch is False, "case: tiny_moe_boundaries"
        assert any(p.startswith("experts.") for p in spec.params), "case: tiny_moe_boundaries"
        assert "gate.weight" in spec.params, "case: tiny_moe_boundaries"

    # ---- case: tiny_hf_llama_arch ----
    planner = ShardingPlanner()
    assert planner._get_architecture(tiny_hf_llama) == "llama", "case: tiny_hf_llama_arch"


# ==========================================================================
# Source: test_s0_param_role.py
# S0.1 + S1.1: ParamRole enum completeness + ParameterClassifier default rules.
# ==========================================================================

class _NamedModel(nn.Module):
    """Register parameters in order to control FQNs exactly."""
    def __init__(self, names):
        super().__init__()
        for n in names:
            # Register dotted names into nested modules
            *path, leaf = n.split(".")
            obj = self
            for p in path:
                if not hasattr(obj, p):
                    setattr(obj, p, nn.Module())
                obj = getattr(obj, p)
            obj.register_parameter(leaf, nn.Parameter(__import__("torch").zeros(2)))


def test_param_role_and_classifier():
    """ParamRole enum completeness + ParameterClassifier default rule coverage."""
    clf = ParameterClassifier()

    def role(name):
        model = _NamedModel([name])
        return clf.classify(model)[name]

    # ---- case: fourteen_roles ----
    assert len(ParamRole) == 14, "case: fourteen_roles"

    # ---- case: expected_role_names ----
    expected = {
        "COLWISE", "ROWWISE", "NORM", "EMBED", "LM_HEAD", "MOE_GATE",
        "MOE_EXPERT", "SHARED_EXPERT", "FUSED_QKV", "FUSED_GATE_UP",
        "BIAS", "REPLICATED", "SPECIAL", "SKIP",
    }
    assert {r.name for r in ParamRole} == expected, "case: expected_role_names"

    # ---- case: each_role_hit ----
    # At least one hit case for each of the 13 roles the default rules can
    # produce (SPECIAL/SKIP included; REPLICATED is only assigned via
    # ARCH_OVERRIDES, see test_s1_mla_deepseek.py).
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
        assert role(name) == want, f"case: each_role_hit {name}"

    # ---- case: ln_rule_no_false_positive ----
    # ln_ / norm rules must not false-positive on linear / kernel.
    assert role("model.linear.weight") == ParamRole.SKIP, "case: ln_rule_no_false_positive"
    assert role("model.kernel.weight") == ParamRole.SKIP, "case: ln_rule_no_false_positive"
    assert role("model.layers.0.ln_1.weight") == ParamRole.NORM, "case: ln_rule_no_false_positive"

    # ---- case: shared_experts_not_moe_expert ----
    # shared_experts must hit SHARED_EXPERT before experts.
    assert role("m.mlp.shared_experts.w1") == ParamRole.SHARED_EXPERT, "case: shared_experts_not_moe_expert"
    assert role("m.mlp.experts.w1") == ParamRole.MOE_EXPERT, "case: shared_experts_not_moe_expert"

    # ---- case: gate_proj_not_moe_gate ----
    # Dense gate_proj must not be misclassified as MOE_GATE.
    assert role("m.mlp.gate_proj.weight") == ParamRole.COLWISE, "case: gate_proj_not_moe_gate"

    # ---- case: expert_gate_proj_is_moe_expert ----
    # Per-expert gate_proj (experts.N.gate_proj) belongs to MOE_EXPERT.
    assert role("m.mlp.experts.0.gate_proj.weight") == ParamRole.MOE_EXPERT, "case: expert_gate_proj_is_moe_expert"

    # ---- case: unmatched_returns_skip ----
    model = _NamedModel(["a.b.c"])
    assert clf.classify(model)["a.b.c"] == ParamRole.SKIP, "case: unmatched_returns_skip"

    # ---- case: default_rules_structure ----
    # F1: rules are (patterns, role, mode) triples; mode ∈ segment-aware constants
    rules = _build_default_rules()
    for rule in rules:
        pats, r, mode = rule
        assert isinstance(pats, list) and isinstance(r, ParamRole), "case: default_rules_structure"
        assert mode in (SEGMENT_EXACT, SEGMENT_SUBSTRING), "case: default_rules_structure"

    # ---- case: shared_expert_gate_not_shared_expert ----
    # F1 segment-exact match: the shared_expert_gate segment != the
    # shared_expert segment (the misclassification source in
    # accuracy_problem.md 10.1); classified as COLWISE by default (qwen2moe
    # is explicitly overridden to REPLICATED via ARCH_OVERRIDES).
    assert role("m.mlp.shared_expert_gate.weight") != ParamRole.SHARED_EXPERT, "case: shared_expert_gate_not_shared_expert"
    assert role("m.mlp.shared_expert.weight") == ParamRole.SHARED_EXPERT, "case: shared_expert_gate_not_shared_expert"
    assert role("m.mlp.shared_experts.weight") == ParamRole.SHARED_EXPERT, "case: shared_expert_gate_not_shared_expert"

    # ---- case: dotted_pattern_keeps_path_substring ----
    # F1 compatibility: dotted patterns keep the legacy full-path substring semantics.
    assert role("model.layers.0.self_attn.q_proj.weight") == ParamRole.COLWISE, "case: dotted_pattern_keeps_path_substring"

    # ---- case: segment_substring_within_one_segment ----
    # F1 segment substring: the fragment matches within a single segment (no cross-segment).
    assert role("m.mlp.experts.gate_up_proj.weight") == ParamRole.MOE_EXPERT, "case: segment_substring_within_one_segment"


# ==========================================================================
# Source: test_s0_placement_utils.py
# S0.3: resolve_placements / _multi_dim / _normalize_out_fields.
# ==========================================================================

def test_resolve_placements():
    """resolve_placements: axis reordering, missing-axis fill, key interop."""
    # ---- case: axis_order_follows_mesh_dim_names ----
    named = {TP: Shard(0), CP: Replicate(), EP: Shard(0)}
    # Mesh axis order (ep, cp, tp) → output reordered accordingly
    out = resolve_placements(named, ("ep", "cp", "tp"))
    assert out == [Shard(0), Replicate(), Shard(0)], "case: axis_order_follows_mesh_dim_names"

    # ---- case: missing_axis_fills_replicate ----
    named = {TP: Shard(1)}
    out = resolve_placements(named, ("tp", "cp", "ep"))
    assert out == [Shard(1), Replicate(), Replicate()], "case: missing_axis_fills_replicate"

    # ---- case: extra_keys_dropped ----
    named = {TP: Shard(1), CP: Shard(1), EP: Replicate()}
    out = resolve_placements(named, ("tp",))
    assert out == [Shard(1)], "case: extra_keys_dropped"

    # ---- case: str_enum_key_interop ----
    # Plain string keys interoperate with MeshAxisName keys.
    named = {"tp": Shard(0)}
    assert resolve_placements(named, ("tp",)) == [Shard(0)], "case: str_enum_key_interop"


def test_multi_dim():
    # ---- case: none_dims_filtered ----
    out = _multi_dim(tp=Shard(0), cp=Replicate(), ep=None)
    assert EP not in out and out[TP] == Shard(0) and out[CP] == Replicate(), "case: none_dims_filtered"

    # ---- case: all_none_empty ----
    assert not _multi_dim(), "case: all_none_empty"


def test_normalize_out_fields():
    """_normalize_out_fields: scalar shorthand wrapping, idempotence, None passthrough."""
    # ---- case: scalar_shorthand_wrapped ----
    spec = ModuleShardingSpec(out_src={TP: Partial(), CP: Replicate()})
    _normalize_out_fields(spec)
    assert spec.out_src == {"output": {TP: Partial(), CP: Replicate()}}, "case: scalar_shorthand_wrapped"

    # ---- case: dict_contract_untouched ----
    spec = ModuleShardingSpec(
        out_src={"hidden_states": {TP: Shard(1)}},
        out_dst={"output": {TP: Replicate()}},
    )
    _normalize_out_fields(spec)
    assert spec.out_src == {"hidden_states": {TP: Shard(1)}}, "case: dict_contract_untouched"
    assert spec.out_dst == {"output": {TP: Replicate()}}, "case: dict_contract_untouched"

    # ---- case: none_untouched ----
    spec = ModuleShardingSpec(out_src=None, out_dst=None)
    _normalize_out_fields(spec)
    assert spec.out_src is None and spec.out_dst is None, "case: none_untouched"

    # ---- case: idempotent ----
    spec = ModuleShardingSpec(out_src={TP: Partial()})
    _normalize_out_fields(spec)
    _normalize_out_fields(spec)
    assert spec.out_src == {"output": {TP: Partial()}}, "case: idempotent"


# ==========================================================================
# Source: test_s0_spec_fields.py
# S0.2: ShardingPlan / ModuleShardingSpec fields aligned with 05 §3.1-3.2.
# ==========================================================================

def test_spec_and_plan_fields():
    """ShardingPlan / ModuleShardingSpec default fields and mutable-field independence."""
    # ---- case: spec_defaults ----
    spec = ModuleShardingSpec()
    # 2026-08-05 "absent means inherit, present means obey": contract fields
    # default to None (undeclared), semantically distinct from an explicit
    # empty {} (no sharding / no contract); plan output is materialized into
    # concrete dicts by _normalize_contract_fields
    assert spec.params is None, "case: spec_defaults"
    assert spec.in_src is None, "case: spec_defaults"
    assert spec.in_dst is None, "case: spec_defaults"
    assert spec.out_src is None, "case: spec_defaults"
    assert spec.out_dst is None, "case: spec_defaults"
    assert spec.out_names is None, "case: spec_defaults"
    assert spec.is_boundary is True, "case: spec_defaults"
    # Internal flags exist and default to False
    assert spec._is_terminal is False, "case: spec_defaults"
    assert spec.region_dispatch is None, "case: spec_defaults"
    assert spec._needs_cp_attn is False, "case: spec_defaults"

    # ---- case: plan_defaults ----
    plan = ShardingPlan()
    assert plan.modules == {}, "case: plan_defaults"
    assert plan.sequence_parallel is True, "case: plan_defaults"
    assert plan.loss_parallel is False, "case: plan_defaults"
    assert plan.special_handlers == {}, "case: plan_defaults"
    assert plan.mesh_dim_names == (), "case: plan_defaults"
    assert plan.tied_pairs == [], "case: plan_defaults"

    # ---- case: spec_mutable_fields_independent ----
    # Explicitly constructed mutable fields are independent (under None
    # defaults, the concrete dicts are owned by the constructor).
    a, b = ModuleShardingSpec(params={}), ModuleShardingSpec(params={})
    a.params["w"] = {}
    assert b.params == {}, "case: spec_mutable_fields_independent"


# ==========================================================================
# Architecture constraint: components/distributed has zero dependencies on
# recipes/_transformers/models/datasets/trainer
# ==========================================================================


FORBIDDEN = ("recipes", "_transformers", "hyper_parallel.models",
             "datasets", "trainer")


def _imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module


def test_zero_dependency_lint():
    """components/distributed has zero dependencies on recipes/models/datasets/trainer."""
    pkg_dir = Path(dist_pkg.__file__).parent
    checked = 0
    for py in pkg_dir.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        checked += 1
        for mod in _imports_of(py):
            for bad in FORBIDDEN:
                assert bad not in mod, f"{py} has forbidden dependency {mod} (contains {bad})"
    assert checked >= 8, f"only checked {checked} files; lint coverage insufficient"
