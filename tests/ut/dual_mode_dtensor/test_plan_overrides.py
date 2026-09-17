# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_plan_overrides.py: core merged suite (feature-combination slim version, 13 cases).

Sources: test_s1_plan_overrides.py, test_s1_injections.py
"""

import copy
import logging
import pytest
import torch
from torch import nn
from hyper_parallel.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.distributed._builder.function_module import (
    FunctionModule,
)
from hyper_parallel.distributed._builder.applier import _preflight_compute_injection
from hyper_parallel.distributed.recipe_spec import (
    CP,
    DP,
    EP,
    ModuleShardingSpec,
    TP,
    resolve_placements,
)
try:
    from hyper_parallel.trainer.config import (
        PlanOverride,
        entries_to_plan_overrides,
    )
    _HAS_TRAINER_CONFIG = True
except ImportError:
    # trainer.config pulls in replacement / checkpoint conversion, which
    # require a newer transformers than some CI gates provide.
    _HAS_TRAINER_CONFIG = False
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)
from tests.ut.dual_mode_dtensor.conftest import (
    _meta_mesh,
    cp_sdpa_hf_injection,
    ep_archetype_injection,
)


# ==========================================================================
# Source: test_s1_plan_overrides.py
# S1.13: ShardingPlanner(plan_overrides=...) —— unified override channel (05 §3.6.7 + unified refactor).
# ==========================================================================

def _attn_override_spec(key="x"):
    """Handwritten spec for a custom multi-input attention: contract keys are the real signature parameter names."""
    return ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0), CP: Replicate()},
            "k_proj.weight": {TP: Shard(0), CP: Replicate()},
            "v_proj.weight": {TP: Shard(0), CP: Replicate()},
            "o_proj.weight": {TP: Shard(1), CP: Replicate()},
        },
        in_src={key: {TP: Shard(1)}},
        in_dst={key: {TP: Replicate()}},
        out_src={TP: Partial()},   # scalar shorthand; normalized to {"output": ...} on merge
        out_dst={TP: Shard(1)},
    )


def test_override_merge_full_declaration(tiny_llama, make_mesh):
    """merge semantics: a full user declaration is equivalent to wholesale
    replacement; _needs_cp_attn is inherited from the derived spec; the scalar
    shorthand is normalized."""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": _attn_override_spec(key="x"),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    spec = plan.modules["model.layers.0.self_attn"]
    # user spec takes effect: contract key is "x"
    assert set(spec.in_src) == {"x"}
    assert set(spec.in_dst) == {"x"}
    assert tuple(resolve_placements(spec.in_src["x"], ("tp",))) == (Shard(1),)
    # structural flags inherited from the derived spec (attention template -> True)
    assert spec._needs_cp_attn is True
    assert spec.region_dispatch is None
    # scalar shorthand normalized
    assert set(spec.out_src) == {"output"}
    assert tuple(resolve_placements(spec.out_src["output"], ("tp",))) == (Partial(),)
    # _is_terminal is uniformly marked by Phase 5 (non-terminal)
    assert spec._is_terminal is False
    # param sharding declarations take the user-declared values (field-granularity replacement)
    assert spec.params["q_proj.weight"][TP] == Shard(0)
    assert spec.params["o_proj.weight"][TP] == Shard(1)

    # uncovered modules keep the template-derived results
    other = plan.modules["model.layers.1.self_attn"]
    assert set(other.in_src) == {"hidden_states"}


@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_merge_and_inheritance_semantics(tiny_llama, tiny_hf_native_moe,
                                         make_mesh, caplog):
    """merge/inheritance semantics family: empty fields inherit, internal flags
    inherit, "unset inherits, set wins", sentinels auto/none, no chain-contract
    checks."""

    # ── case: injection_only_spec_inherits_contracts ─────────────────────
    # An override with injection fields only (params/contracts all empty) ->
    # inherits the derived spec's params and I/O contracts; only the injection
    # fields are written.
    mesh = make_mesh((1,), ("tp",))
    baseline = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    derived = baseline.modules["model.layers.0.mlp"]

    def my_compute(module, x):
        return x
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(local_compute_fn=my_compute, region_dispatch=False),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    case = "injection_only_spec_inherits_contracts"
    assert spec.local_compute_fn is my_compute, f"case: {case}"
    # params/contracts fully inherit the derived values
    assert spec.params == derived.params, f"case: {case}"
    assert spec.in_src == derived.in_src, f"case: {case}"
    assert spec.in_dst == derived.in_dst, f"case: {case}"
    assert spec.out_src == derived.out_src, f"case: {case}"
    assert spec.out_dst == derived.out_dst, f"case: {case}"

    # ── case: partial_contract_override_inherits_rest ────────────────────
    # override only in_dst (all-gather changed to identity); other contract fields inherit.
    case = "partial_contract_override_inherits_rest"
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(
            in_dst={"hidden_states": {TP: Shard(1)}}),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec.in_dst["hidden_states"][TP] == Shard(1), f"case: {case}"   # user value
    assert spec.in_src == derived.in_src, f"case: {case}"                  # inherited
    assert spec.params == derived.params, f"case: {case}"                  # inherited

    # ── case: internal_flags_always_inherit ──────────────────────────────
    # merge never rewrites internal flags: _ep_size/_ep_stack derived by D-10
    # are kept even when the user spec holds default values.
    case = "internal_flags_always_inherit"
    mesh_dp_tp = _meta_mesh((4, 2), ("dp", "tp"))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(
            local_compute_fn=lambda m, x: x, region_dispatch=False),
    })
    plan = planner.plan(tiny_hf_native_moe, mesh_dp_tp, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 2, f"case: {case}"
    assert spec._ep_stack, f"case: {case}"                      # D-09 stacking metadata kept
    assert spec.region_dispatch is False, f"case: {case}"       # user-explicit declaration (a non-None injection field wins)

    # ── case: unset_inherits_derived ─────────────────────────────────────
    # unset None (field omitted) -> inherit the derived value (synonymous with 'auto').
    case = "unset_inherits_derived"
    spec_in = ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                 region_dispatch=False)  # contracts all undeclared
    assert spec_in.params is None, f"case: {case}"
    plan = ShardingPlanner(plan_overrides={
        "*.self_attn": spec_in}).plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.self_attn"]
    assert "q_proj.weight" in spec.params, f"case: {case}"     # derived inheritance
    assert spec.inner_wrapper == "sdpa_hf", f"case: {case}"    # injection field takes effect
    # plan output normalization: undeclared fields land as concrete dicts, never None
    for s in plan.modules.values():
        assert s.params is not None, f"case: {case}"
        assert s.in_src is not None and s.in_dst is not None, f"case: {case}"

    # ── case: explicit_empty_clears_merge ────────────────────────────────
    # explicit {} -> clears the derived value (params={} = no param sharding at this boundary, pure I/O stitching).
    case = "explicit_empty_clears_merge"
    plan = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(params={}),
    }, allow_uncovered_params=True).plan(tiny_llama, mesh, tp_size=2)
    assert plan.modules["model.layers.0.self_attn"].params == {}, f"case: {case}"
    # I/O contract undeclared -> still inherits the derived value
    assert "hidden_states" in plan.modules[
        "model.layers.0.self_attn"].in_dst, f"case: {case}"

    # ── case: sentinel_auto_explicit_inherit ─────────────────────────────
    # params="auto": explicitly declare template derivation — same result as leaving it unset.
    case = "sentinel_auto_explicit_inherit"
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(
            params="auto", in_src="auto", in_dst="auto",
            out_src="auto", out_dst="auto"),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec.params == derived.params, f"case: {case}"
    assert spec.in_src == derived.in_src, f"case: {case}"
    assert spec.out_src == derived.out_src, f"case: {case}"

    # ── case: sentinel_none_clears_params ────────────────────────────────
    # params="none": explicitly clear the derived param sharding (other fields inherit).
    # allow_uncovered_params: this case deliberately clears mlp params (escape
    # hatch of the F4b coverage check; only used in override-mechanism unit tests).
    case = "sentinel_none_clears_params"
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(params="none"),
    }, allow_uncovered_params=True)
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec.params == {}, f"case: {case}"                  # cleared
    assert spec.in_dst == derived.in_dst, f"case: {case}"      # rest inherited

    # ── case: yaml_sentinels_merge ───────────────────────────────────────
    # Sentinels passed through YAML: 'none' clears the derived in_src/in_dst
    # (the D-14 mirror constraint requires both to be cleared together).
    case = "yaml_sentinels_merge"
    overrides = entries_to_plan_overrides([PlanOverride(
        match="*.self_attn", in_src="none", in_dst="none")])
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.self_attn"]
    assert spec.in_src == {}, f"case: {case}"                  # cleared by 'none'
    assert spec.in_dst == {}, f"case: {case}"                  # cleared by 'none'

    # ── case: chain_conflict_no_check ────────────────────────────────────
    # D-14: adjacent-contract consistency is no longer checked statically or at
    # runtime — when the override spec disagrees with the upstream out_dst
    # there is no warning and no error; the declaration is kept verbatim.
    case = "chain_conflict_no_check"
    caplog.clear()
    # mlp declares in_src=Replicate, but the upstream post_attention_layernorm out_dst=Shard(1)
    bad_mlp = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Replicate()}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": bad_mlp})
    with caplog.at_level(logging.WARNING):
        plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert "chain contract mismatch" not in caplog.text, f"case: {case}"
    # the declaration is not rewritten; the plan is generated as usual
    declared = plan.modules["model.layers.0.mlp"].in_src["hidden_states"]
    assert tuple(resolve_placements(declared, ("tp",))) == (Replicate(),), f"case: {case}"


def test_glob_exact_merge_and_warnings(tiny_llama, make_mesh, caplog):
    """glob/exact merge and warning family: glob hit merge, exact per-field
    precedence, glob-miss warning, partial-params override warning,
    FunctionModule uncovered warning."""

    # ── case: glob_key_merges_all_hits ───────────────────────────────────
    # glob key: one override covers all matched boundaries (each inherits its own contract).
    case = "glob_key_merges_all_hits"
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                 region_dispatch=False),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    for i in (0, 1):
        spec = plan.modules[f"model.layers.{i}.self_attn"]
        assert spec.inner_wrapper == "sdpa_hf", f"case: {case}"
        assert spec.params["q_proj.weight"], f"case: {case}"    # contract inherited
    assert plan.modules["lm_head"].inner_wrapper is None, f"case: {case}"

    # ── case: exact_wins_per_field_over_glob ─────────────────────────────
    # exact + glob both hit: merge field by field in entry order — non-empty
    # fields of exact (processed later) win, the other glob fields still apply.
    case = "exact_wins_per_field_over_glob"
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                 region_dispatch=False),
        "model.layers.0.self_attn": ModuleShardingSpec(
            inner_wrapper="sdpa_qkv", inner_target="self", region_dispatch=False),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    spec0 = plan.modules["model.layers.0.self_attn"]
    assert spec0.inner_wrapper == "sdpa_qkv", f"case: {case}"     # exact overrides glob
    assert spec0.inner_target == "self", f"case: {case}"
    # layers.1 only gets the glob
    assert plan.modules["model.layers.1.self_attn"].inner_wrapper == "sdpa_hf", f"case: {case}"

    # ── case: glob_miss_warns ────────────────────────────────────────────
    case = "glob_miss_warns"
    caplog.clear()
    planner = ShardingPlanner(plan_overrides={
        "*.self_atn": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                 region_dispatch=False),
    })
    with caplog.at_level(logging.WARNING):
        planner.plan(tiny_llama, mesh, tp_size=2)
    assert "hit no boundary spec" in caplog.text, f"case: {case}"

    # ── case: partial_params_replace_warns ───────────────────────────────
    # partial override: dropped derived params are listed in a WARNING
    # (visibility safeguard).
    case = "partial_params_replace_warns"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                params={"q_proj.weight": {TP: Shard(0)}}),
        }, allow_uncovered_params=True).plan(tiny_llama, mesh, tp_size=2)
    assert "strips the derived sharding" in caplog.text, f"case: {case}"
    assert "o_proj.weight" in caplog.text, f"case: {case}"

    # ── case: full_params_replace_no_warn ────────────────────────────────
    # full override (the complete set of derived keys) -> no WARNING.
    case = "full_params_replace_no_warn"
    caplog.clear()
    full = {k: {TP: Shard(0)} for k in (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")}
    with caplog.at_level(logging.WARNING):
        ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(params=full),
        }).plan(tiny_llama, mesh, tp_size=2)
    assert "strips the derived sharding" not in caplog.text, f"case: {case}"

    # ── case: function_module_uncovered_warns ────────────────────────────
    # DX guard: FunctionModule with no spec coverage -> plan() warning; with an
    # override covering it -> no warning and the spec is inserted (tutorial §10.8).
    case = "function_module_uncovered_warns"

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out

    tiny_llama.model.layers[0].helper_fn = FunctionModule(_Fn)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    assert "helper_fn" in caplog.text, f"case: {case}"
    assert "no boundary spec" in caplog.text, f"case: {case}"

    caplog.clear()
    spec = ModuleShardingSpec(
        params={}, region_dispatch=False,
        in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}},
        out_src={"output": {TP: Shard(1)}}, out_dst={"output": {TP: Shard(1)}},
    )
    with caplog.at_level(logging.WARNING):
        plan = ShardingPlanner(plan_overrides={
            "model.layers.0.helper_fn": spec}).plan(tiny_llama, mesh, tp_size=2)
    assert "no boundary spec" not in caplog.text, f"case: {case}"
    assert "model.layers.0.helper_fn" in plan.modules, f"case: {case}"


@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_override_error_paths(tiny_llama, make_mesh):
    """error-path family: plan-time fail-fast cases (bad sentinel / sentinel in
    insert mode / no derived boundary hit / misspelled fqn / wrong type / D-14
    double-sharding and missing in_src / insert all-unset / YAML insert
    sentinel)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: bad_sentinel_raises ────────────────────────────────────────
    # unknown string value -> fail-fast listing the legal sentinels.
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(params="auto2"),
    })
    with pytest.raises(ValueError, match="auto"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: sentinel_rejected_in_insert_mode ───────────────────────────
    # sentinels are meaningless in insert mode (no derived boundary hit) -> fail-fast.
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0": ModuleShardingSpec(
            params="auto",
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={"output": {TP: Shard(1)}},
            out_dst={"output": {TP: Shard(1)}},
        ),
    })
    with pytest.raises(ValueError, match="insert"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: injection_only_on_unmatched_fqn_raises ─────────────────────
    # injection-fields-only spec hitting no derived boundary -> fail-fast (merge
    # inheritance only applies to derived boundaries; insertion must fully
    # self-declare).
    # Note the fqn must really exist in named_modules (otherwise the spelling
    # check fires first) — model.layers.0 is a real module but not a derived
    # boundary.
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0": ModuleShardingSpec(inner_target="self", inner_wrapper="sdpa_hf",
                                 region_dispatch=False),
    })
    with pytest.raises(ValueError, match="hit no planner-derived boundary"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: invalid_fqn_raises ─────────────────────────────────────────
    # fqn hits no named_modules entry (typo) -> fail-fast ValueError.
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_atn": _attn_override_spec(),   # typo
    })
    with pytest.raises(ValueError, match="named_modules"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: wrong_type_raises ──────────────────────────────────────────
    # override values must be ModuleShardingSpec.
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": {"params": {}},
    })
    with pytest.raises(TypeError, match="ModuleShardingSpec"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: param_double_declaration_raises ────────────────────────────
    # outer spec.params declares params of the inner boundary subtree ->
    # ValueError (invariant 1: unique param ownership; double-sharding is a
    # silent error in production).
    case = "param_double_declaration_raises"
    block = ModuleShardingSpec(
        # self_attn.q_proj.weight is already declared by the derived attention boundary
        params={"self_attn.q_proj.weight": {TP: Shard(0)}},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
    with pytest.raises(ValueError, match="exactly one boundary") as exc:
        planner.plan(tiny_llama, mesh, tp_size=2)
    assert "self_attn.q_proj.weight" in str(exc.value), f"case: {case}"
    assert "model.layers.0" in str(exc.value), f"case: {case}"

    # ── case: leaf_override_double_declaration_raises ────────────────────
    # overriding a leaf module (q_proj) of a derived boundary — its params
    # conflict with the ancestor boundary -> same unique-ownership error
    # (nesting itself is legal; double-sharding is not).
    leaf = ModuleShardingSpec(
        params={"weight": {TP: Shard(0)}},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn.q_proj": leaf,
    })
    with pytest.raises(ValueError, match="exactly one boundary"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: missing_in_src_raises ──────────────────────────────────────
    # full-declaration enforcement (Scenario 1 backfill has been removed):
    # non-empty in_dst with empty in_src -> plan-time ValueError.
    block = ModuleShardingSpec(
        params={},
        in_src={},                                    # <- empty: no longer backfilled after D-14
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
    with pytest.raises(ValueError, match="in_src"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: insert_all_unset_fails ─────────────────────────────────────
    # insert mode: everything undeclared (all None) -> fail-fast.
    tiny_llama.model.layers[0].extra = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="hit no planner-derived boundary"):
        ShardingPlanner(plan_overrides={
            "model.layers.0.extra": ModuleShardingSpec(),
        }).plan(tiny_llama, mesh, tp_size=2)

    # ── case: yaml_insert_with_sentinel_rejected ─────────────────────────
    # insert mode + sentinel -> fail-fast (no derived values to inherit).
    overrides = entries_to_plan_overrides([PlanOverride(
        match="model.layers.0.extra",
        params={"weight": {"tp": "replicate"}},
        in_src="auto",
    )])
    with pytest.raises(ValueError, match="insert"):
        ShardingPlanner(plan_overrides=overrides).plan(
            tiny_llama, mesh, tp_size=2)


def test_axis_and_dp_validation(tiny_llama, make_mesh):
    """override placement axis-name/value validation family (plan-time
    fail-fast, 2026-08-05) + coordinate-system convention (05 §3): declaring a
    DP placement in an override -> fail-first."""

    # ── case: unknown_axis_raises ────────────────────────────────────────
    # misspelled axis name (tp2) -> fail-fast (otherwise resolve_placements silently ignores it).
    mesh = make_mesh((1,), ("tp",))
    spec = ModuleShardingSpec(params={"q_proj.weight": {"tp2": Shard(0)}})
    with pytest.raises(ValueError, match="unknown axis"):
        ShardingPlanner(plan_overrides={
            "*.self_attn": spec}).plan(tiny_llama, mesh, tp_size=2)

    # ── case: non_placement_value_raises ─────────────────────────────────
    # placement value is not a Placement object (e.g. a string mistakenly passed from Python) -> fail-fast.
    spec = ModuleShardingSpec(params={"q_proj.weight": {TP: "shard(0)"}})
    with pytest.raises(TypeError, match="Placement"):
        ShardingPlanner(plan_overrides={
            "*.self_attn": spec}).plan(tiny_llama, mesh, tp_size=2)

    # ── case: dp_declaration_raises ──────────────────────────────────────
    # plan = single dp slice; an override declaring a DP placement -> fail-first
    # ValueError (educational error) instead of silently dropping or selectively
    # keeping it.
    # Cover each field one by one: params / in_src / out_dst (both the scalar
    # shorthand and the nested form must be intercepted).
    base = {
        "params": {},
        "in_src": {"x": {TP: Shard(1)}},
        "in_dst": {"x": {TP: Shard(1)}},
        "out_src": {"output": {TP: Shard(1)}},
        "out_dst": {"output": {TP: Shard(1)}},
    }
    bad_variants = [
        {**base, "params": {"q_proj.weight": {DP: Shard(0), TP: Replicate()}}},
        {**base, "in_src": {"x": {DP: Shard(0), TP: Shard(1)}}},
        {**base, "out_dst": {DP: Shard(0), TP: Shard(1)}},   # scalar shorthand
    ]
    for kwargs in bad_variants:
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.self_attn": ModuleShardingSpec(**kwargs),
        })
        with pytest.raises(ValueError, match="declaring a DP placement is not allowed"):
            planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: ep_virtual_axis_allowed ────────────────────────────────────
    # 'ep' is a virtual axis (TP-extend-EP expert-sharding coordinate system) -> no error.
    # allow_uncovered_params: the partial params replacement leaves other params
    # uncovered (F4b escape hatch; only used in the axis-validation unit test).
    spec = ModuleShardingSpec(params={"q_proj.weight": {EP: Shard(0)}})
    ShardingPlanner(plan_overrides={
        "*.self_attn": spec}, allow_uncovered_params=True).plan(
        tiny_llama, mesh, tp_size=2)

    # ── case: mesh_axis_accepted ─────────────────────────────────────────
    # legal axis name (inside the mesh) works fine — regression guard.
    case = "mesh_axis_accepted"
    spec = ModuleShardingSpec(params={"q_proj.weight": {TP: Shard(0)}})
    plan = ShardingPlanner(plan_overrides={
        "*.self_attn": spec}, allow_uncovered_params=True).plan(
        tiny_llama, mesh, tp_size=2)
    assert plan.modules["model.layers.0.self_attn"].params[
        "q_proj.weight"][TP] == Shard(0), f"case: {case}"


def test_derive_false(tiny_llama, make_mesh, caplog):
    """derive=False family: template derivation is disabled; the plan contains
    only the specs explicitly declared via plan_overrides (all insert mode) —
    replaces the post-processing plan.modules pruning style (multimodal
    encoder_dp ViT bridging scenario: any derived TP collective inside the
    subtree is mathematically wrong)."""
    mesh = make_mesh((1,), ("tp",))

    def _bridge_spec():
        return ModuleShardingSpec(
            params={},
            region_dispatch=False,
            in_src={"input_ids": {TP: Replicate()}},
            in_dst={"input_ids": {TP: Replicate()}},
            out_src={"output": {TP: Replicate()}},
            out_dst={"output": {TP: Replicate()}},
        )

    # ── case: derive_false_yields_only_overrides ─────────────────────────
    # derive=False: plan.modules == override key set (zero derivation).
    case = "derive_false_yields_only_overrides"
    bridge = _bridge_spec()
    plan = ShardingPlanner(plan_overrides={"": bridge}, derive=False).plan(
        tiny_llama, mesh, tp_size=2)
    assert set(plan.modules) == {""}, f"case: {case}"
    # contrast: with the default derive=True the same model derives inner boundaries
    full = ShardingPlanner(plan_overrides={"": bridge}).plan(
        tiny_llama, mesh, tp_size=2)
    assert "model.layers.0.self_attn" in full.modules, f"case: {case}"
    assert len(full.modules) > 1, f"case: {case}"

    # ── case: derive_false_insert_requires_self_declaration ──────────────
    # with derive=False there is nothing to inherit: an injection-fields-only
    # spec -> insert-mode fail-fast (all undeclared).
    with pytest.raises(ValueError, match="hit no planner-derived boundary"):
        ShardingPlanner(
            plan_overrides={
                "model.layers.0.self_attn": ModuleShardingSpec(
                    inner_target="self", inner_wrapper="sdpa_hf",
                    region_dispatch=False)},
            derive=False).plan(tiny_llama, mesh, tp_size=2)

    # ── case: derive_false_sentinel_rejected_with_reason ─────────────────
    # with derive=False the 'auto'/'none' sentinels have nothing to inherit/clear
    # from -> fail-fast, and the error names derive=False and the correct form
    # (explicit {}).
    with pytest.raises(ValueError, match="derive=False"):
        ShardingPlanner(
            plan_overrides={"": ModuleShardingSpec(params="auto")},
            derive=False).plan(tiny_llama, mesh, tp_size=2)

    # ── case: derive_false_glob_hits_nothing_warns ───────────────────────
    # derive=False: a glob key has no derived boundary to hit -> loud warning
    # (globs never insert).
    case = "derive_false_glob_hits_nothing_warns"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        plan = ShardingPlanner(
            plan_overrides={
                "": _bridge_spec(),
                "*.mlp": ModuleShardingSpec(region_dispatch=False)},
            derive=False).plan(tiny_llama, mesh, tp_size=2)
    assert "hit no boundary spec" in caplog.text, f"case: {case}"
    assert set(plan.modules) == {""}, f"case: {case}"


@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_insert_semantics(tiny_llama, make_mesh):
    """insert semantics family: inserting an unmatched module, D-14
    ancestor/descendant nesting allowed, explicit-empty insert, full-declaration
    YAML insert."""

    # ── case: insert_for_missed_module ───────────────────────────────────
    # insert semantics: a module for which the planner generates no spec and
    # which has no ancestor/descendant relationship with any derived boundary
    # (here the top-level bypass dropout) can be inserted and participates in
    # chain propagation.
    case = "insert_for_missed_module"
    mesh = make_mesh((1,), ("tp",))
    # top-level bypass module: not covered by the planner, and a sibling (not
    # ancestor/descendant) of every derived boundary
    tiny_llama.model.dropout = nn.Dropout(0.0)
    identity = ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.dropout": identity})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    assert "model.dropout" in plan.modules, f"case: {case}"
    # insertion position (named_modules order): model.norm -> model.dropout ->
    # lm_head; all referenced downstream -> non-terminal; lm_head stays terminal
    assert plan.modules["model.norm"]._is_terminal is False, f"case: {case}"
    assert plan.modules["model.dropout"]._is_terminal is False, f"case: {case}"
    assert plan.modules["lm_head"]._is_terminal is True, f"case: {case}"

    # ── case: nested_outer_allowed ───────────────────────────────────────
    # D-14 (05 §13) nested specs: overriding model.layers.0 as the outer
    # boundary (params={} I/O contract only), inner derived boundaries kept ->
    # plan succeeds; the outer spec is inserted and inner specs are untouched.
    case = "nested_outer_allowed"
    block = ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert "model.layers.0" in plan.modules, f"case: {case}"
    # inner derived boundaries remain as usual (not taken over by the outer one)
    assert "model.layers.0.self_attn" in plan.modules, f"case: {case}"
    assert plan.modules["model.layers.0.self_attn"].params["q_proj.weight"], f"case: {case}"

    # ── case: explicit_empty_insert_allowed ──────────────────────────────
    # insert mode: explicit {} is a legal declaration (pure I/O stitching boundary).
    case = "explicit_empty_insert_allowed"
    tiny_llama.model.layers[0].extra = nn.Linear(4, 4)
    spec_in = ModuleShardingSpec(
        params={},
        in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}},
        out_src={"output": {TP: Shard(1)}},
        out_dst={"output": {TP: Shard(1)}})
    plan = ShardingPlanner(plan_overrides={
        "model.layers.0.extra": spec_in}, allow_uncovered_params=True
    ).plan(tiny_llama, mesh, tp_size=2)
    assert plan.modules["model.layers.0.extra"].params == {}, f"case: {case}"

    # ── case: yaml_insert_mode_full_declaration ──────────────────────────
    # insert mode via YAML: a module missed by the templates fully self-declares its contract.
    case = "yaml_insert_mode_full_declaration"
    overrides = entries_to_plan_overrides([PlanOverride(
        match="model.layers.0.extra",
        params={"weight": {"tp": "replicate"}},
        in_src={"x": {"tp": "shard(1)"}},
        in_dst={"x": {"tp": "shard(1)"}},
        out_src={"output": {"tp": "shard(1)"}},
        out_dst={"output": {"tp": "shard(1)"}},
    )])
    plan = ShardingPlanner(plan_overrides=overrides,
                           allow_uncovered_params=True).plan(
        tiny_llama, mesh, tp_size=2)
    inserted = plan.modules["model.layers.0.extra"]
    assert inserted.params["weight"][TP] == Replicate(), f"case: {case}"


def test_injection_gating_fields(tiny_llama, make_mesh):
    """inner_wrap/region_dispatch/local_compute_fn gating family: injection
    fields do not rewrite gating flags (declarations are mutually non-nesting);
    fields survive deepcopy; plan() does not pollute the caller's spec."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: user_set_region_dispatch ───────────────────────────────────
    # region_dispatch is publicly configurable (05 §3.6.7): the user explicitly
    # sets True on a custom module -> kept after merge (not overridden when the
    # template does not infer the flag); modules whose template infers True are
    # force-set even if the user leaves it unset (guards against numeric errors).
    case = "user_set_region_dispatch"
    # the user explicitly sets True on a custom data-related module (mlp used here for demonstration)
    custom = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
        region_dispatch=False,
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": custom})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    # the user-explicit value is kept (the mlp template's region_dispatch=False does not clear it)
    assert plan.modules["model.layers.0.mlp"].region_dispatch is False, f"case: {case}"
    # uncovered mlp keeps the template's False
    assert plan.modules["model.layers.1.mlp"].region_dispatch is None, f"case: {case}"

    # ── case: inner_wrap_fields_not_mutating_flags ───────────────────────
    # inner-wrap custom entry (05 §4.4.2): after declaring
    # inner_target/inner_wrapper, the inner-wrap gate is derived by the applier's
    # resolution chain — **does not rewrite _needs_cp_attn** (declarations are
    # mutually non-nesting); the fields survive deepcopy.
    case = "inner_wrap_fields_not_mutating_flags"
    base = {
        "params": {
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        "in_src": {"hidden_states": {TP: Shard(1)}},
        "in_dst": {"hidden_states": {TP: Shard(1)}},   # identity
        "out_src": {TP: Shard(1)},
        "out_dst": {TP: Shard(1)},
    }
    # declaring inner_target -> _needs_cp_attn keeps its original value (False), not implicitly set
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(inner_target="self", **base),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._needs_cp_attn is False, f"case: {case}"
    assert spec.inner_target == "self", f"case: {case}"

    # inner_wrapper (str registry name / callable) -> likewise not rewritten, survives deepcopy
    def my_wrapper(target, cp_mesh):
        return None
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(inner_target="self", inner_wrapper=my_wrapper,
                                                 **base, region_dispatch=False),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._needs_cp_attn is False, f"case: {case}"
    assert spec.inner_wrapper is my_wrapper, f"case: {case}"

    # ── case: local_compute_fn_gate_derived ──────────────────────────────
    # local-region custom compute (05 §4.4.3): after declaring
    # local_compute_fn, the skeleton gate is derived by the applier's resolution
    # chain — **does not rewrite region_dispatch** (declarations are mutually
    # non-nesting); the callable survives deepcopy.
    case = "local_compute_fn_gate_derived"

    def my_compute(module, hidden_states):
        return module.gate_proj(hidden_states)

    custom = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
        local_compute_fn=my_compute, region_dispatch=False)
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": custom})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    # gate derivation: region_dispatch keeps its declared value (False), not implicitly rewritten
    assert spec.region_dispatch is False, f"case: {case}"
    assert spec.local_compute_fn is my_compute, f"case: {case}"

    # ── case: input_spec_not_mutated ─────────────────────────────────────
    # plan() deepcopies the user spec: normalization/flag-marking/chain backfill
    # never pollute the caller's object; repeatable.
    case = "input_spec_not_mutated"
    user_spec = _attn_override_spec(key="x")
    snapshot = copy.deepcopy(user_spec)
    planner = ShardingPlanner(plan_overrides={"model.layers.0.self_attn": user_spec})

    planner.plan(tiny_llama, mesh, tp_size=2)
    assert user_spec.out_src == snapshot.out_src, f"case: {case}"     # not rewritten by normalization
    assert user_spec._needs_cp_attn is False, f"case: {case}"         # not rewritten by template completion
    assert user_spec._is_terminal is False, f"case: {case}"           # not rewritten by Phase 5
    assert user_spec.in_src == snapshot.in_src, f"case: {case}"       # not rewritten by chain backfill

    # repeated calls yield the same result (no accumulated pollution)
    plan2 = planner.plan(tiny_llama, mesh, tp_size=2)
    spec2 = plan2.modules["model.layers.0.self_attn"]
    assert set(spec2.in_src) == {"x"}, f"case: {case}"
    assert spec2._needs_cp_attn is True, f"case: {case}"


@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_yaml_dsl_merge(tiny_llama, make_mesh, caplog):
    """YAML merge family: string DSL -> PlanOverride desugaring -> planner merge, end to end."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: yaml_contract_fields_merge ─────────────────────────────────
    # contract fields are desugared from the YAML string DSL into Placement objects before merging.
    case = "yaml_contract_fields_merge"
    overrides = entries_to_plan_overrides([PlanOverride(
        match="*.self_attn",
        params={
            "q_proj.weight": {"tp": "shard(0)"},
            "k_proj.weight": {"tp": "shard(0)"},
            "v_proj.weight": {"tp": "shard(0)"},
            "o_proj.weight": {"tp": "shard(1)"},
        },
        in_dst={"hidden_states": {"tp": "replicate"}},
        out_src={"tp": "partial"},      # scalar shorthand
    )])
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.self_attn"]
    assert spec.params["q_proj.weight"][TP] == Shard(0), f"case: {case}"
    assert spec.params["o_proj.weight"][TP] == Shard(1), f"case: {case}"
    assert spec.in_dst["hidden_states"][TP] == Replicate(), f"case: {case}"
    # scalar shorthand normalized to {"output": ...}
    assert spec.out_src["output"][TP] == Partial(), f"case: {case}"

    # ── case: yaml_empty_dict_is_explicit_clear ──────────────────────────
    # YAML-form params: {} -> explicit clear (no longer rejected).
    case = "yaml_empty_dict_is_explicit_clear"
    overrides = entries_to_plan_overrides([
        PlanOverride(match="*.self_attn", params={})])
    plan = ShardingPlanner(plan_overrides=overrides,
                           allow_uncovered_params=True).plan(
        tiny_llama, mesh, tp_size=2)
    assert plan.modules["model.layers.0.self_attn"].params == {}, f"case: {case}"

    # ── case: desugared_entry_equivalent_to_handwritten ──────────────────
    # the desugared result and a handwritten {match: spec} go through the same
    # unified channel (glob merge, contract inheritance).
    case = "desugared_entry_equivalent_to_handwritten"
    plan = ShardingPlanner(
        plan_overrides=entries_to_plan_overrides([
            PlanOverride(match="*.self_attn", inner_target="self",
                 inner_wrapper="sdpa_hf"),
        ])).plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.self_attn"]
    assert spec.inner_wrapper == "sdpa_hf", f"case: {case}"
    # planner-derived param sharding/contracts untouched (merge inheritance)
    assert spec.params["q_proj.weight"], f"case: {case}"
    assert "hidden_states" in spec.in_dst, f"case: {case}"
    # both layers hit; unhit boundaries unaffected
    assert plan.modules["model.layers.1.self_attn"].inner_wrapper == "sdpa_hf", f"case: {case}"
    assert plan.modules["lm_head"].inner_wrapper is None, f"case: {case}"

    # ── case: desugared_composes_with_handwritten_overrides ──────────────
    # desugared dict passed together with a handwritten override: exact merge for
    # contracts + glob merge for injection.
    case = "desugared_composes_with_handwritten_overrides"
    override = ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0), },
            "k_proj.weight": {TP: Shard(0)},
            "v_proj.weight": {TP: Shard(0)},
            "o_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": override,
        **entries_to_plan_overrides([
            PlanOverride(match="*.self_attn", inner_target="self",
                 inner_wrapper="sdpa_hf"),
        ]),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.self_attn"]
    # user contract takes effect (field-granularity merge replacement) + injection takes effect
    assert spec.in_dst["hidden_states"][TP] == Replicate(), f"case: {case}"
    assert spec.inner_wrapper == "sdpa_hf", f"case: {case}"
    # the other layer is a derived spec + glob merge injection
    assert plan.modules["model.layers.1.self_attn"].inner_wrapper \
        == "sdpa_hf", f"case: {case}"

    # ── case: desugared_glob_miss_warns ──────────────────────────────────
    case = "desugared_glob_miss_warns"
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ShardingPlanner(
            plan_overrides=entries_to_plan_overrides([
                PlanOverride(match="*.self_atn", inner_wrapper="sdpa_hf"),
            ])).plan(tiny_llama, mesh, tp_size=2)
    assert "hit no boundary spec" in caplog.text, f"case: {case}"


# ==========================================================================
# Source: test_s1_injections.py
# S1.14: PlanOverride desugaring (entries_to_plan_overrides) + explicit-injection preflight (single process).
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_plan_override_desugar():
    """PlanOverride / entries_to_plan_overrides desugaring family (pure unit tests, no model)."""

    # ── case: to_override_basic ──────────────────────────────────────────
    case = "to_override_basic"
    entry = PlanOverride(match="*.self_attn", inner_wrapper="sdpa_hf")
    match, spec = entry.to_override()
    assert match == "*.self_attn", f"case: {case}"
    assert spec.inner_wrapper == "sdpa_hf", f"case: {case}"
    # unset fields keep spec defaults (treated as "unset" under merge semantics, inheriting derived values)
    assert spec.local_compute_fn is None, f"case: {case}"
    assert spec.inner_target is None, f"case: {case}"
    assert spec.region_dispatch is None, f"case: {case}"
    assert spec.tp_divide_attrs is None, f"case: {case}"

    # ── case: tp_divide_attrs_desugar ────────────────────────────────────
    case = "tp_divide_attrs_desugar"
    _, spec = PlanOverride(
        match="*.self_attn",
        tp_divide_attrs=["hidden_size"],
    ).to_override()
    assert spec.tp_divide_attrs == ["hidden_size"], f"case: {case}"

    # ── case: tp_divide_attrs_invalid ────────────────────────────────────
    case = "tp_divide_attrs_invalid"
    for attrs in ("hidden_size", ["bad.name"], ["x", "x"]):
        with pytest.raises(ValueError, match="tp_divide_attrs"):
            PlanOverride(
                match="*.self_attn", tp_divide_attrs=attrs,
            ).to_override()

    # ── case: inner_out_src_desugar ──────────────────────────────────────
    # inner_out_src desugaring: sentinel / single-output DSL / multi-output DSL / illegal value.
    case = "inner_out_src_desugar"
    _, spec = PlanOverride(
        match="*.self_attn", inner_out_src="first_input").to_override()
    assert spec.inner_out_src == "first_input", f"case: {case}"

    _, spec = PlanOverride(
        match="m", inner_out_src={"cp": "shard(2)"}).to_override()
    assert spec.inner_out_src["cp"] == Shard(2), f"case: {case}"

    _, spec = PlanOverride(match="m", inner_out_src={
        "hidden": {"cp": "shard(2)"},
        "aux": {"tp": "partial"},
    }).to_override()
    assert spec.inner_out_src["hidden"]["cp"] == Shard(2), f"case: {case}"
    assert spec.inner_out_src["aux"]["tp"] == Partial(), f"case: {case}"

    with pytest.raises(ValueError, match="first_input"):
        PlanOverride(match="m", inner_out_src="bogus").to_override()

    # ── case: to_override_missing_match_raises ───────────────────────────
    entry = PlanOverride(match="", inner_wrapper="sdpa_hf")
    with pytest.raises(ValueError, match="match"):
        entry.to_override()

    # ── case: to_plan_overrides_merges_same_match ────────────────────────
    # multiple entries with the same match merge field by field (later non-None fields win).
    case = "to_plan_overrides_merges_same_match"
    overrides = entries_to_plan_overrides([
        PlanOverride(match="*.mlp", region_dispatch=False),
        PlanOverride(match="*.mlp", inner_target="self"),
    ])
    assert set(overrides) == {"*.mlp"}, f"case: {case}"
    spec = overrides["*.mlp"]
    assert spec.region_dispatch is False, f"case: {case}"
    assert spec.inner_target == "self", f"case: {case}"

    # ── case: tp_divide_attrs_later_entry_replaces ───────────────────────
    case = "tp_divide_attrs_later_entry_replaces"
    overrides = entries_to_plan_overrides([
        PlanOverride(
            match="*.self_attn", tp_divide_attrs=["hidden_size"],
        ),
        PlanOverride(match="*.self_attn", tp_divide_attrs=[]),
    ])
    assert overrides["*.self_attn"].tp_divide_attrs == [], f"case: {case}"


def test_tp_local_attr_plan(tiny_llama, make_mesh):
    """TP-local attribute family: planner-finalize automatic vs explicit TP-local attributes."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: d17_auto_attrs_require_no_yaml ─────────────────────────────
    case = "d17_auto_attrs_require_no_yaml"
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    attr_plan = plan.modules[
        "model.layers.0.self_attn"
    ]._tp_local_attr_plan
    assert "num_heads" in attr_plan.auto_divide, f"case: {case}"
    assert attr_plan.user_divide == (), f"case: {case}"

    # ── case: explicit_width_attr ────────────────────────────────────────
    case = "explicit_width_attr"
    for layer in tiny_llama.model.layers:
        layer.self_attn.hidden_size = 16
    plan = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(
            tp_divide_attrs=["hidden_size"],
        ),
    }).plan(tiny_llama, mesh, tp_size=2)
    attr_plan = plan.modules[
        "model.layers.0.self_attn"
    ]._tp_local_attr_plan
    assert attr_plan.user_divide == ("hidden_size",), f"case: {case}"

    # ── case: redundant_auto_attr_fails ──────────────────────────────────
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(
            tp_divide_attrs=["num_heads"],
        ),
    })
    with pytest.raises(ValueError, match="D-17"):
        planner.plan(tiny_llama, mesh, tp_size=2)

    # ── case: missing_user_attr_fails ────────────────────────────────────
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(
            tp_divide_attrs=["missing_width"],
        ),
    })
    with pytest.raises(ValueError, match="plain int"):
        planner.plan(tiny_llama, mesh, tp_size=2)


@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_ep_region_dispatch_and_expert_mesh(tiny_hf_native_moe,
                                            tiny_hf_batched_moe, tiny_moe,
                                            make_mesh):
    """EP family: D-10 region_dispatch clear/keep semantics + unified framework
    derivation of the expert mesh (configuring expert_mesh/ep_mesh fails
    fast)."""

    # ── case: per_expert_layout_cleared ──────────────────────────────────
    # HF-native per-expert layout (D-10): region_dispatch is cleared — the
    # module's own forward is not EP-aware, so local_compute_fn must be injected
    # explicitly.
    case = "per_expert_layout_cleared"
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(
        tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 2, f"case: {case}"
    assert spec.region_dispatch is None, f"case: {case}"

    # ── case: batched_layout_cleared ─────────────────────────────────────
    case = "batched_layout_cleared"
    plan = ShardingPlanner().plan(
        tiny_hf_batched_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 4, f"case: {case}"
    assert spec.region_dispatch is None, f"case: {case}"

    # ── case: custom_naming_kept ─────────────────────────────────────────
    # custom naming (w1/w2/w3, pre-stacked by the module author): EP-aware by
    # construction, region_dispatch kept.
    case = "custom_naming_kept"
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 2, f"case: {case}"
    assert spec.region_dispatch is False, f"case: {case}"

    # ── case: ep1_untouched ──────────────────────────────────────────────
    # ep=1 does not enter the D-10 marking path: region_dispatch keeps the template value.
    case = "ep1_untouched"
    mesh_1d = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh_1d, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0, f"case: {case}"
    assert spec.region_dispatch is False, f"case: {case}"

    # ── case: expert_mesh_config_key_rejected ────────────────────────────
    # the factory signature no longer takes expert_mesh — legacy config
    # (expert_mesh=...) fails fast as an "undeclared config key" and lists the
    # legal parameters (clear migration signal).
    from hyper_parallel.distributed._builder.applier import build_expert_mesh
    from hyper_parallel.distributed.expert_parallel.recipes import (
        routed_only_ep_compute_fn,
    )
    from hyper_parallel.distributed._builder.rule_resolver import (
        _resolve_local_compute_fn,
    )
    from hyper_parallel.trainer.config import Target

    user_mesh = build_expert_mesh(mesh, ep_size=4)
    overrides = {"*.mlp": ModuleShardingSpec(
        local_compute_fn=Target(
            routed_only_ep_compute_fn,
            target_path="hyper_parallel.distributed."
                        "recipes.routed_only_ep_compute_fn",
            expert_mesh=user_mesh), region_dispatch=False)}
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    with pytest.raises(ValueError, match="expert_mesh"):
        _resolve_local_compute_fn(
            mlp, spec, mesh, plan.mesh_dim_names, expert_mesh=None)

    # ── case: ep_mesh_context_key_reserved ───────────────────────────────
    # ep_mesh is a framework-reserved context key — configuring it fails fast
    # (the framework derives it uniformly, guaranteeing the a2a communication
    # domain and the expert param sharding domain are the same object).
    overrides = {"*.mlp": ModuleShardingSpec(
        local_compute_fn=Target(
            routed_only_ep_compute_fn,
            target_path="hyper_parallel.distributed."
                        "recipes.routed_only_ep_compute_fn",
            ep_mesh="user-mesh"), region_dispatch=False)}
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    with pytest.raises(ValueError, match="framework-reserved context keys"):
        _resolve_local_compute_fn(
            mlp, spec, mesh, plan.mesh_dim_names, expert_mesh=None)


@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_preflight_fail_fast(tiny_llama, tiny_hf_native_moe, tiny_moe,
                             make_mesh):
    """preflight fail-fast family: cp/ep explicit-injection pre-checks."""

    # ── case: cp_without_inner_wrapper_raises ────────────────────────────
    # cp>1 + attention boundary without inner_wrapper -> fail-fast before apply.
    mesh_cp = _meta_mesh((2,), ("cp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh_cp, cp_size=2)
    with pytest.raises(ValueError, match="inner_wrapper"):
        apply_sharding_plan(tiny_llama, plan, mesh_cp)

    # ── case: cp_with_injection_passes_preflight ─────────────────────────
    # explicit injection passes preflight (later apply stages need a real mesh;
    # this unit test only exercises preflight itself).
    planner = ShardingPlanner(plan_overrides=cp_sdpa_hf_injection())
    plan = planner.plan(tiny_llama, mesh_cp, cp_size=2)
    _preflight_compute_injection(plan, mesh_cp)   # passing means no exception

    # ── case: builtin_cp_wrapper_rejects_dispatch_true ───────────────────
    # Shipped CP wrappers contain collectives and require black-box validation.
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(
            inner_target="self",
            inner_wrapper="sdpa_hf",
            region_dispatch=True,
        )
    })
    plan = planner.plan(tiny_llama, mesh_cp, cp_size=2)
    with pytest.raises(ValueError, match="requires region_dispatch=False"):
        _preflight_compute_injection(plan, mesh_cp)

    # ── case: child_wrapper_requires_inner_output_contract ───────────────
    # A child target cannot inherit the enclosing boundary output layout.
    planner = ShardingPlanner(plan_overrides={
        "*.self_attn": ModuleShardingSpec(
            inner_target="attention_core",
            inner_wrapper="sdpa_hf",
            region_dispatch=False,
        )
    })
    plan = planner.plan(tiny_llama, mesh_cp, cp_size=2)
    with pytest.raises(ValueError, match="inner_out_src.*first_input"):
        _preflight_compute_injection(plan, mesh_cp)

    # ── case: cp_size1_no_check ──────────────────────────────────────────
    # cp axis size=1 (plan already filtered the cp dim) -> no preflight requirement.
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=1)
    _preflight_compute_injection(plan, mesh)

    # ── case: ep_without_compute_fn_raises ───────────────────────────────
    # ep>1 (D-10) + HF-native MoE without local_compute_fn -> fail-fast.
    mesh_ep = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(
        tiny_hf_native_moe, mesh_ep, tp_size=2, ep_size=2)
    with pytest.raises(ValueError, match="local_compute_fn"):
        apply_sharding_plan(tiny_hf_native_moe, plan, mesh_ep)

    # ── case: ep_with_injection_passes_preflight ─────────────────────────
    case = "ep_with_injection_passes_preflight"
    planner = ShardingPlanner(plan_overrides=ep_archetype_injection())
    plan = planner.plan(tiny_hf_native_moe, mesh_ep, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec.local_compute_fn is not None, f"case: {case}"   # Target already overlaid
    _preflight_compute_injection(plan, mesh_ep)

    # ── case: ep_aware_module_region_dispatch_passes ─────────────────────
    # custom EP-aware module (region_dispatch kept) -> passes even without
    # local_compute_fn (the module's forward carries the a2a itself).
    plan = ShardingPlanner().plan(tiny_moe, mesh_ep, tp_size=2, ep_size=2)
    _preflight_compute_injection(plan, mesh_ep)

    # ── case: ep_error_message_teaches_yaml ──────────────────────────────
    # educational error message: includes a paste-ready YAML snippet and the default implementation path.
    case = "ep_error_message_teaches_yaml"
    plan = ShardingPlanner().plan(
        tiny_hf_native_moe, mesh_ep, tp_size=2, ep_size=2)
    with pytest.raises(ValueError) as exc:
        apply_sharding_plan(tiny_hf_native_moe, plan, mesh_ep)
    msg = str(exc.value)
    assert "recipes.qwen2moe_ep_compute_fn" in msg, f"case: {case}"
    assert "region_dispatch" in msg, f"case: {case}"
