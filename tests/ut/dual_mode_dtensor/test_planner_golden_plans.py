# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_planner_golden_plans.py: merged core suite file (feature-combined, compact).

Sources: test_s1_templates.py, test_s1_plan_golden.py, test_s1_mesh_dims.py,
test_s1_boundary_group.py, test_s7_nested.py (nested sub-plan terminal boundary
rules).

The original 36 atomic cases are merged by feature family into 8 combined tests;
each assertion within a merged family keeps its "case: <atomic name>" marker.
The golden-plan e2e tests stay standalone and unmerged.
"""

import pytest
from torch import nn

from hyper_parallel.auto_models.components.distributed import (
    ShardingPlanner,
)
from hyper_parallel.auto_models.components.distributed.param_role import ParamRole
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    CP,
    EP,
    TEMPLATES,
    TP,
    ModuleShardingSpec,
    resolve_placements,
)
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


# ==========================================================================
# Feature family 1: ShardingTemplate + TEMPLATES -- field completeness of the
# 7 templates / I/O contracts (source: test_s1_templates.py, 8 atomic cases merged)
# ==========================================================================

EXPECTED_TEMPLATES = {
    "attention", "mlp", "norm", "embed", "lm_head", "moe_gate", "moe_mlp",
}


def test_templates_contract():
    """7-template enumeration + SP/non-SP field completeness + CP I/O contract + per-template fields."""

    # ---- case: test_seven_templates_enumerated ----
    assert set(TEMPLATES) == EXPECTED_TEMPLATES, \
        "case: seven_templates_enumerated"

    # ---- case: test_field_completeness_sp_and_nosp ----
    # Every template's SP/non-SP I/O fields are fully populated.
    for name, t in TEMPLATES.items():
        for prefix in ("sp", "nosp"):
            assert getattr(t, f"{prefix}_in_src"), \
                f"case: field_completeness {name}.{prefix}_in_src"
            assert getattr(t, f"{prefix}_in_dst"), \
                f"case: field_completeness {name}.{prefix}_in_dst"
            assert getattr(t, f"{prefix}_out_src") is not None, \
                f"case: field_completeness {name}.{prefix}_out_src"
            assert getattr(t, f"{prefix}_out_dst") is not None, \
                f"case: field_completeness {name}.{prefix}_out_dst"

    # ---- case: test_cp_dim_present_in_io_contracts ----
    # I/O contracts declare the CP dim (activations); CP on the parameter side is
    # always Replicate, guaranteed by _multi_dim.
    for name, t in TEMPLATES.items():
        for key, named in t.sp_in_dst.items():
            assert CP in named, \
                f"case: cp_dim_present {name}.sp_in_dst[{key}] missing CP dim"

    # ---- case: test_attention_needs_cp_attn_and_keeps_cp_shard ----
    t = TEMPLATES["attention"]
    assert t.needs_cp_attn is True, "case: attention needs_cp_attn"
    # §6.3.2 asymmetric responsibility: the CP dim of attention sp_in_dst stays Shard(1) (boundary does not gather)
    assert t.sp_in_dst["hidden_states"][CP] == Shard(1), \
        "case: attention sp_in_dst keeps CP shard"
    assert t.sp_out_dst[CP] == Shard(1), \
        "case: attention sp_out_dst keeps CP shard"

    # ---- case: test_moe_mlp_region_dispatch ----
    assert TEMPLATES["moe_mlp"].region_dispatch is False, \
        "case: moe_mlp region_dispatch"
    assert TEMPLATES["moe_mlp"].moe_expert_placement == Shard(0), \
        "case: moe_mlp moe_expert_placement"

    # ---- case: test_moe_gate_out_dst_ep_shard ----
    t = TEMPLATES["moe_gate"]
    assert t.sp_out_dst[EP] == Shard(0), "case: moe_gate sp_out_dst EP"
    assert t.nosp_out_dst[EP] == Shard(0), "case: moe_gate nosp_out_dst EP"

    # ---- case: test_lm_head_out_src_shard_last_dim ----
    # scalar shorthand: {TP: Shard(-1), ...}
    assert TEMPLATES["lm_head"].sp_out_src[TP] == Shard(-1), \
        "case: lm_head sp_out_src shard last dim"

    # ---- case: test_norm_template_all_replicate_params ----
    assert TEMPLATES["norm"].norm_placement == Replicate(), \
        "case: norm all replicate params"


# ==========================================================================
# Feature family 2: planner internal structure checks -- _build_mesh_dim_names
# (S1.8) + Phase 2 _group_by_boundary (S1.3)
# (source: test_s1_mesh_dims.py 4 cases + test_s1_boundary_group.py 7 cases)
# ==========================================================================

class _FakeMesh:
    def __init__(self, names):
        self.mesh_dim_names = names


_MESH_DIM_CASES = [
    # (atomic case name, mesh_dim_names, tp_size, cp_size, ep_size, expected output)
    # filter tp/cp/ep using mesh.mesh_dim_names as the authoritative order
    ("authority_order_from_mesh", ("dp", "ep", "cp", "tp"), 2, 2, 2,
     ("ep", "cp", "tp")),
    # fall back to (tp, cp, ep) when mesh_dim_names is not declared
    ("fallback_order", None, 2, 1, 4, ("tp", "ep")),
    ("size_one_axis_dropped", ("tp", "cp", "ep"), 2, 1, 1, ("tp",)),
    ("dp_axis_never_included", ("dp_shard", "tp"), 2, 1, 1, ("tp",)),
]


def test_planner_internal_structure(tiny_llama):
    """_build_mesh_dim_names 4 rules + _group_by_boundary 7 rules."""

    # ==================== _build_mesh_dim_names ====================
    for case, names, tp, cp, ep, want in _MESH_DIM_CASES:
        out = ShardingPlanner()._build_mesh_dim_names(
            _FakeMesh(names), tp_size=tp, cp_size=cp, ep_size=ep)
        assert out == want, f"case: mesh_dims/{case}"

    # ==================== _group_by_boundary ====================

    # ---- case: test_direct_hit ----
    # Direct boundary hit after stripping the leaf (a norm leaf module is itself a boundary).
    groups = ShardingPlanner()._group_by_boundary(
        {"model.norm.weight": ParamRole.NORM})
    assert set(groups) == {"model.norm"}, "case: boundary/direct_hit"

    # ---- case: test_backtrack_multi_level ----
    # Multi-level backtrack hit: q_proj -> self_attn (leaf guard -> parent attention).
    roles = {
        "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
        "model.layers.0.self_attn.k_proj.weight": ParamRole.COLWISE,
        "model.layers.0.self_attn.v_proj.weight": ParamRole.COLWISE,
        "model.layers.0.self_attn.o_proj.weight": ParamRole.ROWWISE,
    }
    groups = ShardingPlanner()._group_by_boundary(roles)
    assert set(groups) == {"model.layers.0.self_attn"}, \
        "case: boundary/backtrack_multi_level groups"
    assert len(groups["model.layers.0.self_attn"]) == 4, \
        "case: boundary/backtrack_multi_level size"

    # ---- case: test_backtrack_to_root_unknown ----
    # Still unknown after backtracking to the root -> folded into the owning module.
    groups = ShardingPlanner()._group_by_boundary(
        {"zzz.qqq.weight": ParamRole.SKIP})
    assert set(groups) == {"zzz.qqq"}, "case: boundary/backtrack_to_root_unknown"

    # ---- case: test_skip_params_fold_into_boundary ----
    # SKIP params fold upward into the enclosing boundary, no separate group.
    roles = {
        "model.layers.0.mlp.gate_proj.weight": ParamRole.COLWISE,
        "model.layers.0.mlp.up_proj.weight": ParamRole.COLWISE,
        "model.layers.0.mlp.down_proj.weight": ParamRole.ROWWISE,
        "model.layers.0.mlp.some_scale": ParamRole.SKIP,
    }
    groups = ShardingPlanner()._group_by_boundary(roles)
    assert set(groups) == {"model.layers.0.mlp"}, \
        "case: boundary/skip_params_fold_into_boundary groups"
    assert len(groups["model.layers.0.mlp"]) == 4, \
        "case: boundary/skip_params_fold_into_boundary size"

    # ---- case: test_moe_params_fold_into_mlp ----
    # gate + experts share the same moe mlp boundary; per F3, shared_experts
    # becomes an independent nested mlp boundary (its boundary exit holds the
    # RowWise Partial reduction, accuracy_fix_plan.md section 2).
    roles = {
        "model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE,
        "model.layers.0.mlp.experts.w1": ParamRole.MOE_EXPERT,
        "model.layers.0.mlp.experts.w2": ParamRole.MOE_EXPERT,
        "model.layers.0.mlp.shared_experts.w1": ParamRole.SHARED_EXPERT,
    }
    groups = ShardingPlanner()._group_by_boundary(roles)
    assert set(groups) == {"model.layers.0.mlp",
                           "model.layers.0.mlp.shared_experts"}, \
        "case: boundary/moe_params_fold_into_mlp groups"
    assert len(groups["model.layers.0.mlp"]) == 3, \
        "case: boundary/moe_params_fold_into_mlp mlp size"
    assert groups["model.layers.0.mlp.shared_experts"] == [
        ("model.layers.0.mlp.shared_experts.w1", ParamRole.SHARED_EXPERT)], \
        "case: boundary/moe_params_fold_into_mlp shared_experts"

    # ---- case: test_moe_gate_only_group_does_not_anchor ----
    # F3 structural lint: a MOE_GATE-only group does not anchor an MoE boundary
    # (e.g. a scalar-gate Linear misclassified as routing); merged upward.
    groups = ShardingPlanner()._group_by_boundary(
        {"model.layers.0.mlp.gate.weight": ParamRole.MOE_GATE})
    assert "model.layers.0.mlp" not in groups, \
        "case: boundary/moe_gate_only_group_does_not_anchor"

    # ---- case: test_tiny_llama_boundaries ----
    # tiny_llama full boundary set == expected.
    p = ShardingPlanner()
    roles = p._classify_all_params(tiny_llama, "tiny_llama")
    groups = p._group_by_boundary(roles)
    expected = {
        "model.embed_tokens", "model.norm", "lm_head",
        "model.layers.0.input_layernorm", "model.layers.0.self_attn",
        "model.layers.0.post_attention_layernorm", "model.layers.0.mlp",
        "model.layers.1.input_layernorm", "model.layers.1.self_attn",
        "model.layers.1.post_attention_layernorm", "model.layers.1.mlp",
    }
    assert set(groups) == expected, "case: boundary/tiny_llama_boundaries"


# ==========================================================================
# Feature family 3: plan main-entry checks -- global flags + F4 plan-time lint
# (accuracy_fix_plan.md section 2: shard divisibility check + trainable param
# coverage, 5 atomic cases)
# ==========================================================================

class _OddMlpModel(nn.Module):
    """Toy model whose colwise param rows are not divisible by tp_size (mlp boundary)."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 3, bias=False)  # (3, 8): Shard(0) not divisible by tp=2
        self.mlp.up_proj = nn.Linear(8, 4, bias=False)
        self.mlp.down_proj = nn.Linear(4, 8, bias=False)


def test_plan_level_checks(tiny_llama, make_mesh, caplog):
    """plan global flags + F4a/F4b plan-time lint and its escape hatch."""

    # ---- case: test_plan_global_flags ----
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2,
                                  sequence_parallel=False, loss_parallel=True)
    assert plan.sequence_parallel is False, "case: global_flags sequence_parallel"
    assert plan.loss_parallel is True, "case: global_flags loss_parallel"
    lm = plan.modules["lm_head"]
    assert lm.out_dst["output"][TP] == Shard(-1), \
        "case: global_flags loss_parallel=True lm_head out_dst"

    # ---- case: test_shard_divisibility_fails_at_plan_time ----
    # F4a: (3, 8) Shard(0) over tp=2 -> instructive error at plan time (no more
    # waiting for an empty shard at apply time; same class as accuracy_problem.md 10.1).
    mesh = make_mesh((1,), ("tp",))
    with pytest.raises(ValueError, match="not divisible by tp size 2") as exc:
        ShardingPlanner().plan(_OddMlpModel(), mesh, tp_size=2)
    assert "classification" in str(exc.value), \
        "case: f4a divisibility error points at classification"  # points at the most likely misclassification root cause

    # ---- case: test_uncovered_trainable_param_fails ----
    # F4b: a trainable param not covered by any spec.params/special_handlers ->
    # hard error at plan time (gradient-sync semantics must not be silently
    # decided by a consumer-side default).
    tiny_llama.model.layers[0].extra = nn.Linear(4, 4)   # no boundary declared
    mesh = make_mesh((1,), ("tp",))
    with pytest.raises(ValueError, match="coverage check failed") as exc:
        ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    msg = str(exc.value)
    assert "model.layers.0.extra.weight" in msg, "case: f4b names uncovered param"
    assert "allow_uncovered_params" in msg, "case: f4b escape hatch visible"

    # ---- case: test_allow_uncovered_params_downgrades_to_warning ----
    # F4b escape hatch: allow_uncovered_params=True -> downgraded to WARNING
    # (exploratory debugging only).
    import logging
    tiny_llama.model.layers[0].extra = nn.Linear(4, 4)
    mesh = make_mesh((1,), ("tp",))
    with caplog.at_level(logging.WARNING):
        plan = ShardingPlanner(allow_uncovered_params=True).plan(
            tiny_llama, mesh, tp_size=2)
    assert "coverage check failed" in caplog.text, "case: f4b escape warns"
    assert "model.layers.0.extra.weight" in caplog.text, \
        "case: f4b escape names param"
    assert plan.modules, "case: f4b escape plan still produced"  # plan still produced

    # ---- case: test_frozen_param_needs_no_coverage ----
    # requires_grad=False params are exempt from F4b (frozen is explicit semantics).
    extra = nn.Linear(4, 4)
    extra.weight.requires_grad_(False)
    extra.bias.requires_grad_(False)
    tiny_llama.model.layers[0].extra = extra
    mesh = make_mesh((1,), ("tp",))
    ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)   # no error


# ==========================================================================
# Feature family 4 (standalone): ShardingPlanner.plan() main-entry golden diff --
# the golden-plan e2e tests stay standalone and unmerged (source: test_s1_plan_golden.py)
# ==========================================================================

def _assert_placement(named, mesh_dim_names, *want):
    got = tuple(resolve_placements(named, mesh_dim_names))
    assert got == want


def test_tiny_llama_golden_sp_on(tiny_llama, make_mesh):
    """Golden plan for tiny_llama with sequence_parallel=True (full I/O contract)."""
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
    """Golden plan for tiny_llama with sequence_parallel=False (all-Replicate I/O)."""
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
    """Inferred result with real HF FQNs (mock config) matches tiny_llama."""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_llama, mesh, tp_size=2)
    assert "model.layers.0.self_attn" in plan.modules
    attn = plan.modules["model.layers.0.self_attn"]
    assert attn.params["o_proj.weight"][TP] == Shard(1)


def test_tiny_moe_golden(tiny_moe, make_mesh):
    """Golden plan for tiny_moe: gate replicated, experts EP Shard(0) + TP dim shift, I/O contract."""
    mesh = make_mesh((1, 1), ("tp", "ep"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    assert plan.mesh_dim_names == ("tp", "ep")
    moe = plan.modules["model.layers.0.mlp"]
    assert moe.region_dispatch is False
    # gate fully replicated
    assert moe.params["gate.weight"][TP] == Replicate()
    assert moe.params["gate.weight"][EP] == Replicate()
    # experts: EP Shard(0) + TP colwise/rowwise (D-08: TP dims of a 3D [E,out,in]
    # weight are shifted -- colwise->Shard(1), rowwise->Shard(2))
    assert moe.params["experts.w1"][EP] == Shard(0)
    assert moe.params["experts.w1"][TP] == Shard(1)
    assert moe.params["experts.w2"][EP] == Shard(0)
    assert moe.params["experts.w2"][TP] == Shard(2)
    assert moe.params["experts.w3"][EP] == Shard(0)
    assert moe.params["experts.w3"][TP] == Shard(1)
    # I/O contract
    dims = plan.mesh_dim_names
    _assert_placement(moe.in_src["x_BLD"], dims, Shard(1), Replicate())
    _assert_placement(moe.in_dst["x_BLD"], dims, Replicate(), Replicate())
    _assert_placement(moe.out_src["output"], dims, Partial(), Replicate())
    _assert_placement(moe.out_dst["output"], dims, Shard(1), Replicate())


# ==========================================================================
# Feature family 5: nested sub-plan -- D-14 plan-time rules
# (test_s7_nested_plan.py 4 cases)
# ==========================================================================

def _identity_block_spec():
    """Identity I/O contract for an outer container boundary (params={})."""
    return ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )


def test_nested_subplan(tiny_llama, make_mesh):
    """D-14 nested spec: 4 plan-time rule cases."""

    # ==================== plan-time rules ====================

    # ---- case: test_terminal_is_last_boundary_only ----
    # Post-D-14 _is_terminal semantics: only the last boundary in forward order
    # (lm_head) is terminal; all others are non-terminal (no longer decided by
    # whether out_dst is referenced).
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    assert plan.modules["lm_head"]._is_terminal is True, \
        "case: terminal_is_last lm_head terminal"
    for fqn, spec in plan.modules.items():
        if fqn != "lm_head":
            assert spec._is_terminal is False, \
                f"case: terminal_is_last non-terminal {fqn}"

    # ---- case: test_terminal_with_nested_outer ----
    # A nested outer spec does not affect the terminal verdict: the outer
    # (model.layers.0) sits mid-forward-order -> non-terminal; lm_head stays terminal.
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(
        plan_overrides={"model.layers.0": _identity_block_spec()})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert plan.modules["model.layers.0"]._is_terminal is False, \
        "case: terminal_with_nested_outer outer non-terminal"
    assert plan.modules["lm_head"]._is_terminal is True, \
        "case: terminal_with_nested_outer lm_head terminal"

    # ---- case: test_root_spec_allowed ----
    # A root spec (fqn "", the whole-LM outer contract) may be legally inserted: it
    # nests with all inner boundaries, params={} triggers no sole-ownership conflict;
    # first in forward order -> non-terminal.
    mesh = make_mesh((1,), ("tp",))
    root = ModuleShardingSpec(
        params={},
        in_src={"input_ids": {TP: Shard(1)}},
        in_dst={"input_ids": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"": root})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert "" in plan.modules, "case: root_spec_allowed root present"
    assert plan.modules[""]._is_terminal is False, \
        "case: root_spec_allowed root non-terminal"
    assert plan.modules["lm_head"]._is_terminal is True, \
        "case: root_spec_allowed lm_head terminal"
    # inner derived boundaries all kept
    assert "model.layers.0.self_attn" in plan.modules, \
        "case: root_spec_allowed inner self_attn kept"
    assert "model.embed_tokens" in plan.modules, \
        "case: root_spec_allowed inner embed kept"

    # ---- case: test_outer_declares_intermediate_params ----
    # The outer spec may declare intermediate params not owned by any inner
    # boundary subtree (no sole-ownership conflict).
    # Setup: hang a non-boundary bypass Linear inside layers.0 (the planner will
    # not generate a boundary for it); the outer spec declares its params ->
    # plan succeeds and the params land in the outer spec.
    mesh = make_mesh((1,), ("tp",))
    bypass = nn.Linear(16, 16, bias=False)
    tiny_llama.model.layers[0].bypass = bypass
    block = ModuleShardingSpec(
        params={"bypass.weight": {TP: Shard(0)}},   # intermediate param, owned by the outer spec
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert plan.modules["model.layers.0"].params["bypass.weight"][TP] == Shard(0), \
        "case: outer_declares_intermediate_params bypass in outer spec"
