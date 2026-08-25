# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s4_ep.py: merged core suite file.

Sources: test_s4_local_compute_fn.py, test_s4_moe_gate_compile.py, test_s5_hf_native_moe.py, test_s6_ep_extend.py

Feature grouping: the original 51 atomic cases are merged into 10 cases by
feature family. Within each family all atomic checks run sequentially, with
section comments ``# ── case: <atomic case name> ──`` marking their origin and
assertions carrying a case-identification message; atomic assertions
(including pytest.raises match checks) are fully covered with no loss.
"""

# Injection factories intentionally keep the complete runtime callback signature,
# and these tests directly validate private planner metadata.
# pylint: disable=unused-argument,protected-access

import functools
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from hyper_parallel.auto_models.components.distributed import ep_compute
from hyper_parallel.auto_models.components.distributed.ep_compute import routed_only_ep_compute_fn
from hyper_parallel.auto_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    _local_swiglu_expert_forward,
    _sigmoid_group_router,
    _softmax_topk_router,
    resolve_swiglu_weights,
    _topk_router_module,
)
from hyper_parallel.auto_models.components.distributed.injection import (
    local_compute,
)
from hyper_parallel.auto_models.components.distributed.precompiled_boundary import PrecompiledBoundary
from hyper_parallel.auto_models.components.distributed.sharding.apply import (
    _StackedExperts,
    _stack_moe_experts,
)
from hyper_parallel.auto_models.components.distributed.sharding_applier import (
    _apply_phase_c,
    _expert_mesh_layout,
    _rewrap_local_outputs,
    _resolve_local_compute_fn,
    _wrap_local_region_forward,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    CP,
    EP,
    ModuleShardingSpec,
    ShardingPlan,
    TEMPLATES,
    TP,
    _normalize_out_fields,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner
try:
    from hyper_parallel.auto_models.trainer.config import Target
    _HAS_TRAINER_CONFIG = True
except ImportError:
    # trainer.config pulls in model_transform / checkpoint conversion, which
    # require a newer transformers than some CI gates provide.
    _HAS_TRAINER_CONFIG = False
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Replicate,
    Shard,
)
from tests.ut.dual_mode_dtensor.conftest import _ensure_pg


# ==========================================================================
# Shared helpers (merged from the private helpers of the original test
# classes/modules)
# ==========================================================================

class _TinyMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = _TinyMod()

    def forward(self, x):
        return self.mod(x)


def _identity_spec():
    return _normalize_out_fields(ModuleShardingSpec(
        in_src={"x": {TP: Shard(1)}},
        in_dst={"x": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    ))


def _wrap_region(mod, spec, mesh, validate_mode):
    """Wrap the local region along the apply path: boundary + resolve + wrap."""
    boundary = PrecompiledBoundary(spec, mesh, ("tp",))
    compute_fn = _resolve_local_compute_fn(
        mod, spec, mesh, ("tp",), expert_mesh=None)
    _wrap_local_region_forward(
        mod, boundary, spec, mesh, ("tp",),
        validate_mode=validate_mode, compute_fn=compute_fn)


class _TinyMoeMod(nn.Module):
    """Minimal MoE-shaped module for archetype factory tests (gate + experts
    [+ shared_expert + shared_expert_gate])."""
    def __init__(self, with_shared=False):
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        if with_shared:
            self.shared_expert = nn.Linear(4, 4)
            self.shared_expert_gate = nn.Linear(4, 1, bias=False)


class _FakeEpMesh:
    def __getitem__(self, name):
        assert name == "ep"
        return self

    def get_group(self, name):
        return f"group-{name}"

    def size(self):
        return 2


def _capture_ep_primitives(monkeypatch):
    """Patch EP primitives and return the metadata captured by the fakes."""
    captured = {}

    def fake_compute(module, hidden_states, *, router_fn, ep_group):
        captured.update(router_fn=router_fn, ep_group=ep_group)
        return hidden_states

    def fake_bind(module, ep_size):
        captured.update(bound_module=module, ep_size=ep_size)

    monkeypatch.setattr(ep_compute, "ep_routed_forward", fake_compute)
    monkeypatch.setattr(ep_compute, "bind_local_expert_forward", fake_bind)
    return captured


class _FakeMesh:
    mesh_dim_names = ("tp", "ep")


def _moe_gate_spec():
    t = TEMPLATES["moe_gate"]
    spec = ModuleShardingSpec(
        in_src=t.sp_in_src,
        in_dst=t.sp_in_dst,
        out_src=t.sp_out_src,
        out_dst=t.sp_out_dst,
    )
    return _normalize_out_fields(spec)


def _meta_mesh(shape, names):
    """Metadata-only mesh (planner tests need no real process groups, but
    DeviceMesh construction requires the default PG to exist -- same
    rationale as make_mesh's _ensure_pg)."""
    _ensure_pg()
    n = 1
    for s in shape:
        n *= s
    return init_device_mesh("cpu", tuple(shape), mesh_dim_names=tuple(names),
                            rank_list=tuple(range(n)), init_backend=False)


# ==========================================================================
# Family 1: declaration resolution / region_dispatch injection discipline /
# template boundary compilation
# Sources: TestRewrapLocalOutputs, TestRegionDispatchDeclaration, moe_gate template
# ==========================================================================

def test_declaration_and_region_dispatch(make_mesh, monkeypatch):
    """Declarative contract family: out_names rewrapping, region_dispatch
    explicit-declaration discipline, moe_gate template EP redistribute
    compilation."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_preserves_list_and_wraps_all_declared_outputs ──
    # Every declared Tensor output is wrapped and list remains list.
    spec = ModuleShardingSpec(
        out_src={
            "hidden": {TP: Replicate()},
            "aux": {TP: Shard(0)},
        },
        out_names=["hidden", "aux", "metadata"],
    )
    calls = []

    def fake_from_local(tensor, device_mesh, placements):
        calls.append((tensor, device_mesh, placements))
        return f"wrapped-{len(calls)}"

    monkeypatch.setattr(
        "hyper_parallel.auto_models.components.distributed.sharding_applier.DTensor.from_local",
        fake_from_local,
    )
    outputs = [torch.ones(2), torch.zeros(2), None]
    result = _rewrap_local_outputs(outputs, spec, mesh, ("tp",), "TestModule")

    assert isinstance(result, list), "case: preserves_list_and_wraps_all_declared_outputs"
    assert result == ["wrapped-1", "wrapped-2", None], \
        "case: preserves_list_and_wraps_all_declared_outputs"
    assert len(calls) == 2, "case: preserves_list_and_wraps_all_declared_outputs"

    # ── case: test_declared_output_index_out_of_range_fails ──
    # A stale out_names contract fails with boundary context.
    spec = ModuleShardingSpec(
        out_src={"aux": {TP: Replicate()}},
        out_names=["hidden", "aux"],
    )
    with pytest.raises(ValueError, match="TestModule.*only 1 output"):
        _rewrap_local_outputs(
            (torch.ones(2),), spec, mesh, ("tp",), "TestModule")

    # ── case: test_local_compute_fn_without_region_dispatch_fails ──
    # region_dispatch injection discipline (no default): declaring an
    # injection must be explicit.
    @local_compute
    def my_fn(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x
        return compute_fn
    spec = _identity_spec()
    spec.local_compute_fn = my_fn
    with pytest.raises(ValueError, match="region_dispatch"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_redundant_true_without_injection_fails ──
    # region_dispatch=True without an injection -> fail-fast (a plain
    # boundary passes through naturally, so the declaration is redundant).
    model = _TinyModel()
    spec = _identity_spec()
    spec.region_dispatch = True
    plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
    with pytest.raises(ValueError, match="redundant"):
        _apply_phase_c(model, plan, mesh, validate_mode=False)

    # ── case: test_moe_gate_out_plan_has_ep_redistribute ──
    # moe_gate template EP redistribute (out_dst {EP: Shard(0)}) compilation.
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.out_plan) == 1, "case: moe_gate_out_plan_has_ep_redistribute"
    op = b.out_plan[0]
    ep_idx = ("tp", "ep").index("ep")
    # out_dst EP dim: Replicate -> Shard(0)
    assert op.src_placements[ep_idx] == Replicate(), \
        "case: moe_gate_out_plan_has_ep_redistribute"
    assert op.dst_placements[ep_idx] == Shard(0), \
        "case: moe_gate_out_plan_has_ep_redistribute"
    assert op.collective_type == "redistribute", \
        "case: moe_gate_out_plan_has_ep_redistribute"

    # ── case: test_moe_gate_in_plan_tp_allgather ──
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.in_plan) == 1, "case: moe_gate_in_plan_tp_allgather"
    assert b.in_plan[0].collective_type == "all_gather", \
        "case: moe_gate_in_plan_tp_allgather"


# ==========================================================================
# Family 2: local_compute_fn resolution chain + Target factory form
# Sources: TestResolveLocalComputeFn (resolution), TestTargetLocalComputeFn
# (happy path)
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_local_compute_fn_resolution(make_mesh):
    """Contract checks for the local-compute factory resolution chain and the
    Target factory form (the YAML sharding.injections carrier)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_user_fn_wins_even_with_ep_size ──
    # local_compute_fn is ring 1 of the resolution chain: the user fn is
    # returned directly even when the _ep_size metadata is present (the
    # built-in EP auto-injection path has been removed; _ep_size only drives
    # parameter sharding).
    built = []

    @local_compute
    def my_fn(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x
        built.append(compute_fn)
        return compute_fn

    spec = _identity_spec()
    spec.local_compute_fn = my_fn
    spec.region_dispatch = False
    spec._ep_size = 2
    fn = _resolve_local_compute_fn(
        _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
    assert isinstance(fn, functools.partial), \
        "case: user_fn_wins_even_with_ep_size"
    # Factory form: the factory is invoked once at apply time, and the
    # partial binds the compute_fn returned by the factory.
    assert len(built) == 1 and fn.func is built[0], \
        "case: user_fn_wins_even_with_ep_size"

    # ── case: test_region_dispatch_resolves_to_module_forward ──
    # region_dispatch pure gating (no user fn) -> the module's own forward.
    mod = _TinyMod()
    spec = _identity_spec()
    spec.region_dispatch = False
    fn = _resolve_local_compute_fn(
        mod, spec, mesh, ("tp",), expert_mesh=None)
    assert fn == mod.forward, \
        "case: region_dispatch_resolves_to_module_forward"  # pylint: disable=comparison-with-callable

    # ── cases: three forms in which resolution yields None (table-driven) ──
    # test_ep_size_alone_returns_none: after the rework, _ep_size>0 no longer
    # injects any compute -- with no local_compute_fn and
    # region_dispatch=False -> None (the apply-side preflight fail-fasts on
    # this).
    # test_inner_wrapper_does_not_resolve_module_forward: when an
    # inner_wrapper hosts the local computation, the whole module forward is
    # not chosen as the skeleton.
    # test_no_declaration_returns_none: neither source present -> None (the
    # module takes no skeleton, and the gate derives to False).
    none_cases = [
        ("ep_size_alone_returns_none",
         lambda s: setattr(s, "_ep_size", 2)),
        ("inner_wrapper_does_not_resolve_module_forward",
         lambda s: (setattr(s, "region_dispatch", False),
                    setattr(s, "inner_wrapper", "sdpa_hf"))),
        ("no_declaration_returns_none", lambda s: None),
    ]
    for label, mutate in none_cases:
        spec = _identity_spec()
        mutate(spec)
        fn = _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
        assert fn is None, f"case: {label}"

    # ── case: test_target_factory_built_with_context ──
    # Target factory: built at apply time (the generic context
    # module/mesh/expert_mesh is filtered by signature), and the returned
    # compute fn is partial-bound with module.
    seen = []

    @local_compute
    def my_factory(module, mesh, tp_mesh, cp_mesh, ep_mesh):
        seen.append((module, mesh, tp_mesh, cp_mesh, ep_mesh))

        def compute_fn(mod, x):
            return mod.lin(x) * 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = Target(
        my_factory, target_path="tests.my_factory")
    spec.region_dispatch = False
    fn = _resolve_local_compute_fn(
        mod, spec, mesh, ("tp",), expert_mesh=None)
    assert isinstance(fn, functools.partial), \
        "case: target_factory_built_with_context"
    assert seen and seen[0][0] is mod, \
        "case: target_factory_built_with_context"   # module context injected
    assert seen[0][2] is mesh["tp"], \
        "case: target_factory_built_with_context"   # tp_mesh filled by framework
    assert seen[0][3] is None, \
        "case: target_factory_built_with_context"   # no cp axis -> cp_mesh=None
    assert seen[0][4] is None, \
        "case: target_factory_built_with_context"   # no EP -> ep_mesh=None
    x = torch.randn(2, 4)
    torch.testing.assert_close(fn(x), mod.lin(x) * 2,
                               msg="case: target_factory_built_with_context")

    # ── case: test_config_keys_pass_through_untouched ──
    # Config keys are purely user-owned: the framework only fills the context
    # and performs no auto-population -- an unconfigured key reaches the
    # factory with its default (None); a configured key passes through
    # untouched.
    seen = []

    @local_compute
    def cfg_factory(mesh, tp_mesh, cp_mesh, ep_mesh, block_size=None):
        seen.append(block_size)

        def compute_fn(mod, x):
            return x
        return compute_fn

    spec = _identity_spec()
    spec.local_compute_fn = Target(
        cfg_factory, target_path="tests.cfg_factory")
    spec.region_dispatch = False
    _resolve_local_compute_fn(
        _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)
    assert seen == [None], \
        "case: config_keys_pass_through_untouched"  # framework does not fill config keys

    spec2 = _identity_spec()
    spec2.local_compute_fn = Target(
        cfg_factory, target_path="tests.cfg_factory", block_size=128)
    spec2.region_dispatch = False
    _resolve_local_compute_fn(
        _TinyMod(), spec2, mesh, ("tp",), expert_mesh=None)
    assert seen[-1] == 128, \
        "case: config_keys_pass_through_untouched"  # explicit user config passes through


# ==========================================================================
# Family 3: custom compute_fn executes inside the local region
# (combination-scenario level, kept standalone)
# Sources: TestResolveLocalComputeFn::test_derived_gate_via_apply_path,
#          TestLocalRegionWithCustomComputeFn
# ==========================================================================

def test_custom_compute_fn_executes_in_region(make_mesh):
    """End-to-end execution of a custom compute_fn: apply derives the gated
    injection, and in both production and validate modes the compute_fn
    receives (module, local tensor)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_derived_gate_via_apply_path ──
    # Derived gating end-to-end: region_dispatch=False + local_compute_fn ->
    # _apply_phase_c still injects the skeleton and executes the custom fn
    # (the gate does not read the stored bool).
    calls = []

    @local_compute
    def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            calls.append(x)
            return module.lin(x) * 3
        return compute_fn

    model = _TinyModel()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute
    spec.region_dispatch = False   # injection discipline: explicit declaration (black-box hosting)
    plan = ShardingPlan(modules={"mod": spec}, mesh_dim_names=("tp",))
    _apply_phase_c(model, plan, mesh, validate_mode=False)

    x = torch.randn(2, 4)
    out = model.mod(x)
    assert len(calls) == 1, \
        "case: derived_gate_via_apply_path"   # custom fn executed -> skeleton injected
    torch.testing.assert_close(out, model.mod.lin(x) * 3,
                               msg="case: derived_gate_via_apply_path")

    # ── case: test_custom_compute_fn_runs_in_region ──
    # production: the custom compute_fn receives (module, local tensor), and
    # its output returns as local via the skeleton boundary exit.
    calls = []

    @local_compute
    def my_compute2(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            calls.append((module, x))
            return module.lin(x) * 2   # custom logic: scale by 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute2
    spec.region_dispatch = False
    _wrap_region(mod, spec, mesh, validate_mode=False)

    x = torch.randn(2, 4)
    out = mod(x)
    assert calls and calls[0][0] is mod, "case: custom_compute_fn_runs_in_region"
    torch.testing.assert_close(out, mod.lin(x) * 2,
                               msg="case: custom_compute_fn_runs_in_region")

    # ── case: test_custom_compute_fn_validate_mode ──
    # validate: DTensor inputs are unwrapped by the skeleton -- the compute_fn
    # still receives a local tensor (no mode awareness), and the exit rewraps
    # then unwraps on return.
    seen = []

    @local_compute
    def my_compute3(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            seen.append(x)
            return module.lin(x)
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute3
    spec.region_dispatch = False
    _wrap_region(mod, spec, mesh, validate_mode=True)

    x = torch.randn(2, 4)
    out = mod(x)
    assert len(seen) == 1, "case: custom_compute_fn_validate_mode"
    assert not isinstance(seen[0], DTensor), \
        "case: custom_compute_fn_validate_mode"   # always local inside compute_fn
    assert not isinstance(out, DTensor), \
        "case: custom_compute_fn_validate_mode"   # skeleton exit always unwraps
    torch.testing.assert_close(out, mod.lin(x),
                               msg="case: custom_compute_fn_validate_mode")


# ==========================================================================
# Family 4: error path A -- local_compute_fn / Target contract fail-fast
# Sources: the 6 raises cases of TestTargetLocalComputeFn
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_local_compute_fn_contract_errors(make_mesh):
    """local_compute_fn injection discipline and contract fail-fast
    (sequential pytest.raises)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_target_bad_return_raises ──
    # Target factory returns a non-callable -> TypeError (contract: must
    # return a compute fn).
    @local_compute
    def bad_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
        return 42

    spec = _identity_spec()
    spec.local_compute_fn = Target(bad_factory, target_path="tests.bad")
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="local_compute_fn"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_target_undecorated_factory_raises ──
    # Injection discipline: Target pointing at an undecorated factory ->
    # fail-fast hinting at @local_compute.
    spec = _identity_spec()
    spec.local_compute_fn = Target(lambda: 42, target_path="tests.bad")
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="@local_compute"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_plain_callable_undecorated_raises ──
    # Injection discipline: an undecorated callable -> fail-fast hinting at
    # @local_compute.
    spec = _identity_spec()
    spec.local_compute_fn = lambda module, x: x
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="@local_compute"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_compute_fn_param_mismatch_raises ──
    # Principle 1: compute fn parameters not matching the original forward ->
    # fail-fast at apply time.
    @local_compute
    def bad_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, hidden):   # original forward's parameter name is x
            return hidden
        return compute_fn

    spec = _identity_spec()
    spec.local_compute_fn = bad_compute
    spec.region_dispatch = False
    with pytest.raises(TypeError, match="same-named"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_target_typo_config_key_raises ──
    # Configuring a key the factory does not declare (typo in rounter) ->
    # fail-fast listing the legal parameters -- config keys bind by name, and
    # a typo must not be silently swallowed.
    spec = _identity_spec()
    spec.local_compute_fn = Target(
        routed_only_ep_compute_fn,
        target_path="hyper_parallel.auto_models.components.distributed."
                    "ep_compute.routed_only_ep_compute_fn",
        blok_size="oops")                     # typo: should be block_size
    spec.region_dispatch = False
    with pytest.raises(ValueError, match="undeclared keys"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)

    # ── case: test_target_reserved_context_key_raises ──
    # Context keys are framework-reserved names: a user configuring mesh in
    # Target -> fail-fast (the mesh family may only be filled by the
    # framework).
    spec = _identity_spec()
    spec.local_compute_fn = Target(
        routed_only_ep_compute_fn,
        target_path="hyper_parallel.auto_models.components.distributed."
                    "ep_compute.routed_only_ep_compute_fn",
        mesh="oops")
    spec.region_dispatch = False
    with pytest.raises(ValueError, match="framework-reserved context keys"):
        _resolve_local_compute_fn(
            _TinyMod(), spec, mesh, ("tp",), expert_mesh=None)


# ==========================================================================
# Family 5: built-in EP archetype factories (qwen2/qwen3/mixtral/custom
# combination sweep)
# Sources: the 6 happy-path cases of TestEpArchetypeFactories
# ==========================================================================

def test_ep_archetype_factories(make_mesh, monkeypatch):
    """Built-in EP archetype factories (ep_compute.py): use of the mesh family
    context + explicit router selection + each archetype's combine formula
    (accuracy_fix_plan.md section 3 E2)."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_mesh_family_used_directly ──
    # ep_mesh is filled by the framework; the factory directly takes
    # ep_mesh.get_group("ep") and hands it to ep_routed_forward; the router
    # is embedded (default softmax top-k); no tp_group -- the TP
    # communication of a nested boundary is self-contained by the
    # sub-boundary (contract in ep_utils).
    module = _TinyMoeMod()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.routed_only_ep_compute_fn(
        module=module, mesh=mesh, tp_mesh=mesh["tp"], cp_mesh=None,
        ep_mesh=_FakeEpMesh())
    compute_fn(module, torch.randn(2, 4))
    assert captured["ep_group"] == "group-ep", "case: mesh_family_used_directly"
    assert captured["bound_module"] is module, "case: mesh_family_used_directly"
    assert captured["ep_size"] == 2, "case: mesh_family_used_directly"
    # Embedded router: default softmax top-k (the framework does not decide
    # the router; spec has no such field)
    assert captured["router_fn"] is _softmax_topk_router, \
        "case: mesh_family_used_directly"

    # ── case: test_qwen2moe_factory_combines_shared_and_gate ──
    # qwen2moe archetype combine formula: routed + sigmoid(gate(x)) * shared(x)
    # -- the shared_expert call is an ordinary submodule call (nested
    # boundary contract), with no compensating communication.
    captured = _capture_ep_primitives(monkeypatch)
    module = _TinyMoeMod(with_shared=True)
    compute_fn = ep_compute.qwen2moe_ep_compute_fn(
        module=module, mesh=None, tp_mesh=None, cp_mesh=None,
        ep_mesh=_FakeEpMesh())
    x = torch.randn(2, 4)
    out = compute_fn(module, x)
    expected = x + torch.sigmoid(module.shared_expert_gate(x)) * module.shared_expert(x)
    torch.testing.assert_close(
        out, expected, msg="case: qwen2moe_factory_combines_shared_and_gate")
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen2moe"], \
        "case: qwen2moe_factory_combines_shared_and_gate"

    # ── case: test_qwen3_factory_embeds_topk_router ──
    # Qwen3-MoE uses its explicit TopKRouter factory.
    module = _TinyMoeMod()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.qwen3moe_ep_compute_fn(
        module=module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_FakeEpMesh(),
    )
    compute_fn(module, torch.randn(2, 4))

    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen3moe"], \
        "case: qwen3_factory_embeds_topk_router"
    assert captured["bound_module"] is module, \
        "case: qwen3_factory_embeds_topk_router"

    # ── case: test_mixtral_factory_uses_tuple_router_and_training_jitter ──
    # Mixtral 5.12 uses its tuple router and jitters the expert input in training.
    module = _TinyMoeMod()
    module.jitter_noise = 0.2
    module.train()
    captured = _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.mixtral_ep_compute_fn(
        module=module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_FakeEpMesh(),
    )
    hidden_states = torch.ones(2, 4)
    torch.manual_seed(17)
    output = compute_fn(module, hidden_states)
    torch.manual_seed(17)
    expected = hidden_states * torch.empty_like(hidden_states).uniform_(0.8, 1.2)

    torch.testing.assert_close(
        output, expected,
        msg="case: mixtral_factory_uses_tuple_router_and_training_jitter")
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["mixtral"], \
        "case: mixtral_factory_uses_tuple_router_and_training_jitter"

    # ── case: test_mixtral_factory_disables_jitter_in_eval ──
    # Mixtral evaluation preserves hidden states even when jitter is configured.
    module = _TinyMoeMod()
    module.jitter_noise = 0.2
    module.eval()
    _capture_ep_primitives(monkeypatch)
    compute_fn = ep_compute.mixtral_ep_compute_fn(
        module=module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_FakeEpMesh(),
    )
    hidden_states = torch.randn(2, 4)
    torch.testing.assert_close(
        compute_fn(module, hidden_states), hidden_states,
        msg="case: mixtral_factory_disables_jitter_in_eval")

    # ── case: test_custom_factory_embeds_its_router ──
    # The router is part of the injected function: a custom factory
    # references a MOE_ROUTER_ADAPTERS adapter by name and writes it into its
    # own compute fn -- the framework takes no part in the selection.
    captured = _capture_ep_primitives(monkeypatch)

    @local_compute
    def qwen3moe_ep_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
        ep_group = ep_mesh.get_group("ep")

        def compute_fn(module, hidden_states):
            return ep_compute.ep_routed_forward(
                module, hidden_states,
                router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
                ep_group=ep_group)
        return compute_fn

    fn = qwen3moe_ep_factory(
        mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=_FakeEpMesh())
    fn(_TinyMoeMod(), torch.randn(2, 4))
    assert captured["router_fn"] is MOE_ROUTER_ADAPTERS["qwen3moe"], \
        "case: custom_factory_embeds_its_router"
    assert captured["ep_group"] == "group-ep", \
        "case: custom_factory_embeds_its_router"


# ==========================================================================
# Family 6: error path B -- factory / stack / planner fail-fast
# Sources: the 2 raises cases of TestEpArchetypeFactories,
#          test_stack_moe_experts_rejects_bias, test_planner_ep_extend_invalid
# ==========================================================================

def test_factory_and_planner_error_paths(monkeypatch, tiny_hf_native_moe):
    """Fail-fast for factory interface assertions / missing ep_mesh / stack
    bias restriction / planner EP-extension parameter validation (sequential
    pytest.raises)."""
    # ── case: test_planner_ep_extend_invalid ──
    # ep_size exceeding the dense region / not dividing it / num_experts not
    # divisible -> ValueError.
    # mesh (1,2) D=2: ep=4 > D -> error
    mesh = _meta_mesh((1, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    # mesh (4,2) D=8: ep=3 does not divide D -> error
    mesh8 = _meta_mesh((4, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=3)
    # mesh (4,2) D=8: ep=8 is legal but num_experts=4 not divisible by ep=8 -> error
    with pytest.raises(ValueError, match="num_experts"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=8)

    # ── case: test_interface_assertion_fails_fast ──
    # Wrong archetype chosen (module lacks shared_expert/shared_expert_gate)
    # -> ValueError at apply time, listing the module's actual submodule
    # names.
    _capture_ep_primitives(monkeypatch)
    module = _TinyMoeMod(with_shared=False)
    with pytest.raises(ValueError, match="shared_expert") as exc_info:
        ep_compute.qwen2moe_ep_compute_fn(
            module=module, mesh=None, tp_mesh=None, cp_mesh=None,
            ep_mesh=_FakeEpMesh())
    msg = str(exc_info.value)
    assert "gate" in msg and "experts" in msg, \
        "case: interface_assertion_fails_fast"   # actual submodule names visible

    # ── case: test_factory_requires_ep_mesh ──
    # Non-EP boundary (framework fills ep_mesh=None) -> config error
    # fail-fast.
    mesh = type("M", (), {"mesh_dim_names": ("tp",)})()
    with pytest.raises(ValueError, match="ep_mesh"):
        ep_compute.routed_only_ep_compute_fn(
            module=_TinyMoeMod(), mesh=mesh, tp_mesh=None, cp_mesh=None,
            ep_mesh=None)

    # ── case: test_stack_moe_experts_rejects_bias ──
    # An expert with bias -> NotImplementedError (v1 limitation).
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    mlp.experts[0].gate_proj.bias = nn.Parameter(torch.zeros(32))
    ep_stack = {"experts.gate_proj": [f"experts.{i}.gate_proj.weight" for i in range(4)]}
    with pytest.raises(NotImplementedError, match="bias"):
        _stack_moe_experts(mlp, ep_stack)


# ==========================================================================
# Family 7: region_dispatch=True dispatch-through validation
# Sources: TestLocalRegionDispatchThrough (combination-scenario level, kept
# as its own family)
# ==========================================================================

def test_region_dispatch_through(make_mesh):
    """region_dispatch=True: validate dispatches through the injected
    function (pure standard ops), strategy propagation covers the injected
    code, and out_src is upgraded from declarative rewrapping to true
    validation; production behavior is unchanged."""
    mesh = make_mesh((1,), ("tp",))

    # ── case: test_dispatch_through_validate ──
    # validate: DTensor enters the injected function directly (no to_local),
    # and the propagation result matching the out_src declaration -> passes;
    # production behavior unchanged.
    seen = {}

    @local_compute
    def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            seen["x_is_dtensor"] = isinstance(x, DTensor)
            return x * 2 + x              # pure pointwise: dispatchable
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute
    spec.region_dispatch = True       # injected code uses pure standard ops
    _wrap_region(mod, spec, mesh, validate_mode=True)

    # The boundary entry of a size-1 mesh does not wrap into DTensor
    # (degenerate skip) -- feed a DTensor directly (equivalent to the input
    # passed from the outer boundary in the D-14 nested scenario).
    x = torch.randn(2, 4)
    dt = DTensor.from_local(x, mesh, (Shard(1),))
    out = mod(dt)
    assert seen["x_is_dtensor"] is True, \
        "case: dispatch_through_validate"     # validate dispatch-through: injected fn sees DTensor
    assert not isinstance(out, DTensor), \
        "case: dispatch_through_validate"     # skeleton exit always unwraps
    torch.testing.assert_close(out, x * 3, msg="case: dispatch_through_validate")

    # ── case: test_dispatch_through_out_src_mismatch_fails ──
    # True validation: the propagation result (pointwise -> Shard(1)
    # preserved) disagrees with the declared out_src (Shard(0)) ->
    # fail-fast -- black-box mode cannot catch this class of injected-code
    # bug.
    @local_compute
    def my_compute2(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x * 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.out_src = {"output": {TP: Shard(0)}}   # a lying declaration
    spec.local_compute_fn = my_compute2
    spec.region_dispatch = True
    _wrap_region(mod, spec, mesh, validate_mode=True)

    dt = DTensor.from_local(torch.randn(2, 4), mesh, (Shard(1),))
    with pytest.raises(Exception, match="out_src"):
        mod(dt)

    # ── case: test_dispatch_through_production_unchanged ──
    # production: region_dispatch=True changes no branching (local
    # pass-through).
    seen = {}

    @local_compute
    def my_compute3(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            seen["x_is_dtensor"] = isinstance(x, DTensor)
            return x * 2
        return compute_fn

    mod = _TinyMod()
    spec = _identity_spec()
    spec.local_compute_fn = my_compute3
    spec.region_dispatch = True
    _wrap_region(mod, spec, mesh, validate_mode=False)

    x = torch.randn(2, 4)
    out = mod(x)
    assert seen["x_is_dtensor"] is False, \
        "case: dispatch_through_production_unchanged"   # production always local
    torch.testing.assert_close(out, x * 2,
                               msg="case: dispatch_through_production_unchanged")


# ==========================================================================
# Family 8: planner EP marking + stack handler
# Sources: test_planner_marks_hf_native_moe, test_planner_no_mark_without_ep,
#          test_planner_pre_stacked_d10_ep_extend, test_stack_moe_experts
# ==========================================================================

def test_planner_ep_marking_and_stack(tiny_hf_native_moe, tiny_moe, make_mesh):
    """Planner EP marking for HF-native / custom-named MoE (stacked metadata
    + TP-extend-EP contract) and the _stack_moe_experts stacking handler."""
    # ── case: test_planner_marks_hf_native_moe ──
    # per-expert params + ep>1 -> stacked metadata + TP-extend-EP contract
    # (D-09a/D-10).
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)

    spec = plan.modules["model.layers.0.mlp"]
    # Numeric-field guard in effect: the boundary aggregates at mlp, with no
    # per-expert boundaries
    assert not any("experts.0" in fqn for fqn in plan.modules), \
        "case: planner_marks_hf_native_moe"

    # stacked entries (D-10 TP-extend-EP: only {EP: S0} expert-dim sharding,
    # no TP key, no second axis)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_marks_hf_native_moe"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_marks_hf_native_moe"

    # per-expert entries removed; router fully replicated
    assert not any("experts.0" in k for k in spec.params), \
        "case: planner_marks_hf_native_moe"
    assert spec.params["gate.weight"][TP] == Replicate(), \
        "case: planner_marks_hf_native_moe"

    # _ep_stack metadata: stacked name -> source paths ordered by expert idx
    assert set(spec._ep_stack) == {
        "experts.gate_proj", "experts.up_proj", "experts.down_proj"}, \
        "case: planner_marks_hf_native_moe"
    assert spec._ep_stack["experts.gate_proj"] == [
        f"experts.{i}.gate_proj.weight" for i in range(4)], \
        "case: planner_marks_hf_native_moe"
    # TP-extend-EP: _ep_size = ep_size, boundary identity
    assert spec._ep_size == 2, "case: planner_marks_hf_native_moe"
    assert spec.in_dst["x_BLD"][TP] == Shard(1), \
        "case: planner_marks_hf_native_moe"

    # ── case: test_planner_no_mark_without_ep ──
    # ep=1 -> no stacking, per-expert entries kept (TP-only semantics
    # correct).
    mesh1 = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh1, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_stack == {}, "case: planner_no_mark_without_ep"
    assert "experts.0.gate_proj.weight" in spec.params, \
        "case: planner_no_mark_without_ep"
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0), \
        "case: planner_no_mark_without_ep"

    # ── case: test_planner_pre_stacked_d10_ep_extend ──
    # Custom naming (experts.w1 3D) -> D-10 TP-extend-EP path:
    # {EP: Shard(0)}, no TP key, SP-in identity boundary, _ep_stack empty
    # (already stacked).
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 2, "case: planner_pre_stacked_d10_ep_extend"
    assert spec._ep_stack == {}, "case: planner_pre_stacked_d10_ep_extend"
    # Custom-named w1/w2/w3 -> expert params only {EP: Shard(0)}, no TP key
    for proj in ("w1", "w2", "w3"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_pre_stacked_d10_ep_extend"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_pre_stacked_d10_ep_extend"
    assert spec.in_dst["x_BLD"][TP] == Shard(1), \
        "case: planner_pre_stacked_d10_ep_extend"   # SP-in identity

    # ── case: test_stack_moe_experts ──
    # Stacking handler: stacked values == original per-expert values,
    # original params removed.
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    orig = {
        proj: torch.stack([getattr(mlp.experts[i], proj).weight.data
                           for i in range(4)])
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
    ep_stack = plan.modules["model.layers.0.mlp"]._ep_stack

    _stack_moe_experts(mlp, ep_stack)

    assert isinstance(mlp.experts, _StackedExperts), "case: stack_moe_experts"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = getattr(mlp.experts, proj)
        assert stacked.shape == orig[proj].shape, "case: stack_moe_experts"
        torch.testing.assert_close(stacked, orig[proj],
                                   msg="case: stack_moe_experts")
    # Original per-expert params removed
    assert not any("experts.0" in n
                   for n, _ in mlp.named_parameters()), "case: stack_moe_experts"


# ==========================================================================
# Family 9: router adapters / SwiGLU weight resolution / expert forward
# utilities
# Sources: test_softmax_topk_router, testresolve_swiglu_weights_*,
#          test_local_expert_forward_uses_declared_activation,
#          test_topk_router_module_adapter, test_sigmoid_group_router_adapter
# ==========================================================================

def test_router_and_expert_utils(tiny_hf_native_moe, tiny_hf_batched_moe):
    """Router adapters (softmax top-k / TopKRouter module / sigmoid group)
    plus SwiGLU weight resolution and local expert forward."""
    # ── case: test_softmax_topk_router ──
    # The default adapter matches the routing semantics of the toy model's
    # forward.
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    topk_idx, topk_w = _softmax_topk_router(mlp, hidden)
    logits = mlp.gate(hidden).view(-1, 4)
    w = logits.softmax(-1)
    ref_w, ref_idx = w.topk(2, dim=-1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    assert torch.equal(topk_idx, ref_idx), "case: softmax_topk_router"
    torch.testing.assert_close(topk_w, ref_w, msg="case: softmax_topk_router")
    assert MOE_ROUTER_ADAPTERS["default"] is _softmax_topk_router, \
        "case: softmax_topk_router"

    # ── case: testresolve_swiglu_weights_two_naming_families ──
    # Both naming families gate/up/down_proj and w1/w2/w3 resolve; a missing
    # matrix raises.
    class Holder(nn.Module):  # pylint: disable=abstract-method
        pass

    h1 = Holder()
    h1.gate_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.up_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.down_proj = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = resolve_swiglu_weights(h1)
    assert g is h1.gate_proj and u is h1.up_proj and d is h1.down_proj, \
        "case: resolve_swiglu_weights_two_naming_families"

    h2 = Holder()
    h2.w1 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w3 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w2 = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = resolve_swiglu_weights(h2)
    assert g is h2.w1 and u is h2.w3 and d is h2.w2, \
        "case: resolve_swiglu_weights_two_naming_families"

    with pytest.raises(NotImplementedError, match="SwiGLU"):
        resolve_swiglu_weights(Holder())

    # ── case: testresolve_swiglu_weights_fused_layout ──
    # D-11 fused layout: gate_up_proj + down_proj -> (fused, None, down).
    h = Holder()
    h.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
    h.down_proj = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = resolve_swiglu_weights(h)
    assert g is h.gate_up_proj and u is None and d is h.down_proj, \
        "case: resolve_swiglu_weights_fused_layout"

    # automodel naming (gate_and_up_projs/down_projs) is isomorphic
    h2 = Holder()
    h2.gate_and_up_projs = nn.Parameter(torch.randn(4, 16, 8))
    h2.down_projs = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = resolve_swiglu_weights(h2)
    assert g is h2.gate_and_up_projs and u is None and d is h2.down_projs, \
        "case: resolve_swiglu_weights_fused_layout"

    # ── case: test_local_expert_forward_uses_declared_activation ──
    # EP expert computation honors the model activation instead of forcing SiLU.
    class Experts(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.local_expert_count = 1
            self.gate_up_proj = nn.Parameter(torch.randn(1, 8, 4))
            self.down_proj = nn.Parameter(torch.randn(1, 4, 4))
            self._ep_act_fn = torch.tanh

    experts = Experts()
    hidden_states = torch.randn(3, 4)
    expert_indices = torch.zeros(3, dtype=torch.long)
    output = _local_swiglu_expert_forward(experts, hidden_states, expert_indices)
    gate_states, up_states = F.linear(hidden_states, experts.gate_up_proj[0]).chunk(2, dim=-1)
    expected = F.linear(torch.tanh(gate_states) * up_states, experts.down_proj[0])
    torch.testing.assert_close(
        output, expected, msg="case: local_expert_forward_uses_declared_activation")

    mlp = tiny_hf_batched_moe.model.layers[0].mlp

    # ── case: test_topk_router_module_adapter ──
    # Qwen2/Qwen3/Mixtral adapter: directly takes the indices and scores of
    # the TopKRouter.
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    idx, w = _topk_router_module(mlp, hidden)
    _, ref_w, ref_idx = mlp.gate(hidden)
    assert torch.equal(idx, ref_idx), "case: topk_router_module_adapter"
    torch.testing.assert_close(w, ref_w, msg="case: topk_router_module_adapter")
    assert MOE_ROUTER_ADAPTERS["qwen2moe"] is _topk_router_module, \
        "case: topk_router_module_adapter"
    assert MOE_ROUTER_ADAPTERS["qwen3moe"] is _topk_router_module, \
        "case: topk_router_module_adapter"
    assert MOE_ROUTER_ADAPTERS["mixtral"] is _topk_router_module, \
        "case: topk_router_module_adapter"

    # ── case: test_sigmoid_group_router_adapter ──
    # deepseekv3/glm4moe adapter: sigmoid + correction bias + norm + scaling
    # (n_group=1 skips the group filter), matching a hand-computed reference.
    class Gate(nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(e, h) * 0.02)
            self.register_buffer("e_score_correction_bias", torch.randn(e) * 0.01)

        def forward(self, x):
            return F.linear(  # pylint: disable=not-callable
                x.view(-1, x.shape[-1]).float(), self.weight.float()
            )

    class MoE(nn.Module):  # pylint: disable=abstract-method
        def __init__(self):
            super().__init__()
            self.gate = Gate(4, 16)
            self.top_k = 2
            self.n_group = 1
            self.norm_topk_prob = True
            self.routed_scaling_factor = 2.5

    torch.manual_seed(5)
    moe = MoE()
    hidden = torch.randn(2, 3, 16)
    idx, w = _sigmoid_group_router(moe, hidden)

    logits = moe.gate(hidden)
    scores = logits.sigmoid()
    choice = scores + moe.gate.e_score_correction_bias
    ref_idx = choice.topk(2, dim=-1, sorted=False)[1]
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 2.5
    assert torch.equal(idx, ref_idx), "case: sigmoid_group_router_adapter"
    torch.testing.assert_close(w, ref_w.to(w.dtype),
                               msg="case: sigmoid_group_router_adapter")


# ==========================================================================
# Family 10: planner TP-extend-EP contracts + expert mesh layout
# Sources: test_planner_ep_extend_contract, test_planner_ep1_no_extend,
#          test_planner_batched_contract, test_planner_batched_ep1_no_mark,
#          test_expert_mesh_layout_mapping
# ==========================================================================

def test_planner_ep_extend_contracts(tiny_hf_native_moe, tiny_hf_batched_moe,
                                     make_mesh):
    """D-10/D-11 TP-extend-EP planner contracts (identity boundary, expert
    params sharded only by {EP: Shard(0)}, router fully replicated) and the
    derived expert mesh layout mapping."""
    # ── case: test_planner_ep_extend_contract ──
    # mesh (dp=4, tp=2), ep=4 -> extended EP groups {0,1,2,3}/{4,5,6,7}
    # (user example).
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4, \
        "case: planner_ep_extend_contract"   # ep_size is the extended EP group size
    assert spec._ep_stack, "case: planner_ep_extend_contract"  # stacked metadata unchanged

    # expert params: only {EP: Shard(0)} (expert-dim sharding), no TP key,
    # no second axis
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_ep_extend_contract"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_ep_extend_contract"
        assert len(p) == 2, \
            "case: planner_ep_extend_contract"   # only CP(Replicate) + EP keys

    # router fully replicated (local chunk computation)
    assert spec.params["gate.weight"][TP] == Replicate(), \
        "case: planner_ep_extend_contract"

    # Boundary contract identity (SP-in): in_dst/out_src/out_dst all
    # TP Shard(1)
    assert spec.in_dst["x_BLD"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"
    assert spec.out_src["output"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"
    assert spec.out_dst["output"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"
    # in_src unchanged from the upstream contract (chained validation passes)
    assert spec.in_src["x_BLD"][TP] == Shard(1), \
        "case: planner_ep_extend_contract"

    # ── case: test_planner_ep1_no_extend ──
    # ep=1 -> no TP-extend-EP, per-expert entries kept (TP-only semantics
    # correct).
    mesh1 = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh1, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0, "case: planner_ep1_no_extend"
    assert spec._ep_stack == {}, "case: planner_ep1_no_extend"
    assert "experts.0.gate_proj.weight" in spec.params, \
        "case: planner_ep1_no_extend"
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0), \
        "case: planner_ep1_no_extend"

    # ── case: test_planner_batched_contract ──
    # D-11 batched layout (experts.gate_up_proj [E,2I,H]): no stacking
    # needed, marked directly with {EP: Shard(0)}; arch=qwen3moe ->
    # TopKRouter module adapter.
    plan = ShardingPlanner().plan(tiny_hf_batched_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4, "case: planner_batched_contract"
    assert spec._ep_stack == {}, \
        "case: planner_batched_contract"   # batched is stacked by nature; no stacking needed

    # expert params: only {EP: Shard(0)} (expert-dim sharding), no TP key,
    # no second axis
    for proj in ("gate_up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0), "case: planner_batched_contract"
        assert TP not in p and p[CP] == Replicate(), \
            "case: planner_batched_contract"
        assert len(p) == 2, "case: planner_batched_contract"

    # router (TopKRouter.weight) fully replicated; boundary identity
    assert spec.params["gate.weight"][TP] == Replicate(), \
        "case: planner_batched_contract"
    assert spec.in_dst["x_BLD"][TP] == Shard(1), "case: planner_batched_contract"
    assert spec.out_src["output"][TP] == Shard(1), \
        "case: planner_batched_contract"
    assert spec.out_dst["output"][TP] == Shard(1), \
        "case: planner_batched_contract"

    # ── case: test_planner_batched_ep1_no_mark ──
    # batched layout with ep=1 -> no marking (_ep_size == 0).
    #
    # Writing fused weights as bare TP Shard(1) (the old D-08 semantics) is
    # fail-fasted by _finalize_fused_expert_tp_guard -- contiguous block
    # sharding is incompatible with the in-forward chunk; a legal
    # configuration must override to TP Replicate (guard solution 2, and
    # out_src's TP must likewise be changed from the template-derived Partial
    # to Replicate).
    def rep():
        return {TP: Replicate(), CP: Replicate()}
    overrides = {"*.mlp": ModuleShardingSpec(
        params={
            "gate.weight": rep(),
            "experts.gate_up_proj": rep(),
            "experts.down_proj": rep(),
        },
        out_src={"output": rep()},
    )}
    plan = ShardingPlanner(plan_overrides=overrides).plan(
        tiny_hf_batched_moe, mesh1, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0, "case: planner_batched_ep1_no_mark"
    assert spec.params["experts.gate_up_proj"][TP] == Replicate(), \
        "case: planner_batched_ep1_no_mark"

    # ── case: test_expert_mesh_layout_mapping ──
    # Derived expert mesh: the dense region is flattened into (edp, ep), and
    # an EP group is ep_size consecutive ranks in the flattened order
    # (spanning the TP group first, then extending to adjacent dp ranks).
    # mesh rank = d*2 + t
    # ep=4 (user example): EP groups {0,1,2,3} / {4,5,6,7} -- spanning
    # 2 TP groups x 2 dp
    shape, names, rank_list = _expert_mesh_layout(mesh, ("dp", "tp"), 4)
    assert shape == (2, 4), "case: expert_mesh_layout_mapping"
    assert names == ("edp", "ep"), "case: expert_mesh_layout_mapping"
    assert rank_list == (0, 1, 2, 3, 4, 5, 6, 7), \
        "case: expert_mesh_layout_mapping"

    # ep=2: EP groups {0,1}/{2,3}/{4,5}/{6,7} -- i.e. the TP groups
    shape, names, _ = _expert_mesh_layout(mesh, ("dp", "tp"), 2)
    assert shape == (4, 2), "case: expert_mesh_layout_mapping"
    assert names == ("edp", "ep"), "case: expert_mesh_layout_mapping"

    # Non-divisible -> error
    with pytest.raises(ValueError, match="must divide"):
        _expert_mesh_layout(mesh, ("dp", "tp"), 3)
