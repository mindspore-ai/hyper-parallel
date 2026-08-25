# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_cp_attention_wrappers.py: merged core suite (feature-combination slim edition, 12 cases).

Sources: test_s3_inner_attn_detect.py, test_s3_shard_batch.py, test_s3_shard_seq_lens.py
Also merged test_mla_dsa_cp_wrapper.py: special CP wrapper handlers for MLA/DSA attention.

Organized by feature family: each test function runs all atomic checks of its
family in sequence; section comments mark the atomic case names; assertions
carry "case: <atomic case name>" identification messages; error paths are
verified sequentially via _expect_raise.
"""

import re
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.auto_models.components.distributed.cp_utils import (
    _shard_seq_lens_for_cp,
    shard_batch_for_cp,
)
from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
    INNER_WRAPPER_REGISTRY,
    _slice_sequence,
    is_flex_attention,
    is_hf_style_attention,
    is_sdpa_attention,
    mla_dsa_ulysses_cp_wrapper,
    sdpa_qkv_cp_wrapper,
)
from hyper_parallel.auto_models.components.distributed.sharding_applier import (
    _resolve_inner_target,
    _resolve_inner_wrapper,
    _wrap_inner_attention,
)
from hyper_parallel.auto_models.components.distributed.injection import (
    inner_wrapper,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
    TP,
)
try:
    from hyper_parallel.auto_models.trainer.config import Target
    _HAS_TRAINER_CONFIG = True
except ImportError:
    # trainer.config pulls in model_transform / checkpoint conversion, which
    # require a newer transformers than some CI gates provide.
    _HAS_TRAINER_CONFIG = False
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Replicate,
    Shard,
)


def _expect_raise(case, exc, match, fn, *args, **kwargs):
    """Sequential error-path check: fn must raise exc and its message must
    re.search match.

    Same semantics as pytest.raises(match=...), but failures report
    "case: <atomic case name>".
    """
    try:
        fn(*args, **kwargs)
    except exc as err:
        assert re.search(match, str(err)), (
            f"case: {case}: message {err!r} does not match {match!r}")
        return
    except BaseException as err:
        raise AssertionError(
            f"case: {case}: expected {exc.__name__}, got {err!r}") from err
    raise AssertionError(f"case: {case}: no {exc.__name__} raised")


# ==========================================================================
# Shared test fixture modules (used by the inner attention resolution chain)
# ==========================================================================

class _Cfg:
    _attn_implementation = "sdpa"


class HFSdpaAttention(nn.Module):
    """HF style: holds q/k/v_proj, forward(hidden_states)."""

    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, hidden_states):
        return hidden_states


class NeMoAttention(nn.Module):
    """NeMo style: inner_attention submodule with forward(q,k,v)."""

    class Inner(nn.Module):
        def forward(self, q, k, v):
            return q

    def __init__(self):
        super().__init__()
        self.inner_attention = self.Inner()

    def forward(self, hidden_states):
        return hidden_states


class FlexHFAattention(HFSdpaAttention):
    class _FlexCfg:
        _attn_implementation = "flex_attention"

    def __init__(self):
        super().__init__()
        self.config = self._FlexCfg()


class _Bare(nn.Module):
    """Bare module without submodules (inner_target='self' scenario)."""

    def forward(self, x):
        return x


class _FakeCpMesh:
    """Single-process fake cp mesh (size=1, only for testing injection paths,
    no communication)."""

    def size(self):
        return 1


# ==========================================================================
# Family 1: declaration resolution - explicit inner_target resolution
# ==========================================================================

def test_resolve_inner_target_declaration():
    """Explicit inner_target resolution (the attention-domain auto-locating
    heuristic was removed on 2026-08-10)."""

    # ── case: undeclared_raises ── inner_target not declared → fail-fast
    _expect_raise("undeclared_raises", ValueError, "inner_target",
                  _resolve_inner_target, NeMoAttention(),
                  spec=ModuleShardingSpec())

    # ── case: user_inner_target_attr ── explicit attribute name → resolves to that submodule
    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention")
    assert _resolve_inner_target(m, spec=spec) is m.inner_attention, \
        "case: user_inner_target_attr"

    # ── case: user_inner_target_self ── 'self' → the module itself is the target
    m = _Bare()
    spec = ModuleShardingSpec(inner_target="self")
    assert _resolve_inner_target(m, spec=spec) is m, \
        "case: user_inner_target_self"

    # ── case: user_inner_target_missing_raises ── typo → fail-fast
    spec = ModuleShardingSpec(inner_target="core_atn")  # typo
    _expect_raise("user_inner_target_missing_raises", ValueError,
                  "inner_target", _resolve_inner_target, _Bare(), spec=spec)


# ==========================================================================
# Family 2: declaration gating + registry (lookup / extension / undecorated rejection)
# ==========================================================================

def test_resolve_inner_wrapper_declaration_and_registry():
    """Declaration gating and registry lookup/extension/rejection."""
    # ── case: no_declaration_returns_none ── no declaration → None (derivation gating)
    resolved = _resolve_inner_wrapper(
        NeMoAttention(), ModuleShardingSpec(), _FakeCpMesh(), None, ())
    assert resolved is None, "case: no_declaration_returns_none"

    # ── case: needs_cp_attn_alone_returns_none ── _needs_cp_attn is only
    # template metadata and no longer triggers any injection
    spec = ModuleShardingSpec(_needs_cp_attn=True)
    assert _resolve_inner_wrapper(
        NeMoAttention(), spec, _FakeCpMesh(), None, ()) is None, \
        "case: needs_cp_attn_alone_returns_none (NeMo)"
    assert _resolve_inner_wrapper(
        HFSdpaAttention(), spec, _FakeCpMesh(), None, ()) is None, \
        "case: needs_cp_attn_alone_returns_none (HF)"

    # ── case: inner_target_without_wrapper_raises ── inner_target only
    # locates the target; declaring it alone → fail-fast
    spec = ModuleShardingSpec(inner_target="inner_attention")
    _expect_raise("inner_target_without_wrapper_raises", ValueError,
                  "inner_wrapper", _resolve_inner_wrapper,
                  NeMoAttention(), spec, _FakeCpMesh(), None, ())

    # ── case: wrapper_without_inner_target_raises ── the two fields must be declared explicitly as a pair
    spec = ModuleShardingSpec(inner_wrapper="sdpa_qkv", region_dispatch=False)
    _expect_raise("wrapper_without_inner_target_raises", ValueError,
                  "inner_target", _resolve_inner_wrapper,
                  _Bare(), spec, _FakeCpMesh(), None, ())

    # ── case: str_registry_lookup ── inner_wrapper='sdpa_qkv' (str) →
    # explicitly picks a registry scheme (HF modules may also explicitly pick
    # the qkv path; the user is responsible)
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper="sdpa_qkv", region_dispatch=False)
    name, _, _ = _resolve_inner_wrapper(
        HFSdpaAttention(), spec, _FakeCpMesh(), None, ())
    assert name == "sdpa_qkv", "case: str_registry_lookup"

    # ── case: str_unknown_name_raises ── unregistered name → fail-fast and list available names
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper="sdpa_hff", region_dispatch=False)  # typo
    _expect_raise("str_unknown_name_raises", ValueError,
                  "INNER_WRAPPER_REGISTRY", _resolve_inner_wrapper,
                  NeMoAttention(), spec, _FakeCpMesh(), None, ())

    # ── case: user_registry_extension ── a user-registered named scheme can be referenced by name
    calls = []

    @inner_wrapper
    def my_fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        calls.append(target_module)

    INNER_WRAPPER_REGISTRY["test_custom"] = my_fn
    try:
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="test_custom", region_dispatch=False)
        name, target, apply_fn = _resolve_inner_wrapper(
            NeMoAttention(), spec, _FakeCpMesh(), None, ())
        assert name == "test_custom", "case: user_registry_extension"
        apply_fn()
        assert calls == [target], "case: user_registry_extension"
    finally:
        INNER_WRAPPER_REGISTRY.pop("test_custom")

    # ── case: undecorated_registry_entry_raises ── undecorated function in
    # the registry → fail-fast (the four built-in paths are decorated)
    INNER_WRAPPER_REGISTRY["test_undecorated"] = lambda t, cm: None
    try:
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="test_undecorated", region_dispatch=False)
        _expect_raise("undecorated_registry_entry_raises", TypeError,
                      "@inner_wrapper", _resolve_inner_wrapper,
                      NeMoAttention(), spec, _FakeCpMesh(), None, ())
    finally:
        INNER_WRAPPER_REGISTRY.pop("test_undecorated")


# ==========================================================================
# Family 3: callable / Target forms + generalization without a CP axis
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_callable_target_forms_and_no_cp_generalization():
    """Callable/Target declaration forms and generalization without a CP axis."""
    # ── case: callable_custom ── inner_wrapper=callable → ('custom', target)
    m = NeMoAttention()

    @inner_wrapper
    def my_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        pass

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=my_wrapper, region_dispatch=False)
    name, target, _ = _resolve_inner_wrapper(
        m, spec, _FakeCpMesh(), None, ())
    assert name == "custom", "case: callable_custom"
    assert target is m.inner_attention, "case: callable_custom"

    # ── case: target_builtin_inplace ── Target points to a built-in function
    # in the repo: replaced in place after build; the name is target_path
    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=Target(
        sdpa_qkv_cp_wrapper,
        target_path="hyper_parallel.auto_models.components.distributed."
                    "cp_wrappers.sdpa_qkv_cp_wrapper"), region_dispatch=False)
    name, target, apply_fn = _resolve_inner_wrapper(
        m, spec, _FakeCpMesh(), None, ())
    assert name.endswith("sdpa_qkv_cp_wrapper"), "case: target_builtin_inplace"
    assert target is m.inner_attention, "case: target_builtin_inplace"
    apply_fn()
    q = torch.randn(1, 1, 2, 4)
    assert m.inner_attention(q, q, q) is not None, "case: target_builtin_inplace"

    # ── case: target_factory_returning_callable ── Target factory returns a
    # callable → applied per the custom wrapper contract (target, cp_mesh)
    received = []

    @inner_wrapper
    def my_factory(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        @inner_wrapper
        def inner(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            received.append((target_module, cp_mesh))
        return inner

    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=Target(
        my_factory, target_path="tests.my_factory"), region_dispatch=False)
    cp_mesh = _FakeCpMesh()
    _, target, apply_fn = _resolve_inner_wrapper(m, spec, cp_mesh, None, ())
    apply_fn()
    assert received == [(target, cp_mesh)], "case: target_factory_returning_callable"

    # ── case: context_filled_by_name ── all required mesh-family parameters
    # receive framework-provided values (cp_mesh/ep_mesh are None when there
    # is no cp/ep axis)
    seen = {}

    @inner_wrapper
    def ctx_factory(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        seen.update(target_module=target_module, cp_mesh=cp_mesh,
                    ep_mesh=ep_mesh)
        # implicit None return: in-place replacement style (this check does not actually replace)

    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=Target(
        ctx_factory, target_path="tests.my_factory"), region_dispatch=False)
    cp_mesh = _FakeCpMesh()
    _, target, apply_fn = _resolve_inner_wrapper(m, spec, cp_mesh, None, ())
    apply_fn()
    assert seen["target_module"] is target, "case: context_filled_by_name"
    assert seen["cp_mesh"] is cp_mesh, "case: context_filled_by_name"
    assert seen["ep_mesh"] is None, "case: context_filled_by_name"

    # ── case: custom_callable_fires_without_cp ── no cp axis (cp_mesh=None):
    # the custom callable still resolves and applies (declaration means
    # application; no more CP gating)
    m = NeMoAttention()
    received = []

    @inner_wrapper
    def no_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        received.append((target_module, cp_mesh))

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=no_cp_wrapper,
                              inner_out_src="first_input", region_dispatch=False)
    name = _wrap_inner_attention(m, None, spec=spec)
    assert name == "custom", "case: custom_callable_fires_without_cp"
    assert received == [(m.inner_attention, None)], \
        "case: custom_callable_fires_without_cp"
    assert spec._resolved_inner_target == "inner_attention", \
        "case: custom_callable_fires_without_cp"

    # ── case: user_registered_name_fires_without_cp ── a user-registered
    # named scheme is not a built-in CP wrapper → also usable without a cp axis
    calls = []

    @inner_wrapper
    def no_cp_fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        calls.append(cp_mesh)

    INNER_WRAPPER_REGISTRY["test_no_cp"] = no_cp_fn
    try:
        spec = ModuleShardingSpec(inner_target="inner_attention",
                                  inner_wrapper="test_no_cp",
                                  inner_out_src="first_input", region_dispatch=False)
        name = _wrap_inner_attention(NeMoAttention(), None, spec=spec)
        assert name == "test_no_cp", "case: user_registered_name_fires_without_cp"
        assert calls == [None], "case: user_registered_name_fires_without_cp"
    finally:
        INNER_WRAPPER_REGISTRY.pop("test_no_cp")


# ==========================================================================
# Family 4: error paths (resolution/config/runtime fail-fast merged, sequential raises)
# ==========================================================================

@pytest.mark.skipif(not _HAS_TRAINER_CONFIG,
                    reason="trainer.config import chain needs newer transformers")
def test_error_paths_combined(make_mesh):
    """Resolution/config/runtime fail-fast error paths, checked sequentially."""
    # ── case: wrong_type_raises ── inner_wrapper is neither str/callable/Target
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper=123, region_dispatch=False)
    _expect_raise("wrong_type_raises", TypeError, "inner_wrapper",
                  _resolve_inner_wrapper, NeMoAttention(), spec,
                  _FakeCpMesh(), None, ())

    # ── case: target_bad_return_raises ── Target returns neither None nor a callable
    @inner_wrapper
    def bad_factory(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        return 42

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=Target(
        bad_factory, target_path="tests.bad_factory"), region_dispatch=False)
    _, _, apply_fn = _resolve_inner_wrapper(
        NeMoAttention(), spec, _FakeCpMesh(), None, ())
    _expect_raise("target_bad_return_raises", TypeError, "inner_wrapper",
                  apply_fn)

    # ── case: target_undecorated_factory_raises ── Target points to an undecorated function
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper=Target(
        lambda: None, target_path="tests.undecorated"), region_dispatch=False)
    _expect_raise("target_undecorated_factory_raises", TypeError,
                  "@inner_wrapper", _resolve_inner_wrapper,
                  NeMoAttention(), spec, _FakeCpMesh(), None, ())

    # ── case: target_typo_config_key_raises ── key not declared by the
    # wrapper (typo: cp_mesg) → fail-fast and list the valid parameters
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper=Target(
        sdpa_qkv_cp_wrapper,
        target_path="hyper_parallel.auto_models.components.distributed."
                    "cp_wrappers.sdpa_qkv_cp_wrapper",
        cp_mesg="oops"), region_dispatch=False)                      # typo: should be cp_mesh
    _expect_raise("target_typo_config_key_raises", ValueError, "cp_mesh",
                  _resolve_inner_wrapper, NeMoAttention(), spec,
                  _FakeCpMesh(), None, ())

    # ── case: builtin_str_without_cp_raises ── built-in CP scheme (str) + no
    # cp axis → wrapper self-check fail-fast, pointing to local_compute_fn
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper="sdpa_hf", region_dispatch=False)
    _expect_raise("builtin_str_without_cp_raises", ValueError,
                  "local_compute_fn", _wrap_inner_attention,
                  HFSdpaAttention(), None, spec=spec)

    # ── case: builtin_callable_without_cp_raises ── built-in CP scheme
    # (callable passed directly) + no cp axis → fail-fast
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=sdpa_qkv_cp_wrapper, region_dispatch=False)
    _expect_raise("builtin_callable_without_cp_raises", ValueError,
                  "active cp axis", _wrap_inner_attention,
                  NeMoAttention(), None, spec=spec)

    # ── case: builtin_target_without_cp_raises ── built-in CP scheme (Target
    # form) + no cp axis → fail-fast
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=Target(
        sdpa_qkv_cp_wrapper,
        target_path="hyper_parallel.auto_models.components.distributed."
                    "cp_wrappers.sdpa_qkv_cp_wrapper"), region_dispatch=False)
    _expect_raise("builtin_target_without_cp_raises", ValueError,
                  "active cp axis", _wrap_inner_attention,
                  NeMoAttention(), None, spec=spec)

    # ── case: sdpa_hf_not_fired_raises ── firing detection: 'sdpa_hf'
    # intercepts but the module never calls F.sdpa internally → RuntimeError
    class FakeHFAttn(HFSdpaAttention):
        def forward(self, hidden_states):
            # custom implementation that never calls F.scaled_dot_product_attention
            return self.q_proj(hidden_states)

    m = FakeHFAttn()
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper="sdpa_hf", region_dispatch=False)
    _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
    _expect_raise("sdpa_hf_not_fired_raises", RuntimeError,
                  "did not intercept", m, torch.randn(1, 2, 8))

    # ── case: inner_without_declaration_fails ── case B (inner submodule)
    # without an inner_out_src declaration → fail-fast at installation
    def _make_passthrough_wrapper():
        @inner_wrapper
        def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
            orig = target_module.forward

            def fwd(q, k, v):
                return orig(q, k, v)
            target_module.forward = fwd
        return w

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_passthrough_wrapper(),
                              region_dispatch=False)
    _expect_raise("inner_without_declaration_fails", ValueError,
                  "inner_out_src", _wrap_inner_attention,
                  NeMoAttention(), _FakeCpMesh(), spec=spec)

    # ── case: bad_sentinel_fails ── illegal string sentinel for inner_out_src
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_passthrough_wrapper(),
                              inner_out_src="bogus", region_dispatch=False)
    _expect_raise("bad_sentinel_fails", ValueError, "first_input",
                  _wrap_inner_attention, NeMoAttention(), _FakeCpMesh(),
                  spec=spec)

    # ── case: first_input_tuple_output_fails ── first_input sentinel +
    # multiple outputs → runtime fail-fast pointing to an explicit declaration
    mesh = make_mesh((1,), ("tp",))
    m = NeMoAttention()

    @inner_wrapper
    def tuple_out_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        orig = target_module.forward

        def fwd(q, k, v):
            return orig(q, k, v), q   # multiple outputs
        target_module.forward = fwd

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=tuple_out_wrapper,
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = DTensor.from_local(torch.randn(1, 2, 4, 8), mesh, (Replicate(),))
    _expect_raise("first_input_tuple_output_fails", RuntimeError,
                  "only supports a single output", m.inner_attention, q, q, q)

    # ── case: wrapper_without_region_dispatch_fails ── inner_wrapper declared
    # but region_dispatch missing → fail-fast (no default)
    @inner_wrapper
    def no_dispatch_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        pass

    spec = ModuleShardingSpec(inner_wrapper=no_dispatch_wrapper)
    _expect_raise("wrapper_without_region_dispatch_fails", ValueError,
                  "region_dispatch", _resolve_inner_wrapper,
                  NeMoAttention(), spec, _FakeCpMesh(), None, ())


# ==========================================================================
# Family 5: apply write-back (_resolved_inner_wrapper/_resolved_inner_target + takeover)
# ==========================================================================

def test_apply_writeback_and_takeover():
    """Apply-time write-back of resolved names/targets and custom takeover."""
    # ── case: apply_writes_resolved_name ── spec._resolved_inner_wrapper is
    # written back after application (plan introspection)
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper="sdpa_qkv",
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(NeMoAttention(), _FakeCpMesh(), spec=spec)
    assert spec._resolved_inner_wrapper == "sdpa_qkv", \
        "case: apply_writes_resolved_name"

    # ── case: apply_without_declaration_returns_none ── no declaration →
    # returns None and does not inject (even with _needs_cp_attn metadata)
    spec = ModuleShardingSpec(_needs_cp_attn=True)
    m = NeMoAttention()
    orig_fwd = m.inner_attention.forward
    assert _wrap_inner_attention(m, _FakeCpMesh(), spec=spec) is None, \
        "case: apply_without_declaration_returns_none"
    assert m.inner_attention.forward == orig_fwd, \
        "case: apply_without_declaration_returns_none"

    # ── case: custom_callable_takeover ── custom callable takes over
    # entirely: invoked with (target, cp_mesh) and replaces forward
    calls = []

    @inner_wrapper
    def my_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        calls.append((target_module, cp_mesh))
        orig = target_module.forward

        def wrapped(x, *args, **kwargs):
            return orig(x)

        target_module.forward = wrapped

    m = _Bare()
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper=my_cp_wrapper, region_dispatch=False)
    mesh = _FakeCpMesh()
    _wrap_inner_attention(m, mesh, spec=spec)
    assert calls and calls[0][1] is mesh, "case: custom_callable_takeover"
    assert spec._resolved_inner_wrapper == "custom", \
        "case: custom_callable_takeover"
    # after wrapping, the forward signature is rewritten dynamically; static
    # checks would false-positive against the original signature
    assert m.forward(torch.tensor(1), None, None) is not None, \
        "case: custom_callable_takeover"  # pylint: disable=too-many-function-args

    # ── case: custom_callable_with_inner_target ── inner_target +
    # inner_wrapper combination: target is the user-specified submodule
    m = NeMoAttention()
    received = []

    @inner_wrapper
    def target_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        received.append(target_module)

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=target_wrapper,
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
    assert received == [m.inner_attention], \
        "case: custom_callable_with_inner_target"

    # ── case: resolved_inner_target_written_back ── resolution result made
    # visible: spec._resolved_inner_target writes back the attribute name/"self"
    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper="sdpa_qkv",
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
    assert spec._resolved_inner_target == "inner_attention", \
        "case: resolved_inner_target_written_back"

    m2 = HFSdpaAttention()
    spec2 = ModuleShardingSpec(inner_target="self",
                               inner_wrapper="sdpa_hf", region_dispatch=False)
    _wrap_inner_attention(m2, _FakeCpMesh(), spec=spec2)
    assert spec2._resolved_inner_target == "self", \
        "case: resolved_inner_target_written_back"

    # ── case: str_pinned_wrapper_applied ── inner_wrapper='sdpa_qkv'
    # explicitly pinned: the registry function is applied
    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper="sdpa_qkv",
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
    assert spec._resolved_inner_wrapper == "sdpa_qkv", \
        "case: str_pinned_wrapper_applied"
    q = torch.randn(1, 1, 2, 4)
    assert m.inner_attention(q, q, q) is not None, \
        "case: str_pinned_wrapper_applied"


# ==========================================================================
# Family 6: decorator validation + injection discipline (injection.py)
# ==========================================================================

def test_decorator_and_injection_discipline():
    """@inner_wrapper decorator validation and injection discipline."""
    # ── case: undecorated_callable_raises ── undecorated callable → fail-fast
    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper=lambda t, cm: None, region_dispatch=False)
    _expect_raise("undecorated_callable_raises", TypeError, "@inner_wrapper",
                  _resolve_inner_wrapper, NeMoAttention(), spec,
                  _FakeCpMesh(), None, ())

    # ── case: wrong_kind_decorator_raises ── wrong decorator kind
    # (@local_compute used on an inner_wrapper) → fail-fast
    from hyper_parallel.auto_models.components.distributed.injection import (
        local_compute,
    )

    @local_compute
    def my_compute(mesh, tp_mesh, cp_mesh, ep_mesh):
        def compute_fn(module, x):
            return x
        return compute_fn

    spec = ModuleShardingSpec(inner_target="self",
                              inner_wrapper=my_compute, region_dispatch=False)
    _expect_raise("wrong_kind_decorator_raises", TypeError,
                  "wrong decorator kind", _resolve_inner_wrapper,
                  NeMoAttention(), spec, _FakeCpMesh(), None, ())

    # ── case: decorator_requires_mesh_family ── missing any of
    # mesh/tp_mesh/cp_mesh/ep_mesh → fail-fast at import time
    with pytest.raises(TypeError, match="missing required context parameters"):  # case: decorator_requires_mesh_family
        @inner_wrapper
        def bad_missing(target_module, mesh):   # missing tp_mesh/cp_mesh/ep_mesh
            pass

    # ── case: decorator_rejects_context_default ── context parameters must not have defaults
    with pytest.raises(TypeError, match="must not have a default"):  # case: decorator_rejects_context_default
        @inner_wrapper
        def bad_default(target_module, mesh, tp_mesh, cp_mesh, ep_mesh=None):
            pass

    # ── case: decorator_rejects_var_kwargs ── *args/**kwargs rejected at import time
    with pytest.raises(TypeError, match="\\*args/\\*\\*kwargs"):  # case: decorator_rejects_var_kwargs
        @inner_wrapper
        def bad_kwargs(target_module, mesh, tp_mesh, cp_mesh, ep_mesh, **ctx):
            pass

    # ── case: incompatible_replacement_forward_raises ── the replacement
    # forward cannot accept the original forward's required arguments → fail-fast
    m = NeMoAttention()   # inner forward(q, k, v)

    @inner_wrapper
    def bad_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        def wrapped(q):     # drops k/v -- incompatible with the original forward
            return q
        target_module.forward = wrapped

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=bad_wrapper,
                              inner_out_src="first_input", region_dispatch=False)
    _expect_raise("incompatible_replacement_forward_raises", TypeError,
                  "incompatible with the original forward",
                  _wrap_inner_attention, m, _FakeCpMesh(), spec=spec)


# ==========================================================================
# Family 7: dual-mode adapter - validate rewrap (DTensor ↔ local conversion managed)
# ==========================================================================

def _dtensor(mesh, local):
    return DTensor.from_local(local, mesh, (Replicate(),))


def _make_validate_wrapper(seen=None):
    """User wrapper: records whether q is a DTensor, then calls the original forward."""

    @inner_wrapper
    def w(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        orig = target_module.forward

        def fwd(q, k, v):
            if seen is not None:
                seen["q_is_dtensor"] = isinstance(q, DTensor)
            return orig(q, k, v)
        target_module.forward = fwd
    return w


def test_dual_mode_adapter_rewrap(make_mesh):
    """Validate-mode rewrap: DTensor/local conversion is managed by the adapter."""
    # ── case: first_input_rewrap_validate ── validate: DTensor inputs are
    # to_local'd (the user only sees local tensors); the output is rewrapped
    # into a DTensor with the first input's placements
    mesh = make_mesh((1,), ("tp",))
    m = NeMoAttention()
    seen = {}
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_validate_wrapper(seen),
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = _dtensor(mesh, torch.randn(1, 2, 4, 8))
    out = m.inner_attention(q, q, q)
    assert seen["q_is_dtensor"] is False, "case: first_input_rewrap_validate"
    assert isinstance(out, DTensor), "case: first_input_rewrap_validate"
    assert tuple(out.placements) == (Replicate(),), \
        "case: first_input_rewrap_validate"

    # ── case: production_passthrough ── production (local inputs): passthrough, zero conversion
    m = NeMoAttention()
    seen = {}
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_validate_wrapper(seen),
                              inner_out_src="first_input", region_dispatch=False)
    _wrap_inner_attention(m, _FakeCpMesh(), spec=spec)
    x = torch.randn(1, 2, 4, 8)
    out = m.inner_attention(x, x, x)
    assert seen["q_is_dtensor"] is False, "case: production_passthrough"
    assert not isinstance(out, DTensor), "case: production_passthrough"
    torch.testing.assert_close(out, x, msg="case: production_passthrough")

    # ── case: explicit_placement_rewrap ── case B explicit placement
    # declaration: rewrap per the declaration (the spec form holds Placement
    # objects; YAML strings are resolved during desugaring)
    mesh = make_mesh((1,), ("tp",))
    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_validate_wrapper(),
                              inner_out_src={TP: Replicate()}, region_dispatch=False)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = _dtensor(mesh, torch.randn(1, 2, 4, 8))
    out = m.inner_attention(q, q, q)
    assert isinstance(out, DTensor), "case: explicit_placement_rewrap"
    assert tuple(out.placements) == (Replicate(),), \
        "case: explicit_placement_rewrap"

    # ── case: self_target_uses_boundary_out_src ── case A (target=self):
    # rewrap per the boundary spec.out_src declaration
    mesh = make_mesh((1,), ("tp",))
    m = HFSdpaAttention()

    @inner_wrapper
    def hf_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        orig = target_module.forward

        def fwd(hidden_states, *args, **kwargs):
            return orig(hidden_states)
        target_module.forward = fwd

    spec = ModuleShardingSpec(inner_target="self", inner_wrapper=hf_wrapper,
                              out_src={"output": {TP: Replicate()}}, region_dispatch=False)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    h = _dtensor(mesh, torch.randn(1, 2, 8))
    out = m(h)
    assert isinstance(out, DTensor), "case: self_target_uses_boundary_out_src"
    assert tuple(out.placements) == (Replicate(),), \
        "case: self_target_uses_boundary_out_src"

    # ── case: multi_output_declared ── case B multiple outputs:
    # {name: placement} rewrapped positionally in declared key order
    mesh = make_mesh((1,), ("tp",))

    class Pair(nn.Module):
        def forward(self, q, k, v):
            return q, k + v

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = Pair()

    m = Outer()
    orig = m.inner_attention.forward

    @inner_wrapper
    def multi_out_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        def fwd(q, k, v):
            a, b = orig(q, k, v)
            return a, b
        target_module.forward = fwd

    spec = ModuleShardingSpec(
        inner_target="inner_attention",
        inner_wrapper=multi_out_wrapper,
        inner_out_src={"a": {TP: Replicate()}, "b": {TP: Replicate()}}, region_dispatch=False)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = _dtensor(mesh, torch.randn(1, 2, 4, 8))
    a, b = m.inner_attention(q, q, q)
    assert isinstance(a, DTensor) and isinstance(b, DTensor), \
        "case: multi_output_declared"
    assert tuple(b.placements) == (Replicate(),), "case: multi_output_declared"


# ==========================================================================
# Family 8: region_dispatch=True dispatch-through validation (DTensor passed
# straight into the user forward + real validation)
# ==========================================================================

def test_dispatch_through_validation(make_mesh):
    """region_dispatch=True: DTensors pass straight through with real validation."""
    # ── case: dispatch_through_first_input ── validate: the user forward sees
    # DTensors (dispatch-through); propagation result == first input's layout → passes
    mesh = make_mesh((1,), ("tp",))
    m = NeMoAttention()
    seen = {}
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_validate_wrapper(seen),
                              inner_out_src="first_input",
                              region_dispatch=True)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = _dtensor(mesh, torch.randn(1, 2, 4, 8))
    out = m.inner_attention(q, q, q)
    assert seen["q_is_dtensor"] is True, "case: dispatch_through_first_input"
    assert isinstance(out, DTensor), "case: dispatch_through_first_input"
    assert tuple(out.placements) == (Replicate(),), \
        "case: dispatch_through_first_input"

    # ── case: dispatch_through_mismatch_fails ── real validation: explicit
    # declaration {tp: Shard(0)}, but the propagation result is Replicate →
    # PlacementMismatchError
    mesh = make_mesh((1,), ("tp",))
    m = NeMoAttention()
    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=_make_validate_wrapper(),
                              inner_out_src={TP: Shard(0)},
                              region_dispatch=True)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = _dtensor(mesh, torch.randn(1, 2, 4, 8))
    _expect_raise("dispatch_through_mismatch_fails", Exception,
                  "inner_out_src", m.inner_attention, q, q, q)

    # ── case: dispatch_through_broken_chain_fails ── lie detection: declares
    # True but the injected code unwraps the DTensor to local (breaking the
    # dispatch chain) → fail-fast
    mesh = make_mesh((1,), ("tp",))
    m = NeMoAttention()

    @inner_wrapper
    def broken_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
        orig = target_module.forward

        def fwd(q, k, v):
            return orig(q, k, v).to_local()   # breaks the dispatch chain
        target_module.forward = fwd

    spec = ModuleShardingSpec(inner_target="inner_attention",
                              inner_wrapper=broken_wrapper,
                              inner_out_src="first_input",
                              region_dispatch=True)
    _wrap_inner_attention(m, None, spec=spec, mesh=mesh,
                          mesh_dim_names=("tp",))
    q = _dtensor(mesh, torch.randn(1, 2, 4, 8))
    _expect_raise("dispatch_through_broken_chain_fails", RuntimeError,
                  "not a DTensor", m.inner_attention, q, q, q)


# ==========================================================================
# Family 9: style-detection helpers + CP sdpa trigger condition
# ==========================================================================

def test_style_helpers_and_sdpa_call_condition():
    """Style-detection helpers (public utilities of cp_wrappers for custom
    wrapper authors) + the D-04 trigger condition (based on CP semantics
    rather than a q_len!=kv_len shape comparison)."""

    # ── case: hf_style_by_signature ──
    assert is_hf_style_attention(HFSdpaAttention()) is True, \
        "case: hf_style_by_signature"

    # ── case: nemo_style_not_hf ──
    inner = NeMoAttention().inner_attention
    assert is_hf_style_attention(inner) is False, "case: nemo_style_not_hf"

    # ── case: sdpa_detection ──
    assert is_sdpa_attention(HFSdpaAttention()) is True, "case: sdpa_detection"
    assert is_flex_attention(HFSdpaAttention()) is False, "case: sdpa_detection"

    # ── case: flex_detection ──
    assert is_flex_attention(FlexHFAattention()) is True, "case: flex_detection"
    assert is_sdpa_attention(FlexHFAattention()) is False, "case: flex_detection"

    # ── case: is_causal_kept_when_cp_inactive ── cp_size=1: is_causal is
    # passed through unchanged; no explicit mask substitution
    from hyper_parallel.auto_models.components.distributed.cp_wrappers import (
        _cp_sdpa_call,
    )
    received = {}

    def fake_sdpa(q, k, v, **kwargs):
        received.update(kwargs)
        return q

    q = torch.randn(1, 1, 4, 2)
    _cp_sdpa_call(fake_sdpa, _FakeCpMesh(), q, q, q, {"is_causal": True})
    assert received.get("is_causal") is True, \
        "case: is_causal_kept_when_cp_inactive"
    assert "attn_mask" not in received, "case: is_causal_kept_when_cp_inactive"


# ==========================================================================
# Family 10: shard_batch_for_cp + _shard_seq_lens_for_cp (table-driven)
# ==========================================================================

class FakeCpMesh:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def _batch(seq_len=10):
    return {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(100, 100 + seq_len).unsqueeze(0),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
        "qkv_format": "thd",
    }


SENTINEL = -1000


def _run_shard_seq_lens(seq_lens, seq_lens_padded, cp_rank, chunk):
    return _shard_seq_lens_for_cp(seq_lens, seq_lens_padded,
                                  cp_rank=cp_rank, chunk=chunk)


def test_cp_sharding_helpers():
    """shard_batch_for_cp and _shard_seq_lens_for_cp, table-driven checks."""
    # ── case: cp_size1_passthrough ──
    b = _batch()
    out = shard_batch_for_cp(b, FakeCpMesh(1, 0))
    assert out is b, "case: cp_size1_passthrough"

    # ── case: equal_split_no_pad ── S=8 split evenly, verified per rank
    b = _batch(seq_len=8)
    for rank, slc in ((0, slice(0, 4)), (1, slice(4, 8))):
        out = shard_batch_for_cp(b, FakeCpMesh(2, rank))
        torch.testing.assert_close(out["input_ids"], b["input_ids"][:, slc],
                                   msg=f"case: equal_split_no_pad rank{rank}")
        torch.testing.assert_close(out["labels"], b["labels"][:, slc],
                                   msg=f"case: equal_split_no_pad rank{rank}")
        assert out["qkv_format"] == "thd", f"case: equal_split_no_pad rank{rank}"

    # ── case: pad_to_2cp_multiple ── S=10 padded to 12 (multiple of 2*cp=4)
    # → chunk=6; the last rank's chunk contains the pad region
    b = _batch(seq_len=10)
    out0 = shard_batch_for_cp(b, FakeCpMesh(2, 0))
    out1 = shard_batch_for_cp(b, FakeCpMesh(2, 1))
    assert out0["input_ids"].shape[1] == 6, "case: pad_to_2cp_multiple"
    assert out1["input_ids"].shape[1] == 6, "case: pad_to_2cp_multiple"
    # rank0: the original first 6
    torch.testing.assert_close(out0["input_ids"], b["input_ids"][:, :6],
                               msg="case: pad_to_2cp_multiple")
    # N6: rank1's pad region has label=-100, input_ids=0, position_ids increasing consecutively
    torch.testing.assert_close(out1["labels"][0, -2:],
                               torch.tensor([-100, -100]),
                               msg="case: pad_to_2cp_multiple")
    torch.testing.assert_close(out1["input_ids"][0, -2:],
                               torch.tensor([0, 0]),
                               msg="case: pad_to_2cp_multiple")
    torch.testing.assert_close(out1["position_ids"][0, -2:],
                               torch.tensor([10, 11]),
                               msg="case: pad_to_2cp_multiple")
    # rank1's valid region == the original [6:10]
    torch.testing.assert_close(out1["input_ids"][0, :4],
                               b["input_ids"][0, 6:10],
                               msg="case: pad_to_2cp_multiple")

    # ── case: non_tensor_passthrough ──
    b = _batch(seq_len=8)
    b["meta"] = "info"
    out = shard_batch_for_cp(b, FakeCpMesh(2, 0))
    assert out["meta"] == "info", "case: non_tensor_passthrough"

    # ── case: seq_lens_recomputed ── rank1 region [4,8): one pack truncated to 4
    b = _batch(seq_len=8)
    b["seq_lens"] = torch.tensor([[8, -1000]])
    b["seq_lens_padded"] = torch.tensor([[8, -1000]])
    out = shard_batch_for_cp(b, FakeCpMesh(2, 1))
    assert out["seq_lens"][0, 0].item() == 4, "case: seq_lens_recomputed"

    # ── case: pack_fully_inside ── pack [0,4) fully inside rank0 [0,4)
    sl = torch.tensor([[4, -1000]])
    slp = torch.tensor([[4, -1000]])
    out_lens, out_pad = _run_shard_seq_lens(sl, slp, cp_rank=0, chunk=4)
    assert out_lens[0, 0].item() == 4, "case: pack_fully_inside"
    assert out_pad[0, 0].item() == 4, "case: pack_fully_inside"

    # ── case: pack_crosses_lo_boundary ── pack [0,6) crosses rank1 lo=4 →
    # truncated to [4,6), length 2
    sl = torch.tensor([[6, -1000]])
    slp = torch.tensor([[6, -1000]])
    out_lens, out_pad = _run_shard_seq_lens(sl, slp, cp_rank=1, chunk=4)
    assert out_lens[0, 0].item() == 2, "case: pack_crosses_lo_boundary"
    assert out_pad[0, 0].item() == 2, "case: pack_crosses_lo_boundary"

    # ── case: pack_crosses_hi_boundary ── pack [2,8) crosses rank0 hi=4 →
    # truncated to [2,4), length 2 (a preceding pack builds the offset)
    sl = torch.tensor([[2, 6, -1000]])
    slp = torch.tensor([[2, 6, -1000]])
    out_lens, out_pad = _run_shard_seq_lens(sl, slp, cp_rank=0, chunk=4)
    assert out_lens[0, 0].item() == 2, "case: pack_crosses_hi_boundary"
    assert out_lens[0, 1].item() == 2, "case: pack_crosses_hi_boundary"
    # sentinel padding
    assert out_lens.shape[1] == 2, "case: pack_crosses_hi_boundary"

    # ── case: pack_fully_outside ── pack [0,4) fully outside rank1 [4,8) →
    # skipped (guard against emptiness → sentinel)
    sl = torch.tensor([[4, -1000]])
    slp = torch.tensor([[4, -1000]])
    out_lens, out_pad = _run_shard_seq_lens(sl, slp, cp_rank=1, chunk=4)
    # max_local_packs=0→1 emptiness guard
    assert out_lens.shape == (1, 1), "case: pack_fully_outside"
    assert out_lens[0, 0].item() == SENTINEL, "case: pack_fully_outside"

    # ── case: sentinel_terminates ── packs after the sentinel are not processed
    sl = torch.tensor([[4, -1000, 4]])
    slp = torch.tensor([[4, -1000, 4]])
    out_lens, _ = _run_shard_seq_lens(sl, slp, cp_rank=0, chunk=8)
    assert out_lens.shape[1] == 1, "case: sentinel_terminates"
    assert out_lens[0, 0].item() == 4, "case: sentinel_terminates"

    # ── case: padded_covers_separator ── seq_lens_padded includes the
    # separator: pack has 3 real tokens + 1 pad; boundary crossing is
    # truncated by the padded length
    sl = torch.tensor([[3, -1000]])
    slp = torch.tensor([[4, -1000]])
    out_lens, out_pad = _run_shard_seq_lens(sl, slp, cp_rank=0, chunk=2)
    # pack [0,4) crosses hi=2: real tokens truncated to [0,2) → 2; pad region [0,2) → 2
    assert out_lens[0, 0].item() == 2, "case: padded_covers_separator"
    assert out_pad[0, 0].item() == 2, "case: padded_covers_separator"
    # rank1 [2,4): real tokens [2,3) → 1; pad [2,4) → 2
    out_lens1, out_pad1 = _run_shard_seq_lens(sl, slp, cp_rank=1, chunk=2)
    assert out_lens1[0, 0].item() == 1, "case: padded_covers_separator"
    assert out_pad1[0, 0].item() == 2, "case: padded_covers_separator"

    # ── case: per_rank_asymmetry ── N5: rank0 and rank1 have different pack
    # intersections → different recomputed results (asserted per rank)
    sl = torch.tensor([[5, 3, -1000]])
    slp = torch.tensor([[5, 3, -1000]])
    out0, _ = _run_shard_seq_lens(sl, slp, cp_rank=0, chunk=4)
    out1, _ = _run_shard_seq_lens(sl, slp, cp_rank=1, chunk=4)
    # rank0 [0,4): pack1 [0,5) truncated → 4; pack2 is outside → 1 entry total
    assert out0[0, 0].item() == 4, "case: per_rank_asymmetry"
    # rank1 [4,8): pack1 [0,5) truncated → 1; pack2 [5,8) complete → 3
    assert out1[0, 0].item() == 1, "case: per_rank_asymmetry"
    assert out1[0, 1].item() == 3, "case: per_rank_asymmetry"


# ==========================================================================
# Family 11: MLA/DSA handler validation (registration / precondition fail-fast / sequence slicing)
# ==========================================================================

class _FakeCPMesh:
    """Minimal CP mesh used to exercise wrapper validation and injection."""

    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank
        self._group = object()

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    def get_group(self):
        return self._group


class _FakeCPContext:
    size = 2
    rank = 1


def _text_forward(self, inputs_embeds=None, **kwargs):
    del self, kwargs
    return inputs_embeds


ToyTextModel = type(
    "ToyTextModel",
    (nn.Module,),
    {"__module__": __name__, "forward": _text_forward},
)


ToyMLAAttention = type(
    "ToyMLAAttention",
    (nn.Module,),
    {"__module__": __name__, "attention_type": "mla"},
)


class _ToyModel(nn.Module):
    """Small module tree matching the MLA/DSA discovery contract."""

    def __init__(self, *, heads=8, index_heads=4):
        super().__init__()
        self.text_model = ToyTextModel()
        self.attention = ToyMLAAttention()
        self.config = SimpleNamespace(text_config=SimpleNamespace(
            num_attention_heads=heads,
            index_num_attention_heads=index_heads,
            dsa_dense_warm_up=False,
            apply_FA_rescale=True,
            use_fused_sink_fa=False,
        ))

    def forward(self, inputs_embeds=None):
        return self.text_model.forward(inputs_embeds=inputs_embeds)


class _ToyKLLoss:
    @staticmethod
    def apply(*args):
        return args[0]


def test_mla_dsa_handler_validation():
    """MLA/DSA handler registration, precondition fail-fast, sequence slicing."""
    # ── case: wrapper_is_registered ── registry entry + decorator metadata
    assert INNER_WRAPPER_REGISTRY["mla_dsa_ulysses"] is (
        mla_dsa_ulysses_cp_wrapper), "case: wrapper_is_registered"
    injection_meta = getattr(mla_dsa_ulysses_cp_wrapper, "_injection_meta")
    assert injection_meta.kind == "inner_wrapper", "case: wrapper_is_registered"

    # ── case: requires_active_cp_mesh ── cp_mesh=None / size=1 → fail-fast
    for label, cp_mesh in (("None", None), ("size1", _FakeCPMesh(size=1))):
        _expect_raise(f"requires_active_cp_mesh[{label}]", ValueError,
                      "active CP mesh", mla_dsa_ulysses_cp_wrapper,
                      _ToyModel(), None, None, cp_mesh, None)

    # ── case: validates_head_divisibility ── heads=7 is not divisible by the cp size
    _expect_raise("validates_head_divisibility", ValueError,
                  "num_attention_heads", mla_dsa_ulysses_cp_wrapper,
                  _ToyModel(heads=7), None, None, _FakeCPMesh(), None)

    # ── case: requires_config ── missing config → fail-fast
    model = _ToyModel()
    del model.config
    _expect_raise("requires_config", ValueError, "target_module.config",
                  mla_dsa_ulysses_cp_wrapper, model, None, None,
                  _FakeCPMesh(), None)

    # ── case: requires_non_none_text_config ── text_config=None → fail-fast
    model = _ToyModel()
    model.config.text_config = None
    _expect_raise("requires_non_none_text_config", ValueError,
                  "non-None text_config", mla_dsa_ulysses_cp_wrapper,
                  model, None, None, _FakeCPMesh(), None)

    # ── case: requires_head_config ── missing num_attention_heads /
    # index_num_attention_heads → fail-fast
    for missing_name in ("num_attention_heads", "index_num_attention_heads"):
        model = _ToyModel()
        delattr(model.config.text_config, missing_name)
        _expect_raise(f"requires_head_config[{missing_name}]", ValueError,
                      missing_name, mla_dsa_ulysses_cp_wrapper,
                      model, None, None, _FakeCPMesh(), None)

    # ── case: slice_sequence_uses_cp_rank ── slices by cp rank and stays contiguous
    tensor = torch.arange(16).reshape(2, 8)
    actual = _slice_sequence(tensor, 1, _FakeCPContext())
    torch.testing.assert_close(actual, tensor[:, 4:],
                               msg="case: slice_sequence_uses_cp_rank")
    assert actual.is_contiguous(), "case: slice_sequence_uses_cp_rank"


# ==========================================================================
# Family 12: MLA/DSA model-side adaptation one-shot configuration
# (combined scenario level, kept standalone)
# ==========================================================================

def _define_attention_symbols(monkeypatch):
    """Define model-module symbols adapted by the MLA/DSA wrapper."""
    module = sys.modules[__name__]

    def apply_mome(hidden_states, mome_mask, conv, use_fused):
        del mome_mask, conv, use_fused
        return hidden_states

    def mla_backend(module, query, key, value, attention_mask, **kwargs):
        del module, key, value, attention_mask, kwargs
        return query

    def sparse_backend(module, query, key, value, attention_mask, **kwargs):
        del module, key, value, attention_mask, kwargs
        return query, None, None

    def indexer(module, index_query, index_key, merge_weight,
                actual_q_len, actual_kv_len):
        del module, index_key, merge_weight, actual_q_len, actual_kv_len
        return index_query

    monkeypatch.setattr(module, "_apply_mome", apply_mome, raising=False)
    monkeypatch.setattr(module, "ATTENTION_FUNCTIONS", {
        "npu_fa_rescale": mla_backend,
        "dsa_sparse_attention": sparse_backend,
    }, raising=False)
    monkeypatch.setattr(
        module, "dsa_lightning_indexer_forward", indexer, raising=False)
    monkeypatch.setattr(
        module, "SparseLightningIndexerKLLossTrainFunction", _ToyKLLoss,
        raising=False)
    return module, apply_mome, mla_backend, sparse_backend, indexer


def test_mla_dsa_wrapper_configures_adaptations(monkeypatch):
    """Verify that one wrapper call configures every model-side adaptation."""
    module, old_mome, old_mla, old_sparse, old_indexer = (
        _define_attention_symbols(monkeypatch))
    model = _ToyModel()
    original_model_forward = model.forward
    original_text_forward = model.text_model.forward

    mla_dsa_ulysses_cp_wrapper(
        model, None, None, _FakeCPMesh(), None)

    assert model.forward != original_model_forward
    assert model.text_model.forward != original_text_forward
    assert getattr(module, "_apply_mome") is not old_mome
    assert module.ATTENTION_FUNCTIONS["npu_fa_rescale"] is not old_mla
    assert module.ATTENTION_FUNCTIONS["dsa_sparse_attention"] is not old_sparse
    assert module.dsa_lightning_indexer_forward is not old_indexer
    assert module.SparseLightningIndexerKLLossTrainFunction is not _ToyKLLoss
    context = getattr(model, "_hyper_ulysses_context")
    assert context.cp_mesh.size() == 2

    inputs = torch.randn(1, 8, 4)
    output = model.text_model.forward(inputs_embeds=inputs, use_cache=False)
    torch.testing.assert_close(output, inputs[:, :4])
